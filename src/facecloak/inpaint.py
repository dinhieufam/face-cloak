"""Stage 2: the attacker. SD 1.5 inpainting keeps the face (MediaPipe FACE_OVAL) and regenerates the rest.

Branches per image:
    clean   clean.png               uncloaked control (what the attacker gets without protection)
    cloak   cloak_eps<E>.png        this experiment's cloak
    random  random_eps<E>.png       random ±ε noise, same budget (is it the adversarial direction or just noise?)

The clean and random branches do not depend on the cloak method, so their outputs are cached under
<root>/_shared/inpaint_<settings-hash>/ and reused by every experiment; only the cloak branch is regenerated.
Each output PNG has a sidecar JSON with its safety-checker flag, so generations.csv can be rebuilt from disk.
"""
from __future__ import annotations

import io
import json
import time
from pathlib import Path

import pandas as pd
import torch
from PIL import Image

from . import env
from .config import Paths, dump, get_paths
from .masks import ensure_masks


def build_variants(cfg: dict, paths: Paths) -> pd.DataFrame:
    manifest = paths.cloak_dir / "manifest.csv"
    if not manifest.exists():
        raise FileNotFoundError(f"{manifest} not found: run the cloak stage for '{cfg['exp_name']}' first.")
    man = pd.read_csv(manifest)
    eps_filter = cfg["inpaint"].get("eps")
    if eps_filter is not None:
        man = man[man.eps.isin(eps_filter)]
    rows = []
    for img_id, g in man.groupby("img_id"):
        r0 = g.iloc[0]
        rows.append(dict(img_id=img_id, source=r0.source, branch="clean", eps=0, input_path=r0.clean_path))
        for _, r in g.iterrows():
            rows.append(dict(img_id=img_id, source=r.source, branch="cloak", eps=int(r.eps), input_path=r.cloak_path))
            if cfg["inpaint"]["include_random"] and isinstance(r.get("random_path"), str):
                rows.append(dict(img_id=img_id, source=r.source, branch="random", eps=int(r.eps),
                                 input_path=r.random_path))
    return pd.DataFrame(rows)


def output_path(paths: Paths, v, pid: int, seed: int) -> Path:
    tag = v.branch if v.branch == "clean" else f"{v.branch}_eps{v.eps}"
    base = paths.inpaint_dir if v.branch == "cloak" else paths.shared_inpaint
    return base / "outputs" / v.img_id / tag / f"p{pid}_s{seed}.png"


def jpeg(image: Image.Image, quality: int | None) -> Image.Image:
    """Re-encode as JPEG (robustness check: the platform recompresses the upload before the attacker gets it)."""
    if quality is None:
        return image
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=int(quality))
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def load_pipeline(cfg: dict, device: str):
    from diffusers import StableDiffusionInpaintPipeline
    extra = {} if cfg["inpaint"]["keep_safety_checker"] else dict(safety_checker=None, requires_safety_checker=False)
    dtype = torch.float16 if (device == "cuda" and cfg["model"]["fp16"]) else torch.float32
    pipe = StableDiffusionInpaintPipeline.from_pretrained(cfg["model"]["id"], torch_dtype=dtype, **extra).to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def run_inpaint(cfg: dict) -> pd.DataFrame:
    ic, paths = cfg["inpaint"], get_paths(cfg)
    device = env.get_device()
    print(env.describe())
    paths.inpaint_dir.mkdir(parents=True, exist_ok=True)
    dump(cfg, paths.inpaint_dir / "config.json")

    variants = build_variants(cfg, paths)
    items = [dict(img_id=i, clean_path=str(paths.abs(p)))
             for i, p in variants[variants.branch == "clean"][["img_id", "input_path"]].itertuples(index=False)]
    masks, _ = ensure_masks(cfg, paths, items)
    variants = variants[variants.img_id.isin(masks)].reset_index(drop=True)

    prompt_ids, seeds = list(ic["prompt_ids"]), list(ic["seeds"])
    jobs = []
    for v in variants.itertuples():
        for pid in prompt_ids:
            missing = [s for s in seeds if not output_path(paths, v, pid, s).exists()]
            if missing:
                jobs.append((v, pid, missing))
    n_total = len(variants) * len(prompt_ids) * len(seeds)
    todo = sum(len(j[2]) for j in jobs)
    print(f"{len(variants)} inputs x {len(prompt_ids)} prompts x {len(seeds)} seeds = {n_total} inpaintings; "
          f"{n_total - todo} on disk, {todo} to generate.")

    if jobs:
        pipe = load_pipeline(cfg, device)
        t0, done_now = time.time(), 0
        for k, (v, pid, miss) in enumerate(jobs):
            image = jpeg(Image.open(paths.abs(v.input_path)).convert("RGB"), ic.get("jpeg_quality"))
            mask = Image.open(masks[v.img_id]["inpaint"]).convert("L")
            W, H = image.size
            # one generator per seed: seed s gives the same image alone or in a batch, in every branch
            out = pipe(prompt=ic["prompts"][pid], negative_prompt=ic["negative_prompt"], image=image,
                       mask_image=mask, height=H, width=W, num_inference_steps=ic["num_steps"],
                       guidance_scale=ic["guidance"], num_images_per_prompt=len(miss),
                       generator=[torch.Generator(device=device).manual_seed(s) for s in miss])
            nsfw = out.nsfw_content_detected or [False] * len(miss)
            for img, s, flag in zip(out.images, miss, nsfw):
                p = output_path(paths, v, pid, s)
                p.parent.mkdir(parents=True, exist_ok=True)
                img.save(p)
                p.with_suffix(".json").write_text(json.dumps({"nsfw": bool(flag)}))
            done_now += len(miss)
            if (k + 1) % 5 == 0 or k == len(jobs) - 1:
                rate = (time.time() - t0) / done_now
                print(f"[{k + 1}/{len(jobs)}] {v.img_id} {v.branch} ε={v.eps} p{pid} | {rate:.1f}s/img | "
                      f"ETA {(todo - done_now) * rate / 60:.1f} min")
        del pipe
        if device == "cuda":
            torch.cuda.empty_cache()

    rows = []
    for v in variants.itertuples():
        for pid in prompt_ids:
            for s in seeds:
                p = output_path(paths, v, pid, s)
                if not p.exists():
                    continue
                meta = json.loads(p.with_suffix(".json").read_text()) if p.with_suffix(".json").exists() else {}
                rows.append(dict(img_id=v.img_id, source=v.source, branch=v.branch, eps=v.eps, breaker="none",
                                 prompt_id=pid, seed=s, input_path=v.input_path, out_path=paths.rel(p),
                                 nsfw=bool(meta.get("nsfw", False))))
    gens = pd.DataFrame(rows)
    gens.to_csv(paths.inpaint_dir / "generations.csv", index=False)
    print(f"generations.csv: {len(gens)} outputs ({int(gens.nsfw.sum()) if len(gens) else 0} flagged by the "
          f"safety checker)")
    return gens
