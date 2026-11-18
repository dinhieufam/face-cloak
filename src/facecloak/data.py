"""Input images (FFHQ via kagglehub, local selfies) and small tensor/image helpers."""
from __future__ import annotations

import os
import random
import re
import zlib
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

from .config import Paths, resolve_path


# ----------------------------------------------------------------------------- image helpers
def to_square(img: Image.Image, size: int = 512) -> Image.Image:
    """Fix EXIF rotation, center-crop to a square, resize to `size` (LANCZOS)."""
    img = ImageOps.exif_transpose(img).convert("RGB")
    w, h = img.size
    s = min(w, h)
    left, top = (w - s) // 2, (h - s) // 2
    img = img.crop((left, top, left + s, top + s))
    if s != size:
        img = img.resize((size, size), Image.LANCZOS)
    return img


def load_batch(paths, device) -> torch.Tensor:
    """(B,3,H,W) float tensor in [0,1]."""
    arr = np.stack([np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0 for p in paths])
    return torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous().to(device)


def load_image(path, size: int, device) -> torch.Tensor:
    """Any image file -> (1,3,size,size) tensor in [0,1] (square-cropped)."""
    with Image.open(path) as im:
        arr = np.asarray(to_square(im, size), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)[None].to(device)


def to_pil(x: torch.Tensor) -> Image.Image:
    """(3,H,W) in [0,1] -> uint8 PIL image."""
    return Image.fromarray((x.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8))


def quantize(x: torch.Tensor) -> torch.Tensor:
    """Round to uint8 levels: a real upload is 8-bit, so every metric is measured after quantization."""
    return (x.clamp(0, 1) * 255).round() / 255


def seeded_uniform(img_id: str, eps_int: int, salt: str, shape, seed: int) -> torch.Tensor:
    """U[-1,1] noise seeded from (img_id, ε, salt, seed): identical whatever the batch composition."""
    s = zlib.crc32(f"{img_id}|{eps_int}|{salt}|{seed}".encode()) % (2**31)
    g = torch.Generator().manual_seed(s)
    return torch.rand(shape, generator=g) * 2 - 1


def psnr(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Per-image PSNR in dB, inputs in [0,1]."""
    mse = ((a - b) ** 2).flatten(1).mean(1).clamp_min(1e-12)
    return 10 * torch.log10(1.0 / mse)


# ----------------------------------------------------------------------------- dataset
def read_ids(path: str) -> list[str]:
    with open(path) as f:
        return [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]


def fetch_ffhq(kaggle_dataset: str, file_id: str) -> str:
    """Download a single FFHQ file (kagglehub fetches only the requested file)."""
    import kagglehub
    return kagglehub.dataset_download(kaggle_dataset, path=f"{file_id}.png")


def _register(paths: Paths, img_id: str, source: str, size: int, src_path=None, original="") -> dict:
    clean_path = paths.shared_images / img_id / "clean.png"
    if not clean_path.exists():
        clean_path.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(src_path) as im:
            to_square(im, size).save(clean_path)
    return dict(img_id=img_id, source=source, original=original, clean_path=str(clean_path))


def prepare_images(cfg: dict, paths: Paths) -> list[dict]:
    """Collect inputs into <root>/_shared/images/<id>/clean.png and return one dict per image.

    Order: explicit FFHQ ids (config list, or the dev-ids file), then `n_ffhq_random` random FFHQ ids,
    then local selfies. `data.max_images` truncates the list (quick tests).
    """
    dc = cfg["data"]
    size = int(dc["img_size"])
    items: list[dict] = []

    ffhq_ids = dc.get("ffhq_ids")
    if ffhq_ids is None and dc.get("ids_file"):
        ffhq_ids = read_ids(resolve_path(cfg, dc["ids_file"]))
    ffhq_ids = [f"{int(i):05d}" for i in (ffhq_ids or [])]
    if dc.get("max_images"):
        ffhq_ids = ffhq_ids[: dc["max_images"]]

    def add_ffhq(fid: str) -> bool:
        img_id = f"ffhq_{fid}"
        try:
            if (paths.shared_images / img_id / "clean.png").exists():
                items.append(_register(paths, img_id, "ffhq", size, original=f"{fid}.png"))
            else:
                items.append(_register(paths, img_id, "ffhq", size, fetch_ffhq(dc["kaggle_dataset"], fid),
                                       original=f"{fid}.png"))
            return True
        except Exception as e:  # missing file in the Kaggle mirror, auth error, ...
            print(f"  x {fid}.png: {type(e).__name__}: {str(e)[:160]}")
            return False

    got = [fid for fid in ffhq_ids if add_ffhq(fid)]
    if ffhq_ids and not got:
        print("No FFHQ file could be downloaded. Check Kaggle credentials (see docs/setup.md).")

    n_random = int(dc.get("n_ffhq_random") or 0)
    rng, tried, fails = random.Random(cfg["cloak"]["seed"]), set(got), 0
    target_n = len(got) + n_random
    while len(got) < target_n and fails < 8:
        fid = f"{rng.randrange(*dc['ffhq_id_range']):05d}"
        if fid in tried:
            continue
        tried.add(fid)
        if add_ffhq(fid):
            got.append(fid)
            fails = 0
        else:
            fails += 1

    for p in dc.get("selfie_paths") or []:
        p = resolve_path(cfg, p)
        stem = re.sub(r"[^A-Za-z0-9_-]+", "_", Path(p).stem).strip("_")
        try:
            items.append(_register(paths, f"selfie_{stem}", "selfie", size, p, original=os.path.basename(p)))
        except Exception as e:
            print(f"  x {p}: {e}")

    if dc.get("max_images"):
        items = items[: dc["max_images"]]
    assert items, "No input images: set data.ids_file / data.ffhq_ids or data.selfie_paths."
    print(f"Images ready: {len(items)} ({sum(i['source'] == 'ffhq' for i in items)} FFHQ, "
          f"{sum(i['source'] == 'selfie' for i in items)} selfies) in {paths.shared_images}")
    return items
