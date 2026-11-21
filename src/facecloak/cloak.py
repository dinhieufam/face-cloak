"""Stage 1: compute cloaks for every (image, ε) and write <exp>/cloak_out/manifest.csv.

Resumable: any (image, ε) that already has a PNG and a manifest row is skipped.
"""
from __future__ import annotations

import random
import time

import numpy as np
import pandas as pd
import torch

from . import env, viz
from .attacks import latent_mse, load_vae, make_encoder, make_roundtrip, pgd_encoder_attack
from .config import dump, get_paths, resolve_path
from .data import load_batch, load_image, prepare_images, psnr, quantize, seeded_uniform, to_pil
from .masks import attacker_view, ensure_masks, load_face_masks


def run_cloak(cfg: dict) -> pd.DataFrame:
    cc, paths = cfg["cloak"], get_paths(cfg)
    device = env.get_device()
    print(env.describe())
    print(f"Experiment '{cfg['exp_name']}' -> {paths.cloak_dir}")
    if device != "cuda":
        print("WARNING: no CUDA GPU, PGD will be slow (fp32).")

    (paths.cloak_dir / "images").mkdir(parents=True, exist_ok=True)
    (paths.cloak_dir / "figures").mkdir(parents=True, exist_ok=True)
    dump(cfg, paths.cloak_dir / "config.json")

    loss_kind = cc.get("loss", "latent")
    if loss_kind not in ("latent", "identity"):
        raise ValueError(f"cloak.loss must be 'latent' or 'identity', got {loss_kind!r}")
    use_id = loss_kind == "identity"
    if use_id and cc["untargeted"]:
        raise ValueError("cloak.loss=identity and cloak.untargeted are mutually exclusive")
    no_target = cc["untargeted"] or use_id     # latent columns are then measured against E(x)

    items = prepare_images(cfg, paths)
    masks = None
    if cc["mask_aware"] or use_id:             # identity loss needs the face box (from the clean-image mask)
        masks, _ = ensure_masks(cfg, paths, items)
        items = [it for it in items if it["img_id"] in masks]

    fp16 = bool(cfg["model"]["fp16"]) and device == "cuda"
    vae = load_vae(cfg["model"]["id"], device)
    encode = make_encoder(vae, device, fp16)
    size = int(cfg["data"]["img_size"])
    if use_id:
        from .identity import face_boxes, load_recognizer, make_decoder, make_identity_objective
        recognizer, id_size = load_recognizer(cc.get("id_model", "facenet_vggface2"), device)
        decode = make_decoder(vae, device, fp16)
        print(f"Identity loss through the VAE with {cc.get('id_model', 'facenet_vggface2')} ({id_size}px crops)")

    # target image in [0,1]; its latent is computed per batch (needed when the mask differs per image)
    target_img = None
    if not no_target:
        target_img = (torch.full((1, 3, size, size), 0.5, device=device) if cc["target"] == "gray"
                      else load_image(resolve_path(cfg, cc["target"]), size, device))

    manifest = paths.cloak_dir / "manifest.csv"
    rows = pd.read_csv(manifest).to_dict("records") if manifest.exists() else []
    done = {(r["img_id"], int(r["eps"])) for r in rows if (paths.root / r["cloak_path"]).exists()}
    todo_all = [(e, it) for e in cc["eps"] for it in items if (it["img_id"], e) not in done]
    print(f"{len(done)} (image, ε) pairs already done, {len(todo_all)} to do.")

    bs = cc["batch_size"] if cc["batch_size"] not in (None, "auto") else env.auto_batch_size(device)
    rng = random.Random(cc["seed"])
    histories: dict[int, list[torch.Tensor]] = {}
    t_start, n_done_now = time.time(), 0
    for eps_int in cc["eps"]:
        eps = eps_int / 255.0
        todo = [it for e, it in todo_all if e == eps_int]
        i = 0
        while i < len(todo):
            batch = todo[i:i + bs]
            try:
                t0 = time.time()
                x = load_batch([b["clean_path"] for b in batch], device)
                m_face = load_face_masks([masks[b["img_id"]]["face"] for b in batch], device) if masks else None
                m = m_face if cc["mask_aware"] else None

                def view(t):
                    return attacker_view(t, m) if m is not None else t

                with torch.no_grad():
                    z_clean = encode(view(x))
                    z_tgt = z_clean if no_target else encode(view(target_img.expand_as(x)))
                obj = (make_identity_objective(encode, decode, recognizer, id_size, x, face_boxes(m_face),
                                               device, fp16) if use_id else None)
                init = (torch.stack([seeded_uniform(b["img_id"], eps_int, "init", x.shape[1:], cc["seed"])
                                     for b in batch]) if cc["random_init"] else None)
                delta, hist = pgd_encoder_attack(
                    encode, x, z_tgt, eps, cc["n_iters"], cc["step_frac"], init,
                    mask=m, mask_jitter_px=int(cc["mask_jitter_px"]), momentum=float(cc["momentum"]),
                    untargeted=bool(cc["untargeted"]), rng=rng, objective=obj)
                x_adv = quantize(x + delta)
                with torch.no_grad():
                    z_adv = encode(view(x_adv))
                    L_clean, L_adv = latent_mse(z_clean, z_tgt), latent_mse(z_adv, z_tgt)
                    D_adv = latent_mse(z_adv, z_clean)
                    if use_id:
                        C_clean, C_adv = obj(view(x)), obj(view(x_adv))
                    if cc["save_random_baseline"]:
                        signs = torch.stack([seeded_uniform(b["img_id"], eps_int, "rand", x.shape[1:], cc["seed"])
                                             .sign() for b in batch]).to(device)
                        x_rnd = quantize(x + eps * signs)
                        z_rnd = encode(view(x_rnd))
                        L_rnd, D_rnd = latent_mse(z_rnd, z_tgt), latent_mse(z_rnd, z_clean)
                        C_rnd = obj(view(x_rnd)) if use_id else None
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if bs == 1:
                    raise
                bs = max(1, bs // 2)
                print(f"Out of GPU memory -> batch size lowered to {bs}")
                continue
            dt = time.time() - t0

            for j, b in enumerate(batch):
                d = paths.cloak_dir / "images" / b["img_id"]
                d.mkdir(parents=True, exist_ok=True)
                cloak_path = d / f"cloak_eps{eps_int}.png"
                to_pil(x_adv[j]).save(cloak_path)
                ratio = float("nan") if no_target else (L_adv[j] / L_clean[j]).item()
                row = dict(
                    exp=cfg["exp_name"], img_id=b["img_id"], source=b["source"], original=b["original"],
                    eps=eps_int, n_iters=cc["n_iters"], step_frac=cc["step_frac"],
                    target=("(identity)" if use_id else "(untargeted)" if cc["untargeted"] else cc["target"]),
                    loss=loss_kind, mask_aware=bool(cc["mask_aware"]),
                    mask_jitter_px=int(cc["mask_jitter_px"]), momentum=float(cc["momentum"]),
                    untargeted=bool(cc["untargeted"]),
                    clean_path=paths.rel(b["clean_path"]), cloak_path=paths.rel(cloak_path),
                    latent_mse_clean=L_clean[j].item(), latent_mse_cloak=L_adv[j].item(), loss_ratio_cloak=ratio,
                    latent_dist_cloak=D_adv[j].item(),
                    linf_255=round(((x_adv[j] - x[j]).abs().max() * 255).item()),
                    psnr_cloak=psnr(x_adv[j:j + 1], x[j:j + 1]).item(), seconds_per_image=dt / len(batch))
                if use_id:  # surrogate (FaceNet) cosine to the clean face after the VAE round trip
                    row.update(id_cos_clean=C_clean[j].item(), id_cos_cloak=C_adv[j].item())
                if cc["save_random_baseline"]:
                    rnd_path = d / f"random_eps{eps_int}.png"
                    to_pil(x_rnd[j]).save(rnd_path)
                    row.update(random_path=paths.rel(rnd_path), latent_mse_random=L_rnd[j].item(),
                               loss_ratio_random=(float("nan") if no_target
                                                  else (L_rnd[j] / L_clean[j]).item()),
                               latent_dist_random=D_rnd[j].item(),
                               psnr_random=psnr(x_rnd[j:j + 1], x[j:j + 1]).item())
                    if use_id:
                        row.update(id_cos_random=C_rnd[j].item())
                rows = [r for r in rows if not (r["img_id"] == b["img_id"] and int(r["eps"]) == eps_int)] + [row]
            pd.DataFrame(rows).to_csv(manifest, index=False)  # save after every batch
            histories.setdefault(eps_int, []).append(hist)

            i += len(batch)
            n_done_now += len(batch)
            eta = (time.time() - t_start) / n_done_now * (len(todo_all) - n_done_now)
            extra = f" | id cos {C_clean.mean().item():.3f}→{C_adv.mean().item():.3f}" if use_id else ""
            print(f"ε={eps_int:>2}/255 | {i}/{len(todo)} | batch {dt:5.1f}s | "
                  f"latent dist cloak→clean {D_adv.mean().item():.3f}{extra} | ETA {eta / 60:5.1f} min")

    df = pd.read_csv(manifest)
    print(f"Done. Manifest: {manifest} ({len(df)} rows)")
    if histories:
        hist_np = {f"eps{e}": torch.cat(h, dim=1).numpy() for e, h in histories.items()}
        np.savez(paths.cloak_dir / f"loss_history_{time.strftime('%Y%m%d-%H%M%S')}.npz", **hist_np)
        viz.plot_convergence(hist_np, paths.cloak_dir / "figures" / "pgd_convergence.png",
                             untargeted=bool(cc["untargeted"]), identity=use_id)
    viz.plot_cloak_sanity(df, paths, make_roundtrip(vae, device, fp16), device,
                          paths.cloak_dir / "figures" / f"sanity_eps{max(cc['eps'])}.png", show_eps=max(cc["eps"]))
    return df


def summarize_cloak(df: pd.DataFrame) -> pd.DataFrame:
    agg = {"latent_dist_cloak": "mean", "loss_ratio_cloak": "mean", "psnr_cloak": "mean",
           "linf_255": "max", "img_id": "count"}
    if "latent_dist_random" in df:
        agg.update(latent_dist_random="mean", loss_ratio_random="mean", psnr_random="mean")
    for c in ("id_cos_clean", "id_cos_cloak", "id_cos_random"):
        if c in df:
            agg[c] = "mean"
    return df.groupby(["source", "eps"]).agg(agg).rename(columns={"img_id": "n_images"}).round(3)
