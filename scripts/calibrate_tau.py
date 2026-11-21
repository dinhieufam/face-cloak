#!/usr/bin/env python
"""Step 0: one fixed identity threshold τ shared by every experiment.

Embeds N clean FFHQ faces (not in the dev set) with the same RetinaFace + ArcFace used for evaluation,
takes all N·(N−1)/2 impostor pairs (different people) and sets τ at a 1% false-accept rate.
Writes <output_root>/tau.json, which `eval.tau_rule: fixed` reads. No diffusion needed; CPU is fine.

python scripts/calibrate_tau.py --config configs/base.yaml [--n 300] [--far 0.01]
"""
import argparse
import json
import random

import numpy as np
from PIL import Image

from facecloak.config import get_paths, load_config, resolve_path
from facecloak.data import fetch_ffhq, read_ids, to_square
from facecloak.evaluate import face_embed, get_face_models

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--n", type=int, default=300, help="number of faces (impostor pairs = n(n-1)/2)")
    ap.add_argument("--far", type=float, default=0.01, help="target false-accept rate")
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    cfg = load_config(args.config)
    paths, ec, dc = get_paths(cfg), cfg["eval"], cfg["data"]
    cache = paths.shared / "tau_images"
    cache.mkdir(parents=True, exist_ok=True)
    dev = {f"{int(i):05d}" for i in read_ids(resolve_path(cfg, dc["ids_file"]))} if dc.get("ids_file") else set()
    models = get_face_models(ec["detector"], ec["recognizer"], ec["det_conf"])

    rng, tried, embs, ids = random.Random(args.seed), set(dev), [], []
    while len(embs) < args.n and len(tried) < 50 * args.n:
        fid = f"{rng.randrange(*dc['ffhq_id_range']):05d}"
        if fid in tried:
            continue
        tried.add(fid)
        p = cache / f"{fid}.png"
        if not p.exists():
            try:
                with Image.open(fetch_ffhq(dc["kaggle_dataset"], fid)) as im:
                    to_square(im, int(dc["img_size"])).save(p)
            except Exception as e:
                print(f"  x {fid}: {type(e).__name__}")
                continue
        e, *_ = face_embed(models, p)
        if e is not None:
            embs.append(e)
            ids.append(fid)
            if len(embs) % 50 == 0:
                print(f"  {len(embs)}/{args.n} faces")

    E = np.stack(embs)
    sims = (E @ E.T)[np.triu_indices(len(E), 1)]
    tau = float(np.quantile(sims, 1 - args.far))
    out = dict(tau=tau, far=args.far, n_faces=len(E), n_pairs=int(len(sims)), impostor_mean=float(sims.mean()),
               impostor_max=float(sims.max()), detector=ec["detector"], recognizer=ec["recognizer"],
               seed=args.seed, ffhq_ids=ids)
    paths.tau_file.parent.mkdir(parents=True, exist_ok=True)
    paths.tau_file.write_text(json.dumps(out, indent=2))
    print(f"τ @ FAR {args.far:.0%} = {tau:.3f} from {len(sims)} impostor pairs ({len(E)} faces) -> {paths.tau_file}")
