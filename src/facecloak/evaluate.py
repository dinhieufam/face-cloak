"""Stage 3: identity evaluation with UniFace (RetinaFace detection + ArcFace embeddings).

sim = cosine(ArcFace(edited output), ArcFace(clean.png)). Protected ⇔ no face found or sim < τ.
Rows flagged by the safety checker are excluded from metrics (a black image is not "protected").

τ (identity threshold), `eval.tau_rule`:
    fixed   eval.tau if set, else <root>/tau.json from scripts/calibrate_tau.py   (use this to compare experiments)
    far1    1% false-accept rate on impostor pairs among this run's source faces
    eer     equal error rate between impostor pairs and clean-branch (genuine) edits
"""
from __future__ import annotations

import itertools
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import env, viz
from .config import Paths, dump, get_paths

_MODELS = None


def get_face_models(detector: str, recognizer: str, det_conf: float):
    global _MODELS
    if _MODELS is None:
        from uniface import ArcFace, RetinaFace
        from uniface.constants import ArcFaceWeights, RetinaFaceWeights
        _MODELS = (RetinaFace(model_name=RetinaFaceWeights(detector), confidence_threshold=det_conf),
                   ArcFace(model_name=ArcFaceWeights(recognizer)))
    return _MODELS


def iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def face_embed(models, path, ref_box=None):
    """-> (L2-normalised embedding or None, det_conf, iou_with_ref, n_faces, bbox).

    With several faces, take the one overlapping the source face box most (so a background face that SD
    invented is not picked); otherwise the largest.
    """
    import cv2
    detector, recognizer = models
    bgr = cv2.imread(str(path))  # UniFace expects BGR
    faces = detector.detect(bgr)
    if not faces:
        return None, np.nan, np.nan, 0, None

    def area(f):
        return (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])

    if ref_box is not None:
        ious = [iou(f.bbox, ref_box) for f in faces]
        j = int(np.argmax(ious)) if max(ious) > 0 else int(np.argmax([area(f) for f in faces]))
        ov = ious[j]
    else:
        j, ov = int(np.argmax([area(f) for f in faces])), np.nan
    f = faces[j]
    emb = recognizer.get_normalized_embedding(bgr, f.landmarks).ravel()
    return emb / np.linalg.norm(emb), float(f.confidence), ov, len(faces), np.asarray(f.bbox, float)


def eer_threshold(genuine, impostor):
    ts = np.unique(np.concatenate([genuine, impostor]))
    far = np.array([(impostor >= t).mean() for t in ts])
    frr = np.array([(genuine < t).mean() for t in ts])
    d = np.abs(far - frr)
    ks = np.flatnonzero(d == d.min())  # several thresholds can tie -> take the middle one
    k = int(ks[len(ks) // 2])
    return float(ts[k]), float((far[k] + frr[k]) / 2)


def resolve_tau(cfg: dict, paths: Paths, impostor: np.ndarray, genuine: np.ndarray) -> tuple[float, dict]:
    ec = cfg["eval"]
    far1 = float(np.quantile(impostor, 0.99)) if len(impostor) else np.nan
    eer_t, eer = eer_threshold(genuine, impostor) if len(genuine) and len(impostor) else (np.nan, np.nan)
    info = dict(tau_far1_run=far1, tau_eer_run=eer_t, eer_run=eer, n_impostor=len(impostor), n_genuine=len(genuine))
    rule = ec["tau_rule"]
    if rule == "fixed":
        if ec.get("tau") is not None:
            return float(ec["tau"]), dict(info, tau_source="config eval.tau")
        if paths.tau_file.exists():
            t = json.loads(paths.tau_file.read_text())
            return float(t["tau"]), dict(info, tau_source=str(paths.tau_file))
        print(f"WARNING: tau_rule=fixed but no eval.tau and no {paths.tau_file}. Run scripts/calibrate_tau.py. "
              f"Falling back to far1 on {len(impostor)} impostor pairs (unreliable with few images).")
        rule = "far1"
    tau = far1 if rule == "far1" else eer_t
    return tau, dict(info, tau_source=f"{rule} (this run)")


def run_eval(cfg: dict) -> dict:
    ec, paths = cfg["eval"], get_paths(cfg)
    env.mount_drive_if_needed(str(paths.root))
    gens = pd.read_csv(paths.inpaint_dir / "generations.csv")
    models = get_face_models(ec["detector"], ec["recognizer"], ec["det_conf"])
    out_dir = paths.inpaint_dir

    # source faces (clean.png)
    src = {}
    for img_id, p in gens[gens.branch == "clean"][["img_id", "input_path"]].drop_duplicates().itertuples(index=False):
        e, _, _, _, box = face_embed(models, paths.abs(p))
        if e is None:
            print(f"UniFace found no face in clean {img_id}: excluded from evaluation")
            continue
        src[img_id] = dict(emb=e, box=box)

    # similarity of the attack INPUTS (does the cloak alone already change identity?)
    input_sim = {}
    for v in gens[["img_id", "branch", "eps", "input_path"]].drop_duplicates().itertuples(index=False):
        if v.img_id in src:
            e, *_ = face_embed(models, paths.abs(v.input_path), src[v.img_id]["box"])
            input_sim[(v.img_id, v.branch, v.eps)] = float(e @ src[v.img_id]["emb"]) if e is not None else np.nan

    # outputs (resumable on out_path)
    res_csv = out_dir / "results.csv"
    res_rows = pd.read_csv(res_csv).to_dict("records") if res_csv.exists() else []
    res_rows = [r for r in res_rows if r["out_path"] in set(gens.out_path)]
    seen = {r["out_path"] for r in res_rows}
    todo = gens[~gens.out_path.isin(seen) & gens.img_id.isin(src)]
    t0 = time.time()
    for n, g in enumerate(todo.itertuples(index=False), 1):
        e, conf, ov, nf, _ = face_embed(models, paths.abs(g.out_path), src[g.img_id]["box"])
        res_rows.append(dict(g._asdict(), face_found=e is not None, det_conf=conf, iou=ov, n_faces=nf,
                             sim=float(e @ src[g.img_id]["emb"]) if e is not None else np.nan,
                             sim_input=input_sim.get((g.img_id, g.branch, g.eps), np.nan)))
        if n % 200 == 0:
            pd.DataFrame(res_rows).to_csv(res_csv, index=False)
            print(f"  evaluated {n}/{len(todo)} ({(time.time() - t0) / n:.2f}s/img)")
    res = pd.DataFrame(res_rows)
    res.to_csv(res_csv, index=False)
    print(f"results.csv: {len(res)} rows | faces found in {res.face_found.mean():.1%} of outputs")

    # threshold
    ids = sorted(src)
    imp = np.array([src[a]["emb"] @ src[b]["emb"] for a, b in itertools.combinations(ids, 2)
                    if not (a.startswith("selfie_") and b.startswith("selfie_"))])  # selfies may share a person
    ev = res[~res.nsfw.astype(bool)]
    genuine = ev[(ev.branch == "clean") & ev.face_found].sim.values
    tau, tau_info = resolve_tau(cfg, paths, imp, genuine)
    print(f"τ = {tau:.3f} ({tau_info['tau_source']})")

    ev = ev.assign(protected=(~ev.face_found.astype(bool)) | (ev.sim < tau))
    summary = (ev.groupby(["branch", "eps"])
                 .agg(n=("sim", "size"), face_rate=("face_found", "mean"), mean_sim=("sim", "mean"),
                      median_sim=("sim", "median"), mean_sim_input=("sim_input", "mean"),
                      protection_rate=("protected", "mean"))
                 .round(3))
    nsfw_rate = res.groupby(["branch", "eps"]).nsfw.mean().rename("nsfw_rate").round(3)
    summary = summary.join(nsfw_rate)
    summary.to_csv(out_dir / "summary.csv")

    ctrl = ev[ev.branch == "clean"].set_index(["img_id", "prompt_id", "seed"]).sim
    paired = (ev[ev.branch != "clean"].join(ctrl.rename("sim_ctrl"), on=["img_id", "prompt_id", "seed"])
                .assign(delta=lambda d: d.sim_ctrl - d.sim))
    paired_summary = (paired.groupby(["branch", "eps"]).delta.agg(["mean", "std", "count"]).round(3)
                      .rename(columns={"mean": "delta_mean", "std": "delta_std", "count": "n_pairs"}))
    paired_summary.to_csv(out_dir / "paired.csv")
    (out_dir / "tau.json").write_text(json.dumps(dict(tau=tau, **tau_info), indent=2, default=float))
    dump(cfg, out_dir / "config.json")

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    viz.plot_similarity(imp, genuine, ev, summary, tau, fig_dir / "similarity_and_protection.png")
    viz.plot_edit_grid(res, paths, cfg, fig_dir)
    return dict(summary=summary, paired=paired_summary, results=res, tau=tau, tau_info=tau_info)


def compare_experiments(root, exps=None) -> pd.DataFrame:
    """One row per (experiment, branch, ε): the results table of docs/experiments.md."""
    root = Path(root)
    exps = exps or sorted(p.name for p in root.iterdir()
                          if p.is_dir() and not p.name.startswith("_") and (p / "inpaint_out" / "summary.csv").exists())
    tables = []
    for exp in exps:
        d = root / exp
        s = pd.read_csv(d / "inpaint_out" / "summary.csv")
        p = pd.read_csv(d / "inpaint_out" / "paired.csv") if (d / "inpaint_out" / "paired.csv").exists() else None
        if p is not None:
            s = s.merge(p, on=["branch", "eps"], how="left")
        man = d / "cloak_out" / "manifest.csv"
        if man.exists():
            m = pd.read_csv(man).groupby("eps").agg(psnr_cloak=("psnr_cloak", "mean"),
                                                    latent_dist_cloak=("latent_dist_cloak", "mean")).reset_index()
            s = s.merge(m.assign(branch="cloak"), on=["branch", "eps"], how="left")
        tau = json.loads((d / "inpaint_out" / "tau.json").read_text())["tau"] if (d / "inpaint_out" / "tau.json").exists() else np.nan
        tables.append(s.assign(exp=exp, tau=tau))
    if not tables:
        return pd.DataFrame()
    out = pd.concat(tables, ignore_index=True)
    first = ["exp", "branch", "eps", "n", "mean_sim", "delta_mean", "protection_rate", "nsfw_rate", "tau"]
    return out[[c for c in first if c in out] + [c for c in out if c not in first]].round(3)
