#!/usr/bin/env python
"""Phase 5: rebuild every report table and figure from <output_root> and render the report draft.

python scripts/make_report.py [--config configs/base.yaml] [--template docs/report_template.md]

Writes <output_root>/report/: tables/*.csv + *.md, figures/*.png, report.md (the template with every
{{table:...}}, {{fig:...}} and {{value:...}} placeholder filled). Nothing in the report is typed by hand.

Experiments are grouped by name: E*/C* = main table (E0 is the reference), R_* = robustness checks
(each compared with its own clean branch and its own random control).
"""
import argparse
import os
import json
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from facecloak.config import get_paths, load_config  # noqa: E402

# reference palette (dataviz skill, light mode): series blue/orange, ink, recessive grid
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1", "#fcfcfb"
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 9})


CHANGES = {"E0_baseline": "PhotoGuard PGD, gray target (reference)", "E1_pattern_target": "target = colour pattern",
           "E1b_checker_target": "target = 8-px checkerboard", "E2_mask_aware": "loss on attacker's view + mask jitter",
           "E3_momentum": "momentum PGD (μ = 0.9)", "E4_untargeted": "untargeted (push away from E(x))",
           "E5_identity": "identity loss (FaceNet) through the VAE"}


def change_of(exp_dir: Path) -> str:
    """Short description: CHANGES, else the config's header comment ('# C1 (combination): E5 + E3. ...')."""
    if exp_dir.name in CHANGES:
        return CHANGES[exp_dir.name]
    cfg = json.loads((exp_dir / "cloak_out" / "config.json").read_text()) if (exp_dir / "cloak_out" / "config.json").exists() else {}
    f = cfg.get("config_file")
    if f and Path(f).exists():
        first = Path(f).read_text().split("\n", 1)[0].lstrip("# ").strip()
        return re.sub(r"^[A-Z]\w*(\s*\([^)]*\))?:\s*", "", first).split(". ")[0].rstrip(".")
    return ""


def load_results(exp_dir: Path, tau: float) -> pd.DataFrame:
    r = pd.read_csv(exp_dir / "inpaint_out" / "results.csv")
    r = r[~r.nsfw.astype(bool)].copy()
    r["protected"] = (~r.face_found.astype(bool)) | (r.sim < tau)
    return r


def paired_delta(a: pd.DataFrame, b: pd.DataFrame, rng: np.random.Generator):
    """Δ = mean of sim_b − sim_a (b = reference, a = cloak; > 0 = cloak lowers identity) over matched (img, prompt, seed) pairs with a face in both, and a bootstrap 95% CI."""
    k = ["img_id", "prompt_id", "seed"]
    m = a[k + ["sim"]].merge(b[k + ["sim"]], on=k, suffixes=("_a", "_b")).dropna()
    d = (m.sim_b - m.sim_a).values
    if len(d) == 0:
        return np.nan, np.nan, np.nan, 0
    boot = rng.choice(d, (5000, len(d))).mean(1)
    return d.mean(), *np.quantile(boot, [0.025, 0.975]), len(d)


def fmt(x, nd=3):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def summarize(name, change, cloak, clean, rnd, man, rng):
    d_clean = paired_delta(cloak, clean, rng)
    d_rnd = paired_delta(cloak, rnd, rng) if rnd is not None else (np.nan,) * 3 + (0,)
    return dict(exp=name, change=change, n=len(cloak), mean_sim=cloak.sim.mean(),
                delta_vs_clean=d_clean[0], delta_vs_random=d_rnd[0], delta_vs_random_lo=d_rnd[1],
                delta_vs_random_hi=d_rnd[2], n_pairs_random=d_rnd[3], protection_rate=cloak.protected.mean(),
                face_rate=cloak.face_found.mean(),
                latent_dist_cloak=man.latent_dist_cloak.mean() if man is not None else np.nan,
                psnr_cloak=man.psnr_cloak.mean() if man is not None else np.nan,
                id_cos_cloak=man.id_cos_cloak.mean() if man is not None and "id_cos_cloak" in man else np.nan)


def ref_row(name, df):
    return dict(exp=name, change="", n=len(df), mean_sim=df.sim.mean(), protection_rate=df.protected.mean(),
                face_rate=df.face_found.mean())


def md_table(df: pd.DataFrame, cols: dict) -> str:
    head = "| " + " | ".join(cols.values()) + " |\n|" + "---|" * len(cols) + "\n"
    rows = []
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r.get(c, np.nan)
            if c == "delta_vs_random" and not np.isnan(r.get("delta_vs_random_lo", np.nan)):
                cells.append(f"{v:+.3f} [{r.delta_vs_random_lo:+.3f}, {r.delta_vs_random_hi:+.3f}]")
            elif c in ("delta_vs_clean",) and isinstance(v, float) and not np.isnan(v):
                cells.append(f"{v:+.3f}")
            elif c == "psnr_cloak":
                cells.append(fmt(v, 1))
            elif c in ("protection_rate", "nsfw_rate", "face_rate"):
                cells.append("—" if pd.isna(v) else f"{v:.0%}")
            elif isinstance(v, (float, np.floating)):
                cells.append(fmt(float(v)))
            else:
                cells.append("" if pd.isna(v) else str(v))
        rows.append("| " + " | ".join(cells) + " |")
    return head + "\n".join(rows) + "\n"


MAIN_COLS = {"exp": "ID", "change": "Change", "n": "n", "mean_sim": "mean sim ↓", "delta_vs_clean": "Δ vs clean ↑",
             "delta_vs_random": "Δ vs random [95% CI] ↑", "protection_rate": "protection ↑", "nsfw_rate": "NSFW",
             "latent_dist_cloak": "latent dist", "psnr_cloak": "PSNR (dB)", "id_cos_cloak": "FaceNet cos (E5)"}


def bar_figure(main: pd.DataFrame, clean_v, rnd_v, col, xlabel, out, pct=False):
    d = main[main.exp.str.match(r"^[EC]")].iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.4, 0.42 * len(d) + 1.0))
    ax.barh(d.exp, d[col], color=BLUE, height=0.6, edgecolor=SURFACE, linewidth=2)
    for y, v in enumerate(d[col]):
        ax.text(v + 0.005, y, f"{v:.0%}" if pct else f"{v:.3f}", va="center", color=INK2, fontsize=8)
    refs = [(v, lab, ls) for v, lab, ls in [(clean_v, "clean", "-"), (rnd_v, "random ±ε", "--")]
            if v is not None and not np.isnan(v)]
    lo = min(v for v, _, _ in refs) if refs else None
    for v, lab, ls in refs:   # labels above the plot, the lower reference right-aligned so they never collide
        ax.axvline(v, color=MUTED, ls=ls, lw=1.2)
        ax.text(v, len(d) - 0.3, f"{lab} {v:.0%}" if pct else f"{lab} {v:.3f}", color=INK2, fontsize=8, va="bottom",
                ha="right" if (v == lo and len(refs) > 1) else "left")
    ax.set_xlim(0, 1.0 if not pct else max(1.0, d[col].max() + 0.1))
    ax.set_ylim(-0.6, len(d) + 0.3)
    ax.set_xlabel(xlabel)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def convergence_figure(root: Path, exps, out):
    fig, ax = plt.subplots(figsize=(6, 3.4))
    drawn = 0
    for exp, color in zip(exps, [BLUE, ORANGE]):
        files = sorted((root / exp / "cloak_out").glob("loss_history_*.npz"))
        if not files:
            continue
        h = np.load(files[-1])["eps16"]
        y = (h / h[0:1]).mean(1)
        ax.plot(y, color=color, lw=2, label=exp)
        ax.text(len(y) - 1, y[-1], f" {exp}", color=INK2, fontsize=8, va="center")
        drawn += 1
    if not drawn:
        plt.close(fig)
        return False
    ax.set_yscale("log")
    ax.set_xlabel("PGD iteration")
    ax.set_ylabel("latent loss / initial (mean of 10 images)")
    ax.grid(color=GRID, lw=0.8)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return True


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--template", default="docs/report_template.md")
    ap.add_argument("--docs-out", default="docs/REPORT.md",
                    help="committed copy of the report; '' to skip. Charts are copied next to it (docs/figures/report/); "
                         "figures showing FFHQ faces stay in <output_root>/report/figures and are only linked (not committed)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    root = get_paths(cfg).root
    rep = root / "report"
    (rep / "tables").mkdir(parents=True, exist_ok=True)
    (rep / "figures").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    tau = cfg["eval"].get("tau")
    tau_info = {}
    for f in [Path(cfg["_repo_root"]) / "data" / "tau_calibration.json", root / "tau.json"]:
        if f.exists():
            tau_info = json.loads(f.read_text())
            break
    tau = float(tau if tau is not None else tau_info["tau"])

    exps = sorted(p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith("_")
                  and not p.name.endswith("_quick") and (p / "inpaint_out" / "results.csv").exists())
    res = {e: load_results(root / e, tau) for e in exps}
    nsfw = {e: pd.read_csv(root / e / "inpaint_out" / "results.csv").query("branch == 'cloak'").nsfw.mean() for e in exps}
    mans = {e: (pd.read_csv(root / e / "cloak_out" / "manifest.csv") if (root / e / "cloak_out" / "manifest.csv").exists()
                else None) for e in exps}
    assert "E0_baseline" in res, f"E0_baseline not finished under {root}"
    e0 = res["E0_baseline"]
    clean0, rnd0 = e0[e0.branch == "clean"], e0[(e0.branch == "random") & (e0.eps == 16)]

    # ---- main table: clean / random references, then every E*/C* experiment's cloak branch at ε=16
    rows = [ref_row("clean", clean0), dict(ref_row("random", rnd0), change="±16/255 sign noise",
                                           delta_vs_clean=paired_delta(rnd0, clean0, rng)[0],
                                           nsfw_rate=pd.read_csv(root / "E0_baseline/inpaint_out/results.csv")
                                           .query("branch == 'random'").nsfw.mean())]
    rows[0]["nsfw_rate"] = pd.read_csv(root / "E0_baseline/inpaint_out/results.csv").query("branch == 'clean'").nsfw.mean()
    main_order = sorted((e for e in exps if re.match(r"^[EC]", e)), key=lambda e: (e[0] != "E", e))  # E0…E5, then C*
    for e in main_order:
        r = res[e]
        man = mans[e][mans[e].eps == 16] if mans[e] is not None else None
        rows.append(dict(summarize(e, change_of(root / e), r[(r.branch == "cloak") & (r.eps == 16)], clean0, rnd0,
                                   man, rng), nsfw_rate=nsfw[e]))
    main = pd.DataFrame(rows)
    main.round(4).to_csv(rep / "tables" / "main.csv", index=False)
    (rep / "tables" / "main.md").write_text(md_table(main, MAIN_COLS))

    # ---- robustness: R_<base>_<what>, each against its own clean / random branches
    rrows = []
    for e in exps:
        if not e.startswith("R_"):
            continue
        r, man = res[e], mans[e]
        for eps in sorted(r[r.branch == "cloak"].eps.unique()):
            cl = r[r.branch == "clean"]
            rn = r[(r.branch == "random") & (r.eps == eps)]
            s = summarize(f"{e} (ε={eps})", "", r[(r.branch == "cloak") & (r.eps == eps)], cl,
                          rn if len(rn) else None, man[man.eps == eps] if man is not None else None, rng)
            rrows.append(dict(s, clean_mean_sim=cl.sim.mean(), random_mean_sim=rn.sim.mean() if len(rn) else np.nan))
    robust = pd.DataFrame(rrows)
    rob_cols = {"exp": "run", "n": "n", "clean_mean_sim": "clean sim", "random_mean_sim": "random sim",
                "mean_sim": "cloak sim ↓", "delta_vs_clean": "Δ vs clean ↑", "delta_vs_random": "Δ vs random [95% CI] ↑",
                "protection_rate": "protection ↑", "psnr_cloak": "PSNR (dB)"}
    if len(robust):
        robust.round(4).to_csv(rep / "tables" / "robustness.csv", index=False)
    (rep / "tables" / "robustness.md").write_text(md_table(robust, rob_cols) if len(robust) else "_No robustness runs (R_*) found._\n")

    # ---- figures
    figs = {}
    bar_figure(main, clean0.sim.mean(), rnd0.sim.mean(), "mean_sim",
               "mean ArcFace sim, deepfake vs. clean photo (lower is better)",
               rep / "figures" / "mean_sim.png")
    figs["mean_sim"] = "figures/mean_sim.png"
    bar_figure(main, clean0.protected.mean(), rnd0.protected.mean(), "protection_rate",
               f"protection rate: no face or sim < τ = {tau:.3f}", rep / "figures" / "protection.png",
               pct=True)
    figs["protection"] = "figures/protection.png"
    if convergence_figure(root, ["E0_baseline", "E3_momentum"], rep / "figures" / "convergence_E0_E3.png"):
        figs["convergence"] = "figures/convergence_E0_E3.png"
    for e in exps:
        for f in sorted((root / e / "inpaint_out" / "figures").glob("grid_*.png"))[:1]:
            shutil.copy(f, rep / "figures" / f"grid_{e}.png")
            figs[f"grid:{e}"] = f"figures/grid_{e}.png"
        f = root / e / "cloak_out" / "figures" / "sanity_eps16.png"
        if f.exists():
            shutil.copy(f, rep / "figures" / f"sanity_{e}.png")
            figs[f"sanity:{e}"] = f"figures/sanity_{e}.png"

    # ---- values
    cl = main.set_index("exp")
    best = main[main.exp.str.match(r"^[EC]")].sort_values("mean_sim").iloc[0]
    values = dict(tau=f"{tau:.3f}", tau_n_faces=str(tau_info.get("n_faces", "?")),
                  tau_n_pairs=str(tau_info.get("n_pairs", "?")), tau_far=f"{tau_info.get('far', 0.01):.0%}",
                  n_images=str(clean0.img_id.nunique()), n_outputs=str(len(clean0)),
                  clean_sim=fmt(cl.loc["clean", "mean_sim"]), random_sim=fmt(cl.loc["random", "mean_sim"]),
                  e0_sim=fmt(cl.loc["E0_baseline", "mean_sim"]),
                  e0_vs_random=f"{cl.loc['E0_baseline', 'delta_vs_random']:+.3f}",
                  best_exp=best.exp, best_sim=fmt(best.mean_sim), best_vs_random=f"{best.delta_vs_random:+.3f}",
                  best_protection=f"{best.protection_rate:.0%}", experiments=", ".join(exps))
    for _, r in pd.concat([main, robust], ignore_index=True).iterrows():   # per-row values: sim:E1_pattern_target, ...
        k = r.exp.replace(" (ε=", "@").rstrip(")")                          # robustness rows: R_E0_eps8@8
        values[f"sim:{k}"] = fmt(r.mean_sim)
        values[f"prot:{k}"] = "—" if pd.isna(r.get("protection_rate")) else f"{r.protection_rate:.0%}"
        if not pd.isna(r.get("delta_vs_random", np.nan)):
            values[f"dr:{k}"] = f"{r.delta_vs_random:+.3f} [{r.delta_vs_random_lo:+.3f}, {r.delta_vs_random_hi:+.3f}]"
        if not pd.isna(r.get("nsfw_rate", np.nan)):
            values[f"nsfw:{k}"] = f"{r.nsfw_rate:.0%}"
        if not pd.isna(r.get("psnr_cloak", np.nan)):
            values[f"psnr:{k}"] = f"{r.psnr_cloak:.1f}"
    for e in exps:   # cloak-stage numbers per experiment (ε = 16): latent distance, surrogate cosine, seconds/image
        m = mans[e]
        if m is None:
            continue
        m16 = m[m.eps == 16]
        values[f"ld:{e}"] = fmt(m16.latent_dist_cloak.mean(), 1)
        values[f"spi:{e}"] = f"{m16.seconds_per_image.mean():.0f}"
        if "id_cos_cloak" in m16:
            values[f"idcos:{e}"] = fmt(m16.id_cos_cloak.mean())
            values[f"idcos_clean:{e}"] = fmt(m16.id_cos_clean.mean())
    if mans.get("E0_baseline") is not None and "latent_dist_random" in mans["E0_baseline"]:
        values["ld:random"] = fmt(mans["E0_baseline"].query("eps == 16").latent_dist_random.mean(), 1)

    # ---- legacy Colab baseline (archived from the team's shared folder): results/baseline_pgd_gray/
    leg = Path(cfg["_repo_root"]) / "results" / "baseline_pgd_gray"
    if (leg / "inpaint_out" / "results.csv").exists():
        lr = pd.read_csv(leg / "inpaint_out" / "results.csv")
        lr = lr[~lr.nsfw.astype(bool)]
        lc, lk, lrn = (lr[lr.branch == b_] for b_ in ("clean", "cloak", "random"))
        lm = pd.read_csv(leg / "cloak_out" / "manifest.csv")
        allr = pd.read_csv(leg / "inpaint_out" / "results.csv")
        values.update({
            "legacy_n_images": str(allr.img_id.nunique()), "legacy_n_prompts": str(allr.prompt_id.nunique()),
            "legacy_n_seeds": str(allr.seed.nunique()), "legacy_n_outputs": str(len(allr)),
            "legacy_n_nsfw": str(int(allr.nsfw.sum())), "legacy_iters": str(int(lm[lm.eps == 16].n_iters.iloc[0])),
            "legacy_clean_sim": fmt(lc.sim.mean()), "legacy_cloak_sim": fmt(lk.sim.mean()),
            "legacy_random_sim": fmt(lrn.sim.mean()),
            "legacy_n_cloak": str(len(lk)), "legacy_n_random": str(len(lrn)),
            "legacy_dc": f"{paired_delta(lk, lc, rng)[0]:+.3f}", "legacy_dr_rand_clean": f"{paired_delta(lrn, lc, rng)[0]:+.3f}",
            "legacy_dr": f"{paired_delta(lk, lrn, rng)[0]:+.3f}",
            "legacy_tau": fmt(json.loads((leg / "inpaint_out" / "config.json").read_text()).get("tau_far1", np.nan)),
        })
    (rep / "values.json").write_text(json.dumps(values, indent=2))

    # ---- render the template
    tpl = Path(args.template)
    tpl = tpl if tpl.is_absolute() else Path(cfg["_repo_root"]) / tpl
    text = re.sub(r"<!--.*?-->\s*", "", tpl.read_text(), flags=re.S)   # drop the template's own comments
    tables = {"main": (rep / "tables" / "main.md").read_text(), "robustness": (rep / "tables" / "robustness.md").read_text()}

    def render(fig_link) -> str:
        def sub(m):
            kind, key = m.group(1), m.group(2)
            if kind == "table":
                return tables.get(key, f"_(missing table {key})_")
            if kind == "fig":
                return fig_link(key) if key in figs else f"_(missing figure {key})_"
            return values.get(key, f"_(missing value {key})_")
        out = re.sub(r"\{\{(value):([^{}]+)\}\}", sub, text)      # values first (they may sit inside a fig name)
        return re.sub(r"\{\{(table|fig):([^{}]+)\}\}", sub, out)

    out = render(lambda k: f"![{k}]({figs[k]})")
    (rep / "report.md").write_text(out)

    if args.docs_out:
        docs_md = Path(args.docs_out)
        docs_md = docs_md if docs_md.is_absolute() else Path(cfg["_repo_root"]) / docs_md
        chart_dir = docs_md.parent / "figures" / "report"
        chart_dir.mkdir(parents=True, exist_ok=True)

        def docs_link(k):
            src = rep / figs[k]
            if k.startswith(("grid:", "sanity:")):   # contains FFHQ faces (CC BY-NC-SA): link the local copy only
                rel = os.path.relpath(src, docs_md.parent)
                return (f"[{k} — figure with FFHQ faces, not committed; generate it with "
                        f"`python scripts/make_report.py`]({rel})")
            shutil.copy(src, chart_dir / src.name)
            return f"![{k}]({os.path.relpath(chart_dir / src.name, docs_md.parent)})"

        docs_md.write_text(render(docs_link).replace("*CRP II, draft generated from `runs/`.",
                                                     "*CRP II. Generated by `scripts/make_report.py` from `runs/`; do not edit by hand.", 1))
        print(f"Wrote {docs_md} (+ charts in {chart_dir})")
    missing = re.findall(r"_\(missing [^)]*\)_", out)
    print(main[["exp", "n", "mean_sim", "delta_vs_clean", "delta_vs_random", "protection_rate"]].round(3).to_string(index=False))
    print(f"\nWrote {rep}/report.md, tables/, figures/" + (f" — {len(missing)} MISSING placeholders: {missing}" if missing else ""))
