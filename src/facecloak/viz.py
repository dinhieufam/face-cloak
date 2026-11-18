"""Figures. Every function saves a PNG and closes the figure (works headless on a workstation);
notebooks display the saved files with `facecloak.viz.show(path)`."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt  # falls back to the Agg backend automatically on headless machines
from PIL import Image


def show(path) -> None:
    """Display a saved figure inline in Jupyter/Colab (no-op outside notebooks)."""
    try:
        from IPython.display import Image as IPImage, display
        display(IPImage(filename=str(path)))
    except Exception:
        pass


def _save(fig, out) -> None:
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)


def plot_convergence(histories: dict, out, untargeted: bool = False) -> None:
    """histories: {"eps16": (iters+1, n_images) array of latent MSE to the target}."""
    fig = plt.figure(figsize=(6, 3.5))
    for name, h in sorted(histories.items()):
        h = h / h[0:1]  # normalise each image by its value at the start
        plt.plot(h.mean(1), label=name.replace("eps", "ε=") + "/255")
    plt.yscale("log")
    plt.xlabel("PGD iteration")
    plt.ylabel("latent distance to clean / initial" if untargeted else "latent loss / initial (mean)")
    plt.title("PGD convergence")
    plt.legend()
    plt.grid(alpha=.3)
    _save(fig, out)


def plot_cloak_sanity(df: pd.DataFrame, paths, roundtrip, device, out, show_eps: int, n_show: int = 4) -> None:
    """clean | cloak | |δ| amplified | VAE reconstruction of clean / cloak / random."""
    from .data import load_batch, to_pil
    sel = df[df.eps == show_eps].head(n_show)
    if sel.empty:
        return
    has_rnd = "random_path" in sel and sel["random_path"].notna().all()
    cols = ["clean", f"cloak ε={show_eps}", "|δ| (amplified)", "VAE recon: clean", "VAE recon: cloak"] + \
           (["VAE recon: random"] if has_rnd else [])
    fig, axes = plt.subplots(len(sel), len(cols), figsize=(2.4 * len(cols), 2.5 * len(sel)), squeeze=False)
    for r, (_, row) in enumerate(sel.iterrows()):
        ps = [row.clean_path, row.cloak_path] + ([row.random_path] if has_rnd else [])
        xs = load_batch([paths.abs(p) for p in ps], device)
        recon = roundtrip(xs)
        amp = ((xs[1] - xs[0]).abs() * 255 / show_eps).clamp(0, 1)
        panels = [xs[0], xs[1], amp, recon[0], recon[1]] + ([recon[2]] if has_rnd else [])
        for c, p in enumerate(panels):
            axes[r, c].imshow(to_pil(p))
            axes[r, c].axis("off")
            if r == 0:
                axes[r, c].set_title(cols[c], fontsize=9)
        axes[r, 0].text(-8, xs.shape[-1] / 2, row.img_id, rotation=90, va="center", ha="right", fontsize=8)
    plt.tight_layout()
    _save(fig, out)


def plot_similarity(imp, genuine, ev: pd.DataFrame, summary: pd.DataFrame, tau: float, out) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    bins = np.linspace(-0.3, 1.0, 50)
    if len(imp):
        ax[0].hist(imp, bins, alpha=.5, density=True, label="impostor (different people)")
    if len(genuine):
        ax[0].hist(genuine, bins, alpha=.5, density=True, label="genuine: clean-branch edits")
    eps_max = ev.eps.max()
    for b in ["cloak", "random"]:
        s = ev[(ev.branch == b) & (ev.eps == eps_max) & ev.face_found.astype(bool)].sim
        if len(s):
            ax[0].hist(s, bins, histtype="step", lw=2, density=True, label=f"{b} ε={eps_max} edits")
    ax[0].axvline(tau, color="k", ls="--", label=f"τ={tau:.2f}")
    ax[0].set_xlabel("ArcFace cosine similarity to source")
    ax[0].legend(fontsize=8)
    ax[0].set_title("Similarity distributions")

    c = summary.reset_index()
    base = c[c.branch == "clean"].protection_rate.mean() if (c.branch == "clean").any() else np.nan
    for b, mk in [("cloak", "o-"), ("random", "s--")]:
        d = c[c.branch == b]
        if len(d):
            ax[1].plot(d.eps, d.protection_rate, mk, label=b)
    ax[1].axhline(base, color="gray", ls=":", label="clean control")
    ax[1].set_xlabel("ε (/255)")
    ax[1].set_ylabel("protection success rate")
    ax[1].set_ylim(-0.02, 1.02)
    ax[1].set_title("Protection vs ε (no breaker)")
    ax[1].legend()
    ax[1].grid(alpha=.3)
    plt.tight_layout()
    _save(fig, out)


def plot_edit_grid(res: pd.DataFrame, paths, cfg: dict, fig_dir, n_show: int = 4) -> None:
    """source | kept face | edited clean / cloak / random for the first prompt and seed."""
    pid, seed = cfg["inpaint"]["prompt_ids"][0], cfg["inpaint"]["seeds"][0]
    eps_show = int(res[res.branch == "cloak"].eps.max())
    img_ids = sorted(res.img_id.unique())[:n_show]
    cols = ["source (clean)", "kept face", "edit: clean", f"edit: cloak ε={eps_show}", f"edit: random ε={eps_show}"]
    fig, axes = plt.subplots(len(img_ids), len(cols), figsize=(2.6 * len(cols), 2.7 * len(img_ids)), squeeze=False)
    for r, i in enumerate(img_ids):
        def pick(branch, eps):
            q = res[(res.img_id == i) & (res.branch == branch) & (res.eps == eps) &
                    (res.prompt_id == pid) & (res.seed == seed)]
            return q.iloc[0] if len(q) else None
        clean = pick("clean", 0)
        panels = [(Image.open(paths.abs(clean.input_path)) if clean is not None else None, i),
                  (Image.open(paths.masks / f"{i}_face.png"), f"dilate={paths.dilate_px}px")]
        for b, e in [("clean", 0), ("cloak", eps_show), ("random", eps_show)]:
            q = pick(b, e)
            if q is None:
                panels.append((None, "n/a"))
            else:
                t = "no face" if not q.face_found else f"sim={q.sim:.2f}"
                panels.append((Image.open(paths.abs(q.out_path)), t + (" (nsfw)" if q.nsfw else "")))
        for c_, (im, t) in enumerate(panels):
            a = axes[r, c_]
            a.axis("off")
            if im is not None:
                a.imshow(im, cmap="gray" if c_ == 1 else None)
            a.set_title((cols[c_] + "\n" if r == 0 else "") + t, fontsize=8)
    prompt = cfg["inpaint"]["prompts"][pid]
    plt.suptitle(f"[{cfg['exp_name']}] prompt {pid}: “{prompt}”, seed {seed}", fontsize=9)
    plt.tight_layout()
    _save(fig, Path(fig_dir) / f"grid_p{pid}_s{seed}_eps{eps_show}.png")
