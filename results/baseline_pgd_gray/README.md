# baseline_pgd_gray: legacy Colab snapshot (2026-11-18)

Copied from Google Drive (`MyDrive/CRPII/{cloak_out,inpaint_out}`) and produced by the two notebooks in
`notebooks/legacy/`. **Archived for reference; these are not the E0 reference numbers.** See
[docs/baseline_results.md](../../docs/baseline_results.md) for the numbers and their caveats.

- `cloak_out/`: PGD cloaks with a gray target. It holds `clean.png` for 35 FFHQ images. The ε=16 cloaks
  (2 images) ran only **50 iterations** (QUICK_TEST); the ε=8 cloaks (4 images) ran 200 iterations.
  `manifest.csv` uses the legacy layout: paths are relative to `cloak_out/`.
- `inpaint_out/`: SD 1.5 inpainting of `ffhq_00004` and `ffhq_25247` (clean / cloak ε=16 / random ε=16 ×
  3 prompts × 3 seeds), plus ArcFace results. The τ in `config.json` (−0.018) comes from a single impostor pair
  and is not meaningful.

The images are FFHQ (CC BY-NC-SA 4.0) and derivatives of it, for non-commercial research use only.
