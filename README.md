# face-cloak

**Adversarial face cloaks against diffusion-inpainting deepfakes.**

An attacker takes a photo of someone, keeps the face, and uses Stable Diffusion 1.5 inpainting to put that
person in a new scene (a police station, a nightclub, ...). A *cloak* is a small perturbation δ with
‖δ‖∞ ≤ ε, added to the photo before it is shared, so that the face in the resulting deepfake is
**no longer recognised as the same person** by a face recognizer (ArcFace).

- **Baseline:** PhotoGuard-style encoder attack. PGD pulls the VAE latent of the photo toward the latent of a flat gray image.
- **Goal:** find a cloak whose post-deepfake ArcFace similarity is **lower than the baseline at the same ε**.

```
clean.png ─► Stage 1: cloak (PGD on the SD VAE encoder) ─► cloak.png
                                                              │   MediaPipe face mask (from clean.png)
                                                              ▼
              Stage 2: SD 1.5 inpainting (keep face, regenerate the rest) ─► edited.png
                                                              ▼
              Stage 3: RetinaFace + ArcFace: sim = cos(edited, clean); protected ⇔ no face or sim < τ
```

## Quickstart

**Google Colab (T4):** open [`notebooks/colab_quickstart.ipynb`](notebooks/colab_quickstart.ipynb) in Colab,
choose `EXP`, and run all cells. Results are written to `runs/` inside the cloned repo; the last cell zips them for download.

**Local GPU workstation (e.g. NVIDIA A5000):**

```bash
git clone https://github.com/dinhieufam/face-cloak.git && cd face-cloak
python -m venv .venv && source .venv/bin/activate
pip install -e ".[eval,dev]"

# τ is already fixed in configs/base.yaml (eval.tau = 0.1493, from scripts/calibrate_tau.py)
python scripts/run_experiment.py --config configs/experiments/E0_baseline.yaml   # cloak -> inpaint -> eval
python scripts/compare_experiments.py                                            # table across experiments
```

Add `quick_test=true` at the end of any command for a few-minute smoke test (2 images, 50 PGD iterations).
Full instructions, credentials and troubleshooting: [docs/setup.md](docs/setup.md).

## Experiments

Every experiment is a small YAML file that inherits [`configs/base.yaml`](configs/base.yaml) and changes
**one** thing. The ε, mask, prompt, seeds and the 10 dev images stay the same.

| ID | Config | Change | Question |
|---|---|---|---|
| E0 | `E0_baseline.yaml` | none | Reference numbers (200 PGD iterations, 10 images) |
| E1 | `E1_pattern_target.yaml` | target = dense pattern | Does a structured target destroy identity better than gray? |
| E2 | `E2_mask_aware.yaml` | loss on the attacker's view + mask jitter | Does optimising what the encoder actually sees help? |
| E3 | `E3_momentum.yaml` | momentum PGD (MI-FGSM) | Does a better optimiser help? |
| E4 | `E4_untargeted.yaml` | push away from E(x) | Is a target needed at all? |
| E5 | (planned) | identity loss through the VAE | Optimise the metric we evaluate on |

Protocol and results table: [docs/experiments.md](docs/experiments.md).
Full report (method, results, robustness, threat model, limitations): [docs/REPORT.md](docs/REPORT.md),
regenerated from `runs/` by `python scripts/make_report.py`.
Current baseline numbers and their caveats: [docs/baseline_results.md](docs/baseline_results.md).

## Repository layout

```
configs/                 base.yaml (shared protocol) + experiments/E*.yaml (one change each)
src/facecloak/           the package
  attacks.py             PGD encoder attack and its variants (target, mask-aware, momentum, untargeted)
  cloak.py               Stage 1 runner (resumable, writes manifest.csv)
  inpaint.py             Stage 2: SD 1.5 inpainting attack (resumable, shared clean/random cache)
  evaluate.py            Stage 3: RetinaFace + ArcFace, threshold τ, summaries, cross-experiment table
  masks.py, data.py      MediaPipe face masks, attacker view, FFHQ download, image helpers
  config.py, env.py      YAML configs and output layout; Colab vs. local detection, device, batch size
scripts/                 command-line entry points (run_*, calibrate_tau, compare_experiments, make_targets)
notebooks/               colab_quickstart.ipynb; legacy/ holds the original exploratory notebooks
data/dev_ids.txt         the fixed 10-image FFHQ dev set
assets/targets/          target images for E1
results/                 archived results (baseline snapshot from the legacy notebooks)
docs/                    REPORT, setup, pipeline, experiments, baseline results
tests/                   CPU unit tests (pytest), no model downloads
```

## How it works

See [docs/pipeline.md](docs/pipeline.md) for the threat model, each stage, and one detail that drives E2:
the inpainting pipeline's VAE only ever sees **the face on a mid-gray background**, not the full photo.

## Data and models

- Faces come from [FFHQ](https://github.com/NVlabs/ffhq-dataset) through the Kaggle mirror
  `arnaud58/flickrfaceshq-dataset-ffhq`. FFHQ is released under **CC BY-NC-SA 4.0** and each image keeps its own
  Flickr license. The images under `results/` are FFHQ images and their derivatives, shared for non-commercial
  research only. Image ids are kept in every manifest for attribution.
- Models: `stable-diffusion-v1-5/stable-diffusion-inpainting` (diffusers), MediaPipe Face Landmarker,
  RetinaFace and ArcFace through [UniFace](https://github.com/yakhyo/uniface).
