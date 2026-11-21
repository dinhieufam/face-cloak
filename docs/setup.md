# Setup

The same code runs on Google Colab and on a local CUDA workstation. Two things differ, and both are detected
automatically in `src/facecloak/env.py`:

| | Colab (T4, 15 GB) | Local workstation (A5000, 24 GB) |
|---|---|---|
| Output root | `/content/face-cloak/runs` (= `<repo>/runs`; zip it before the runtime is recycled) | `<repo>/runs` |
| PGD batch size (`cloak.batch_size: auto`) | 4 | 8 |
| Precision | fp16 autocast on CUDA | fp16 autocast on CUDA |

Override the output root with `paths.output_root=/some/dir` on the command line, or with the environment
variable `FACECLOAK_OUTPUT_ROOT`. Results stay comparable across machines because everything that defines a run
is in git: the configs and the fixed identity threshold `eval.tau` in `configs/base.yaml`. CPU and Apple-Silicon (MPS) machines also work, in fp32 and much more slowly,
which is enough for debugging.

## Google Colab

1. Open `notebooks/colab_quickstart.ipynb` from GitHub in Colab (*File → Open notebook → GitHub*).
2. *Runtime → Change runtime type → T4 GPU*.
3. Set `EXP` (and optionally `OVERRIDES`), then *Run all*.

The notebook clones this repo to `/content/face-cloak`, installs it with `pip install -e ".[eval]"`, and calls
the same functions as the scripts. When the repo changes, re-run the clone cell: it does a `git pull`.

## Local workstation (NVIDIA A5000)

```bash
git clone https://github.com/dinhieufam/face-cloak.git && cd face-cloak
python -m venv .venv && source .venv/bin/activate      # or: conda create -n facecloak python=3.11
pip install torch --index-url https://download.pytorch.org/whl/cu124   # match your CUDA driver
pip install -e ".[eval,dev]"
pip install --no-deps facenet-pytorch                    # E5 only: its pins conflict with current torch
pytest                                                   # CPU unit tests
```

On a Slurm cluster, `slurm/run_experiment.sbatch <config> [k=v ...]` runs one experiment on one GPU and
`slurm/run_array.sbatch <list.txt>` runs one line of a list per array task (`--array=1-N%2`). Both source
`slurm/env.sh`, which sets the env path and keeps caches off `/home` (`HF_HOME`, `KAGGLEHUB_CACHE`, `TORCH_HOME`).

`[eval]` installs `uniface[gpu]`, which pulls `onnxruntime-gpu`. That needs CUDA 12 and cuDNN 9; the CUDA wheels
of PyTorch ship both. If UniFace reports that it runs on CPU, evaluation still works (about 0.1–0.3 s per image).
**Never install `onnxruntime` and `onnxruntime-gpu` together**, because the CPU build shadows the GPU one.
On a machine without an NVIDIA GPU, use `pip install -e ".[eval-cpu,dev]"` instead.

Long runs over SSH: use `tmux` or `nohup ... &`. Every stage resumes from what is already on disk.

## Credentials

- **Kaggle** (FFHQ images): public downloads often work without logging in. If you get a 401 or 403 error,
  create a token at kaggle.com → *Settings → API → Create New Token*. Then either
  - Colab: add `KAGGLE_USERNAME` and `KAGGLE_KEY` in the 🔑 *Secrets* panel and enable notebook access, or
  - local: put `kaggle.json` in `~/.kaggle/` (chmod 600), or export `KAGGLE_USERNAME` and `KAGGLE_KEY`.
- **Hugging Face**: the SD 1.5 inpainting weights are public. Setting `HF_TOKEN` only raises rate limits.
  On a shared workstation, set `HF_HOME` to a disk with about 10 GB free.

## Running

```bash
# Step 0 is done: τ = 0.1493 is fixed in configs/base.yaml (eval.tau). Only re-run to reproduce it:
# python scripts/calibrate_tau.py --config configs/base.yaml     # -> <output_root>/tau.json

# report: every table/figure + the report draft from runs/ -> <output_root>/report/
python scripts/make_report.py

# a whole experiment, or stage by stage
python scripts/run_experiment.py   --config configs/experiments/E1_pattern_target.yaml
python scripts/run_cloak.py        --config configs/experiments/E1_pattern_target.yaml
python scripts/run_inpaint_eval.py --config configs/experiments/E1_pattern_target.yaml

# overrides: any key from configs/base.yaml, dotted
python scripts/run_experiment.py --config configs/experiments/E0_baseline.yaml quick_test=true
python scripts/run_cloak.py --config configs/experiments/E3_momentum.yaml cloak.eps=[8,16] cloak.batch_size=4

# inspect the resolved config and output folders without running anything
python scripts/run_cloak.py --config configs/experiments/E2_mask_aware.yaml --print-config

# comparison table across all finished experiments -> <output_root>/comparison.csv
python scripts/compare_experiments.py
```

`quick_test=true` writes to `<exp>_quick/` so a smoke test never mixes with a real run.

## Output layout

```
<output_root>/
  tau.json                              output of calibrate_tau.py (the value used is eval.tau in base.yaml)
  report/                               make_report.py: tables/, figures/, report.md
  comparison.csv                        scripts/compare_experiments.py
  _shared/images/<img_id>/clean.png     512×512 preprocessed inputs (all experiments)
  _shared/masks/dilate<d>/              <id>_face.png (white = keep), <id>_inpaint.png (white = regenerate)
  _shared/inpaint_<hash>/outputs/       clean- and random-branch inpaintings, reused by every experiment
                                        (<hash> = model, prompts, steps, guidance, ...; changing them starts a new cache)
  <exp>/cloak_out/                      config.json, manifest.csv, images/<id>/{cloak,random}_eps<E>.png,
                                        loss_history_*.npz, figures/
  <exp>/inpaint_out/                    config.json, generations.csv, results.csv, summary.csv, paired.csv,
                                        tau.json, outputs/ (cloak branch), figures/
```

All paths stored in CSVs are relative to `<output_root>`, so a run folder can be moved or zipped as a whole.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `NaN latent loss` | `model.fp16=false` (slower, fp32) |
| A Kaggle download hangs | Each file is retried after 90 s (`fetch_ffhq`); persistent failures are skipped and logged |
| CUDA out of memory in Stage 1 | Handled automatically: the batch size is halved and the batch retried. Or set `cloak.batch_size=2`. |
| A config change seems to have no effect | Both stages skip work that is already on disk. Give the run a new `exp_name`, or delete `<exp>/`. |
| Many black outputs | The safety checker flagged them. They are excluded from metrics and reported as `nsfw_rate`. Prompt 0 is flagged least. |
| `No face found by MediaPipe` | The image is skipped for every branch. Inspect `_shared/images/<id>/clean.png`. |
