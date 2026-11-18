# Baseline results so far (legacy notebooks, 2026-11-18)

Source: `results/baseline_pgd_gray/`, produced on Colab by `notebooks/legacy/`.
Setup: 2 FFHQ images (`ffhq_00004`, `ffhq_25247`), ε = 16/255, gray target, 3 prompts × 3 seeds.

| branch | n (after NSFW removal) | mean sim | sim of the input | paired Δsim vs. clean |
|---|---|---|---|---|
| clean | 18 | 0.843 | 1.000 | — |
| cloak ε=16 | 14 | **0.696** | 0.944 | 0.135 ± 0.069 |
| random ε=16 | 18 | 0.767 | 0.921 | 0.077 ± 0.032 |

Per image: `ffhq_00004` scores 0.90 (clean) / 0.71 (cloak) / 0.83 (random); `ffhq_25247` scores 0.79 / 0.69 / 0.70.

## Caveats: why these are not the reference numbers

1. **The evaluated ε=16 cloaks ran only 50 PGD iterations.** They come from a `QUICK_TEST` run
   (`manifest.csv` has `n_iters=50`). The 200-iteration run has so far produced ε=8 for 4 images only.
2. **τ = −0.018 is meaningless.** Two images give a single impostor pair, so `protection_rate = 0` everywhere
   says nothing. Even 10 images give only 45 pairs, and their 99% quantile is essentially the maximum.
   **Fix:** `scripts/calibrate_tau.py` estimates τ once from about 300 FFHQ faces, and that value stays fixed for
   every experiment. For reference, ArcFace cosine thresholds in the literature are typically 0.3–0.4.
3. **NSFW filtering biases the mean.** 4 of 54 outputs were blacked out by the safety checker, 3 of them in the
   cloak branch of `ffhq_00004`, so the cloak mean is computed over 14 outputs instead of 18. **Fix:** use prompt 0
   only (flagged least) and report `nsfw_rate`.
4. **The cloak beats random noise by only about 0.06** (paired Δ 0.135 vs. 0.077). Most of the drop in similarity
   comes from adding *any* perturbation, not from the adversarial direction.
5. **Why a gray target only blurs the face.** Pulling the latent toward gray removes colour, contrast and
   high-frequency detail, but face geometry (eye shape, nose, bone structure) survives, and ArcFace is fairly robust
   to colour and contrast. In addition, the cloak is optimised on the **full image** while the attacker's encoder
   sees only **the face on gray** (see [pipeline.md](pipeline.md)), so part of the budget is wasted on the background.
   This motivates E1 and E2.

E0 in [experiments.md](experiments.md) re-runs this baseline correctly: 200 iterations, 10 images, fixed τ,
prompt 0, seeds {0, 1}.
