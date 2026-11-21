<!-- Report template. Rendered by `python scripts/make_report.py` into docs/REPORT.md and <output_root>/report/report.md.
     Placeholders: {{table:main|robustness}}, {{fig:<name>}}, {{value:<key>}} (keys in <output_root>/report/values.json).
     Never type a result into this file: every number comes from runs/ (or results/baseline_pgd_gray/) through the script. -->

# Face cloaks against inpainting deepfakes: progress report

*CRP II, draft generated from `runs/`. Covers the earlier Colab work (shared-folder snapshot) and the runs on VISHC Server 2.*

## 1. The idea in brief

An attacker takes a public photo, keeps the person's face and has **Stable Diffusion 1.5 inpainting** generate a
new scene around it, for example a park, a police station or a nightclub. We want a **cloak**: a small change δ to
the photo, with every pixel moved by at most ε = 16/255, applied *before* the photo is posted. After the attacker's
edit, the face should no longer be recognised as the same person.

```
photo ─► cloak (PGD on the SD VAE encoder) ─► posted photo ─► attacker: SD 1.5 inpainting
         keeps the face (MediaPipe face oval), new background ─► deepfake ─► ArcFace: same person?
```

- **Metric:** `sim` is the ArcFace cosine similarity between the face in the deepfake and the clean photo. Lower is
  better.
- **Protected:** a deepfake counts as protected if no face is found or `sim < τ`.
- **Controls:** we always compare with the deepfake of the **clean** photo and with **random ±ε noise** of the same
  budget. A cloak is only interesting if it beats random noise. The paired Δ compares the same image, prompt and
  seed.
- **Pipeline property** (`docs/pipeline.md`): the attacker's VAE only sees the face on mid-gray, and the face in
  the output is VAE-decoded rather than pasted back. So perturbing the VAE changes the face the attacker gets.

## 2. Starting point: what was done before (Colab, team shared folder)

Archived in `results/baseline_pgd_gray/` (notebooks in `notebooks/legacy/`). Notes in `docs/baseline_results.md`.

- **Motivation.** Reproduce PhotoGuard's encoder attack (Salman et al., 2023) on an inpainting deepfake and check
  whether it protects identity, not just image quality.
- **Method.**
  - **Cloak:** PGD pulls the VAE latent of the photo toward the latent of a flat gray image, with the loss on the
    full image.
  - **Attack and evaluation:** SD 1.5 inpainting keeps the MediaPipe face oval. UniFace RetinaFace + ArcFace
    measure the similarity.
  - **Scale:** {{value:legacy_n_images}} images × {{value:legacy_n_prompts}} prompts × {{value:legacy_n_seeds}} seeds
    = {{value:legacy_n_outputs}} edits, on a Colab T4.
- **Result.**

  | branch | mean sim | paired Δ vs clean |
  |---|---|---|
  | clean | {{value:legacy_clean_sim}} | — |
  | random ±16/255 | {{value:legacy_random_sim}} | {{value:legacy_dr_rand_clean}} |
  | cloak | {{value:legacy_cloak_sim}} | {{value:legacy_dc}} |

  The cloak lowered similarity, but beat random noise by only {{value:legacy_dr}}.
- **Why these were not usable as reference numbers:**
  - The ε=16 cloaks ran only {{value:legacy_iters}} PGD iterations (a quick test).
  - Only {{value:legacy_n_images}} images were evaluated.
  - τ was estimated from a single impostor pair ({{value:legacy_tau}}, meaningless).
  - {{value:legacy_n_nsfw}} of {{value:legacy_n_outputs}} outputs were blacked out by the safety checker, most of
    them cloaks.
- **Code before this work:** the notebooks had been restructured into the `facecloak` package (configs, scripts,
  tests), but nothing had been run with it yet.

## 3. What I did on VISHC Server 2

### 3.1 Infrastructure and a fixed protocol

- **Motivation.** Get reference numbers that are reproducible and comparable with teammates' Colab runs, and that
  have a meaningful threshold.
- **Method.**
  - **Environment:** a conda env on `/mnt`, model caches off `/home`, and Slurm scripts in `slurm/`: one GPU per
    run, sweeps as arrays limited to 2 GPUs at a time.
  - **Fixed protocol:** {{value:n_images}} FFHQ dev images, prompt 0 ("park"), seeds {0, 1}, giving
    {{value:n_outputs}} edits per branch. Every experiment changes one setting relative to the baseline.
  - **Fixed τ:** calibrated once from {{value:tau_n_faces}} held-out FFHQ faces ({{value:tau_n_pairs}} impostor
    pairs, false-accept rate {{value:tau_far}}). It is stored in `configs/base.yaml`, so every git clone uses the
    same value.
  - **No shared Drive:** outputs go to `runs/` on every machine. Teammates clone the repo on Colab.
  - **Download fix:** Kaggle downloads occasionally hung forever, so they now time out and retry.
- **Result.**
  - τ = **{{value:tau}}**. This is lower than the 0.3–0.4 often quoted for ArcFace, but those values correspond to
    a much stricter false-accept rate. At 1%, 0.149 is the expected value.
  - A cloak takes about {{value:spi:E0_baseline}} s per image for latent losses and about {{value:spi:E5_identity}}
    s per image for the identity loss.

### 3.2 E0: the baseline, re-run properly

- **Motivation.** Get a reference that every other method is compared with.
- **Method.** PhotoGuard as before: gray target, full-image loss, 200 PGD iterations, step size halved twice,
  random start.
- **Result.**
  - Mean sim **{{value:sim:E0_baseline}}**, against {{value:clean_sim}} for clean and {{value:random_sim}} for random.
  - Δ vs random: {{value:dr:E0_baseline}}. The cloak's advantage over plain noise is small.
  - Protection: {{value:prot:E0_baseline}}.
  - Safety-checker flags: {{value:nsfw:E0_baseline}} of cloak outputs. The washed-out gray faces are often flagged
    as NSFW.
  - Sanity check: the VAE reconstruction of the cloaked photo collapses to a gray blur, so the attack does what it
    is designed to do. The identity still survives the attacker's diffusion model.

### 3.3 E1 / E1b: a different target image

- **Motivation.** A gray target only blurs the face, and the diffusion model then re-synthesises a plausible face
  of the same person. A structured target might break the face itself.
- **Method.** Pull the latent toward a dense colour pattern (E1) or a plain 8-px checkerboard (E1b, an ablation)
  instead of gray.
- **Result.**
  - E1: **{{value:sim:E1_pattern_target}}**, Δ vs random {{value:dr:E1_pattern_target}}. This is the best latent
    target.
  - E1b: {{value:sim:E1b_checker_target}}.
  - The colour pattern leaves textured artefacts across the face instead of a blur.

### 3.4 E2: optimise what the attacker actually sees

- **Motivation.** The attacker's VAE only sees the face on mid-gray, so a cloak optimised on the full photo wastes
  budget on the background.
- **Method.**
  - Compute the loss on `face·m + 0.5·(1−m)`, the attacker's view of the photo.
  - Randomly grow or shrink the mask by up to 10 px each step, so the cloak does not depend on one exact mask.
- **Result.**
  - Mean sim {{value:sim:E2_mask_aware}}, only slightly better than E0. The objective is still the weak gray target.
  - The latent distance is lower (latent dist {{value:ld:E2_mask_aware}} vs {{value:ld:E0_baseline}}) because only
    the face region is optimised.

### 3.5 E3: momentum PGD

- **Motivation.** Check whether the optimiser, rather than the objective, is the bottleneck.
- **Method.** MI-FGSM (Dong et al., 2018) with momentum 0.9.
- **Result.**
  - Mean sim {{value:sim:E3_momentum}}, a modest gain.
  - Momentum starts more slowly but ends at a lower latent loss (figure below).

{{fig:convergence}}

### 3.6 E4: untargeted

- **Motivation.** Instead of choosing a target, push the latent as far from the photo's own latent as possible.
- **Method.** Maximise ‖E(x+δ) − E(x)‖².
- **Result.**
  - The latent distance is by far the largest ({{value:ld:E4_untargeted}}, against {{value:ld:E0_baseline}} for E0),
    yet mean sim is only {{value:sim:E4_untargeted}}.
  - Distance in latent space is a poor proxy for identity.

### 3.7 E5: optimise identity directly (new)

- **Motivation.** E1–E4 all optimise a latent proxy. We should attack identity itself, but without cheating by
  optimising against the evaluator.
- **Method.**
  - Minimise the cosine similarity between the face after the VAE round trip, crop(D(E(x+δ))), and the clean face.
  - Use **FaceNet** (VGGFace2), a recognizer *different* from the ArcFace evaluator, so a good result shows transfer.
  - Code: `src/facecloak/identity.py`.
- **Result.**
  - Mean sim **{{value:sim:E5_identity}}**, half of E0. Δ vs random: {{value:dr:E5_identity}}.
  - The FaceNet cosine goes from {{value:idcos_clean:E5_identity}} to {{value:idcos:E5_identity}}, so the surrogate
    is completely fooled. Much of that transfers to ArcFace, but not all of it.
  - About 3–4× slower than the latent attacks, because gradients go through the VAE decoder.

### 3.8 Combinations of the winners

- **Motivation.** Check whether the best single changes add up.
- **Method.**
  - C1 = E5 + momentum.
  - C2 = E5 + the attacker's view with mask jitter.
  - C3 = E1 pattern + momentum.
- **Result.**
  - **C2 is the best configuration: {{value:sim:C2_identity_maskaware}}**, Δ vs random
    {{value:dr:C2_identity_maskaware}}.
  - C2 is the only method with non-zero protection at the strict τ ({{value:prot:C2_identity_maskaware}}).
  - Momentum adds nothing on top of E5 (C1, {{value:sim:C1_identity_momentum}}) or E1 (C3,
    {{value:sim:C3_pattern_momentum}}).

Example edits, showing the clean, cloaked and random inputs after the attack:

{{fig:grid:C2_identity_maskaware}} · {{fig:grid:E0_baseline}}

### 3.9 Robustness of E0, E5 and C2

- **Motivation.** Make sure the gains are not specific to one prompt, seed, mask or budget, and check a realistic
  upload: platforms recompress images as JPEG.
- **Method.** Keep the defender's cloak fixed and change one thing on the attacker's side:
  - prompts 1–2 (police station, nightclub)
  - seeds 0–3
  - a mask dilated by 10 px
  - ε = 8/255 (this one needs new cloaks)
  - JPEG quality 75 before the attack
- **Result.**
  - **Prompts and seeds:** the ranking holds. On prompts 1–2: C2 {{value:sim:R_C2_prompts12@16}}, E5
    {{value:sim:R_E5_prompts12@16}}, E0 {{value:sim:R_E0_prompts12@16}}.
  - **Mask dilation:** the advantage is unchanged.
  - **Half the budget (ε = 8):** C2 still reaches {{value:sim:R_C2_eps8@8}} (PSNR {{value:psnr:R_C2_eps8@8}} dB),
    far better than E0 at full budget.
  - **JPEG 75 is the weak point.** E0's advantage over random almost disappears ({{value:dr:R_E0_jpeg75@16}}). C2
    rises to {{value:sim:R_C2_jpeg75@16}} and E5 to {{value:sim:R_E5_jpeg75@16}}, still well below random noise.
  - **Run-to-run noise:** one E0 re-run gave {{value:sim:R_E0_eps8@16}} against {{value:sim:E0_baseline}}.
    Differences of about 0.02 are within noise.

{{table:robustness}}

## 4. Summary

{{table:main}}

{{fig:mean_sim}}

- **Every cloak beats random noise**, but the PhotoGuard baseline only barely does ({{value:dr:E0_baseline}}). Its
  gray target blurs the face, and the attacker's diffusion model restores the identity.
- **What matters most is what we optimise.**
  - Among latent targets, a structured target is better than gray: {{value:sim:E1_pattern_target}} vs
    {{value:sim:E0_baseline}}.
  - Optimising identity through the VAE with a different recognizer is better still: {{value:sim:E5_identity}}.
  - The optimiser (momentum, {{value:sim:E3_momentum}}) and the untargeted objective ({{value:sim:E4_untargeted}}) matter much less.
- **Best configuration: identity loss on the attacker's view (C2).** Mean sim {{value:sim:C2_identity_maskaware}},
  down from {{value:clean_sim}} without a cloak and {{value:sim:E0_baseline}} for the baseline. Protection at the
  strict τ is {{value:prot:C2_identity_maskaware}}.
- The gains hold across prompts, seeds, mask size and a halved budget. **JPEG recompression is the main weakness.**

## 5. Limitations

- **Small evaluation:** {{value:n_images}} images and one main prompt. Confidence intervals are wide, and
  differences of about 0.02 are within run-to-run noise.
- **One attacker** (SD 1.5 inpainting) **and one evaluator** (UniFace ArcFace R50). Other editors (SDXL, InstantID)
  and other recognizers are untested.
- **White-box assumptions:** the defender knows the attacker's VAE and that a face-oval mask is kept. Only the mask
  is jittered.
- **No adaptive attacker:** denoising, upscaling and diffusion purification are not tested. JPEG alone already
  weakens every cloak.
- **Strict τ:** at FAR 1%, almost no deepfake counts as "protected", so we mostly report similarity rather than a
  yes/no outcome.
- **NSFW exclusions:** up to a quarter of cloak outputs are blacked out by the safety checker and excluded, which
  changes n per method.
- **Cost:** the identity loss takes about {{value:spi:E5_identity}} s per image on an A5000, about 3.5× the latent
  attacks.

## 6. Next steps

1. **JPEG-robust cloaks:** train through a differentiable JPEG (expectation over transformations, EOT) in the C2
   loss, then re-test at quality 75 and 90.
2. **Close the transfer gap:** optimise against an ensemble of recognizers (FaceNet + an IResNet/AdaFace model),
   and evaluate with a recognizer that none of them includes.
3. **Combine identity and texture:** `λ·latent(pattern target) + identity`, and more PGD iterations for C2.
4. **Scale up the evaluation:** more images (e.g. 50–100), all three prompts, and a second attacker (SDXL
   inpainting).
5. **Adaptive attacks:** JPEG at other qualities, resizing, denoising and diffusion purification of the upload.
6. **Report:** turn this report into slides, and settle with the team on whether protection should also be
   reported at a less strict τ.

## References

- Salman et al., *Raising the Cost of Malicious AI-Powered Image Editing* (PhotoGuard), ICML 2023.
- Madry et al., *Towards Deep Learning Models Resistant to Adversarial Attacks* (PGD), ICLR 2018.
- Dong et al., *Boosting Adversarial Attacks with Momentum* (MI-FGSM), CVPR 2018.
- Deng et al., *ArcFace*, CVPR 2019; Deng et al., *RetinaFace*, CVPR 2020; UniFace (yakhyo/uniface).
- Schroff et al., *FaceNet*, CVPR 2015; Cao et al., *VGGFace2*, FG 2018.
- Rombach et al., *High-Resolution Image Synthesis with Latent Diffusion Models*, CVPR 2022 (SD inpainting).
- Karras et al., *A Style-Based Generator Architecture for GANs* (FFHQ, CC BY-NC-SA 4.0); MediaPipe Face Landmarker.
