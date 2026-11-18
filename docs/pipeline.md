# Pipeline and threat model

## Threat model

- **Defender** (photo owner) adds a cloak δ, ‖δ‖∞ ≤ ε, before posting the photo. The defender knows the
  attacker's model family (SD 1.5 inpainting, white-box VAE) but not the exact mask, prompt or seed.
- **Attacker** takes the posted 8-bit image, masks everything except the face, and inpaints a new scene
  with SD 1.5.
- **Success for the defender:** the face in the edited image is not matched to the original identity by
  ArcFace (cosine similarity below τ), or no face is found at all.

## Stage 1: cloak (`facecloak.cloak`, `facecloak.attacks`)

Only the VAE of `stable-diffusion-v1-5/stable-diffusion-inpainting` (about 80M parameters) is loaded.
Pixels live in [0,1], and the VAE receives `2x − 1`.

Baseline objective (PhotoGuard encoder attack):

    min_δ ‖E(x+δ) − E(x_gray)‖²   s.t.  ‖δ‖∞ ≤ ε,  x+δ ∈ [0,1]

- E is the **mean** of the encoder's latent distribution, so the loss is deterministic.
- Sign-PGD runs for 200 iterations with step α = ε/8. α is halved at 50% and again at 75% of the iterations.
  The start point is random, U[−ε, ε], seeded from (image id, ε, seed), so it does not depend on the batch.
- The backward pass uses the per-image *sum* of squared errors. Only the gradient's sign is used, and the sum
  keeps fp16 gradients from underflowing.
- The **best iterate** (lowest loss) is kept, then quantized to uint8. A real upload is 8-bit, so every number
  reported is measured after quantization.
- A **random ±ε** image with the same budget is saved as a control. Comparing against it shows whether the
  effect comes from the adversarial direction or just from adding noise.

Sanity check (`cloak_out/figures/sanity_eps*.png`): the VAE reconstruction of the cloak collapses toward a
gray, washed-out face, while the clean and random-noise images reconstruct normally.

Variants (E1–E4) are keyword arguments of `pgd_encoder_attack`. With the defaults, the function reproduces the
original notebook's update rule bit for bit; `tests/test_attacks.py` checks this.

## Stage 2: inpainting attack (`facecloak.inpaint`)

- The **mask** is MediaPipe Face Landmarker's `FACE_OVAL` polygon (36 landmarks, forehead to chin), computed once
  from `clean.png`. Every branch reuses it, so neither the cloak nor the noise can move the mask.
- There are three branches per image, and they all use the same mask, prompt and seeds:
  `clean` (no protection), `cloak` (this experiment), and `random` (noise control).
- Seed *s* uses its own `torch.Generator`, so it produces the same image whether it runs alone or in a batch.
- Outputs flagged by the safety checker come back black. They are recorded and **excluded** from metrics, so a
  black image is never counted as "protected".

### What the attacker's VAE actually sees

From `diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_inpaint.py`:

```python
masked_image = init_image * (mask_condition < 0.5)    # init_image in [-1, 1]
masked_image_latents = self._encode_vae_image(masked_image, generator=generator)
```

- With `strength = 1` (the default), the initial latents are pure noise. The photo therefore reaches the UNet
  **only** through `masked_image_latents`.
- The regenerated area is set to 0 in [−1, 1], which is **mid-gray (0.5) in [0, 1]**, not black. The encoder
  sees *the face on a gray background*, never the full photo.
- The pipeline does not paste the original pixels back. The face in the output is decoded by the VAE, so the
  pixel-level cloak never reaches the output directly; it acts only through the latent.

Consequences:

1. A cloak optimised on the full image spends part of its budget on pixels the attacker throws away.
2. To match the attacker, the loss must use `x·m + 0.5·(1 − m)` (`facecloak.masks.attacker_view`), not `x·m`,
   which would make the background black.

E2 tests exactly this.

## Stage 3: evaluation (`facecloak.evaluate`)

- [UniFace](https://github.com/yakhyo/uniface): RetinaFace (`retinaface_r50`) gives the face box and 5-point
  landmarks; ArcFace (`arcface_resnet`) gives an L2-normalised 512-d embedding.
- **Source embedding:** from `clean.png`. If an output has several faces, the one that overlaps the source face
  box most is used, so a background face that SD invented is not picked.
- `sim` is the cosine similarity between output and source. `sim_input` is the same measure on the attack
  *input*, which tells whether the cloak alone already changes identity before any editing.
- **Threshold τ** (`eval.tau_rule`):
  - `fixed`: `eval.tau`, or `<output_root>/tau.json` from `scripts/calibrate_tau.py` (FAR 1% on ~45k impostor
    pairs). Use this when comparing experiments.
  - `far1` or `eer`: estimated from this run's own images. This is unreliable with few images; with 2 images
    there is a single impostor pair.
- **Protected** ⇔ no face found, or `sim < τ`.
- **Reported:** `mean_sim`, `protection_rate`, `face_rate`, `nsfw_rate` (`summary.csv`), and the paired
  Δ = sim(clean branch) − sim(cloak branch) for the same image, prompt and seed (`paired.csv`).
