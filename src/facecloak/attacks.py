"""PGD encoder attack on the Stable Diffusion VAE (PhotoGuard-style) and its variants.

Baseline objective (targeted):
    min_δ ‖E(x+δ) − E(x_target)‖²   s.t. ‖δ‖∞ ≤ ε, x+δ ∈ [0,1]

Variants, all switched by keyword arguments (defaults reproduce the original notebook exactly):
    mask          optimise what the inpainting pipeline's VAE sees: attacker_view(x+δ, m)          (E2)
    mask_jitter_px  random dilate/erode of that mask every step (EOT over the attacker's mask)   (E2)
    momentum      MI-FGSM: g ← μ·g + ∇/mean|∇|, step along sign(g)                             (E3)
    untargeted    maximise ‖E(x+δ) − E(x)‖² instead (caller passes z_tgt = E(x))                (E4)
    objective     any per-image loss f(x_in) to minimise instead of the latent distance, e.g. the
                  identity loss of facecloak.identity (E5); `encode`/`z_tgt` are then unused
"""
from __future__ import annotations

import random
from typing import Callable

import torch

from . import env
from .masks import attacker_view, jitter_mask

Encoder = Callable[[torch.Tensor], torch.Tensor]


def load_vae(model_id: str, device: str):
    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae").to(device).eval()
    vae.requires_grad_(False)
    return vae


def make_encoder(vae, device: str, fp16: bool) -> Encoder:
    """x in [0,1] -> mean of the VAE latent distribution (deterministic), returned in fp32."""
    def encode_mean(x01: torch.Tensor) -> torch.Tensor:
        with env.autocast(device, fp16):
            z = vae.encode(x01 * 2 - 1).latent_dist.mean
        return z.float()
    return encode_mean


def make_roundtrip(vae, device: str, fp16: bool):
    """x in [0,1] -> VAE decode(encode(x)) in [0,1] (for sanity plots)."""
    @torch.no_grad()
    def roundtrip(x01: torch.Tensor) -> torch.Tensor:
        with env.autocast(device, fp16):
            y = vae.decode(vae.encode(x01 * 2 - 1).latent_dist.mean).sample
        return ((y.float() + 1) / 2).clamp(0, 1)
    return roundtrip


def latent_mse(z: torch.Tensor, z_tgt: torch.Tensor) -> torch.Tensor:
    """Per-image mean squared error in latent space."""
    return ((z - z_tgt) ** 2).flatten(1).mean(1)


def pgd_encoder_attack(encode: Encoder, x: torch.Tensor, z_tgt: torch.Tensor, eps: float, n_iters: int,
                       step_frac: float, init: torch.Tensor | None = None, *, mask: torch.Tensor | None = None,
                       mask_jitter_px: int = 0, momentum: float = 0.0, untargeted: bool = False,
                       rng: random.Random | None = None,
                       objective: Callable[[torch.Tensor], torch.Tensor] | None = None):
    """Sign-PGD with L∞ budget `eps` (on the [0,1] scale).

    x: (B,3,H,W) in [0,1]. init: U[-1,1] noise of x's shape (scaled by eps) or None for zero init.
    Step α = step_frac·ε, halved at 50% and again at 75% of the iterations. Keeps the best iterate per image.
    For the backward pass the per-image *sum* of squared errors is used: only the sign of the gradient matters
    and the sum keeps fp16 gradients from underflowing.

    Returns (best δ (B,3,H,W), history (n_iters+1, B) of the latent MSE to z_tgt, or of `objective`).
    With `mask` + jitter, the gradient uses a jittered mask but the final evaluation uses the true mask.
    """
    alpha0 = step_frac * eps
    if objective is not None and untargeted:
        raise ValueError("objective and untargeted are mutually exclusive")
    sgn = -1.0 if untargeted else 1.0          # untargeted: minimise −distance
    if untargeted and init is None:
        # ‖E(x+δ) − E(x)‖² has exactly zero gradient at δ = 0, so start from a small random point
        init = 0.1 * (torch.rand_like(x) * 2 - 1)
    delta = torch.zeros_like(x) if init is None else init.to(x.device) * eps
    delta = (x + delta).clamp(0, 1) - x
    best_delta = delta.clone()
    best_loss = torch.full((x.shape[0],), float("inf"), device=x.device)
    g = torch.zeros_like(x) if momentum > 0 else None
    history = []
    for it in range(n_iters + 1):
        last = it == n_iters
        delta.requires_grad_(not last)
        with torch.set_grad_enabled(not last):
            x_in = x + delta
            if mask is not None:
                m = mask if last else jitter_mask(mask, mask_jitter_px, rng)
                x_in = attacker_view(x_in, m)
            if objective is not None:
                dist = objective(x_in)              # per-image loss to minimise
                total = dist.sum()
            else:
                z = encode(x_in)
                sq = ((z - z_tgt) ** 2).flatten(1)
                dist = sq.mean(1)
                total = sq.sum()
            loss = sgn * dist
        if torch.isnan(loss).any():
            raise RuntimeError("NaN latent loss: set model.fp16=false and re-run.")
        with torch.no_grad():
            lv = loss.detach()
            better = lv < best_loss
            best_loss = torch.where(better, lv, best_loss)
            best_delta[better] = delta.detach()[better]
            history.append(dist.detach().cpu())
        if last:
            break
        grad, = torch.autograd.grad(sgn * total, delta)
        if g is not None:
            g = momentum * g + grad / grad.abs().mean(dim=(1, 2, 3), keepdim=True).clamp_min(1e-12)
            step = g.sign()
        else:
            step = grad.sign()
        alpha = alpha0 * (0.5 if it >= n_iters // 2 else 1.0) * (0.5 if it >= 3 * n_iters // 4 else 1.0)
        with torch.no_grad():
            delta = delta.detach() - alpha * step
            delta = delta.clamp(-eps, eps)
            delta = (x + delta).clamp(0, 1) - x
    return best_delta, torch.stack(history)
