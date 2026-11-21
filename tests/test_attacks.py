"""PGD encoder attack on a tiny random conv 'encoder' (CPU, no downloads)."""
import random

import pytest
import torch

from facecloak.attacks import latent_mse, pgd_encoder_attack


@pytest.fixture
def setup():
    torch.manual_seed(0)
    enc = torch.nn.Sequential(torch.nn.Conv2d(3, 8, 4, stride=4), torch.nn.Tanh(), torch.nn.Conv2d(8, 4, 2, stride=2))
    enc.requires_grad_(False)
    x = torch.rand(2, 3, 32, 32)
    init = torch.rand_like(x) * 2 - 1
    return enc, x, init


def legacy_pgd(encode_mean, x, z_tgt, eps, n_iters, step_frac, init=None):
    """Verbatim copy of pgd_encoder_attack from notebooks/legacy/crpII_cloak_pgd.ipynb (§5)."""
    alpha0 = step_frac * eps
    delta = torch.zeros_like(x) if init is None else init.to(x.device) * eps
    delta = (x + delta).clamp(0, 1) - x
    best_delta = delta.clone()
    best_loss = torch.full((x.shape[0],), float("inf"), device=x.device)
    history = []
    for it in range(n_iters + 1):
        last = (it == n_iters)
        delta.requires_grad_(not last)
        with torch.set_grad_enabled(not last):
            z = encode_mean(x + delta)
            sq = ((z - z_tgt) ** 2).flatten(1)
            loss = sq.mean(1)
        with torch.no_grad():
            lv = loss.detach()
            better = lv < best_loss
            best_loss = torch.where(better, lv, best_loss)
            best_delta[better] = delta.detach()[better]
            history.append(lv.cpu())
        if last:
            break
        grad, = torch.autograd.grad(sq.sum(), delta)
        alpha = alpha0 * (0.5 if it >= n_iters // 2 else 1.0) * (0.5 if it >= 3 * n_iters // 4 else 1.0)
        with torch.no_grad():
            delta = delta.detach() - alpha * grad.sign()
            delta = delta.clamp(-eps, eps)
            delta = (x + delta).clamp(0, 1) - x
    return best_delta, torch.stack(history)


def test_default_args_match_legacy_exactly(setup):
    enc, x, init = setup
    z_tgt = enc(torch.full_like(x, 0.5))
    d_new, h_new = pgd_encoder_attack(enc, x, z_tgt, 16 / 255, 20, 0.125, init)
    d_old, h_old = legacy_pgd(enc, x, z_tgt, 16 / 255, 20, 0.125, init)
    assert torch.equal(d_new, d_old)
    assert torch.equal(h_new, h_old)


@pytest.mark.parametrize("kwargs", [{}, {"momentum": 0.9}, {"untargeted": True},
                                    {"mask": "face", "mask_jitter_px": 2}])
def test_budget_and_box(setup, kwargs):
    enc, x, init = setup
    eps = 8 / 255
    if kwargs.get("mask") == "face":
        m = torch.zeros(2, 1, 32, 32)
        m[:, :, 8:24, 8:24] = 1
        kwargs = dict(kwargs, mask=m, rng=random.Random(0))
    z_tgt = enc(x) if kwargs.get("untargeted") else enc(torch.full_like(x, 0.5))
    delta, hist = pgd_encoder_attack(enc, x, z_tgt, eps, 15, 0.125, init, **kwargs)
    assert delta.abs().max() <= eps + 1e-6
    assert (x + delta).min() >= 0 and (x + delta).max() <= 1
    assert hist.shape == (16, 2)


def test_targeted_reduces_loss(setup):
    enc, x, _ = setup
    z_tgt = enc(torch.full_like(x, 0.5))
    delta, _ = pgd_encoder_attack(enc, x, z_tgt, 16 / 255, 30, 0.125)
    assert (latent_mse(enc(x + delta), z_tgt) < latent_mse(enc(x), z_tgt)).all()


def test_untargeted_increases_distance(setup):
    enc, x, _ = setup
    z_clean = enc(x)
    delta, _ = pgd_encoder_attack(enc, x, z_clean, 16 / 255, 30, 0.125, untargeted=True)
    assert (latent_mse(enc(x + delta), z_clean) > 0).all()


def test_mask_aware_leaves_background_untouched_without_init(setup):
    enc, x, _ = setup
    m = torch.zeros(2, 1, 32, 32)
    m[:, :, 8:24, 8:24] = 1
    z_tgt = enc(torch.full_like(x, 0.5))
    delta, _ = pgd_encoder_attack(enc, x, z_tgt, 16 / 255, 10, 0.125, mask=m)
    assert delta[:, :, :8, :].abs().max() == 0          # zero gradient outside the face
    assert delta[:, :, 8:24, 8:24].abs().max() > 0


def test_objective_replaces_latent_loss_and_is_minimised(setup):
    """E5 path: a generic per-image objective (here a tiny fake 'recognizer' cosine) goes down under PGD."""
    enc, x, init = setup
    torch.manual_seed(1)
    rec = torch.nn.Sequential(torch.nn.Conv2d(3, 6, 4, stride=4), torch.nn.Flatten())
    rec.requires_grad_(False)
    ref = torch.nn.functional.normalize(rec(x), dim=1)

    def obj(x_in):
        return (torch.nn.functional.normalize(rec(x_in), dim=1) * ref).sum(1)

    delta, hist = pgd_encoder_attack(None, x, None, 16 / 255, 30, 0.125, init, objective=obj)
    assert delta.abs().max() <= 16 / 255 + 1e-6
    assert hist.shape == (31, 2)
    assert (obj(x + delta) < obj(x) - 1e-3).all()
    assert torch.allclose(hist.min(0).values, obj(x + delta), atol=1e-5)   # best iterate is returned
    with pytest.raises(ValueError):
        pgd_encoder_attack(None, x, None, 16 / 255, 2, 0.125, objective=obj, untargeted=True)


def test_identity_crop_and_boxes():
    from facecloak.identity import crop, face_boxes
    m = torch.zeros(2, 1, 64, 64)
    m[0, :, 10:30, 20:40] = 1
    boxes = face_boxes(m, margin=0.0)
    assert torch.allclose(boxes[0], torch.tensor([20.0, 10.0, 40.0, 30.0]))
    assert torch.allclose(boxes[1], torch.tensor([0.0, 0.0, 64.0, 64.0]))   # empty mask -> whole image
    x = torch.rand(2, 3, 64, 64, requires_grad=True)
    c = crop(x, boxes, 16)
    assert c.shape == (2, 3, 16, 16)
    c.sum().backward()
    assert x.grad[0, :, 10:30, 20:40].abs().sum() > 0 and x.grad[0, :, 40:, :].abs().sum() == 0
