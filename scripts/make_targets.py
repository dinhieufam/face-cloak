#!/usr/bin/env python
"""Generate dense, high-contrast target images for E1 (target-image ablation).

Usage: python scripts/make_targets.py   -> writes assets/targets/*.png (512x512, deterministic)
"""
import os

import numpy as np
from PIL import Image

S = 512
OUT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "assets", "targets"))
os.makedirs(OUT, exist_ok=True)
yy, xx = np.mgrid[0:S, 0:S].astype(np.float32) / S


def save(name, arr):
    Image.fromarray((np.clip(arr, 0, 1) * 255).round().astype(np.uint8)).save(os.path.join(OUT, name))
    print("wrote", os.path.join(OUT, name))


# 1) pattern.png: multi-scale colour gratings at several orientations + fine checkerboard.
#    Lots of energy at the 8-px scale the VAE (downsample x8) is most sensitive to.
rng = np.random.default_rng(0)
img = np.zeros((S, S, 3), np.float32)
for c in range(3):
    acc = np.zeros((S, S), np.float32)
    for _ in range(3):
        f = rng.uniform(4, 24)                      # cycles per image
        th = rng.uniform(0, np.pi)
        ph = rng.uniform(0, 2 * np.pi)
        acc += np.sin(2 * np.pi * f * (xx * np.cos(th) + yy * np.sin(th)) + ph)
    img[..., c] = 0.5 + 0.5 * np.tanh(2.5 * acc)    # saturate -> high contrast, far from mid-gray
checker = ((np.floor(xx * S / 8) + np.floor(yy * S / 8)) % 2)[..., None]
pattern = 0.7 * img + 0.3 * checker
save("pattern.png", pattern)

# 2) checker8.png: pure black/white 8-px checkerboard (simplest high-frequency target, for a sanity ablation)
save("checker8.png", np.repeat(checker, 3, axis=2))
