"""Face masks (MediaPipe FACE_OVAL) and the attacker's view of an image.

Diffusers convention: in the *inpaint* mask white = regenerate, black = keep. We also save the inverse
(`_face.png`, white = kept face) because that is what the cloak and the plots need.
"""
from __future__ import annotations

import random
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .config import Paths

# Face-oval landmark indices of the 468/478-point MediaPipe face mesh, in polygon order
# (forehead -> right cheek -> chin -> left cheek).
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377,
             152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]


# ----------------------------------------------------------------------------- torch-side helpers
def attacker_view(x01: torch.Tensor, face_mask: torch.Tensor) -> torch.Tensor:
    """What the SD inpainting pipeline feeds its VAE.

    diffusers computes `masked_image = init_image * (mask < 0.5)` with `init_image` in [-1, 1], so the
    regenerated area becomes 0 in [-1, 1], i.e. mid-gray 0.5 in [0, 1] (NOT black).
    `face_mask` is (B,1,H,W) with 1 = kept face.
    """
    return x01 * face_mask + 0.5 * (1 - face_mask)


def jitter_mask(face_mask: torch.Tensor, max_px: int, rng: random.Random | None = None) -> torch.Tensor:
    """Randomly dilate (+k) or erode (-k) a binary mask by up to `max_px` pixels (EOT over the attacker's mask)."""
    if max_px <= 0:
        return face_mask
    k = (rng or random).randint(-max_px, max_px)
    if k == 0:
        return face_mask
    r = abs(k)

    def pool(t):
        return F.max_pool2d(t, 2 * r + 1, stride=1, padding=r)

    return pool(face_mask) if k > 0 else 1 - pool(1 - face_mask)


def load_face_masks(paths, device) -> torch.Tensor:
    """List of `_face.png` paths -> (B,1,H,W) float tensor, 1 = kept face."""
    m = np.stack([np.asarray(Image.open(p).convert("L"), np.float32) / 255.0 for p in paths])
    return torch.from_numpy((m > 0.5).astype(np.float32))[:, None].to(device)


# ----------------------------------------------------------------------------- MediaPipe
_LANDMARKER = None


def get_landmarker(model_dir: Path, url: str, min_conf: float):
    global _LANDMARKER
    if _LANDMARKER is None:
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python import vision as mp_vision
        model_path = Path(model_dir) / "face_landmarker.task"
        if not model_path.exists():
            model_path.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(url, model_path)
        _LANDMARKER = mp_vision.FaceLandmarker.create_from_options(mp_vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)), num_faces=1,
            min_face_detection_confidence=min_conf, min_face_presence_confidence=min_conf))
    return _LANDMARKER


def face_oval_mask(landmarker, rgb: np.ndarray, dilate_px: int = 0):
    """rgb: HxWx3 uint8 -> uint8 mask (255 = face to keep), or None if no face is found."""
    import cv2
    import mediapipe as mp
    h, w = rgb.shape[:2]
    res = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)))
    if not res.face_landmarks:
        return None
    lm = res.face_landmarks[0]
    poly = np.array([[lm[i].x * w, lm[i].y * h] for i in FACE_OVAL], dtype=np.float32)
    poly = np.clip(poly.round(), 0, [w - 1, h - 1]).astype(np.int32)
    mask = np.zeros((h, w), np.uint8)
    cv2.fillPoly(mask, [poly], 255)
    if dilate_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1))
        mask = cv2.dilate(mask, k)
    return mask


def ensure_masks(cfg: dict, paths: Paths, items: list[dict]) -> tuple[dict, list[str]]:
    """Compute masks once per image from clean.png (so a cloak can never move the mask).

    Returns ({img_id: {"face": path, "inpaint": path}}, [img_ids without a detected face]).
    """
    mc = cfg["mask"]
    paths.masks.mkdir(parents=True, exist_ok=True)
    ok, failed = {}, []
    for it in items:
        face_p = paths.masks / f"{it['img_id']}_face.png"
        inp_p = paths.masks / f"{it['img_id']}_inpaint.png"
        if not inp_p.exists():
            lmk = get_landmarker(paths.shared, mc["landmarker_url"], mc["min_face_conf"])
            m = face_oval_mask(lmk, np.asarray(Image.open(it["clean_path"]).convert("RGB")), mc["dilate_px"])
            if m is None:
                failed.append(it["img_id"])
                continue
            Image.fromarray(m).save(face_p)
            Image.fromarray(255 - m).save(inp_p)
        ok[it["img_id"]] = {"face": str(face_p), "inpaint": str(inp_p)}
    if failed:
        print(f"No face found by MediaPipe in {len(failed)} image(s), skipped: {failed}")
    return ok, failed
