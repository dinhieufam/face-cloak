"""E5: identity loss through the SD VAE with a differentiable face recognizer.

    min_δ  cos( F(crop(D(E(x_in)))), F(crop(x)) )        x_in = x+δ, or attacker_view(x+δ, m) with mask_aware

D(E(·)) is the inpainting VAE round trip: the face in an SD inpainting output is VAE-decoded, not pasted back
(see docs/pipeline.md), so this is the face the attacker's output starts from. F is a PyTorch recognizer that is
deliberately *different* from the evaluator (UniFace ArcFace, ONNX): FaceNet InceptionResnetV1 trained on
VGGFace2. A low post-deepfake ArcFace similarity is then evidence of transfer, not of overfitting to the metric.

The face crop is a fixed box per image, taken from the MediaPipe FACE_OVAL mask of clean.png (+ margin), so the
crop never moves with δ. The reference embedding F(crop(x)) is of the clean photo, as in the evaluation.
"""
from __future__ import annotations

from typing import Callable

import torch
import torch.nn.functional as F

from . import env

RECOGNIZERS = ("facenet_vggface2", "facenet_casia")


def load_recognizer(name: str, device: str) -> tuple[torch.nn.Module, int]:
    """-> (frozen recognizer mapping standardized (B,3,S,S) crops to embeddings, input size S)."""
    if name not in RECOGNIZERS:
        raise ValueError(f"cloak.id_model must be one of {RECOGNIZERS}, got {name!r}")
    from facenet_pytorch import InceptionResnetV1  # pip install --no-deps facenet-pytorch (see docs/setup.md)
    net = InceptionResnetV1(pretrained=name.split("_", 1)[1]).to(device).eval()
    net.requires_grad_(False)
    return net, 160


def facenet_standardize(x01: torch.Tensor) -> torch.Tensor:
    """facenet-pytorch's fixed_image_standardization: (x·255 − 127.5) / 128."""
    return (x01 * 255 - 127.5) / 128


def face_boxes(face_mask: torch.Tensor, margin: float = 0.15) -> torch.Tensor:
    """(B,1,H,W) binary face mask -> (B,4) square boxes [x1,y1,x2,y2] around the mask, grown by `margin`."""
    B, _, H, W = face_mask.shape
    boxes = []
    for b in range(B):
        ys, xs = torch.nonzero(face_mask[b, 0] > 0.5, as_tuple=True)
        if len(ys) == 0:
            boxes.append(torch.tensor([0.0, 0.0, W, H]))
            continue
        x1, x2, y1, y2 = xs.min().item(), xs.max().item() + 1, ys.min().item(), ys.max().item() + 1
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        half = max(x2 - x1, y2 - y1) * (1 + margin) / 2
        boxes.append(torch.tensor([cx - half, cy - half, cx + half, cy + half]))
    return torch.stack(boxes).float().to(face_mask.device)


def crop(x: torch.Tensor, boxes: torch.Tensor, size: int) -> torch.Tensor:
    """Differentiable bilinear crop of box b from image b, resized to size×size (pads with zeros off-image)."""
    from torchvision.ops import roi_align
    rois = torch.cat([torch.arange(len(boxes), device=x.device, dtype=x.dtype)[:, None], boxes.to(x.dtype)], 1)
    return roi_align(x, rois, output_size=size, spatial_scale=1.0, sampling_ratio=2, aligned=True)


def make_decoder(vae, device: str, fp16: bool):
    """latent mean -> image in [0,1] (differentiable), fp32 out."""
    def decode(z: torch.Tensor) -> torch.Tensor:
        with env.autocast(device, fp16):
            y = vae.decode(z).sample
        return ((y.float() + 1) / 2).clamp(0, 1)
    return decode


def make_identity_objective(encode: Callable, decode: Callable, recognizer: torch.nn.Module, size: int,
                            x_clean: torch.Tensor, boxes: torch.Tensor, device: str, fp16: bool):
    """-> obj(x_in) = per-image cosine similarity to the clean face (minimise it)."""
    def embed(img01: torch.Tensor) -> torch.Tensor:
        with env.autocast(device, fp16):
            e = recognizer(facenet_standardize(crop(img01, boxes, size)))
        return F.normalize(e.float(), dim=1)

    with torch.no_grad():
        ref = embed(x_clean)

    def objective(x_in: torch.Tensor) -> torch.Tensor:
        return (embed(decode(encode(x_in))) * ref).sum(1)
    return objective
