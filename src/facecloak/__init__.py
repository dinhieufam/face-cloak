"""facecloak: adversarial face cloaks against diffusion-inpainting deepfakes.

Stage 1 (`facecloak.cloak`)     PGD encoder attack on the Stable Diffusion VAE.
Stage 2 (`facecloak.inpaint`)   SD 1.5 inpainting attack that keeps the face and regenerates the rest.
Stage 3 (`facecloak.evaluate`)  RetinaFace + ArcFace identity similarity, threshold, protection rate.
"""

__version__ = "0.1.0"
