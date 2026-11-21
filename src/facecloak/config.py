"""YAML experiment configs with inheritance, CLI overrides and a fixed on-disk layout.

An experiment file names its parent (`inherits: ../base.yaml`) and overrides only the keys it changes.
Overrides from the command line use dotted keys: `cloak.n_iters=50 quick_test=true`.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import yaml

def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _load_raw(path: Path) -> dict:
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    parent = raw.pop("inherits", None)
    if parent:
        raw = deep_merge(_load_raw((path.parent / parent).resolve()), raw)
    return raw


def apply_overrides(cfg: dict, overrides) -> dict:
    """`["cloak.eps=[8,16]", "quick_test=true"]` -> nested assignment; values are parsed as YAML."""
    cfg = copy.deepcopy(cfg)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must look like key.sub=value, got {item!r}")
        key, value = item.split("=", 1)
        node = cfg
        parts = key.strip().split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(value)
    return cfg


def find_repo_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / "pyproject.toml").exists():
            return p
    return Path.cwd()


def apply_quick_test(cfg: dict) -> dict:
    """Smoke-test settings: 2 images, largest ε only, 50 PGD iterations, 1 prompt, 1 seed.

    The experiment name gets a `_quick` suffix so smoke tests never mix with real runs (both stages resume
    from what is on disk).
    """
    cfg = copy.deepcopy(cfg)
    cfg["exp_name"] = f"{cfg['exp_name']}_quick"
    cfg["data"]["max_images"] = 2
    cfg["cloak"]["eps"] = [max(cfg["cloak"]["eps"])]
    cfg["cloak"]["n_iters"] = 50
    cfg["inpaint"]["prompt_ids"] = cfg["inpaint"]["prompt_ids"][:1]
    cfg["inpaint"]["seeds"] = cfg["inpaint"]["seeds"][:1]
    return cfg


def load_config(path, overrides=()) -> dict:
    path = Path(path).resolve()
    cfg = apply_overrides(_load_raw(path), overrides)
    if cfg.get("quick_test"):
        cfg = apply_quick_test(cfg)
    cfg["_config_path"] = str(path)
    cfg["_repo_root"] = str(find_repo_root(path.parent))
    return cfg


def resolve_path(cfg: dict, p: str | None) -> str | None:
    """Absolute paths pass through; relative ones are taken relative to the repo root."""
    if p is None:
        return None
    p = os.path.expanduser(str(p))
    return p if os.path.isabs(p) else str(Path(cfg["_repo_root"]) / p)


@dataclass(frozen=True)
class Paths:
    """On-disk layout of one experiment.

    <root>/_shared/images/<img_id>/clean.png       preprocessed 512x512 source, shared by every experiment
    <root>/_shared/masks/dilate<d>/<img_id>_*.png    MediaPipe face masks (face = keep, inpaint = regenerate)
    <root>/_shared/inpaint_<hash>/...                clean/random-branch inpaintings, reused across experiments
    <root>/<exp>/cloak_out/                          manifest.csv, config.json, images/<id>/cloak_eps*.png, figures/
    <root>/<exp>/inpaint_out/                        generations.csv, results.csv, summary.csv, outputs/, figures/
    <root>/tau.json                                  fixed identity threshold from scripts/calibrate_tau.py
    """
    root: Path
    exp: str
    dilate_px: int
    inpaint_key: str

    @property
    def shared(self) -> Path:
        return self.root / "_shared"

    @property
    def shared_images(self) -> Path:
        return self.shared / "images"

    @property
    def masks(self) -> Path:
        return self.shared / "masks" / f"dilate{self.dilate_px}"

    @property
    def shared_inpaint(self) -> Path:
        return self.shared / f"inpaint_{self.inpaint_key}"

    @property
    def exp_dir(self) -> Path:
        return self.root / self.exp

    @property
    def cloak_dir(self) -> Path:
        return self.exp_dir / "cloak_out"

    @property
    def inpaint_dir(self) -> Path:
        return self.exp_dir / "inpaint_out"

    @property
    def tau_file(self) -> Path:
        return self.root / "tau.json"

    def rel(self, p) -> str:
        """Paths stored in CSVs are relative to the output root, so a run folder can be moved or zipped."""
        return os.path.relpath(p, self.root)

    def abs(self, rel_path: str) -> Path:
        return self.root / rel_path


def inpaint_settings_key(cfg: dict) -> str:
    """Short hash of everything that changes an inpainting output for a fixed input, mask, prompt and seed."""
    ic = cfg["inpaint"]
    keyed = dict(model=cfg["model"]["id"], neg=ic["negative_prompt"], steps=ic["num_steps"],
                 guidance=ic["guidance"], safety=ic["keep_safety_checker"], prompts=ic["prompts"],
                 dilate=cfg["mask"]["dilate_px"])
    if ic.get("jpeg_quality") is not None:  # only when set, so existing caches keep their hash
        keyed["jpeg"] = int(ic["jpeg_quality"])
    return hashlib.sha1(json.dumps(keyed, sort_keys=True).encode()).hexdigest()[:8]


def get_paths(cfg: dict) -> Paths:
    root = os.environ.get("FACECLOAK_OUTPUT_ROOT") or cfg["paths"].get("output_root") or "runs"  # -> <repo>/runs
    return Paths(root=Path(resolve_path(cfg, root)), exp=cfg["exp_name"], dilate_px=int(cfg["mask"]["dilate_px"]),
                 inpaint_key=inpaint_settings_key(cfg))


def public(cfg: dict) -> dict:
    """Config without private bookkeeping keys, for config.json dumps."""
    return {k: v for k, v in cfg.items() if not k.startswith("_")}


def dump(cfg: dict, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(dict(public(cfg), config_file=cfg.get("_config_path")), f, indent=2)
