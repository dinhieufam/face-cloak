"""Shared command-line handling for scripts/*.py."""
from __future__ import annotations

import argparse
import json

from .config import get_paths, load_config, public


def parse(description: str, argv=None) -> dict:
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--config", required=True, help="experiment YAML, e.g. configs/experiments/E0_baseline.yaml")
    ap.add_argument("--print-config", action="store_true", help="print the resolved config and paths, then exit")
    ap.add_argument("overrides", nargs="*", help="dotted overrides, e.g. cloak.n_iters=50 quick_test=true")
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.overrides)
    if args.print_config:
        p = get_paths(cfg)
        print(json.dumps(public(cfg), indent=2))
        print(f"\noutput_root : {p.root}\ncloak_dir   : {p.cloak_dir}\ninpaint_dir : {p.inpaint_dir}\n"
              f"shared      : {p.shared}\nmasks       : {p.masks}\ntau_file    : {p.tau_file}")
        raise SystemExit(0)
    return cfg
