#!/usr/bin/env python
"""Stages 2+3: inpainting attack and ArcFace evaluation of an experiment whose cloaks already exist.

python scripts/run_inpaint_eval.py --config configs/experiments/E0_baseline.yaml [k=v ...]
"""
from facecloak.cli import parse
from facecloak.evaluate import run_eval
from facecloak.inpaint import run_inpaint

if __name__ == "__main__":
    cfg = parse(__doc__)
    run_inpaint(cfg)
    out = run_eval(cfg)
    print(out["summary"].to_string())
    print(out["paired"].to_string())
