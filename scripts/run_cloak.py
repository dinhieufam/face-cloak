#!/usr/bin/env python
"""Stage 1: compute cloaks.  python scripts/run_cloak.py --config configs/experiments/E0_baseline.yaml [k=v ...]"""
from facecloak.cli import parse
from facecloak.cloak import run_cloak, summarize_cloak

if __name__ == "__main__":
    cfg = parse(__doc__)
    print(summarize_cloak(run_cloak(cfg)).to_string())
