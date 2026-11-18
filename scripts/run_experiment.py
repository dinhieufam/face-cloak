#!/usr/bin/env python
"""All stages for one experiment: cloak -> inpaint -> evaluate.

python scripts/run_experiment.py --config configs/experiments/E1_pattern_target.yaml [k=v ...]
"""
from facecloak.cli import parse
from facecloak.cloak import run_cloak, summarize_cloak
from facecloak.evaluate import run_eval
from facecloak.inpaint import run_inpaint

if __name__ == "__main__":
    cfg = parse(__doc__)
    print(summarize_cloak(run_cloak(cfg)).to_string())
    run_inpaint(cfg)
    out = run_eval(cfg)
    print(out["summary"].to_string())
    print(out["paired"].to_string())
