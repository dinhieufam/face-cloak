#!/usr/bin/env python
"""Print / save one comparison table across experiments (reads <output_root>/<exp>/inpaint_out/summary.csv).

python scripts/compare_experiments.py --config configs/base.yaml [--exps E0_baseline E1_pattern_target]
"""
import argparse

from facecloak.config import get_paths, load_config
from facecloak.evaluate import compare_experiments

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/base.yaml", help="any config; only used to locate output_root")
    ap.add_argument("--exps", nargs="*", default=None, help="experiment names (default: all with a summary.csv)")
    args = ap.parse_args()
    root = get_paths(load_config(args.config)).root
    table = compare_experiments(root, args.exps)
    print(table.to_string(index=False) if len(table) else f"No finished experiments under {root}")
    if len(table):
        table.to_csv(root / "comparison.csv", index=False)
        print(f"\nSaved {root / 'comparison.csv'}")
