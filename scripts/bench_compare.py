#!/usr/bin/env python3
"""
bench_compare.py  —  Compare two Google Benchmark JSON result files.

Usage:
    python3 scripts/bench_compare.py baseline.json current.json [--threshold 0.10]

Exit code:
    0  All benchmarks within threshold (no regression)
    1  One or more benchmarks regressed beyond threshold
"""

import argparse
import json
import sys


def load_benchmarks(path: str) -> dict:
    """Load benchmark JSON and return dict: name -> mean real_time (ns)."""
    with open(path) as f:
        data = json.load(f)
    result = {}
    for bm in data.get("benchmarks", []):
        # Only consider aggregate "mean" entries when repetitions > 1
        agg = bm.get("aggregate_name", "")
        if agg and agg != "mean":
            continue
        name = bm["name"].removesuffix("_mean")
        result[name] = bm["real_time"]
    return result


def compare(baseline: dict, current: dict, threshold: float) -> bool:
    """Print comparison table; return True if all within threshold."""
    all_ok = True
    keys = sorted(set(baseline) & set(current))

    col_w = max((len(k) for k in keys), default=30) + 2
    print(f"\n{'Benchmark':<{col_w}}  {'Baseline':>12}  {'Current':>12}  {'Delta':>10}  {'Status':>8}")
    print("-" * (col_w + 50))

    for name in keys:
        base = baseline[name]
        curr = current[name]
        delta = (curr - base) / base if base != 0 else 0.0
        status = "✓ OK"
        if delta > threshold:
            status = "✗ REGRESSED"
            all_ok = False
        elif delta < -threshold:
            status = "↑ IMPROVED"
        print(f"{name:<{col_w}}  {base:>11.2f}  {curr:>11.2f}  {delta:>+9.1%}  {status:>8}")

    # Keys only in current (new benchmarks)
    new_keys = sorted(set(current) - set(baseline))
    if new_keys:
        print(f"\nNew benchmarks (no baseline):")
        for name in new_keys:
            print(f"  {name}: {current[name]:.2f} ns")

    # Keys only in baseline (removed benchmarks)
    removed_keys = sorted(set(baseline) - set(current))
    if removed_keys:
        print(f"\nRemoved benchmarks:")
        for name in removed_keys:
            print(f"  {name}")

    return all_ok


def main():
    parser = argparse.ArgumentParser(description="Compare Google Benchmark JSON results")
    parser.add_argument("baseline", help="Baseline JSON file")
    parser.add_argument("current",  help="Current JSON file")
    parser.add_argument("--threshold", type=float, default=0.10,
                        help="Regression threshold (fraction, default 0.10 = 10%%)")
    args = parser.parse_args()

    baseline = load_benchmarks(args.baseline)
    current  = load_benchmarks(args.current)

    if not baseline:
        print("WARNING: baseline file has no benchmark results", file=sys.stderr)
        sys.exit(0)
    if not current:
        print("ERROR: current file has no benchmark results", file=sys.stderr)
        sys.exit(1)

    ok = compare(baseline, current, args.threshold)
    print(f"\n{'All benchmarks within {:.0%} threshold'.format(args.threshold) if ok else 'REGRESSION DETECTED (>{:.0%} slowdown)'.format(args.threshold)}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
