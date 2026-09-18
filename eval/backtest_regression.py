
"""
净值曲线差分。
用法:
    python3 eval/backtest_regression.py baseline.json v2.json
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List


def diff_equity(baseline_curve, candidate_curve, window=20):
    n = min(len(baseline_curve), len(candidate_curve))
    if n < window + 1:
        return {"error": "too_short"}

    b = [x / baseline_curve[0] for x in baseline_curve[:n]]
    c = [x / candidate_curve[0] for x in candidate_curve[:n]]

    windows = []
    for i in range(window, n):
        b_ret = b[i] / b[i - window] - 1
        c_ret = c[i] / c[i - window] - 1
        windows.append({
            "end_idx": i,
            "baseline_ret": b_ret,
            "candidate_ret": c_ret,
            "diff": c_ret - b_ret,
        })

    max_div = max(windows, key=lambda w: abs(w["diff"]))
    big_diffs = [w for w in windows if abs(w["diff"]) > 0.03]

    return {
        "max_divergence": max_div,
        "n_big_diffs": len(big_diffs),
        "top_divergences": sorted(big_diffs, key=lambda w: -abs(w["diff"]))[:5],
    }


def diff_report(baseline_path, candidate_path):
    b = json.loads(Path(baseline_path).read_text())
    c = json.loads(Path(candidate_path).read_text())
    out = {}
    for sym in set(b["per_symbol"]) & set(c["per_symbol"]):
        bc = b["per_symbol"][sym].get("equity_curve")
        cc = c["per_symbol"][sym].get("equity_curve")
        if not bc or not cc:
            continue
        out[sym] = diff_equity(bc, cc)
    return out


def print_regression(diffs):
    print(f"\n{'='*60}")
    print(f"  Equity Curve Divergence")
    print(f"{'='*60}")
    for sym, d in diffs.items():
        if "error" in d:
            print(f"  {sym}: {d['error']}")
            continue
        max_d = d["max_divergence"]
        print(f"\n  {sym}:")
        print(f"    Max divergence @ bar {max_d['end_idx']}: {max_d['diff']*100:+.2f}%")
        if d["n_big_diffs"] > 0:
            print(f"    ⚠️  {d['n_big_diffs']} windows with |diff| > 3%")
            for w in d["top_divergences"]:
                print(f"      @ bar {w['end_idx']:>4}: {w['diff']*100:+.2f}%")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline")
    parser.add_argument("candidate")
    args = parser.parse_args()
    print_regression(diff_report(args.baseline, args.candidate))
