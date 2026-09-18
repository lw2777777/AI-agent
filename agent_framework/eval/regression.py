"""
净值曲线差分。

即使总收益不变,也可能是"某段行情表现完全换了"。
这个工具回答:改了什么之后,哪一段行情的结果不一样了?
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List


def diff_equity(
    baseline_curve: List[float],
    candidate_curve: List[float],
    window: int = 20,
) -> Dict[str, Any]:
    """按窗口对比两条净值曲线"""
    n = min(len(baseline_curve), len(candidate_curve))
    if n < window + 1:
        return {"error": "too_short"}

    b = [x / baseline_curve[0] for x in baseline_curve[:n]]
    c = [x / candidate_curve[0] for x in candidate_curve[:n]]

    windows = []
    for i in range(window, n):
        b_ret = b[i] / b[i - window] - 1
        c_ret = c[i] / c[i - window] - 1
        diff = c_ret - b_ret
        windows.append({
            "end_idx": i,
            "baseline_ret": b_ret,
            "candidate_ret": c_ret,
            "diff": diff,
        })

    max_div = max(windows, key=lambda w: abs(w["diff"]))
    big_diffs = [w for w in windows if abs(w["diff"]) > 0.03]

    return {
        "max_divergence": max_div,
        "n_big_diffs": len(big_diffs),
        "top_divergences": sorted(big_diffs, key=lambda w: -abs(w["diff"]))[:5],
    }


def diff_report(baseline_path: str, candidate_path: str) -> Dict[str, Any]:
    b = json.loads(Path(baseline_path).read_text())
    c = json.loads(Path(candidate_path).read_text())

    b_syms = b["per_symbol"]
    c_syms = c["per_symbol"]
    out = {}

    for sym in set(b_syms) & set(c_syms):
        bc = b_syms[sym].get("equity_curve")
        cc = c_syms[sym].get("equity_curve")
        if not bc or not cc:
            continue
        out[sym] = diff_equity(bc, cc)

    return out


def print_regression(diffs: Dict[str, Any]) -> None:
    print(f"\n{'='*60}")
    print(f"  Equity Curve Divergence")
    print(f"{'='*60}")
    for sym, d in diffs.items():
        if "error" in d:
            print(f"  {sym}: {d['error']}")
            continue
        max_d = d["max_divergence"]
        print(f"\n  {sym}:")
        print(f"    Max divergence @ bar {max_d['end_idx']}: "
              f"{max_d['diff']*100:+.2f}%")
        print(f"    Baseline 20-bar ret:  {max_d['baseline_ret']*100:+.2f}%")
        print(f"    Candidate 20-bar ret: {max_d['candidate_ret']*100:+.2f}%")
        if d["n_big_diffs"] > 0:
            print(f"    ⚠️  {d['n_big_diffs']} windows with |diff| > 3%")
            print(f"    Top divergences:")
            for w in d["top_divergences"]:
                print(f"      @ bar {w['end_idx']:>4}: {w['diff']*100:+.2f}%")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline")
    parser.add_argument("candidate")
    args = parser.parse_args()

    diffs = diff_report(args.baseline, args.candidate)
    print_regression(diffs)