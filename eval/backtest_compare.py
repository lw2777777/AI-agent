
"""
对比两次量化回测报告。
用法:
    python3 eval/backtest_compare.py baseline.json v2.json
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List


def load_report(path: str) -> Dict[str, Any]:
    return json.loads(Path(path).read_text())


def compare(baseline: Dict, candidate: Dict) -> Dict[str, Any]:
    b_agg = baseline["aggregate"]
    c_agg = candidate["aggregate"]

    summary_delta = {
        "avg_return":     c_agg["avg_return"]     - b_agg["avg_return"],
        "avg_sharpe":     c_agg["avg_sharpe"]     - b_agg["avg_sharpe"],
        "avg_drawdown":   c_agg["avg_drawdown"]   - b_agg["avg_drawdown"],
        "avg_win_rate":   c_agg["avg_win_rate"]   - b_agg["avg_win_rate"],
        "avg_trades":     c_agg["avg_trades"]     - b_agg["avg_trades"],
        "worst_drawdown": c_agg["worst_drawdown"] - b_agg["worst_drawdown"],
        "std_return":     c_agg["std_return"]     - b_agg["std_return"],
    }

    b_syms = baseline["per_symbol"]
    c_syms = candidate["per_symbol"]
    per_symbol_delta = {}
    regressions = []
    improvements = []

    for sym in set(b_syms) & set(c_syms):
        b = b_syms[sym]
        c = c_syms[sym]
        if "error" in b or "error" in c:
            continue
        d = {
            "return":   c["total_return"]  - b["total_return"],
            "sharpe":   c["sharpe"]        - b["sharpe"],
            "drawdown": c["max_drawdown"]  - b["max_drawdown"],
            "trades":   c["total_trades"]  - b["total_trades"],
            "win_rate": c["win_rate"]      - b["win_rate"],
        }
        per_symbol_delta[sym] = d

        if d["return"] < -0.02 or d["drawdown"] > 0.03:
            regressions.append({
                "symbol": sym, "return_delta": d["return"],
                "drawdown_delta": d["drawdown"], "sharpe_delta": d["sharpe"],
            })
        elif d["return"] > 0.02 and d["drawdown"] < 0.01:
            improvements.append({
                "symbol": sym, "return_delta": d["return"],
                "sharpe_delta": d["sharpe"],
            })

    return {
        "baseline_tag": baseline["config"]["tag"],
        "candidate_tag": candidate["config"]["tag"],
        "baseline_notes": baseline["config"].get("notes", ""),
        "candidate_notes": candidate["config"].get("notes", ""),
        "summary_delta": summary_delta,
        "per_symbol_delta": per_symbol_delta,
        "regressions": regressions,
        "improvements": improvements,
    }


def print_comparison(diff: Dict) -> None:
    print(f"\n{'='*60}")
    print(f"  Compare: {diff['baseline_tag']} → {diff['candidate_tag']}")
    print(f"{'='*60}")
    if diff["baseline_notes"]:
        print(f"  baseline:  {diff['baseline_notes']}")
    if diff["candidate_notes"]:
        print(f"  candidate: {diff['candidate_notes']}")

    d = diff["summary_delta"]
    def _arrow(v, good_positive=True):
        if abs(v) < 1e-6:
            return "  "
        return "✅" if (v > 0) == good_positive else "⚠️"

    print(f"\n  avg_return     Δ: {d['avg_return']*100:+.2f}%   {_arrow(d['avg_return'])}")
    print(f"  avg_sharpe     Δ: {d['avg_sharpe']:+.3f}      {_arrow(d['avg_sharpe'])}")
    print(f"  avg_win_rate   Δ: {d['avg_win_rate']*100:+.2f}%   {_arrow(d['avg_win_rate'])}")
    print(f"  avg_trades     Δ: {d['avg_trades']:+.1f}      {_arrow(d['avg_trades'], good_positive=False)}")
    print(f"  worst_drawdown Δ: {d['worst_drawdown']*100:+.2f}%   {_arrow(d['worst_drawdown'], good_positive=False)}")
    print(f"  std_return     Δ: {d['std_return']*100:+.2f}%   {_arrow(d['std_return'], good_positive=False)}")

    if diff["regressions"]:
        print(f"\n  🔴 Regressions ({len(diff['regressions'])}):")
        for r in diff["regressions"]:
            print(f"    - {r['symbol']}: return={r['return_delta']*100:+.2f}%, "
                  f"dd={r['drawdown_delta']*100:+.2f}%, sharpe={r['sharpe_delta']:+.3f}")
    if diff["improvements"]:
        print(f"\n  🟢 Improvements ({len(diff['improvements'])}):")
        for i in diff["improvements"]:
            print(f"    - {i['symbol']}: return={i['return_delta']*100:+.2f}%, "
                  f"sharpe={i['sharpe_delta']:+.3f}")
    print(f"{'='*60}\n")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline")
    parser.add_argument("candidate")
    args = parser.parse_args()

    b = load_report(args.baseline)
    c = load_report(args.candidate)
    print_comparison(compare(b, c))


if __name__ == "__main__":
    main()
