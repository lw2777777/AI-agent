
"""
批量回测运行器。
用法:
    python3 eval/run_batch.py --tag baseline --symbols 300750,600519,000001
"""
from __future__ import annotations
import json
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# 让脚本找到项目根目录的 backtest_multi_agent.py
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from backtest_multi_agent import backtest_symbol, load_data, BacktestResult


@dataclass
class BatchConfig:
    tag: str
    symbols: List[str]
    start_date: str = "2023-01-01"
    end_date: Optional[str] = None
    initial_capital: float = 100_000.0
    warmup: int = 60
    max_bars: int = 500
    agent_config: Dict[str, Any] = field(default_factory=dict)
    notes: str = ""


@dataclass
class BatchReport:
    config: BatchConfig
    timestamp: str
    per_symbol: Dict[str, Dict[str, Any]]
    aggregate: Dict[str, Any]


def run_batch(config: BatchConfig, output_dir: str = "./backtest_runs",
              verbose: bool = True) -> BatchReport:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    per_symbol: Dict[str, Dict[str, Any]] = {}

    for sym in config.symbols:
        if verbose:
            print(f"\n{'='*60}\n  Backtesting {sym} (tag={config.tag})\n{'='*60}")
        try:
            data = load_data(sym, start_date=config.start_date, end_date=config.end_date)
            if data is None or data.empty:
                per_symbol[sym] = {"error": "no_data"}
                continue

            t0 = time.time()
            result, trades = backtest_symbol(
                sym, data,
                initial_capital=config.initial_capital,
                warmup=config.warmup,
                max_bars=config.max_bars,
                config=config.agent_config,
            )
            elapsed = time.time() - t0

            per_symbol[sym] = {
                **_result_to_dict(result),
                "elapsed_sec": round(elapsed, 2),
                "n_bars": min(len(data), config.warmup + config.max_bars),
            }
        except Exception as e:
            traceback.print_exc()
            per_symbol[sym] = {"error": f"{type(e).__name__}: {e}"}

    aggregate = _aggregate(per_symbol)
    report = BatchReport(
        config=config,
        timestamp=datetime.now().isoformat(),
        per_symbol=per_symbol,
        aggregate=aggregate,
    )

    fname = f"{config.tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    fpath = out / fname
    with open(fpath, "w", encoding="utf-8") as f:
        json.dump(asdict(report), f, ensure_ascii=False, indent=2, default=str)

    if verbose:
        _print_summary(aggregate)
        print(f"\n💾 Saved: {fpath}")

    return report


def _result_to_dict(res: BacktestResult) -> Dict[str, Any]:
    reason_counts: Dict[str, int] = {}
    for t in res.trades:
        r = t.get("reason", "unknown")
        reason_counts[r] = reason_counts.get(r, 0) + 1
    pf = res.profit_factor
    if pf == float("inf") or (isinstance(pf, float) and pf != pf):
        pf = None

    return {
        "total_return": round(res.total_return, 6),
        "sharpe": round(res.sharpe, 4),
        "max_drawdown": round(res.max_drawdown, 6),
        "total_trades": res.total_trades,
        "win_rate": round(res.win_rate, 4),
        "profit_factor": round(pf, 4) if pf is not None else None,
        "avg_win": round(res.avg_win, 4),
        "avg_loss": round(res.avg_loss, 4),
        "equity_curve": res.equity_curve,
        "trades_by_reason": reason_counts,
    }


def _aggregate(per_symbol: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    valid = {k: v for k, v in per_symbol.items() if "error" not in v}
    if not valid:
        return {"n_symbols": 0}

    import statistics
    returns = [v["total_return"] for v in valid.values()]
    sharpes = [v["sharpe"] for v in valid.values()]
    dds = [v["max_drawdown"] for v in valid.values()]
    win_rates = [v["win_rate"] for v in valid.values()]
    trades = [v["total_trades"] for v in valid.values()]

    return {
        "n_symbols": len(valid),
        "avg_return": round(sum(returns) / len(returns), 6),
        "avg_sharpe": round(sum(sharpes) / len(sharpes), 4),
        "avg_drawdown": round(sum(dds) / len(dds), 6),
        "avg_win_rate": round(sum(win_rates) / len(win_rates), 4),
        "avg_trades": round(sum(trades) / len(trades), 1),
        "best_return": round(max(returns), 6),
        "worst_return": round(min(returns), 6),
        "worst_drawdown": round(max(dds), 6),
        "std_return": round(statistics.pstdev(returns), 6) if len(returns) > 1 else 0.0,
        "total_trades": sum(trades),
    }


def _print_summary(agg: Dict[str, Any]) -> None:
    print(f"\n{'─'*60}")
    print("  Aggregate")
    print(f"{'─'*60}")
    print(f"  Symbols:       {agg.get('n_symbols', 0)}")
    print(f"  Avg return:    {agg.get('avg_return', 0)*100:+.2f}%")
    print(f"  Avg sharpe:    {agg.get('avg_sharpe', 0):.2f}")
    print(f"  Worst dd:      {agg.get('worst_drawdown', 0)*100:.2f}%")
    print(f"  Avg win rate:  {agg.get('avg_win_rate', 0)*100:.1f}%")
    print(f"  Total trades:  {agg.get('total_trades', 0)}")
    print(f"  Return std:    {agg.get('std_return', 0)*100:.2f}% (越低越稳)")


# ── 默认 agent 参数 ──
DEFAULT_AGENT_CONFIG: Dict[str, Any] = {
    "voting_threshold": 0.40,
    "learning_rate": 0.05,
    "min_samples_to_learn": 20,
    "alpha_config": {"min_confidence": 0.45},
    "execution_config": {"min_rr": 1.5, "total_capital": 100_000},
    "risk_config": {
        "max_daily_loss_pct": 0.05,
        "trailing_stop_dd": 0.20,
        "max_drawdown_threshold": 0.15,
    },
}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--symbols", default="300750")
    parser.add_argument("--start", default="2023-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--bars", type=int, default=500)
    parser.add_argument("--notes", default="")
    parser.add_argument("--output", default="./backtest_runs")
    parser.add_argument("--config", default=None,
                        help="agent config JSON 路径,不填则用内置")
    args = parser.parse_args()

    if args.config:
        agent_config = json.loads(Path(args.config).read_text())
    else:
        agent_config = DEFAULT_AGENT_CONFIG

    cfg = BatchConfig(
        tag=args.tag,
        symbols=[s.strip() for s in args.symbols.split(",")],
        start_date=args.start,
        end_date=args.end,
        max_bars=args.bars,
        agent_config=agent_config,
        notes=args.notes,
    )
    run_batch(cfg, output_dir=args.output)


if __name__ == "__main__":
    main()
