"""
批量回测运行器。

把单次 backtest_symbol() 扩展为:
  - 多标的 × 多时间段
  - 每次结果存成 JSON
  - 支持 --tag 标记版本
"""
from __future__ import annotations
import json
import time
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from backtest_multi_agent import backtest_symbol, load_data


@dataclass
class RunConfig:
    symbols: List[str]
    start_date: str = "2023-01-01"
    end_date: Optional[str] = None
    initial_capital: float = 100_000.0
    warmup: int = 60
    max_bars: int = 500
    agent_config: Dict[str, Any] = field(default_factory=dict)
    tag: str = "default"          # ★ 版本标记,如 "v1.2-baseline"


@dataclass
class RunReport:
    config: RunConfig
    timestamp: str
    per_symbol: Dict[str, Dict[str, Any]]   # symbol → 指标
    aggregate: Dict[str, Any]               # 汇总


def run_batch(config: RunConfig, output_dir: str = "./backtest_runs") -> RunReport:
    """跑一批回测,存报告"""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    per_symbol = {}
    for sym in config.symbols:
        data = load_data(sym, start_date=config.start_date, end_date=config.end_date)
        if data is None or data.empty:
            per_symbol[sym] = {"error": "no_data"}
            continue

        res, trades = backtest_symbol(
            sym, data,
            initial_capital=config.initial_capital,
            warmup=config.warmup,
            max_bars=config.max_bars,
            config=config.agent_config,
        )
        per_symbol[sym] = _result_to_dict(res)

    # 聚合
    valid = [v for v in per_symbol.values() if "error" not in v]
    aggregate = {
        "n_symbols": len(valid),
        "avg_return": sum(v["total_return"] for v in valid) / len(valid) if valid else 0,
        "avg_sharpe": sum(v["sharpe"] for v in valid) / len(valid) if valid else 0,
        "worst_drawdown": min((v["max_drawdown"] for v in valid), default=0),
        "total_trades": sum(v["total_trades"] for v in valid),
        "avg_win_rate": sum(v["win_rate"] for v in valid) / len(valid) if valid else 0,
    }

    report = RunReport(
        config=config,
        timestamp=datetime.now().isoformat(),
        per_symbol=per_symbol,
        aggregate=aggregate,
    )

    # 保存:文件名带 tag + 时间
    fname = f"{config.tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out / fname, "w", encoding="utf-8") as f:
        json.dump(asdict(report), f, ensure_ascii=False, indent=2, default=str)

    return report


def _result_to_dict(res) -> Dict[str, Any]:
    """BacktestResult → dict"""
    return {
        "total_return": res.total_return,
        "sharpe": res.sharpe,
        "max_drawdown": res.max_drawdown,
        "total_trades": res.total_trades,
        "win_rate": res.win_rate,
        "profit_factor": res.profit_factor,
        "avg_win": res.avg_win,
        "avg_loss": res.avg_loss,
        "equity_curve": res.equity_curve,
        # 从 trades 里提取更多统计
        "trades_by_reason": _count_by_key(res.trades, "reason"),
    }


def _count_by_key(items: List[Dict], key: str) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for it in items:
        k = it.get(key, "unknown")
        out[k] = out.get(k, 0) + 1
    return out