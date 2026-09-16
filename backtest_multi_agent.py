"""
backtest_multi_agent.py
四层 Agent 架构回测脚本
  Market → Alpha → (Risk) → Execution → Orchestrator
"""
from __future__ import annotations

import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ── 项目路径 ─────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from my_agents.multi_agent_orchestrator import MultiAgentOrchestrator, Decision
from my_agents.base_agent import Action

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("backtest")


# ══════════════════════════════════════════════════════════
# 数据加载
# ══════════════════════════════════════════════════════════


CACHE_DIR = Path(__file__).parent / "data_cache"
CACHE_DIR.mkdir(exist_ok=True)


def load_data(
    symbol: str,
    start_date: str = "2023-01-01",
    end_date: Optional[str] = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    加载 A 股日线数据。

    Args:
        symbol:     6 位代码，如 "300750"
        start_date: "YYYY-MM-DD"
        end_date:   默认今天
        use_cache:  是否使用本地缓存

    Returns:
        DataFrame(index=DatetimeIndex, columns=[open, high, low, close, volume])
    """
    end_date = end_date or pd.Timestamp.now().strftime("%Y-%m-%d")
    cache_file = CACHE_DIR / f"{symbol}_{start_date}_{end_date}.parquet"

    # ── 1. 缓存 ──────────────────────────
    if use_cache and cache_file.exists():
        try:
            df = pd.read_parquet(cache_file)
            if len(df) >= 60:
                logger.info("✅ 缓存命中: %s (%d 条)", cache_file.name, len(df))
                return df
        except Exception as e:
            logger.warning("缓存读取失败: %s", e)

    # ── 2. 依次尝试多个数据源 ────────────
    df = None

    # 2a. akshare 东财接口（最常用）
    df = _try_akshare_em(symbol, start_date, end_date)
    if df is not None and len(df) >= 60:
        _save_cache(df, cache_file)
        return df

    # 2b. akshare 新浪接口
    df = _try_akshare_sina(symbol, start_date, end_date)
    if df is not None and len(df) >= 60:
        _save_cache(df, cache_file)
        return df

    # 2c. akshare 腾讯接口
    df = _try_akshare_tencent(symbol, start_date, end_date)
    if df is not None and len(df) >= 60:
        _save_cache(df, cache_file)
        return df

    # 2d. baostock 兜底（最稳）
    df = _try_baostock(symbol, start_date, end_date)
    if df is not None and len(df) >= 60:
        _save_cache(df, cache_file)
        return df

    # ── 3. 全部失败 ──────────────────────
    logger.error("❌ 所有数据源都失败，无法加载 %s", symbol)
    return pd.DataFrame()


# ══════════════════════════════════════════════════════════
# 数据源实现
# ══════════════════════════════════════════════════════════
def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """统一列名、类型、索引"""
    df = df.copy()
    df.columns = [str(c).lower() for c in df.columns]

    rename_map = {
        "日期": "date", "开盘": "open", "收盘": "close",
        "最高": "high", "最低": "low", "成交量": "volume",
        "date": "date", "open": "open", "close": "close",
        "high": "high", "low": "low", "volume": "volume",
        "vol": "volume", "amount": "amount",
    }
    df = df.rename(columns=rename_map)

    need = ["open", "high", "low", "close", "volume"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"缺少列: {missing}")

    df = df[["date"] + need] if "date" in df.columns else df[need]
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")

    df = df[need].astype(float)
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df = df.dropna()
    return df


def _save_cache(df: pd.DataFrame, path: Path) -> None:
    try:
        df.to_parquet(path)
        logger.info("💾 已缓存: %s (%d 条)", path.name, len(df))
    except Exception as e:
        logger.warning("缓存写入失败: %s", e)


def _try_akshare_em(symbol: str, start: str, end: str) -> Optional[pd.DataFrame]:
    """akshare 东财接口"""
    try:
        import akshare as ak
        logger.info("尝试 akshare-东财: %s", symbol)
        df = ak.stock_zh_a_hist(
            symbol=symbol,
            period="daily",
            start_date=start.replace("-", ""),
            end_date=end.replace("-", ""),
            adjust="qfq",
        )
        if df is None or df.empty:
            return None
        df = _normalize(df)
        logger.info("✅ akshare-东财: %d 条", len(df))
        return df
    except Exception as e:
        logger.warning("akshare-东财 失败: %s", e)
        return None


def _try_akshare_sina(symbol: str, start: str, end: str) -> Optional[pd.DataFrame]:
    """akshare 新浪接口"""
    try:
        import akshare as ak
        logger.info("尝试 akshare-新浪: %s", symbol)
        # 新浪接口需要带交易所前缀
        prefix = "sh" if symbol.startswith(("6", "9")) else "sz"
        df = ak.stock_zh_a_daily(
            symbol=f"{prefix}{symbol}",
            start_date=start.replace("-", ""),
            end_date=end.replace("-", ""),
            adjust="qfq",
        )
        if df is None or df.empty:
            return None
        # 新浪接口返回的 index 是 date
        if "date" not in df.columns:
            df = df.reset_index().rename(columns={"index": "date"})
        df = _normalize(df)
        logger.info("✅ akshare-新浪: %d 条", len(df))
        return df
    except Exception as e:
        logger.warning("akshare-新浪 失败: %s", e)
        return None


def _try_akshare_tencent(symbol: str, start: str, end: str) -> Optional[pd.DataFrame]:
    """akshare 腾讯接口"""
    try:
        import akshare as ak
        logger.info("尝试 akshare-腾讯: %s", symbol)
        prefix = "sh" if symbol.startswith(("6", "9")) else "sz"
        df = ak.stock_zh_a_hist_tx(
            symbol=f"{prefix}{symbol}",
            start_date=start.replace("-", ""),
            end_date=end.replace("-", ""),
            adjust="qfq",
        )
        if df is None or df.empty:
            return None
        df = _normalize(df)
        logger.info("✅ akshare-腾讯: %d 条", len(df))
        return df
    except Exception as e:
        logger.warning("akshare-腾讯 失败: %s", e)
        return None


def _try_baostock(symbol: str, start: str, end: str) -> Optional[pd.DataFrame]:
    """
    baostock 兜底：免费、稳定、无需 token。
    缺点：需要先 login/logout。
    """
    try:
        import baostock as bs
    except ImportError:
        logger.warning("baostock 未安装，跳过。安装: pip install baostock")
        return None

    try:
        logger.info("尝试 baostock: %s", symbol)
        lg = bs.login()
        if lg.error_code != "0":
            logger.warning("baostock 登录失败: %s", lg.error_msg)
            return None

        # 代码格式：sz.300750 / sh.600519
        if symbol.startswith(("6", "9")):
            bs_code = f"sh.{symbol}"
        else:
            bs_code = f"sz.{symbol}"

        rs = bs.query_history_k_data_plus(
            bs_code,
            "date,open,high,low,close,volume",
            start_date=start,
            end_date=end,
            frequency="d",
            adjustflag="2",   # 前复权
        )
        if rs.error_code != "0":
            logger.warning("baostock 查询失败: %s", rs.error_msg)
            bs.logout()
            return None

        rows = []
        while rs.next():
            rows.append(rs.get_row_data())
        bs.logout()

        if not rows:
            return None

        df = pd.DataFrame(rows, columns=["date", "open", "high", "low", "close", "volume"])
        df = _normalize(df)
        logger.info("✅ baostock: %d 条", len(df))
        return df
    except Exception as e:
        logger.warning("baostock 失败: %s", e)
        try:
            bs.logout()
        except Exception:
            pass
        return None


# ══════════════════════════════════════════════════════════
# CLI 测试
# ══════════════════════════════════════════════════════════
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    import sys
    symbol = sys.argv[1] if len(sys.argv) > 1 else "300750"

    df = load_data(symbol, start_date="2023-01-01")
    if df.empty:
        print("❌ 数据加载失败")
        sys.exit(1)

    print(f"\n✅ {symbol} 数据概览:")
    print(f"   行数: {len(df)}")
    print(f"   区间: {df.index[0].date()} ~ {df.index[-1].date()}")
    print(f"   列:   {list(df.columns)}")
    print(f"\n前 3 行:\n{df.head(3)}")
    print(f"\n后 3 行:\n{df.tail(3)}")


# ══════════════════════════════════════════════════════════
# 回测结果
# ══════════════════════════════════════════════════════════
@dataclass
class BacktestResult:
    total_return: float = 0.0
    sharpe: float = 0.0
    max_drawdown: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    equity_curve: List[float] = field(default_factory=list)
    trades: List[Dict[str, Any]] = field(default_factory=list)


# ══════════════════════════════════════════════════════════
# 单标回测
# ══════════════════════════════════════════════════════════
def backtest_symbol(
    symbol: str,
    data: pd.DataFrame,
    initial_capital: float = 100_000.0,
    warmup: int = 60,
    max_bars: int = 500,
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[BacktestResult, List[Dict[str, Any]]]:

    logger.info("=" * 60)
    logger.info("回测 %s | bars=%d | 起始资金=%.0f", symbol, len(data), initial_capital)
    logger.info("=" * 60)

    orchestrator = MultiAgentOrchestrator(config or {})





    stats = {
        "total_bars": 0,
        "orchestrator_hold": 0,
        "orchestrator_buy": 0,
        "orchestrator_sell": 0,
        "position_opened": 0,
        "position_closed": 0,
        "execution_veto": 0,       # 开仓时 cash 不够 / pos_size=0
        "sell_but_flat": 0,        # ← 加这一行
        "buy_but_holding": 0, 
        "hold_and_same_signal": 0,      # ★ 加这一行
        "conf_buckets": {},        # 置信度分布
    }




    
    # ── 交易状态 ──────────────────────────
    cash = initial_capital
    position_shares = 0.0
    entry_price = 0.0
    stop_loss = 0.0
    take_profit = 0.0

    # ══════════════════════════════════════════════════════
    # P2-1 关键状态：待回填的 decision_id
    # ══════════════════════════════════════════════════════
    # 用一个显式的 pending 结构，记录"哪个 decision 对应哪笔开仓"
    # 而不是只存一个 decision_id（避免多笔重叠时丢信息）
    pending: Optional[Dict[str, Any]] = None
    # pending = {
    #     "decision_id": str,
    #     "entry_price": float,
    #     "direction": int,      # +1 多 / -1 空
    #     "entry_bar": int,
    # }

    equity_curve: List[float] = []
    trades: List[Dict[str, Any]] = []
    consecutive_losses = 0
    peak_value = initial_capital

    day_start_value = initial_capital    # 当日开盘净值
    last_day = None                      # 上一根 bar 的日期
    daily_trades = 0 
    n = min(len(data), warmup + max_bars)

    for i in range(warmup, n):
        window = data.iloc[: i + 1]
        bar = data.iloc[i]
        price = float(bar["close"])

        current_day = bar.name.date() if hasattr(bar.name, "date") else None
        if current_day != last_day:
            day_start_value = cash + position_shares * price
            daily_trades = 0
            consecutive_losses = 0  
            last_day = current_day
           

        # ══════════════════════════════════════════════════
        # P2-1：先回填上一根 K 线的实际收益
        # ══════════════════════════════════════════════════
        if pending is not None:
            ret = (price - pending["entry_price"]) / pending["entry_price"]
            if pending["direction"] < 0:
                ret = -ret
            orchestrator.update_actual_return(pending["decision_id"], ret)
            pending = None

        # ── 持仓管理：止损 / 止盈 ──
        if position_shares != 0:
            closed_pnl = 0.0
            hit = None
            if position_shares > 0:
                if price <= stop_loss:
                    hit = "stop"
                elif price >= take_profit:
                    hit = "tp"
            else:
                if price >= stop_loss:
                    hit = "stop"
                elif price <= take_profit:
                    hit = "tp"

            if hit:
                if position_shares > 0:
                    closed_pnl = (price - entry_price) * position_shares
                    cash += position_shares * price
                else:
                    closed_pnl = (entry_price - price) * abs(position_shares)
                    cash += abs(position_shares) * (2 * entry_price - price)

                trades.append({
                    "side": "SELL" if position_shares > 0 else "BUY",
                    "price": price,
                    "shares": abs(position_shares),
                    "pnl": closed_pnl,
                    "reason": hit,
                    "bar": i,
                })

                if closed_pnl < 0:
                    consecutive_losses += 1
                else:
                    consecutive_losses = 0

                position_shares = 0.0

        # ── 构造组合状态 ──
        total_value = cash + position_shares * price
        peak_value = max(peak_value, total_value)

        portfolio = {}
        if position_shares != 0:
            portfolio[symbol] = {
                "shares": position_shares,
                "avg_price": entry_price,
            }

        portfolio_state = {
            "total_value": total_value,
            "cash": cash,
            "long_value": max(position_shares, 0) * price,
            "short_value": max(-position_shares, 0) * price,
            "peak_value": peak_value,
            "daily_pnl": total_value - day_start_value,
            "daily_trades": daily_trades,
            "consecutive_losses": consecutive_losses,
        }

        # ── 调 Orchestrator ──
        try:
            decision = orchestrator.analyze(
                data=window,
                symbol=symbol,
                portfolio=portfolio,
                portfolio_state=portfolio_state,
            )

            logger.info(
             "DEBUG decision: action=%s force_close=%s conf=%.3f pos=%.4f",
             decision.action,
             getattr(decision, "force_close", "NO_ATTR"),
             decision.confidence,
             decision.position_size,
)
        except Exception:
            logger.exception("[%s] analyze 失败 @ %s", symbol, bar.name)
            equity_curve.append(total_value)
            continue




   
        stats["total_bars"] += 1
        if decision.action == "hold":
            stats["orchestrator_hold"] += 1
        elif decision.action == "buy":
            stats["orchestrator_buy"] += 1
        elif decision.action == "sell":
            stats["orchestrator_sell"] += 1

        # 置信度分布（每 0.1 一个桶）
        if decision.action != "hold":
            bucket = round(decision.confidence, 1)
            stats["conf_buckets"][bucket] = stats["conf_buckets"].get(bucket, 0) + 1

        # 非 hold 时打印详情
        if decision.action != "hold":
            logger.info(
                "信号 @ %s | action=%s conf=%.3f pos=%.4f entry=%.2f sl=%.2f tp=%.2f",
                bar.name, decision.action, decision.confidence,
                decision.position_size, decision.entry_price,
                decision.stop_loss, decision.take_profit,
            )
        





        # ══════════════════════════════════════════════════
        # P2-1：开仓时记录 decision_id，下一根 K 线回填
        # ══════════════════════════════════════════════════
                # ══════════════════════════════════════════════════════
        # 6. 执行决策（四分支：开多 / 平多 / 开空 / 平空）
        # ══════════════════════════════════════════════════════

         # ★★★ 最高优先级：FORCE_CLOSE 只平仓，不开仓 ★★★
        if decision.action != Action.HOLD.value and getattr(decision, "force_close", False):
            logger.info(
                "DEBUG FORCE_CLOSE block: position_shares=%.2f, portfolio=%s, price=%.2f",
                position_shares, portfolio, price,
            )
            if position_shares > 0:
                pnl = (price - entry_price) * position_shares
                cash += position_shares * price
                trades.append({
                    "side": "SELL", "price": price,
                    "shares": position_shares, "pnl": pnl,
                    "bar": i, "reason": "force_close",
                })
                position_shares = 0.0
                entry_price = 0.0
                stats["position_closed"] += 1
                if pending_decision_id:
                    ret = (price - pending_entry_price) / pending_entry_price
                    orchestrator.update_actual_return(pending_decision_id, ret)
                    pending_decision_id = None
            elif position_shares < 0:
                shares_short = abs(position_shares)
                pnl = (entry_price - price) * shares_short
                cash += shares_short * entry_price + pnl
                trades.append({
                    "side": "BUY", "price": price,
                    "shares": shares_short, "pnl": pnl,
                    "bar": i, "reason": "force_close",
                })
                position_shares = 0.0
                entry_price = 0.0
                stats["position_closed"] += 1
                if pending_decision_id:
                    ret = (price - pending_entry_price) / pending_entry_price
                    orchestrator.update_actual_return(pending_decision_id, -ret)
                    pending_decision_id = None
            logger.info(
                "DEBUG FORCE_CLOSE end: position_shares=%.2f, cash=%.2f",
                position_shares, cash,
            )
            equity_curve.append(cash + position_shares * price)
            orchestrator.reset_risk_state_after_close(cash)  
            continue




        pos_pct = float(decision.position_size)

        # ── 分支 1：空仓 + BUY → 开多 ──────────────
        if position_shares == 0 and decision.action == Action.BUY.value:
            if pos_pct > 0 and decision.entry_price > 0:
                notional = total_value * pos_pct
                shares = notional / decision.entry_price
                if cash >= notional:
                    stats["position_opened"] += 1
                    cash -= notional
                    position_shares = shares
                    entry_price = decision.entry_price
                    stop_loss = decision.stop_loss
                    take_profit = decision.take_profit
                    pending_decision_id = decision.decision_id
                    pending_entry_price = entry_price
                    trades.append({
                        "side": "BUY", "price": entry_price,
                        "shares": shares, "pnl": 0.0,
                        "bar": i, "reason": "entry",
                    })
                else:
                    stats["execution_veto"] += 1

        # ── 分支 2：持多 + SELL → 平多 ────────────
        elif position_shares > 0 and decision.action == Action.SELL.value:
            pnl = (price - entry_price) * position_shares
            cash += position_shares * price
            stats["position_closed"] += 1
            trades.append({
                "side": "SELL", "price": price,
                "shares": position_shares, "pnl": pnl,
                "bar": i, "reason": "signal",
            })
            if pending_decision_id:
                ret = (price - pending_entry_price) / pending_entry_price
                orchestrator.update_actual_return(pending_decision_id, ret)
                pending_decision_id = None
            position_shares = 0.0
            entry_price = 0.0

        # ── 分支 3：空仓 + SELL → 开空 ────────────
        elif position_shares == 0 and decision.action == Action.SELL.value:
            stats["sell_but_flat"] = stats.get("sell_but_flat", 0) + 1
            if pos_pct > 0 and decision.entry_price > 0:
                notional = total_value * pos_pct
                shares = notional / decision.entry_price
                # 做空：不扣现金（保证金另算），记录负仓位
                # 简化模型：占用等额现金作为保证金
                if cash >= notional:
                    stats["position_opened"] += 1
                    cash -= notional                     # 冻结保证金
                    position_shares = -shares
                    entry_price = decision.entry_price
                    stop_loss = decision.stop_loss       # 做空时 SL 在上方
                    take_profit = decision.take_profit   # TP 在下方
                    pending_decision_id = decision.decision_id
                    pending_entry_price = entry_price
                    trades.append({
                        "side": "SHORT", "price": entry_price,
                        "shares": shares, "pnl": 0.0,
                        "bar": i, "reason": "entry",
                    })
                else:
                    stats["execution_veto"] += 1

        # ── 分支 4：持空 + BUY → 平空 ─────────────
        elif position_shares < 0 and decision.action == Action.BUY.value:
            shares_short = abs(position_shares)
            pnl = (entry_price - price) * shares_short
            # 返还保证金 + 盈亏
            cash += shares_short * entry_price + pnl
            stats["position_closed"] += 1
            trades.append({
                "side": "COVER", "price": price,
                "shares": shares_short, "pnl": pnl,
                "bar": i, "reason": "signal",
            })
            if pending_decision_id:
                ret = (price - pending_entry_price) / pending_entry_price
                orchestrator.update_actual_return(pending_decision_id, -ret)
                pending_decision_id = None
            position_shares = 0.0
            entry_price = 0.0

                # ── 分支 5：持多 + BUY → 已有仓位，不重复开仓 ────
        elif position_shares > 0 and decision.action == Action.BUY.value:
            stats["hold_and_same_signal"] = stats.get("hold_and_same_signal", 0) + 1

        # ── 分支 6：持空 + SELL → 已有仓位，不重复开仓 ──
        elif position_shares < 0 and decision.action == Action.SELL.value:
            stats["hold_and_same_signal"] = stats.get("hold_and_same_signal", 0) + 1


        equity_curve.append(cash + position_shares * price)

    # ── 计算指标 ──
    equity = pd.Series(equity_curve)
    result = _compute_metrics(equity, trades, initial_capital)





    logger.info("─" * 60)
    logger.info("信号统计 %s", symbol)
    logger.info("  执行层否决:       %d", stats["execution_veto"])
    logger.info("  空仓时 SELL 次数: %d", stats.get("sell_but_flat", 0))   # ★ 加这一行
    logger.info("  置信度分布:       %s", dict(sorted(stats["conf_buckets"].items())))
    logger.info("  总 K 线数:        %d", stats["total_bars"])
    logger.info("  Orchestrator HOLD: %d (%.1f%%)",
                stats["orchestrator_hold"],
                stats["orchestrator_hold"] / max(stats["total_bars"], 1) * 100)
    logger.info("  Orchestrator BUY:  %d (%.1f%%)",
                stats["orchestrator_buy"],
                stats["orchestrator_buy"] / max(stats["total_bars"], 1) * 100)
    logger.info("  Orchestrator SELL: %d (%.1f%%)",
                stats["orchestrator_sell"],
                stats["orchestrator_sell"] / max(stats["total_bars"], 1) * 100)
    logger.info("  实际开仓:         %d", stats["position_opened"])
    logger.info("  执行层否决:       %d", stats["execution_veto"])
    logger.info("  空仓时 SELL 次数: %d", stats.get("sell_but_flat", 0))
    logger.info("  持仓时同向信号:   %d", stats.get("hold_and_same_signal", 0))   # ★ 加这一行
    logger.info("  置信度分布:       %s", dict(sorted(stats["conf_buckets"].items())))
    logger.info("─" * 60)









    logger.info("总收益: %.2f%%", result.total_return * 100)
    logger.info("夏普:   %.2f", result.sharpe)
    logger.info("最大回撤: %.2f%%", result.max_drawdown * 100)
    logger.info("交易次数: %d", result.total_trades)
    logger.info("胜率:   %.2f%%", result.win_rate * 100)

    return result, trades
    

def _compute_metrics(
    equity: pd.Series,
    trades: List[Dict[str, Any]],
    initial_capital: float,
) -> BacktestResult:
    res = BacktestResult()
    if len(equity) < 2:
        return res

    res.equity_curve = equity.tolist()

    total_return = equity.iloc[-1] / equity.iloc[0] - 1
    res.total_return = float(total_return)

    returns = equity.pct_change().dropna()
    if returns.std() > 0:
        res.sharpe = float(returns.mean() / returns.std() * np.sqrt(252))

    running_max = equity.expanding().max()
    dd = (equity - running_max) / running_max
    res.max_drawdown = float(abs(dd.min()))

    closed = [t for t in trades if t.get("pnl", 0) != 0 or t.get("reason") in ("stop", "tp")]
    res.total_trades = len(closed)
    if closed:
        pnls = [t["pnl"] for t in closed]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]
        res.winning_trades = len(wins)
        res.win_rate = len(wins) / len(closed)
        res.avg_win = float(np.mean(wins)) if wins else 0.0
        res.avg_loss = float(np.mean(losses)) if losses else 0.0
        if losses and sum(losses) != 0:
            res.profit_factor = abs(sum(wins) / sum(losses))
        elif wins:
            res.profit_factor = float("inf")

    res.trades = trades
    return res


# ══════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════
def main():
    symbols = ["300750"]   # 可加 ["600519", "000001"]

    # ── 关键：调低阈值，避免信号被层层压制 ──
    config = {
        "voting_threshold": 0.40,     # 原 0.55 → 0.40
        "learning_rate": 0.05,
        "min_samples_to_learn": 20,
        "alpha_config": {
            "min_confidence": 0.45,   # 原 0.52 → 0.45
        },
        "execution_config": {
            "min_rr": 1.5,            # 原 1.8 → 1.5
            "total_capital": 100_000,
        },
        "risk_config": {
            "max_daily_loss_pct": 0.05,
            "trailing_stop_dd": 0.20,       # 12% → 20%
            "max_drawdown_threshold": 0.15,
        },
    }

    all_results = {}
    for sym in symbols:
        data = load_data(sym, start_date="2023-01-01")
        if data is None or data.empty:
            logger.error("[%s] 无数据，跳过", sym)
            continue

        try:
            res, trades = backtest_symbol(
                sym, data,
                initial_capital=100_000,
                warmup=60,
                max_bars=300,
                config=config,
            )
            all_results[sym] = res
        except Exception:
            logger.exception("[%s] 回测失败", sym)

    # ── 汇总 ──
    logger.info("=" * 60)
    logger.info("汇总")
    for sym, r in all_results.items():
        logger.info(
            "%s | 收益=%.2f%% 夏普=%.2f 回撤=%.2f%% 交易=%d 胜率=%.1f%%",
            sym, r.total_return * 100, r.sharpe,
            r.max_drawdown * 100, r.total_trades, r.win_rate * 100,
        )

    # ── 保存 ──
    out = ROOT / "backtest_results.csv"
    rows = []
    for sym, r in all_results.items():
        rows.append({
            "symbol": sym,
            "total_return": r.total_return,
            "sharpe": r.sharpe,
            "max_drawdown": r.max_drawdown,
            "total_trades": r.total_trades,
            "win_rate": r.win_rate,
            "profit_factor": r.profit_factor,
        })
    pd.DataFrame(rows).to_csv(out, index=False)
    logger.info("结果已保存: %s", out)


if __name__ == "__main__":
    main()