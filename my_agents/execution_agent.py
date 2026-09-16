"""
ExecutionAgent v5
把 AlphaAgent 的方向信号转换成可执行订单参数

"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .base_agent import Action, AgentOutput, BaseAgent, MarketRegime, REQUIRED_COLS

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════
#        枚举 & 常量
# ══════════════════════════════════════════════════════════
class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP_LIMIT = "stop_limit"


class StopStrategy(str, Enum):
    ATR = "atr"
    SUPPORT_RESISTANCE = "sr"
    VOLATILITY = "volatility"


REGIME_TP_MULTIPLIER: Dict[str, float] = {
    MarketRegime.TRENDING_UP.value: 3.5,
    MarketRegime.TRENDING_DOWN.value: 3.5,
    MarketRegime.BREAKOUT.value: 4.0,
    MarketRegime.RANGING.value: 2.0,
    MarketRegime.VOLATILE.value: 1.8,
    MarketRegime.REVERSAL.value: 2.5,
    "unknown": 3.0,
}


# ══════════════════════════════════════════════════════════
#       数据结构
# ══════════════════════════════════════════════════════════
@dataclass
class SlippageEstimate:
    """
    [REMOVED] impact_bps：原 Kyle's Lambda 项已随 MarketAgent 删除。
    [REFACTOR-08] spread_bps 只在有盘口数据时填写，无盘口置 0。
    """
    base_bps: float
    volume_bps: float
    spread_bps: float
    impact_bps: float
    total_bps: float
    total_price: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "slip_base_bps": round(self.base_bps, 3),
            "slip_volume_bps": round(self.volume_bps, 3),
            "slip_spread_bps": round(self.spread_bps, 3),
            "slip_impact_bps": round(self.impact_bps, 3),
            "slip_total_bps": round(self.total_bps, 3),
            "slip_total_price": round(self.total_price, 8),
        }


@dataclass
class OrderParams:
    order_type: OrderType
    entry: float
    limit_price: Optional[float]
    stop_loss: float
    take_profit: float
    stop_pct: float
    tp_pct: float
    risk_reward: float
    position_size: float
    position_shares: float
    notional: float
    slippage: SlippageEstimate
    stop_strategy: StopStrategy
    size_multiplier: float = 1.0
    size_breakdown: Dict[str, float] = field(default_factory=dict)
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "order_type": self.order_type.value,
            "entry": round(self.entry, 8),
            "limit_price": round(self.limit_price, 8) if self.limit_price else None,
            "stop_loss": round(self.stop_loss, 8),
            "take_profit": round(self.take_profit, 8),
            "stop_pct": round(self.stop_pct, 6),
            "tp_pct": round(self.tp_pct, 6),
            "risk_reward": round(self.risk_reward, 4),
            "position_size": round(self.position_size, 4),
            "position_shares": round(self.position_shares, 4),
            "notional": round(self.notional, 2),
            "stop_strategy": self.stop_strategy.value,
            "size_multiplier": round(self.size_multiplier, 4),
            "size_breakdown": {k: round(v, 4) for k, v in self.size_breakdown.items()},
            **self.slippage.to_dict(),
        }


@dataclass
class TradeState:
    """
    [SCOPE-07] 仅保留 last_signal_ts 用于审计/调试。
    日内次数、日内盈亏、连败计数全部移交 RiskAgent。
    """
    last_signal_ts: float = 0.0


# ══════════════════════════════════════════════════════════
# 配置
# ══════════════════════════════════════════════════════════
DEFAULT_CONFIG: Dict[str, Any] = {
    # ATR
    "atr_period": 14,
    "atr_stop_multiplier": 1.5,
    "atr_tp_multiplier": 3.0,

    # 止损边界
    "min_stop_pct": 0.004,
    "max_stop_pct": 0.05,

    # 止盈边界
    "min_rr": 1.8,
    "max_rr": 6.0,

    # 止损策略
    "stop_strategy": StopStrategy.ATR.value,
    "sr_buffer_atr_mult": 0.3,

    # 基础仓位
    "risk_per_trade": 0.01,
    "max_position_pct": 0.20,
    "kelly_fraction": 0.25,
    "vol_target": 0.15,
    "total_capital": 100_000,

    # 滑点
    "slippage_base_bps": 2.0,
    "slippage_vol_k": 0.5,
    "slippage_max_bps": 20.0,

    # 限价单
    "use_limit_order": True,
    "limit_offset_atr_mult": 0.1,

    # ── 单笔硬门槛（[SCOPE-03] 只保留"可成交性"两项）──
    "hard_min_volume": 100_000,
    "hard_min_atr_pct": 0.0005,

    # ── 仓位调整链（[SCOPE-04] 只保留 ATR + 滑点）──
    "size_adj": {
        "atr_high_pct": 0.03,
        "atr_high_min_mult": 0.5,
        "atr_low_pct": 0.002,
        "atr_low_min_mult": 0.7,
        "slip_soft_bps": 5.0,
        "slip_hard_bps": 15.0,
        "slip_min_mult": 0.5,
        "size_floor": 0.20,
    },

    # 交易时段假设（年化波动率折算用）
    "trading_hours_per_day": 6.5,
    "trading_days_per_year": 252,
    "default_bar_seconds": 300.0,

    # [SCOPE-06] 来自 RiskAgent 的默认值
    "default_risk_size_mult": 1.0,
}


# ══════════════════════════════════════════════════════════
# ExecutionAgent
# ══════════════════════════════════════════════════════════
class ExecutionAgent(BaseAgent):
    name = "execution"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        merged = {**DEFAULT_CONFIG, **(config or {})}
        if config and "size_adj" in config:
            merged["size_adj"] = {**DEFAULT_CONFIG["size_adj"], **config["size_adj"]}
        super().__init__(merged)
        self._validate_config()
        self._trade_states: Dict[str, TradeState] = {}
        self._last_bar_seconds: float = self.config["default_bar_seconds"]

    # ══════════════════════════════════════
    # 主入口
    # ══════════════════════════════════════
    def run(
        self,
        data: pd.DataFrame,
        symbol: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> AgentOutput:

        context = context or {}

        err = self.validate_ohlcv(data, min_len=30)
        if err:
            return self._hold(err, as_warning=True)

        # ── Alpha 方向 ────────────────────────
        alpha_action_raw = context.get("alpha_action", Action.HOLD.value)
        alpha_conf = float(context.get("alpha_confidence", 0.0))

        if isinstance(alpha_action_raw, Action):
            action_enum = alpha_action_raw
        else:
            try:
                action_enum = Action(alpha_action_raw)
            except ValueError:
                return self._hold(f"invalid alpha_action={alpha_action_raw!r}")

        if action_enum == Action.HOLD or alpha_conf <= 0:
            return self._hold("no directional input from alpha")

        # ── [SCOPE-06] 消费 RiskAgent 的 gate / size_mult ──
        risk_gate = str(context.get("risk_gate", "clear")).lower()
        if risk_gate == "block":
            return self._hold(f"risk_gate=block（RiskAgent 否决）")
        risk_size_mult = float(
            context.get("risk_size_mult", self.config["default_risk_size_mult"])
        )
        if not (0.0 < risk_size_mult <= 1.0):
            # 只在 (0,1] 内合法；>1 说明上游想放大，拒绝
            risk_size_mult = float(np.clip(risk_size_mult, 0.0, 1.0))

        is_buy = action_enum == Action.BUY

        df = data.dropna(subset=list(REQUIRED_COLS)).copy()
        price = float(df["close"].iloc[-1])
        atr = self.atr(df, self.config["atr_period"])
        atr_pct = atr / price if price > 0 else 0.0

        if isinstance(df.index, pd.DatetimeIndex) and len(df) >= 2:
            delta = (df.index[-1] - df.index[-2]).total_seconds()
            if delta > 0:
                self._last_bar_seconds = float(delta)

        # ── MarketAgent 特征 ──────────────────
        market_feat = context.get("market_features", {}) or {}
        regime = market_feat.get("regime", "unknown")
        ob_spread_pct = float(market_feat.get("ob_spread_pct", 0.0))
        support = float(market_feat.get("support", df["low"].tail(20).min()))
        resistance = float(market_feat.get("resistance", df["high"].tail(20).max()))

        # ══════════════════════════════════════
        # 单笔硬门槛（[SCOPE-03] 只保留 H1/H1b）
        # ══════════════════════════════════════
        gate_result = self._hard_gates(
            df=df, price=price, atr_pct=atr_pct,
        )
        if gate_result:
            return self._hold(gate_result)

        # ══════════════════════════════════════
        # 滑点估算
        # ══════════════════════════════════════
        slippage = self._estimate_slippage(
            price=price, atr=atr, df=df,
            ob_spread_pct=ob_spread_pct,
            is_buy=is_buy,
        )

        # ══════════════════════════════════════
        # 止损 / 止盈
        # ══════════════════════════════════════
        entry = price + slippage.total_price if is_buy else price - slippage.total_price
        stop_pct, stop_strategy = self._calc_stop(
            entry=entry, atr=atr, is_buy=is_buy,
            support=support, resistance=resistance, df=df,
        )
        tp_pct = self._calc_tp(
            entry=entry, atr=atr, stop_pct=stop_pct,
            is_buy=is_buy, regime=regime,
            resistance=resistance, support=support,
        )

        rr = tp_pct / stop_pct if stop_pct > 0 else 0.0
        if rr < self.config["min_rr"]-1e-6:
            return self._hold(f"RR={rr:.2f} < min={self.config['min_rr']:.2f}")

        # ══════════════════════════════════════
        # 基础仓位（Kelly + 波动率）
        # ══════════════════════════════════════
        base_position_size, base_shares = self._calc_position(
            entry=entry, stop_pct=stop_pct,
            alpha_conf=alpha_conf, atr=atr, rr=rr,
        )

        # ══════════════════════════════════════
        # 单笔仓位调整链（[SCOPE-04] ATR + 滑点）
        # ══════════════════════════════════════
        local_mult, breakdown, adj_notes = self._size_adjustment_chain(
            atr_pct=atr_pct,
            slippage_bps=slippage.total_bps,
        )
        breakdown["risk_gate"] = risk_size_mult

        # ── 最终乘数 = 本地 × RiskAgent ────────
        total_mult = float(np.clip(
            local_mult * risk_size_mult,
            self.config["size_adj"]["size_floor"],
            1.0,
        ))
        breakdown["total"] = total_mult

        position_size = base_position_size * total_mult
        position_shares = base_shares * total_mult

        # ── 订单类型选择 ──────────────────────
        order_type, limit_price = self._select_order_type(
            price=price, atr=atr, is_buy=is_buy,
        )

        ref_price = limit_price if limit_price else entry
        if is_buy:
            stop_price = ref_price * (1 - stop_pct)
            tp_price = ref_price * (1 + tp_pct)
        else:
            stop_price = ref_price * (1 + stop_pct)
            tp_price = ref_price * (1 - tp_pct)

        notional = position_shares * ref_price

        # ── 审计时间戳（[SCOPE-07]）───────────
        state = self._get_state(symbol)
        if isinstance(df.index, pd.DatetimeIndex):
            state.last_signal_ts = df.index[-1].timestamp()

        reasons = [
            f"entry={entry:.6f}",
            f"SL={stop_price:.6f}({stop_pct:.2%})",
            f"TP={tp_price:.6f}({tp_pct:.2%})",
            f"RR={rr:.2f}",
            f"base_size={base_position_size:.2%}",
            f"local_mult={local_mult:.3f}",
            f"risk_mult={risk_size_mult:.3f}",
            f"final_size={position_size:.2%}",
            f"order={order_type.value}",
            f"stop_strategy={stop_strategy.value}",
            f"slip={slippage.total_bps:.2f}bps",
            f"regime={regime}",
            f"risk_gate={risk_gate}",
        ] + adj_notes

        params = OrderParams(
            order_type=order_type,
            entry=entry,
            limit_price=limit_price,
            stop_loss=stop_price,
            take_profit=tp_price,
            stop_pct=stop_pct,
            tp_pct=tp_pct,
            risk_reward=rr,
            position_size=position_size,
            position_shares=position_shares,
            notional=notional,
            slippage=slippage,
            stop_strategy=stop_strategy,
            size_multiplier=total_mult,
            size_breakdown=breakdown,
            reasons=reasons,
        )

        return AgentOutput(
            agent_name=self.name,
            action=action_enum,
            confidence=round(alpha_conf, 4),
            reasons=reasons,
            features=params.to_dict(),
        )

    # ══════════════════════════════════════
    # 单笔硬门槛（[SCOPE-03] 只保留可成交性）
    # ══════════════════════════════════════
    def _hard_gates(
        self,
        df: pd.DataFrame,
        price: float,
        atr_pct: float,
    ) -> Optional[str]:
        """
        只负责"这笔能不能成交"，不负责"今天还能不能交易"。
        日内次数 / 日内亏损 由 RiskAgent 判定。
        """
        # H1：成交量 —— 无法成交的底线
        avg_vol = float(df["volume"].tail(20).mean())
        if avg_vol < self.config["hard_min_volume"]:
            return f"H1: avg_vol={avg_vol:.0f} < min={self.config['hard_min_volume']}"

        # H1b：价格僵死 —— ATR% 过低，止损/止盈都会失真
        if atr_pct < self.config["hard_min_atr_pct"]:
            return f"H1b: atr_pct={atr_pct:.4%} < min={self.config['hard_min_atr_pct']:.4%}"

        return None

    # ══════════════════════════════════════
    # 单笔仓位调整链（[SCOPE-04] ATR + 滑点）
    # ══════════════════════════════════════
    def _size_adjustment_chain(
        self,
        atr_pct: float,
        slippage_bps: float,
    ) -> Tuple[float, Dict[str, float], List[str]]:
        """
        乘法降仓：ATR + 滑点。
        连败项已移交 RiskAgent。
        """
        cfg = self.config["size_adj"]
        breakdown: Dict[str, float] = {}
        notes: List[str] = []

        # ── 1. ATR 降仓 ────────────────────────
        if atr_pct > cfg["atr_high_pct"]:
            atr_mult = self._linear_ramp(
                value=atr_pct,
                soft=cfg["atr_high_pct"],
                hard=cfg["atr_high_pct"] * 2.0,
                min_mult=cfg["atr_high_min_mult"],
            )
            notes.append(f"adj_atr_high={atr_mult:.2f}({atr_pct:.2%})")
        elif atr_pct < cfg["atr_low_pct"]:
            atr_mult = cfg["atr_low_min_mult"]
            notes.append(f"adj_atr_low={atr_mult:.2f}({atr_pct:.4%})")
        else:
            atr_mult = 1.0
        breakdown["atr"] = atr_mult

        # ── 2. 滑点降仓 ────────────────────────
        slip_mult = self._linear_ramp(
            value=slippage_bps,
            soft=cfg["slip_soft_bps"],
            hard=cfg["slip_hard_bps"],
            min_mult=cfg["slip_min_mult"],
        )
        breakdown["slippage"] = slip_mult
        if slip_mult < 1.0:
            notes.append(f"adj_slip={slip_mult:.2f}({slippage_bps:.1f}bps)")

        local_total = float(np.clip(
            atr_mult * slip_mult,
            cfg["size_floor"],
            1.0,
        ))
        breakdown["local_total"] = local_total

        return local_total, breakdown, notes

    @staticmethod
    def _linear_ramp(
        value: float, soft: float, hard: float, min_mult: float
    ) -> float:
        if value <= soft:
            return 1.0
        if value >= hard:
            return float(min_mult)
        ratio = (value - soft) / max(hard - soft, 1e-8)
        return float(1.0 - ratio * (1.0 - min_mult))

    # ══════════════════════════════════════
    # 动态滑点估算
    # ══════════════════════════════════════
    def _estimate_slippage(
        self,
        price: float, atr: float, df: pd.DataFrame,
        ob_spread_pct: float,
        is_buy: bool,
    ) -> SlippageEstimate:
        """
        滑点 = 基础 + 成交量调整 + 盘口价差。
        无盘口数据时 spread_bps = 0。
        """
        base_bps = self.config["slippage_base_bps"]

        avg_vol = float(df["volume"].tail(20).mean())
        cur_vol = float(df["volume"].iloc[-1])
        vol_ratio = cur_vol / avg_vol if avg_vol > 0 else 1.0
        vol_bps = self.config["slippage_vol_k"] * max(0.0, 1.0 - vol_ratio) * 5.0

        if ob_spread_pct > 0:
            spread_bps = ob_spread_pct * 10_000 * 0.5
        else:
            spread_bps = 0.0

        total_bps = float(np.clip(
            base_bps + vol_bps + spread_bps,
            base_bps,
            self.config["slippage_max_bps"],
        ))
        total_price = price * total_bps / 10_000

        return SlippageEstimate(
            base_bps=base_bps,
            volume_bps=vol_bps,
            spread_bps=spread_bps,
            impact_bps=0.0,
            total_bps=total_bps,
            total_price=total_price,
        )

    # ══════════════════════════════════════
    # 止损策略
    # ══════════════════════════════════════
    def _calc_stop(
        self, entry: float, atr: float, is_buy: bool,
        support: float, resistance: float, df: pd.DataFrame,
    ) -> Tuple[float, StopStrategy]:

        strategy_name = self.config["stop_strategy"]

        atr_stop_pct = float(np.clip(
            atr * self.config["atr_stop_multiplier"] / entry,
            self.config["min_stop_pct"],
            self.config["max_stop_pct"],
        ))

        if strategy_name == StopStrategy.SUPPORT_RESISTANCE.value:
            buffer = atr * self.config["sr_buffer_atr_mult"]
            if is_buy and 0 < support < entry:
                sr_stop_price = support - buffer
                sr_stop_pct = (entry - sr_stop_price) / entry
            elif (not is_buy) and resistance > entry:
                sr_stop_price = resistance + buffer
                sr_stop_pct = (sr_stop_price - entry) / entry
            else:
                sr_stop_pct = atr_stop_pct

            stop_pct = float(np.clip(
                sr_stop_pct, self.config["min_stop_pct"], self.config["max_stop_pct"]
            ))
            return stop_pct, StopStrategy.SUPPORT_RESISTANCE

        elif strategy_name == StopStrategy.VOLATILITY.value:
            returns = df["close"].pct_change().dropna()
            vol_20 = float(returns.tail(20).std()) if len(returns) >= 20 else 0.0
            vol_stop_pct = float(np.clip(
                vol_20 * 2.5,
                self.config["min_stop_pct"],
                self.config["max_stop_pct"],
            ))
            return vol_stop_pct, StopStrategy.VOLATILITY

        return atr_stop_pct, StopStrategy.ATR

    # ══════════════════════════════════════
    # 止盈策略
    # ══════════════════════════════════════
    def _calc_tp(
        self,
        entry: float, atr: float, stop_pct: float,
        is_buy: bool, regime: str,
        resistance: float, support: float,
    ) -> float:
        regime_mult = REGIME_TP_MULTIPLIER.get(regime, self.config["atr_tp_multiplier"])
        atr_tp_pct = atr * regime_mult / entry

        min_acceptable = stop_pct * self.config["min_rr"]

        candidates: List[float] = []
        if atr_tp_pct >= min_acceptable:
            candidates.append(atr_tp_pct)

        if is_buy and resistance > entry:
            sr = (resistance - entry) / entry
            if sr >= min_acceptable:
                candidates.append(sr)
        elif (not is_buy) and 0 < support < entry:
            sr = (entry - support) / entry
            if sr >= min_acceptable:
                candidates.append(sr)

        swing_range = resistance - support
        if swing_range > 0:
            if is_buy:
                fib_price = support + swing_range * 0.618
                if fib_price > entry:
                    fib = (fib_price - entry) / entry
                    if fib >= min_acceptable:
                        candidates.append(fib)
            else:
                fib_price = resistance - swing_range * 0.618
                if 0 < fib_price < entry:
                    fib = (entry - fib_price) / entry
                    if fib >= min_acceptable:
                        candidates.append(fib)

        if not candidates:
            return min_acceptable

        return float(np.clip(
            min(candidates),
            min_acceptable,
            stop_pct * self.config["max_rr"],
        ))

    # ══════════════════════════════════════
    # 基础仓位计算（Kelly + 波动率）
    # ══════════════════════════════════════
    def _calc_position(
        self, entry: float, stop_pct: float,
        alpha_conf: float, atr: float, rr: float,
    ) -> Tuple[float, float]:
        capital = self.config["total_capital"]
        risk_pct = self.config["risk_per_trade"]

        base_shares = (capital * risk_pct) / (entry * stop_pct) if stop_pct > 0 else 0.0
        conf_scale = float(np.clip((alpha_conf - 0.5) / 0.5, 0.2, 1.0))

        bars_per_year = self._infer_bars_per_year()
        annual_vol = (atr / entry) * np.sqrt(bars_per_year) if entry > 0 else 0.15
        vol_scale = float(np.clip(
            self.config["vol_target"] / max(annual_vol, 0.01),
            0.3, 1.5,
        ))

        win_rate = float(np.clip(alpha_conf, 0.5, 0.9))
        odds = max(rr, self.config["min_rr"])
        kelly_raw = (win_rate * odds - (1 - win_rate)) / odds
        kelly_frac = float(np.clip(
            kelly_raw * self.config["kelly_fraction"],
            0.0, self.config["max_position_pct"],
        ))

        shares = min(
            base_shares * conf_scale * vol_scale,
            (capital * kelly_frac) / entry,
            (capital * self.config["max_position_pct"]) / entry,
        )
        shares = max(shares, 0.0)
        position_size = (shares * entry) / capital if capital > 0 else 0.0

        return float(position_size), float(shares)

    # ══════════════════════════════════════
    # 订单类型选择
    # ══════════════════════════════════════
    def _select_order_type(
        self, price: float, atr: float, is_buy: bool,
    ) -> Tuple[OrderType, Optional[float]]:
        if not self.config["use_limit_order"]:
            return OrderType.MARKET, None

        offset = atr * self.config["limit_offset_atr_mult"]
        limit_price = (price - offset) if is_buy else (price + offset)
        return OrderType.LIMIT, float(limit_price)

    # ══════════════════════════════════════
    # 工具方法
    # ══════════════════════════════════════
    def _infer_bars_per_year(self) -> float:
        sec_per_day = self.config["trading_hours_per_day"] * 3600
        sec_per_year = sec_per_day * self.config["trading_days_per_year"]
        bar_sec = max(self._last_bar_seconds, 1.0)
        return sec_per_year / bar_sec

    def _get_state(self, symbol: str) -> TradeState:
        if symbol not in self._trade_states:
            self._trade_states[symbol] = TradeState()
        return self._trade_states[symbol]

    def _hold(self, reason: str, as_warning: bool = False) -> AgentOutput:
        kwargs: Dict[str, Any] = {
            "agent_name": self.name,
            "action": Action.HOLD,
            "confidence": 0.0,
        }
        if as_warning:
            kwargs["warnings"] = [reason]
        else:
            kwargs["reasons"] = [reason]
        return AgentOutput(**kwargs)

    def _validate_config(self) -> None:
        assert self.config["min_stop_pct"] < self.config["max_stop_pct"], \
            "min_stop_pct 必须小于 max_stop_pct"
        assert self.config["min_rr"] > 0, "min_rr 必须大于 0"
        assert 0 < self.config["risk_per_trade"] < 0.1, \
            "risk_per_trade 应在 0~10% 之间"
        assert 0 < self.config["kelly_fraction"] <= 1.0, \
            "kelly_fraction 应在 (0, 1] 之间"
        assert self.config["trading_hours_per_day"] > 0, \
            "trading_hours_per_day 必须大于 0"
        assert 0.0 < self.config["default_risk_size_mult"] <= 1.0, \
            "default_risk_size_mult 应在 (0, 1]"