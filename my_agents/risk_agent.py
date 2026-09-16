"""
RiskAgent(中低频日内精简版)

职责边界（与 ExecutionAgent 明确分工）：
  - RiskAgent:组合层 + 累计层风险(回撤、敞口、日内累计、VaR/CVaR、连败)
  - ExecutionAgent单笔交易参数(仓位、止损、止盈、滑点、订单类型)

"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .base_agent import Action, AgentOutput, BaseAgent, REQUIRED_COLS, RiskLevel

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════
# 枚举
# ══════════════════════════════════════════════════════════
class RiskGate(str, Enum):
    """[DEL-7] 三级闸门"""
    CLEAR = "clear"       # 放行
    REDUCE = "reduce"     # 降仓（禁止加仓，允许平仓；仓位乘数由调用方处理）
    BLOCK = "block"       # 禁止所有新开仓
    FORCE_CLOSE = "force_close"


# ══════════════════════════════════════════════════════════
# 数据结构
# ══════════════════════════════════════════════════════════
@dataclass
class PortfolioState:
    """组合实时状态（[DEL-8] 去掉 margin / leverage 相关字段）"""
    total_value: float = 0.0
    cash: float = 0.0
    long_value: float = 0.0          # 多头市值
    short_value: float = 0.0         # 空头市值（绝对值）
    peak_value: float = 0.0
    current_drawdown: float = 0.0
    max_drawdown: float = 0.0
    consecutive_losses: int = 0
    daily_pnl: float = 0.0
    daily_trades: int = 0
    timestamp: float = 0.0


    @property
    def net_exposure(self) -> float:
        eq = max(self.total_value, 1.0)
        return (self.long_value - self.short_value) / eq

    @property
    def gross_exposure(self) -> float:
        eq = max(self.total_value, 1.0)
        return (self.long_value + self.short_value) / eq


@dataclass
class PositionRisk:
    """单个持仓的风险摘要"""
    symbol: str
    shares: float
    avg_price: float
    current_price: float
    position_value: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    weight: float = 0.0


# ══════════════════════════════════════════════════════════
# 配置
# ══════════════════════════════════════════════════════════
DEFAULT_CONFIG: Dict[str, Any] = {
    # VaR / CVaR（同置信度，避免双重计数）
    "var_confidence": 0.95,
    "cvar_confidence": 0.95,
    "var_lookback": 60,               # 日内：60 根足够

    # 仓位限制
    "max_single_position": 0.20,      # 单标的最大 20%

    # 敞口
    "max_net_exposure": 0.80,
    "max_gross_exposure": 1.0,        # 日内现货通常 ≤ 1

    # 回撤控制（日内口径）
    "max_drawdown_threshold": 0.08,   # 8% 回撤触发降仓
    "trailing_stop_dd": 0.12,         # 12% 浮动止损（平仓）
    "max_daily_loss_pct": 0.03,       # 日内 3% 停手

    # 连续亏损
    "max_consecutive_losses": 4,      # 触发降仓

    # 风险评分（[DEL-1/4/5] 去掉 correlation / cvar 重复项）
    "risk_score_weights": {
        "var": 0.40,
        "drawdown": 0.40,
        "exposure": 0.20,
    },
    "risk_thresholds": {
        RiskLevel.LOW.value: 3.0,
        RiskLevel.MEDIUM.value: 6.0,
        RiskLevel.HIGH.value: 8.0,
    },

    # 数据
    "min_data_points": 30,
    "atr_period": 14,                 # 仍暴露给上层参考，但不在此处做止损
}


# ══════════════════════════════════════════════════════════
# RiskAgent
# ══════════════════════════════════════════════════════════
class RiskAgent(BaseAgent):
    name = "risk"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        merged = {**DEFAULT_CONFIG, **(config or {})}
        for k in ("risk_score_weights", "risk_thresholds"):
            if config and k in config:
                merged[k] = {**DEFAULT_CONFIG[k], **config[k]}
        super().__init__(merged)
        self._validate_config()
        self._portfolio_state = PortfolioState()

    # ══════════════════════════════════════
    # 公开接口：更新组合状态（Orchestrator 调用）
    # ══════════════════════════════════════
    def update_portfolio_state(
        self,
        total_value: float,
        cash: float,
        long_value: float = 0.0,
        short_value: float = 0.0,
        daily_pnl: float = 0.0,
        daily_trades: int = 0,
        consecutive_losses: int = 0,
        timestamp: Optional[float] = None,
    ) -> None:
        s = self._portfolio_state
        s.total_value = total_value
        s.cash = cash
        s.long_value = long_value
        s.short_value = short_value
        s.daily_pnl = daily_pnl
        s.daily_trades = daily_trades
        s.consecutive_losses = consecutive_losses
        s.timestamp = timestamp or pd.Timestamp.now().timestamp()

        if total_value > s.peak_value:
            s.peak_value = total_value
        if s.peak_value > 0:
            s.current_drawdown = (s.peak_value - total_value) / s.peak_value
            s.max_drawdown = max(s.max_drawdown, s.current_drawdown)
        else:
        # ← 必须加这个分支
            s.peak_value = max(total_value, 1.0)
            s.current_drawdown = 0.0

    def reset_daily(self) -> None:
        """每日开盘前调用"""
        s = self._portfolio_state
        s.daily_pnl = 0.0
        s.daily_trades = 0
        s.consecutive_losses = 0

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
        portfolio: Dict[str, Dict[str, Any]] = context.get("portfolio", {}) or {}

        portfolio_prices: Dict[str, float] = context.get("portfolio_prices", {}) or {}

        # ── 数据校验 ──────────────────────────
        err = self.validate_ohlcv(data, min_len=self.config["min_data_points"])
        if err:
            return self._default_output(err, is_warning=True)

        df = data.dropna(subset=list(REQUIRED_COLS)).copy()
        price = float(df["close"].iloc[-1])

        # ══════════════════════════════════════
        # 一、单标的风险
        # ══════════════════════════════════════
        returns = df["close"].pct_change().dropna()
        lookback = self.config["var_lookback"]
        r_window = returns.tail(lookback)

        var95 = self._var_historical(r_window, self.config["var_confidence"])
        cvar95 = self._cvar_historical(r_window, self.config["cvar_confidence"])
        max_dd = self._max_drawdown(df["close"])
        vol_20d = float(r_window.tail(20).std() * np.sqrt(252)) if len(r_window) >= 20 else 0.0

        # 波动率历史分位（用整段 returns）
        vol_percentile = self._vol_percentile(returns, window=20)

        # ══════════════════════════════════════
        # 二、组合层风险
        # ══════════════════════════════════════
        position_risks = self._analyze_positions(portfolio, price, symbol,portfolio_prices)
        net_exp = self._portfolio_state.net_exposure
        gross_exp = self._portfolio_state.gross_exposure

        # 单标的权重校验（当前 symbol 在组合中的占比）
        cur_weight = next(
            (p.weight for p in position_risks if p.symbol == symbol), 0.0
        )

        # ══════════════════════════════════════
        # 三、风险评分 & 分级
        # ══════════════════════════════════════
        score = self._risk_score(
            var=var95,
            dd=self._portfolio_state.current_drawdown,
            net_exp=net_exp,
            gross_exp=gross_exp,
        )
        level = self._risk_level(score)

        # ══════════════════════════════════════
        # 四、三级风控闸门
        # ══════════════════════════════════════
        daily_loss_pct = self._portfolio_state.daily_pnl / max(
            self._portfolio_state.total_value, 1.0
        )
        gate, gate_reasons = self._risk_gate(
            level=level,
            dd=self._portfolio_state.current_drawdown,
            net_exp=net_exp,
            gross_exp=gross_exp,
            daily_loss_pct=daily_loss_pct,
            consecutive_losses=self._portfolio_state.consecutive_losses,
            cur_weight=cur_weight,
        )

        # ══════════════════════════════════════
        # 五、输出动作（RiskAgent 只投否决票）
        # ══════════════════════════════════════
        if gate == RiskGate.BLOCK:
            action = Action.HOLD
            conf = 0.9
        elif gate == RiskGate.REDUCE:
            action = Action.HOLD
            conf = 0.7
        else:
            action = Action.HOLD
            conf = float(np.clip(1.0 - score / 10.0, 0.1, 1.0))

        # ══════════════════════════════════════
        # 六、特征 & 理由
        # ══════════════════════════════════════
        features = {
            "risk_level": level,
            "risk_score": round(score, 2),
            "risk_gate": gate.value,

            # 单标的风险
            "var_95": round(var95, 4),
            "cvar_95": round(cvar95, 4),
            "max_drawdown": round(max_dd, 4),
            "vol_20d": round(vol_20d, 4),
            "vol_percentile": round(vol_percentile, 4),

            # 组合层
            "net_exposure": round(net_exp, 4),
            "gross_exposure": round(gross_exp, 4),
            "current_drawdown": round(self._portfolio_state.current_drawdown, 4),
            "max_drawdown_portfolio": round(self._portfolio_state.max_drawdown, 4),
            "daily_pnl": round(self._portfolio_state.daily_pnl, 2),
            "daily_trades": self._portfolio_state.daily_trades,
            "consecutive_losses": self._portfolio_state.consecutive_losses,
            "cur_symbol_weight": round(cur_weight, 4),

            # 持仓摘要
            "positions": [
                {
                    "symbol": p.symbol,
                    "weight": round(p.weight, 4),
                    "unrealized_pnl_pct": round(p.unrealized_pnl_pct, 4),
                }
                for p in position_risks
            ],
        }

        reasons = [
            f"gate={gate.value}",
            f"risk_level={level}",
            f"score={score:.1f}",
            f"dd={self._portfolio_state.current_drawdown:.2%}",
            f"net_exp={net_exp:.2%}",
        ] + gate_reasons

        return AgentOutput(
            agent_name=self.name,
            action=action,
            confidence=round(conf, 4),
            reasons=reasons,
            features=features,
        )

    # ══════════════════════════════════════
    # 单标的风险度量（[DEL-4] 历史分位法）
    # ══════════════════════════════════════
    @staticmethod
    def _var_historical(returns: pd.Series, confidence: float) -> float:
        """历史分位法 VaR（简单、鲁棒、无依赖）"""
        if len(returns) < 20:
            return 0.02
        cutoff = np.percentile(returns, (1 - confidence) * 100)
        return float(abs(cutoff))

    @staticmethod
    def _cvar_historical(returns: pd.Series, confidence: float) -> float:
        """[DEL-5] 只保留 CVaR，不再单独算 ES"""
        if len(returns) < 20:
            return 0.03
        cutoff = np.percentile(returns, (1 - confidence) * 100)
        tail = returns[returns <= cutoff]
        if tail.empty:
            return float(abs(cutoff))
        return float(abs(tail.mean()))

    @staticmethod
    def _max_drawdown(prices: pd.Series) -> float:
        if len(prices) < 2:
            return 0.0
        cummax = prices.expanding().max()
        dd = (prices - cummax) / cummax
        return float(abs(dd.min()))

    @staticmethod
    def _vol_percentile(returns: pd.Series, window: int = 20) -> float:
        """当前 20 根滚动波动率在历史中的分位"""
        if len(returns) < window * 2:
            return 0.5
        rolling = returns.rolling(window).std().dropna()
        if rolling.empty:
            return 0.5
        cur = rolling.iloc[-1]
        return float((rolling < cur).mean())

    # ══════════════════════════════════════
    # 组合层
    # ══════════════════════════════════════
    def _analyze_positions(
        self, portfolio: Dict[str, Dict[str, Any]],
        current_price: float, current_symbol: str,
        portfolio_prices: Optional[Dict[str, float]] = None,
    ) -> List[PositionRisk]:
        total_value = max(self._portfolio_state.total_value, 1.0)
        prices = portfolio_prices or {}
        risks: List[PositionRisk] = []

        for sym, pos in portfolio.items():
            shares = float(pos.get("shares", 0))
            avg_price = float(pos.get("avg_price", 0))
            if shares == 0:
                continue

            # 优先级：外部传入 > 当前 symbol 的 price > avg_price 兜底
            if sym in prices and prices[sym] > 0:
                price = float(prices[sym])
            elif sym == current_symbol:
                price = current_price
            else:
                price = avg_price

            pos_value = shares * price
            unrealized = (price - avg_price) * shares
            unrealized_pct = (price / avg_price - 1.0) if avg_price > 0 else 0.0

            risks.append(PositionRisk(
                symbol=sym,
                shares=shares,
                avg_price=avg_price,
                current_price=price,
                position_value=pos_value,
                unrealized_pnl=unrealized,
                unrealized_pnl_pct=unrealized_pct,
                weight=pos_value / total_value,
            ))
        return risks

    # ══════════════════════════════════════
    # 风险评分（[DEL-1/4/5] 三项）
    # ══════════════════════════════════════
    def _risk_score(
        self, var: float, dd: float, net_exp: float, gross_exp: float,
    ) -> float:
        """
        三项加权风险评分 [0, 10]：
          var     : 5% VaR → 10 分
          drawdown: 20% 回撤 → 10 分
          exposure: gross 超过 max_gross 的倍数
        """
        w = self.config["risk_score_weights"]

        var_s = min(var / 0.05 * 10, 10.0)
        dd_s = min(dd / 0.20 * 10, 10.0)

        # 敞口评分：用 gross 与上限比值，超过 1 倍上限 → 10 分
        max_gross = max(self.config["max_gross_exposure"], 1e-8)
        exp_s = min(gross_exp / max_gross * 10, 10.0)

        score = (
            w["var"] * var_s
            + w["drawdown"] * dd_s
            + w["exposure"] * exp_s
        )
        return float(np.clip(score, 0.0, 10.0))

    def _risk_level(self, score: float) -> str:
        t = self.config["risk_thresholds"]
        if score < t[RiskLevel.LOW.value]:
            return RiskLevel.LOW.value
        if score < t[RiskLevel.MEDIUM.value]:
            return RiskLevel.MEDIUM.value
        if score < t[RiskLevel.HIGH.value]:
            return RiskLevel.HIGH.value
        return RiskLevel.CRITICAL.value

    # ══════════════════════════════════════
    # 三级风控闸门
    # ══════════════════════════════════════
    def _risk_gate(
        self, level: str, dd: float,
        net_exp: float, gross_exp: float, daily_loss_pct: float,
        consecutive_losses: int, cur_weight: float,
    ) -> Tuple[RiskGate, List[str]]:
        """
        [DEL-7] 三级联动，优先级 BLOCK > REDUCE > CLEAR
        """
        reasons: List[str] = []

        # ── BLOCK：禁止所有新开仓 ─────────────
        if level == RiskLevel.CRITICAL.value:
            reasons.append("risk_level=CRITICAL")
            return RiskGate.BLOCK, reasons

        if abs(daily_loss_pct) >= self.config["max_daily_loss_pct"]:
            reasons.append(f"daily_loss={daily_loss_pct:.2%} 超限")
            return RiskGate.BLOCK, reasons

        if dd >= self.config["trailing_stop_dd"]:
            reasons.append(f"drawdown={dd:.2%} 触发强制平仓")
            return RiskGate.FORCE_CLOSE, reasons

        if consecutive_losses >= self.config["max_consecutive_losses"]:
            reasons.append(f"consecutive_losses={consecutive_losses}")
            return RiskGate.BLOCK, reasons

        if abs(net_exp) >= self.config["max_net_exposure"]:
            reasons.append(f"net_exposure={net_exp:.2%} 超限")
            return RiskGate.BLOCK, reasons

        if gross_exp >= self.config["max_gross_exposure"]:
            reasons.append(f"gross_exposure={gross_exp:.2%} 超限")
            return RiskGate.BLOCK, reasons

        if cur_weight >= self.config["max_single_position"]:
            reasons.append(f"symbol_weight={cur_weight:.2%} 超单标的上限")
            return RiskGate.BLOCK, reasons

        # ── REDUCE：禁止加仓，允许平仓 ────────
        if dd >= self.config["max_drawdown_threshold"]:
            reasons.append(f"drawdown={dd:.2%} 超阈值，降仓")
            return RiskGate.REDUCE, reasons

        if level == RiskLevel.HIGH.value:
            reasons.append("risk_level=HIGH，降仓")
            return RiskGate.REDUCE, reasons

        if abs(net_exp) >= self.config["max_net_exposure"] * 0.8:
            reasons.append(f"net_exposure={net_exp:.2%} 接近上限")
            return RiskGate.REDUCE, reasons

        if cur_weight >= self.config["max_single_position"] * 0.8:
            reasons.append(f"symbol_weight={cur_weight:.2%} 接近上限")
            return RiskGate.REDUCE, reasons

        # ── CLEAR ─────────────────────────────
        return RiskGate.CLEAR, ["all clear"]

    # ══════════════════════════════════════
    # 工具
    # ══════════════════════════════════════
    def _default_output(
        self, reason: str, is_warning: bool = False
    ) -> AgentOutput:
        kwargs: Dict[str, Any] = {
            "agent_name": self.name,
            "action": Action.HOLD,
            "confidence": 0.0,
            "features": {
                "risk_level": RiskLevel.MEDIUM.value,
                "risk_score": 5.0,
                "risk_gate": RiskGate.CLEAR.value,
            },
        }
        if is_warning:
            kwargs["warnings"] = [reason]
        else:
            kwargs["reasons"] = [reason]
        return AgentOutput(**kwargs)

    def _validate_config(self) -> None:
        assert 0 < self.config["var_confidence"] < 1, "var_confidence 必须在 (0,1)"
        assert 0 < self.config["cvar_confidence"] < 1, "cvar_confidence 必须在 (0,1)"
        assert 0 < self.config["max_drawdown_threshold"] < 1, \
            "max_drawdown_threshold 必须在 (0,1)"
        assert self.config["trailing_stop_dd"] > self.config["max_drawdown_threshold"], \
            "trailing_stop_dd 必须大于 max_drawdown_threshold"
        assert 0 < self.config["max_daily_loss_pct"] < 1, \
            "max_daily_loss_pct 必须在 (0,1)"

        # 归一化评分权重
        w = self.config["risk_score_weights"]
        total = sum(w.values())
        if total <= 0:
            n = len(w)
            self.config["risk_score_weights"] = {k: 1.0 / n for k in w}
        else:
            self.config["risk_score_weights"] = {k: v / total for k, v in w.items()}