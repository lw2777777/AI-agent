"""
AlphaAgent
信号生成层:OHLCV + MarketAgent features → 方向 + 置信度 + 结构化理由

保留信号:
  TREND      : TREND_FOLLOW + MACD
  REVERT     : MEAN_REVERT + BB
  ORDER_BOOK : ORDER_BOOK

删除的信号及理由：
  - MOMENTUM  : 与 TREND_FOLLOW + MACD 严重重叠
  - OBV       : 量能信息可由 MACD 动量间接反映，归一化不稳定
  - RSI       : 与 BB 同为超买超卖,BB 自带波动率上下文
  - VWAP      : 与 MEAN_REVERT 的支撑阻力高度重叠

 bug 修复：
  [FIX-01] _sig_trend_follow 冗余 and 条件
  [FIX-02] 归一化分母问题 → 饱和归一化
  [FIX-03] _df_hash 缓存指纹碰撞
  [FIX-04] MACD 强度归一化
  [FIX-07] 冲突惩罚死代码
  [FIX-10] 指标缓存 LRU
  [FIX-11] 缺列提前返回
"""
from __future__ import annotations

import hashlib
import logging
from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .base_agent import Action, AgentOutput, BaseAgent, MarketRegime, REQUIRED_COLS

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════
# 常量 & 枚举
# ══════════════════════════════════════════════════════════
class SignalType(str, Enum):
    TREND_FOLLOW = "trend_follow"
    MACD = "macd"
    MEAN_REVERT = "mean_revert"
    BB = "bb"
    ORDER_BOOK = "order_book"


class SignalDimension(str, Enum):
    TREND = "trend"
    REVERT = "revert"
    ORDER_BOOK = "order_book"


SIGNAL_TO_DIM: Dict[SignalType, SignalDimension] = {
    SignalType.TREND_FOLLOW: SignalDimension.TREND,
    SignalType.MACD: SignalDimension.TREND,
    SignalType.MEAN_REVERT: SignalDimension.REVERT,
    SignalType.BB: SignalDimension.REVERT,
    SignalType.ORDER_BOOK: SignalDimension.ORDER_BOOK,
}


# 各 Regime 下允许激活的信号类型（已删掉冗余信号）
REGIME_SIGNAL_MAP: Dict[str, List[SignalType]] = {
    MarketRegime.TRENDING_UP.value: [
        SignalType.TREND_FOLLOW, SignalType.MACD, SignalType.ORDER_BOOK,
    ],
    MarketRegime.TRENDING_DOWN.value: [
        SignalType.TREND_FOLLOW, SignalType.MACD, SignalType.ORDER_BOOK,
    ],
    MarketRegime.RANGING.value: [
        SignalType.MEAN_REVERT, SignalType.BB, SignalType.ORDER_BOOK,
    ],
    MarketRegime.VOLATILE.value: [
        SignalType.BB, SignalType.ORDER_BOOK,
    ],
    MarketRegime.BREAKOUT.value: [
        SignalType.TREND_FOLLOW, SignalType.MACD, SignalType.ORDER_BOOK,
    ],
    MarketRegime.REVERSAL.value: [
        SignalType.MEAN_REVERT, SignalType.BB,
    ],
    "unknown": list(SignalType),
}


# ══════════════════════════════════════════════════════════
# 数据结构
# ══════════════════════════════════════════════════════════
@dataclass
class SignalDetail:
    signal_type: SignalType
    direction: int
    raw_score: float
    normalized_score: float
    weight: float
    weighted_score: float
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.signal_type.value,
            "direction": self.direction,
            "raw_score": round(self.raw_score, 4),
            "normalized_score": round(self.normalized_score, 4),
            "weight": self.weight,
            "weighted_score": round(self.weighted_score, 4),
            "reason": self.reason,
        }


@dataclass
class DimensionVote:
    dimension: SignalDimension
    direction: int
    score: float
    members: List[SignalDetail]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dimension": self.dimension.value,
            "direction": self.direction,
            "score": round(self.score, 4),
            "members": [m.to_dict() for m in self.members],
        }


@dataclass
class ComputedIndicators:
    bb_upper: float
    bb_mid: float
    bb_lower: float
    macd_line: float
    macd_signal: float
    macd_hist: float
    macd_hist_norm: float
    vwap: float
    atr: float


# ══════════════════════════════════════════════════════════
# 配置
# ══════════════════════════════════════════════════════════
DEFAULT_CONFIG: Dict[str, Any] = {
    # 布林带
    "bb_period": 20,
    "bb_std": 2.0,
    "bb_squeeze_threshold": 0.02,

    # MACD
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,
    "macd_hist_norm_window": 20,

    # 均值回归
    "mr_zone_pct": 0.2,

    # 盘口
    "ob_strong_imbalance": 0.5,

    # 置信度
    "min_confidence": 0.45,
    "conflict_penalty": 0.15,
    "max_confidence": 0.95,
    "conf_scale": 2.0,

    # 维度权重
    "dimension_weights": {
        SignalDimension.TREND.value: 1.6,
        SignalDimension.REVERT.value: 1.3,
        SignalDimension.ORDER_BOOK.value: 1.4,
    },
    # 维度内成员权重
    "signal_weights": {
        SignalType.TREND_FOLLOW.value: 1.2,
        SignalType.MACD.value: 1.0,
        SignalType.MEAN_REVERT.value: 1.0,
        SignalType.BB.value: 0.9,
        SignalType.ORDER_BOOK.value: 1.0,
    },

    # 数据
    "min_data_points": 60,
    "atr_period": 14,
    "indicator_cache_size": 8,
}


# ══════════════════════════════════════════════════════════
# AlphaAgent
# ══════════════════════════════════════════════════════════
class AlphaAgent(BaseAgent):
    name = "alpha"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        merged = {**DEFAULT_CONFIG, **(config or {})}

        for nested in ("signal_weights", "dimension_weights"):
            if config and nested in config:
                merged[nested] = {**DEFAULT_CONFIG[nested], **config[nested]}

        super().__init__(merged)
        self._validate_config()

        self._indicator_cache: "OrderedDict[str, ComputedIndicators]" = OrderedDict()

    # ══════════════════════════════════════
    # 主入口
    # ══════════════════════════════════════
    def run(
        self,
        data: pd.DataFrame,
        symbol: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> AgentOutput:

        missing = [c for c in REQUIRED_COLS if c not in data.columns]
        if missing:
            return AgentOutput(
                agent_name=self.name, action=Action.HOLD,
                confidence=0.0,
                warnings=[f"missing columns: {missing}"],
            )

        err = self.validate_ohlcv(data, min_len=self.config["min_data_points"])
        if err:
            return AgentOutput(
                agent_name=self.name, action=Action.HOLD,
                confidence=0.0, warnings=[err],
            )

        df = data.dropna(subset=list(REQUIRED_COLS)).copy()
        close = df["close"]
        price = float(close.iloc[-1])

        # ── MarketAgent 特征 ──────────────────
        ctx = context or {}
        market_feat = ctx.get("market_features", {}) or {}
        regime = market_feat.get("regime", "unknown")
        ob_imbalance = float(market_feat.get("ob_depth_imbalance", 0.0))
        has_order_book = "ob_depth_imbalance" in market_feat and ob_imbalance != 0.0
        ob_large_bid_wall = False
        ob_large_ask_wall = False
        support = float(market_feat.get("support", df["low"].tail(20).min()))
        resistance = float(market_feat.get("resistance", df["high"].tail(20).max()))

        ind = self._get_indicators(df)
        allowed = self._allowed_signals(regime, has_order_book)

        # ══════════════════════════════════════
        # 信号收集
        # ══════════════════════════════════════
        signals: List[SignalDetail] = []

        if SignalType.TREND_FOLLOW in allowed:
            sig = self._sig_trend_follow(regime)
            if sig:
                signals.append(sig)

        if SignalType.MACD in allowed:
            sig = self._sig_macd(ind.macd_line, ind.macd_signal, ind.macd_hist_norm)
            if sig:
                signals.append(sig)

        if SignalType.MEAN_REVERT in allowed:
            sig = self._sig_mean_revert(price, support, resistance)
            if sig:
                signals.append(sig)

        if SignalType.BB in allowed:
            sig = self._sig_bb(price, ind.bb_upper, ind.bb_lower, ind.bb_mid)
            if sig:
                signals.append(sig)

        if SignalType.ORDER_BOOK in allowed and ob_imbalance != 0.0:
            sig = self._sig_order_book(
                ob_imbalance, ob_large_bid_wall, ob_large_ask_wall
            )
            if sig:
                signals.append(sig)

        # ══════════════════════════════════════
        # 汇总
        # ══════════════════════════════════════
        if not signals:
            return AgentOutput(
                agent_name=self.name, action=Action.HOLD,
                confidence=0.0, reasons=["no active signal"],
                features=self._build_features(ind, regime),
            )

        action, conf, reasons, dim_votes = self._aggregate_signals(signals)

        if conf < self.config["min_confidence"]:
            return AgentOutput(
                agent_name=self.name, action=Action.HOLD,
                confidence=round(conf, 4),
                reasons=reasons + ["below min_confidence"],
                features={
                    **self._build_features(ind, regime),
                    "signal_details": [s.to_dict() for s in signals],
                    "dimension_votes": [d.to_dict() for d in dim_votes],
                },
            )

        return AgentOutput(
            agent_name=self.name,
            action=action,
            confidence=round(
                float(np.clip(conf, 0.0, self.config["max_confidence"])), 4
            ),
            reasons=reasons,
            features={
                **self._build_features(ind, regime),
                "signal_details": [s.to_dict() for s in signals],
                "dimension_votes": [d.to_dict() for d in dim_votes],
                "active_signal_count": len(signals),
            },
        )

    # ══════════════════════════════════════
    # 信号生成
    # ══════════════════════════════════════
    def _sig_trend_follow(self, regime: str) -> Optional[SignalDetail]:
        if regime == MarketRegime.TRENDING_UP.value:
            direction, reason = 1, "regime=trending_up → 趋势做多"
        elif regime == MarketRegime.TRENDING_DOWN.value:
            direction, reason = -1, "regime=trending_down → 趋势做空"
        elif regime == MarketRegime.BREAKOUT.value:
            direction, reason = 1, "regime=breakout → 突破做多"
        else:
            return None
        return self._make_signal(SignalType.TREND_FOLLOW, direction, 1.0, 1.0, reason)

    def _sig_macd(
        self, macd_line: float, signal_line: float, hist_norm: float
    ) -> Optional[SignalDetail]:
        if macd_line > signal_line:
            strength = float(np.clip(abs(hist_norm), 0.0, 1.0))
            return self._make_signal(SignalType.MACD, 1, abs(hist_norm), strength,
                                     f"MACD golden_cross hist_norm={hist_norm:.3f}")
        elif macd_line < signal_line:
            strength = float(np.clip(abs(hist_norm), 0.0, 1.0))
            return self._make_signal(SignalType.MACD, -1, abs(hist_norm), strength,
                                     f"MACD death_cross hist_norm={hist_norm:.3f}")
        return None

    def _sig_mean_revert(
        self, price: float, support: float, resistance: float
    ) -> Optional[SignalDetail]:
        span = resistance - support
        if span <= 0:
            return None
        zone = self.config["mr_zone_pct"]
        pos = (price - support) / span

        if pos < zone:
            strength = float(np.clip(1 - pos / zone, 0.0, 1.0))
            return self._make_signal(
                SignalType.MEAN_REVERT, 1, strength, strength,
                f"near support pos={pos:.2f} strength={strength:.2f}",
            )
        elif pos > (1 - zone):
            strength = float(np.clip((pos - (1 - zone)) / zone, 0.0, 1.0))
            return self._make_signal(
                SignalType.MEAN_REVERT, -1, strength, strength,
                f"near resistance pos={pos:.2f} strength={strength:.2f}",
            )
        return None

    def _sig_bb(
        self, price: float, bb_u: float, bb_l: float, bb_m: float
    ) -> Optional[SignalDetail]:
        if bb_m > 0:
            width_pct = (bb_u - bb_l) / bb_m
            if width_pct < self.config["bb_squeeze_threshold"]:
                return None

        if price < bb_l:
            strength = float(np.clip((bb_l - price) / max(bb_m - bb_l, 1e-8), 0.0, 1.0))
            return self._make_signal(SignalType.BB, 1, bb_l - price, strength,
                                     f"below BB lower price={price:.4f} bb_l={bb_l:.4f}")
        elif price > bb_u:
            strength = float(np.clip((price - bb_u) / max(bb_u - bb_m, 1e-8), 0.0, 1.0))
            return self._make_signal(SignalType.BB, -1, price - bb_u, strength,
                                     f"above BB upper price={price:.4f} bb_u={bb_u:.4f}")
        return None

    def _sig_order_book(
        self, imbalance: float, large_bid: bool, large_ask: bool
    ) -> Optional[SignalDetail]:
        th = self.config["ob_strong_imbalance"]
        if abs(imbalance) < th * 0.5:
            return None

        wall_discount = 0.7 if (large_bid or large_ask) else 1.0

        if imbalance > th:
            strength = float(np.clip(imbalance / 1.0, 0.0, 1.0)) * wall_discount
            note = "large_bid_wall detected" if large_bid else ""
            return self._make_signal(SignalType.ORDER_BOOK, 1, imbalance, strength,
                                     f"bid pressure={imbalance:.3f} {note}".strip())
        elif imbalance < -th:
            strength = float(np.clip(abs(imbalance) / 1.0, 0.0, 1.0)) * wall_discount
            note = "large_ask_wall detected" if large_ask else ""
            return self._make_signal(SignalType.ORDER_BOOK, -1, abs(imbalance), strength,
                                     f"ask pressure={imbalance:.3f} {note}".strip())
        return None

    # ══════════════════════════════════════
    # 汇总：维度内融合 → 维度间投票
    # ══════════════════════════════════════
    def _aggregate_signals(
        self, signals: List[SignalDetail]
    ) -> Tuple[Action, float, List[str], List[DimensionVote]]:
        # ── 1. 维度内融合 ─────────────────────
        dim_buckets: Dict[SignalDimension, List[SignalDetail]] = {}
        for s in signals:
            dim_buckets.setdefault(SIGNAL_TO_DIM[s.signal_type], []).append(s)

        dim_votes: List[DimensionVote] = []
        for dim, members in dim_buckets.items():
            buy = sum(m.weighted_score for m in members if m.direction == 1)
            sell = sum(m.weighted_score for m in members if m.direction == -1)
            net = buy - sell
            if net == 0:
                continue
            direction = 1 if net > 0 else -1
            denom = sum(m.weight for m in members) or 1.0
            score = float(np.clip(abs(net) / denom, 0.0, 1.0))
            dim_votes.append(DimensionVote(
                dimension=dim, direction=direction,
                score=score, members=members,
            ))

        if not dim_votes:
            return Action.HOLD, 0.0, [s.reason for s in signals], []

        # ── 2. 维度间投票 ─────────────────────
        dw = self.config["dimension_weights"]
        buy_score = sum(
            dw.get(v.dimension.value, 1.0) * v.score
            for v in dim_votes if v.direction == 1
        )
        sell_score = sum(
            dw.get(v.dimension.value, 1.0) * v.score
            for v in dim_votes if v.direction == -1
        )

        reasons = [s.reason for s in signals]

        # ── 3. 冲突惩罚 ──────────────────────
        total = buy_score + sell_score
        conflict_ratio = min(buy_score, sell_score) / total if total > 0 else 0.0
        penalty = conflict_ratio * self.config["conflict_penalty"]

        # ── 4. 饱和归一化 ─────────────────────
        scale = self.config["conf_scale"]
        if buy_score > sell_score:
            raw = buy_score - sell_score
            conf = float(1.0 - np.exp(-raw / scale)) - penalty
            return Action.BUY, float(np.clip(conf, 0.0, 1.0)), reasons, dim_votes
        elif sell_score > buy_score:
            raw = sell_score - buy_score
            conf = float(1.0 - np.exp(-raw / scale)) - penalty
            return Action.SELL, float(np.clip(conf, 0.0, 1.0)), reasons, dim_votes

        return Action.HOLD, 0.0, reasons + ["buy=sell conflict"], dim_votes

    # ══════════════════════════════════════
    # 指标计算（LRU 缓存）
    # ══════════════════════════════════════
    def _get_indicators(self, df: pd.DataFrame) -> ComputedIndicators:
        key = self._df_hash(df)
        if key in self._indicator_cache:
            self._indicator_cache.move_to_end(key)
            return self._indicator_cache[key]

        close = df["close"]
        bb_u, bb_m, bb_l = self._bb(close)
        macd_l, macd_sig, macd_hist, macd_hist_norm = self._macd(close)
        vwap = self._vwap(df)
        atr = self.atr(df, self.config["atr_period"])

        ind = ComputedIndicators(
            bb_upper=bb_u, bb_mid=bb_m, bb_lower=bb_l,
            macd_line=macd_l, macd_signal=macd_sig,
            macd_hist=macd_hist, macd_hist_norm=macd_hist_norm,
            vwap=vwap, atr=atr,
        )

        max_size = self.config["indicator_cache_size"]
        self._indicator_cache[key] = ind
        while len(self._indicator_cache) > max_size:
            self._indicator_cache.popitem(last=False)

        return ind

    # ══════════════════════════════════════
    # 技术指标
    # ══════════════════════════════════════
    @staticmethod
    def _vwap(df: pd.DataFrame) -> float:
        tv = df["volume"].sum()
        if tv == 0:
            return float(df["close"].mean())
        tp = (df["high"] + df["low"] + df["close"]) / 3
        return float((tp * df["volume"]).sum() / tv)

    def _bb(self, prices: pd.Series) -> Tuple[float, float, float]:
        p, s = self.config["bb_period"], self.config["bb_std"]
        if len(prices) < p:
            v = float(prices.iloc[-1])
            return v, v, v
        m = prices.rolling(p).mean()
        sd = prices.rolling(p).std(ddof=0)
        return (
            float((m + s * sd).iloc[-1]),
            float(m.iloc[-1]),
            float((m - s * sd).iloc[-1]),
        )

    def _macd(self, prices: pd.Series) -> Tuple[float, float, float, float]:
        fast = self.config["macd_fast"]
        slow = self.config["macd_slow"]
        sig = self.config["macd_signal"]
        nw = self.config["macd_hist_norm_window"]

        if len(prices) < slow + sig:
            return 0.0, 0.0, 0.0, 0.0

        ema_f = prices.ewm(span=fast, adjust=False).mean()
        ema_s = prices.ewm(span=slow, adjust=False).mean()
        macd_line = ema_f - ema_s
        macd_sig = macd_line.ewm(span=sig, adjust=False).mean()
        hist = macd_line - macd_sig

        hist_std = float(hist.tail(nw).std(ddof=0)) or 1e-8
        hist_norm = float(np.clip(float(hist.iloc[-1]) / (2.0 * hist_std), -1.0, 1.0))

        return (
            float(macd_line.iloc[-1]),
            float(macd_sig.iloc[-1]),
            float(hist.iloc[-1]),
            hist_norm,
        )

    # ══════════════════════════════════════
    # 工具
    # ══════════════════════════════════════
    def _allowed_signals(self, regime: str, hurst: float, has_order_book: bool = True) -> List[SignalType]:
        base = list(REGIME_SIGNAL_MAP.get(regime, list(SignalType)))

        # 无盘口数据时移除 ORDER_BOOK
        if not has_order_book:
            base = [s for s in base if s != SignalType.ORDER_BOOK]
            
        return base

    def _make_signal(
        self,
        sig_type: SignalType,
        direction: int,
        raw_score: float,
        normalized_score: float,
        reason: str,
    ) -> SignalDetail:
        weight = self.config["signal_weights"].get(sig_type.value, 1.0)
        norm = float(np.clip(normalized_score, 0.0, 1.0))
        return SignalDetail(
            signal_type=sig_type,
            direction=direction,
            raw_score=raw_score,
            normalized_score=norm,
            weight=weight,
            weighted_score=norm * weight,
            reason=reason,
        )

    def _build_features(self, ind: ComputedIndicators, regime: str) -> Dict[str, Any]:
        return {
            "bb_upper": round(ind.bb_upper, 8),
            "bb_mid": round(ind.bb_mid, 8),
            "bb_lower": round(ind.bb_lower, 8),
            "macd_line": round(ind.macd_line, 8),
            "macd_signal": round(ind.macd_signal, 8),
            "macd_hist": round(ind.macd_hist, 8),
            "macd_hist_norm": round(ind.macd_hist_norm, 4),
            "vwap": round(ind.vwap, 8),
            "atr": round(ind.atr, 8),
            "regime": regime,
        }

    @staticmethod
    def _df_hash(df: pd.DataFrame) -> str:
        tail = df.tail(5)[list(REQUIRED_COLS)]
        try:
            payload = tail.values.tobytes()
        except Exception:
            payload = repr(tail.values).encode()
        key = f"{len(df)}_{hashlib.md5(payload).hexdigest()}"
        return hashlib.md5(key.encode()).hexdigest()

    def _validate_config(self) -> None:
        required_keys = [
            "bb_period", "min_confidence",
            "signal_weights", "dimension_weights",
        ]
        for k in required_keys:
            if k not in self.config:
                raise ValueError(f"AlphaAgent config missing required key: {k!r}")
        if not 0 < self.config["min_confidence"] < 1:
            raise ValueError("min_confidence 必须在 (0, 1) 之间")

        missing = [
            st.value for st in SignalType
            if st.value not in self.config["signal_weights"]
        ]
        if missing:
            raise ValueError(f"signal_weights 缺少: {missing}")

        missing_dim = [
            d.value for d in SignalDimension
            if d.value not in self.config["dimension_weights"]
        ]
        if missing_dim:
            raise ValueError(f"dimension_weights 缺少: {missing_dim}")