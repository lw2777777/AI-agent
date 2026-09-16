"""
MarketAgent
识别市场状态，不产生交易方向。
  - 删除 Hurst / Kyle's Lambda / MFI / pivot_points 等冗余项
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .base_agent import (
    Action, AgentOutput, BaseAgent, MarketRegime, REQUIRED_COLS
)

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────
# 配置
# ──────────────────────────────────────────
DEFAULT_CONFIG: Dict[str, Any] = {
    # 均线
    "ema_fast": 12,
    "ema_slow": 26,

    # ADX
    "adx_period": 14,
    "adx_trend_threshold": 22.0,
    "adx_strong_threshold": 35.0,

    # ATR / 波动率
    "atr_period": 14,
    "vol_high_percentile": 0.75,
    "vol_low_percentile": 0.25,
    "vol_rolling_window": 20,

    # 成交量
    "volume_avg_period": 20,
    "volume_spike_ratio": 1.5,      # 放量阈值

    # 支撑阻力
    "sr_window": 20,

    # 盘口
    "ob_depth_levels": 5,
    "ob_imbalance_threshold": 0.35,  # 买卖失衡阈值（用于突破确认）

    # 突破
    "breakout_atr_buffer": 0.2,      # 突破需超出阻力 N 倍 ATR，防假突破

    # 数据要求
    "min_data_points": 40,

    # 置信度权重（可回测拟合后覆盖）
    "conf_weights": {
        "adx": 0.35,
        "vol": 0.20,
        "ob": 0.20,
        "flow": 0.25,
    },
}


# ──────────────────────────────────────────
# 盘口数据结构
# ──────────────────────────────────────────
@dataclass
class OrderBookSnapshot:
    """盘口快照（只保留决策相关的两项派生指标）"""
    timestamp: pd.Timestamp
    bids: List[Tuple[float, float]]  # [(price, size), ...] 降序
    asks: List[Tuple[float, float]]  # [(price, size), ...] 升序

    bid_ask_spread_pct: float = 0.0   # 价差百分比
    mid_price: float = 0.0
    depth_imbalance: float = 0.0      # [-1,1]，正=买盘厚

    def compute(self, depth_levels: int = 5) -> "OrderBookSnapshot":
        if not self.bids or not self.asks:
            return self

        best_bid = self.bids[0][0]
        best_ask = self.asks[0][0]
        self.mid_price = (best_bid + best_ask) / 2

        if self.mid_price > 0:
            self.bid_ask_spread_pct = (best_ask - best_bid) / self.mid_price

        bid_vol = sum(s for _, s in self.bids[:depth_levels])
        ask_vol = sum(s for _, s in self.asks[:depth_levels])
        total = bid_vol + ask_vol
        self.depth_imbalance = (bid_vol - ask_vol) / total if total > 0 else 0.0
        return self

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ob_spread_pct": round(self.bid_ask_spread_pct, 6),
            "ob_depth_imbalance": round(self.depth_imbalance, 4),
        }


# ──────────────────────────────────────────
# 主 Agent
# ──────────────────────────────────────────
class MarketAgent(BaseAgent):
    name = "market"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        merged = {**DEFAULT_CONFIG, **(config or {})}
        # 合并嵌套的 conf_weights
        if config and "conf_weights" in config:
            merged["conf_weights"] = {
                **DEFAULT_CONFIG["conf_weights"],
                **config["conf_weights"],
            }
        super().__init__(merged)

    # ══════════════════════════════════════
    # 主入口
    # ══════════════════════════════════════
    def run(
        self,
        data: pd.DataFrame,
        symbol: str,
        context: Optional[Dict[str, Any]] = None,
        order_book: Optional[OrderBookSnapshot] = None,
    ) -> AgentOutput:
        err = self.validate_ohlcv(data, min_len=self.config["min_data_points"])
        if err:
            logger.warning("[%s][market] %s", symbol, err)
            return AgentOutput(
                agent_name=self.name, action=Action.HOLD,
                confidence=0.0, warnings=[err],
            )

        df = data.dropna(subset=list(REQUIRED_COLS)).copy()
        close = df["close"]
        last_price = float(close.iloc[-1])

        # ── 核心指标 ──────────────────────────
        ema_f = float(close.ewm(span=self.config["ema_fast"], adjust=False).mean().iloc[-1])
        ema_s = float(close.ewm(span=self.config["ema_slow"], adjust=False).mean().iloc[-1])
        adx = self._adx(df, self.config["adx_period"])

        atr_val = self.atr(df, self.config["atr_period"])
        atr_pct = atr_val / last_price if last_price > 0 else 0.0

        returns = close.pct_change().dropna()
        vol_rank = self._vol_rank(returns)

        avg_vol = float(df["volume"].tail(self.config["volume_avg_period"]).mean())
        cur_vol = float(df["volume"].iloc[-1])
        vol_ratio = cur_vol / avg_vol if avg_vol > 0 else 1.0

        support, resistance = self._support_resistance(df)
        vwap = self._vwap(df)
        active_buy_ratio = self._active_buy_ratio(df)

        # ── 盘口 ──────────────────────────────
        ob_features: Dict[str, Any] = {}
        if order_book is not None:
            order_book.compute(depth_levels=self.config["ob_depth_levels"])
            ob_features = order_book.to_dict()

        ob_imbalance = ob_features.get("ob_depth_imbalance", 0.0)

        # ── Regime 分类 ───────────────────────
        regime = self._classify_regime(
            ema_f=ema_f, ema_s=ema_s, adx=adx,
            atr_pct=atr_pct, vol_rank=vol_rank, vol_ratio=vol_ratio,
            support=support, resistance=resistance, last_price=last_price,
            ob_imbalance=ob_imbalance,
            prev_adx=self._adx_prev,      # 由 _adx 内部缓存
        )

        hour = self._trading_session(df)

        # ── 特征汇总 ──────────────────────────
        features: Dict[str, Any] = {
            "regime": regime.value,
            "adx": round(adx, 3),
            "ema_fast": round(ema_f, 8),
            "ema_slow": round(ema_s, 8),
            "atr": round(atr_val, 8),
            "atr_pct": round(atr_pct, 6),
            "vol_rank": round(vol_rank, 4),
            "volume_ratio": round(vol_ratio, 4),
            "vwap": round(vwap, 8),
            "vwap_deviation_pct": round(
                (last_price - vwap) / vwap if vwap > 0 else 0.0, 6
            ),
            "support": round(support, 8),
            "resistance": round(resistance, 8),
            "active_buy_ratio": round(active_buy_ratio, 4),
            "hour": hour,
            **ob_features,
        }

        reasons = self._build_reasons(
            regime, adx, atr_pct, vol_rank, vol_ratio,
            active_buy_ratio, ob_features,
        )

        conf = self._regime_confidence(
            adx=adx,
            vol_rank=vol_rank,
            ob_imbalance=ob_imbalance,
            active_buy_ratio=active_buy_ratio,
        )

        return AgentOutput(
            agent_name=self.name,
            action=Action.HOLD,
            confidence=conf,
            reasons=reasons,
            features=features,
        )

    # ══════════════════════════════════════
    # 核心指标
    # ══════════════════════════════════════
    def _vol_rank(self, returns: pd.Series) -> float:
        """当前波动率在历史中的百分位 [0,1]"""
        w = self.config["vol_rolling_window"]
        rolling_vol = returns.rolling(w).std().dropna()
        if len(rolling_vol) < w:
            return 0.5
        return float((rolling_vol < rolling_vol.iloc[-1]).mean())

    def _vwap(self, df: pd.DataFrame) -> float:
        total_vol = df["volume"].sum()
        if total_vol == 0:
            return float(df["close"].mean())
        typical = (df["high"] + df["low"] + df["close"]) / 3
        return float((typical * df["volume"]).sum() / total_vol)

    def _adx(self, df: pd.DataFrame, period: int) -> float:
        """只返回 ADX 值；同时缓存上一根 ADX 用于 REVERSAL 判定"""
        self._adx_prev = 0.0
        if len(df) < period * 2:
            return 0.0

        high, low, close = df["high"], df["low"], df["close"]
        prev_close = close.shift(1)

        tr = pd.concat(
            [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
        ).max(axis=1)

        up, down = high.diff(), -low.diff()
        plus_dm = np.where((up > down) & (up > 0), up, 0.0)
        minus_dm = np.where((down > up) & (down > 0), down, 0.0)

        atr_s = tr.rolling(period).mean()
        plus_di = (
            100 * pd.Series(plus_dm, index=df.index).rolling(period).mean()
            / atr_s.replace(0, np.nan)
        )
        minus_di = (
            100 * pd.Series(minus_dm, index=df.index).rolling(period).mean()
            / atr_s.replace(0, np.nan)
        )
        denom = (plus_di + minus_di).replace(0, np.nan)
        dx = 100 * (plus_di - minus_di).abs() / denom
        adx_series = dx.rolling(period).mean()

        cur = adx_series.iloc[-1]
        prev = adx_series.iloc[-2] if len(adx_series) >= 2 else cur
        self._adx_prev = float(prev) if pd.notna(prev) else 0.0
        return float(cur) if pd.notna(cur) else 0.0

    def _support_resistance(self, df: pd.DataFrame) -> Tuple[float, float]:
        w = self.config["sr_window"]
        recent = df.tail(w)
        return float(recent["low"].min()), float(recent["high"].max())

    def _active_buy_ratio(self, df: pd.DataFrame) -> float:
        """
        收盘价在高低区间中的位置，估算主动买入比例。
        ⚠️ 趋势 K 线会失真，仅作参考。真实场景应用 tick 数据。
        """
        hl = (df["high"] - df["low"]).replace(0, np.nan)
        ratio = (df["close"] - df["low"]) / hl
        val = ratio.tail(self.config["volume_avg_period"]).mean()
        return float(val) if pd.notna(val) else 0.5

    # ══════════════════════════════════════
    # Regime 分类
    # ══════════════════════════════════════
    def _classify_regime(
        self,
        ema_f: float, ema_s: float,
        adx: float, atr_pct: float, vol_rank: float, vol_ratio: float,
        support: float, resistance: float, last_price: float,
        ob_imbalance: float, prev_adx: float,
    ) -> MarketRegime:
        cfg = self.config
        th = cfg["adx_trend_threshold"]
        strong_th = cfg["adx_strong_threshold"]
        atr_abs = atr_pct * last_price
        buffer = cfg["breakout_atr_buffer"] * atr_abs

        # ── 1. 突破：真正突破关键位 + 放量/盘口确认 ──
        breakout_up = (
            last_price > resistance + buffer
            and vol_ratio >= cfg["volume_spike_ratio"]
        )
        breakout_down = (
            last_price < support - buffer
            and vol_ratio >= cfg["volume_spike_ratio"]
        )
        if breakout_up or breakout_down:
            return MarketRegime.BREAKOUT

        # ── 2. 反转：ADX 从高位回落 + 价格穿越 EMA ──
        # ADX 高说明此前是趋势，ADX 回落说明趋势动能衰竭
        adx_falling = adx < prev_adx and prev_adx >= strong_th
        price_cross = (
            (ema_f > ema_s and last_price < ema_f)   # 上升趋势中被跌破
            or (ema_f < ema_s and last_price > ema_f)  # 下降趋势中被突破
        )
        if adx_falling and price_cross:
            return MarketRegime.REVERSAL

        # ── 3. 高波动震荡：无趋势 + 波动极端 ──
        if vol_rank > cfg["vol_high_percentile"] and adx < th:
            return MarketRegime.VOLATILE

        # ── 4. 趋势 ──
        if adx >= th:
            return MarketRegime.TRENDING_UP if ema_f > ema_s else MarketRegime.TRENDING_DOWN

        # ── 5. 默认震荡 ──
        return MarketRegime.RANGING

    # ══════════════════════════════════════
    # 置信度（可配置权重）
    # ══════════════════════════════════════
    def _regime_confidence(
        self,
        adx: float,
        vol_rank: float,
        ob_imbalance: float,
        active_buy_ratio: float,
    ) -> float:
        # ADX 强度
        adx_score = float(np.clip(adx / 40.0, 0.0, 1.0))
        # 波动率明确性（远离 0.5 越明确）
        vol_score = abs(vol_rank - 0.5) * 2.0
        # 盘口失衡明确性
        ob_score = abs(ob_imbalance)
        # 资金流明确性（active_buy_ratio 远离 0.5）
        flow_score = abs(active_buy_ratio - 0.5) * 2.0

        w = self.config["conf_weights"]
        scores = {
            "adx": adx_score,
            "vol": vol_score,
            "ob": ob_score,
            "flow": flow_score,
        }
        total_w = sum(w.values()) or 1.0
        conf = sum(w[k] * scores[k] for k in scores) / total_w
        return float(np.clip(conf, 0.0, 1.0))

    # ══════════════════════════════════════
    # 工具
    # ══════════════════════════════════════
    def _trading_session(self, df: pd.DataFrame) -> Optional[int]:
        if isinstance(df.index, pd.DatetimeIndex):
            return int(df.index[-1].hour)
        return None

    def _build_reasons(
        self,
        regime: MarketRegime,
        adx: float,
        atr_pct: float,
        vol_rank: float,
        vol_ratio: float,
        active_buy_ratio: float,
        ob_features: Dict[str, Any],
    ) -> List[str]:
        reasons = [
            f"regime={regime.value}",
            f"ADX={adx:.1f}",
            f"ATR%={atr_pct:.3%}",
            f"vol_rank={vol_rank:.2f}",
            f"vol_ratio={vol_ratio:.2f}",
            f"active_buy={active_buy_ratio:.2f}",
        ]
        if ob_features:
            reasons += [
                f"ob_imbalance={ob_features.get('ob_depth_imbalance', 0):.3f}",
                f"ob_spread_pct={ob_features.get('ob_spread_pct', 0):.4%}",
            ]
        return reasons