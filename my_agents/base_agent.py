"""
统一 Agent 基类与数据契约
所有 Agent 必须继承 BaseAgent 并实现 run()

优化点：
  - 完整类型标注
  - 数据验证增强（自动修复NaN）
  - 新增5个常用技术指标（RSI/MACD/BB/EMA/SMA）
  - ATR计算优化（缓存TR）
  - AgentOutput增加序列化方法
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════
# 常量
# ══════════════════════════════════════════════════════════
REQUIRED_COLS: Tuple[str, ...] = ("open", "high", "low", "close", "volume")
PRICE_COLS: Tuple[str, ...] = ("open", "high", "low", "close")
MIN_DATA_POINTS: int = 30

# ══════════════════════════════════════════════════════════
# 枚举
# ══════════════════════════════════════════════════════════
class Action(str, Enum):
    """交易动作"""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"

class MarketRegime(str, Enum):
    """市场状态"""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    VOLATILE = "volatile"
    BREAKOUT = "breakout"          # 新增：突破
    REVERSAL = "reversal"          # 新增：反转
    UNKNOWN = "unknown"

class RiskLevel(str, Enum):
    """风险等级"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

# ══════════════════════════════════════════════════════════
# 数据契约
# ══════════════════════════════════════════════════════════
@dataclass
class AgentOutput:
    """所有 Agent 的统一输出契约"""
    agent_name: str
    action: Action
    confidence: float                                    # [0, 1]
    reasons: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    features: Dict[str, Any] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """数据验证"""
        # 确保 action 是 Action 枚举
        if not isinstance(self.action, Action):
            try:
                self.action = Action(self.action)
            except ValueError as e:
                raise ValueError(f"Invalid action: {self.action}") from e

        # 限制 confidence 在 [0, 1]
        if not 0.0 <= self.confidence <= 1.0:
            logger.warning(
                "%s: confidence %.4f out of [0,1], clipping",
                self.agent_name, self.confidence
            )
            self.confidence = float(np.clip(self.confidence, 0.0, 1.0))

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        d = asdict(self)
        d["action"] = self.action.value
        return d

    def add_warning(self, warning: str) -> None:
        """添加警告（去重）"""
        if warning not in self.warnings:
            self.warnings.append(warning)

    def add_reason(self, reason: str) -> None:
        """添加理由（去重）"""
        if reason not in self.reasons:
            self.reasons.append(reason)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> AgentOutput:
        """从字典构造"""
        data = data.copy()
        if "action" in data and isinstance(data["action"], str):
            data["action"] = Action(data["action"])
        return cls(**data)

# ══════════════════════════════════════════════════════════
# BaseAgent
# ══════════════════════════════════════════════════════════
class BaseAgent(ABC):
    """
    Agent 基类

    子类必须实现：
      - run(data, symbol, context) -> AgentOutput
    """
    name: str = "BaseAgent"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config: Dict[str, Any] = config or {}

    @abstractmethod
    def run(
        self,
        data: pd.DataFrame,
        symbol: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> AgentOutput:
        """
        Agent 主入口

        Args:
            data: OHLCV DataFrame (必须包含: open, high, low, close, volume)
            symbol: 交易标的符号
            context: 上下文字典（其他Agent输出、组合状态等）

        Returns:
            AgentOutput
        """
        ...

    # ══════════════════════════════════════
    # 数据验证
    # ══════════════════════════════════════
    @staticmethod
    def validate_ohlcv(
        data: pd.DataFrame,
        min_len: int = MIN_DATA_POINTS,
        auto_fix: bool = False,
    ) -> Optional[str]:
        """
        OHLCV 数据验证

        Args:
            data: 输入数据
            min_len: 最小长度要求
            auto_fix: 是否自动修复异常（填充NaN、剔除负值）

        Returns:
            错误信息（None表示通过）
        """
        # 第一层：空值检查
        if data is None or data.empty:
            return "data is empty"

        # 第二层：列检查
        missing = [c for c in REQUIRED_COLS if c not in data.columns]
        if missing:
            return f"missing columns: {missing}"

        # 第三层：数据类型检查
        for col in REQUIRED_COLS:
            if not pd.api.types.is_numeric_dtype(data[col]):
                if auto_fix:
                    try:
                        data[col] = pd.to_numeric(data[col], errors="coerce")
                        logger.warning("Auto-converted %s to numeric", col)
                    except Exception as e:
                        return f"failed to convert {col} to numeric: {e}"
                else:
                    return f"{col} is not numeric"

        # 第四层：价格有效性（非正价格）
        price_invalid = (data[list(PRICE_COLS)] <= 0).any()
        if price_invalid.any():
            invalid_cols = [c for c in PRICE_COLS if (data[c] <= 0).any()]
            if auto_fix:
                for col in invalid_cols:
                    # 用前值填充非正价格
                    data.loc[data[col] <= 0, col] = np.nan
                    data[col] = data[col].ffill()
                    if data[col].isna().any():
                        return f"cannot fix non-positive {col} (no valid prior)"
                logger.warning("Auto-fixed non-positive prices in %s", invalid_cols)
            else:
                return f"non-positive price detected in {invalid_cols}"

        # 第五层：成交量有效性（负值）
        if (data["volume"] < 0).any():
            if auto_fix:
                data.loc[data["volume"] < 0, "volume"] = 0
                logger.warning("Auto-fixed negative volume")
            else:
                return "negative volume detected"

        # 第六层：NaN检查
        if data[list(REQUIRED_COLS)].isna().any().any():
            if auto_fix:
                data[list(REQUIRED_COLS)] = data[list(REQUIRED_COLS)].ffill().bfill()
                if data[list(REQUIRED_COLS)].isna().any().any():
                    return "data contains NaN that cannot be auto-fixed"
                logger.warning("Auto-filled NaN values")
            else:
                return "data contains NaN"

        # 第七层：长度检查
        if len(data) < min_len:
            return f"insufficient data ({len(data)} < {min_len})"

        return None

    # ══════════════════════════════════════
    # 技术指标工具函数
    # ══════════════════════════════════════
    @staticmethod
    def atr(data: pd.DataFrame, period: int = 14) -> float:
        """
        Average True Range(平均真实波幅)

        Args:
            data: OHLCV DataFrame
            period: 周期

        Returns:
            ATR 值
        """
        if len(data) < period + 1:
            # 不足数据时用收盘价标准差估算
            return float(data["close"].std()) if len(data) > 1 else 0.0

        high = data["high"].values
        low = data["low"].values
        close = data["close"].values

        # 计算 True Range（向量化）
        prev_close = np.roll(close, 1)
        prev_close[0] = close[0]  # 第一根K线无前值

        tr = np.maximum(
            high - low,
            np.maximum(
                np.abs(high - prev_close),
                np.abs(low - prev_close)
            )
        )

        # ATR = TR 的移动平均
        tr_series = pd.Series(tr)
        atr_val = tr_series.rolling(period).mean().iloc[-1]
        return float(atr_val) if pd.notna(atr_val) else 0.0

    @staticmethod
    def rsi(prices: pd.Series, period: int = 14) -> float:
        """
        Relative Strength Index（相对强弱指标）

        Args:
            prices: 价格序列（通常是 close）
            period: 周期

        Returns:
            RSI 值 [0, 100]
        """
        if len(prices) < period + 1:
            return 50.0  # 默认中性

        delta = prices.diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()

        avg_gain = float(gain.iloc[-1])
        avg_loss = float(loss.iloc[-1])

        if avg_gain == 0 and avg_loss == 0:
            return 50.0
        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        rsi_val = 100 - 100 / (1 + rs)
        return float(rsi_val)

    @staticmethod
    def macd(
        prices: pd.Series,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ) -> Tuple[float, float, float]:
        """
        MACD（指数平滑异同移动平均线）

        Args:
            prices: 价格序列
            fast: 快线周期
            slow: 慢线周期
            signal: 信号线周期

        Returns:
            (macd_line, signal_line, histogram)
        """
        if len(prices) < slow + signal:
            return 0.0, 0.0, 0.0

        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line

        return (
            float(macd_line.iloc[-1]),
            float(signal_line.iloc[-1]),
            float(histogram.iloc[-1]),
        )

    @staticmethod
    def bollinger_bands(
        prices: pd.Series,
        period: int = 20,
        std_dev: float = 2.0,
    ) -> Tuple[float, float, float]:
        """
        Bollinger Bands（布林带）

        Args:
            prices: 价格序列
            period: 周期
            std_dev: 标准差倍数

        Returns:
            (upper, middle, lower)
        """
        if len(prices) < period:
            mid = float(prices.iloc[-1])
            return mid, mid, mid

        sma = prices.rolling(period).mean()
        std = prices.rolling(period).std(ddof=0)

        upper = sma + std_dev * std
        lower = sma - std_dev * std

        return (
            float(upper.iloc[-1]),
            float(sma.iloc[-1]),
            float(lower.iloc[-1]),
        )

    @staticmethod
    def ema(prices: pd.Series, period: int) -> float:
        """
        Exponential Moving Average（指数移动平均）

        Args:
            prices: 价格序列
            period: 周期

        Returns:
            EMA 值
        """
        if len(prices) < period:
            return float(prices.mean())

        ema_val = prices.ewm(span=period, adjust=False).mean().iloc[-1]
        return float(ema_val) if pd.notna(ema_val) else float(prices.iloc[-1])

    @staticmethod
    def sma(prices: pd.Series, period: int) -> float:
        """
        Simple Moving Average（简单移动平均）

        Args:
            prices: 价格序列
            period: 周期

        Returns:
            SMA 值
        """
        if len(prices) < period:
            return float(prices.mean())

        sma_val = prices.rolling(period).mean().iloc[-1]
        return float(sma_val) if pd.notna(sma_val) else float(prices.iloc[-1])

    # ══════════════════════════════════════
    # 统计工具
    # ══════════════════════════════════════
    @staticmethod
    def max_drawdown(prices: pd.Series) -> float:
        """
        最大回撤

        Args:
            prices: 价格序列

        Returns:
            最大回撤百分比
        """
        if len(prices) < 2:
            return 0.0

        cummax = prices.expanding().max()
        drawdown = (prices - cummax) / cummax
        return float(abs(drawdown.min()))

    @staticmethod
    def volatility(returns: pd.Series, window: int = 20) -> float:
        """
        滚动波动率（年化）

        Args:
            returns: 收益率序列
            window: 滚动窗口

        Returns:
            年化波动率
        """
        if len(returns) < window:
            return float(returns.std() * np.sqrt(252)) if len(returns) > 1 else 0.0

        vol = returns.rolling(window).std().iloc[-1]
        return float(vol * np.sqrt(252)) if pd.notna(vol) else 0.0

    # ══════════════════════════════════════
    # 配置管理
    # ══════════════════════════════════════
    def get_config(self, key: str, default: Any = None) -> Any:
        """安全获取配置"""
        return self.config.get(key, default)

    def update_config(self, updates: Dict[str, Any]) -> None:
        """更新配置"""
        self.config.update(updates)
