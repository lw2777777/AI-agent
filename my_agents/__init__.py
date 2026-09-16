"""
AI Agent 多智能体系统

层级：
  MarketAgent            → 市场状态识别（不产生交易方向）
  AlphaAgent             → 信号生成层（方向 + 置信度）
  RiskAgent              → 组合层风控（闸门 + 降仓系数）
  ExecutionAgent         → 订单参数转换（止损/止盈/仓位）
  MultiAgentOrchestrator → 协调层
  ReflexionEngine        → 反思与在线学习
"""

# ── 基类与数据契约 ─────────────────────────
from .base_agent import (
    Action,
    AgentOutput,
    BaseAgent,
    MarketRegime,
    RiskLevel,
    REQUIRED_COLS,
    PRICE_COLS,
    MIN_DATA_POINTS,
)

# ── 四层 Agent ────────────────────────────
from .market_agent import (
    MarketAgent,
    OrderBookSnapshot,
)

from .alpha_agent import (
    AlphaAgent,
    SignalType,
    SignalDimension,
    SignalDetail,
    DimensionVote,
    ComputedIndicators,
    SIGNAL_TO_DIM,
    REGIME_SIGNAL_MAP,
)

from .risk_agent import (
    RiskAgent,
    RiskGate,
    PortfolioState,
    PositionRisk,
)

from .execution_agent import (
    ExecutionAgent,
    OrderType,
    StopStrategy,
    OrderParams,
    SlippageEstimate,
    TradeState,
    REGIME_TP_MULTIPLIER,
)

# ── 协调层 ────────────────────────────────
from .multi_agent_orchestrator import (
    MultiAgentOrchestrator,
    Decision,
)

# ── 反思层 ────────────────────────────────
from .reflexion_engine import (
    ReflexionEngine,
    ReflexionMemory,
    StrategyInsight,
    EmotionalState,
)


__all__ = [
    # base
    "Action",
    "AgentOutput",
    "BaseAgent",
    "MarketRegime",
    "RiskLevel",
    "REQUIRED_COLS",
    "PRICE_COLS",
    "MIN_DATA_POINTS",

    # market
    "MarketAgent",
    "OrderBookSnapshot",

    # alpha
    "AlphaAgent",
    "SignalType",
    "SignalDimension",
    "SignalDetail",
    "DimensionVote",
    "ComputedIndicators",
    "SIGNAL_TO_DIM",
    "REGIME_SIGNAL_MAP",

    # risk
    "RiskAgent",
    "RiskGate",
    "PortfolioState",
    "PositionRisk",

    # execution
    "ExecutionAgent",
    "OrderType",
    "StopStrategy",
    "OrderParams",
    "SlippageEstimate",
    "TradeState",
    "REGIME_TP_MULTIPLIER",

    # orchestrator
    "MultiAgentOrchestrator",
    "Decision",

    # reflexion
    "ReflexionEngine",
    "ReflexionMemory",
    "StrategyInsight",
    "EmotionalState",
]

__version__ = "2.0.0"