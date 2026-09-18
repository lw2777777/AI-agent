"""
MultiAgentOrchestrator: 协调 alpha / market / execution / risk 四个 Agent

核心逻辑：
  1. MarketAgent 先跑 → 输出市场状态特征
  2. AlphaAgent、RiskAgent 并行跑
  3. 投票：
     - Alpha 投方向票(BUY/SELL/HOLD)
     - Market 提供 regime 调整系数（不投票）
     - Risk 提供风险等级调整系数（不投票）
  4. ExecutionAgent 做参数转换（止损/止盈/仓位）
  5. 在线学习：根据实际收益调整 AlphaAgent 权重

修复点：
  - 投票逻辑简化：只 AlphaAgent 投票
  - 置信度计算修正：避免过度压制
  - 权重学习修正：只对方向票计分
  - RiskAgent 集成修正：传入组合状态
  - 已禁用做空: SELL 仅用于平多，不再开空
"""
from __future__ import annotations

import logging
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .alpha_agent import AlphaAgent
from .base_agent import Action, AgentOutput, MarketRegime, RiskLevel
from .execution_agent import ExecutionAgent
from .market_agent import MarketAgent
from .reflexion_engine import ReflexionEngine
from .risk_agent import RiskAgent

logger = logging.getLogger(__name__)

ACTION_BUY = Action.BUY.value
ACTION_SELL = Action.SELL.value
ACTION_HOLD = Action.HOLD.value

@dataclass
class Decision:
    """最终决策输出"""
    symbol: str
    timestamp: pd.Timestamp
    action: str                                      # buy/sell/hold
    confidence: float                                # [0, 1]
    entry_price: float
    stop_loss: float
    take_profit: float
    position_size: float
    risk_level: str
    force_close: bool = False
    reasons: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    agent_votes: Dict[str, str] = field(default_factory=dict)
    agent_confidences: Dict[str, float] = field(default_factory=dict)
    market_features: Dict[str, Any] = field(default_factory=dict)
    decision_id: str = field(default_factory=lambda: uuid.uuid4().hex)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["timestamp"] = self.timestamp.isoformat()
        return d

@dataclass
class _PerfRecord:
    """性能记录（用于在线学习）"""
    decision_id: str
    alpha_vote: str
    alpha_confidence: float
    actual_return: Optional[float] = None

class MultiAgentOrchestrator:
    name = "MultiAgentOrchestrator"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        cfg = config or {}
        self.config = cfg

        # Agent 初始化
        self.market_agent = MarketAgent(cfg.get("market_config"))
        self.alpha_agent = AlphaAgent(cfg.get("alpha_config"))
        self.execution_agent = ExecutionAgent(cfg.get("execution_config"))
        self.risk_agent = RiskAgent(cfg.get("risk_config"))
        self.reflexion = ReflexionEngine(cfg.get("reflexion_config"))

        # 参数
        self.voting_threshold = float(cfg.get("voting_threshold", 0.3))
        self.agent_timeout = float(cfg.get("agent_timeout", 5.0))
        self.max_history = int(cfg.get("max_history", 1000))

        # 学习参数
        self.learning_rate = float(cfg.get("learning_rate", 0.1))
        self.min_samples_to_learn = int(cfg.get("min_samples_to_learn", 20))

        self.memory = MemoryManager()
        self._current_episode_id: Optional[str] = None
        # 状态
        self._alpha_weight = 1.0  # AlphaAgent 的可信度权重（动态调整）
        self._weight_lock = threading.Lock()

        self._history: List[_PerfRecord] = []
        self._id_index: Dict[str, _PerfRecord] = {}
        self._history_lock = threading.Lock()

        logger.info(
            "%s initialized | voting_threshold=%.2f | learning_rate=%.3f",
            self.name, self.voting_threshold, self.learning_rate
        )

    # ══════════════════════════════════════
    # 公开接口
    # ══════════════════════════════════════
    def analyze(
        self,
        data: pd.DataFrame,
        symbol: str = "UNKNOWN",
        portfolio: Optional[Dict[str, Dict[str, Any]]] = None,
        portfolio_state: Optional[Dict[str, Any]] = None,
    ) -> Decision:
        """
        多 Agent 协同分析

        Args:
            data: OHLCV DataFrame
            symbol: 交易标的
            portfolio: 持仓字典 {symbol: {shares, avg_price, ...}}
            portfolio_state: 组合状态 {total_value, cash, margin_used, ...}

        Returns:
            Decision
        """
        portfolio = portfolio or {}
        portfolio_state = portfolio_state or {}

        # ── 数据校验 ────────────────────────
        if data is None or data.empty or "close" not in data.columns:
            return self._default_decision(symbol, "invalid data")

        price = float(data["close"].iloc[-1])

        # ══════════════════════════════════════
        # Stage 1: MarketAgent 先跑
        # ══════════════════════════════════════
        market_out = self.market_agent.run(data, symbol)
        market_features = market_out.features
        regime = market_features.get("regime", MarketRegime.UNKNOWN.value)

        if portfolio_state:
            self.risk_agent.update_portfolio_state(
                total_value=float(portfolio_state.get("total_value", 100_000)),
                cash=float(portfolio_state.get("cash", 100_000)),
                long_value=float(portfolio_state.get("long_value", 0)),
                short_value=float(portfolio_state.get("short_value", 0)),
                daily_pnl=float(portfolio_state.get("daily_pnl", 0)),
                daily_trades=int(portfolio_state.get("daily_trades", 0)),
                consecutive_losses=int(portfolio_state.get("consecutive_losses", 0)),
                timestamp=float(
                    portfolio_state.get(
                        "timestamp", pd.Timestamp.now().timestamp()
                    )
                ),
           )

        # ══════════════════════════════════════
        # Stage 2: AlphaAgent、RiskAgent 并行
        # ══════════════════════════════════════
        alpha_ctx = {"market_features": market_features}
        risk_ctx = {
            "portfolio": portfolio,
            "portfolio_state": portfolio_state,
            "portfolio_prices": {symbol: price},
        }

        with ThreadPoolExecutor(max_workers=2) as ex:
            f_alpha = ex.submit(self.alpha_agent.run, data, symbol, alpha_ctx)
            f_risk = ex.submit(self.risk_agent.run, data, symbol, risk_ctx)

            alpha_out = self._safe_result(f_alpha, "alpha", symbol)
            risk_out = self._safe_result(f_risk, "risk", symbol)

        # ══════════════════════════════════════
        # Stage 3: 投票（只 Alpha 投方向票）
        # ══════════════════════════════════════
        final_action, final_confidence = self._vote(
            alpha_out, market_out, risk_out
        )

        # ★ strategy memory 调整置信度
        regime = market_features.get("regime", "unknown")   # 你文件里其实已有了
        signal_types = self._extract_signal_types(alpha_out)

        mem_ctx = self.memory.retrieve_for_decision(
               symbol=symbol,
               regime=regime,
               signal_types=signal_types,
               query=f"{symbol} {regime}",
               top_k=5,
             )

        if final_action != ACTION_HOLD and signal_types:
            signal_types = self._extract_signal_types(alpha_out)  # 已排序
            reflexion_mult, reasons = self.reflexion.get_confidence_multiplier(
                regime=regime,
                signal_types=signal_types,   # ★ 传完整列表，reflexion 内部会遍历+按样本量加权
                hour=market_features.get("hour"),  )
        
        # ── P1-3：Reflexion 置信度乘子 ──────────
        # 只对方向票生效（HOLD 不用乘）
        if final_action != ACTION_HOLD:
            reflexion_mult, reflexion_reasons = self.reflexion.get_confidence_multiplier(
                regime=regime,
                signal_types=self._extract_signal_types(alpha_out),
                hour=market_features.get("hour"),
            )
            if reflexion_mult != 1.0:
                final_confidence = float(np.clip(
                    final_confidence * reflexion_mult, 0.0, 1.0
                ))
                # 二次阈值检查：乘子可能把置信度拉到阈值以下
                if final_confidence < self.voting_threshold:
                    final_action = ACTION_HOLD

        # ExecutionAgent 的 DEFAULT_CONFIG 里 total_capital 硬编码为 100_000。
        # 实盘 / 回测里组合总价值会变，必须每根 K 线同步，否则：
        #   - total_value > 100_000 → 算出的 shares 偏小 → 实际仓位不足
        #   - total_value < 100_000 → 算出的 shares 偏大 → 实际仓位超限

        if portfolio_state and portfolio_state.get("total_value", 0) > 0:
            self.execution_agent.config["total_capital"] = float(
                portfolio_state["total_value"]
            )

        # ══════════════════════════════════════════════════
        # ★ Stage 3.5：强制平仓优先（仅多头）
        # ══════════════════════════════════════════════════
        risk_gate = risk_out.features.get("risk_gate", "clear")

        is_force_close = False
        if risk_gate == "force_close":
            # 当前有持仓 → 发反向信号平仓（仅多头）
            has_long = any(p.get("shares", 0) > 0 for p in portfolio.values())

            if has_long:
                final_action = ACTION_SELL
                final_confidence = 1.0
                is_force_close = True
                logger.warning(
                    "[%s] FORCE_CLOSE triggered → close long via SELL", symbol
                )
            else:
                # 无持仓，忽略强平
                final_action = ACTION_HOLD
                final_confidence = 0.0

        # ══════════════════════════════════════
        # Stage 4: ExecutionAgent 参数转换
        # ══════════════════════════════════════
        if final_action == ACTION_HOLD:
            exec_out = AgentOutput(
                agent_name="execution",
                action=Action.HOLD,
                confidence=final_confidence,
                reasons=["orchestrator decided hold"],
                features={
                    "entry": price,
                    "stop_loss": 0.0,
                    "take_profit": 0.0,
                    "position_size": 0.0,
                },
            )

        else:
            # ══════════════════════════════════════════════════
            # ★ 区分"开仓"和"平仓"（仅多头）
            # ══════════════════════════════════════════════════
            is_opening = self._is_opening_signal(final_action, portfolio)
            is_closing = self._is_closing_signal(final_action, portfolio)

            if is_closing:
                # 平仓信号：绕过 risk_gate，强制放行
                risk_gate = "clear"
                risk_size_mult = 1.0
                logger.debug(
                    "[%s] closing signal %s, bypass risk_gate",
                    symbol, final_action,
                )
                # ★ 这里：平仓 episode
                if self._current_episode_id is not None:
                    self.memory.close_trade(
                    episode_id=self._current_episode_id,
                    exit_time=pd.Timestamp.now().isoformat(),
                    exit_price=price,
                    exit_reason="signal",
                    symbol=symbol,
                    regime=regime,
                    signal_types=self._extract_signal_types(alpha_out), 
                    pnl=None,                          # 平仓时还不知道真实盈亏 → 先 None
                    pnl_pct=None,                      # 同上
                    max_drawdown=0.0,                  # 同上，事后补
                    holding_bars=None,                 # 事后补，或现在算
                    reflection=None,                   # 事后由 reflexion.reflect() 补
                    lesson=None,  )                     # 同上
                    self._current_episode_id = None
                    logger.info("[%s] Closed episode via closing signal", symbol)
            else:
                # 开仓信号：读 RiskAgent 的闸门
                risk_gate = risk_out.features.get("risk_gate", "clear")
                risk_size_mult = self._risk_gate_to_size_mult(risk_gate)

                if risk_gate == "block":
                    logger.info(
                        "[%s] RiskAgent blocked opening %s: %s",
                        symbol, final_action, risk_out.reasons,
                    )
                    final_action = ACTION_HOLD
                    final_confidence = 0.0

            if final_action == ACTION_HOLD:
                  # 被 RiskAgent 否决后，改成 HOLD 输出
                   exec_out = AgentOutput(
                       agent_name="execution",
                       action=Action.HOLD,
                       confidence=0.0,
                       reasons=["risk gate blocked opening"],
                       features={
                           "entry": price,
                           "stop_loss": 0.0,
                           "take_profit": 0.0,
                           "position_size": 0.0,
                       },
                   )
            else:
                exec_ctx = {
                    "alpha_action": final_action,
                    "alpha_confidence": final_confidence,
                    "risk_gate": risk_gate,               # ★ 新增
                    "risk_size_mult": risk_size_mult,     # ★ 新增
                    "market_features": market_features,   # ★ 新增（P0-4）
            }
                exec_out = self.execution_agent.run(data, symbol, exec_ctx)

                 # ExecutionAgent 自身也可能否决
                if exec_out.action == Action.HOLD:
                    logger.info(
                        "[%s] execution vetoed %s: %s",
                        symbol, final_action, exec_out.reasons,
                    )
                    final_action = ACTION_HOLD
                    final_confidence = 0.0



                # ★ Stage 4.5：开仓 episode 记录（仅新开仓）
        if (
            final_action != ACTION_HOLD
            and is_opening                       # ★ 关键：只在新开仓时记
            and not self._current_episode_id     # 防止重复开
        ):
            signal_types = self._extract_signal_types(alpha_out)
            self._current_episode_id = self.memory.open_trade(
                symbol=symbol,
                regime=regime,
                signal_types=signal_types,
                action=final_action,
                entry_time=pd.Timestamp.now().isoformat(),
                entry_price=float(exec_out.features.get("entry", price)),
                position_size=float(exec_out.features.get("position_size", 0)),
                agent_votes={
                    "alpha": alpha_out.action.value,
                    "market": market_out.action.value,
                    "risk": risk_out.action.value,
                    "execution": exec_out.action.value,
                },
                agent_confidences={
                    "alpha": round(alpha_out.confidence, 4),
                    "market": round(market_out.confidence, 4),
                    "risk": round(risk_out.confidence, 4),
                    "execution": round(exec_out.confidence, 4),
                },
                market_features=market_features,
            )
            logger.info("[%s] Opened episode: %s", symbol, self._current_episode_id)


        # ══════════════════════════════════════
        # 组装 Decision
        # ══════════════════════════════════════
        decision = Decision(
            symbol=symbol,
            timestamp=pd.Timestamp.now(),
            action=final_action,
            confidence=round(final_confidence, 4),
            entry_price=float(exec_out.features.get("entry", price)),
            stop_loss=float(exec_out.features.get("stop_loss", 0.0)),
            take_profit=float(exec_out.features.get("take_profit", 0.0)),
            position_size=float(exec_out.features.get("position_size", 0.0)),
            risk_level=str(risk_out.features.get("risk_level", RiskLevel.MEDIUM.value)),
            force_close=is_force_close,
            reasons=[
                f"alpha: {alpha_out.reasons}",
                f"market: regime={regime}",
                f"risk: {risk_out.features.get('risk_level')}",
                f"execution: {exec_out.reasons}",
            ],
            warnings=(
                market_out.warnings
                + alpha_out.warnings
                + risk_out.warnings
                + exec_out.warnings
            ),
            agent_votes={
                "alpha": alpha_out.action.value,
                "market": market_out.action.value,
                "risk": risk_out.action.value,
                "execution": exec_out.action.value,
            },
            agent_confidences={
                "alpha": round(alpha_out.confidence, 4),
                "market": round(market_out.confidence, 4),
                "risk": round(risk_out.confidence, 4),
                "execution": round(exec_out.confidence, 4),
            },
            market_features=market_features,
        )

        # 记录用于学习
        self._record(decision, alpha_out)
        self._try_learn()

        return decision

    def update_actual_return(self, decision_id: str, actual_return: float, max_drawdown: float = 0.0,) -> bool:
        """
        更新决策的实际收益（用于在线学习）

        Args:
            decision_id: Decision.decision_id
            actual_return: 实际收益率（如 0.05 表示 5%）

        Returns:
            是否成功更新
        """
        with self._history_lock:
            rec = self._id_index.get(decision_id)
            if rec is None:
                logger.warning("unknown decision_id=%s", decision_id)
                return False
            if rec.actual_return is not None:
                logger.warning("decision_id=%s already has return", decision_id)
                return False
            rec.actual_return = float(actual_return)
            # 顺便把决策也取出来，供 reflect 使用
            decision_ref = None
            for d in getattr(self, "_decision_cache", []):
                if d.decision_id == decision_id:
                    decision_ref = d
                    break

        # ── P1-3：Reflexion 反思 ────────────────
        if decision_ref is not None:
            self._reflect_on_decision(decision_ref, actual_return, max_drawdown)

        return True

    def reset_risk_state_after_close(self, current_cash: float) -> None:
        """ FORCE_CLOSE 平仓后调用，重置 RiskAgent 的累计回撤状态。
                避免"回撤超限 → 强平 → 空仓仍超限 → 再强平"的死循环。

                Args:
                    current_cash: 平仓后的当前现金（作为新的 peak_value 基准）
        """
        s = self.risk_agent._portfolio_state
        s.peak_value = max(current_cash, 1.0)
        s.current_drawdown = 0.0
        s.max_drawdown = 0.0
                # 连败计数也清零（平仓已发生，不应继续累计）
        s.consecutive_losses = 0
        logger.info(
            "[RISK RESET] peak_value=%.2f, dd=0, consecutive_losses=0",
            s.peak_value,
        )

    def get_performance(self) -> Dict[str, Any]:
        """获取性能统计"""
        with self._history_lock:
            hist = list(self._history)
        done = [r for r in hist if r.actual_return is not None]

        with self._weight_lock:
            alpha_weight = self._alpha_weight

        if done:
            avg_return = float(np.mean([r.actual_return for r in done]))
            win_rate = float(np.mean([r.actual_return > 0 for r in done]))
        else:
            avg_return = 0.0
            win_rate = 0.0

        return {
            "total_decisions": len(hist),
            "evaluated_decisions": len(done),
            "avg_return": round(avg_return, 6),
            "win_rate": round(win_rate, 4),
            "alpha_weight": round(alpha_weight, 4),
            "reflexion": self.reflexion.get_performance_summary(),
        }

    # ══════════════════════════════════════
    # 内部方法
    # ══════════════════════════════════════
    def _safe_result(self, future, name: str, symbol: str) -> AgentOutput:
        """安全获取 Future 结果（处理超时和异常）"""
        try:
            return future.result(timeout=self.agent_timeout)
        except TimeoutError:
            logger.warning("[%s] agent=%s timeout", symbol, name)
        except Exception:
            logger.exception("[%s] agent=%s exception", symbol, name)

        return AgentOutput(
            agent_name=name,
            action=Action.HOLD,
            confidence=0.0,
            warnings=[f"{name} failed"],
        )

    @staticmethod
    def _risk_gate_to_size_mult(gate: str) -> float:
        """
        把 RiskAgent 的 gate 映射成仓位乘数。
        clear  → 1.0
        reduce → 0.5
        block  → 0.0
        """
        return {
            "clear": 1.0,
            "reduce": 0.5,
            "block": 0.0,
            "force_close": 1.0,
        }.get(str(gate).lower(), 1.0)

    def _vote(
        self,
        alpha: AgentOutput,
        market: AgentOutput,
        risk: AgentOutput,
    ) -> Tuple[str, float]:
        """
        投票逻辑

        规则：
          1. 只有 AlphaAgent 投方向票
          2. MarketAgent 的 regime 影响置信度（乘子）
          3. RiskAgent 的 risk_level 影响置信度（乘子）
          4. 最终置信度 = alpha_conf × regime_mult × risk_mult × alpha_weight

        Args:
            alpha: AlphaAgent 输出
            market: MarketAgent 输出
            risk: RiskAgent 输出

        Returns:
            (action, confidence)
        """
        # ─ Alpha 投票 ─────────────────────────
        alpha_action = alpha.action.value
        alpha_conf = alpha.confidence

        # ─ Market Regime 调整系数 ─────────────
        regime = market.features.get("regime", MarketRegime.UNKNOWN.value)
        regime_mult = {
            MarketRegime.TRENDING_UP.value: 1.2,       # 趋势上涨：增强做多
            MarketRegime.TRENDING_DOWN.value: 1.2,     # 趋势下跌：增强看跌
            MarketRegime.RANGING.value: 0.8,           # 震荡：降低方向性
            MarketRegime.VOLATILE.value: 0.8,          # 波动剧烈：大幅降低
            MarketRegime.BREAKOUT.value: 1.3,          # 突破：增强
            MarketRegime.REVERSAL.value: 0.9,          # 反转：适度降低
            MarketRegime.UNKNOWN.value: 0.7,           # 未知：保守
        }.get(regime, 1.0)

        # 特殊规则：趋势方向与信号方向不一致时，压制
        if regime == MarketRegime.TRENDING_UP.value and alpha_action == ACTION_SELL:
            regime_mult *= 0.5
        elif regime == MarketRegime.TRENDING_DOWN.value and alpha_action == ACTION_BUY:
            regime_mult *= 0.5

        # ─ Risk Level 调整系数 ────────────────
        risk_level = risk.features.get("risk_level", RiskLevel.MEDIUM.value)
        risk_mult = {
            RiskLevel.LOW.value: 1.0,
            RiskLevel.MEDIUM.value: 0.9,
            RiskLevel.HIGH.value: 0.6,
            RiskLevel.CRITICAL.value: 0.0,  # 完全阻止
            "extreme": 0.0,
        }.get(risk_level, 0.8)

        # ─ Alpha Weight（在线学习得到）────────
        with self._weight_lock:
            alpha_weight = self._alpha_weight

        # ─ 最终置信度 ─────────────────────────
        final_confidence = alpha_conf * regime_mult * risk_mult * alpha_weight
        final_confidence = float(np.clip(final_confidence, 0.0, 1.0))

        # ─ 阈值检查 ───────────────────────────
        if final_confidence < self.voting_threshold or alpha_action == ACTION_HOLD:
            return ACTION_HOLD, round(final_confidence, 4)

        return alpha_action, round(final_confidence, 4)

    # ══════════════════════════════════════
    # 辅助：判断开仓 / 平仓（仅多头）
    # ══════════════════════════════════════
    @staticmethod
    def _is_opening_signal(action: str, portfolio: Dict[str, Dict[str, Any]]) -> bool:
        """判断信号是否是"新开仓"：空仓 BUY"""
        has_long = any(p.get("shares", 0) > 0 for p in portfolio.values())

        if action == ACTION_BUY:
            return not has_long
        return False

    @staticmethod
    def _is_closing_signal(action: str, portfolio: Dict[str, Dict[str, Any]]) -> bool:
        """判断信号是否是"平仓"：持多 SELL"""
        has_long = any(p.get("shares", 0) > 0 for p in portfolio.values())

        if action == ACTION_SELL:
            return has_long
        return False

    @staticmethod
    def _extract_signal_types(alpha_out: AgentOutput) -> List[str]:
        """从 AlphaAgent 输出里提取激活的信号类型"""
        details = alpha_out.features.get("signal_details", [])
        return list({d.get("type") for d in details if d.get("type")})

    def _record(self, decision: Decision, alpha_out: AgentOutput) -> None:
        """记录决策（用于在线学习）"""
        rec = _PerfRecord(
            decision_id=decision.decision_id,
            alpha_vote=alpha_out.action.value,
            alpha_confidence=alpha_out.confidence,
        )

        with self._history_lock:
            self._history.append(rec)
            self._id_index[decision.decision_id] = rec

            # 缓存完整 Decision 供 reflect 使用
            if not hasattr(self, "_decision_cache"):
                self._decision_cache: List[Decision] = []
            self._decision_cache.append(decision)
            if len(self._decision_cache) > self.max_history:
                self._decision_cache.pop(0)
            # 限制历史长度
            if len(self._history) > self.max_history:
                old = self._history.pop(0)
                self._id_index.pop(old.decision_id, None)

    def _try_learn(self) -> None:
        """
        在线学习：根据实际收益调整 AlphaAgent 权重

        规则：
          - 只有方向票（BUY/SELL）才参与学习
          - 做多正收益 or 看跌负收益 → 正样本
          - 做多负收益 or 看跌正收益 → 负样本
          - 使用贝叶斯平滑后的胜率更新权重
        """
        with self._history_lock:
            done = [r for r in self._history if r.actual_return is not None]

        if len(done) < self.min_samples_to_learn:
            return

        # 只取最近 50 个样本
        recent = done[-50:]

        # 计算 AlphaAgent 的胜率
        correct = 0
        total = 0
        for rec in recent:
            if rec.alpha_vote == ACTION_HOLD:
                continue  # HOLD 不参与学习

            total += 1
            ret = rec.actual_return

            # 判断是否预测正确
            if (rec.alpha_vote == ACTION_BUY and ret > 0) or \
               (rec.alpha_vote == ACTION_SELL and ret < 0):
                correct += 1

        if total == 0:
            return

        # 贝叶斯平滑（伪计数）
        alpha_prior = 1.0
        beta_prior = 1.0
        win_rate = (correct + alpha_prior) / (total + alpha_prior + beta_prior)

        # 目标权重：胜率映射到 [0.5, 1.5]
        target_weight = 0.5 + win_rate

        # 指数移动平均更新
        with self._weight_lock:
            old_weight = self._alpha_weight
            new_weight = (1 - self.learning_rate) * old_weight + \
                         self.learning_rate * target_weight
            new_weight = float(np.clip(new_weight, 0.3, 2.0))  # 防止极端值
            self._alpha_weight = new_weight

        logger.info(
            "alpha_weight updated: %.4f → %.4f (win_rate=%.2f%%, n=%d)",
            old_weight, new_weight, win_rate * 100, total
        )

    def _reflect_on_decision(
        self,
        decision: Decision,
        actual_return: float,
        max_drawdown: float,
    ) -> None:
        """
        P1-3:把一次决策的结果喂给 ReflexionEngine
        """
        try:
            # 从 decision 里提取 signal_type（如果 alpha 输出里有）
            signal_types = []
            for r in decision.reasons:
                # reasons 里形如 "alpha: [...]"，粗略提取即可
                if "signal_details" in str(r):
                    signal_types.append("unknown")
                    break

            self.reflexion.reflect(
                decision={
                    "action": decision.action,
                    "confidence": decision.confidence,
                    "regime": decision.market_features.get("regime", "unknown"),
                    "signal_type": signal_types[0] if signal_types else None,
                    "hour": decision.market_features.get("hour"),
                    "strategy_type": decision.action,
                },
                outcome={
                    "profit": actual_return * 100.0,      # 转成百分数
                    "max_drawdown": abs(max_drawdown),    # 正数幅度
                },
                context={
                    "regime": decision.market_features.get("regime", "unknown"),
                },
            )
        except Exception:
            logger.exception("Reflexion reflect() 失败")

    @staticmethod
    def _default_decision(symbol: str, reason: str) -> Decision:
        """默认决策（HOLD）"""
        return Decision(
            symbol=symbol,
            timestamp=pd.Timestamp.now(),
            action=ACTION_HOLD,
            confidence=0.0,
            entry_price=0.0,
            stop_loss=0.0,
            take_profit=0.0,
            position_size=0.0,
            risk_level=RiskLevel.MEDIUM.value,
            reasons=[reason],
        )