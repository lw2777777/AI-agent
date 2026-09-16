"""
Reflexion Engine v2
- 与四层 Agent 架构 (Market/Alpha/Execution/Risk) 对齐
- regime-aware 洞察 + 置信度乘子闭环
- 单位/符号统一:profit=百分数,max_drawdown=正数幅度
"""
from __future__ import annotations

import json
import logging
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple 
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════
# 数据结构
# ══════════════════════════════════════════════════════════
class EmotionalState(str, Enum):
    CONFIDENT = "confident"
    ANXIOUS = "anxious"
    NEUTRAL = "neutral"


@dataclass
class ReflexionMemory:
    timestamp: pd.Timestamp
    decision: Dict[str, Any]
    outcome: Dict[str, Any]
    context: Dict[str, Any]
    reflection: str
    improvement_suggestions: List[str]
    decision_quality: float
    emotional_state: EmotionalState
    score_breakdown: Dict[str, float] = field(default_factory=dict)


@dataclass
class StrategyInsight:
    pattern: str
    description: str
    success_count: int = 0
    fail_count: int = 0
    confidence: float = 0.5
    examples: List[Dict[str, Any]] = field(default_factory=list)
    recommendation: str = ""

    @property
    def total(self) -> int:
        return self.success_count + self.fail_count

    @property
    def success_rate(self) -> float:
        return self.success_count / self.total if self.total else 0.0

    @property
    def score(self) -> float:
        """样本量折扣后的得分，避免小样本霸榜"""
        sample_factor = min(self.total / 5.0, 1.0)
        return self.success_rate * self.confidence * sample_factor

    def update(self, success: bool, example: Dict[str, Any], max_examples: int = 30):
        if success:
            self.success_count += 1
        else:
            self.fail_count += 1

        self.examples.append(example)
        if len(self.examples) > max_examples:
            self.examples = self.examples[-max_examples:]

        # Beta 后验均值（拉普拉斯平滑）
        alpha = 1 + self.success_count
        beta = 1 + self.fail_count
        self.confidence = alpha / (alpha + beta)


# ══════════════════════════════════════════════════════════
# 引擎
# ══════════════════════════════════════════════════════════
class ReflexionEngine:
    name = "ReflexionEngine"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

        memory_size = int(self.config.get("memory_size", 1000))
        self.memory: deque = deque(maxlen=memory_size)

        self.insights: List[StrategyInsight] = []
        self._insight_index: Dict[str, int] = {}
        self.performance_history: List[Dict[str, Any]] = []

        self.learning_rate = float(self.config.get("learning_rate", 0.08))
        self.min_learning_rate = float(self.config.get("min_learning_rate", 0.01))
        self.max_learning_rate = float(self.config.get("max_learning_rate", 0.20))

        # 只保留真正会被触发的维度
        self.weights = self.config.get("weights", {
            "accuracy": 0.25,   # 若 Orchestrator 提供 expected/actual_direction
            "profit":   0.40,   # 主维度
            "risk":     0.20,   # 依赖 outcome["max_drawdown"]
            "timing":   0.15,   # 若 Orchestrator 提供 timing_score
        })
        self.weights = self._normalize_weights(self.weights)

        # 乘子映射边界
        self.mult_min = float(self.config.get("mult_min", 0.5))
        self.mult_max = float(self.config.get("mult_max", 1.5))
        self.min_samples_for_mult = int(self.config.get("min_samples_for_mult", 10))

    # ══════════════════════════════════════
    # Public API
    # ══════════════════════════════════════
    def reflect(
        self,
        decision: Dict[str, Any],
        outcome: Dict[str, Any],
        context: Dict[str, Any],
    ) -> ReflexionMemory:
        try:
            quality, breakdown = self._evaluate_decision_quality(decision, outcome, context)
            reflection = self._generate_reflection(decision, outcome, quality, context)
            improvements = self._suggest_improvements(decision, outcome, quality, context)
            emotional_state = self._assess_emotional_state(quality, outcome)

            ts = pd.Timestamp.now()
            memory = ReflexionMemory(
                timestamp=ts,
                decision=decision,
                outcome=outcome,
                context=context,
                reflection=reflection,
                improvement_suggestions=improvements,
                decision_quality=quality,
                emotional_state=emotional_state,
                score_breakdown=breakdown,
            )
            self.memory.append(memory)
            self.performance_history.append({
                "quality": quality,
                "timestamp": ts,
                "profit": outcome.get("profit", 0.0),
            })

            self._update_insights(memory)
            self._adjust_strategy(memory)
            return memory
        except Exception:
            logger.exception("Reflection error")
            return self._default_memory(decision, outcome, context)

    def get_insights(self, top_n: int = 10) -> List[StrategyInsight]:
        return sorted(self.insights, key=lambda x: x.score, reverse=True)[:top_n]

    def get_insights_by_regime(self, regime: str, top_n: int = 5) -> List[StrategyInsight]:
        """返回指定市场状态相关的洞察"""
        if not regime:
            return []
        prefix = f"regime_{regime}"
        return [
            i for i in self.insights
            if i.pattern.startswith(prefix)
        ][:top_n]

    # ── 核心新增：置信度乘子 ──
    def get_confidence_multiplier(
        self,
        regime: Optional[str] = None,
        signal_types: Optional[List[str]] = None,
        hour: Optional[int] = None,
    ) -> Tuple[float, List[str]]:
        """
        根据历史 insights 返回置信度乘子。

        Returns:
            (multiplier, reasons)
            multiplier ∈ [mult_min, mult_max]，无有效数据时返回 1.0
        """
        signal_types = signal_types or []
        relevant: List[StrategyInsight] = []

        if regime:
            # 组合 pattern（更精确）
            for st in signal_types:
                key = f"regime_{regime}_signal_{st}"
                if key in self._insight_index:
                    relevant.append(self.insights[self._insight_index[key]])
            # regime 整体
            key = f"regime_{regime}"
            if key in self._insight_index:
                relevant.append(self.insights[self._insight_index[key]])

        if hour is not None:
            key = f"hour_{int(hour)}"
            if key in self._insight_index:
                relevant.append(self.insights[self._insight_index[key]])

        if not relevant:
            return 1.0, []

        total_n = sum(i.total for i in relevant)
        if total_n < self.min_samples_for_mult:
            return 1.0, [f"insufficient_samples(n={total_n})"]

        weighted_sr = sum(i.success_rate * i.total for i in relevant) / total_n
        # 胜率 0.5 → 1.0；0.7 → 1.2；0.3 → 0.8
        multiplier = float(np.clip(
            0.5 + weighted_sr,
            self.mult_min,
            self.mult_max,
        ))

        reasons = [
            f"reflexion_mult={multiplier:.3f}",
            f"hist_win_rate={weighted_sr:.2%}(n={total_n})",
        ]
        return multiplier, reasons

    def get_performance_summary(self) -> Dict[str, Any]:
        if not self.performance_history:
            return {
                "avg_quality": 0.0,
                "quality_std": 0.0,
                "trend": "neutral",
                "total_decisions": 0,
                "success_rate": 0.0,
                "avg_profit": 0.0,
                "learning_rate": self.learning_rate,
                "by_regime": {},
            }

        hist = self.performance_history[-200:]
        qualities = np.array([h["quality"] for h in hist], dtype=float)
        profits = np.array([h.get("profit", 0.0) for h in hist], dtype=float)

        trend = "stable"
        if len(qualities) >= 20:
            recent = qualities[-10:].mean()
            prev = qualities[-20:-10].mean()
            if recent > prev + 1e-3:
                trend = "improving"
            elif recent < prev - 1e-3:
                trend = "declining"

        # 按 regime 分解
        regime_buckets: Dict[str, List[float]] = {}
        for m in self.memory:
            r = (
                m.context.get("regime")
                or m.context.get("market_condition")
                or "unknown"
            )
            regime_buckets.setdefault(str(r), []).append(m.decision_quality)

        by_regime = {
            r: {
                "count": len(v),
                "avg_quality": round(float(np.mean(v)), 4),
                "success_rate": round(float(np.mean([q > 0.5 for q in v])), 4),
            }
            for r, v in regime_buckets.items()
        }

        return {
            "avg_quality": float(qualities.mean()),
            "quality_std": float(qualities.std(ddof=0)),
            "trend": trend,
            "total_decisions": len(self.performance_history),
            "success_rate": float((qualities > 0.5).mean()),
            "avg_profit": float(profits.mean()),
            "learning_rate": float(self.learning_rate),
            "by_regime": by_regime,
        }

    # ══════════════════════════════════════
    # 持久化
    # ══════════════════════════════════════
    def save_state(self, filepath: str) -> None:
        state = {
            "learning_rate": self.learning_rate,
            "performance_history": self.performance_history[-5000:],
            "insights": [
                {
                    "pattern": i.pattern,
                    "description": i.description,
                    "success_count": i.success_count,
                    "fail_count": i.fail_count,
                    "confidence": i.confidence,
                    "examples": i.examples[-10:],
                    "recommendation": i.recommendation,
                }
                for i in self.insights
            ],
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, default=str, indent=2)
        logger.info("ReflexionEngine saved to %s (%d insights)", filepath, len(self.insights))

    def load_state(self, filepath: str) -> None:
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                state = json.load(f)
        except FileNotFoundError:
            logger.info("ReflexionEngine state file not found: %s", filepath)
            return

        self.learning_rate = float(state.get("learning_rate", self.learning_rate))
        self.learning_rate = float(np.clip(
            self.learning_rate, self.min_learning_rate, self.max_learning_rate
        ))
        self.performance_history = state.get("performance_history", [])

        self.insights = []
        self._insight_index = {}
        for idx, raw in enumerate(state.get("insights", [])):
            insight = StrategyInsight(
                pattern=raw["pattern"],
                description=raw.get("description", ""),
                success_count=int(raw.get("success_count", 0)),
                fail_count=int(raw.get("fail_count", 0)),
                confidence=float(raw.get("confidence", 0.5)),
                examples=raw.get("examples", []),
                recommendation=raw.get("recommendation", ""),
            )
            self.insights.append(insight)
            self._insight_index[insight.pattern] = idx

        logger.info("ReflexionEngine loaded from %s (%d insights)", filepath, len(self.insights))

    # ══════════════════════════════════════
    # 决策质量评估
    # ══════════════════════════════════════
    def _evaluate_decision_quality(
        self,
        decision: Dict[str, Any],
        outcome: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Tuple[float, Dict[str, float]]:
        """
        约定：
          - outcome["profit"]        : 百分数，如 2.5 表示 +2.5%
          - outcome["max_drawdown"]  : 正数幅度，如 0.08 表示 8%
          - outcome["timing_score"]  : [0, 1]，可选
        """
        score = 0.0
        total_weight = 0.0
        breakdown: Dict[str, float] = {}

        # 1) Accuracy
        if "expected_direction" in decision and "actual_direction" in outcome:
            w = self.weights["accuracy"]
            acc = 1.0 if decision["expected_direction"] == outcome["actual_direction"] else 0.0
            score += w * acc
            total_weight += w
            breakdown["accuracy"] = acc

        # 2) Profit（百分数口径）
        if "profit" in outcome:
            w = self.weights["profit"]
            p = float(outcome["profit"])
            profit_score = 1.0 / (1.0 + np.exp(-p / 2.0))  # p=0 → 0.5
            score += w * profit_score
            total_weight += w
            breakdown["profit"] = float(profit_score)

        # 3) Risk（正数幅度口径）
        if "max_drawdown" in outcome:
            w = self.weights["risk"]
            dd = abs(float(outcome["max_drawdown"]))
            if dd < 0.02:
                risk_score = 1.0
            elif dd < 0.05:
                risk_score = 0.8
            elif dd < 0.10:
                risk_score = 0.5
            else:
                risk_score = 0.2
            score += w * risk_score
            total_weight += w
            breakdown["risk"] = risk_score

        # 4) Timing
        if "timing_score" in outcome:
            w = self.weights["timing"]
            t = float(np.clip(outcome["timing_score"], 0, 1))
            score += w * t
            total_weight += w
            breakdown["timing"] = t

        quality = score / total_weight if total_weight > 0 else 0.5
        return float(np.clip(quality, 0.0, 1.0)), breakdown

    # ══════════════════════════════════════
    # 反思文本
    # ══════════════════════════════════════
    def _generate_reflection(
        self,
        decision: Dict[str, Any],
        outcome: Dict[str, Any],
        quality: float,
        context: Dict[str, Any],
    ) -> str:
        msgs: List[str] = []

        if quality >= 0.8:
            msgs.append("决策质量高，执行和风控整体有效。")
        elif quality >= 0.5:
            msgs.append("决策质量中等，信号强度与执行细节仍可优化。")
        else:
            msgs.append("决策质量偏低，需要收紧触发条件并降低风险敞口。")

        p = float(outcome.get("profit", 0.0))
        msgs.append(f"本次收益: {p:.2f}%")

        # max_drawdown 统一用正数幅度
        dd = abs(float(outcome.get("max_drawdown", 0.0)))
        if dd >= 0.10:
            msgs.append("最大回撤较大，风险超阈值。")
        elif dd >= 0.05:
            msgs.append("最大回撤可控，但可继续优化止损。")
        else:
            msgs.append("回撤控制较好。")

        regime = context.get("regime") or context.get("market_condition")
        if regime:
            msgs.append(f"市场状态: {regime}")

        return " ".join(msgs)

    def _suggest_improvements(
        self,
        decision: Dict[str, Any],
        outcome: Dict[str, Any],
        quality: float,
        context: Dict[str, Any],
    ) -> List[str]:
        suggestions: List[str] = []

        if quality < 0.5:
            suggestions += [
                "提高入场阈值（提高最小置信度）",
                "减少仓位（例如下调每笔风险）",
                "增加信号确认层（如趋势过滤/成交量过滤）",
            ]

        if float(outcome.get("profit", 0)) < 0:
            suggestions += [
                "缩短止损触发时间窗口，避免亏损扩大",
                "降低高波动时段交易频率",
            ]

        if abs(float(outcome.get("max_drawdown", 0))) >= 0.10:
            suggestions += [
                "下调最大仓位上限",
                "启用相关性约束，避免同方向集中暴露",
            ]

        if float(decision.get("confidence", 0.5)) < 0.6 and float(outcome.get("profit", 0)) < 0:
            suggestions.append("仅在高置信度信号下开仓")

        regime = str(context.get("regime", "")).lower()
        if regime in ("ranging", "sideways"):
            suggestions.append("震荡市优先使用均值回归策略，减少追涨杀跌")

        dedup = list(dict.fromkeys(suggestions))
        return dedup if dedup else ["当前策略表现稳定，保持小步迭代。"]

    def _assess_emotional_state(
        self, quality: float, outcome: Dict[str, Any]
    ) -> EmotionalState:
        profit = float(outcome.get("profit", 0))
        if quality > 0.7 and profit > 0:
            return EmotionalState.CONFIDENT
        if quality < 0.3 and profit < 0:
            return EmotionalState.ANXIOUS
        return EmotionalState.NEUTRAL

    # ══════════════════════════════════════
    # Insights
    # ══════════════════════════════════════
    def _update_insights(self, memory: ReflexionMemory):
        patterns = self._extract_patterns(memory)

        for pattern, info in patterns.items():
            success = bool(info["success"])
            example = info["example"]

            if pattern in self._insight_index:
                i = self.insights[self._insight_index[pattern]]
                i.update(success=success, example=example)
            else:
                new_insight = StrategyInsight(
                    pattern=pattern,
                    description=info["description"],
                    recommendation=info["recommendation"],
                )
                new_insight.update(success=success, example=example)
                self._insight_index[pattern] = len(self.insights)
                self.insights.append(new_insight)

        max_insights = int(self.config.get("max_insights", 200))
        keep_top = int(self.config.get("keep_top_insights", 100))
        if len(self.insights) > max_insights:
            self.insights = sorted(self.insights, key=lambda x: x.score, reverse=True)[:keep_top]
            self._insight_index = {ins.pattern: idx for idx, ins in enumerate(self.insights)}

    def _extract_patterns(self, memory: ReflexionMemory) -> Dict[str, Dict[str, Any]]:
        patterns: Dict[str, Dict[str, Any]] = {}
        d, o, c = memory.decision, memory.outcome, memory.context

        # 成功定义：profit > 0
        success = 1 if float(o.get("profit", 0.0)) > 0 else 0
        example = {"decision": d, "outcome": o, "context": c}

        regime = c.get("regime") or c.get("market_condition") or d.get("regime")
        signal_type = d.get("strategy_type") or d.get("signal_type")
        hour = d.get("hour") if d.get("hour") is not None else c.get("hour")

        # 1) regime 整体
        if regime:
            key = f"regime_{regime}"
            patterns[key] = {
                "description": f"{regime} 市场状态下的整体表现",
                "success": success,
                "example": example,
                "recommendation": f"{regime} 环境下调整信号权重",
            }

        # 2) regime × signal
        if regime and signal_type:
            key = f"regime_{regime}_signal_{signal_type}"
            patterns[key] = {
                "description": f"{regime} 下 {signal_type} 信号表现",
                "success": success,
                "example": example,
                "recommendation": f"{regime} 时优先/抑制 {signal_type}",
            }

        # 3) 纯 signal（跨 regime 汇总）
        if signal_type:
            key = f"signal_{signal_type}"
            patterns[key] = {
                "description": f"{signal_type} 信号整体表现",
                "success": success,
                "example": example,
                "recommendation": f"评估 {signal_type} 信号有效性",
            }

        # 4) 时段
        if hour is not None:
            key = f"hour_{int(hour)}"
            patterns[key] = {
                "description": f"{hour} 点时段表现",
                "success": success,
                "example": example,
                "recommendation": f"评估 {hour} 点时段是否值得交易",
            }

        # 5) 置信度分桶
        conf = float(d.get("confidence", 0.5))
        conf_bucket = "high" if conf >= 0.7 else ("mid" if conf >= 0.55 else "low")
        key = f"confidence_{conf_bucket}"
        patterns[key] = {
            "description": f"{conf_bucket} 置信度信号表现",
            "success": success,
            "example": example,
            "recommendation": f"优先保留 {conf_bucket} 置信度且历史胜率高的信号",
        }

        return patterns

    # ══════════════════════════════════════
    # 学习率调整（均值回归型，避免单调衰减）
    # ══════════════════════════════════════
    def _adjust_strategy(self, memory: ReflexionMemory):
        q = memory.decision_quality
        target_lr = 0.08

        if q > 0.7:
            # 高质量：向 0.15 上限靠拢，但增速递减
            self.learning_rate += 0.005 * (1.0 - self.learning_rate / 0.15)
        elif q < 0.3:
            # 低质量：缓慢降
            self.learning_rate -= 0.002
        else:
            # 中性：缓慢回归目标
            self.learning_rate += 0.001 * (target_lr - self.learning_rate)

        self.learning_rate = float(np.clip(
            self.learning_rate,
            self.min_learning_rate,
            self.max_learning_rate,
        ))

    # ══════════════════════════════════════
    # Utils
    # ══════════════════════════════════════
    @staticmethod
    def _normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
        total = sum(max(v, 0) for v in weights.values())
        if total <= 0:
            n = len(weights)
            return {k: 1.0 / n for k in weights}
        return {k: max(v, 0) / total for k, v in weights.items()}

    def _default_memory(
        self,
        decision: Dict[str, Any],
        outcome: Dict[str, Any],
        context: Dict[str, Any],
    ) -> ReflexionMemory:
        return ReflexionMemory(
            timestamp=pd.Timestamp.now(),
            decision=decision,
            outcome=outcome,
            context=context,
            reflection="反思过程异常，返回默认结果。",
            improvement_suggestions=["检查输入字段完整性与数值范围"],
            decision_quality=0.5,
            emotional_state=EmotionalState.NEUTRAL,
            score_breakdown={},
        )