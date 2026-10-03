"""
llm_agents.py — MarketAgent / AlphaAgent 的 LLM 增强版

设计：
  - 结构化行情：规则（原 MarketAgent / AlphaAgent 逻辑）
  - 非结构化文本：LLM（NewsProvider → LLM 情绪抽取）
  - AlphaAgent：规则产信号 → LLM 做最终方向判断
  - LLM 失败 → 降级到规则结果（不阻塞回测）
  - 不依赖 response_format；prompt 约束 + 解析容错

适配 LLMAdapter：
  - chat(messages, ...) → {"content", "tool_calls", "finish_reason", "usage"}
  - 使用实例级 usage 累计（adapter.get_usage()）
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .base_agent import Action, AgentOutput
from .alpha_agent import AlphaAgent
from .market_agent import MarketAgent
from agent_framework.core.llm_adapter import LLMAdapter
from .news_provider import NewsItem, NewsProvider, get_news_provider

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════
# JSON 解析工具
# ══════════════════════════════════════════════════════════
def _try_parse_json(content: Optional[str]) -> Dict[str, Any]:
    """从 LLM 文本输出里抽 JSON，失败返回 {}"""
    if not content:
        return {}
    content = content.strip()

    # 1. 直接试
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    # 2. 剥 markdown ```json ... ```
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    # 3. 抓第一个 { 到最后一个 }
    start, end = content.find("{"), content.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(content[start:end + 1])
        except json.JSONDecodeError:
            pass

    logger.warning("JSON 抽取失败: %s", content[:200])
    return {}


def _call_llm_json(
    adapter: LLMAdapter,
    system: str,
    user: str,
) -> Dict[str, Any]:
    """调 adapter，返回解析后的 dict；失败返回 {}"""
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    try:
        resp = adapter.chat(messages, temperature=0.0)
    except Exception as e:
        logger.warning("LLM 调用异常: %s", e)
        return {}

    return _try_parse_json(resp.get("content"))


# ══════════════════════════════════════════════════════════
# MarketAgent + LLM 情绪
# ══════════════════════════════════════════════════════════
SENTIMENT_SYSTEM_PROMPT = """你是一个金融新闻情绪分析器。
输入：一批关于某只股票的新闻标题/摘要。
输出：严格的 JSON，字段如下：
{
  "sentiment": -1.0 到 1.0 的浮点数（负=利空，正=利多，0=中性）,
  "event_type": "earnings|mna|regulation|analyst|product|insider|other",
  "impact": "low|medium|high",
  "reason": "一句话中文说明"
}
只输出 JSON，不要任何其他文字、不要 markdown 代码块。"""


class LLMMarketAgent(MarketAgent):
    """
    在 MarketAgent 基础上：
      - 保留所有结构化行情逻辑
      - 新增：拉取新闻 → LLM 抽取情绪 → 注入 features
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        llm: Optional[LLMAdapter] = None,
        news_provider: Optional[NewsProvider] = None,
    ):
        super().__init__(config)
        self._llm = llm
        self._news = news_provider or get_news_provider(use_real=False)

        self.use_llm_sentiment = bool(self.get_config("use_llm_sentiment", True))
        self.news_lookback_hours = int(self.get_config("news_lookback_hours", 24))
        self.news_max_items = int(self.get_config("news_max_items", 8))
        self.min_sentiment_items = int(self.get_config("min_sentiment_items", 1))

        self._sentiment_cache: Dict[tuple, Dict[str, Any]] = {}

    def run(self, data, symbol, context=None, order_book=None):
        # 1. 规则层
        base = super().run(data, symbol, context, order_book)

        if not self.use_llm_sentiment or self._llm is None:
            return base

        # 2. 拉新闻
        try:
            if isinstance(data.index, pd.DatetimeIndex):
                as_of = data.index[-1].to_pydatetime()
            else:
                as_of = pd.Timestamp.now().to_pydatetime()
        except Exception:
            as_of = pd.Timestamp.now().to_pydatetime()

        news = self._news.fetch(
            symbol=symbol,
            as_of=as_of,
            lookback_hours=self.news_lookback_hours,
            max_items=self.news_max_items,
        )

        if len(news) < self.min_sentiment_items:
            base.features.update({
                "news_sentiment": 0.0,
                "news_event_type": "none",
                "news_impact": "low",
                "news_count": 0,
            })
            return base

        # 3. LLM 抽取情绪
        date_key = as_of.strftime("%Y-%m-%d")
        sent = self._extract_sentiment(symbol, news, date_key=date_key)
        raw_sentiment = float(sent.get("sentiment", 0.0))
        event_type = sent.get("event_type", "other")

        # ════════════════════════════════════════════════════
        # ★★★ ④ 新增：RAG 历史新闻校准（加在这里）★★★
        # ════════════════════════════════════════════════════
        memory_manager = (context or {}).get("memory_manager")
        calibrated_sentiment = raw_sentiment
        calibration_note = "no_calibration"

        if memory_manager is not None and abs(raw_sentiment) > 0.3:
            try:
                polarity = "利多" if raw_sentiment > 0 else "利空"
                query = f"新闻事件 {event_type} {polarity} 后市场表现"

                historical = memory_manager.retrieve_rag(
                    query=query, top_k=3,
                )

                if historical:
                    pnls = []
                    for h in historical:
                        meta = h.get("metadata", {})
                        if meta.get("type") == "news_case" and "pnl_pct" in meta:
                            pnls.append(float(meta["pnl_pct"]))

                    if pnls:
                        avg_hist = float(np.mean(pnls))
                        if avg_hist * raw_sentiment > 0:
                        # 历史验证了当前情绪 → 加强
                            calibrated_sentiment = float(
                                np.clip(raw_sentiment * 1.15, -1, 1)
                            )
                            calibration_note = (
                                f"validated by {len(pnls)} cases "
                                f"(avg_pnl={avg_hist:+.2%})"
                            )

                        else:
                        # 历史推翻了当前情绪 → 削弱
                            calibrated_sentiment = float(
                                np.clip(raw_sentiment * 0.7, -1, 1)
                            )
                            calibration_note = (
                                f"weakened by {len(pnls)} cases "
                                f"(avg_pnl={avg_hist:+.2%})"
                            )
            except Exception:
                logger.exception("MarketAgent RAG 校准失败，降级到原始情绪")



        base.features.update({
            "news_sentiment":calibrated_sentiment,      
            "news_sentiment_raw": raw_sentiment, 
            "news_event_type": sent.get("event_type", "other"),
            "news_impact": sent.get("impact", "low"),
            "news_reason": sent.get("reason", ""),
            "news_count": len(news),
            "news_calibration": calibration_note,   
        })

        
        if abs(calibrated_sentiment) > 0.5:
            base.confidence = float(np.clip(base.confidence * 1.1, 0.0, 1.0))
            base.add_reason(f"news_sentiment={calibrated_sentiment:+.2f}")

        return base

    def _extract_sentiment(
        self,
        symbol: str,
        news: List[NewsItem],
        date_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        # ── 缓存命中检查 ──
        cache_key = (symbol, date_key) if date_key else None
        if cache_key is not None and cache_key in self._sentiment_cache:
            return self._sentiment_cache[cache_key]

        news_text = "\n".join(
            f"- [{n.timestamp}] {n.title}" for n in news[: self.news_max_items]
        )
        user_prompt = f"股票代码：{symbol}\n最近新闻：\n{news_text}"

        parsed = _call_llm_json(
            self._llm,
            system=SENTIMENT_SYSTEM_PROMPT,
            user=user_prompt,
        )

        if not parsed:
            result = {
                "sentiment": 0.0,
                "event_type": "other",
                "impact": "low",
                "reason": "llm_failed",
            }
        else:
            result = {
                "sentiment": float(np.clip(float(parsed.get("sentiment", 0.0)), -1.0, 1.0)),
                "event_type": str(parsed.get("event_type", "other")),
                "impact": str(parsed.get("impact", "low")),
                "reason": str(parsed.get("reason", "")),
            }

        # ── 写缓存（失败也缓存，避免同一日反复重试）──
        if cache_key is not None:
            self._sentiment_cache[cache_key] = result

        return result

# ══════════════════════════════════════════════════════════
# AlphaAgent + LLM 决策
# ══════════════════════════════════════════════════════════
ALPHA_SYSTEM_PROMPT = """你是一个量化交易决策助手。
你会收到一份压缩后的市场特征快照（不是原始K线），包括：
- 规则层算出的信号（signals）
- 市场状态（regime）
- 新闻情绪（news_sentiment）
你的任务：判断是否应该 BUY / SELL / HOLD，以及置信度。

输出严格 JSON：
{
  "action": "buy|sell|hold",
  "confidence": 0.0 到 1.0 的浮点数,
  "reason": "一句话中文说明"
}
注意：
- 只有当多个信号同向且情绪支持时才给高置信度
- 信号冲突时给 HOLD
- 不要臆造未提供的信息
只输出 JSON，不要 markdown 代码块。"""


class LLMAlphaAgent(AlphaAgent):
    """
    在 AlphaAgent 基础上：
      - 保留规则信号生成
      - 规则输出压缩成紧凑 prompt → LLM 做最终判断
      - LLM 失败 → 降级到规则结果
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        llm: Optional[LLMAdapter] = None,
    ):
        super().__init__(config)
        self._llm = llm

        self.use_llm_decision = bool(self.get_config("use_llm_decision", True))
        self.llm_override_threshold = float(
            self.get_config("llm_override_threshold", 0.15)
        )

    def run(self, data, symbol, context=None):
        # 1. 规则层
        rule_out = super().run(data, symbol, context)

        if not self.use_llm_decision or self._llm is None:
            return rule_out

        if rule_out.warnings and "missing" in str(rule_out.warnings):
            return rule_out

        # 2. 构造压缩 prompt
        market_features = (context or {}).get("market_features", {}) or {}
        user_prompt = self._build_prompt(rule_out, market_features, symbol)

        # 3. LLM 判断
        parsed = _call_llm_json(
            self._llm,
            system=ALPHA_SYSTEM_PROMPT,
            user=user_prompt,
        )

        if not parsed:
            rule_out.add_warning("llm_failed")
            return rule_out

        # 4. 融合
        return self._merge(rule_out, parsed)

    def _build_prompt(
        self,
        rule_out: AgentOutput,
        market_features: Dict[str, Any],
        symbol: str,
    ) -> str:
        signal_details = rule_out.features.get("signal_details", [])
        compact_signals = [
            {
                "type": s.get("type"),
                "direction": s.get("direction"),
                "score": s.get("normalized_score"),
            }
            for s in signal_details
        ]

        snapshot = {
            "symbol": symbol,
            "rule_action": rule_out.action.value,
            "rule_confidence": round(rule_out.confidence, 3),
            "regime": market_features.get("regime"),
            "adx": market_features.get("adx"),
            "atr_pct": market_features.get("atr_pct"),
            "news_sentiment": market_features.get("news_sentiment", 0.0),
            "news_event_type": market_features.get("news_event_type", "none"),
            "signals": compact_signals,
            "rule_reasons": rule_out.reasons[:5],
        }

        if self._memory is not None:
            regime = market_features.get("regime", "unknown")
            signal_types = [s.get("type") for s in rule_out.features.get("signal_details", [])]
            query = f"{regime} {signal_types[0] if signal_types else ''} 决策经验"

            lessons = self._memory.retrieve_rag(
               query=query,
               regime=regime,
               top_k=3,
            )
            snapshot["historical_lessons"] = [
              {"lesson": r["content"], "score": r["score"]}
              for r in lessons
           ]

        return json.dumps(snapshot, ensure_ascii=False, indent=2)


    def _merge(self, rule_out: AgentOutput, llm_parsed: Dict[str, Any]) -> AgentOutput:
        llm_action_str = str(llm_parsed.get("action", "hold")).lower()
        llm_conf = float(np.clip(float(llm_parsed.get("confidence", 0.0)), 0.0, 1.0))
        llm_reason = str(llm_parsed.get("reason", ""))

        try:
            llm_action = Action(llm_action_str)
        except ValueError:
            llm_action = Action.HOLD
            llm_conf = 0.0

        rule_action = rule_out.action
        rule_conf = rule_out.confidence

        if rule_action == llm_action and rule_action != Action.HOLD:
            merged_action = rule_action
            merged_conf = 0.4 * rule_conf + 0.6 * llm_conf
            merge_note = "rule+llm agree"
        elif rule_action == Action.HOLD and llm_action != Action.HOLD:
            merged_action = llm_action
            merged_conf = llm_conf * 0.7
            merge_note = "llm only"
        elif llm_action == Action.HOLD:
            merged_action = Action.HOLD
            merged_conf = max(rule_conf * 0.3, 0.0)
            merge_note = "llm veto"
        else:
            if llm_conf >= self.llm_override_threshold:
                merged_action = llm_action
                merged_conf = llm_conf * 0.6
                merge_note = "llm override"
            else:
                merged_action = Action.HOLD
                merged_conf = 0.0
                merge_note = "conflict"

        new_features = dict(rule_out.features)
        new_features.update({
            "llm_action": llm_action.value,
            "llm_confidence": round(llm_conf, 4),
            "llm_reason": llm_reason,
            "merge_note": merge_note,
        })

        return AgentOutput(
            agent_name=self.name,
            action=merged_action,
            confidence=round(float(np.clip(merged_conf, 0.0, 1.0)), 4),
            reasons=rule_out.reasons + [f"llm: {llm_reason} [{merge_note}]"],
            warnings=rule_out.warnings,
            features=new_features,
        ) 