"""
baseline_react.py — 朴素 ReAct baseline

设计（作为对照）:
  - 单 Agent,无多 Agent 分工
  - 无 Reflexion、无 Memory
  - 直接把最近 N 根 OHLCV 序列化成文本给 LLM
  - 每根 K 线一次调用，无特征压缩

这个 baseline 代表"不做任何工程优化的 LLM 交易 Agent",
用来对比多 Agent 框架的 token 效率和决策质量。
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .base_agent import Action
from agent_framework.core.llm_adapter import LLMAdapter

logger = logging.getLogger(__name__)


BASELINE_REACT_SYSTEM_PROMPT = """你是一个量化交易 Agent,使用 ReAct 风格思考。

你会看到某只股票最近 N 根 K 线的完整数(open/high/low/close/volume)。
请分析后决定:BUY / SELL / HOLD,并给出置信度。

输出严格 JSON:
{
  "thought": "你的简短分析(1-2 句）",
  "action": "buy|sell|hold",
  "confidence": 0.0 到 1.0 的浮点数,
  "reason": "一句话中文说明"
}
只输出 JSON,不要 markdown 代码块。"""


@dataclass
class BaselineDecision:
    action: str = "hold"
    confidence: float = 0.0
    thought: str = ""
    reason: str = ""
    ok: bool = False


class NaiveReActBaseline:
    """
    朴素 ReAct:直接把原始 K 线序列化给 LLM。
    无 Market / Risk / Execution / Reflexion / Memory。
    """

    def __init__(
        self,
        llm: LLMAdapter,
        config: Optional[Dict[str, Any]] = None,
    ):
        self._llm = llm
        cfg = config or {}
        self.lookback_bars = int(cfg.get("lookback_bars", 20))
        self.min_data_points = int(cfg.get("min_data_points", 30))

    def run(
        self,
        data: pd.DataFrame,
        symbol: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> BaselineDecision:
        if data is None or len(data) < self.min_data_points:
            return BaselineDecision(action="hold", ok=False)

        # 1. 序列化最近 N 根 K 线
        window = data.tail(self.lookback_bars)
        user_prompt = self._serialize(window, symbol)

        # 2. LLM
        messages = [
            {"role": "system", "content": BASELINE_REACT_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        try:
            resp = self._llm.chat(messages, temperature=0.0)
        except Exception as e:
            logger.warning("baseline LLM 失败: %s", e)
            return BaselineDecision(action="hold", ok=False)

        parsed = self._parse(resp.get("content"))
        if not parsed:
            return BaselineDecision(action="hold", ok=False)

        action_str = str(parsed.get("action", "hold")).lower()
        if action_str not in ("buy", "sell", "hold"):
            action_str = "hold"

        return BaselineDecision(
            action=action_str,
            confidence=float(np.clip(float(parsed.get("confidence", 0.0)), 0.0, 1.0)),
            thought=str(parsed.get("thought", "")),
            reason=str(parsed.get("reason", "")),
            ok=True,
        )

    def _serialize(self, df: pd.DataFrame, symbol: str) -> str:
        """把 K 线序列化成紧凑文本——这是 baseline 的 token 消耗来源"""
        lines = [f"股票：{symbol}", f"最近 {len(df)} 根 K 线："]
        for idx, row in df.iterrows():
            date_str = idx.strftime("%Y-%m-%d") if hasattr(idx, "strftime") else str(idx)
            lines.append(
                f"{date_str} O={row['open']:.2f} H={row['high']:.2f} "
                f"L={row['low']:.2f} C={row['close']:.2f} V={int(row['volume'])}"
            )
        return "\n".join(lines)

    @staticmethod
    def _parse(content: Optional[str]) -> Dict[str, Any]:
        if not content:
            return {}
        content = content.strip()
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass
        start, end = content.find("{"), content.rfind("}")
        if start != -1 and end > start:
            try:
                return json.loads(content[start:end + 1])
            except json.JSONDecodeError:
                pass
        return {}