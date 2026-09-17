"""
Context Manager - 上下文压缩

职责：
  1. 检查 messages 是否超过预算
  2. 超预算时压缩：
     - 保留 system
     - 把旧历史摘要成一段
     - 保留最近 N 轮完整对话
  3. 工具结果裁剪：长输出截断 + 保留首尾
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .token_counter import estimate_messages_tokens, estimate_message_tokens

logger = logging.getLogger(__name__)


# 摘要提示词：要求 LLM 输出"信息密集、无冗余"的摘要
SUMMARIZE_PROMPT = """You are summarizing an ongoing agent task for context compression.

## Task
{task}

## Conversation So Far
{history}

## Instructions
Summarize the above in a dense, information-preserving way:
1. What has been tried and what worked/failed
2. Key facts, values, and observations discovered
3. Current state: what's been done, what's still pending
4. Any important errors, constraints, or user requirements

Output a concise summary (max 500 words). Do NOT include filler like "The agent then...". 
Just the essential information for continuing the task.
"""


@dataclass
class CompressionResult:
    """压缩结果"""
    messages: List[Dict[str, Any]]
    compressed: bool
    original_tokens: int
    final_tokens: int
    summary_tokens: int = 0
    dropped_messages: int = 0
    truncated_tool_results: int = 0


class ContextManager:
    """
    上下文管理器

    用法：
        cm = ContextManager(max_tokens=8000, keep_recent=4, llm=llm)
        result = cm.compress(messages, task=query)
        messages = result.messages
    """

    def __init__(
        self,
        max_tokens: int = 8000,
        keep_recent: int = 4,          # 保留最近 N 条 message
        summary_max_tokens: int = 800,  # 摘要本身不超过这个
        tool_result_max_chars: int = 2000,  # 单条 tool 结果最长
        tool_result_head_chars: int = 800,  # 截断时保留头部
        tool_result_tail_chars: int = 400,  # 截断时保留尾部
        llm: Optional[Any] = None,
    ):
        self.max_tokens = max_tokens
        self.keep_recent = keep_recent
        self.summary_max_tokens = summary_max_tokens
        self.tool_result_max_chars = tool_result_max_chars
        self.tool_result_head_chars = tool_result_head_chars
        self.tool_result_tail_chars = tool_result_tail_chars
        self.llm = llm

        # 缓存的摘要（避免每次都重新生成）
        self._cached_summary: Optional[str] = None

    # ══════════════════════════════════════
    # 主入口
    # ══════════════════════════════════════

    def compress(
        self,
        messages: List[Dict[str, Any]],
        task: str = "",
    ) -> CompressionResult:
        """
        压缩 messages。

        Args:
            messages: 完整消息列表
            task: 原始任务描述（用于摘要 prompt）

        Returns:
            CompressionResult
        """
        original_tokens = estimate_messages_tokens(messages)

        # ── Step 1：先做工具结果裁剪（低成本）──
        messages, truncated = self._truncate_tool_results(messages)
        after_truncate_tokens = estimate_messages_tokens(messages)

        # ── Step 2：还在预算内？直接返回 ──
        if after_truncate_tokens <= self.max_tokens:
            return CompressionResult(
                messages=messages,
                compressed=False,
                original_tokens=original_tokens,
                final_tokens=after_truncate_tokens,
                truncated_tool_results=truncated,
            )

        # ── Step 3：超预算 → 摘要压缩 ──
        system_msg = messages[0]  # 一定是 system
        rest = messages[1:]

        # 保留最近 keep_recent 条
        recent = rest[-self.keep_recent:]
        old = rest[:-self.keep_recent]

        if not old:
            # 没东西可压缩，只能硬截断 tool 结果
            logger.warning("over budget but nothing to compress")
            return CompressionResult(
                messages=messages,
                compressed=False,
                original_tokens=original_tokens,
                final_tokens=after_truncate_tokens,
                truncated_tool_results=truncated,
            )

        # 生成摘要
        summary = self._generate_summary(old, task)

        # 重组 messages
        summary_msg = {
            "role": "user",
            "content": (
                f"[Prior context summary]\n{summary}\n\n"
                f"[End summary — {len(old)} earlier messages compressed]"
            ),
        }

        new_messages = [system_msg, summary_msg] + recent

        final_tokens = estimate_messages_tokens(new_messages)

        logger.info(
            "context compressed: %d msgs / %d tokens → %d msgs / %d tokens "
            "(summary=%d tokens, dropped=%d msgs)",
            len(messages), original_tokens,
            len(new_messages), final_tokens,
            estimate_message_tokens(summary_msg), len(old),
        )

        return CompressionResult(
            messages=new_messages,
            compressed=True,
            original_tokens=original_tokens,
            final_tokens=final_tokens,
            summary_tokens=estimate_message_tokens(summary_msg),
            dropped_messages=len(old),
            truncated_tool_results=truncated,
        )

    # ══════════════════════════════════════
    # 工具结果裁剪
    # ══════════════════════════════════════

    def _truncate_tool_results(
        self, messages: List[Dict[str, Any]]
    ) -> tuple[List[Dict[str, Any]], int]:
        """
        裁剪过长的工具结果。
        保留头 + 尾，中间用 "... [truncated N chars] ..." 替代。
        """
        out = []
        truncated = 0

        for m in messages:
            if m.get("role") != "tool":
                out.append(m)
                continue

            content = m.get("content", "")
            if not isinstance(content, str) or len(content) <= self.tool_result_max_chars:
                out.append(m)
                continue

            head = content[: self.tool_result_head_chars]
            tail = content[-self.tool_result_tail_chars:]
            omitted = len(content) - len(head) - len(tail)

            new_content = (
                f"{head}\n\n"
                f"... [truncated {omitted} chars] ...\n\n"
                f"{tail}"
            )
            out.append({**m, "content": new_content})
            truncated += 1

        return out, truncated

    # ══════════════════════════════════════
    # 摘要生成
    # ══════════════════════════════════════

    def _generate_summary(
        self, old_messages: List[Dict[str, Any]], task: str
    ) -> str:
        """用 LLM 生成摘要"""
        if self.llm is None:
            # 没 LLM，用朴素截断
            return self._fallback_summary(old_messages)

        # 把旧消息格式化成文本
        history_text = self._format_messages_for_summary(old_messages)

        prompt = SUMMARIZE_PROMPT.format(
            task=task or "(unknown task)",
            history=history_text,
        )

        try:
            resp = self.llm.chat(
                messages=[{"role": "user", "content": prompt}],
                tools=None,
                tool_choice=None,
            )
            summary = resp.get("content", "").strip()
            if not summary:
                return self._fallback_summary(old_messages)
            return summary
        except Exception:
            logger.exception("summary generation failed, using fallback")
            return self._fallback_summary(old_messages)

    @staticmethod
    def _format_messages_for_summary(
        messages: List[Dict[str, Any]]
    ) -> str:
        """把 messages 格式化成摘要 prompt 的输入"""
        lines = []
        for m in messages:
            role = m.get("role", "?")
            content = m.get("content", "")

            if role == "tool":
                lines.append(f"[Tool Result] {str(content)[:500]}")
            elif m.get("tool_calls"):
                for tc in m["tool_calls"]:
                    fn = tc.get("function", {})
                    lines.append(
                        f"[Assistant calls {fn.get('name')}] {fn.get('arguments', '')[:200]}"
                    )
            else:
                lines.append(f"[{role}] {str(content)[:500]}")

        return "\n".join(lines)

    @staticmethod
    def _fallback_summary(messages: List[Dict[str, Any]]) -> str:
        """无 LLM 时的朴素摘要：列关键动作"""
        actions = []
        for m in messages:
            if m.get("tool_calls"):
                for tc in m["tool_calls"]:
                    fn = tc.get("function", {})
                    actions.append(f"- called {fn.get('name')}")
            elif m.get("role") == "tool":
                content = str(m.get("content", ""))[:100]
                actions.append(f"- got result: {content}")

        if not actions:
            return "(no notable actions in prior context)"

        return "Prior actions:\n" + "\n".join(actions[-20:])

    # ══════════════════════════════════════
    # 查询
    # ══════════════════════════════════════

    def estimate_tokens(self, messages: List[Dict[str, Any]]) -> int:
        return estimate_messages_tokens(messages)

    def should_compress(self, messages: List[Dict[str, Any]]) -> bool:
        return self.estimate_tokens(messages) > self.max_tokens