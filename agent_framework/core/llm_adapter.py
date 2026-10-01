"""
LLM Adapter - 统一不同 LLM Provider 的调用接口

目标：让 CoreAgent 不需要知道底层是 OpenAI 还是 Anthropic。

补丁：新增 usage 累计（线程安全），供评测脚本统计 token 消耗。
"""
from __future__ import annotations

import json
import logging
import threading
from typing import Any, Dict, List, Optional

from openai import OpenAI

logger = logging.getLogger(__name__)


class LLMAdapter:
    """统一 LLM 接口"""

    def __init__(self, provider: str, api_key: str, model: str, **kwargs):
        self.provider = provider
        self.model = model
        self._client = None

        if provider == "openai":
            self._client = OpenAI(api_key=api_key)
        elif provider == "anthropic":
            from anthropic import Anthropic
            self._client = Anthropic(api_key=api_key)
        elif provider == "deepseek":
            self._client = OpenAI(
                api_key=api_key, base_url="https://api.deepseek.com"
            )
        else:
            raise ValueError(f"Unknown provider: {provider}")

        # ── usage 累计（线程安全）──
        self._usage_lock = threading.Lock()
        self._usage: Dict[str, int] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "call_count": 0,
        }

    # ══════════════════════════════════════
    # 主入口
    # ══════════════════════════════════════
    def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: str = "auto",
        temperature: float = 0.0,
    ) -> Dict[str, Any]:
        """
        统一聊天接口

        Returns:
            {
                "content": str | None,
                "tool_calls": [{"id", "name", "arguments"}],
                "finish_reason": str,
                "usage": {"prompt_tokens", "completion_tokens"},
            }
        """
        if self.provider in ("deepseek", "openai"):
            return self._chat_openai(messages, tools, tool_choice, temperature)
        elif self.provider == "anthropic":
            return self._chat_anthropic(messages, tools, tool_choice, temperature)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    # ══════════════════════════════════════
    # OpenAI / DeepSeek
    # ══════════════════════════════════════
    def _chat_openai(self, messages, tools, tool_choice, temperature):
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice

        resp = self._client.chat.completions.create(**kwargs)
        msg = resp.choices[0].message

        tool_calls = []
        for tc in (msg.tool_calls or []):
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                logger.warning(
                    "Invalid tool arguments: %s", tc.function.arguments
                )
                args = {"_raw": tc.function.arguments}

            tool_calls.append({
                "id": tc.id,
                "name": tc.function.name,
                "arguments": args,
            })

        # ── usage 累计 ──
        self._accumulate(
            prompt=getattr(resp.usage, "prompt_tokens", 0),
            completion=getattr(resp.usage, "completion_tokens", 0),
        )

        return {
            "content": msg.content,
            "tool_calls": tool_calls,
            "finish_reason": resp.choices[0].finish_reason,
            "usage": {
                "prompt_tokens": getattr(resp.usage, "prompt_tokens", 0),
                "completion_tokens": getattr(resp.usage, "completion_tokens", 0),
            },
        }

    # ══════════════════════════════════════
    # Anthropic
    # ══════════════════════════════════════
    def _chat_anthropic(self, messages, tools, tool_choice, temperature):
        system = None
        filtered_messages = []
        for m in messages:
            if m["role"] == "system":
                system = m["content"]
            else:
                filtered_messages.append(m)

        kwargs = {
            "model": self.model,
            "messages": filtered_messages,
            "max_tokens": 4096,
            "temperature": temperature,
        }
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = [
                {
                    "name": t["function"]["name"],
                    "description": t["function"]["description"],
                    "input_schema": t["function"]["parameters"],
                }
                for t in tools
            ]

        resp = self._client.messages.create(**kwargs)

        content_text = ""
        tool_calls = []
        for block in resp.content:
            if block.type == "text":
                content_text += block.text
            elif block.type == "tool_use":
                tool_calls.append({
                    "id": block.id,
                    "name": block.name,
                    "arguments": block.input,
                })

        prompt_tokens = getattr(resp.usage, "input_tokens", 0)
        completion_tokens = getattr(resp.usage, "output_tokens", 0)

        # ── usage 累计 ──
        self._accumulate(prompt=prompt_tokens, completion=completion_tokens)

        return {
            "content": content_text or None,
            "tool_calls": tool_calls,
            "finish_reason": resp.stop_reason,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            },
        }

    # ══════════════════════════════════════
    # usage 工具
    # ══════════════════════════════════════
    def _accumulate(self, prompt: int, completion: int) -> None:
        with self._usage_lock:
            self._usage["prompt_tokens"] += int(prompt)
            self._usage["completion_tokens"] += int(completion)
            self._usage["total_tokens"] += int(prompt) + int(completion)
            self._usage["call_count"] += 1

    def get_usage(self) -> Dict[str, int]:
        """返回累计 token 消耗（供评测统计）"""
        with self._usage_lock:
            return dict(self._usage)

    def reset_usage(self) -> None:
        """重置累计（每个任务开始前调用）"""
        with self._usage_lock:
            self._usage = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "call_count": 0,
            }