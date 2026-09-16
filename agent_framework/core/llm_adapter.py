"""
LLM Adapter - 统一不同 LLM Provider 的调用接口

目标：让 CoreAgent 不需要知道底层是 OpenAI 还是 Anthropic。
"""
from __future__ import annotations

import json
import logging
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
            self._client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        else:
            raise ValueError(f"Unknown provider: {provider}")

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
        if self.provider == "deepseek":
            return self._chat_openai(messages, tools, tool_choice, temperature)
        elif self.provider == "openai":
            return self._chat_openai(messages, tools, tool_choice, temperature)
        elif self.provider == "anthropic":
            return self._chat_anthropic(messages, tools, tool_choice, temperature)

    # ────────────────────────────────────
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
                logger.warning(f"Invalid tool arguments: {tc.function.arguments}")
                args = {"_raw": tc.function.arguments}

            tool_calls.append({
                "id": tc.id,
                "name": tc.function.name,
                "arguments": args,
            })

        return {
            "content": msg.content,
            "tool_calls": tool_calls,
            "finish_reason": resp.choices[0].finish_reason,
            "usage": {
                "prompt_tokens": resp.usage.prompt_tokens,
                "completion_tokens": resp.usage.completion_tokens,
            },
        }

    def _chat_anthropic(self, messages, tools, tool_choice, temperature):
        # Anthropic 的格式和 OpenAI 略有不同
        # system message 单独传
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
            # 转换 OpenAI schema → Anthropic schema
            kwargs["tools"] = [
                {
                    "name": t["function"]["name"],
                    "description": t["function"]["description"],
                    "input_schema": t["function"]["parameters"],
                }
                for t in tools
            ]

        resp = self._client.messages.create(**kwargs)

        # 解析 content blocks
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

        return {
            "content": content_text or None,
            "tool_calls": tool_calls,
            "finish_reason": resp.stop_reason,
            "usage": {
                "prompt_tokens": resp.usage.input_tokens,
                "completion_tokens": resp.usage.output_tokens,
            },
        }