"""
LLM Adapter - 统一不同 LLM Provider 的调用接口

补丁：
  1. usage 累计（线程安全）
  2. 并发控制（Semaphore，类级共享）
  3. 速率限制（滑动窗口，类级共享）
  4. 重试 + 指数退避（针对 429 / 5xx）
"""
from __future__ import annotations

import json
import logging
import threading
import time
from collections import deque
from typing import Any, Dict, List, Optional

from openai import OpenAI

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════
# 滑动窗口限流器（线程安全）
# ══════════════════════════════════════════════════════════
class _SlidingWindowLimiter:
    """
    在 window_sec 秒内最多允许 max_calls 次调用。
    超限时阻塞等待。
    """
    def __init__(self, max_calls: int, window_sec: float = 60.0):
        self.max_calls = max(1, int(max_calls))
        self.window_sec = float(window_sec)
        self._timestamps: deque = deque()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        while True:
            wait = 0.0
            with self._lock:
                now = time.monotonic()
                # 清理窗口外的
                while self._timestamps and now - self._timestamps[0] >= self.window_sec:
                    self._timestamps.popleft()

                if len(self._timestamps) < self.max_calls:
                    self._timestamps.append(now)
                    return
                # 需要等待最老的那个过期
                wait = self.window_sec - (now - self._timestamps[0])

            if wait > 0:
                time.sleep(min(wait, 1.0))


# ══════════════════════════════════════════════════════════
# 类级共享状态（所有实例共用）
# ══════════════════════════════════════════════════════════
_CLASS_LOCK = threading.Lock()
_CLASS_SEMAPHORE: Optional[threading.Semaphore] = None
_CLASS_RPM_LIMITER: Optional[_SlidingWindowLimiter] = None
_CLASS_MAX_CONCURRENCY: Optional[int] = None
_CLASS_RPM: Optional[int] = None


def _get_class_resources(max_concurrency: int, rpm: int):
    """懒初始化类级共享资源；第一次调用时的参数生效"""
    global _CLASS_SEMAPHORE, _CLASS_RPM_LIMITER
    global _CLASS_MAX_CONCURRENCY, _CLASS_RPM

    with _CLASS_LOCK:
        if _CLASS_SEMAPHORE is None:
            _CLASS_SEMAPHORE = threading.Semaphore(max_concurrency)
            _CLASS_MAX_CONCURRENCY = max_concurrency
            logger.info("LLMAdapter 全局并发上限 = %d", max_concurrency)

        if _CLASS_RPM_LIMITER is None and rpm > 0:
            _CLASS_RPM_LIMITER = _SlidingWindowLimiter(max_calls=rpm, window_sec=60.0)
            _CLASS_RPM = rpm
            logger.info("LLMAdapter 全局 RPM 上限 = %d", rpm)

    return _CLASS_SEMAPHORE, _CLASS_RPM_LIMITER


def reset_class_resources() -> None:
    """测试用：重置全局限流状态"""
    global _CLASS_SEMAPHORE, _CLASS_RPM_LIMITER
    global _CLASS_MAX_CONCURRENCY, _CLASS_RPM
    with _CLASS_LOCK:
        _CLASS_SEMAPHORE = None
        _CLASS_RPM_LIMITER = None
        _CLASS_MAX_CONCURRENCY = None
        _CLASS_RPM = None


# ══════════════════════════════════════════════════════════
# LLMAdapter
# ══════════════════════════════════════════════════════════
class LLMAdapter:
    """统一 LLM 接口（线程安全 + 限流 + 重试）"""

    def __init__(
        self,
        provider: str,
        api_key: str,
        model: str,
        max_concurrency: int = 8,
        rpm: int = 300,
        max_retries: int = 3,
        retry_backoff: float = 1.5,
        request_timeout: float = 60.0,
        **kwargs,
    ):
        self.provider = provider
        self.model = model
        self.max_retries = int(max_retries)
        self.retry_backoff = float(retry_backoff)
        self._client = None

        if provider == "openai":
            self._client = OpenAI(api_key=api_key, timeout=request_timeout)
        elif provider == "anthropic":
            from anthropic import Anthropic
            self._client = Anthropic(api_key=api_key, timeout=request_timeout)
        elif provider == "deepseek":
            self._client = OpenAI(
                api_key=api_key,
                base_url="https://api.deepseek.com",
                timeout=request_timeout,
            )
        else:
            raise ValueError(f"Unknown provider: {provider}")

        # ── usage 累计（实例级）──
        self._usage_lock = threading.Lock()
        self._usage: Dict[str, int] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "call_count": 0,
            "retry_count": 0,      # ← 新增：重试次数统计
        }

        # ── 并发控制（类级共享）──
        self._sem, self._rpm_limiter = _get_class_resources(
            max_concurrency=max_concurrency, rpm=rpm,
        )

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
        # ── 限流闸门（两层）──
        if self._rpm_limiter is not None:
            self._rpm_limiter.acquire()      # 速率限制

        with self._sem:                       # 并发限制
            return self._chat_with_retry(
                messages, tools, tool_choice, temperature,
            )

    def _chat_with_retry(self, messages, tools, tool_choice, temperature):
        last_exc: Optional[Exception] = None

        for attempt in range(self.max_retries):
            try:
                if self.provider in ("deepseek", "openai"):
                    return self._chat_openai(messages, tools, tool_choice, temperature)
                elif self.provider == "anthropic":
                    return self._chat_anthropic(messages, tools, tool_choice, temperature)
            except Exception as e:
                last_exc = e
                # 判断是否值得重试
                if not self._is_retryable(e) or attempt == self.max_retries - 1:
                    raise

                wait = self.retry_backoff ** attempt
                logger.warning(
                    "LLM 调用失败（attempt %d/%d），%.1fs 后重试: %s",
                    attempt + 1, self.max_retries, wait, e,
                )
                with self._usage_lock:
                    self._usage["retry_count"] += 1
                time.sleep(wait)

        if last_exc:
            raise last_exc
        raise RuntimeError("unreachable")

    @staticmethod
    def _is_retryable(exc: Exception) -> bool:
        """429 / 5xx / 网络超时可重试；4xx 参数错不可重试"""
        msg = str(exc).lower()
        if "rate" in msg or "429" in msg:
            return True
        if "timeout" in msg or "connection" in msg:
            return True
        if any(f" {code}" in msg or f"({code})" in msg for code in (500, 502, 503, 504)):
            return True
        return False

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
                logger.warning("Invalid tool arguments: %s", tc.function.arguments)
                args = {"_raw": tc.function.arguments}

            tool_calls.append({
                "id": tc.id,
                "name": tc.function.name,
                "arguments": args,
            })

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
    # usage
    # ══════════════════════════════════════
    def _accumulate(self, prompt: int, completion: int) -> None:
        with self._usage_lock:
            self._usage["prompt_tokens"] += int(prompt)
            self._usage["completion_tokens"] += int(completion)
            self._usage["total_tokens"] += int(prompt) + int(completion)
            self._usage["call_count"] += 1

    def get_usage(self) -> Dict[str, int]:
        with self._usage_lock:
            return dict(self._usage)

    def reset_usage(self) -> None:
        with self._usage_lock:
            self._usage = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "call_count": 0,
                "retry_count": 0,
            }