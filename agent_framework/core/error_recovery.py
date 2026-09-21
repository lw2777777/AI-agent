# core/error_recovery.py
"""
Error Recovery - 错误分类、降级、补偿

与 RetryPolicy / CircuitBreaker 的分工：
  RetryPolicy     → 处理"瞬时错误"（网络抖动、限流）
  CircuitBreaker  → 处理"持续性故障"（服务挂了）
  ErrorRecovery   → 处理"语义错误"（参数错、工具选错、模型答非所问）

三层关系:Retry 失败 → ErrorRecovery 决策 → 决定重试/降级/放弃
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class ErrorClass(str, Enum):
    """错误分类"""
    TRANSIENT = "transient"          # 瞬时：重试可解（网络、限流）
    INVALID_INPUT = "invalid_input"  # 输入错：需要修正参数
    TOOL_FAILURE = "tool_failure"    # 工具执行失败：可换工具
    MODEL_FAILURE = "model_failure"  # 模型输出异常：可换模型/换 prompt
    RESOURCE_EXHAUSTED = "resource"  # 预算耗尽：必须停止
    PERMISSION = "permission"        # 权限拒绝：不可恢复
    UNKNOWN = "unknown"


class RecoveryAction(str, Enum):
    RETRY = "retry"                  # 原样重试
    RETRY_WITH_HINT = "retry_hint"   # 带错误提示重试
    FALLBACK_TOOL = "fallback_tool"  # 换工具
    FALLBACK_MODEL = "fallback_model"# 换模型
    DEGRADE = "degrade"              # 降级返回部分结果
    ABORT = "abort"                  # 放弃


@dataclass
class RecoveryDecision:
    action: RecoveryAction
    reason: str
    hint: Optional[str] = None       # 给 LLM 的修正提示
    fallback_tool: Optional[str] = None
    fallback_model: Optional[str] = None
    max_attempts: int = 1


# ══════════════════════════════════════════════════
# 错误分类器
# ══════════════════════════════════════════════════
class ErrorClassifier:
    """把异常/错误字符串映射到 ErrorClass"""

    # 可扩展的规则表
    _PATTERNS = [
        (ErrorClass.TRANSIENT, [
            "timeout", "timed out", "connection reset", "connection refused",
            "rate limit", "429", "503", "502", "504", "temporarily unavailable",
            "network", "socket",
        ]),
        (ErrorClass.INVALID_INPUT, [
            "invalid argument", "validation error", "missing required",
            "schema", "typeerror", "valueerror", "keyerror",
        ]),
        (ErrorClass.TOOL_FAILURE, [
            "tool not found", "tool execution failed", "no such tool",
            "handler", "attributeerror",
        ]),
        (ErrorClass.MODEL_FAILURE, [
            "malformed", "json decode", "parse error", "empty response",
            "content filter", "context length",
        ]),
        (ErrorClass.PERMISSION, [
            "permission denied", "forbidden", "403", "unauthorized",
            "guardrail", "blocked by",
        ]),
        (ErrorClass.RESOURCE_EXHAUSTED, [
            "budget", "quota exceeded", "token limit", "out of memory",
        ]),
    ]

    @classmethod
    def classify(cls, error: Any) -> ErrorClass:
        text = f"{type(error).__name__}: {error}".lower()

        for err_cls, keywords in cls._PATTERNS:
            if any(k in text for k in keywords):
                return err_cls

        # 按异常类型兜底
        if isinstance(error, (TimeoutError, ConnectionError)):
            return ErrorClass.TRANSIENT
        if isinstance(error, (ValueError, TypeError, KeyError)):
            return ErrorClass.INVALID_INPUT
        if isinstance(error, PermissionError):
            return ErrorClass.PERMISSION

        return ErrorClass.UNKNOWN


# ══════════════════════════════════════════════════
# 恢复策略
# ══════════════════════════════════════════════════
class ErrorRecovery:
    """
    错误恢复决策器。

    用法：
        recovery = ErrorRecovery(fallback_models=["gpt-4o-mini"])
        decision = recovery.decide(error, context)
        if decision.action == RecoveryAction.ABORT:
            raise
    """

    def __init__(
        self,
        fallback_models: Optional[List[str]] = None,
        fallback_tools: Optional[Dict[str, str]] = None,
        on_degrade: Optional[Callable[[str], str]] = None,
    ):
        """
        Args:
            fallback_models: 降级模型列表（按优先级）
            fallback_tools: 工具降级映射 {'primary_tool': 'backup_tool'}
            on_degrade: 降级时生成部分答案的函数
        """
        self.fallback_models = fallback_models or []
        self.fallback_tools = fallback_tools or {}
        self.on_degrade = on_degrade

        # 每个 error_class 的尝试次数
        self._attempts: Dict[str, int] = {}

    def decide(
        self,
        error: Any,
        context: Optional[Dict[str, Any]] = None,
    ) -> RecoveryDecision:
        """
        Args:
            error: 异常对象或错误字符串
            context: {tool_name, model, step_index, partial_answer, ...}
        """
        context = context or {}
        err_cls = ErrorClassifier.classify(error)
        key = err_cls.value
        self._attempts[key] = self._attempts.get(key, 0) + 1
        n = self._attempts[key]

        if err_cls == ErrorClass.TRANSIENT:
            if n <= 3:
                return RecoveryDecision(
                    action=RecoveryAction.RETRY,
                    reason=f"transient error, attempt {n}/3",
                )
            return RecoveryDecision(
                action=RecoveryAction.DEGRADE,
                reason="transient error persists, degrading",
            )

        if err_cls == ErrorClass.INVALID_INPUT:
            if n <= 2:
                return RecoveryDecision(
                    action=RecoveryAction.RETRY_WITH_HINT,
                    reason="invalid input, retry with hint",
                    hint=(
                        f"Your previous call failed with: {error}. "
                        f"Please check the argument schema and try again with corrected values."
                    ),
                )
            return RecoveryDecision(
                action=RecoveryAction.DEGRADE,
                reason="invalid input persists",
            )

        if err_cls == ErrorClass.TOOL_FAILURE:
            tool = context.get("tool_name")
            fallback = self.fallback_tools.get(tool)
            if fallback and n <= 1:
                return RecoveryDecision(
                    action=RecoveryAction.FALLBACK_TOOL,
                    reason=f"tool '{tool}' failed, switching to '{fallback}'",
                    fallback_tool=fallback,
                )
            return RecoveryDecision(
                action=RecoveryAction.DEGRADE,
                reason="tool failure, no fallback available",
            )

        if err_cls == ErrorClass.MODEL_FAILURE:
            if self.fallback_models and n <= len(self.fallback_models):
                return RecoveryDecision(
                    action=RecoveryAction.FALLBACK_MODEL,
                    reason=f"model failure, switching to {self.fallback_models[n-1]}",
                    fallback_model=self.fallback_models[n - 1],
                )
            if n <= 2:
                return RecoveryDecision(
                    action=RecoveryAction.RETRY,
                    reason="model failure, plain retry",
                )
            return RecoveryDecision(
                action=RecoveryAction.DEGRADE,
                reason="model failure persists",
            )

        if err_cls == ErrorClass.RESOURCE_EXHAUSTED:
            return RecoveryDecision(
                action=RecoveryAction.ABORT,
                reason="resource exhausted, cannot continue",
            )

        if err_cls == ErrorClass.PERMISSION:
            return RecoveryDecision(
                action=RecoveryAction.ABORT,
                reason="permission denied, not recoverable",
            )

        # UNKNOWN
        if n <= 1:
            return RecoveryDecision(
                action=RecoveryAction.RETRY,
                reason="unknown error, one retry",
            )
        return RecoveryDecision(
            action=RecoveryAction.DEGRADE,
            reason="unknown error persists",
        )

    def reset(self) -> None:
        """新一轮 run 开始时调用"""
        self._attempts.clear()

    def degrade(self, partial: str) -> str:
        """生成降级答案"""
        if self.on_degrade:
            return self.on_degrade(partial)
        return f"[degraded] {partial}" if partial else "[degraded] No result available."