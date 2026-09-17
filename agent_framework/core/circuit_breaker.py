"""
Circuit Breaker - 熔断器
连续失败达到阈值后，中断执行，返回部分结果。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class CircuitState(str, Enum):
    CLOSED = "closed"      # 正常
    OPEN = "open"          # 熔断中
    HALF_OPEN = "half_open"  # 试探恢复


@dataclass
class CircuitBreaker:
    """
    连续失败熔断器。

    语义：
      - 每次成功调用 reset()，失败计数归零；
      - 每次失败调用 record_failure()，计数 +1；
      - 计数达到 threshold 时，state 变为 OPEN；
      - OPEN 状态下的调用应被拒绝。
    """
    threshold: int = 5
    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    total_failures: int = 0
    total_successes: int = 0

    def record_success(self) -> None:
        self.failure_count = 0
        self.total_successes += 1
        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.CLOSED
            logger.info("circuit breaker closed (recovered)")

    def record_failure(self) -> None:
        self.failure_count += 1
        self.total_failures += 1
        if self.failure_count >= self.threshold:
            self.state = CircuitState.OPEN
            logger.warning(
                "circuit breaker OPEN after %d consecutive failures",
                self.failure_count,
            )

    def is_open(self) -> bool:
        return self.state == CircuitState.OPEN

    def reset(self) -> None:
        self.state = CircuitState.CLOSED
        self.failure_count = 0

    def snapshot(self) -> Dict[str, Any]:
        return {
            "state": self.state.value,
            "failure_count": self.failure_count,
            "total_failures": self.total_failures,
            "total_successes": self.total_successes,
        }