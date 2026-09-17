"""
Retry policy - 指数退避重试
"""
from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass
from typing import Callable, TypeVar, Optional, Tuple, Type

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class RetryPolicy:
    """重试策略"""
    max_attempts: int = 3
    base_delay: float = 1.0          # 秒
    max_delay: float = 30.0
    exponential_base: float = 2.0
    jitter: bool = True              # 加随机抖动，避免惊群
    retryable_exceptions: Tuple[Type[Exception], ...] = (Exception,)

    def compute_delay(self, attempt: int) -> float:
        """计算第 attempt 次重试前的等待时间（attempt 从 0 开始）"""
        delay = self.base_delay * (self.exponential_base ** attempt)
        delay = min(delay, self.max_delay)
        if self.jitter:
            # 加 ±25% 抖动
            delay *= random.uniform(0.75, 1.25)
        return delay


def retry_call(
    fn: Callable[[], T],
    policy: RetryPolicy,
    on_retry: Optional[Callable[[int, Exception, float], None]] = None,
) -> T:
    """
    带重试的调用。

    Args:
        fn: 无参可调用对象
        policy: 重试策略
        on_retry: 回调 (attempt, exception, delay)

    Returns:
        fn() 的返回值

    Raises:
        最后一次尝试的异常
    """
    last_exc: Optional[Exception] = None

    for attempt in range(policy.max_attempts):
        try:
            return fn()
        except policy.retryable_exceptions as e:
            last_exc = e
            is_last = (attempt == policy.max_attempts - 1)
            if is_last:
                logger.error(
                    "retry_call exhausted after %d attempts: %s",
                    policy.max_attempts, e,
                )
                raise

            delay = policy.compute_delay(attempt)
            logger.warning(
                "retry_call attempt %d/%d failed (%s), retrying in %.2fs",
                attempt + 1, policy.max_attempts, type(e).__name__, delay,
            )
            if on_retry:
                on_retry(attempt, e, delay)
            time.sleep(delay)

    # 理论上不可达
    raise last_exc if last_exc else RuntimeError("retry_call unreachable")