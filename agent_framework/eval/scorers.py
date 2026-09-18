"""
常用打分函数。

每个打分函数签名: (actual: Any, expected: Any, meta: Dict) -> float  (0~1)
"""
from __future__ import annotations

import logging
import re
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)

ScorerFn = Callable[[Any, Any, Dict], float]


def exact_match(actual: Any, expected: Any, meta: Dict) -> float:
    """严格相等"""
    return 1.0 if str(actual).strip() == str(expected).strip() else 0.0


def contains_match(actual: Any, expected: Any, meta: Dict) -> float:
    """子串匹配（大小写不敏感）"""
    a = str(actual).lower()
    e = str(expected).lower()
    return 1.0 if e in a else 0.0


def numeric_close(
    actual: Any, expected: Any, meta: Dict
) -> float:
    """
    数值接近度打分。
    提取策略:
      1. 优先匹配 '=' / '＝' 后面的第一个数字 (如 "123 × 456 = 56088")
      2. 否则取文本里最后一个数字 (如 "结果是 42 哦")
      3. 都没有 → 0.0
    meta 可含 'tolerance'（默认 1e-6）和 'rel_tolerance'（默认 None）。
    返回 1.0（完全匹配）到 0.0（差得远），中间线性插值。
    """
    text = str(actual)

    # ── 提取数字 ──
    a: Optional[float] = None
    
    # 策略 0: 加粗的最终答案，如 **64**
    m = re.search(r"\*\*\s*(-?\d+\.?\d*)\s*\*\*", text)
    if m:
        try:
            a = float(m.group(1))
        except ValueError:
            a = None


     # 策略 1: 等号后的数字
    matches = re.findall(r"[=＝]\s*(-?\d+\.?\d*)", text)
    if matches:
               try:
                    a = float(matches[-1])   # 取最后一个等号后的数字
               except ValueError:
                    a = None

    # 策略 2: 文本中最后一个数字
    if a is None:
        nums = re.findall(r"-?\d+\.?\d*", text)
        if nums:
            try:
                a = float(nums[-1])
            except ValueError:
                a = None

    if a is None:
        return 0.0
    try:
        e = float(expected)
    except (ValueError, TypeError):
        return 0.0

    tol = float(meta.get("tolerance", 1e-6))
    rel_tol = meta.get("rel_tolerance")

    abs_err = abs(a - e)
    if abs_err <= tol:
        return 1.0

    if rel_tol:
        rel_err = abs_err / max(abs(e), 1e-9)
        if rel_err <= rel_tol:
            return 1.0
        # 相对误差在 [rel_tol, 2*rel_tol] 之间线性衰减
        return max(0.0, 1.0 - (rel_err - rel_tol) / rel_tol)

    # 没有相对容差：绝对误差在 [tol, 10*tol] 之间线性衰减
    return max(0.0, 1.0 - (abs_err - tol) / (9 * tol))


def tool_called(actual: Any, expected: Any, meta: Dict) -> float:
    """
    检查某个工具是否被调用过。
    meta: {'steps': [AgentStep...], 'tool_name': 'calculate'}
    """
    steps = meta.get("steps", [])
    tool_name = meta.get("tool_name") or expected
    for s in steps:
        action = getattr(s, "action", None)
        if action == tool_name:
            return 1.0
    return 0.0


def no_error(actual: Any, expected: Any, meta: Dict) -> float:
    """没有发生错误（success 全为 True）"""
    steps = meta.get("steps", [])
    if not steps:
        return 0.0
    return 1.0 if all(getattr(s, "success", True) for s in steps) else 0.0


def llm_judge(
    actual: Any, expected: Any, meta: Dict
) -> float:
    """
    LLM-as-judge。需要 meta['judge_llm']（一个 LLMAdapter 实例）。
    meta 可含 'criteria'（判断标准，默认通用）。
    """
    judge = meta.get("judge_llm")
    if judge is None:
        logger.warning("llm_judge: no judge_llm provided, returning 0.0")
        return 0.0

    criteria = meta.get(
        "criteria",
        "Rate how well the ACTUAL answer matches the EXPECTED answer, "
        "on a scale of 0 to 10. Output only the number.",
    )

    prompt = f"""{criteria}

EXPECTED: {expected}
ACTUAL: {actual}

Score (0-10):"""


    try:
        resp = judge.chat([{"role": "user", "content": prompt}])
        text = resp.get("content", "")
        m = re.search(r"\d+", text)
        if not m:
            return 0.0
        score = float(m.group())
        return max(0.0, min(1.0, score / 10.0))
    except Exception as e:
        logger.warning("llm_judge failed: %s", e)
        return 0.0


# ═══════════════════════════════════════════
# 新增:挑战类 scorer
# ═══════════════════════════════════════════

def graceful_error(actual: Any, expected: Any, meta: Optional[Dict] = None) -> float:
    """
    评估 Agent 面对错误时的表现:
      - 优雅报错(明确说"无法完成") → 1.0
      - 编造答案(幻觉)              → 0.0
      - 崩溃 / 异常未处理            → 0.0
    """
    meta = meta or {}
    text = str(actual).lower()

    # 明显崩溃 / 空答案
    if not text.strip():
        return 0.0

    # Agent 是否明确承认失败
    graceful_markers = [
        "无法", "不能", "失败", "错误", "不存在", "找不到",
        "cannot", "unable", "not found", "error", "failed",
        "抱歉", "sorry",
    ]
    has_marker = any(m in text for m in graceful_markers)

    # 是否编造内容(出现具体数字/路径但实际不存在)
    # 这里简化:有 graceful marker 且无"凭空数据" → 1.0
    if has_marker:
        return 1.0
    return 0.0


def no_hallucination(actual: Any, expected: Any, meta: Optional[Dict] = None) -> float:
    """
    检查答案里是否包含 forbidden 内容。
    meta['forbidden']: List[str] — 出现即 0 分
    meta['required']:  List[str] — 必须出现
    """
    meta = meta or {}
    text = str(actual).lower()

    for bad in meta.get("forbidden", []):
        if str(bad).lower() in text:
            return 0.0

    for good in meta.get("required", []):
        if str(good).lower() not in text:
            return 0.0

    return 1.0


def answer_in_range(actual: Any, expected: Any, meta: Optional[Dict] = None) -> float:
    """
    数值落在 [expected_lo, expected_hi] 区间即满分。
    meta['range']: (lo, hi)
    """
    meta = meta or {}
    lo, hi = meta.get("range", (expected, expected))
    nums = re.findall(r"-?\d+\.?\d*", str(actual))
    if not nums:
        return 0.0
    try:
        a = float(nums[-1])
    except ValueError:
        return 0.0
    return 1.0 if lo <= a <= hi else 0.0


__all__ = [
    "exact_match", "contains_match", "numeric_close",
    "tool_called", "no_error", "llm_judge",
    "graceful_error", "no_hallucination", "answer_in_range",  # 新增
    "weighted", "all_of", "any_of",
]

    