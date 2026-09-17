"""
内置测试集。

这些任务覆盖 harness 的 4 个核心能力:
  1. 单工具调用
  2. 多步推理
  3. 工具失败恢复
  4. 无工具对话
"""
from __future__ import annotations

from .evaluator import EvalSuite, EvalTask
from .scorers import (
    exact_match,
    contains_match,
    numeric_close,
    tool_called,
    no_error,
)


def suite_basic_math() -> EvalSuite:
    """基础算术：测试单工具调用 + 数值准确"""
    suite = EvalSuite(name="basic_math", description="单工具调用 + 数值")

    suite.add(EvalTask(
        id="math_1",
        query="计算 123 * 456",
        expected=56088,
        scorer=numeric_close,
        meta={"tolerance": 0.01},
        tags=["tool_use", "math"],
    ))
    suite.add(EvalTask(
        id="math_2",
        query="计算 (10 + 20) * 3",
        expected=90,
        scorer=numeric_close,
        meta={"tolerance": 0.01},
        tags=["tool_use", "math"],
    ))
    suite.add(EvalTask(
        id="math_3",
        query="计算 2 的 10 次方",
        expected=1024,
        scorer=numeric_close,
        meta={"tolerance": 0.01},
        tags=["tool_use", "math"],
    ))
    return suite


def suite_tool_use() -> EvalSuite:
    """工具调用正确性"""
    suite = EvalSuite(name="tool_use", description="工具是否被正确调用")

    suite.add(EvalTask(
        id="tool_1",
        query="计算 5 + 5",
        expected="calculate",
        scorer=tool_called,
        meta={"tool_name": "calculate"},
        tags=["tool_use"],
    ))
    suite.add(EvalTask(
        id="tool_2",
        query="格式化这段 JSON: {\"a\":1,\"b\":2}",
        expected="format_json",
        scorer=tool_called,
        meta={"tool_name": "format_json"},
        tags=["tool_use"],
    ))
    return suite


def suite_no_error() -> EvalSuite:
    """全流程无错误"""
    suite = EvalSuite(name="no_error", description="执行过程中无错误")

    suite.add(EvalTask(
        id="clean_1",
        query="计算 100 / 4",
        expected="",
        scorer=no_error,
        tags=["robustness"],
    ))
    suite.add(EvalTask(
        id="clean_2",
        query="现在几点？",
        expected="",
        scorer=no_error,
        tags=["robustness"],
    ))
    return suite


def suite_recovery() -> EvalSuite:
    """错误恢复：工具失败后仍能完成"""
    suite = EvalSuite(name="recovery", description="工具失败后恢复")

    # 这个任务故意触发工具失败（除零）
    suite.add(EvalTask(
        id="recover_1",
        query="先计算 1/0，然后告诉我为什么失败",
        expected="zero",
        scorer=contains_match,
        tags=["recovery"],
    ))
    return suite


def suite_all() -> EvalSuite:
    """所有测试集的合集"""
    suite = EvalSuite(name="all", description="完整测试集")
    for s in [suite_basic_math(), suite_tool_use(), suite_no_error()]:
        suite.tasks.extend(s.tasks)
    return suite