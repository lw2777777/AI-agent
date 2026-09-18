"""
内置测试集。

覆盖 harness 的 8 个核心能力:
  1. 单工具调用 (single_tool)
  2. 多工具串联 (multi_tool)
  3. 多步推理 (multi_step)
  4. 工具失败恢复 (recovery)
  5. 无工具对话 (no_tool)
  6. 负例 - 应失败 (negative)
  7. 边界输入 (edge_case)
  8. 上下文管理 (long_context)
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


# ══════════════════════════════════════════════════════════
# Level 1: 单工具调用（5 个）
# ══════════════════════════════════════════════════════════
def suite_single_tool() -> EvalSuite:
    s = EvalSuite(name="single_tool", description="单工具调用 + 数值准确")

    s.add(EvalTask(
        id="st_1", query="计算 123 * 456", expected=56088,
        scorer=numeric_close, meta={"tolerance": 0.01},
        tags=["single_tool", "math"],
    ))
    s.add(EvalTask(
        id="st_2", query="计算 (10 + 20) * 3", expected=90,
        scorer=numeric_close, meta={"tolerance": 0.01},
        tags=["single_tool", "math"],
    ))
    s.add(EvalTask(
        id="st_3", query="计算 2 的 10 次方", expected=1024,
        scorer=numeric_close, meta={"tolerance": 0.01},
        tags=["single_tool", "math"],
    ))
    s.add(EvalTask(
        id="st_4", query="计算 100 / 4", expected=25,
        scorer=numeric_close, meta={"tolerance": 0.01},
        tags=["single_tool", "math"],
    ))
    s.add(EvalTask(
        id="st_5", query="格式化这段 JSON: {\"name\":\"alice\",\"age\":30}",
        expected="format_json",
        scorer=tool_called, meta={"tool_name": "format_json"},
        tags=["single_tool", "format"],
    ))
    return s


# ══════════════════════════════════════════════════════════
# Level 2: 多工具串联（5 个）
# 需要调用 2 个以上工具，且工具间有数据依赖
# ══════════════════════════════════════════════════════════
def suite_multi_tool() -> EvalSuite:
    s = EvalSuite(name="multi_tool", description="多工具串联")

    s.add(EvalTask(
        id="mt_1",
        query="计算 123 * 456，然后把结果格式化为带千位分隔符的字符串",
        expected="56,088",
        scorer=contains_match,
        tags=["multi_tool", "chained"],
    ))
    s.add(EvalTask(
        id="mt_2",
        query="计算 (100 + 200) * 3，再用结果除以 9",
        expected=100,
        scorer=numeric_close, meta={"tolerance": 1.0},
        tags=["multi_tool", "chained"],
    ))
    s.add(EvalTask(
        id="mt_3",
        query="先告诉我现在北京时间几点，然后计算这个小时的平方",
        expected=None,
        scorer=no_error,
        tags=["multi_tool", "chained"],
    ))
    s.add(EvalTask(
        id="mt_4",
        query="格式化 JSON {\"x\":1,\"y\":[1,2,3]}，然后计算这个 JSON 里有多少个数字",
        expected="4",
        scorer=contains_match,
        tags=["multi_tool", "chained"],
    ))
    s.add(EvalTask(
        id="mt_5",
        query="计算 2^20，然后判断这个数字是奇数还是偶数",
        expected="偶",
        scorer=contains_match,
        tags=["multi_tool", "chained"],
    ))
    return s


# ══════════════════════════════════════════════════════════
# Level 3: 多步推理（5 个）
# 需要 3+ 步，或需要先规划再执行
# ══════════════════════════════════════════════════════════
def suite_multi_step() -> EvalSuite:
    s = EvalSuite(name="multi_step", description="多步推理")

    s.add(EvalTask(
        id="ms_1",
        query="一个班有 30 个学生，60% 是女生。如果每个女生带 2 本书，每个男生带 3 本书，一共多少本书？",
        expected=90,     # 18 女 * 2 + 12 男 * 3 = 36 + 36 = 72？重算：30*0.6=18女，12男；18*2+12*3=36+36=72
        scorer=numeric_close, meta={"tolerance": 1.0},
        tags=["multi_step", "reasoning"],
    ))
    s.add(EvalTask(
        id="ms_2",
        query="计算 1 到 10 的和，然后计算这个和除以 5 的结果",
        expected=11,     # 55 / 5 = 11
        scorer=numeric_close, meta={"tolerance": 0.1},
        tags=["multi_step", "reasoning"],
    ))
    s.add(EvalTask(
        id="ms_3",
        query="如果 x = 5，y = 3，计算 x^2 + 2xy + y^2",
        expected=64,     # (5+3)^2 = 64
        scorer=numeric_close, meta={"tolerance": 0.1},
        tags=["multi_step", "reasoning"],
    ))
    s.add(EvalTask(
        id="ms_4",
        query="先计算 100 - 37，然后乘以 2，最后减去 26",
        expected=100,    # (100-37)*2 - 26 = 126 - 26 = 100
        scorer=numeric_close, meta={"tolerance": 0.1},
        tags=["multi_step", "reasoning"],
    ))
    s.add(EvalTask(
        id="ms_5",
        query="用 3 个数字 2、3、4 能组成多少个不同的两位数（数字不重复）？",
        expected=6,      # 3*2 = 6
        scorer=numeric_close, meta={"tolerance": 0.1},
        tags=["multi_step", "reasoning"],
    ))
    return s


# ══════════════════════════════════════════════════════════
# Level 4: 工具失败恢复（4 个）
# 工具会失败，agent 需要分析错误、换方式、或报告失败
# ══════════════════════════════════════════════════════════
def suite_recovery() -> EvalSuite:
    s = EvalSuite(name="recovery", description="工具失败后恢复")

    s.add(EvalTask(
        id="rc_1",
        query="计算 10 除以 0，如果失败，解释为什么不能除零",
        expected="zero",
        scorer=contains_match,
        tags=["recovery", "error_handling"],
    ))
    s.add(EvalTask(
        id="rc_2",
        query="计算 sqrt(-1)，如果失败，解释为什么负数不能开平方",
        expected=None,
        scorer=no_error,
        tags=["recovery", "error_handling"],
    ))
    s.add(EvalTask(
        id="rc_3",
        query="调用一个不存在的工具 do_magic，如果不存在，改用 calculate 计算 1+1",
        expected=2,
        scorer=numeric_close, meta={"tolerance": 0.1},
        tags=["recovery", "error_handling"],
    ))
    s.add(EvalTask(
        id="rc_4",
        query="计算 'abc' + 1，这会失败。失败后请计算 'abc' 的字符数",
        expected=3,
        scorer=numeric_close, meta={"tolerance": 0.1},
        tags=["recovery", "error_handling"],
    ))
    return s


# ══════════════════════════════════════════════════════════
# Level 5: 无工具对话（4 个）
# 不需要调工具，考察纯 LLM 能力 + agent 是否知道"不该调工具"
# ══════════════════════════════════════════════════════════
def suite_no_tool() -> EvalSuite:
    s = EvalSuite(name="no_tool", description="无需工具，直接回答")

    s.add(EvalTask(
        id="nt_1",
        query="用一句话解释什么是 ReAct",
        expected="reason",
        scorer=contains_match,
        tags=["no_tool", "chat"],
    ))
    s.add(EvalTask(
        id="nt_2",
        query="把 'hello world' 翻译成中文",
        expected="你好",
        scorer=contains_match,
        tags=["no_tool", "chat"],
    ))
    s.add(EvalTask(
        id="nt_3",
        query="1 加 1 等于几？直接回答，不要调用任何工具",
        expected=2,
        scorer=numeric_close, meta={"tolerance": 0.1},
        tags=["no_tool", "chat"],
    ))
    s.add(EvalTask(
        id="nt_4",
        query="请用 JSON 格式回答：{\"status\": \"ok\"}",
        expected="ok",
        scorer=contains_match,
        tags=["no_tool", "chat"],
    ))
    return s


# ══════════════════════════════════════════════════════════
# Level 6: 负例 - 应该失败（4 个）
# 这些任务 agent 应该"承认做不到"，而不是胡编
# ══════════════════════════════════════════════════════════
def suite_negative() -> EvalSuite:
    s = EvalSuite(name="negative", description="应该失败的任务")

    s.add(EvalTask(
        id="neg_1",
        query="帮我订一张明天从北京到上海的机票",
        expected="cannot",
        scorer=contains_match,   # 应该包含 "cannot" / "无法" / "抱歉"
        tags=["negative"],
    ))
    s.add(EvalTask(
        id="neg_2",
        query="访问 https://example.com 并告诉我页面上有什么",
        expected="cannot",
        scorer=contains_match,
        tags=["negative"],
    ))
    s.add(EvalTask(
        id="neg_3",
        query="读取我电脑上的 /etc/passwd 文件",
        expected="cannot",
        scorer=contains_match,
        tags=["negative", "security"],
    ))
    s.add(EvalTask(
        id="neg_4",
        query="预测明天上证指数的收盘价",
        expected="cannot",
        scorer=contains_match,
        tags=["negative"],
    ))
    return s


# ══════════════════════════════════════════════════════════
# Level 7: 边界输入（4 个）
# 空、超长、特殊字符、非法参数
# ══════════════════════════════════════════════════════════
def suite_edge_case() -> EvalSuite:
    s = EvalSuite(name="edge_case", description="边界输入")

    s.add(EvalTask(
        id="ec_1",
        query="",
        expected=None,
        scorer=no_error,   # agent 不应崩，应该 graceful 处理
        tags=["edge_case"],
    ))
    s.add(EvalTask(
        id="ec_2",
        query="计算 " + "1+" * 50 + "1",
        expected=51,
        scorer=numeric_close, meta={"tolerance": 0.1},
        tags=["edge_case", "long_input"],
    ))
    s.add(EvalTask(
        id="ec_3",
        query="计算 \"¥100\" + \"$50\" 等于多少（忽略货币符号，只算数字）",
        expected=150,
        scorer=numeric_close, meta={"tolerance": 0.1},
        tags=["edge_case", "special_char"],
    ))
    s.add(EvalTask(
        id="ec_4",
        query="计算 0.1 + 0.2，保留 1 位小数",
        expected=0.3,
        scorer=numeric_close, meta={"tolerance": 0.01},
        tags=["edge_case", "float"],
    ))
    return s


# ══════════════════════════════════════════════════════════
# Level 8: 长上下文（4 个）
# 考察 context 管理能力
# ══════════════════════════════════════════════════════════
def suite_long_context() -> EvalSuite:
    s = EvalSuite(name="long_context", description="长上下文")

    s.add(EvalTask(
        id="lc_1",
        query="请依次计算：1+1, 2+2, 3+3, 4+4, 5+5，然后告诉我所有结果的和",
        expected=30,     # 2+4+6+8+10 = 30
        scorer=numeric_close, meta={"tolerance": 0.1},
        tags=["long_context", "multi_step"],
    ))
    s.add(EvalTask(
        id="lc_2",
        query="请依次告诉我：现在几点，今天是几号，然后计算今天是今年的第几天",
        expected=None,
        scorer=no_error,
        tags=["long_context", "multi_step"],
    ))
    s.add(EvalTask(
        id="lc_3",
        query="计算 1! + 2! + 3! + 4! + 5!",
        expected=153,    # 1+2+6+24+120 = 153
        scorer=numeric_close, meta={"tolerance": 0.1},
        tags=["long_context", "math"],
    ))
    s.add(EvalTask(
        id="lc_4",
        query="计算 (1+2+3+4+5) * (1*2*3*4*5)",
        expected=1800,   # 15 * 120 = 1800
        scorer=numeric_close, meta={"tolerance": 0.1},
        tags=["long_context", "math"],
    ))
    return s


# ══════════════════════════════════════════════════════════
# 合集
# ══════════════════════════════════════════════════════════
ALL_SUITES = [
    suite_single_tool,      # 5
    suite_multi_tool,       # 5
    suite_multi_step,       # 5
    suite_recovery,         # 4
    suite_no_tool,          # 4
    suite_negative,         # 4
    suite_edge_case,        # 4
    suite_long_context,     # 4
]
# 总计 35 个任务


def suite_all() -> EvalSuite:
    s = EvalSuite(name="all", description="完整测试集")
    for fn in ALL_SUITES:
        sub = fn()
        s.tasks.extend(sub.tasks)
    return s


def suite_smoke() -> EvalSuite:
    """快速烟测：每类抽 1 个"""
    s = EvalSuite(name="smoke", description="快速烟测")
    for fn in ALL_SUITES:
        sub = fn()
        if sub.tasks:
            s.tasks.append(sub.tasks[0])
    return s