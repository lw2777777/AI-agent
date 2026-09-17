"""
运行评估。

用法:
    python -m agent_framework.eval.run_eval
"""
import logging
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("eval")


def make_agent():
    """构造一个全新的 CoreAgent"""
    from agent_framework.core.core_agent import CoreAgent
    from agent_framework.core.llm_adapter import LLMAdapter
    from agent_framework.tools.tools_registry import get_tools_registry

    # 读 API key
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY not set")

    llm = LLMAdapter(provider="deepseek", api_key=api_key, model="deepseek-chat")

    # 拿所有内置工具
    registry = get_tools_registry()
    tools = list(registry._tools.values())

    return CoreAgent(
        name="eval_agent",
        llm=llm,
        tools=tools,
        max_iterations=8,
        verbose=False,
    )


def main():
    from agent_framework.eval.suites import suite_all
    from agent_framework.eval.evaluator import Evaluator

    suite = suite_all()
    logger.info("Running suite: %s (%d tasks)", suite.name, len(suite))

    evaluator = Evaluator(
        agent_factory=make_agent,
        pass_threshold=0.8,
        verbose=True,
    )

    report = evaluator.run(suite)
    print("\n" + report.summary())

    # 保存
    out_path = Path("eval_report.json")
    report.to_json(str(out_path))
    logger.info("Report saved to %s", out_path)


if __name__ == "__main__":
    main()