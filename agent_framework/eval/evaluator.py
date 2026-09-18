"""
Evaluator - 跑任务集、汇总指标
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

ScorerFn = Callable[[Any, Any, Dict], float]


@dataclass
class EvalTask:
    """一个评估任务"""
    id: str
    query: str
    expected: Any = None
    scorer: Optional[ScorerFn] = None     # None → 默认 exact_match
    meta: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "query": self.query,
            "expected": self.expected,
            "tags": self.tags,
            "meta": {k: v for k, v in self.meta.items() if k != "judge_llm"},
        }


@dataclass
class EvalResult:
    """单个任务的评估结果"""
    task_id: str
    score: float
    success: bool
    actual_answer: str
    expected: Any
    steps: int = 0
    duration_s: float = 0.0
    tokens_prompt: int = 0
    tokens_completion: int = 0
    error: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EvalSuite:
    """一组相关任务"""
    name: str
    tasks: List[EvalTask] = field(default_factory=list)
    description: str = ""

    def add(self, task: EvalTask) -> "EvalSuite":
        self.tasks.append(task)
        return self

    def __len__(self) -> int:
        return len(self.tasks)


@dataclass
class EvalReport:
    """评估报告"""
    suite_name: str
    total: int
    passed: int
    failed: int
    avg_score: float
    pass_rate: float
    avg_steps: float
    avg_duration_s: float
    total_tokens: int
    results: List[EvalResult] = field(default_factory=list)
    by_tag: Dict[str, Dict[str, float]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["results"] = [r.to_dict() for r in self.results]
        return d

    def to_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    def summary(self) -> str:
        lines = [
            f"═ Eval Report: {self.suite_name} ═",
            f"  Total:      {self.total}",
            f"  Passed:     {self.passed} ({self.pass_rate:.1%})",
            f"  Failed:     {self.failed}",
            f"  Avg Score:  {self.avg_score:.3f}",
            f"  Avg Steps:  {self.avg_steps:.1f}",
            f"  Avg Time:   {self.avg_duration_s:.2f}s",
            f"  Tokens:     {self.total_tokens}",
        ]
        if self.by_tag:
            lines.append("  By Tag:")
            for tag, stats in self.by_tag.items():
                lines.append(
                    f"    {tag:20s} pass={stats['pass_rate']:.1%} "
                    f"score={stats['avg_score']:.3f} n={int(stats['n'])}"
                )
        return "\n".join(lines)


class Evaluator:
    """
    跑任务集、汇总指标。

    用法:
        evaluator = Evaluator(agent_factory=make_agent)
        report = evaluator.run(suite)
        print(report.summary())
        report.to_json("eval_report.json")
    """

    def __init__(
        self,
        agent_factory: Callable[[], Any],
        pass_threshold: float = 0.8,
        verbose: bool = True,
    ):
        """
        Args:
            agent_factory: 无参函数，返回一个新的 CoreAgent 实例
                           （每个 task 用新 agent，避免状态污染）
            pass_threshold: 判定"通过"的分数阈值
            verbose: 是否打印每个 task 的执行过程
        """
        self.agent_factory = agent_factory
        self.pass_threshold = pass_threshold
        self.verbose = verbose

    def run(self, suite: EvalSuite) -> EvalReport:
        """跑整个 suite"""
        results: List[EvalResult] = []

        for task in suite.tasks:
            if self.verbose:
                logger.info("[eval] running task %s: %s", task.id, task.query[:60])

            result = self._run_one(task)
            results.append(result)

            if self.verbose:
                logger.info(
                    "[eval]   score=%.3f success=%s steps=%d %.2fs",
                    result.score, result.success, result.steps, result.duration_s,
                )

        return self._aggregate(suite.name, results)

    def _run_one(self, task: EvalTask) -> EvalResult:
        """跑单个 task"""
        agent = self.agent_factory()
        t0 = time.time()

        try:
            agent_result = agent.run(task.query)
            duration = time.time() - t0

            # 提取 token 用量
            usage = agent_result.metadata.get("usage", {})
            tokens_prompt = usage.get("prompt_tokens", 0)
            tokens_completion = usage.get("completion_tokens", 0)

            # 打分
            scorer = task.scorer
            if scorer is None:
                from .scorers import exact_match
                scorer = exact_match

            meta = dict(task.meta)
            meta["steps"] = agent_result.steps
            score = float(scorer(agent_result.answer, task.expected, meta))

            return EvalResult(
                task_id=task.id,
                score=score,
                success=agent_result.success and score >= self.pass_threshold,
                actual_answer=str(agent_result.answer)[:500],
                expected=task.expected,
                steps=len(agent_result.steps),
                duration_s=duration,
                tokens_prompt=tokens_prompt,
                tokens_completion=tokens_completion,
                error=agent_result.error,
                tags=task.tags, 
            )

        except Exception as e:
            logger.exception("[eval] task %s crashed", task.id)
            return EvalResult(
                task_id=task.id,
                score=0.0,
                success=False,
                actual_answer="",
                expected=task.expected,
                duration_s=time.time() - t0,
                error=f"{type(e).__name__}: {e}",
            )

    def _aggregate(self, suite_name: str, results: List[EvalResult]) -> EvalReport:
        """汇总指标"""
        n = len(results)
        if n == 0:
            return EvalReport(
                suite_name=suite_name, total=0, passed=0, failed=0,
                avg_score=0.0, pass_rate=0.0, avg_steps=0.0,
                avg_duration_s=0.0, total_tokens=0,
            )

        passed = sum(1 for r in results if r.success)
        avg_score = sum(r.score for r in results) / n
        avg_steps = sum(r.steps for r in results) / n
        avg_dur = sum(r.duration_s for r in results) / n
        total_tokens = sum(r.tokens_prompt + r.tokens_completion for r in results)

        # 按 tag 汇总
        by_tag: Dict[str, Dict[str, float]] = {}
        tag_buckets: Dict[str, List[EvalResult]] = {}
        for r in results:
            for tag in (r.tags or ["untagged"]):
                tag_buckets.setdefault(tag, []).append(r)

        for tag, bucket in tag_buckets.items():
            by_tag[tag] = {
                "n": float(len(bucket)),
                "avg_score": sum(r.score for r in bucket) / len(bucket),
                "pass_rate": sum(1 for r in bucket if r.success) / len(bucket),
            }

        return EvalReport(
            suite_name=suite_name,
            total=n,
            passed=passed,
            failed=n - passed,
            avg_score=avg_score,
            pass_rate=passed / n,
            avg_steps=avg_steps,
            avg_duration_s=avg_dur,
            total_tokens=total_tokens,
            results=results,
            by_tag=by_tag,
        )