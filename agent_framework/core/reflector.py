"""
Reflector - 自我反思机制 (Reflexion)

核心思想 (Shinn et al., 2023):
1. 执行任务
2. 反思执行过程: 哪里做得好? 哪里做得不好?
3. 提炼经验: 下次遇到类似任务应该如何改进?
4. 存储经验: 存入记忆供未来使用

这是Agent自我进化的关键机制!
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ReflectionLevel(Enum):
    """反思层级"""
    STEP = "step"           # 单步反思
    TASK = "task"           # 任务级反思
    PATTERN = "pattern"     # 模式级反思 (跨任务)


@dataclass
class ReflectionResult:
    """反思结果"""
    level: ReflectionLevel
    success: bool
    what_worked: List[str] = field(default_factory=list)
    what_failed: List[str] = field(default_factory=list)
    lessons: List[str] = field(default_factory=list)
    improvements: List[str] = field(default_factory=list)
    confidence: float = 0.7
    raw_analysis: str = ""


class Reflector:
    """
    自我反思器

    功能:
    1. 分析执行过程
    2. 提取经验教训
    3. 生成改进建议
    4. 存储反思结果
    """

    def __init__(
        self,
        llm: Optional[Any] = None,
        memory_store: Optional[Any] = None,
        system_prompt: Optional[str] = None
    ):
        self.llm = llm
        self.memory_store = memory_store
        self._reflections: List[ReflectionResult] = []

        self.system_prompt = system_prompt or self._DEFAULT_SYSTEM_PROMPT

    _DEFAULT_SYSTEM_PROMPT = """
You are a reflective AI that analyzes your own performance.

After completing a task, reflect on:
1. What worked well? (strategies, tools, reasoning)
2. What didn't work? (mistakes, wrong assumptions, tool failures)
3. What lessons can be learned?
4. How to improve next time?

Be specific and actionable. Focus on patterns, not one-off errors.
"""

    def reflect_on_step(
        self,
        step: Dict[str, Any],
        context: Dict[str, Any]
    ) -> ReflectionResult:
        """
        单步反思

        Args:
            step: {
                'thought': str,
                'action': str,
                'action_input': Dict,
                'observation': str,
                'success': bool
            }
            context: 任务上下文

        Returns:
            ReflectionResult
        """
        # 1. 快速规则检查 (不需要LLM)
        quick = self._quick_reflect_step(step)

        # 2. 如果LLM可用，进行深度反思
        if self.llm and (not quick.success or quick.confidence < 0.6):
            deep = self._llm_reflect_step(step, context)
            if deep:
                return deep

        return quick

    def _quick_reflect_step(self, step: Dict[str, Any]) -> ReflectionResult:
        """快速规则反思"""
        result = ReflectionResult(
            level=ReflectionLevel.STEP,
            success=step.get('success', False)
        )

        observation = step.get('observation', '')
        action = step.get('action', '')

        # 成功模式
        if result.success:
            if any(kw in observation.lower() for kw in ['found', 'discovered', 'calculated', 'retrieved']):
                result.what_worked.append(f"Successfully used {action} to get useful information")
            if len(observation) > 100:
                result.what_worked.append("Received comprehensive observation")

        # 失败模式
        else:
            error = step.get('error', '')
            if 'not found' in error.lower():
                result.what_failed.append(f"Tool {action} failed: resource not found")
            if 'timeout' in error.lower():
                result.what_failed.append(f"Tool {action} timeout - consider smaller requests")
            if 'permission' in error.lower():
                result.what_failed.append("Permission error - check access rights")

            result.lessons.append(f"Tool {action} requires more robust error handling")

        # 置信度
        if result.success and result.what_worked:
            result.confidence = 0.8
        elif not result.success and result.what_failed:
            result.confidence = 0.7
        else:
            result.confidence = 0.5

        return result

    def _llm_reflect_step(
        self,
        step: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Optional[ReflectionResult]:
        """LLM深度反思"""
        if not self.llm:
            return None

        prompt = f"""
Reflect on this step:

Task: {context.get('task', 'Unknown')}

Step:
- Thought: {step.get('thought', '')}
- Action: {step.get('action', '')}
- Action Input: {step.get('action_input', {})}
- Observation: {step.get('observation', '')}
- Success: {step.get('success', False)}

Analyze:
1. What worked well?
2. What could be improved?
3. What lesson can be learned?

Return JSON:
{{
    "what_worked": ["item1", "item2"],
    "what_failed": ["item1", "item2"],
    "lessons": ["lesson1", "lesson2"],
    "improvements": ["improvement1", "improvement2"],
    "confidence": 0.7
}}
"""
        try:
            response = self.llm.chat(prompt)
            parsed = self._parse_json_response(response)

            return ReflectionResult(
                level=ReflectionLevel.STEP,
                success=step.get('success', False),
                what_worked=parsed.get('what_worked', []),
                what_failed=parsed.get('what_failed', []),
                lessons=parsed.get('lessons', []),
                improvements=parsed.get('improvements', []),
                confidence=parsed.get('confidence', 0.7),
                raw_analysis=response
            )
        except Exception as e:
            logger.warning(f"LLM reflection failed: {e}")
            return None

    def reflect_on_task(
        self,
        steps: List[Dict[str, Any]],
        task: str,
        final_result: Any
    ) -> ReflectionResult:
        """
        任务级反思

        Args:
            steps: 所有步骤
            task: 原始任务
            final_result: 最终结果

        Returns:
            ReflectionResult
        """
        result = ReflectionResult(
            level=ReflectionLevel.TASK,
            success=final_result.get('success', False)
        )

        # 统计成功/失败步骤
        successful_steps = [s for s in steps if s.get('success', False)]
        failed_steps = [s for s in steps if not s.get('success', False)]

        if successful_steps:
            result.what_worked.append(f"{len(successful_steps)} steps succeeded")

        if failed_steps:
            result.what_failed.append(f"{len(failed_steps)} steps failed")
            # 提取失败原因
            errors = [s.get('error', '') for s in failed_steps if s.get('error')]
            if errors:
                result.what_failed.append(f"Errors: {', '.join(errors[:3])}")

        # 生成改进建议
        if failed_steps:
            result.improvements.append("Add more error handling and fallback strategies")
            result.improvements.append("Consider breaking down complex steps")

        if len(steps) > 10:
            result.improvements.append("Consider using parallel execution for independent steps")

        # 提炼经验
        if result.success:
            result.lessons.append(f"Task '{task[:50]}...' completed successfully")
            # 记录成功模式
            if len(successful_steps) > 3:
                result.lessons.append("Multi-step approach was effective")
        else:
            result.lessons.append(f"Task '{task[:50]}...' failed - need to revisit approach")

        # 如果有LLM，进行深度反思
        if self.llm:
            deep = self._llm_reflect_task(steps, task, final_result)
            if deep:
                return deep

        return result

    def _llm_reflect_task(
        self,
        steps: List[Dict[str, Any]],
        task: str,
        final_result: Any
    ) -> Optional[ReflectionResult]:
        """LLM任务级反思"""
        if not self.llm:
            return None

        steps_summary = "\n".join([
            f"Step {i+1}: {s.get('action', 'N/A')} - {'✓' if s.get('success') else '✗'}"
            for i, s in enumerate(steps)
        ])

        prompt = f"""
Reflect on this complete task:

Task: {task}

Steps:
{steps_summary}

Final Result: {final_result}

Analyze:
1. Overall approach: what worked well in the strategy?
2. What patterns emerged?
3. Key lessons learned?
4. How would you approach this differently next time?

Return JSON:
{{
    "what_worked": ["strategy1", "strategy2"],
    "what_failed": ["mistake1", "mistake2"],
    "lessons": ["lesson1", "lesson2"],
    "improvements": ["improvement1", "improvement2"],
    "confidence": 0.7
}}
"""
        try:
            response = self.llm.chat(prompt)
            parsed = self._parse_json_response(response)

            return ReflectionResult(
                level=ReflectionLevel.TASK,
                success=final_result.get('success', False),
                what_worked=parsed.get('what_worked', []),
                what_failed=parsed.get('what_failed', []),
                lessons=parsed.get('lessons', []),
                improvements=parsed.get('improvements', []),
                confidence=parsed.get('confidence', 0.7),
                raw_analysis=response
            )
        except Exception as e:
            logger.warning(f"LLM task reflection failed: {e}")
            return None

    def store_reflection(
        self,
        reflection: ReflectionResult,
        task_id: str = None
    ) -> Optional[str]:
        """
        存储反思到记忆

        Args:
            reflection: 反思结果
            task_id: 任务ID

        Returns:
            记忆ID
        """
        if not self.memory_store:
            logger.warning("No memory store configured")
            return None

        # 构建记忆内容
        content = f"""
Reflection on {reflection.level.value}:
Success: {reflection.success}
What worked: {', '.join(reflection.what_worked) if reflection.what_worked else 'Nothing specific'}
What failed: {', '.join(reflection.what_failed) if reflection.what_failed else 'Nothing specific'}
Lessons: {', '.join(reflection.lessons) if reflection.lessons else 'No lessons extracted'}
Improvements: {', '.join(reflection.improvements) if reflection.improvements else 'No improvements suggested'}
""".strip()

        return self.memory_store.save(
            content=content,
            memory_type="reflection",
            metadata={
                "level": reflection.level.value,
                "success": reflection.success,
                "confidence": reflection.confidence,
                "task_id": task_id,
                "what_worked": reflection.what_worked,
                "what_failed": reflection.what_failed,
                "lessons": reflection.lessons,
                "improvements": reflection.improvements
            },
            importance=reflection.confidence
        )

    def get_insights_for_task(self, task: str, k: int = 3) -> List[Dict[str, Any]]:
        """
        获取相关反思见解

        Args:
            task: 当前任务
            k: 返回数量

        Returns:
            相关反思列表
        """
        if not self.memory_store:
            return []

        memories = self.memory_store.search(
            query=task,
            k=k,
            memory_type="reflection",
            min_importance=0.5
        )

        return memories

    @staticmethod
    def _parse_json_response(response: str) -> Dict[str, Any]:
        """解析LLM JSON响应"""
        text = response.strip()
        # 去除markdown代码块
        text = re.sub(r'^```(?:json)?\s*', '', text)
        text = re.sub(r'\s*```$', '', text)

        # 提取JSON
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            text = match.group(0)

        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON: {e}")
            return {}

    def get_reflection_history(self) -> List[ReflectionResult]:
        """获取反思历史"""
        return self._reflections


# ==================== 便捷函数 ====================

def create_reflector(
    llm: Optional[Any] = None,
    memory_store: Optional[Any] = None
) -> Reflector:
    """创建反思器"""
    return Reflector(llm=llm, memory_store=memory_store)


def reflect_on_execution(
    steps: List[Dict[str, Any]],
    task: str,
    result: Any,
    llm: Optional[Any] = None
) -> ReflectionResult:
    """
    快速反思执行过程

    Args:
        steps: 执行步骤
        task: 任务描述
        result: 执行结果
        llm: LLM实例 (可选)

    Returns:
        ReflectionResult
    """
    reflector = Reflector(llm=llm)
    return reflector.reflect_on_task(steps, task, result)