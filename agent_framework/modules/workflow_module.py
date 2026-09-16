"""
Workflow Module - 工作流编排

支持的工作流模式:
1. 顺序执行 (Sequential): 按顺序执行步骤
2. 并行执行 (Parallel): 同时执行多个步骤
3. 条件执行 (Conditional): 根据条件选择分支
4. 循环执行 (Loop): 重复执行直到满足条件

工作流步骤类型:
- agent: 调用Agent执行
- tool: 调用工具
- condition: 条件判断
- parallel: 并行执行
- loop: 循环执行
- checkpoint: 保存检查点
"""

from typing import Dict, Any, Optional, List, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
import asyncio
import json
from datetime import datetime

logger = logging.getLogger(__name__)


class StepType(Enum):
    """步骤类型"""
    AGENT = "agent"
    TOOL = "tool"
    CONDITION = "condition"
    PARALLEL = "parallel"
    LOOP = "loop"
    CHECKPOINT = "checkpoint"
    WAIT = "wait"


class WorkflowStatus(Enum):
    """工作流状态"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"


@dataclass
class WorkflowStep:
    """工作流步骤"""
    id: str
    name: str
    step_type: StepType
    config: Dict[str, Any]
    dependencies: List[str] = field(default_factory=list)
    status: WorkflowStatus = WorkflowStatus.PENDING
    result: Any = None
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'name': self.name,
            'type': self.step_type.value,
            'config': self.config,
            'dependencies': self.dependencies,
            'status': self.status.value,
            'result': str(self.result) if self.result else None,
            'error': self.error
        }


class WorkflowModule:
    """
    工作流模块
    
    编排Agent和工具的复杂执行流程
    
    使用示例:
        workflow = WorkflowModule("Analysis Workflow")
        workflow.add_step("fetch_data", StepType.AGENT, 
                          {"query": "Fetch AAPL data"})
        workflow.add_step("analyze", StepType.AGENT,
                          {"query": "Analyze the data"},
                          dependencies=["fetch_data"])
        workflow.add_step("report", StepType.AGENT,
                          {"query": "Generate report"},
                          dependencies=["analyze"])
        
        result = workflow.run()
    """
    
    def __init__(
        self,
        name: str = "Workflow",
        description: Optional[str] = None,
        llm: Optional[Any] = None
    ):
        self.name = name
        self.description = description
        self.llm = llm
        
        self.steps: Dict[str, WorkflowStep] = {}
        self.context: Dict[str, Any] = {}
        self.status: WorkflowStatus = WorkflowStatus.PENDING
        self.results: Dict[str, Any] = {}
        
        self._event_handlers: Dict[str, List[Callable]] = {}
        self._step_counter = 0
    
    def _next_id(self) -> str:
        """生成步骤ID"""
        self._step_counter += 1
        return f"step_{self._step_counter}"
    
    def add_step(
        self,
        name: str,
        step_type: Union[str, StepType],
        config: Dict[str, Any],
        dependencies: List[str] = None,
        step_id: Optional[str] = None
    ) -> str:
        """添加步骤"""
        if isinstance(step_type, str):
            step_type = StepType(step_type)
        
        step_id = step_id or self._next_id()
        
        step = WorkflowStep(
            id=step_id,
            name=name,
            step_type=step_type,
            config=config,
            dependencies=dependencies or []
        )
        self.steps[step_id] = step
        
        logger.debug(f"Added step: {name} ({step_type.value})")
        return step_id
    
    def add_agent_step(
        self,
        name: str,
        query: str,
        agent_config: Dict[str, Any] = None,
        dependencies: List[str] = None
    ) -> str:
        """添加Agent步骤"""
        return self.add_step(
            name=name,
            step_type=StepType.AGENT,
            config={
                'query': query,
                'agent_config': agent_config or {}
            },
            dependencies=dependencies
        )
    
    def add_tool_step(
        self,
        name: str,
        tool_name: str,
        tool_args: Dict[str, Any] = None,
        dependencies: List[str] = None
    ) -> str:
        """添加工具步骤"""
        return self.add_step(
            name=name,
            step_type=StepType.TOOL,
            config={
                'tool_name': tool_name,
                'tool_args': tool_args or {}
            },
            dependencies=dependencies
        )
    
    def add_parallel(
        self,
        name: str,
        steps: List[Dict[str, Any]],
        dependencies: List[str] = None
    ) -> str:
        """添加并行步骤"""
        return self.add_step(
            name=name,
            step_type=StepType.PARALLEL,
            config={'steps': steps},
            dependencies=dependencies
        )
    
    def add_condition(
        self,
        name: str,
        condition: str,
        if_true: List[Dict[str, Any]],
        if_false: Optional[List[Dict[str, Any]]] = None,
        dependencies: List[str] = None
    ) -> str:
        """添加条件步骤"""
        return self.add_step(
            name=name,
            step_type=StepType.CONDITION,
            config={
                'condition': condition,
                'if_true': if_true,
                'if_false': if_false or []
            },
            dependencies=dependencies
        )
    
    def add_loop(
        self,
        name: str,
        steps: List[Dict[str, Any]],
        condition: Optional[str] = None,
        max_iterations: int = 10,
        dependencies: List[str] = None
    ) -> str:
        """添加循环步骤"""
        return self.add_step(
            name=name,
            step_type=StepType.LOOP,
            config={
                'steps': steps,
                'condition': condition,
                'max_iterations': max_iterations
            },
            dependencies=dependencies
        )
    
    def on(self, event_type: str, handler: Callable) -> 'WorkflowModule':
        """注册事件处理器"""
        if event_type not in self._event_handlers:
            self._event_handlers[event_type] = []
        self._event_handlers[event_type].append(handler)
        return self
    
    def _trigger_event(self, event_type: str, data: Any) -> None:
        """触发事件"""
        for handler in self._event_handlers.get(event_type, []):
            try:
                handler(data)
            except Exception as e:
                logger.warning(f"Event handler error: {e}")
    
    async def _execute_step(self, step: WorkflowStep) -> Any:
        """执行单个步骤"""
        step.status = WorkflowStatus.RUNNING
        self._trigger_event('step_start', {'step': step.to_dict()})
        
        try:
            result = None
            
            if step.step_type == StepType.AGENT:
                result = await self._execute_agent_step(step)
            elif step.step_type == StepType.TOOL:
                result = await self._execute_tool_step(step)
            elif step.step_type == StepType.PARALLEL:
                result = await self._execute_parallel_step(step)
            elif step.step_type == StepType.CONDITION:
                result = await self._execute_condition_step(step)
            elif step.step_type == StepType.LOOP:
                result = await self._execute_loop_step(step)
            elif step.step_type == StepType.CHECKPOINT:
                result = await self._execute_checkpoint_step(step)
            elif step.step_type == StepType.WAIT:
                result = await self._execute_wait_step(step)
            
            step.result = result
            step.status = WorkflowStatus.COMPLETED
            self.results[step.id] = result
            
            self._trigger_event('step_complete', {
                'step': step.to_dict(),
                'result': result
            })
            
            return result
            
        except Exception as e:
            step.status = WorkflowStatus.FAILED
            step.error = str(e)
            self._trigger_event('step_error', {
                'step': step.to_dict(),
                'error': str(e)
            })
            raise
    
    async def _execute_