"""
Modules - Agent 功能模块

可插拔的功能模块，每个模块独立且可组合:
1. MemoryModule: 记忆管理 (短期/长期/代理记忆)
2. ReasoningModule: 推理策略 (CoT, ReAct, Tree of Thought)
3. GuardrailsModule: 输入/输出护栏 (安全、合规)
4. TeamModule: 多Agent团队协作
5. WorkflowModule: 工作流编排 (顺序/并行/条件/循环)

使用示例:
    from agent_framework.modules import (
        MemoryModule,
        ReasoningModule,
        GuardrailsModule,
        TeamModule,
        WorkflowModule
    )
    
    # 组合使用
    memory = MemoryModule(user_id="user123")
    reasoning = ReasoningModule(strategy="chain_of_thought")
    guardrails = GuardrailsModule.default_financial()
    
    # 应用到 Agent
    config = {
        "memory": memory.to_agent_config(),
        "reasoning": reasoning.to_agent_config(),
        "guardrails": guardrails.to_agent_config()
    }
"""

from .memory_module import (
    MemoryModule,
    AgenticMemoryModule,
    WorkingMemory,
    EpisodicMemory,
    MemoryItem,
)

from .reasoning_module import (
    ReasoningModule,
    ReasoningTracker,
    ReasoningBuilder,
    REASONING_STRATEGIES,
)

from .guardrails_module import (
    GuardrailsModule,
    FinancialPIIGuardrail,
    PromptInjectionGuardrail,
    TradingComplianceGuardrail,
    OutputValidationGuardrail,
)

from .team_module import (
    TeamModule,
    TeamBuilder,
    TeamRole,
    TeamMode,
)

from .workflow_module import (
    WorkflowModule,
    WorkflowBuilder,
    StepType,
    WorkflowStatus,
    WorkflowStep,
)

__all__ = [
    # Memory
    'MemoryModule',
    'AgenticMemoryModule',
    'WorkingMemory',
    'EpisodicMemory',
    'MemoryItem',
    
    # Reasoning
    'ReasoningModule',
    'ReasoningTracker',
    'ReasoningBuilder',
    'REASONING_STRATEGIES',
    
    # Guardrails
    'GuardrailsModule',
    'FinancialPIIGuardrail',
    'PromptInjectionGuardrail',
    'TradingComplianceGuardrail',
    'OutputValidationGuardrail',
    
    # Team
    'TeamModule',
    'TeamBuilder',
    'TeamRole',
    'TeamMode',
    
    # Workflow
    'WorkflowModule',
    'WorkflowBuilder',
    'StepType',
    'WorkflowStatus',
    'WorkflowStep',
]