"""
Tools - Agent 工具系统

工具是 Agent 执行任务的能力来源，包括:
1. 工具注册中心 (ToolsRegistry): 管理所有可用工具
2. 技能库 (SkillLibrary): 存储可复用的任务配方
3. 终端工具包 (TerminalToolkit): 终端操作工具

使用示例:
    from agent_framework.tools import ToolsRegistry, SkillLibrary, TerminalToolkit
    
    # 注册工具
    registry = ToolsRegistry()
    registry.register("get_price", get_price_handler)
    
    # 获取工具
    tools = registry.get_tools(["get_price", "calculator"])
    
    # 技能库
    skills = SkillLibrary()
    skills.add("stock_analysis", "分析股票", recipe)
"""

from .tools_registry import (
    ToolsRegistry,
    ToolDefinition,
    ToolParameter,
    ToolCategory,
    get_tools_registry,
    get_tool,
    register_tool,
    list_tools,
    call_tool,
)

from .skill_library import (
    SkillLibrary,
    Skill,
    SkillStep,
    SkillRecipe,
    create_skill_library,
    get_skill,
    search_skills,
    add_skill,
)

from .terminal_toolkit import (
    TerminalToolkit,
    TerminalCommand,
    TerminalResult,
    create_terminal_toolkit,
    execute_terminal_command,
)

__all__ = [
    # Tools Registry
    'ToolsRegistry',
    'ToolDefinition',
    'ToolParameter',
    'tool',
    'get_tools',
    'register_tool',
    'list_tools',
    'call_tool',
    
    # Skill Library
    'SkillLibrary',
    'Skill',
    'SkillStep',
    'SkillRecipe',
    'create_skill_library',
    'get_skill',
    'search_skills',
    'add_skill',
    
    # Terminal Toolkit
    'TerminalToolkit',
    'TerminalCommand',
    'TerminalResult',
    'create_terminal_toolkit',
    'execute_terminal_command',
]