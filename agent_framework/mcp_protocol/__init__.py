"""
MCP Protocol - Model Context Protocol 实现

MCP (Model Context Protocol) 是 Anthropic 提出的标准化协议，
用于 Agent 与工具/资源/提示词之间的交互。

核心概念:
1. Tools: Agent 可调用的函数 (tools/list, tools/call)
2. Resources: Agent 可读取的数据 (resources/list, resources/read)
3. Prompts: 预定义的提示词模板 (prompts/list, prompts/get)

实现方式:
- Server: 符合 MCP 规范的服务器 (支持 stdio 和 HTTP 传输)
- Client: 连接 MCP 服务器的客户端
- Tools: 工具注册和调用

使用场景:
1. Agent 通过 MCP 调用外部工具
2. 多 Agent 共享工具集
3. 工具的热插拔
"""

from .server import (
    MCPServer,
    MCPTool,
    MCPResource,
    MCPPrompt,
    MCPMethod,
    MCPRequest,
    MCPResponse,
    create_mcp_server,
    run_mcp_server,
)

from .mcp_tools import (
    MCPTools,
    MCPToolWrapper,
    get_mcp_tools,
    create_mcp_client,
    MCP_AVAILABLE,
    MCP_TOOLS,
)

__all__ = [
    # Server
    'MCPServer',
    'MCPTool',
    'MCPResource',
    'MCPPrompt',
    'MCPMethod',
    'MCPRequest',
    'MCPResponse',
    'create_mcp_server',
    'run_mcp_server',
    # Client
    'MCPTools',
    'MCPToolWrapper',
    'get_mcp_tools',
    'create_mcp_client',
    'MCP_AVAILABLE',
    'MCP_TOOLS',
]