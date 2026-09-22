"""
MCP Server - Model Context Protocol 服务端实现

符合 MCP 规范的服务器，支持:
1. 工具发现 (tools/list)
2. 工具调用 (tools/call)
3. 资源管理 (resources/list, resources/read)
4. 提示词管理 (prompts/list, prompts/get)
5. 初始化握手 (initialize)
6. 心跳检测 (ping)

传输方式:
- stdio: 标准输入输出 (用于子进程)
- HTTP: HTTP 服务器 (用于远程调用)

这是展示你理解 Agent 协议设计的关键模块!
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Callable, Union
from datetime import datetime

logger = logging.getLogger(__name__)


class MCPMethod(Enum):
    """MCP 方法"""
    # 核心
    INITIALIZE = "initialize"
    PING = "ping"
    
    # 工具
    TOOLS_LIST = "tools/list"
    TOOLS_CALL = "tools/call"
    
    # 资源
    RESOURCES_LIST = "resources/list"
    RESOURCES_READ = "resources/read"
    RESOURCES_TEMPLATES_LIST = "resources/templates/list"
    RESOURCES_TEMPLATES_READ = "resources/templates/read"
    
    # 提示词
    PROMPTS_LIST = "prompts/list"
    PROMPTS_GET = "prompts/get"
    
    # 完成
    COMPLETE = "completion/complete"


@dataclass
class MCPTool:
    """MCP 工具定义"""
    name: str
    description: str
    inputSchema: Dict[str, Any]
    handler: Optional[Callable] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.inputSchema
        }


@dataclass
class MCPResource:
    """MCP 资源定义"""
    uri: str
    name: str
    description: str
    mimeType: str = "text/plain"
    handler: Optional[Callable] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "uri": self.uri,
            "name": self.name,
            "description": self.description,
            "mimeType": self.mimeType
        }


@dataclass
class MCPPrompt:
    """MCP 提示词定义"""
    name: str
    description: str
    arguments: List[Dict[str, str]]
    handler: Optional[Callable] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "arguments": self.arguments
        }


@dataclass
class MCPRequest:
    """MCP 请求"""
    id: Union[str, int]
    method: str
    params: Dict[str, Any]
    jsonrpc: str = "2.0"


@dataclass
class MCPResponse:
    """MCP 响应"""
    id: Union[str, int]
    result: Optional[Any] = None
    error: Optional[Dict[str, Any]] = None
    jsonrpc: str = "2.0"
    
    def to_dict(self) -> Dict[str, Any]:
        result = {"jsonrpc": self.jsonrpc, "id": self.id}
        if self.error:
            result["error"] = self.error
        else:
            result["result"] = self.result
        return result


class MCPServer:
    """
    MCP 协议服务器
    
    实现符合 MCP 规范的服务器，支持工具、资源、提示词的管理和调用。
    
    使用示例:
        server = MCPServer(name="My Agent Server")
        
        # 注册工具
        @server.tool(
            name="get_price",
            description="获取股票价格",
            parameters={
                "symbol": {"type": "string", "description": "股票代码"}
            }
        )
        def get_price(symbol: str) -> str:
            return f"${symbol} price: 150.00"
        
        # 注册资源
        server.add_resource(
            uri="memory://recent_queries",
            name="Recent Queries",
            description="最近查询记录",
            handler=lambda: json.dumps(["AAPL", "MSFT"])
        )
        
        # 运行服务器
        server.run(transport="stdio")  # 或 "http"
    """
    
    def __init__(
        self,
        name: str = "MCP Server",
        version: str = "1.0.0",
        instructions: Optional[str] = None
    ):
        self.name = name
        self.version = version
        self.instructions = instructions or "MCP Server for Agent tools"
        
        self.tools: Dict[str, MCPTool] = {}
        self.resources: Dict[str, MCPResource] = {}
        self.prompts: Dict[str, MCPPrompt] = {}
        
        self._initialized = False
        self._client_info: Optional[Dict[str, Any]] = None
        self._request_handlers: Dict[str, Callable] = {}
        
        # 注册默认处理器
        self._register_default_handlers()
    
    def _register_default_handlers(self):
        """注册默认处理器"""
        self._request_handlers = {
            MCPMethod.INITIALIZE.value: self._handle_initialize,
            MCPMethod.PING.value: self._handle_ping,
            MCPMethod.TOOLS_LIST.value: self._handle_tools_list,
            MCPMethod.TOOLS_CALL.value: self._handle_tools_call,
            MCPMethod.RESOURCES_LIST.value: self._handle_resources_list,
            MCPMethod.RESOURCES_READ.value: self._handle_resources_read,
            MCPMethod.PROMPTS_LIST.value: self._handle_prompts_list,
            MCPMethod.PROMPTS_GET.value: self._handle_prompts_get,
        }
    
    # ==================== 装饰器 ====================
    
    def tool(
        self,
        name: Optional[str] = None,
        description: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None
    ) -> Callable:
        """
        工具注册装饰器
        
        Args:
            name: 工具名称 (默认: 函数名)
            description: 工具描述 (默认: 函数文档)
            parameters: 参数模式 (默认: 从函数签名推断)
        
        使用示例:
            @server.tool(
                name="get_weather",
                description="获取天气信息",
                parameters={
                    "city": {"type": "string", "description": "城市名称"}
                }
            )
            def get_weather(city: str) -> str:
                return f"Weather in {city}: Sunny 25°C"
        """
        def decorator(func: Callable) -> Callable:
            tool_name = name or func.__name__
            tool_desc = description or func.__doc__ or "No description"
            tool_params = parameters or self._infer_parameters(func)
            
            self.tools[tool_name] = MCPTool(
                name=tool_name,
                description=tool_desc,
                inputSchema=tool_params,
                handler=func
            )
            
            logger.info(f"Registered MCP tool: {tool_name}")
            return func
        return decorator
    
    def _infer_parameters(self, func: Callable) -> Dict[str, Any]:
        """从函数签名推断参数"""
        import inspect
        
        sig = inspect.signature(func)
        properties = {}
        required = []
        
        for param_name, param in sig.parameters.items():
            if param_name in ('self', 'cls'):
                continue
            
            properties[param_name] = {
                "type": "string",
                "description": f"Parameter {param_name}"
            }
            
            if param.default == inspect.Parameter.empty:
                required.append(param_name)
        
        return {
            "type": "object",
            "properties": properties,
            "required": required
        }
    
    # ==================== 注册方法 ====================
    
    def add_tool(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Any],
        handler: Callable
    ) -> "MCPServer":
        """添加工具"""
        self.tools[name] = MCPTool(
            name=name,
            description=description,
            inputSchema=parameters,
            handler=handler
        )
        logger.info(f"Added MCP tool: {name}")
        return self
    
    def add_resource(
        self,
        uri: str,
        name: str,
        description: str,
        handler: Callable,
        mimeType: str = "text/plain"
    ) -> "MCPServer":
        """添加资源"""
        self.resources[uri] = MCPResource(
            uri=uri,
            name=name,
            description=description,
            mimeType=mimeType,
            handler=handler
        )
        logger.info(f"Added MCP resource: {name} ({uri})")
        return self
    
    def add_prompt(
        self,
        name: str,
        description: str,
        arguments: List[Dict[str, str]],
        handler: Callable
    ) -> "MCPServer":
        """添加提示词"""
        self.prompts[name] = MCPPrompt(
            name=name,
            description=description,
            arguments=arguments,
            handler=handler
        )
        logger.info(f"Added MCP prompt: {name}")
        return self
    
    # ==================== 处理器 ====================
    
    def _handle_initialize(self, request: MCPRequest) -> MCPResponse:
        """处理初始化请求"""
        params = request.params
        self._client_info = params.get("clientInfo", {})
        self._initialized = True
        
        return MCPResponse(
            id=request.id,
            result={
                "protocolVersion": "2024-11-05",
                "serverInfo": {
                    "name": self.name,
                    "version": self.version
                },
                "capabilities": {
                    "tools": {"listChanged": False},
                    "resources": {"listChanged": False, "subscribe": False},
                    "prompts": {"listChanged": False}
                },
                "instructions": self.instructions
            }
        )
    
    def _handle_ping(self, request: MCPRequest) -> MCPResponse:
        """处理心跳请求"""
        return MCPResponse(id=request.id, result={})
    
    def _handle_tools_list(self, request: MCPRequest) -> MCPResponse:
        """处理工具列表请求"""
        return MCPResponse(
            id=request.id,
            result={
                "tools": [t.to_dict() for t in self.tools.values()]
            }
        )
    
    def _handle_tools_call(self, request: MCPRequest) -> MCPResponse:
        """处理工具调用请求"""
        params = request.params
        tool_name = params.get("name")
        arguments = params.get("arguments", {})
        
        tool = self.tools.get(tool_name)
        if not tool:
            return MCPResponse(
                id=request.id,
                error={
                    "code": -32601,
                    "message": f"Tool not found: {tool_name}"
                }
            )
        
        try:
            # 执行工具
            if tool.handler:
                if asyncio.iscoroutinefunction(tool.handler):
                    # 异步处理器需要在事件循环中运行
                    result = asyncio.run(tool.handler(**arguments))
                else:
                    result = tool.handler(**arguments)
            else:
                result = f"Tool {tool_name} executed with {arguments}"
            
            return MCPResponse(
                id=request.id,
                result={
                    "content": [
                        {
                            "type": "text",
                            "text": str(result)
                        }
                    ]
                }
            )
        except Exception as e:
            logger.error(f"Tool {tool_name} execution failed: {e}")
            return MCPResponse(
                id=request.id,
                error={
                    "code": -32000,
                    "message": f"Tool execution failed: {str(e)}"
                }
            )
    
    def _handle_resources_list(self, request: MCPRequest) -> MCPResponse:
        """处理资源列表请求"""
        return MCPResponse(
            id=request.id,
            result={
                "resources": [r.to_dict() for r in self.resources.values()]
            }
        )
    
    def _handle_resources_read(self, request: MCPRequest) -> MCPResponse:
        """处理资源读取请求"""
        params = request.params
        uri = params.get("uri")
        
        resource = self.resources.get(uri)
        if not resource:
            return MCPResponse(
                id=request.id,
                error={
                    "code": -32601,
                    "message": f"Resource not found: {uri}"
                }
            )
        
        try:
            if resource.handler:
                content = resource.handler()
            else:
                content = f"Resource: {uri}"
            
            return MCPResponse(
                id=request.id,
                result={
                    "contents": [
                        {
                            "uri": uri,
                            "mimeType": resource.mimeType,
                            "text": str(content)
                        }
                    ]
                }
            )
        except Exception as e:
            return MCPResponse(
                id=request.id,
                error={
                    "code": -32000,
                    "message": f"Resource read failed: {str(e)}"
                }
            )
    
    def _handle_prompts_list(self, request: MCPRequest) -> MCPResponse:
        """处理提示词列表请求"""
        return MCPResponse(
            id=request.id,
            result={
                "prompts": [p.to_dict() for p in self.prompts.values()]
            }
        )
    
    def _handle_prompts_get(self, request: MCPRequest) -> MCPResponse:
        """处理提示词获取请求"""
        params = request.params
        name = params.get("name")
        arguments = params.get("arguments", {})
        
        prompt = self.prompts.get(name)
        if not prompt:
            return MCPResponse(
                id=request.id,
                error={
                    "code": -32601,
                    "message": f"Prompt not found: {name}"
                }
            )
        
        try:
            if prompt.handler:
                content = prompt.handler(arguments)
            else:
                content = f"Prompt: {name} with {arguments}"
            
            return MCPResponse(
                id=request.id,
                result={
                    "messages": [
                        {
                            "role": "user",
                            "content": {
                                "type": "text",
                                "text": str(content)
                            }
                        }
                    ]
                }
            )
        except Exception as e:
            return MCPResponse(
                id=request.id,
                error={
                    "code": -32000,
                    "message": f"Prompt get failed: {str(e)}"
                }
            )
    
    # ==================== 请求处理 ====================
    
    def handle_request(self, data: Union[str, Dict]) -> MCPResponse:
        """
        处理 MCP 请求
        
        Args:
            data: JSON 字符串或解析后的字典
        
        Returns:
            MCPResponse
        """
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except json.JSONDecodeError as e:
                return MCPResponse(
                    id=0,
                    error={
                        "code": -32700,
                        "message": f"Parse error: {str(e)}"
                    }
                )
        
        request = MCPRequest(
            id=data.get("id", 0),
            method=data.get("method", ""),
            params=data.get("params", {})
        )
        
        # 检查方法
        handler = self._request_handlers.get(request.method)
        if not handler:
            return MCPResponse(
                id=request.id,
                error={
                    "code": -32601,
                    "message": f"Method not found: {request.method}"
                }
            )
        
        # 执行处理器
        try:
            return handler(request)
        except Exception as e:
            logger.error(f"Request handler failed: {e}")
            return MCPResponse(
                id=request.id,
                error={
                    "code": -32000,
                    "message": f"Internal error: {str(e)}"
                }
            )
    
    # ==================== 运行服务器 ====================
    
    def run(
        self,
        transport: str = "stdio",
        host: str = "127.0.0.1",
        port: int = 8765,
        path: str = "/mcp"
    ):
        """
        运行 MCP 服务器
        
        Args:
            transport: 传输方式 ("stdio" 或 "http")
            host: HTTP 服务器主机
            port: HTTP 服务器端口
            path: HTTP 服务器路径
        """
        if transport == "stdio":
            self._run_stdio()
        elif transport == "http":
            self._run_http(host, port, path)
        else:
            raise ValueError(f"Unsupported transport: {transport}")
    
    def _run_stdio(self):
        """通过 stdio 运行 (标准输入输出)"""
        logger.info(f"Starting MCP server (stdio): {self.name}")
        
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            
            response = self.handle_request(line)
            print(json.dumps(response.to_dict()), flush=True)
    
    def _run_http(self, host: str, port: int, path: str):
        """通过 HTTP 运行"""     #短连接
        try:
            from http.server import HTTPServer, BaseHTTPRequestHandler
            import socketserver
            
            class MCPHandler(BaseHTTPRequestHandler):
                def __init__(self, *args, server=None, **kwargs):
                    self.server_instance = server
                    super().__init__(*args, **kwargs)
                
                def do_GET(self):
                    if self.path == "/health":
                        self.send_response(200)
                        self.send_header("Content-Type", "application/json")
                        self.end_headers()
                        self.wfile.write(json.dumps({
                            "status": "ok",
                            "name": self.server_instance.name,
                            "version": self.server_instance.version
                        }).encode())
                    else:
                        self.send_response(404)
                        self.end_headers()
                
                def do_POST(self):
                    if self.path != path:
                        self.send_response(404)
                        self.end_headers()
                        return
                    
                    content_length = int(self.headers.get('Content-Length', 0))
                    body = self.rfile.read(content_length)
                    
                    response = self.server_instance.handle_request(body.decode())
                    
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps(response.to_dict()).encode())
            
            # 创建服务器
            class MCPHTTPServer(HTTPServer):
                def __init__(self, server_address, RequestHandlerClass, mcp_server):
                    self.name = mcp_server.name
                    self.version = mcp_server.version
                    self.mcp_server = mcp_server
                    super().__init__(server_address, RequestHandlerClass)
                
                def handle_request(self, data):
                    return self.mcp_server.handle_request(data)
            
            server = MCPHTTPServer((host, port), MCPHandler, self)
            logger.info(f"Starting MCP server (HTTP) on http://{host}:{port}{path}")
            server.serve_forever()
            
        except ImportError:
            logger.error("HTTP server requires Python's http.server module")
            raise
    
    # ==================== 信息获取 ====================
    
    def get_info(self) -> Dict[str, Any]:
        """获取服务器信息"""
        return {
            "name": self.name,
            "version": self.version,
            "tools": len(self.tools),
            "resources": len(self.resources),
            "prompts": len(self.prompts),
            "initialized": self._initialized,
            "client_info": self._client_info
        }
    
    def list_tools(self) -> List[Dict[str, Any]]:
        """列出所有工具"""
        return [t.to_dict() for t in self.tools.values()]
    
    def list_resources(self) -> List[Dict[str, Any]]:
        """列出所有资源"""
        return [r.to_dict() for r in self.resources.values()]
    
    def list_prompts(self) -> List[Dict[str, Any]]:
        """列出所有提示词"""
        return [p.to_dict() for p in self.prompts.values()]


# ==================== 便捷函数 ====================

def create_mcp_server(
    name: str = "MCP Server",
    version: str = "1.0.0",
    instructions: Optional[str] = None
) -> MCPServer:
    """创建 MCP 服务器"""
    return MCPServer(name=name, version=version, instructions=instructions)


def run_mcp_server(
    tools: Dict[str, Callable],
    transport: str = "stdio",
    host: str = "127.0.0.1",
    port: int = 8765,
    path: str = "/mcp"
):
    """
    快速运行 MCP 服务器
    
    Args:
        tools: {name: handler} 工具字典
        transport: 传输方式
        host: HTTP 主机
        port: HTTP 端口
        path: HTTP 路径
    """
    server = MCPServer()
    
    for name, handler in tools.items():
        server.add_tool(
            name=name,
            description=f"Tool: {name}",
            parameters={"type": "object", "properties": {}},
            handler=handler
        )
    
    server.run(transport=transport, host=host, port=port, path=path)


# ==================== 内置工具示例 ====================

def create_finance_mcp_server() -> MCPServer:
    """创建金融 MCP 服务器"""
    server = MCPServer(
        name="Finance MCP Server",
        version="1.0.0",
        instructions="Financial data and analysis tools"
    )
    
    @server.tool(
        name="get_stock_price",
        description="获取股票当前价格",
        parameters={
            "symbol": {
                "type": "string",
                "description": "股票代码 (如 AAPL)"
            }
        }
    )
    def get_stock_price(symbol: str) -> str:
        """获取股票价格"""
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            info = ticker.info
            return f"{symbol}: ${info.get('regularMarketPrice', 0):.2f}"
        except Exception as e:
            return f"Error getting price: {e}"
    
    @server.tool(
        name="calculate_rsi",
        description="计算RSI指标",
        parameters={
            "symbol": {"type": "string", "description": "股票代码"},
            "period": {"type": "integer", "description": "周期 (默认14)"}
        }
    )
    def calculate_rsi(symbol: str, period: int = 14) -> str:
        """计算RSI"""
        try:
            import yfinance as yf
            import pandas as pd
            
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="2mo")
            
            if hist.empty:
                return f"No data for {symbol}"
            
            close = hist['Close']
            delta = close.diff()
            gain = delta.where(delta > 0, 0).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            current_rsi = rsi.iloc[-1] if not rsi.empty else 0
            signal = "Overbought" if current_rsi > 70 else "Oversold" if current_rsi < 30 else "Neutral"
            
            return f"{symbol} RSI({period}): {current_rsi:.2f} - {signal}"
        except Exception as e:
            return f"Error calculating RSI: {e}"
    
    return server