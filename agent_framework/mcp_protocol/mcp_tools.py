"""
MCP Tools - MCP 客户端和工具封装

功能:
1. MCP 客户端连接
2. 工具调用封装
3. 资源读取封装
4. 与 Agent 框架集成
"""

from __future__ import annotations

import json
import logging
import urllib.request
import urllib.error
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# 检查 MCP 客户端是否可用
MCP_AVAILABLE = False
try:
    from mcp import ClientSession, StdioServerParameters
    MCP_AVAILABLE = True
except ImportError:
    pass


@dataclass
class MCPToolWrapper:
    """MCP 工具包装器 (用于 Agent 调用)"""
    name: str
    description: str
    parameters: Dict[str, Any]
    client: Any  # MCP ClientSession
    _call_count: int = 0
    
    def __call__(self, **kwargs) -> str:
        """调用 MCP 工具"""
        self._call_count += 1
        
        if not self.client:
            return f"Error: MCP client not connected for tool {self.name}"
        
        try:
            result = self.client.call_tool(self.name, kwargs)
            return self._extract_result(result)
        except Exception as e:
            return f"Error calling {self.name}: {str(e)}"
    
    def _extract_result(self, result: Any) -> str:
        """提取工具结果"""
        if hasattr(result, 'content'):
            # MCP 返回格式
            parts = []
            for content in result.content:
                if hasattr(content, 'text'):
                    parts.append(content.text)
                elif hasattr(content, 'data'):
                    parts.append(str(content.data))
            return '\n'.join(parts)
        return str(result)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取调用统计"""
        return {
            'name': self.name,
            'call_count': self._call_count
        }


class MCPTools:
    """
    MCP 工具集 - 连接 MCP 服务器并暴露工具
    
    使用示例:
        tools = MCPTools("http://localhost:8765/mcp")
        tools.connect()
        
        # 获取所有工具
        tool_list = tools.list_tools()
        
        # 调用工具
        result = tools.call_tool("get_stock_price", {"symbol": "AAPL"})
        
        # 转换为 Agent 工具
        agent_tools = tools.to_agent_tools()
    """
    
    def __init__(
        self,
        server_url: Optional[str] = None,
        transport: str = "http",
        command: Optional[str] = None,
        args: Optional[List[str]] = None,
        env: Optional[Dict[str, str]] = None
    ):
        """
        初始化 MCP 工具集
        
        Args:
            server_url: HTTP 服务器 URL (http模式)
            transport: 传输方式 ('http' 或 'stdio')
            command: stdio 模式的命令
            args: stdio 模式的参数
            env: stdio 模式的环境变量
        """
        self.server_url = server_url
        self.transport = transport
        self.command = command
        self.args = args or []
        self.env = env or {}
        
        self._client = None
        self._session = None
        self._tools: Dict[str, Dict[str, Any]] = {}
        self._connected = False
    
    def connect(self) -> bool:
        """连接 MCP 服务器"""
        if self._connected:
            return True
        
        try:
            if self.transport == "http" and self.server_url:
                self._client = MCPHttpClient(self.server_url)
                self._connected = self._client.connect()
            elif self.transport == "stdio" and MCP_AVAILABLE:
                self._client = MCPStdioClient(self.command, self.args, self.env)
                self._connected = self._client.connect()
            else:
                logger.error(f"Unsupported transport or missing URL: {self.transport}")
                return False
            
            if self._connected:
                # 获取工具列表
                self._refresh_tools()
                logger.info(f"Connected to MCP server, {len(self._tools)} tools available")
            
            return self._connected
        except Exception as e:
            logger.error(f"Failed to connect to MCP server: {e}")
            return False
    
    def disconnect(self):
        """断开连接"""
        if self._client:
            self._client.disconnect()
        self._connected = False
        self._tools = {}
    
    def _refresh_tools(self):
        """刷新工具列表"""
        if not self._client:
            return
        
        try:
            tools = self._client.list_tools()
            self._tools = {
                tool['name']: tool
                for tool in tools
            }
        except Exception as e:
            logger.error(f"Failed to list tools: {e}")
    
    def list_tools(self) -> List[Dict[str, Any]]:
        """列出所有工具"""
        if not self._connected:
            self.connect()
        return list(self._tools.values())
    
    def get_tool(self, name: str) -> Optional[Dict[str, Any]]:
        """获取工具定义"""
        if not self._connected:
            self.connect()
        return self._tools.get(name)
    
    def call_tool(self, name: str, arguments: Dict[str, Any]) -> str:
        """
        调用工具
        
        Args:
            name: 工具名称
            arguments: 参数
        
        Returns:
            工具结果
        """
        if not self._connected:
            self.connect()
        
        if not self._client:
            return f"Error: Not connected to MCP server"
        
        try:
            return self._client.call_tool(name, arguments)
        except Exception as e:
            return f"Error calling {name}: {str(e)}"
    
    def to_agent_tools(self) -> List[Callable]:
        """
        转换为 Agent 可用的工具列表
        
        Returns:
            工具函数列表，可直接传给 Agent
        """
        if not self._connected:
            self.connect()
        
        tools = []
        
        for name, tool_def in self._tools.items():
            wrapper = MCPToolWrapper(
                name=name,
                description=tool_def.get('description', ''),
                parameters=tool_def.get('inputSchema', {}),
                client=self._client
            )
            tools.append(wrapper)
        
        return tools
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'connected': self._connected,
            'tools_count': len(self._tools),
            'transport': self.transport,
            'server_url': self.server_url
        }


class MCPHttpClient:
    """MCP HTTP 客户端"""
    
    def __init__(self, url: str):
        self.url = url.rstrip('/')
        self._connected = False
    
    def connect(self) -> bool:
        """建立连接 (健康检查)"""
        try:
            req = urllib.request.Request(
                f"{self.url}/health",
                method="GET"
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                if resp.status == 200:
                    self._connected = True
                    return True
        except Exception as e:
            logger.debug(f"Health check failed: {e}")
        return False
    
    def disconnect(self):
        self._connected = False
    
    def list_tools(self) -> List[Dict[str, Any]]:
        """获取工具列表"""
        if not self._connected:
            return []
        
        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/list",
            "params": {}
        }
        
        response = self._send_request(request)
        if response and 'result' in response:
            return response['result'].get('tools', [])
        return []
    
    def call_tool(self, name: str, arguments: Dict[str, Any]) -> str:
        """调用工具"""
        if not self._connected:
            return "Error: Not connected"
        
        request = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": name,
                "arguments": arguments
            }
        }
        
        response = self._send_request(request)
        if response and 'result' in response:
            content = response['result'].get('content', [])
            texts = [c.get('text', '') for c in content if c.get('type') == 'text']
            return '\n'.join(texts)
        elif response and 'error' in response:
            return f"Error: {response['error'].get('message', 'Unknown')}"
        
        return f"Error: No response from server"
    
    def _send_request(self, request: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """发送请求"""
        try:
            data = json.dumps(request).encode('utf-8')
            req = urllib.request.Request(
                f"{self.url}",
                data=data,
                headers={
                    "Content-Type": "application/json"
                },
                method="POST"
            )
            
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode('utf-8'))
        except urllib.error.URLError as e:
            logger.error(f"HTTP request failed: {e}")
            return None
        except Exception as e:
            logger.error(f"Request error: {e}")
            return None


class MCPStdioClient:
    """MCP Stdio 客户端 (使用 subprocess)"""
    
    def __init__(self, command: str, args: List[str], env: Dict[str, str]):
        self.command = command
        self.args = args
        self.env = env
        self._process = None
        self._connected = False
    
    def connect(self) -> bool:
        """启动子进程连接"""
        if not MCP_AVAILABLE:
            logger.error("MCP library not available")
            return False
        
        try:
            import subprocess
            
            # 构建命令
            cmd = [self.command] + self.args
            
            # 设置环境变量
            import os
            env = os.environ.copy()
            env.update(self.env)
            
            self._process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE