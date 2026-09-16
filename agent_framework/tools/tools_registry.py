"""
Tools Registry - 工具注册中心

功能:
1. 工具注册和发现
2. 工具分类管理
3. 工具生命周期管理
4. 工具参数验证
5. 工具调用统计

设计模式:
- 单例模式 (全局唯一注册中心)
- 工厂模式 (创建工具实例)
- 观察者模式 (工具事件通知)
"""

from typing import Dict, Any, Optional, List, Callable, Type, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
import inspect
import json
from datetime import datetime
from functools import wraps

logger = logging.getLogger(__name__)


class ToolCategory(Enum):
    """工具分类"""
    MARKET = "market"          # 市场数据
    ANALYSIS = "analysis"      # 分析工具
    TRADING = "trading"        # 交易工具
    SEARCH = "search"          # 搜索工具
    UTILITY = "utility"        # 工具类
    SOCIAL = "social"          # 社交媒体
    NEWS = "news"              # 新闻
    RISK = "risk"              # 风险管理
    CUSTOM = "custom"          # 自定义


@dataclass
class ToolParameter:
    """工具参数定义"""
    name: str
    type: str  # string, number, integer, boolean, object, array
    description: str
    required: bool = False
    default: Any = None
    enum: Optional[List[Any]] = None
    
    def to_schema(self) -> Dict[str, Any]:
        """转换为JSON Schema"""
        schema = {
            "type": self.type,
            "description": self.description
        }
        if self.enum:
            schema["enum"] = self.enum
        return schema


@dataclass
class ToolDefinition:
    """工具定义"""
    name: str
    description: str
    category: ToolCategory
    parameters: Dict[str, ToolParameter]
    handler: Callable
    metadata: Dict[str, Any] = field(default_factory=dict)
    is_async: bool = False
    version: str = "1.0.0"
    author: str = "system"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    call_count: int = 0
    last_call: Optional[str] = None
    
    def to_schema(self) -> Dict[str, Any]:
        """转换为JSON Schema (用于LLM)"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        name: param.to_schema()
                        for name, param in self.parameters.items()
                   },
                    "required": [
                        name for name, param in self.parameters.items()
                        if param.required
                ],
            },
        },
    }
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category.value,
            "parameters": {
                name: {
                    "type": param.type,
                    "description": param.description,
                    "required": param.required
                }
                for name, param in self.parameters.items()
            },
            "version": self.version,
            "call_count": self.call_count,
            "created_at": self.created_at
        }


class ToolsRegistry:
    """
    工具注册中心 - 单例模式
    
    管理所有Agent可用的工具
    
    使用示例:
        registry = ToolsRegistry()
        
        # 注册工具
        @registry.register("get_price", "获取股票价格", ToolCategory.MARKET)
        def get_price(symbol: str) -> str:
            return f"${symbol}: 150.00"
        
        # 获取工具
        tool = registry.get("get_price")
        
        # 调用工具
        result = registry.call("get_price", {"symbol": "AAPL"})
        
        # 列出所有工具
        tools = registry.list_tools()
    """
    
    _instance: Optional['ToolsRegistry'] = None
    
    def __new__(cls) -> 'ToolsRegistry':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._tools: Dict[str, ToolDefinition] = {}
        self._by_category: Dict[ToolCategory, List[str]] = {
            cat: [] for cat in ToolCategory
        }
        self._event_handlers: Dict[str, List[Callable]] = {}
        self._initialized = True
        
        # 注册内置工具
        self._register_builtin_tools()
        
        logger.info("ToolsRegistry initialized")
    
    def _register_builtin_tools(self):
        """注册内置工具"""
        # 计算器工具
        @self.register(
            name="calculate",
            description="执行数学计算",
            category=ToolCategory.UTILITY,
            parameters={
                "expression": ToolParameter(
                    name="expression",
                    type="string",
                    description="数学表达式，如 '2 + 2 * 3'",
                    required=True
                )
            }
        )
        def calculate(expression: str) -> str:
            """执行数学计算"""
            try:
                # 安全计算
                allowed = set("0123456789+-*/().% ")
                if not all(c in allowed for c in expression):
                    return "Error: Invalid characters in expression"
                result = eval(expression, {"__builtins__": {}})
                return f"Result: {result}"
            except Exception as e:
                return f"Error: {str(e)}"
        
        # 格式化工具
        @self.register(
            name="format_json",
            description="格式化JSON字符串",
            category=ToolCategory.UTILITY,
            parameters={
                "json_str": ToolParameter(
                    name="json_str",
                    type="string",
                    description="要格式化的JSON字符串",
                    required=True
                ),
                "indent": ToolParameter(
                    name="indent",
                    type="integer",
                    description="缩进空格数",
                    required=False,
                    default=2
                )
            }
        )
        def format_json(json_str: str, indent: int = 2) -> str:
            """格式化JSON"""
            try:
                data = json.loads(json_str)
                return json.dumps(data, indent=indent, ensure_ascii=False)
            except Exception as e:
                return f"Error: {str(e)}"
        
        # 当前时间工具
        @self.register(
            name="get_current_time",
            description="获取当前时间",
            category=ToolCategory.UTILITY,
            parameters={
                "timezone": ToolParameter(
                    name="timezone",
                    type="string",
                    description="时区 (如 'Asia/Shanghai', 'America/New_York')",
                    required=False,
                    default="UTC"
                )
            }
        )
        def get_current_time(timezone: str = "UTC") -> str:
            """获取当前时间"""
            try:
                from datetime import datetime
                import pytz
                tz = pytz.timezone(timezone) if timezone != "UTC" else pytz.UTC
                now = datetime.now(tz)
                return now.strftime("%Y-%m-%d %H:%M:%S %Z")
            except Exception:
                return datetime.now().isoformat()
    
    def register(
        self,
        name: Optional[str] = None,
        description: Optional[str] = None,
        category: ToolCategory = ToolCategory.CUSTOM,
        parameters: Optional[Dict[str, ToolParameter]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        version: str = "1.0.0",
        author: str = "system"
    ) -> Callable:
        """
        工具注册装饰器
        
        使用示例:
            @registry.register("get_price", "获取股票价格", ToolCategory.MARKET)
            def get_price(symbol: str) -> str:
                return f"Price: 150.00"
        """
        def decorator(func: Callable) -> Callable:
            tool_name = name or func.__name__
            tool_desc = description or func.__doc__ or "No description"
            tool_params = parameters or self._infer_parameters(func)
            is_async = inspect.iscoroutinefunction(func)
            
            tool_def = ToolDefinition(
                name=tool_name,
                description=tool_desc,
                category=category,
                parameters=tool_params,
                handler=func,
                metadata=metadata or {},
                is_async=is_async,
                version=version,
                author=author
            )
            
            self._tools[tool_name] = tool_def
            self._by_category[category].append(tool_name)
            
            logger.info(f"Registered tool: {tool_name} ({category.value})")
            return func
        return decorator
    
    def _infer_parameters(self, func: Callable) -> Dict[str, ToolParameter]:
        """从函数签名推断参数"""
        sig = inspect.signature(func)
        doc = inspect.getdoc(func) or ""
        
        params = {}
        for param_name, param in sig.parameters.items():
            if param_name in ('self', 'cls'):
                continue
            
            # 尝试从文档提取描述
            desc = f"Parameter {param_name}"
            if doc:
                lines = doc.split('\n')
                for line in lines:
                    if f":param {param_name}:" in line:
                        desc = line.split(':', 2)[-1].strip()
                        break
            
            param_type = "string"
            if param.annotation != inspect.Parameter.empty:
                if param.annotation in (int, float):
                    param_type = "number" if param.annotation == float else "integer"
                elif param.annotation == bool:
                    param_type = "boolean"
                elif param.annotation == dict:
                    param_type = "object"
                elif param.annotation == list:
                    param_type = "array"
            
            params[param_name] = ToolParameter(
                name=param_name,
                type=param_type,
                description=desc,
                required=param.default == inspect.Parameter.empty,
                default=param.default if param.default != inspect.Parameter.empty else None
            )
        
        return params
    
    def get(self, name: str) -> Optional[ToolDefinition]:
        """获取工具定义"""
        return self._tools.get(name)
    
    def list_tools(
        self,
        category: Optional[ToolCategory] = None,
        include_hidden: bool = False
    ) -> List[Dict[str, Any]]:
        """列出所有工具"""
        tools = []
        for name, tool in self._tools.items():
            if category and tool.category != category:
                continue
            tools.append(tool.to_dict())
        return tools


    
    def get_tools(self, names: List[str], api_keys: Dict[str, str]) -> List[ToolDefinition]:
        """根据名字列表获取工具"""
        return [self._tools[name] for name in names if name in self._tools]
    
    def list_categories(self) -> List[str]:
        """列出所有分类"""
        return [cat.value for cat in ToolCategory]
    
    def call(
        self,
        name: str,
        arguments: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        调用工具
        
        Args:
            name: 工具名称
            arguments: 参数
            context: 上下文 (用于授权等)
        
        Returns:
            {
                'success': bool,
                'result': Any,
                'error': str,
                'call_id': str
            }
        """
        tool = self.get(name)
        if not tool:
            return {
                'success': False,
                'error': f"Tool '{name}' not found",
                'call_id': None
            }
        
        # 验证参数
        validation = self._validate_arguments(tool, arguments)
        if not validation['valid']:
            return {
                'success': False,
                'error': validation['error'],
                'call_id': None
            }
        
        # 执行工具
        try:
            call_id = f"{name}_{datetime.now().timestamp()}"
            
            if tool.is_async:
                # 异步执行
                import asyncio
                result = asyncio.run(tool.handler(**arguments))
            else:
                result = tool.handler(**arguments)
            
            # 更新统计
            tool.call_count += 1
            tool.last_call = datetime.now().isoformat()
            
            self._trigger_event('tool_called', {
                'name': name,
                'arguments': arguments,
                'result': result,
                'call_id': call_id
            })
            
            return {
                'success': True,
                'result': result,
                'error': None,
                'call_id': call_id
            }
            
        except Exception as e:
            logger.error(f"Tool {name} execution failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'call_id': None
            }
    
    def _validate_arguments(
        self,
        tool: ToolDefinition,
        arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """验证工具参数"""
        # 检查必填参数
        for name, param in tool.parameters.items():
            if param.required and name not in arguments:
                return {
                    'valid': False,
                    'error': f"Missing required parameter: {name}"
                }
        
        # 类型检查
        type_map = {
            'string': str,
            'number': (int, float),
            'integer': int,
            'boolean': bool,
            'object': dict,
            'array': list
        }
        
        for name, value in arguments.items():
            if name in tool.parameters:
                param = tool.parameters[name]
                expected_type = type_map.get(param.type)
                if expected_type and not isinstance(value, expected_type):
                    return {
                        'valid': False,
                        'error': f"Parameter '{name}' should be {param.type}, got {type(value).__name__}"
                    }
        
        return {'valid': True}
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        total_calls = sum(t.call_count for t in self._tools.values())
        
        return {
            'total_tools': len(self._tools),
            'by_category': {
                cat.value: len(names)
                for cat, names in self._by_category.items()
            },
            'total_calls': total_calls,
            'most_used': sorted(
                [(name, t.call_count) for name, t in self._tools.items()],
                key=lambda x: x[1],
                reverse=True
            )[:5]
        }
    
    def on(self, event_type: str, handler: Callable) -> 'ToolsRegistry':
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
    
    def clear(self) -> None:
        """清空所有工具"""
        self._tools = {}
        self._by_category = {cat: [] for cat in ToolCategory}
        logger.info("ToolsRegistry cleared")
    
    def to_schemas(self) -> List[Dict[str, Any]]:
        """获取所有工具的JSON Schema (用于LLM)"""
        return [t.to_schema() for t in self._tools.values()]


# ==================== 全局实例 ====================

_registry: Optional[ToolsRegistry] = None


def get_tools_registry() -> ToolsRegistry:
    """获取全局工具注册中心"""
    global _registry
    if _registry is None:
        _registry = ToolsRegistry()
    return _registry


def register_tool(
    name: Optional[str] = None,
    description: Optional[str] = None,
    category: ToolCategory = ToolCategory.CUSTOM
) -> Callable:
    """便捷工具注册装饰器"""
    return get_tools_registry().register(name, description, category)


def get_tool(name: str) -> Optional[ToolDefinition]:
    """获取工具"""
    return get_tools_registry().get(name)


def list_tools(category: Optional[ToolCategory] = None) -> List[Dict[str, Any]]:
    """列出工具"""
    return get_tools_registry().list_tools(category)


def call_tool(name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """调用工具"""
    return get_tools_registry().call(name, arguments)