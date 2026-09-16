# my_tools/sandbox.py
"""
沙箱执行环境 - 安全运行不受信任的代码

DeerFlow风格的安全沙箱:
- 资源限制 (CPU/内存/时间)
- 文件系统隔离
- 网络访问控制
- 代码静态分析
"""

import ast
import resource
import signal
import time
import os
import sys
from typing import Any, Dict, Optional, Callable
from contextlib import contextmanager
import logging

logger = logging.getLogger(__name__)


class SandboxError(Exception):
    """沙箱执行错误"""
    pass


class SandboxConfig:
    """沙箱配置"""
    
    def __init__(
        self,
        max_cpu_time: float = 5.0,      # 最大CPU时间(秒)
        max_memory_mb: int = 256,        # 最大内存(MB)
        max_files: int = 10,             # 最大打开文件数
        allow_network: bool = False,     # 允许网络
        allow_filesystem: bool = False,  # 允许文件系统
        allow_imports: List[str] = None, # 允许的导入模块
        allow_builtins: List[str] = None # 允许的内置函数
    ):
        self.max_cpu_time = max_cpu_time
        self.max_memory_mb = max_memory_mb
        self.max_files = max_files
        self.allow_network = allow_network
        self.allow_filesystem = allow_filesystem
        self.allow_imports = allow_imports or [
            'math', 'random', 'collections', 'itertools',
            'functools', 'datetime', 'typing', 'json'
        ]
        self.allow_builtins = allow_builtins or [
            'abs', 'all', 'any', 'bool', 'dict', 'enumerate',
            'filter', 'float', 'int', 'len', 'list', 'map',
            'max', 'min', 'range', 'round', 'sum', 'zip',
            'isinstance', 'type', 'tuple', 'set', 'str'
        ]


class CodeAnalyzer(ast.NodeVisitor):
    """静态代码分析 - 检测危险操作"""
    
    DANGEROUS_ATTRS = {
        '__import__', 'exec', 'eval', 'compile',
        'open', 'file', 'input', 'raw_input',
        'system', 'popen', 'subprocess',
        'globals', 'locals', 'vars', 'dir',
        'getattr', 'setattr', 'delattr',
        '__class__', '__subclasses__', '__bases__',
        '__globals__', '__code__', '__closure__'
    }
    
    DANGEROUS_MODULES = {
        'os', 'sys', 'subprocess', 'shutil', 'socket',
        'requests', 'urllib', 'ftplib', 'smtplib',
        'ctypes', 'multiprocessing', 'threading',
        'pickle', 'marshal', 'importlib'
    }
    
    def __init__(self, config: SandboxConfig):
        self.config = config
        self.errors = []
        self.imports = set()
    
    def visit_Import(self, node):
        for alias in node.names:
            module = alias.name.split('.')[0]
            if module in self.DANGEROUS_MODULES:
                self.errors.append(f"Forbidden import: {module}")
            elif module not in self.config.allow_imports:
                self.errors.append(f"Import not allowed: {module}")
            self.imports.add(module)
        self.generic_visit(node)
    
    def visit_ImportFrom(self, node):
        if node.module:
            module = node.module.split('.')[0]
            if module in self.DANGEROUS_MODULES:
                self.errors.append(f"Forbidden import from: {module}")
            elif module not in self.config.allow_imports:
                self.errors.append(f"Import from not allowed: {module}")
            self.imports.add(module)
        self.generic_visit(node)
    
    def visit_Call(self, node):
        # 检查危险函数调用
        if isinstance(node.func, ast.Name):
            if node.func.id in ['exec', 'eval', 'compile', '__import__']:
                self.errors.append(f"Forbidden function: {node.func.id}")
        elif isinstance(node.func, ast.Attribute):
            if node.func.attr in self.DANGEROUS_ATTRS:
                self.errors.append(f"Forbidden attribute: {node.func.attr}")
        self.generic_visit(node)
    
    def visit_Attribute(self, node):
        if node.attr in self.DANGEROUS_ATTRS:
            self.errors.append(f"Forbidden attribute access: {node.attr}")
        self.generic_visit(node)
    
    def analyze(self, code: str) -> bool:
        """分析代码，返回是否安全"""
        try:
            tree = ast.parse(code)
            self.visit(tree)
            return len(self.errors) == 0
        except SyntaxError as e:
            self.errors.append(f"Syntax error: {e}")
            return False


class SandboxExecutor:
    """沙箱执行器"""
    
    def __init__(self, config: Optional[SandboxConfig] = None):
        self.config = config or SandboxConfig()
        self._timeout_occurred = False
    
    def _limit_resources(self):
        """设置资源限制"""
        # CPU时间限制
        resource.setrlimit(
            resource.RLIMIT_CPU,
            (self.config.max_cpu_time, self.config.max_cpu_time + 1)
        )
        
        # 内存限制
        max_memory = self.config.max_memory_mb * 1024 * 1024
        resource.setrlimit(
            resource.RLIMIT_AS,
            (max_memory, max_memory)
        )
        
        # 文件限制
        resource.setrlimit(
            resource.RLIMIT_NOFILE,
            (self.config.max_files, self.config.max_files)
        )
    
    def _timeout_handler(self, signum, frame):
        """超时处理"""
        self._timeout_occurred = True
        raise TimeoutError("Execution timed out")
    
    @contextmanager
    def _sandbox_context(self):
        """沙箱上下文管理器"""
        old_signal = signal.signal(signal.SIGALRM, self._timeout_handler)
        signal.alarm(int(self.config.max_cpu_time) + 1)
        
        try:
            self._limit_resources()
            yield
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_signal)
    
    def _create_safe_globals(self) -> Dict[str, Any]:
        """创建安全的全局命名空间"""
        import math, random, collections, itertools, functools, datetime, typing, json
        
        safe_modules = {
            'math': math,
            'random': random,
            'collections': collections,
            'itertools': itertools,
            'functools': functools,
            'datetime': datetime,
            'typing': typing,
            'json': json,
        }
        
        # 仅允许安全的内置函数
        safe_builtins = {
            name: __builtins__[name]
            for name in self.config.allow_builtins
            if name in __builtins__
        }
        
        safe_globals = {
            '__builtins__': safe_builtins,
            '__name__': '__sandbox__',
            '__doc__': None,
        }
        safe_globals.update(safe_modules)
        
        return safe_globals
    
    def execute(
        self,
        code: str,
        context: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        在沙箱中执行代码
        
        Args:
            code: 要执行的代码
            context: 上下文变量
            timeout: 超时时间(秒)
        
        Returns:
            {
                'success': bool,
                'result': Any,
                'error': str,
                'timeout': bool,
                'output': str
            }
        """
        # 1. 静态分析
        analyzer = CodeAnalyzer(self.config)
        if not analyzer.analyze(code):
            return {
                'success': False,
                'error': f"Code analysis failed: {', '.join(analyzer.errors)}",
                'result': None,
                'timeout': False,
                'output': ''
            }
        
        # 2. 准备执行环境
        safe_globals = self._create_safe_globals()
        safe_locals = context or {}
        
        # 3. 捕获输出
        from io import StringIO
        import sys
        
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        
        try:
            # 4. 执行 (带超时)
            if timeout:
                self.config.max_cpu_time = timeout
            
            with self._sandbox_context():
                exec(code, safe_globals, safe_locals)
            
            output = sys.stdout.getvalue()
            
            return {
                'success': True,
                'result': safe_locals.get('result', None),
                'error': None,
                'timeout': False,
                'output': output
            }
            
        except TimeoutError:
            return {
                'success': False,
                'result': None,
                'error': f"Execution timed out after {self.config.max_cpu_time}s",
                'timeout': True,
                'output': sys.stdout.getvalue()
            }
        except Exception as e:
            return {
                'success': False,
                'result': None,
                'error': str(e),
                'timeout': False,
                'output': sys.stdout.getvalue()
            }
        finally:
            sys.stdout = old_stdout
    
    def evaluate_expression(
        self,
        expr: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """在沙箱中计算表达式"""
        code = f"result = ({expr})"
        return self.execute(code, context)


class TradingSandbox:
    """交易策略沙箱 - 专门用于策略测试"""
    
    def __init__(self, config: Optional[SandboxConfig] = None):
        self.config = config or SandboxConfig()
        self.executor = SandboxExecutor(config)
        
        # 默认注入的上下文
        self.default_context = {
            'pd': __import__('pandas'),
            'np': __import__('numpy'),
            'ta': None,  # 技术指标库
        }
    
    def test_strategy(
        self,
        strategy_code: str,
        market_data: pd.DataFrame,
        initial_capital: float = 100000.0
    ) -> Dict[str, Any]:
        """测试交易策略"""
        
        # 注入数据
        context = {
            'data': market_data,
            'capital': initial_capital,
            'position': 0,
            'trades': [],
            'equity': [initial_capital],
            'pd': __import__('pandas'),
            'np': __import__('numpy'),
        }
        context.update(self.default_context)
        
        # 执行策略
        result = self.executor.execute(strategy_code, context)
        
        if result['success']:
            # 计算绩效
            equity = context.get('equity', [initial_capital])
            trades = context.get('trades', [])
            
            return {
                'success': True,
                'final_equity': equity[-1] if equity else initial_capital,
                'total_return': (equity[-1] - initial_capital) / initial_capital * 100,
                'trades': trades,
                'equity_curve': equity,
                'error': None
            }
        else:
            return {
                'success': False,
                'error': result['error'],
                'final_equity': initial_capital,
                'total_return': 0,
                'trades': [],
                'equity_curve': [initial_capital]
            }


# 便捷函数
def create_sandbox(
    max_cpu_time: float = 5.0,
    max_memory_mb: int = 256,
    allow_network: bool = False
) -> SandboxExecutor:
    """创建沙箱执行器"""
    config = SandboxConfig(
        max_cpu_time=max_cpu_time,
        max_memory_mb=max_memory_mb,
        allow_network=allow_network
    )
    return SandboxExecutor(config)


def run_safe_code(
    code: str,
    context: Optional[Dict[str, Any]] = None,
    timeout: float = 5.0
) -> Dict[str, Any]:
    """安全执行代码"""
    executor = create_sandbox(max_cpu_time=timeout)
    return executor.execute(code, context)