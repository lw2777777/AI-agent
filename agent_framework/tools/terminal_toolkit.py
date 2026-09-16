"""
Terminal Toolkit - 终端工具包

提供安全的终端操作工具，用于 Agent 执行系统命令、文件操作等。

功能:
1. 命令执行 (安全沙箱)
2. 文件操作 (读取/写入/列表/删除)
3. 进程管理
4. 环境信息
5. 命令历史

安全设计:
- 命令白名单 (只允许预定义命令)
- 路径限制 (只能在指定目录内操作)
- 超时控制 (防止死循环)
- 输出截断 (防止内存溢出)
- 危险模式检测

使用示例:
    toolkit = TerminalToolkit(
        allowed_commands=['ls', 'cat', 'python'],
        base_path='/tmp/sandbox',
        timeout_seconds=30
    )
    
    # Agent 获取工具
    tools = toolkit.get_tools()
    
    # 执行命令
    result = toolkit.execute_command("ls -la")
    
    # 读取文件
    content = toolkit.read_file("data.txt")
"""

import os
import sys
import subprocess
import shlex
import tempfile
import logging
import time
import re
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

logger = logging.getLogger(__name__)


# ==================== 数据类 ====================

class CommandType(Enum):
    """命令类型"""
    SHELL = "shell"
    PYTHON = "python"
    FILE = "file"
    SYSTEM = "system"


@dataclass
class TerminalCommand:
    """终端命令定义"""
    command: str
    args: List[str] = field(default_factory=list)
    working_dir: Optional[str] = None
    env: Optional[Dict[str, str]] = None
    timeout_seconds: float = 30.0
    max_output_bytes: int = 1024 * 1024  # 1MB
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'command': self.command,
            'args': self.args,
            'working_dir': self.working_dir,
            'timeout_seconds': self.timeout_seconds
        }


@dataclass
class TerminalResult:
    """终端命令结果"""
    success: bool
    stdout: str
    stderr: str
    exit_code: int
    duration: float
    command: str
    truncated: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'success': self.success,
            'stdout': self.stdout,
            'stderr': self.stderr,
            'exit_code': self.exit_code,
            'duration': round(self.duration, 3),
            'command': self.command,
            'truncated': self.truncated
        }
    
    def get_full_output(self) -> str:
        """获取完整输出 (stdout + stderr)"""
        if self.stdout and self.stderr:
            return f"{self.stdout}\n{self.stderr}"
        return self.stdout or self.stderr or ""


# ==================== TerminalToolkit 主类 ====================

class TerminalToolkit:
    """
    终端工具包 - 安全的终端操作
    
    为 Agent 提供安全的命令行操作能力。
    
    安全特性:
    1. 命令白名单: 只允许执行预定义的命令
    2. 路径限制: 只能在 base_path 目录下操作
    3. 超时控制: 防止命令无限执行
    4. 输出截断: 防止输出过大
    5. 危险模式检测: 阻止 rm -rf / 等危险命令
    """
    
    # 默认允许的命令
    DEFAULT_ALLOWED_COMMANDS = [
        # 文件操作
        'ls', 'dir', 'pwd', 'cd', 'cat', 'head', 'tail', 
        'wc', 'find', 'grep', 'sort', 'uniq', 'echo',
        # Python
        'python', 'python3',
        # 系统信息
        'whoami', 'hostname', 'date', 'uptime', 'uname',
        # 网络 (如果允许)
        'ping', 'curl', 'wget',
    ]
    
    # 危险命令模式 (阻止执行)
    DANGEROUS_PATTERNS = [
        r'rm\s+(-rf?|/|/\*)',
        r'>\s*/dev/',
        r'chmod\s+777',
        r'sudo',
        r'su\s+',
        r'passwd',
        r'kill\s+-9',
        r'shutdown',
        r'reboot',
        r'init\s+\d+',
        r'dd\s+if=',
        r'mkfs',
        r'fdisk',
        r'format',
        r'del\s+/f',
        r'rd\s+/s',
        r'curl.*\|\s*(bash|sh)',
        r'wget.*\|\s*(bash|sh)',
        r'python.*-c\s+["\'].*import\s+os',
    ]
    
    def __init__(
        self,
        allowed_commands: Optional[List[str]] = None,
        base_path: Optional[str] = None,
        allow_network: bool = False,
        timeout_seconds: float = 30.0,
        max_output_bytes: int = 1024 * 1024,
        safe_mode: bool = True,
        name: str = "terminal_toolkit"
    ):
        """
        初始化终端工具包
        
        Args:
            allowed_commands: 允许执行的命令列表
            base_path: 基础路径 (所有操作限制在此目录下)
            allow_network: 是否允许网络命令
            timeout_seconds: 默认超时时间
            max_output_bytes: 最大输出字节数
            safe_mode: 是否启用安全模式 (阻止危险命令)
            name: 工具包名称
        """
        self.name = name
        
        # 命令白名单
        self.allowed_commands = set(allowed_commands or self.DEFAULT_ALLOWED_COMMANDS)
        
        # 路径限制
        self.base_path = Path(base_path) if base_path else Path.cwd()
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # 安全配置
        self.allow_network = allow_network
        self.timeout_seconds = timeout_seconds
        self.max_output_bytes = max_output_bytes
        self.safe_mode = safe_mode
        
        # 如果允许网络，添加网络命令
        if allow_network:
            self.allowed_commands.update(['ping', 'curl', 'wget', 'nslookup', 'dig'])
        
        # 编译危险模式正则
        self._dangerous_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.DANGEROUS_PATTERNS
        ]
        
        # 命令历史
        self._history: List[Dict[str, Any]] = []
        self._max_history = 100
        
        # 当前工作目录
        self._cwd = self.base_path
        
        # 缓存
        self._cache: Dict[str, Any] = {}
        
        # 构建 Agno 工具函数
        self._tools: List[Any] = []
        self._build_tools()
        
        logger.info(
            f"TerminalToolkit initialized: base_path={self.base_path}, "
            f"allowed_commands={len(self.allowed_commands)}, safe_mode={safe_mode}"
        )
    
    # ==================== 核心方法 ====================
    
    def _is_safe_command(self, command: str) -> Tuple[bool, str]:
        """
        检查命令是否安全
        
        Returns:
            (is_safe, reason)
        """
        if not self.safe_mode:
            return True, ""
        
        # 检查危险模式
        for pattern in self._dangerous_patterns:
            if pattern.search(command):
                return False, f"Dangerous pattern detected: {pattern.pattern}"
        
        # 检查是否包含管道/重定向等 (需要额外检查)
        if '|' in command or '>' in command or '<' in command:
            # 允许基本的管道，但要检查内容
            parts = command.split('|')
            for part in parts:
                cmd_part = part.strip().split()[0] if part.strip() else ''
                if cmd_part and cmd_part not in self.allowed_commands:
                    return False, f"Command '{cmd_part}' not in allowed list"
        
        return True, ""
    
    def _is_path_allowed(self, path: Path) -> bool:
        """检查路径是否在允许范围内"""
        try:
            resolved = path.resolve()
            base = self.base_path.resolve()
            return str(resolved).startswith(str(base))
        except Exception:
            return False
    
    def _sanitize_path(self, path: str) -> Optional[Path]:
        """清理并验证路径"""
        try:
            # 处理相对路径
            if not Path(path).is_absolute():
                p = self._cwd / path
            else:
                p = Path(path)
            
            # 规范化路径
            p = p.resolve()
            
            # 检查是否在 base_path 内
            if not self._is_path_allowed(p):
                logger.warning(f"Path not allowed: {p}")
                return None
            
            return p
        except Exception as e:
            logger.error(f"Path sanitization failed: {e}")
            return None
    
    def _truncate_output(self, text: str) -> Tuple[str, bool]:
        """截断过长的输出"""
        if len(text) <= self.max_output_bytes:
            return text, False
        
        truncated = text[:self.max_output_bytes]
        truncated += f"\n... [truncated, total {len(text)} bytes]"
        return truncated, True
    
    # ==================== 命令执行 ====================
    
    def execute_command(
        self,
        command: str,
        timeout: Optional[float] = None,
        working_dir: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        capture_output: bool = True
    ) -> TerminalResult:
        """
        执行命令
        
        Args:
            command: 要执行的命令
            timeout: 超时时间 (秒)
            working_dir: 工作目录
            env: 环境变量
            capture_output: 是否捕获输出
        
        Returns:
            TerminalResult
        """
        start_time = time.time()
        
        # 1. 安全检查
        if self.safe_mode:
            is_safe, reason = self._is_safe_command(command)
            if not is_safe:
                logger.warning(f"Blocked unsafe command: {command} - {reason}")
                return TerminalResult(
                    success=False,
                    stdout="",
                    stderr=f"Command blocked: {reason}",
                    exit_code=-1,
                    duration=0,
                    command=command
                )
        
        # 2. 检查命令是否在允许列表中
        cmd_parts = shlex.split(command)
        if not cmd_parts:
            return TerminalResult(
                success=False,
                stdout="",
                stderr="Empty command",
                exit_code=-1,
                duration=0,
                command=command
            )
        
        main_cmd = cmd_parts[0]
        if main_cmd not in self.allowed_commands:
            # 检查是否有别名或完整路径
            cmd_name = Path(main_cmd).name
            if cmd_name not in self.allowed_commands:
                logger.warning(f"Command not allowed: {main_cmd}")
                return TerminalResult(
                    success=False,
                    stdout="",
                    stderr=f"Command '{main_cmd}' not allowed. Allowed: {sorted(self.allowed_commands)}",
                    exit_code=-1,
                    duration=0,
                    command=command
                )
        
        # 3. 确定工作目录
        if working_dir:
            wd = Path(working_dir)
        else:
            wd = self._cwd
        
        # 验证工作目录
        if not self._is_path_allowed(wd):
            return TerminalResult(
                success=False,
                stdout="",
                stderr=f"Working directory not allowed: {wd}",
                exit_code=-1,
                duration=0,
                command=command
            )
        
        # 4. 准备环境变量
        env_vars = os.environ.copy()
        if env:
            env_vars.update(env)
        
        # 5. 执行命令
        try:
            timeout_sec = timeout or self.timeout_seconds
            
            logger.debug(f"Executing: {command} in {wd}")
            
            result = subprocess.run(
                command,
                shell=True,
                cwd=str(wd),
                env=env_vars,
                capture_output=capture_output,
                timeout=timeout_sec,
                text=True
            )
            
            stdout, truncated_out = self._truncate_output(result.stdout or "")
            stderr, truncated_err = self._truncate_output(result.stderr or "")
            
            duration = time.time() - start_time
            
            # 记录历史
            self._add_history({
                'command': command,
                'working_dir': str(wd),
                'exit_code': result.returncode,
                'duration': duration,
                'success': result.returncode == 0,
                'timestamp': datetime.now().isoformat()
            })
            
            return TerminalResult(
                success=result.returncode == 0,
                stdout=stdout,
                stderr=stderr,
                exit_code=result.returncode,
                duration=duration,
                command=command,
                truncated=truncated_out or truncated_err
            )
            
        except subprocess.TimeoutExpired as e:
            logger.warning(f"Command timeout: {command}")
            return TerminalResult(
                success=False,
                stdout="",
                stderr=f"Command timed out after {timeout_sec}s",
                exit_code=-1,
                duration=timeout_sec,
                command=command
            )
        except Exception as e:
            logger.error(f"Command execution failed: {e}")
            return TerminalResult(
                success=False,
                stdout="",
                stderr=f"Execution error: {str(e)}",
                exit_code=-1,
                duration=time.time() - start_time,
                command=command
            )
    
    # ==================== Python 沙箱执行 ====================

def execute_python(
    self,
    code: str,
    timeout: Optional[float] = None,
    working_dir: Optional[str] = None,
    memory_limit_mb: int = 256,
    cpu_limit: float = 1.0
) -> TerminalResult:
    """
    在沙箱中执行 Python 代码 (真正的隔离)

    安全措施:
    1. 文件系统隔离: 只能访问沙箱目录
    2. 模块限制: 禁止 os, subprocess, socket 等危险模块
    3. builtins限制: 删除 eval, exec, globals 等
    4. 内存限制: ulimit -v
    5. CPU限制: ulimit -t
    6. 超时控制: timeout 命令
    7. 输出捕获: 重定向 stdout/stderr

    Args:
        code: Python 代码
        timeout: 超时时间 (秒)
        working_dir: 工作目录 (可选)
        memory_limit_mb: 内存限制 (MB)
        cpu_limit: CPU时间限制 (秒)

    Returns:
        TerminalResult
    """
    timeout_sec = timeout or self.timeout_seconds
    
    # 1. 创建沙箱目录
    import tempfile
    sandbox_dir = tempfile.mkdtemp(dir=str(self.base_path))
    sandbox_path = Path(sandbox_dir)
    
    try:
        # 2. 注入沙箱安全代码
        safe_code = self._inject_sandbox_code(code, sandbox_path)
        
        # 3. 写入脚本
        script_path = sandbox_path / "script.py"
        script_path.write_text(safe_code, encoding='utf-8')
        
        # 4. 准备环境 (清空 Python 路径防止导入外部模块)
        env = os.environ.copy()
        env['PYTHONPATH'] = ''
        env['PYTHONHOME'] = ''
        env['TMPDIR'] = sandbox_dir
        env['TEMP'] = sandbox_dir
        env['TMP'] = sandbox_dir
        
        # 5. 确定工作目录
        wd = Path(working_dir) if working_dir else sandbox_path
        if not self._is_path_allowed(wd):
            return TerminalResult(
                success=False,
                stdout="",
                stderr=f"Working directory not allowed: {wd}",
                exit_code=-1,
                duration=0,
                command="python (sandboxed)"
            )
        
        # 6. 构建沙箱命令
        cmd = self._build_sandbox_command(
            script_path,
            timeout=timeout_sec,
            memory_limit_mb=memory_limit_mb,
            cpu_limit=cpu_limit
        )
        
        logger.debug(f"Executing sandboxed Python: {script_path}")
        
        # 7. 执行
        start_time = time.time()
        
        result = subprocess.run(
            cmd,
            shell=True,
            cwd=str(wd),
            env=env,
            capture_output=True,
            timeout=timeout_sec + 2,  # 给清理留点余量
            text=True
        )
        
        duration = time.time() - start_time
        
        # 8. 记录历史
        self._add_history({
            'command': 'python (sandboxed)',
            'code_length': len(code),
            'working_dir': str(wd),
            'exit_code': result.returncode,
            'duration': duration,
            'success': result.returncode == 0,
            'timestamp': datetime.now().isoformat(),
            'sandbox_dir': sandbox_dir
        })
        
        return TerminalResult(
            success=result.returncode == 0,
            stdout=result.stdout or "",
            stderr=result.stderr or "",
            exit_code=result.returncode,
            duration=duration,
            command="python (sandboxed)",
            truncated=False
        )
        
    except subprocess.TimeoutExpired:
        logger.warning(f"Sandboxed Python timed out after {timeout_sec}s")
        return TerminalResult(
            success=False,
            stdout="",
            stderr=f"Sandboxed execution timed out after {timeout_sec}s",
            exit_code=-1,
            duration=timeout_sec,
            command="python (sandboxed)"
        )
    except Exception as e:
        logger.error(f"Sandboxed Python execution failed: {e}")
        return TerminalResult(
            success=False,
            stdout="",
            stderr=f"Sandbox error: {str(e)}",
            exit_code=-1,
            duration=time.time() - start_time if 'start_time' in locals() else 0,
            command="python (sandboxed)"
        )
    finally:
        # 9. 清理沙箱
        self._cleanup_sandbox(sandbox_path)

def _inject_sandbox_code(self, code: str, sandbox_dir: Path) -> str:
    """
    注入沙箱安全代码
    
    在用户代码周围添加安全层:
    1. 文件系统限制 (只能访问 sandbox_dir)
    2. 模块导入限制 (禁止 os, subprocess, socket 等)
    3. builtins限制 (删除 eval, exec 等)
    4. 输出捕获
    5. 异常处理
    """
    
    # 禁止的模块
    FORBIDDEN_MODULES = [
        'os', 'sys', 'subprocess', 'shutil', 'socket',
        'requests', 'urllib', 'ftplib', 'smtplib',
        'ctypes', 'multiprocessing', 'threading',
        'pickle', 'marshal', 'importlib'
    ]
    
    # 危险的 builtins
    DANGEROUS_BUILTINS = [
        'eval', 'exec', 'compile', 'globals', 'locals', 
        'vars', 'dir', 'open', '__import__', 'breakpoint'
    ]
    
    # 安全的模块 (允许导入)
    SAFE_MODULES = [
        'math', 'random', 'datetime', 'collections',
        'itertools', 'functools', 'typing', 'json',
        're', 'string', 'time', 'statistics', 'decimal',
        'fractions', 'heapq', 'bisect'
    ]
    
    forbidden_str = ', '.join(f"'{m}'" for m in FORBIDDEN_MODULES)
    dangerous_str = ', '.join(f"'{m}'" for m in DANGEROUS_BUILTINS)
    safe_str = ', '.join(f"'{m}'" for m in SAFE_MODULES)
    
    sandbox_code = f'''
# ========== 沙箱安全层 v2 ==========
# 沙箱目录: {sandbox_dir}

import builtins
import sys
import io
from pathlib import Path

# ---------- 1. 文件系统重定向 ----------
_original_open = builtins.open

def _sandbox_open(file, mode='r', *args, **kwargs):
    """文件操作重定向 - 只允许在沙箱目录内操作"""
    try:
        file_path = Path(file).resolve()
    except Exception:
        raise PermissionError(f"Invalid file path: {{file}}")
    
    sandbox_path = Path("{sandbox_dir}").resolve()
    
    # 只允许在沙箱目录内操作
    if not str(file_path).startswith(str(sandbox_path)):
        raise PermissionError(f"Access denied: {{file}} (must be in sandbox)")
    
    return _original_open(file, mode, *args, **kwargs)

# 替换 open
builtins.open = _sandbox_open

# ---------- 2. 禁用危险 builtins ----------
dangerous = [{dangerous_str}]
for _name in dangerous:
    if hasattr(builtins, _name):
        delattr(builtins, _name)

# ---------- 3. 模块导入限制 ----------
_original_import = builtins.__import__

def _sandbox_import(name, *args, **kwargs):
    """模块导入限制 - 禁止危险模块"""
    forbidden = [{forbidden_str}]
    
    # 检查导入的模块
    module_name = name.split('.')[0]
    if module_name in forbidden:
        raise ImportError(f"Module '{{module_name}}' is not allowed in sandbox")
    
    return _original_import(name, *args, **kwargs)

builtins.__import__ = _sandbox_import

# ---------- 4. 捕获输出 ----------
_stdout = sys.stdout
_stderr = sys.stderr
sys.stdout = io.StringIO()
sys.stderr = io.StringIO()

# ---------- 5. 用户代码 ----------
try:
    # 允许导入安全模块
    import math, random, datetime, collections
    import itertools, functools, typing, json, re, string
    
    # ===== 用户代码开始 =====
{self._indent_code(code, 1)}
    # ===== 用户代码结束 =====
    
except Exception as _e:
    import traceback
    _trace = traceback.format_exc()
    print(f"Runtime Error: {{_e}}", file=sys.stderr)
    print(_trace, file=sys.stderr)
    sys.exit(1)

# ---------- 6. 输出还原 ----------
_stdout.write(sys.stdout.getvalue())
_stderr.write(sys.stderr.getvalue())
_stdout.flush()
_stderr.flush()
sys.stdout = _stdout
sys.stderr = _stderr
'''
    
    return sandbox_code

def _indent_code(self, code: str, indent_level: int) -> str:
    """缩进代码"""
    indent = "    " * indent_level
    lines = code.split('\n')
    result = []
    for line in lines:
        if line.strip() or result:  # 保留空行结构
            result.append(f"{indent}{line}" if line.strip() else "")
        else:
            result.append("")
    return '\n'.join(result)

def _build_sandbox_command(
    self,
    script_path: Path,
    timeout: float,
    memory_limit_mb: int,
    cpu_limit: float
) -> str:
    """
    构建带资源限制的命令
    
    使用:
    - ulimit: 限制内存和CPU
    - timeout: 超时控制
    - nice: 降低优先级
    """
    
    # 内存限制 (ulimit -v)
    memory_bytes = memory_limit_mb * 1024 * 1024
    
    # CPU时间限制 (ulimit -t)
    cpu_seconds = int(timeout)
    
    if sys.platform == 'linux' or sys.platform == 'darwin':
        # Linux / macOS: 使用 ulimit
        cmd_parts = [
            f"ulimit -v {memory_bytes}",      # 虚拟内存限制
            f"ulimit -t {cpu_seconds}",        # CPU时间限制
            f"ulimit -f 10240",                # 文件大小限制 (10MB)
            f"nice -n 19",                     # 降低优先级
            f"timeout {timeout}",              # 超时控制
            f"python3 {script_path}"
        ]
        return " && ".join(cmd_parts)
    else:
        # Windows: 简化版本
        return f"python3 {script_path}"

def _cleanup_sandbox(self, sandbox_dir: Path) -> None:
    """清理沙箱目录"""
    try:
        import shutil
        if sandbox_dir.exists():
            shutil.rmtree(sandbox_dir)
            logger.debug(f"Cleaned up sandbox: {sandbox_dir}")
    except Exception as e:
        logger.warning(f"Failed to cleanup sandbox {sandbox_dir}: {e}")

    # ==================== 文件操作 ====================
    
    def read_file(self, path: str) -> Dict[str, Any]:
        """
        读取文件内容
        
        Args:
            path: 文件路径
        
        Returns:
            {
                'success': bool,
                'content': str,
                'error': str,
                'size': int
            }
        """
        file_path = self._sanitize_path(path)
        if not file_path:
            return {
                'success': False,
                'content': '',
                'error': f'Invalid path: {path}',
                'size': 0
            }
        
        if not file_path.exists():
            return {
                'success': False,
                'content': '',
                'error': f'File not found: {path}',
                'size': 0
            }
        
        if not file_path.is_file():
            return {
                'success': False,
                'content': '',
                'error': f'Not a file: {path}',
                'size': 0
            }
        
        try:
            content = file_path.read_text()
            truncated, is_truncated = self._truncate_output(content)
            
            return {
                'success': True,
                'content': truncated,
                'error': None,
                'size': len(content),
                'truncated': is_truncated
            }
        except Exception as e:
            return {
                'success': False,
                'content': '',
                'error': str(e),
                'size': 0
            }
    
    def write_file(self, path: str, content: str, mode: str = 'w') -> Dict[str, Any]:
        """
        写入文件
        
        Args:
            path: 文件路径
            content: 内容
            mode: 写入模式 ('w' 覆盖, 'a' 追加)
        
        Returns:
            {'success': bool, 'error': str, 'size': int}
        """
        file_path = self._sanitize_path(path)
        if not file_path:
            return {
                'success': False,
                'error': f'Invalid path: {path}',
                'size': 0
            }
        
        try:
            # 确保目录存在
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            if mode == 'a':
                with open(file_path, 'a') as f:
                    f.write(content)
            else:
                with open(file_path, 'w') as f:
                    f.write(content)
            
            return {
                'success': True,
                'error': None,
                'size': len(content),
                'path': str(file_path)
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'size': 0
            }
    
    def list_directory(self, path: str = ".", show_hidden: bool = False) -> Dict[str, Any]:
        """
        列出目录内容
        
        Args:
            path: 目录路径
            show_hidden: 是否显示隐藏文件
        
        Returns:
            {
                'success': bool,
                'files': List[Dict],
                'error': str
            }
        """
        dir_path = self._sanitize_path(path)
        if not dir_path:
            return {
                'success': False,
                'files': [],
                'error': f'Invalid path: {path}'
            }
        
        if not dir_path.exists():
            return {
                'success': False,
                'files': [],
                'error': f'Directory not found: {path}'
            }
        
        if not dir_path.is_dir():
            return {
                'success': False,
                'files': [],
                'error': f'Not a directory: {path}'
            }
        
        try:
            files = []
            for item in dir_path.iterdir():
                if not show_hidden and item.name.startswith('.'):
                    continue
                
                stat = item.stat()
                files.append({
                    'name': item.name,
                    'path': str(item),
                    'is_dir': item.is_dir(),
                    'size': stat.st_size if item.is_file() else 0,
                    'modified': datetime.fromtimestamp(stat.st_mtime).isoformat()
                })
            
            # 排序：目录在前，然后按名称
            files.sort(key=lambda x: (not x['is_dir'], x['name']))
            
            return {
                'success': True,
                'files': files,
                'count': len(files),
                'error': None
            }
        except Exception as e:
            return {
                'success': False,
                'files': [],
                'error': str(e)
            }
    
    def delete_file(self, path: str) -> Dict[str, Any]:
        """
        删除文件
        
        Args:
            path: 文件路径
        
        Returns:
            {'success': bool, 'error': str}
        """
        file_path = self._sanitize_path(path)
        if not file_path:
            return {
                'success': False,
                'error': f'Invalid path: {path}'
            }
        
        if not file_path.exists():
            return {
                'success': False,
                'error': f'File not found: {path}'
            }
        
        try:
            if file_path.is_file():
                file_path.unlink()
            else:
                file_path.rmdir()
            
            return {
                'success': True,
                'error': None
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def file_info(self, path: str) -> Dict[str, Any]:
        """
        获取文件信息
        
        Args:
            path: 文件路径
        
        Returns:
            {
                'success': bool,
                'info': Dict,
                'error': str
            }
        """
        file_path = self._sanitize_path(path)
        if not file_path:
            return {
                'success': False,
                'info': {},
                'error': f'Invalid path: {path}'
            }
        
        if not file_path.exists():
            return {
                'success': False,
                'info': {},
                'error': f'File not found: {path}'
            }
        
        try:
            stat = file_path.stat()
            return {
                'success': True,
                'info': {
                    'name': file_path.name,
                    'path': str(file_path),
                    'is_dir': file_path.is_dir(),
                    'is_file': file_path.is_file(),
                    'size': stat.st_size,
                    'created': datetime.fromtimestamp(stat.st_ctime).isoformat(),
                    'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    'accessed': datetime.fromtimestamp(stat.st_atime).isoformat(),
                    'permissions': oct(stat.st_mode)[-3:]
                },
                'error': None
            }
        except Exception as e:
            return {
                'success': False,
                'info': {},
                'error': str(e)
            }
    
    # ==================== 系统信息 ====================
    
    def get_system_info(self) -> Dict[str, Any]:
        """获取系统信息"""
        return {
            'hostname': os.uname().nodename if hasattr(os, 'uname') else 'unknown',
            'platform': sys.platform,
            'python_version': sys.version,
            'current_directory': str(self._cwd),
            'base_path': str(self.base_path),
            'allowed_commands': sorted(self.allowed_commands)
        }
    
    def set_working_directory(self, path: str) -> Dict[str, Any]:
        """设置工作目录"""
        new_path = self._sanitize_path(path)
        if not new_path:
            return {
                'success': False,
                'error': f'Invalid path: {path}'
            }
        
        if not new_path.exists():
            return {
                'success': False,
                'error': f'Directory not found: {path}'
            }
        
        if not new_path.is_dir():
            return {
                'success': False,
                'error': f'Not a directory: {path}'
            }
        
        self._cwd = new_path
        return {
            'success': True,
            'cwd': str(self._cwd),
            'error': None
        }
    
    # ==================== 工具构建 ====================
    
    def _build_tools(self):
        """构建 Agno 工具函数"""
        try:
            from agno.tools import Function
        except ImportError:
            Function = None
            logger.warning("Agno not available, tools will be raw callables")
        
        self._tools = []
        self._functions = []
        
        # 定义工具
        tool_defs = [
            {
                'name': 'execute_command',
                'description': '执行终端命令。参数: command (str) - 要执行的命令',
                'handler': self._execute_command_tool
            },
            {
                'name': 'read_file',
                'description': '读取文件内容。参数: path (str) - 文件路径',
                'handler': self._read_file_tool
            },
            {
                'name': 'write_file',
                'description': '写入文件。参数: path (str), content (str), mode (str, 可选)',
                'handler': self._write_file_tool
            },
            {
                'name': 'list_directory',
                'description': '列出目录内容。参数: path (str, 可选), show_hidden (bool, 可选)',
                'handler': self._list_directory_tool
            },
            {
                'name': 'delete_file',
                'description': '删除文件或目录。参数: path (str)',
                'handler': self._delete_file_tool
            },
            {
                'name': 'file_info',
                'description': '获取文件信息。参数: path (str)',
                'handler': self._file_info_tool
            },
            {
                'name': 'execute_python',
                'description': '执行Python代码。参数: code (str)',
                'handler': self._execute_python_tool
            },
            {
                'name': 'get_system_info',
                'description': '获取系统信息。无参数',
                'handler': self._get_system_info_tool
            },
            {
                'name': 'set_working_directory',
                'description': '设置工作目录。参数: path (str)',
                'handler': self._set_working_directory_tool
            },
        ]
        
        for tool_def in tool_defs:
            if Function is not None:
                try:
                    fn = Function(
                        name=tool_def['name'],
                        description=tool_def['description'],
                        entrypoint=tool_def['handler'],
                    )
                    self._tools.append(fn)
                except Exception as e:
                    logger.debug(f"Failed to create Function for {tool_def['name']}: {e}")
                    self._tools.append(tool_def['handler'])
            else:
                self._tools.append(tool_def['handler'])
        
        logger.info(f"Built {len(self._tools)} tools for TerminalToolkit")
    
    # ==================== 工具处理器 ====================
    
    def _execute_command_tool(self, command: str, timeout: float = None) -> str:
        """工具: 执行命令"""
        result = self.execute_command(command, timeout)
        if result.success:
            return result.stdout or "Command executed successfully"
        return f"Error: {result.stderr}"
    
    def _read_file_tool(self, path: str) -> str:
        #把底层方法包装成Agent可调用的工具 
        #返回的是纯字符串, agent直接使用
        #底层方法返回的是复杂字典结构

        """工具: 读取文件"""
        result = self.read_file(path)
        if result['success']:
            return result['content']
        return f"Error: {result['error']}"
    
    def _write_file_tool(self, path: str, content: str, mode: str = 'w') -> str:
        """工具: 写入文件"""
        result = self.write_file(path, content, mode)
        if result['success']:
            return f"Successfully wrote {result['size']} bytes to {path}"
        return f"Error: {result['error']}"
    
    def _list_directory_tool(self, path: str = ".", show_hidden: bool = False) -> str:
        """工具: 列出目录"""
        result = self.list_directory(path, show_hidden)
        if result['success']:
            if not result['files']:
                return "Directory is empty"
            
            lines = [f"Directory: {path} ({result['count']} items)"]
            for f in result['files']:
                prefix = "📁" if f['is_dir'] else "📄"
                size = f" ({f['size']} bytes)" if not f['is_dir'] else ""
                lines.append(f"  {prefix} {f['name']}{size}")
            return "\n".join(lines)
        return f"Error: {result['error']}"
    
    def _delete_file_tool(self, path: str) -> str:
        """工具: 删除文件"""
        result = self.delete_file(path)
        if result['success']:
            return f"Successfully deleted: {path}"
        return f"Error: {result['error']}"
    
    def _file_info_tool(self, path: str) -> str:
        """工具: 文件信息"""
        result = self.file_info(path)
        if result['success']:
            info = result['info']
            return (
                f"File: {info['name']}\n"
                f"  Path: {info['path']}\n"
                f"  Type: {'Directory' if info['is_dir'] else 'File'}\n"
                f"  Size: {info['size']} bytes\n"
                f"  Modified: {info['modified']}"
            )
        return f"Error: {result['error']}"
    
    def _execute_python_tool(self, code: str, timeout: float = None) -> str:
        """工具: 执行Python代码"""
        result = self.execute_python(code, timeout)
        if result.success:
            return result.stdout or "Code executed successfully"
        return f"Error: {result.stderr}"
    
    def _get_system_info_tool(self) -> str:
        """工具: 系统信息"""
        info = self.get_system_info()
        return (
            f"Hostname: {info['hostname']}\n"
            f"Platform: {info['platform']}\n"
            f"Python: {info['python_version'].split()[0]}\n"
            f"Current Directory: {info['current_directory']}\n"
            f"Base Path: {info['base_path']}\n"
            f"Allowed Commands: {', '.join(info['allowed_commands'][:10])}..."
        )
    
    def _set_working_directory_tool(self, path: str) -> str:
        """工具: 设置工作目录"""
        result = self.set_working_directory(path)
        if result['success']:
            return f"Working directory set to: {result['cwd']}"
        return f"Error: {result['error']}"
    
    # ==================== 历史和管理 ====================
    
    def _add_history(self, entry: Dict[str, Any]) -> None:
        """添加历史记录"""
        self._history.append(entry)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]
    
    def get_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """获取命令历史"""
        return self._history[-limit:]
    
    def clear_history(self) -> None:
        """清空历史"""
        self._history = []
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        total = len(self._history)
        successful = sum(1 for h in self._history if h.get('success', False))
        
        return {
            'total_commands': total,
            'successful': successful,
            'success_rate': successful / total if total > 0 else 0,
            'allowed_commands': len(self.allowed_commands),
            'current_directory': str(self._cwd),
            'base_path': str(self.base_path),
            'safe_mode': self.safe_mode
        }
    
    # ==================== Agno 接口 ====================
    
    def get_tools(self) -> List[Any]:
        """获取工具列表 (给 Agent)"""
        return self._tools
    
    def get_tool_names(self) -> List[str]:
        """获取工具名称列表"""
        return [
            'execute_command', 'read_file', 'write_file', 
            'list_directory', 'delete_file', 'file_info',
            'execute_python', 'get_system_info', 'set_working_directory'
        ]
    
    def add_allowed_command(self, command: str) -> None:
        """添加允许的命令"""
        self.allowed_commands.add(command)
        logger.info(f"Added allowed command: {command}")
    
    def remove_allowed_command(self, command: str) -> None:
        """移除允许的命令"""
        if command in self.allowed_commands:
            self.allowed_commands.remove(command)
            logger.info(f"Removed allowed command: {command}")


# ==================== 便捷函数 ====================

def create_terminal_toolkit(
    base_path: Optional[str] = None,
    allowed_commands: Optional[List[str]] = None,
    allow_network: bool = False,
    **kwargs
) -> TerminalToolkit:
    """
    创建终端工具包
    
    Args:
        base_path: 基础路径
        allowed_commands: 允许的命令列表
        allow_network: 是否允许网络命令
        **kwargs: 其他参数
    
    Returns:
        TerminalToolkit
    """
    return TerminalToolkit(
        allowed_commands=allowed_commands,
        base_path=base_path,
        allow_network=allow_network,
        **kwargs
    )


def execute_terminal_command(
    command: str,
    base_path: Optional[str] = None,
    **kwargs
) -> TerminalResult:
    """
    执行终端命令 (快速接口)
    
    Args:
        command: 命令
        base_path: 基础路径
        **kwargs: 其他参数
    
    Returns:
        TerminalResult
    """
    toolkit = TerminalToolkit(base_path=base_path, **kwargs)
    return toolkit.execute_command(command)