"""
Runner - Agent执行器

功能:
1. 执行Agent任务
2. 支持暂停/恢复
3. 支持取消
4. 任务状态管理
5. 预算控制
6. 事件发射
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable, Union
import logging
from .trace import Tracer
from .core_agent import CoreAgent
from .retry import RetryPolicy
from ..tools.tools_registry import get_tools_registry, ToolCategory


logger = logging.getLogger(__name__)

import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..mcp_protocol.mcp_lifecycle import MCPClientPool

# ── MCP client pool（全局单例，跨任务复用）──
_MCP_POOL: Optional["MCPClientPool"] = None
_MCP_LOCK = threading.Lock()


def _get_mcp_pool() -> "MCPClientPool":
    """全局单例 MCP client pool。避免每次 start_task 都重连。"""
    global _MCP_POOL
    if _MCP_POOL is None:
        with _MCP_LOCK:
            if _MCP_POOL is None:
                from ..mcp_protocol.mcp_lifecycle import MCPClientPool
                _MCP_POOL = MCPClientPool(auto_atexit=True)
    return _MCP_POOL

_DEFAULT_DB = str(Path(__file__).parent.parent.parent.parent / "agent_tasks.db")

# 事件发射器类型
EventEmitter = Callable[[str, Dict[str, Any]], None]


@dataclass
class Task:
    """任务定义"""
    id: str
    query: str
    config: Dict[str, Any]
    status: str  # pending, running, paused, completed, failed, cancelled
    steps: List[Dict[str, Any]]
    result: Optional[Any] = None
    error: Optional[str] = None
    plan: Optional[Dict[str, Any]] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())


class TaskStateManager:
    """任务状态管理器 (SQLite)"""

    def __init__(self, db_path: str = _DEFAULT_DB):
        self.db_path = db_path
        self._ensure_schema()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self) -> None:
        """确保数据库架构存在"""
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS agent_tasks (
                    id TEXT PRIMARY KEY,
                    query TEXT NOT NULL,
                    config_json TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    steps_json TEXT DEFAULT '[]',
                    result TEXT,
                    error TEXT,
                    plan_json TEXT,
                    checkpoint_json TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_tasks_status 
                ON agent_tasks(status)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_tasks_created 
                ON agent_tasks(created_at DESC)
            """)
            # 兼容旧库
            self._migrate(conn)

    def _migrate(self, conn: sqlite3.Connection) -> None:
        cols = {r["name"] for r in conn.execute("PRAGMA table_info(agent_tasks)")}
        if "checkpoint_json" not in cols:
            conn.execute("ALTER TABLE agent_tasks ADD COLUMN checkpoint_json TEXT")


    def create_task(self, query: str, config: Dict[str, Any]) -> str:
        """创建任务"""
        task_id = str(uuid.uuid4())
        now = datetime.now().isoformat()

        with self._conn() as conn:
            conn.execute(
                """INSERT INTO agent_tasks 
                   (id, query, config_json, status, steps_json, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (task_id, query, json.dumps(config), 'pending', '[]', now, now)
            )

        return task_id

    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """获取任务"""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM agent_tasks WHERE id = ?", (task_id,)
            ).fetchone()

        if not row:
            return None

        return {
            'id': row['id'],
            'query': row['query'],
            'config': json.loads(row['config_json']),
            'status': row['status'],
            'steps': json.loads(row['steps_json']),
            'result': row['result'],
            'error': row['error'],
            'plan': json.loads(row['plan_json']) if row['plan_json'] else None,
            'created_at': row['created_at'],
            'updated_at': row['updated_at']
        }

    def update_status(self, task_id: str, status: str) -> None:
        """更新任务状态"""
        now = datetime.now().isoformat()
        with self._conn() as conn:
            conn.execute(
                "UPDATE agent_tasks SET status = ?, updated_at = ? WHERE id = ?",
                (status, now, task_id)
            )

    def save_checkpoint(
        self,
        task_id: str,
        checkpoint: Dict[str, Any],
    ) -> None:
         """
        保存完整断点。checkpoint 建议结构：
        {
            "step_index": int,
            "messages": [...],        # 对话历史
            "scratchpad": {...},      # 中间结果
            "plan": {...},            # 计划
            "last_tool_results": [...],
        }
        """
       
         now = datetime.now().isoformat()
         with self._conn() as conn:
             conn.execute(
                 "UPDATE agent_tasks SET checkpoint_json = ?, updated_at = ? WHERE id = ?",
                 (json.dumps(checkpoint), now, task_id)
             )

    
    def load_checkpoint(self, task_id: str) -> Optional[Dict[str, Any]]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT checkpoint_json FROM agent_tasks WHERE id = ?", (task_id,)
            ).fetchone()
        if not row or not row["checkpoint_json"]:
            return None
        return json.loads(row["checkpoint_json"])



    def complete_task(self, task_id: str, result: Any) -> None:
        """完成任务"""
        now = datetime.now().isoformat()
        with self._conn() as conn:
            conn.execute(
                "UPDATE agent_tasks SET status = 'completed', result = ?, updated_at = ? WHERE id = ?",
                (str(result), now, task_id)
            )

    def fail_task(self, task_id: str, error: str) -> None:
        """任务失败"""
        now = datetime.now().isoformat()
        with self._conn() as conn:
            conn.execute(
                "UPDATE agent_tasks SET status = 'failed', error = ?, updated_at = ? WHERE id = ?",
                (error, now, task_id)
            )

    def save_plan(self, task_id: str, plan: Dict[str, Any]) -> None:
        """保存计划"""
        now = datetime.now().isoformat()
        with self._conn() as conn:
            conn.execute(
                "UPDATE agent_tasks SET plan_json = ?, updated_at = ? WHERE id = ?",
                (json.dumps(plan), now, task_id)
            )

    def list_tasks(
        self,
        status: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """列出任务"""
        sql = "SELECT * FROM agent_tasks"
        params = []

        if status:
            sql += " WHERE status = ?"
            params.append(status)

        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        with self._conn() as conn:
            rows = conn.execute(sql, params).fetchall()

        tasks = []
        for row in rows:
            tasks.append({
                'id': row['id'],
                'query': row['query'],
                'config': json.loads(row['config_json']),
                'status': row['status'],
                'steps': json.loads(row['steps_json']),
                'result': row['result'],
                'error': row['error'],
                'plan': json.loads(row['plan_json']) if row['plan_json'] else None,
                'created_at': row['created_at'],
                'updated_at': row['updated_at']
            })

        return tasks


    def _execute_with_checkpoints(
       self,
       task_id: str,
       query: str,
       config: Dict[str, Any],
       checkpoint: Optional[Dict[str, Any]] = None,
     ):
       agent = self._create_agent(config)
       self._agent = agent

       def on_step(step_index: int, state: Dict[str, Any]) -> None:
        # state 就是 agent.export_state(...) 的返回值
           self.state_mgr.save_checkpoint(task_id, {
               "step_index": step_index,
               "state": state,          # ← 整个快照
          })
           self._emit('task_checkpoint', {
              'task_id': task_id,
              'step_index': step_index,
          })

    # 统一入口：checkpoint=None 就是从零开始
       result = agent.run(
          query,
          config,
          checkpoint=checkpoint["state"] if checkpoint else None,  # ★
          on_step=on_step,                                          # ★
      )
       return result

class AgenticRunner:
    """
    Agent执行器

    功能:
    1. 执行Agent任务
    2. 支持暂停/恢复
    3. 支持取消
    4. 事件发射
    5. 检查点保存
    """

    def __init__(
        self,
        api_keys: Dict[str, str],
        db_path: str = _DEFAULT_DB,
        emit: Optional[EventEmitter] = None
    ):
        self.api_keys = api_keys
        self.db_path = db_path
        self.state_mgr = TaskStateManager(db_path)   #state_mgr 是 TaskStateManager 的实例，负责把任务的状态持久化到 SQLite 数据库里。
        self._emit = emit or (lambda kind, payload: None)
        self._agent = None

    def _ensure_mcp_clients(self, mcp_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        根据 config["mcp"] 确保 client 已连上。返回 {namespace: client}。

        幂等：同名 server 只连一次。pool 是全局单例，所以跨任务复用。
        """
        from ..mcp_protocol.config import load_mcp_configs

        cfgs = load_mcp_configs(mcp_config)
        if not cfgs:
            return {}

        pool = _get_mcp_pool()

        # 只 add 没加过的
        existing = set(pool.names())
        new_cfgs = [c for c in cfgs if c.name not in existing]
        if new_cfgs:
            pool.add_many(new_cfgs)
            results = pool.connect_all()
            for name, ok in results.items():
                if not ok:
                    logger.warning("MCP server '%s' failed to connect", name)

        # 返回当前 pool 里所有 client
        return {name: pool.get(name) for name in pool.names()}

    
    def _create_agent(self, config: Dict[str, Any],tracer: Optional[Tracer] = None):
        """创建Agent"""
        from .core_agent import CoreAgent
        from .retry import RetryPolicy
        from ..tools.tools_registry import ToolsRegistry
        registry = ToolsRegistry(...)

        mcp_clients = self._ensure_mcp_clients(config.get("mcp", {}))
        for ns, client in mcp_clients.items():
            if client is None:
                continue
            try:
                mapping = registry.load_from_mcp(
                    client=client,
                    namespace=ns,
                    category=ToolCategory.CUSTOM,
                    overwrite=False,
                    prefix_sep="_",          # ★ LLM 函数名不能带点
                )
                logger.info("MCP[%s] loaded %d tools: %s",
                            ns, len(mapping), list(mapping))
            except Exception:
                logger.exception("MCP[%s] load failed", ns)



        # 获取工具
        tool_names = config.get('tools', [])
       
        tools = registry.get_tools(tool_names, self.api_keys)
        
        trace_cfg = config.get('trace', {})
        tracer = Tracer(
          service_name=trace_cfg.get('service_name', 'agent-harness'),
          exporter=trace_cfg.get('exporter', 'console'),
          otlp_endpoint=trace_cfg.get('otlp_endpoint'),
          verbose=trace_cfg.get('verbose', False),
      )

        memory_config = config.get('memory', {})
        memory_store = None
        if memory_config.get('enabled', False):
            from .archival_memory import ArchivalMemoryStore
            memory_store = ArchivalMemoryStore(self.db_path)

         # ★ 重试策略
        retry_cfg = config.get('retry', {})
        retry_policy = RetryPolicy(
             max_attempts=retry_cfg.get('max_attempts', 3),
             base_delay=retry_cfg.get('base_delay', 1.0),
             max_delay=retry_cfg.get('max_delay', 10.0),
         )

        # 创建Agent
        return CoreAgent(
            name=config.get('name', 'Agent'),
            llm=self._create_llm(config.get('model', {})),
            tools=tools,
            memory_store=memory_store,
            max_iterations=config.get('max_iterations', 10),
            llm_retry_policy=llm_retry_policy,          # ← 按 core-agent 实际名字
            circuit_threshold=5,
            enable_reflection=config.get('enable_reflection', True),
            verbose=config.get('verbose', True),
            tracer=tracer or Tracer(),
         )

    def _create_llm(self, model_config: Dict[str, Any]):
       from .llm_adapter import LLMAdapter

       provider = model_config.get('provider', 'openai')
       model = model_config.get('model', 'gpt-4o')
       api_key = (
           self.api_keys.get(provider)
           or self.api_keys.get(f'{provider.upper()}_API_KEY')
      )
       if not api_key:
           logger.warning(f"No API key for {provider}, agent will run without LLM")
           return None

       return LLMAdapter(provider=provider, api_key=api_key, model=model)

    def start_task(self, query: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        启动任务

        Args:
            query: 任务描述
            config: 配置

        Returns:
            {'task_id': str, 'success': bool, ...}
        """
        task_id = self.state_mgr.create_task(query, config)
        tracer = Tracer(trace_id=task_id)
        self.state_mgr.update_status(task_id, 'running')

        self._emit('task_started', {'task_id': task_id, 'query': query})

        # 创建Agent
        self._agent = self._create_agent(config)

        # 执行
        try:
            result = self._agent.run(query, config)
            self.state_mgr.complete_task(task_id, result.answer)

            self._emit('task_completed', {
                'task_id': task_id,
                'result': result.answer,
                'steps': len(result.steps)
            })

            return {
                'success': True,
                'task_id': task_id,
                'result': result.answer,
                'steps': len(result.steps)
            }

        except Exception as e:
            self.state_mgr.fail_task(task_id, str(e))
            self._emit('task_error', {'task_id': task_id, 'error': str(e)})

            return {
                'success': False,
                'task_id': task_id,
                'error': str(e)
            }

    def resume_task(self, task_id: str) -> Dict[str, Any]:
        """恢复任务（从断点继续）"""
        task = self.state_mgr.get_task(task_id)
        if not task:
          return {'success': False, 'error': f'Task {task_id} not found'}

        if task['status'] == 'completed':
            return {
               'success': True,
               'task_id': task_id,
               'result': task['result'],
               'already_completed': True
           }

        if task['status'] == 'cancelled':
            return {'success': False, 'task_id': task_id, 'cancelled': True}

    # ★ 1. 读出断点
        checkpoint = task.get('checkpoint')          # 来自 checkpoint_json 字段
        from_step = checkpoint.get('step_index', 0) if checkpoint else 0

        logger.info(
            "Resume task %s from step %s (checkpoint=%s)",
            task_id, from_step, "yes" if checkpoint else "no",
        )

    # 2. 恢复执行
        self.state_mgr.update_status(task_id, 'running')
        self._emit('task_resumed', {'task_id': task_id, 'from_step': from_step})

    # 3. 创建 Agent
        self._agent = self._create_agent(task['config'])

    # ★ 4. 定义 on_step：每轮工具执行完就存快照
        def on_step(step_index: int, state: Dict[str, Any]) -> None:
            self.state_mgr.save_checkpoint(task_id, {
              "step_index": step_index,
              "state": state,
        })
            self._emit('task_checkpoint', {
              'task_id': task_id,
              'step_index': step_index,
        })

        try:
        # ★ 5. 把 checkpoint 传给 run()
            result = self._agent.run(
               task['query'],
               task['config'],
               checkpoint=checkpoint["state"] if checkpoint else None,   # ★
               on_step=on_step,                                           # ★
        )
            self.state_mgr.complete_task(task_id, result.answer)

            self._emit('task_completed', {
               'task_id': task_id,
               'result': result.answer
           })

            return {
              'success': True,
              'task_id': task_id,
              'result': result.answer,
              'resumed_from_step': from_step,
          }

        except Exception as e:
            self.state_mgr.fail_task(task_id, str(e))
            self._emit('task_error', {'task_id': task_id, 'error': str(e)})
            return {
                'success': False,
                'task_id': task_id,
                'error': str(e)
            }
    def cancel_task(self, task_id: str) -> Dict[str, Any]:
        """
        取消任务

        Args:
            task_id: 任务ID

        Returns:
            {'success': bool}
        """
        task = self.state_mgr.get_task(task_id)
        if not task:
            return {'success': False, 'error': f'Task {task_id} not found'}

        if task['status'] in ('completed', 'failed', 'cancelled'):
            return {'success': True, 'message': f'Task already {task["status"]}'}

        self.state_mgr.update_status(task_id, 'cancelled')
        self._emit('task_cancelled', {'task_id': task_id})

        return {'success': True}

    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """获取任务"""
        return self.state_mgr.get_task(task_id)

    def list_tasks(self, status: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        """列出任务"""
        return self.state_mgr.list_tasks(status, limit)


# ==================== 便捷函数 ====================

def create_runner(
    api_keys: Dict[str, str],
    db_path: str = _DEFAULT_DB,
    emit: Optional[EventEmitter] = None
) -> AgenticRunner:
    """创建Runner"""
    return AgenticRunner(api_keys=api_keys, db_path=db_path, emit=emit)


def run_agent(
    query: str,
    config: Dict[str, Any],
    api_keys: Dict[str, str]
) -> Dict[str, Any]:
    """
    快速运行Agent

    Args:
        query: 任务描述
        config: 配置
        api_keys: API密钥

    Returns:
        结果
    """
    runner = AgenticRunner(api_keys=api_keys)
    return runner.start_task(query, config)