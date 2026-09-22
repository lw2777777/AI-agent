# mcp_protocol/mcp_lifecycle.py
"""
MCP 连接生命周期管理

负责：
  1. 启动时按配置创建 client（stdio / sse）
  2. 保持连接（复用 client 实例）
  3. 关闭时统一清理

不负责工具注册——那是 ToolsRegistry 的事。
"""
from __future__ import annotations

import atexit
import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .client import MCPClient

logger = logging.getLogger(__name__)


@dataclass
class MCPServerConfig:
    """一个 MCP server 的连接配置"""
    name: str                          # 命名空间用
    transport: str = "stdio"           # "stdio" | "sse"

    # stdio 专用
    command: Optional[List[str]] = None
    env: Optional[Dict[str, str]] = None
    cwd: Optional[str] = None

    # sse 专用
    url: Optional[str] = None

    headers: Optional[Dict[str, str]] = None 
    # 通用
    timeout: float = 30.0
    enabled: bool = True


class MCPClientPool:
    """
    管理多个 MCP client 的连接池。

    用法：
        pool = MCPClientPool()
        pool.add(MCPServerConfig(name="fs", command=["python", "fs_server.py"]))
        pool.add(MCPServerConfig(name="calc", url="http://localhost:8080/sse",
                                 transport="sse"))
        pool.connect_all()

        # 拿一个 client
        client = pool.get("fs")

        # 关闭全部（也可注册 atexit 自动关）
        pool.close_all()
    """

    def __init__(self, auto_atexit: bool = True):
        self._clients: Dict[str, MCPClient] = {}
        self._configs: Dict[str, MCPServerConfig] = {}
        self._lock = threading.Lock()

        if auto_atexit:
            atexit.register(self.close_all)

    # ── 配置 ────────────────────────────────
    def add(self, cfg: MCPServerConfig) -> None:
        with self._lock:
            self._configs[cfg.name] = cfg

    def add_many(self, cfgs: List[MCPServerConfig]) -> None:
        for c in cfgs:
            self.add(c)

    # ── 连接 ────────────────────────────────
    def connect_all(self) -> Dict[str, bool]:
        results: Dict[str, bool] = {}
        for name, cfg in list(self._configs.items()):
            if not cfg.enabled:
                results[name] = False
                continue
            try:
                self._connect_one(cfg)
                results[name] = True
            except Exception:
                logger.exception("MCP connect failed: %s", name)
                results[name] = False
        return results

    def _connect_one(self, cfg: MCPServerConfig) -> MCPClient:
        if cfg.transport == "stdio":
            if not cfg.command:
                raise ValueError(f"MCP '{cfg.name}': stdio transport needs command")
            client = MCPClient.stdio(
                cfg.command, env=cfg.env, cwd=cfg.cwd, timeout=cfg.timeout,
            )
        elif cfg.transport == "sse":
            if not cfg.url:
                raise ValueError(f"MCP '{cfg.name}': sse transport needs url")
            client = MCPClient.sse(cfg.url, timeout=cfg.timeout,
                                   headers=cfg.headers,   )
        else:
            raise ValueError(f"MCP '{cfg.name}': unknown transport {cfg.transport}")

        with self._lock:
            self._clients[cfg.name] = client
        logger.info("MCP client connected: %s (%s)", cfg.name, cfg.transport)
        return client

    # ── 访问 ────────────────────────────────
    def get(self, name: str) -> Optional[MCPClient]:
        return self._clients.get(name)

    def names(self) -> List[str]:
        return list(self._clients.keys())

    # ── 关闭 ────────────────────────────────
    def close_all(self) -> None:
        with self._lock:
            clients = list(self._clients.items())
            self._clients.clear()

        for name, client in clients:
            try:
                client.close()
                logger.info("MCP client closed: %s", name)
            except Exception:
                logger.exception("MCP close failed: %s", name)
