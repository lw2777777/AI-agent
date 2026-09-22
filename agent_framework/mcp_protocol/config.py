# mcp_protocol/config.py
"""
MCP 配置加载

支持三种输入：
  1. 文件路径（str / Path）
  2. dict：{"mcpServers": {...}} / {"mcp_servers": {...}} / {"servers": [...]}
  3. list[dict]（旧格式）

配置示例（推荐，兼容 MCP 生态）：
    {
      "mcpServers": {
        "filesystem": {
          "type": "stdio",
          "command": "python",
          "args": ["-m", "mcp_server_fs"],
          "cwd": "/tmp/sandbox",
          "env": {"LOG_LEVEL": "info"}
        },
        "search": {
          "type": "sse",
          "url": "$SEARCH_MCP_URL",
          "headers": {"Authorization": "Bearer $SEARCH_TOKEN"}
        }
      }
    }
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from .mcp_lifecycle import MCPServerConfig

logger = logging.getLogger(__name__)


def _resolve_env(value: Any) -> Any:
    """递归解析 $ENV_VAR 占位符。未定义的变量 → 空串。"""
    if isinstance(value, str):
        if value.startswith("$"):
            return os.getenv(value[1:], "")
        return value
    if isinstance(value, dict):
        return {k: _resolve_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_env(v) for v in value]
    return value


def load_mcp_configs(raw: Any) -> List[MCPServerConfig]:
    """
    从多种输入加载配置。见模块 docstring。
    """
    if raw is None:
        return []

    # 文件路径
    if isinstance(raw, (str, Path)):
        p = Path(raw)
        if not p.exists():
            logger.warning("MCP config file not found: %s", p)
            return []
        raw = json.loads(p.read_text(encoding="utf-8"))

    # 解析 $ENV_VAR
    raw = _resolve_env(raw)

    # 统一成 list[dict]
    if isinstance(raw, dict):
        # 优先官方格式
        servers = (
            raw.get("mcpServers")
            or raw.get("mcp_servers")
            or raw.get("servers")
            or {}
        )
        # dict 形式 → list
        if isinstance(servers, dict):
            items = [{"name": k, **v} for k, v in servers.items()]
        else:
            items = servers
    elif isinstance(raw, list):
        items = raw
    else:
        logger.warning("unsupported MCP config shape: %r", type(raw))
        return []

    configs: List[MCPServerConfig] = []
    for item in items:
        try:
            configs.append(_parse_one(item))
        except Exception:
            logger.exception("bad MCP config entry: %r", item)
    return configs


def _parse_one(item: Dict[str, Any]) -> MCPServerConfig:
    name = item.get("name")
    if not name:
        raise ValueError("MCP server config missing 'name'")

    # 兼容 type / transport
    transport = item.get("type") or item.get("transport") or "stdio"

    # command 支持 str 或 list
    command = item.get("command")
    if isinstance(command, str):
        # 生态格式：command 是字符串，args 单独给
        args = item.get("args") or []
        command = [command] + list(args)
    # command 是 list 就用 list

    return MCPServerConfig(
        name=name,
        transport=transport,
        command=command,
        env=item.get("env"),
        cwd=item.get("cwd"),
        url=item.get("url"),
        headers=item.get("headers"),           # ★ 新增
        timeout=float(item.get("timeout", 30.0)),
        enabled=bool(item.get("enabled", True)),
    )