# mcp_protocol/client.py
"""
MCP Client - 连接 MCP Server

用法:
    # stdio: 起一个子进程跑 server
    client = MCPClient.stdio(["python", "-m", "my_server"])
    tools = client.list_tools()
    result = client.call_tool("add", {"a": 1, "b": 2})
    client.close()

    # sse: 连远端
    client = MCPClient.sse("http://127.0.0.1:8080/sse")
    ...
"""
from __future__ import annotations
import json
import logging
import subprocess
import threading
import time
from queue import Empty, Queue
from typing import Any, Dict, List, Optional

import urllib.request
import urllib.error
import os
from dotenv import load_dotenv


load_dotenv() 

logger = logging.getLogger(__name__)

PROTOCOL_VERSION = "2024-11-05"


class MCPError(Exception):
    def __init__(self, code: int, message: str):
        super().__init__(f"[{code}] {message}")
        self.code = code
        self.message = message


class MCPClient:
    """MCP 客户端（基类，不要直接实例化）"""

    def __init__(self):
        self._next_id = 1
        self._id_lock = threading.Lock()
        self.server_info: Dict[str, Any] = {}
        self.capabilities: Dict[str, Any] = {}

    # ── 工厂 ──────────────────────────────
    @classmethod
    def stdio(
        cls,
        cmd: List[str],
        env: Optional[Dict[str, str]] = None,
        cwd: Optional[str] = None,
        timeout: float = 30.0,
    ) -> "StdioMCPClient":
        return StdioMCPClient(cmd, env=env, cwd=cwd, timeout=timeout)

    @classmethod
    def sse(cls, url: str, timeout: float = 30.0, headers: Optional[Dict[str, str]] = None) -> "SSEMCPClient":
        return SSEMCPClient(url, timeout=timeout, headers=headers)

    # ── 公共 API ──────────────────────────
    def initialize(self) -> Dict[str, Any]:
        resp = self._request("initialize", {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": {"name": "agent-harness-client", "version": "1.0.0"},
        })
        self.server_info = resp.get("serverInfo", {})
        self.capabilities = resp.get("capabilities", {})
        self._notify("notifications/initialized", {})
        return resp

    def list_tools(self) -> List[Dict[str, Any]]:
        return self._request("tools/list", {}).get("tools", [])

    def call_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        resp = self._request("tools/call", {"name": name, "arguments": arguments})
        if resp.get("isError"):
            raise MCPError(-32000, f"tool '{name}' failed: {resp.get('content')}")
        return self._extract_text(resp)

    def list_resources(self) -> List[Dict[str, Any]]:
        return self._request("resources/list", {}).get("resources", [])

    def read_resource(self, uri: str) -> str:
        resp = self._request("resources/read", {"uri": uri})
        contents = resp.get("contents", [])
        if contents:
            return contents[0].get("text", "")
        return ""

    def list_prompts(self) -> List[Dict[str, Any]]:
        return self._request("prompts/list", {}).get("prompts", [])

    def get_prompt(self, name: str, arguments: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return self._request("prompts/get", {
            "name": name, "arguments": arguments or {},
        })

    def close(self) -> None:
        pass

    # ── 底层：请求 / 通知 ─────────────────
    def _alloc_id(self) -> int:
        with self._id_lock:
            i = self._next_id
            self._next_id += 1
            return i

    @staticmethod
    def _extract_text(resp: Dict[str, Any]) -> Any:
        """把 MCP content 数组转成字符串（多块用 \n 拼）"""
        contents = resp.get("content", [])
        if not contents:
            return ""
        parts = []
        for c in contents:
            if c.get("type") == "text":
                parts.append(c.get("text", ""))
            elif c.get("type") == "resource":
                parts.append(json.dumps(c.get("resource", {}), ensure_ascii=False))
            elif c.get("type") == "image":
                parts.append(f"[image {c.get('mimeType', 'unknown')}]")
            else:
                parts.append(str(c))
        return "\n".join(parts)

    def _request(self, method: str, params: Dict[str, Any]) -> Any:
        raise NotImplementedError

    def _notify(self, method: str, params: Dict[str, Any]) -> None:
        raise NotImplementedError


# ══════════════════════════════════════════════
# stdio 客户端
# ══════════════════════════════════════════════
class StdioMCPClient(MCPClient):
    def __init__(
        self,
        cmd: List[str],
        env: Optional[Dict[str, str]] = None,
        cwd: Optional[str] = None,
        timeout: float = 30.0,
    ):
        super().__init__()
        self.timeout = timeout

        # P0: 合并系统环境变量，避免子进程丢 PATH
        merged_env = os.environ.copy()
        if env:
            merged_env.update(env)

        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env, cwd=cwd, text=True, bufsize=1,
        )
        self._pending: Dict[int, Queue] = {}
        self._pending_lock = threading.Lock()
        self._reader_alive = True

        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()
        self._stderr_reader = threading.Thread(target=self._stderr_loop, daemon=True)
        self._stderr_reader.start()

        # 初始化
        self.initialize()

    def _read_loop(self) -> None:
        try:
            for line in self.proc.stdout:
                line = line.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning("client: bad json: %s", line[:200])
                    continue

                msg_id = msg.get("id")
                if msg_id is None:
                    continue
                with self._pending_lock:
                    q = self._pending.pop(msg_id, None)
                if q:
                    q.put(msg)
        except Exception:
            logger.exception("stdio reader crashed")
        finally:
            self._reader_alive = False
            # 唤醒所有等待者
            with self._pending_lock:
                for q in self._pending.values():
                    q.put({"error": {"code": -32000, "message": "server closed"}})
                self._pending.clear()

    def _stderr_loop(self) -> None:
        try:
            for line in self.proc.stderr:
                logger.debug("[server stderr] %s", line.rstrip())
        except Exception:
            pass

    def _request(self, method: str, params: Dict[str, Any]) -> Any:
        if not self._reader_alive:
            raise MCPError(-32000, "server process not alive")

        msg_id = self._alloc_id()
        q: Queue = Queue()
        with self._pending_lock:
            self._pending[msg_id] = q

        payload = {"jsonrpc": "2.0", "id": msg_id, "method": method, "params": params}
        self.proc.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self.proc.stdin.flush()

        try:
            resp = q.get(timeout=self.timeout)
        except Empty:
            with self._pending_lock:
                self._pending.pop(msg_id, None)
            raise MCPError(-32000, f"request timeout: {method}")

        if "error" in resp:
            e = resp["error"]
            raise MCPError(e.get("code", -32000), e.get("message", "unknown"))
        return resp.get("result", {})

    def _notify(self, method: str, params: Dict[str, Any]) -> None:
        payload = {"jsonrpc": "2.0", "method": method, "params": params}
        try:
            self.proc.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
            self.proc.stdin.flush()
        except Exception:
            logger.exception("notify failed")

    def close(self) -> None:
        try:
            self.proc.terminate()
            self.proc.wait(timeout=3)
        except Exception:
            try:
                self.proc.kill()
            except Exception:
                pass


# ══════════════════════════════════════════════
# SSE 客户端
# ══════════════════════════════════════════════
class SSEMCPClient(MCPClient):
    """
    简易 SSE 客户端。
    流程：
      1. GET /sse 建立长连接，拿到 /message?session_id=xxx
      2. POST /message?session_id=xxx 发请求
      3. 响应通过 SSE 长连接返回，reader 线程接住并投递给 pending
    """

    def __init__(self, sse_url: str, timeout: float = 30.0, headers: Optional[Dict[str, str]] = None):
        super().__init__()
        self.sse_url = sse_url
        self.timeout = timeout
        self.message_url: Optional[str] = None

        self._pending: Dict[int, Queue] = {}
        self._pending_lock = threading.Lock()
        self._endpoint_ready = threading.Event()

        self._reader = threading.Thread(target=self._sse_loop, daemon=True)
        self._reader.start()

        self.headers = headers or {}

        # 等 endpoint 事件
        if not self._endpoint_ready.wait(timeout=10):
            raise MCPError(-32000, "SSE endpoint handshake timeout")

        self.initialize()

    def _sse_loop(self) -> None:
        try:
            req = urllib.request.Request(
                self.sse_url, headers={"Accept": "text/event-stream", **self.headers},
            )
            with urllib.request.urlopen(req, timeout=None) as resp:
                event_type = None
                for raw in resp:
                    line = raw.decode("utf-8").rstrip("\n")
                    if line.startswith("event:"):
                        event_type = line.split(":", 1)[1].strip()
                    elif line.startswith("data:"):
                        data = line.split(":", 1)[1].strip()
                        if event_type == "endpoint":
                            # data 里是 /message?session_id=xxx
                            from urllib.parse import urljoin
                            base = self.sse_url.rsplit("/", 1)[0]
                            self.message_url = urljoin(base + "/", data.lstrip("/"))
                            self._endpoint_ready.set()
                        elif event_type == "message":
                            try:
                                msg = json.loads(data)
                            except json.JSONDecodeError:
                                continue
                            msg_id = msg.get("id")
                            if msg_id is None:
                                continue
                            with self._pending_lock:
                                q = self._pending.pop(msg_id, None)
                            if q:
                                q.put(msg)
                    elif line == "":
                        event_type = None
        except Exception:
            logger.exception("SSE reader crashed")
            with self._pending_lock:
                for q in self._pending.values():
                    q.put({"error": {"code": -32000, "message": "SSE closed"}})
                self._pending.clear()

    def _request(self, method: str, params: Dict[str, Any]) -> Any:
        if not self.message_url:
            raise MCPError(-32000, "endpoint not ready")

        msg_id = self._alloc_id()
        q: Queue = Queue()
        with self._pending_lock:
            self._pending[msg_id] = q

        payload = {"jsonrpc": "2.0", "id": msg_id, "method": method, "params": params}
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")

        try:
            req = urllib.request.Request(
                self.message_url, data=body,
                headers={"Content-Type": "application/json", **self.headers} ,method="POST",
            )
            urllib.request.urlopen(req, timeout=self.timeout).read()
        except urllib.error.HTTPError as e:
            with self._pending_lock:
                self._pending.pop(msg_id, None)
            raise MCPError(-32000, f"HTTP {e.code}")

        try:
            resp = q.get(timeout=self.timeout)
        except Empty:
            with self._pending_lock:
                self._pending.pop(msg_id, None)
            raise MCPError(-32000, f"request timeout: {method}")

        if "error" in resp:
            e = resp["error"]
            raise MCPError(e.get("code", -32000), e.get("message", "unknown"))
        return resp.get("result", {})

    def _notify(self, method: str, params: Dict[str, Any]) -> None:
        if not self.message_url:
            return
        payload = {"jsonrpc": "2.0", "method": method, "params": params}
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        try:
            req = urllib.request.Request(
                self.message_url, data=body,
                headers={"Content-Type": "application/json",**self.headers }, method="POST",
            )
            urllib.request.urlopen(req, timeout=self.timeout).read()
        except Exception:
            logger.exception("notify failed")

    def close(self) -> None:
        # reader 线程是 daemon，进程退出自动结束
        pass
