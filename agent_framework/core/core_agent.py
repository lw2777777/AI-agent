"""
Core Agent - 完整的Agent实现

整合所有功能:
1. ReAct循环 (推理+行动)
2. 记忆系统 (工作记忆 + 长期记忆)
3. 反思机制 (自我改进)
4. 工具调用
5. 多Agent协作支持
6. MCP协议支持
"""

from collections import deque
import hashlib

from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import logging
from datetime import datetime

from .memory.semantic_memory import ArchivalMemoryStore
from .reflector import Reflector, ReflectionResult
from .trace import Tracer
import time
from .context_manager import ContextManager
from .retry import RetryPolicy, retry_call
from .circuit_breaker import CircuitBreaker
from .error_recovery import ErrorRecovery, RecoveryAction

from typing import Tuple

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# ★ FIX: 结构化工具结果
# ─────────────────────────────────────────────
@dataclass
class ToolResult:
    """结构化工具结果，替代脆弱的 startswith('Error:')"""
    success: bool
    content: str
    error: Optional[str] = None
    error_type: Optional[str] = None   # "validation" | "execution" | "guardrail" | "not_found"

    def to_observation(self) -> str:
        return self.content if self.success else f"Error: {self.content}"


def _validate_against_schema(inputs: Dict[str, Any], schema: Dict[str, Any]) -> Tuple[bool, str]:
    """轻量 JSON Schema 子集校验：type / required / enum / minimum / maximum。"""
    params = schema.get("parameters", {}) or {}
    props = params.get("properties", {}) or {}
    required = set(params.get("required", []) or [])

    if not isinstance(inputs, dict):
        return False, f"arguments must be an object, got {type(inputs).__name__}"

    missing = [k for k in required if k not in inputs or inputs[k] is None]
    if missing:
        return False, f"missing required field(s): {', '.join(missing)}"

    type_map = {
        "string": str, "integer": int, "number": (int, float),
        "boolean": bool, "array": list, "object": dict,
    }
    for key, val in inputs.items():
        spec = props.get(key)
        if not spec:
            continue
        expected = spec.get("type")
        if expected in ("integer", "number") and isinstance(val, bool):
            return False, f"field '{key}' expects {expected}, got bool"
        if expected in type_map and not isinstance(val, type_map[expected]):
            return False, f"field '{key}' expects {expected}, got {type(val).__name__}"
        if "enum" in spec and val not in spec["enum"]:
            return False, f"field '{key}' must be one of {spec['enum']}, got {val!r}"
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            if "minimum" in spec and val < spec["minimum"]:
                return False, f"field '{key}' must be >= {spec['minimum']}, got {val}"
            if "maximum" in spec and val > spec["maximum"]:
                return False, f"field '{key}' must be <= {spec['maximum']}, got {val}"

    return True, ""


class AgentState(Enum):
    IDLE = "idle"
    THINKING = "thinking"
    ACTING = "acting"
    OBSERVING = "observing"
    REFLECTING = "reflecting"
    DONE = "done"
    ERROR = "error"


@dataclass
class AgentStep:
    thought: str
    action: Optional[str] = None
    action_input: Optional[Dict] = None
    observation: Optional[str] = None
    success: bool = True
    error: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class AgentResult:
    success: bool
    answer: str
    steps: List[AgentStep]
    iterations: int
    state: AgentState
    error: Optional[str] = None
    reflection: Optional[ReflectionResult] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class CoreAgent:
    def __init__(
        self,
        llm_retry_policy: Optional[RetryPolicy] = None,
        circuit_threshold: int = 5,
        guardrails: Optional[Any] = None,
        memory_manager: Optional[Any] = None,
        context_max_tokens: int = 8000,
        context_keep_recent: int = 4,
        enable_context_compression: bool = True,
        name: str = "CoreAgent",
        llm: Optional[Any] = None,
        tools: Optional[List[Any]] = None,
        memory_store: Optional[ArchivalMemoryStore] = None,
        max_iterations: int = 10,
        enable_reflection: bool = True,
        verbose: bool = True,
        tracer: Optional[Tracer] = None,
        **kwargs,
    ):
        self.name = name
        self.llm = llm
        self.tools = self._index_tools(tools or [])
        self.memory_store = memory_store
        self.max_iterations = max_iterations
        self.enable_reflection = enable_reflection
        self.verbose = verbose

        self.state = AgentState.IDLE
        self.steps: List[AgentStep] = []
        self.context: Dict[str, Any] = {}

        if enable_reflection and llm is not None:
            try:
                self._reflector = Reflector(
                    llm=self._make_reflector_llm(llm),
                    memory_store=memory_store,
                )
            except Exception:
                logger.exception("Reflector init failed; task-level reflection disabled")
                self._reflector = None
        else:
            self._reflector = None

        self.tracer = tracer or Tracer()
        self.llm_retry_policy = llm_retry_policy or RetryPolicy()
        self._circuit = CircuitBreaker(threshold=circuit_threshold)
        self.guardrails = guardrails

        self.enable_context_compression = enable_context_compression
        self._context_manager = ContextManager(
            max_tokens=context_max_tokens,
            keep_recent=context_keep_recent,
            llm=self._make_summary_llm(llm),
        )
        self.memory_manager = memory_manager

        self.error_recovery = ErrorRecovery(
            fallback_models=kwargs.get("fallback_models", []),
            fallback_tools=kwargs.get("fallback_tools", {}),
            on_degrade=lambda partial: self._partial_answer(),
        )

        self._compression_stats = {
            "total_compressions": 0,
            "total_tokens_saved": 0,
        }

        self._loop_window = kwargs.get("loop_guard_window", 10)
        self._loop_exact_limit = kwargs.get("exact_repeat_limit", 3)
        self._loop_pattern_limit = kwargs.get("loop_pattern_limit", 2)
        self._loop_history: deque = deque(maxlen=self._loop_window)
        self._loop_embedder = kwargs.get("loop_guard_embedder")
        self._loop_vec_cache: Dict[str, Any] = {}

        self.high_risk_tools = kwargs.get(
            "high_risk_tools",
            {"transfer", "delete", "send_email", "execute_sql"},
        )
        self.always_reflect = kwargs.get("always_reflect", False)

        self.system_prompt = self._build_system_prompt()

    # ─────────────────────────────────────────────
    # LLM 适配器
    # ─────────────────────────────────────────────
    def _make_summary_llm(self, llm):
        if llm is None:
            return None

        class _SummaryLLM:
            def __init__(self, inner, agent):
                self._inner = inner
                self._agent = agent

            def chat(self, messages, tools=None, tool_choice=None, **kwargs):
                def _do():
                    return self._inner.chat(messages=messages, tools=None, tool_choice=None)
                try:
                    return retry_call(_do, self._agent.llm_retry_policy)
                except Exception:
                    logger.exception("summary LLM failed after retries")
                    raise

        return _SummaryLLM(llm, self)

    def _make_reflector_llm(self, llm):
        if llm is None:
            return None
        agent = self

        class _ReflectorLLM:
            def __init__(self, inner):
                self._inner = inner

            def chat(self, prompt, **kwargs):
                if isinstance(prompt, str):
                    messages = [{"role": "user", "content": prompt}]
                else:
                    messages = prompt

                def _do():
                    return self._inner.chat(messages=messages, tools=None, tool_choice=None)
                return retry_call(_do, agent.llm_retry_policy)

        return _ReflectorLLM(llm)

    # ─────────────────────────────────────────────
    # 工具索引 / 提示词
    # ─────────────────────────────────────────────
    def _index_tools(self, tools: List[Any]) -> Dict[str, Any]:
        indexed = {}
        for tool in tools:
            name = getattr(tool, "name", tool.__class__.__name__)
            indexed[name] = tool
        return indexed

    def _build_system_prompt(self) -> str:
        tool_descriptions = self._format_tools()
        return f"""You are {self.name}, an AI assistant that uses tools to accomplish tasks.

    ## Available Tools
    {tool_descriptions}

    ## Instructions
    1. Think step by step about what to do
    2. Use the provided tools when needed
    3. Observe results and continue
    4. When the task is complete, provide your final answer as plain text

    Important:
    - Do not describe tool calls in text. Use the tool calling mechanism directly.
    - When you have enough information, respond with plain text (no tool call).
    - If a tool fails, analyze the error and decide whether to retry or try another approach.
    """

    def _format_tools(self) -> str:
        if not self.tools:
            return "No tools available."
        lines = []
        for name, tool in self.tools.items():
            desc = getattr(tool, "description", "No description")
            lines.append(f"- {name}: {desc}")
            schema = None
            if hasattr(tool, "to_schema"):
                try:
                    schema = tool.to_schema()
                except Exception:
                    schema = None
            if schema and "parameters" in schema:
                props = schema["parameters"].get("properties", {})
                required = set(schema["parameters"].get("required", []))
                for pname, pschema in props.items():
                    req = " (required)" if pname in required else ""
                    pdesc = pschema.get("description", "")
                    ptype = pschema.get("type", "any")
                    lines.append(f"    - {pname}: {ptype}{req} - {pdesc}")
        return "\n".join(lines)

    def _format_history(self) -> str:
        if not self.steps:
            return "No previous steps."
        lines = []
        for i, step in enumerate(self.steps, 1):
            lines.append(f"Step {i}:")
            lines.append(f"  Thought: {step.thought}")
            if step.action:
                lines.append(f"  Action: {step.action}")
                lines.append(f"  Input: {json.dumps(step.action_input or {})}")
            if step.observation:
                obs = step.observation[:200] + "..." if len(step.observation) > 200 else step.observation
                lines.append(f"  Observation: {obs}")
            lines.append("")
        return "\n".join(lines)

    def _get_memory_context(self, query: str, current_thought: str = "") -> str:
        if not self.memory_manager:
            return ""
        ctx = self.memory_manager.retrieve_for_decision(
            symbol=self.context.get("symbol", ""),
            regime=self.context.get("regime", "unknown"),
            signal_types=self.context.get("signal_types", []),
            query=f"{query} {current_thought}".strip(),
            top_k=5,
            agent_id=self.name,
        )
        parts = []
        if ctx["strategy"]:
            parts.append("## Strategy Experience (current regime)")
            for s in ctx["strategy"]:
                parts.append(
                    f"- {s['signal_type']}: n={s['n_trades']} "
                    f"win_rate={s['win_rate']:.2f} avg_pnl={s['avg_pnl']:.4f} "
                    f"sharpe_like={s['sharpe_like']:.2f}"
                )
        if ctx["episodic"]:
            parts.append("\n## Similar Past Trades")
            for e in ctx["episodic"][:3]:
                parts.append(
                    f"- {e['symbol']} {e['action']} @ {e['entry_time']} "
                    f"→ {e['exit_reason']} pnl_pct={e.get('pnl_pct', 0):.4f}"
                )
        if ctx["semantic"]:
            parts.append("\n## Relevant Lessons")
            for m in ctx["semantic"][:3]:
                parts.append(f"- {m['content'][:200]}")
        return "\n".join(parts)

    def _call_llm_with_retry(self, messages, tool_schemas):
        def _do_call():
            return self.llm.chat(
                messages=messages,
                tools=tool_schemas if tool_schemas else None,
                tool_choice="auto" if tool_schemas else None,
            )

        def _on_retry(attempt, exc, delay):
            logger.warning(
                "LLM call retry %d after %s, waiting %.1fs",
                attempt + 1, type(exc).__name__, delay,
            )

        return retry_call(_do_call, self.llm_retry_policy, on_retry=_on_retry)

    # ─────────────────────────────────────────────
    # ★ FIX: 工具执行（返回 ToolResult）
    # ─────────────────────────────────────────────
    def _execute_tool(self, name: str, inputs: Dict) -> ToolResult:
        tool = self.tools.get(name)
        if not tool:
            return ToolResult(
                success=False,
                content=f"Tool '{name}' not found. Available: {list(self.tools.keys())}",
                error_type="not_found",
            )

        schema = None
        if hasattr(tool, "to_schema"):
            try:
                schema = tool.to_schema()
            except Exception:
                schema = None
        if schema:
            ok, err = _validate_against_schema(inputs or {}, schema)
            if not ok:
                return ToolResult(
                    success=False,
                    content=(
                        f"Invalid arguments for '{name}': {err}. "
                        f"Please fix the arguments and retry."
                    ),
                    error_type="validation",
                )

        if self.guardrails is not None:
            verdict = self.guardrails.check(
                tool_name=name,
                arguments=inputs,
                session_id=self.context.get("session_id"),
            )
            if not verdict.allowed:
                logger.warning(
                    "Guardrail blocked %s [%s]: %s",
                    name, verdict.risk_level.value, verdict.reason,
                )
                return ToolResult(
                    success=False,
                    content=f"blocked by guardrail [{verdict.risk_level.value}]: {verdict.reason}",
                    error_type="guardrail",
                )
            if verdict.modified_args is not None:
                inputs = verdict.modified_args

        try:
            handler = getattr(tool, "handler", None) or tool
            result = handler(**(inputs or {}))
            return ToolResult(success=True, content=str(result))
        except Exception as e:
            logger.exception(f"Tool {name} failed")
            return ToolResult(
                success=False,
                content=f"{type(e).__name__}: {e}",
                error_type="execution",
            )

    # ─────────────────────────────────────────────
    # ★ FIX: 净化 LLM 返回的 tool_calls
    # ─────────────────────────────────────────────
    def _sanitize_tool_calls(self, raw_tool_calls: List[Any]) -> List[Dict[str, Any]]:
        cleaned = []
        for tc in raw_tool_calls or []:
            if not isinstance(tc, dict):
                logger.warning("drop non-dict tool_call: %r", tc)
                continue
            name = tc.get("name")
            call_id = tc.get("id")
            args = tc.get("arguments")
            if not name or not call_id:
                logger.warning("drop tool_call missing name/id: %r", tc)
                continue
            if args is None:
                args = {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    logger.warning("tool_call arguments not JSON: %r", args)
                    args = {}
            if not isinstance(args, dict):
                logger.warning("tool_call arguments not object: %r", args)
                args = {}
            cleaned.append({"name": name, "arguments": args, "id": call_id})
        return cleaned

    # ─────────────────────────────────────────────
    # 主入口
    # ─────────────────────────────────────────────
    def run(
        self,
        query: str,
        config: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
        stream: bool = False,
        checkpoint: Optional[Dict[str, Any]] = None,
        on_step: Optional[Callable[[int, Dict[str, Any]], None]] = None,
    ) -> AgentResult:
        # ★ FIX: 提前初始化，保证 finally 安全
        messages: List[Dict[str, Any]] = []

        try:
            with self.tracer.trace("agent.run", **{
                "run.query": query[:200],
                "agent.name": self.name,
                "agent.max_iterations": self.max_iterations,
            }):
                self.tracer.emit(
                    "run_start",
                    step_index=-1,
                    query=query[:200],
                    session_id=session_id,
                    agent_name=self.name,
                    max_iterations=self.max_iterations,
                )

                self.state = AgentState.THINKING

                # ── 1. 初始化：断点恢复 or 全新 ──
                if checkpoint:
                    self.import_state(checkpoint)
                    messages = list(checkpoint.get("messages", []))
                    start_iter = int(checkpoint.get("next_iteration", 0))
                    if self.verbose:
                        logger.info(
                            "♻️ Resume from iteration %d, %d steps, %d messages",
                            start_iter, len(self.steps), len(messages),
                        )
                    if not messages or messages[0].get("role") != "system":
                        messages.insert(0, {"role": "system", "content": self.system_prompt})
                else:
                    self.steps = []
                    start_iter = 0
                    self.context = {
                        "query": query,
                        "config": config or {},
                        "session_id": session_id,
                        "start_time": datetime.now(),
                    }
                    if session_id:
                        messages = self._load_session_messages(session_id)
                    else:
                        messages = []
                    if not messages or messages[0].get("role") != "system":
                        messages.insert(0, {"role": "system", "content": self.system_prompt})
                    memory_context = self._get_memory_context(query)
                    user_content = query
                    if memory_context:
                        user_content = f"{memory_context}\n\n## Task\n{query}"
                    messages.append({"role": "user", "content": user_content})

                if self.verbose:
                    logger.info(f"🚀 Agent {self.name} starting task: {query[:100]}...")

                # ── 2. 工具 schema（★ FIX: 移出 else，两条路径都要有）──
                tool_schemas = [tool.to_schema() for tool in self.tools.values()]

                # ★ FIX: 循环防护 reset（移出 else）
                self._loop_history.clear()
                self._loop_vec_cache.clear()

                # ── 3. 主循环 ──
                for i in range(start_iter, self.max_iterations):
                    if self.verbose:
                        logger.info(f"🔄 Iteration {i+1}/{self.max_iterations}")
                    self.state = AgentState.THINKING

                    # 上下文压缩
                    if self.enable_context_compression:
                        try:
                            comp = self._context_manager.compress(messages, task=query)
                            if comp.compressed:
                                messages = comp.messages
                                self._compression_stats["total_compressions"] += 1
                                saved = comp.original_tokens - comp.final_tokens
                                self._compression_stats["total_tokens_saved"] += saved
                                logger.info(
                                    "🗜️ context compressed: %d→%d tokens "
                                    "(saved %d, dropped %d msgs, truncated %d tools)",
                                    comp.original_tokens, comp.final_tokens,
                                    saved, comp.dropped_messages, comp.truncated_tool_results,
                                )
                                self.tracer.emit(
                                    "context_compressed",
                                    step_index=i,
                                    original_tokens=comp.original_tokens,
                                    final_tokens=comp.final_tokens,
                                    saved_tokens=saved,
                                    dropped_messages=comp.dropped_messages,
                                    truncated_tool_results=comp.truncated_tool_results,
                                    summary_tokens=comp.summary_tokens,
                                )
                        except Exception:
                            logger.exception("context compression failed; continuing with full context")

                    # ── LLM 调用 ──
                    t0 = time.time()
                    self.tracer.emit(
                        "llm_call",
                        step_index=i,
                        model=getattr(self.llm, "model", "unknown"),
                        n_messages=len(messages),
                        n_tools=len(tool_schemas),
                    )

                    try:
                        resp = self._call_llm_with_retry(messages, tool_schemas)
                        self._circuit.record_success()
                    except Exception as e:
                        logger.exception("LLM call failed after retries")
                        self._circuit.record_failure()

                        decision = self.error_recovery.decide(
                            e, context={"step_index": i, "model": getattr(self.llm, "model", None)}
                        )
                        self.tracer.emit("error_recovery", step_index=i,
                                         action=decision.action.value, reason=decision.reason)

                        if decision.action == RecoveryAction.RETRY:
                            continue

                        if decision.action == RecoveryAction.FALLBACK_MODEL:
                            self._swap_model(decision.fallback_model)
                            continue

                        if decision.action == RecoveryAction.DEGRADE:
                            result = AgentResult(
                                success=False,
                                answer=self.error_recovery.degrade(self._partial_answer()),
                                steps=self.steps, iterations=i + 1,
                                state=AgentState.ERROR, error=str(e),
                            )
                            self._finalize(result, query, session_id)
                            self.tracer.emit("run_end", step_index=i, success=False,
                                             iterations=i + 1, error=f"llm_error: {e}")
                            return result

                        if self._circuit.is_open():
                            result = AgentResult(
                                success=False,
                                answer=self._partial_answer(),
                                steps=self.steps,
                                iterations=i + 1,
                                state=AgentState.ERROR,
                                error=f"circuit_breaker_open: {e}",
                                metadata={"circuit": self._circuit.snapshot()},
                            )
                            self._finalize(result, query, session_id)
                            self.tracer.emit("run_end", step_index=i, success=False,
                                             iterations=i + 1, error=f"circuit_breaker_open: {e}")
                            return result

                        result = AgentResult(
                            success=False, answer=f"LLM error: {e}",
                            steps=self.steps, iterations=i + 1,
                            state=AgentState.ERROR, error=str(e),
                        )
                        self._finalize(result, query, session_id)
                        self.tracer.emit("run_end", step_index=i, success=False,
                                         iterations=i + 1, error=f"llm_error: {e}")
                        return result

                    llm_ms = (time.time() - t0) * 1000
                    usage = resp.get("usage", {})
                    self.tracer.emit(
                        "llm_response",
                        step_index=i,
                        duration_ms=llm_ms,
                        content_len=len(resp.get("content") or ""),
                        n_tool_calls=len(resp.get("tool_calls", [])),
                        prompt_tokens=usage.get("prompt_tokens", 0),
                        completion_tokens=usage.get("completion_tokens", 0),
                        finish_reason=resp.get("finish_reason", ""),
                    )

                    content = resp.get("content")
                    # ★ FIX: 净化 tool_calls
                    tool_calls = self._sanitize_tool_calls(resp.get("tool_calls", []))

                    if self.verbose and content:
                        logger.info(f"💭 Thought: {content[:100]}...")

                    # ── 4. 无工具调用 → 完成 ──
                    if not tool_calls:
                        self.state = AgentState.DONE
                        messages.append({"role": "assistant", "content": content or ""})
                        step = AgentStep(thought=content or "", observation=content or "")
                        self.steps.append(step)

                        result = AgentResult(
                            success=True,
                            answer=content or "",
                            steps=self.steps,
                            iterations=i + 1,
                            state=AgentState.DONE,
                            metadata={
                                "query": query,
                                "session_id": session_id,
                                "usage": resp.get("usage", {}),
                                "messages": messages,
                                "compression_stats": dict(self._compression_stats),
                            },
                        )
                        result.reflection = self._reflect_on_task(query, result)
                        self.tracer.emit("run_end", step_index=i, success=True,
                                         iterations=i + 1, reason="completed")
                        if session_id:
                            self._save_session_messages(session_id, messages)
                        if self.verbose:
                            logger.info(f"✅ Completed in {i+1} iterations")
                        return result

                    # ── 5. 有工具调用 → 执行 ──
                    self.state = AgentState.ACTING
                    messages.append({
                        "role": "assistant",
                        "content": content,
                        "tool_calls": [
                            {
                                "id": tc["id"],
                                "type": "function",
                                "function": {
                                    "name": tc["name"],
                                    "arguments": json.dumps(tc["arguments"], ensure_ascii=False),
                                },
                            }
                            for tc in tool_calls
                        ],
                    })

                    for tc in tool_calls:
                        action_name = tc["name"]
                        action_input = tc["arguments"]
                        call_id = tc["id"]

                        # 循环检测
                        stop, reason = self._loop_check(action_name, action_input)
                        if stop:
                            logger.warning("🔁 Loop detected: %s", reason)
                            self.tracer.emit(
                                "loop_detected", step_index=i, tool=action_name,
                                reason=reason, history_len=len(self.steps),
                            )
                            self.state = AgentState.ERROR
                            result = AgentResult(
                                success=False,
                                answer=f"[loop-guard] Agent stopped: {reason}",
                                steps=self.steps,
                                iterations=i + 1,
                                state=AgentState.ERROR,
                                error=f"loop_detected: {reason}",
                                metadata={"loop_guard": {"reason": reason, "tool": action_name}},
                            )
                            self._finalize(result, query, session_id)
                            return result

                        t0 = time.time()
                        self.tracer.emit(
                            "tool_call", step_index=i, tool=action_name,
                            arguments=action_input, call_id=call_id,
                        )
                        if self.verbose:
                            logger.info(
                                f"🔧 Action: {action_name} "
                                f"{json.dumps(action_input, ensure_ascii=False)[:200]}"
                            )

                        tool_result: ToolResult = self._execute_tool(action_name, action_input)
                        self.state = AgentState.OBSERVING
                        tool_ms = (time.time() - t0) * 1000
                        success = tool_result.success
                        observation = tool_result.to_observation()

                        # ★ FIX: 冗余检测（可选，启用 _loop_observe）
                        try:
                            stop2, reason2 = self._loop_observe(action_name, action_input, observation)
                            if stop2:
                                logger.warning("🔁 Redundant loop: %s", reason2)
                                self.tracer.emit(
                                    "loop_detected", step_index=i, tool=action_name,
                                    reason=reason2, history_len=len(self.steps),
                                )
                                self.state = AgentState.ERROR
                                result = AgentResult(
                                    success=False,
                                    answer=f"[loop-guard] Agent stopped: {reason2}",
                                    steps=self.steps,
                                    iterations=i + 1,
                                    state=AgentState.ERROR,
                                    error=f"loop_detected: {reason2}",
                                    metadata={"loop_guard": {"reason": reason2, "tool": action_name}},
                                )
                                self._finalize(result, query, session_id)
                                return result
                        except Exception:
                            logger.exception("loop_observe failed (ignored)")

                        if success:
                            self._circuit.record_success()
                        elif tool_result.error_type == "validation":
                            pass  # 参数错，LLM 可自愈，不计熔断
                        else:
                            self._circuit.record_failure()
                            observation = (
                                f"{observation}\n\n"
                                f"[System] Failure type={tool_result.error_type}, "
                                f"consecutive failure #{self._circuit.failure_count}. "
                                f"Consider a different approach or tool."
                            )

                        step = AgentStep(
                            thought=content or "",
                            action=action_name,
                            action_input=action_input,
                            observation=str(observation),
                            success=success,
                            error=tool_result.content if not success else None,
                        )
                        self.steps.append(step)

                        messages.append({
                            "role": "tool",
                            "tool_call_id": call_id,
                            "content": str(observation),
                        })

                        self.tracer.emit(
                            "tool_result", step_index=i, duration_ms=tool_ms,
                            tool=action_name, call_id=call_id,
                            success=success, result_preview=str(observation)[:200],
                        )
                        if not success:
                            self.tracer.emit(
                                "error", step_index=i, phase="tool_call",
                                tool=action_name, call_id=call_id,
                                error=str(observation)[:500],
                            )

                        if self._circuit.is_open():
                            logger.error(
                                "circuit breaker open after %d consecutive failures",
                                self._circuit.failure_count,
                            )
                            result = AgentResult(
                                success=False,
                                answer=self._partial_answer(),
                                steps=self.steps,
                                iterations=i + 1,
                                state=AgentState.ERROR,
                                error="circuit_breaker_open",
                                metadata={"circuit": self._circuit.snapshot()},
                            )
                            self._finalize(result, query, session_id)
                            return result

                        if session_id:
                            self._save_session_messages(session_id, messages)

                        if self.verbose:
                            logger.info(f"👁️ Observation: {str(observation)[:100]}")

                        if self._should_reflect({
                            "step_index": i,
                            "tool_name": action_name,
                            "success": success,
                            "error": str(observation) if not success else None,
                            "history": self.steps,
                        }):
                            reflection = self._reflect_on_step(
                                query=query,
                                tool_name=action_name,
                                tool_args=action_input,
                                tool_result=observation,
                                error=str(observation) if not success else None,
                                history=self.steps,
                            )
                            if reflection:
                                messages.append({
                                    "role": "user",
                                    "content": f"[self-reflection] {reflection}",
                                })

                        if success:
                            self._store_successful_step(step, query)

                    if session_id:
                        self._save_session_messages(session_id, messages)

                    if on_step is not None:
                        try:
                            on_step(i + 1, self.export_state(
                                messages=messages,
                                next_iteration=i + 1,
                            ))
                        except Exception:
                            logger.exception("on_step callback failed (ignored)")

                # ── 6. 超出最大迭代 ──
                self.state = AgentState.ERROR
                self.tracer.emit(
                    "run_end",
                    step_index=self.max_iterations - 1,
                    success=False,
                    reason="max_iterations_exceeded",
                    iterations=self.max_iterations,
                )
                if session_id:
                    self._save_session_messages(session_id, messages)

                result = AgentResult(
                    success=False,
                    answer="Reached maximum iterations",
                    steps=self.steps,
                    iterations=self.max_iterations,
                    state=AgentState.ERROR,
                    error="max_iterations_exceeded",
                    metadata={"compression_stats": dict(self._compression_stats)},
                )
                result.reflection = self._reflect_on_task(query, result)
                self._finalize(result, query, session_id)
                return result

        finally:
            if session_id:
                try:
                    self._save_session_messages(session_id, messages)
                except Exception:
                    logger.exception("save session failed")

            if self.tracer is not None:
                try:
                    self.tracer.shutdown()
                except Exception:
                    logger.exception("Tracer shutdown failed")

    # ─────────────────────────────────────────────
    # ★ FIX: 补齐缺失方法
    # ─────────────────────────────────────────────
    def _swap_model(self, fallback_model: Optional[str]):
        """切换 LLM 到备用模型（若适配器支持）"""
        if not fallback_model:
            return
        try:
            if hasattr(self.llm, "model"):
                self.llm.model = fallback_model
            elif hasattr(self.llm, "set_model"):
                self.llm.set_model(fallback_model)
            logger.info("swapped model to %s", fallback_model)
        except Exception:
            logger.exception("swap model failed (ignored)")

    @staticmethod
    def _safe_jsonable(obj: Any) -> Any:
        """把对象转成可 JSON 序列化的形式（递归）"""
        try:
            json.dumps(obj)
            return obj
        except (TypeError, ValueError):
            pass
        if isinstance(obj, dict):
            return {k: CoreAgent._safe_jsonable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [CoreAgent._safe_jsonable(v) for v in obj]
        if isinstance(obj, datetime):
            return obj.isoformat()
        return str(obj)

    @staticmethod
    def _step_to_dict(step: AgentStep) -> Dict[str, Any]:
        return {
            "thought": step.thought,
            "action": step.action,
            "action_input": step.action_input,
            "observation": step.observation,
            "success": step.success,
            "error": step.error,
            "timestamp": step.timestamp.isoformat() if step.timestamp else None,
        }

    @staticmethod
    def _dict_to_step(d: Dict[str, Any]) -> AgentStep:
        ts = d.get("timestamp")
        try:
            ts_obj = datetime.fromisoformat(ts) if ts else datetime.now()
        except Exception:
            ts_obj = datetime.now()
        return AgentStep(
            thought=d.get("thought", ""),
            action=d.get("action"),
            action_input=d.get("action_input"),
            observation=d.get("observation"),
            success=d.get("success", True),
            error=d.get("error"),
            timestamp=ts_obj,
        )

    def _finalize(self, result: AgentResult, query: str, session_id: Optional[str]):
        if not self.memory_manager:
            return
        if self.memory_store is None:
            logger.debug("no memory_store; skip memory write")
            return

        try:
            if result.success:
                content = f"Task succeeded: {query[:100]}. Answer: {result.answer[:200]}"
                importance = 0.6
            else:
                content = f"Task failed: {query[:100]}. Error: {result.error}"
                importance = 0.8
            self.memory_store.save(
                content=content,
                agent_id=self.name,
                memory_type="task",
                metadata={
                    "session_id": session_id,
                    "success": result.success,
                    "iterations": result.iterations,
                },
                importance=importance,
            )
        except Exception:
            logger.exception("task memory write failed")

        if result.reflection:
            try:
                for lesson in getattr(result.reflection, "lessons", []):
                    self.memory_store.save(
                        content=lesson,
                        agent_id=self.name,
                        memory_type="lesson",
                        metadata={"query": query},
                        importance=0.9,
                    )
            except Exception:
                logger.exception("reflection memory write failed")

    # ─────────────────────────────────────────────
    # 反思
    # ─────────────────────────────────────────────
    def _should_reflect(self, context: Dict[str, Any]) -> bool:
        if not getattr(self, "enable_reflection", True):
            return False
        if not context.get("success", True):
            return True

        tool_name = context.get("tool_name")
        history = context.get("history") or []

        def _get_tool(s):
            if isinstance(s, dict):
                return s.get("tool") or s.get("action")
            return getattr(s, "action", None)

        def _get_success(s):
            if isinstance(s, dict):
                return s.get("success", True)
            return getattr(s, "success", True)

        if tool_name:
            same_tool_calls = [s for s in history if _get_tool(s) == tool_name]
            if len(same_tool_calls) >= 2:
                return True

        if tool_name and history:
            last = history[-1]
            if _get_tool(last) != tool_name and not _get_success(last):
                return True

        if tool_name and tool_name in getattr(self, "high_risk_tools", set()):
            return True

        return False

    def _steps_to_dicts(self, steps: List[AgentStep]) -> List[Dict[str, Any]]:
        return [
            {
                "thought": s.thought,
                "action": s.action,
                "action_input": s.action_input,
                "observation": s.observation,
                "success": s.success,
                "error": s.error,
            }
            for s in steps
        ]

    def _reflect_on_step(
        self, query: str, tool_name: str, tool_args: Any,
        tool_result: Any, error: Optional[str], history: List[Any],
    ) -> str:
        if not getattr(self, "enable_reflection", True):
            return ""
        t0 = time.time()
        history_text = "\n".join(
            f"  Step {i}: {s}" for i, s in enumerate(history[-5:])
        ) or "  (无)"

        prompt = f"""你在执行一个 Agent 任务,刚刚某一步出了问题。请做一次简短的反思(80 字以内)，重点是"下一步怎么改"。

## 用户目标
{query}

## 最近执行
{history_text}

## 刚刚这一步
工具: {tool_name}
参数: {tool_args}
结果: {tool_result}
错误: {error or "(无)"}

## 反思要求
1. 一句话判断问题出在哪（参数错？工具选错？思路错？）
2. 给出下一步具体动作：换工具 / 换参数 / 换思路 / 直接回答
请直接输出，不要用 markdown 标题。"""

        try:
            resp = self.llm.chat(
                messages=[{"role": "user", "content": prompt}],
                tools=None, tool_choice=None, temperature=0.2,
            )
            reflection = ""
            if isinstance(resp, dict):
                reflection = (resp.get("content") or "").strip()
            self.tracer.emit(
                "reflection_step",
                duration_ms=(time.time() - t0) * 1000,
                content_len=len(reflection),
                success=bool(reflection),
            )
            if self.verbose and reflection:
                logger.info(f"🪞 Step reflection: {reflection[:100]}...")
            return reflection
        except Exception as e:
            logger.warning(f"Step reflection failed (ignored): {e}")
            return ""

    def _reflect_on_task(self, query: str, result: AgentResult) -> Optional[ReflectionResult]:
        if not getattr(self, "enable_reflection", True):
            return None
        if self._reflector is None:
            return None
        if result.success and len(result.steps) <= 1 and not getattr(self, "always_reflect", False):
            return None

        t0 = time.time()
        try:
            steps_as_dicts = self._steps_to_dicts(result.steps)
            final_result_dict = {
                "success": result.success,
                "answer": result.answer,
                "error": result.error,
                "iterations": result.iterations,
            }
            reflection = self._reflector.reflect_on_task(
                steps=steps_as_dicts, task=query, final_result=final_result_dict,
            )
            self.tracer.emit(
                "reflection_task",
                duration_ms=(time.time() - t0) * 1000,
                success=bool(reflection),
                n_lessons=len(getattr(reflection, "lessons", []) or []),
                n_improvements=len(getattr(reflection, "improvements", []) or []),
            )
            if self.verbose and reflection:
                logger.info(
                    f"🪞 Task reflection: "
                    f"{len(reflection.lessons)} lessons, "
                    f"{len(reflection.improvements)} improvements"
                )
            return reflection
        except Exception:
            logger.exception("Task reflection failed (ignored)")
            return None

    # ─────────────────────────────────────────────
    # session 存储
    # ─────────────────────────────────────────────
    def _load_session_messages(self, session_id: str) -> List[Dict[str, Any]]:
        if not hasattr(self, "_session_store"):
            self._session_store: Dict[str, List[Dict[str, Any]]] = {}
        return list(self._session_store.get(session_id, []))

    def _save_session_messages(self, session_id: str, messages: List[Dict[str, Any]]) -> None:
        if not hasattr(self, "_session_store"):
            self._session_store = {}
        self._session_store[session_id] = list(messages)

    def _store_successful_step(self, step: AgentStep, query: str):
        if not self.memory_store:
            return
        content = f"Successfully used {step.action} for task: {query[:50]}..."
        self.memory_store.save(
            content=content,
            agent_id=self.name,
            memory_type="experience",
            metadata={
                "action": step.action,
                "query": query,
                "observation": step.observation[:200],
            },
            importance=0.7,
        )

    # ─────────────────────────────────────────────
    # 断点：导出 / 导入
    # ─────────────────────────────────────────────
    def export_state(
        self,
        messages: Optional[List[Dict[str, Any]]] = None,
        next_iteration: Optional[int] = None,
    ) -> Dict[str, Any]:
        return {
            "version": 1,
            "agent_name": self.name,
            "state": self.state.value,
            "context": self._safe_jsonable(self.context),
            "steps": [self._step_to_dict(s) for s in self.steps],
            "messages": self._safe_jsonable(messages or []),
            "next_iteration": (
                next_iteration if next_iteration is not None else len(self.steps)
            ),
            "loop_history": [
                {"fp": h.get("fp"), "tool": h.get("tool")}
                for h in self._loop_history
            ],
            "compression_stats": dict(self._compression_stats),
            "saved_at": datetime.now().isoformat(),
        }

    def import_state(self, checkpoint: Dict[str, Any]) -> None:
        state_str = checkpoint.get("state", AgentState.IDLE.value)
        try:
            self.state = AgentState(state_str)
        except ValueError:
            self.state = AgentState.IDLE

        self.context = checkpoint.get("context", {}) or {}
        self.steps = [self._dict_to_step(d) for d in checkpoint.get("steps", [])]

        self._loop_history.clear()
        for h in checkpoint.get("loop_history", []):
            self._loop_history.append({
                "fp": h.get("fp"), "tool": h.get("tool"), "vec": None,
            })

        stats = checkpoint.get("compression_stats")
        if isinstance(stats, dict):
            self._compression_stats.update(stats)

    def restore(self, checkpoint: Dict[str, Any]) -> None:
        inner = checkpoint.get("state", checkpoint)
        self.import_state(inner)
        logger.info(
            "Agent '%s' restored: state=%s, steps=%d, next_iter=%s",
            self.name, self.state.value, len(self.steps), inner.get("next_iteration"),
        )

    # ─────────────────────────────────────────────
    # 其他工具方法
    # ─────────────────────────────────────────────
    def get_response_content(self, response) -> str:
        if isinstance(response, AgentResult):
            return response.answer
        if hasattr(response, "answer"):
            return response.answer
        return str(response)

    def add_tool(self, name: str, tool: Any):
        self.tools[name] = tool

    def reset(self):
        self.state = AgentState.IDLE
        self.steps = []
        self.context = {}
        self._compression_stats = {"total_compressions": 0, "total_tokens_saved": 0}

    def get_stats(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "state": self.state.value,
            "total_steps": len(self.steps),
            "tools": list(self.tools.keys()),
            "max_iterations": self.max_iterations,
            "enable_reflection": self.enable_reflection,
        }

    # ─────────────────────────────────────────────
    # 循环防护
    # ─────────────────────────────────────────────
    def _loop_fingerprint(self, tool: str, args: Any) -> str:
        payload = json.dumps(
            {"tool": tool, "args": args or {}}, sort_keys=True, default=str,
        )
        return hashlib.md5(payload.encode()).hexdigest()

    def _loop_result_fingerprint(self, result: Any) -> str:
        payload = json.dumps(result, sort_keys=True, default=str)
        return hashlib.md5(payload.encode()).hexdigest()

    def _loop_vec(self, tool: str, args: Any):
        if not self._loop_embedder:
            return None
        fp = self._loop_fingerprint(tool, args)
        if fp not in self._loop_vec_cache:
            text = json.dumps({"tool": tool, "args": args or {}}, sort_keys=True, default=str)
            self._loop_vec_cache[fp] = self._loop_embedder(text)
        return self._loop_vec_cache[fp]

    def _loop_detect_cycle(self) -> bool:
        if len(self._loop_history) < 4:
            return False
        fps = [h["fp"] for h in self._loop_history]
        for period in (2, 3):
            if len(fps) < period * self._loop_pattern_limit:
                continue
            pattern = fps[-period:]
            match = 0
            for i in range(len(fps) - period, -1, -period):
                if fps[i - period:i] == pattern:
                    match += 1
                else:
                    break
            if match >= self._loop_pattern_limit:
                return True
        return False

    def _loop_check(self, tool: str, args: Any) -> Tuple[bool, str]:
        fp = self._loop_fingerprint(tool, args)
        repeats = sum(1 for h in self._loop_history if h["fp"] == fp)
        if repeats >= self._loop_exact_limit:
            return True, (
                f"Exact repeat: '{tool}' repeated {repeats + 1} times with same args"
            )
        if self._loop_embedder:
            v = self._loop_vec(tool, args)
            if v is not None:
                for h in self._loop_history:
                    if h.get("vec") is None:
                        continue
                    sim = self._loop_cosine(v, h["vec"])
                    if sim >= 0.95 and h["fp"] != fp:
                        return True, (
                            f"Semantic repeat: sim={sim:.3f} with previous '{h['tool']}'"
                        )
        if self._loop_detect_cycle():
            return True, "Cyclic pattern detected (A-B-A-B)"
        self._loop_history.append({
            "fp": fp, "tool": tool, "vec": self._loop_vec(tool, args),
            "result_fp": None, "result": None,
        })
        return False, ""

    def _loop_observe(self, tool: str, args: Any, result: Any) -> Tuple[bool, str]:
        fp = self._loop_fingerprint(tool, args)
        result_fp = self._loop_result_fingerprint(result)
        for h in reversed(self._loop_history):
            if h["fp"] == fp and h.get("result_fp") is None:
                h["result_fp"] = result_fp
                h["result"] = result
                break
        same_io = [
            h for h in self._loop_history
            if h["fp"] == fp and h.get("result_fp") == result_fp
        ]
        if len(same_io) >= self._loop_exact_limit:
            return True, (
                f"Redundant repeat: '{tool}' produced identical result "
                f"{len(same_io)} times (same input & same output)"
            )
        return False, ""

    @staticmethod
    def _loop_cosine(a, b) -> float:
        import numpy as np
        a, b = np.asarray(a), np.asarray(b)
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1e-10
        return float(a @ b / denom)

    def _partial_answer(self) -> str:
        """熔断时返回部分答案：★ FIX 改用 step.success 判断"""
        for step in reversed(self.steps):
            if step.success and step.observation:
                return f"[partial] {step.observation}"
            if step.thought:
                return f"[partial] {step.thought}"
        return "[partial] No usable result before circuit breaker opened."


# ==================== 便捷函数 ====================

def create_agent(
    name: str = "Agent",
    llm: Optional[Any] = None,
    tools: Optional[List[Any]] = None,
    **kwargs,
) -> CoreAgent:
    return CoreAgent(name=name, llm=llm, tools=tools, **kwargs)