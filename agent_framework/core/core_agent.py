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


logger = logging.getLogger(__name__)


class AgentState(Enum):
    """Agent状态"""
    IDLE = "idle"
    THINKING = "thinking"
    ACTING = "acting"
    OBSERVING = "observing"
    REFLECTING = "reflecting"
    DONE = "done"
    ERROR = "error"


@dataclass
class AgentStep:
    """Agent执行步骤"""
    thought: str
    action: Optional[str] = None
    action_input: Optional[Dict] = None
    observation: Optional[str] = None
    success: bool = True
    error: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class AgentResult:
    """Agent执行结果"""
    success: bool
    answer: str
    steps: List[AgentStep]
    iterations: int
    state: AgentState
    error: Optional[str] = None
    reflection: Optional[ReflectionResult] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class CoreAgent:
    """
    核心Agent - 完整的ReAct Agent

    功能:
    1. ReAct循环 (Thought → Action → Observation)
    2. 记忆系统 (短期 + 长期)
    3. 反思机制 (自我改进)
    4. 工具调用
    5. 流式输出支持
    """

    def __init__(
        self,
        llm_retry_policy: Optional[RetryPolicy] = None,
        circuit_threshold: int = 5,
        guardrails: Optional[Any] = None,
        memory_manager: Optional[Any] = None,
    # ★ 新增：上下文压缩
        context_max_tokens: int = 8000,
        context_keep_recent: int = 4,
        enable_context_compression: bool = True,
        # ★ 新增：错误恢复
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
        """
        初始化Agent

        Args:
            name: Agent名称
            llm: LLM实例
            tools: 工具列表
            memory_store: 记忆存储
            max_iterations: 最大迭代次数
            enable_reflection: 是否启用反思
            verbose: 是否输出日志
        """
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
        self._reflector: Optional[Reflector] = None
        self.tracer = tracer or Tracer()
        
        self.llm_retry_policy = llm_retry_policy  or RetryPolicy()
        self._circuit = CircuitBreaker(threshold=circuit_threshold)
        self.guardrails = guardrails  

        self.enable_context_compression = enable_context_compression
        self._context_manager = ContextManager(
            max_tokens=context_max_tokens,
            keep_recent=context_keep_recent,
            llm=self._make_summary_llm(llm),  # ★ 带重试的包装
        )
        self.memory_manager = memory_manager

        self.error_recovery = ErrorRecovery(
            fallback_models=kwargs.get("fallback_models", []),
            fallback_tools=kwargs.get("fallback_tools", {}),
            on_degrade=lambda partial: self._partial_answer(),
       )

    # 压缩统计
        self._compression_stats = {
           "total_compressions": 0,
           "total_tokens_saved": 0,
        }

        # 反思相关开关
        self.high_risk_tools = kwargs.get(
            "high_risk_tools",
            {"transfer", "delete", "send_email", "execute_sql"},
        )
        self.always_reflect = kwargs.get("always_reflect", False)

        # 系统提示词
        self.system_prompt = self._build_system_prompt()


    def _make_summary_llm(self, llm):
        """
      包装 LLM,让摘要调用也走重试策略。
      如果 llm 是 None,返回 None（ContextManager 会 fallback）。
       """
        if llm is None:
            return None

        class _SummaryLLM:
            def __init__(self, inner, agent):
               self._inner = inner
               self._agent = agent
            def chat(self, messages, tools=None, tool_choice=None, **kwargs):
            # 摘要不传 tools
                def _do():
                    return self._inner.chat(
                        messages=messages,
                        tools=None,
                        tool_choice=None,
                    )
                try:
                    return retry_call(_do, self._agent.llm_retry_policy)
                except Exception:
                    logger.exception("summary LLM failed after retries")
                # 抛出去让 ContextManager 走 fallback
                    raise

        return _SummaryLLM(llm, self)


    # ─────────────────────────────────────────────
    # 工具索引 / 提示词
    # ─────────────────────────────────────────────
    def _index_tools(self, tools: List[Any]) -> Dict[str, Any]:
        """索引工具"""
        indexed = {}
        for tool in tools:
            name = getattr(tool, "name", tool.__class__.__name__)
            indexed[name] = tool
        return indexed

    def _build_system_prompt(self) -> str:
        """构建系统提示词"""
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
        """格式化工具列表"""
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
        """格式化历史步骤"""
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
                obs = (
                    step.observation[:200] + "..."
                    if len(step.observation) > 200
                    else step.observation
                )
                lines.append(f"  Observation: {obs}")
            lines.append("")

        return "\n".join(lines)

    def _get_memory_context(self, query: str, current_thought: str = "") -> str:
        """获取记忆上下文"""
      
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
                  f"win_rate={s['win_rate']:.2f} "
                  f"avg_pnl={s['avg_pnl']:.4f} "
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
        """Layer 1: LLM 调用带重试"""
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
            ) #装饰包装模式 代码逻辑分离 不改原代码、动态增强功能、高度复用、灵活组合、易于维护      横切关注点分离

        return retry_call(_do_call, self.llm_retry_policy, on_retry=_on_retry)
    # ─────────────────────────────────────────────
    # 工具执行
    # ─────────────────────────────────────────────
    def _execute_tool(self, name: str, inputs: Dict) -> str:
        # ★ Layer 0: guardrails
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

                return (
                    f"Error: blocked by guardrail "
                    f"[{verdict.risk_level.value}]: {verdict.reason}"
                )
            # 允许改写参数（如路径规范化）
            if verdict.modified_args is not None:
                inputs = verdict.modified_args

        # 原有逻辑
        tool = self.tools.get(name)
        if not tool:
            return f"Error: Tool '{name}' not found"
        try:
            handler = getattr(tool, 'handler', None)
            if handler is None:
                handler = tool
            result = handler(**inputs)
            return str(result)
        except Exception as e:
            logger.exception(f"Tool {name} failed")
            return f"Error: {type(e).__name__}: {str(e)}"
    # ─────────────────────────────────────────────
    # 主入口
    # ─────────────────────────────────────────────
    def run(
        self,
        query: str,
        config: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
        stream: bool = False,
    ) -> AgentResult:
        """
        执行任务

        Args:
            query: 任务描述
            config: 配置覆盖
            session_id: 会话ID
            stream: 是否流式输出

        Returns:
            AgentResult
        """
        try:
            # ★ root span：包住整个 run
            with self.tracer.trace("agent.run", **{
                "run.query": query[:200],
                "agent.name": self.name,
                "agent.max_iterations": self.max_iterations,
            }) as root:

                # ★ trace: 开始
                self.tracer.emit(
                    "run_start",
                    step_index=-1,
                    query=query[:200],
                    session_id=session_id,
                    agent_name=self.name,
                    max_iterations=self.max_iterations,
                )

                self.state = AgentState.THINKING
                self.steps = []
                self.context = {
                    "query": query,
                    "config": config or {},
                    "session_id": session_id,
                    "start_time": datetime.now(),
                }

                if self.verbose:
                    logger.info(f"🚀 Agent {self.name} starting task: {query[:100]}...")

                # ── 1. 构造初始 messages（优先从 session 加载）──
                if session_id:
                    messages: List[Dict[str, Any]] = self._load_session_messages(session_id)
                else:
                    messages = []

                if not messages or messages[0].get("role") != "system":
                    messages.insert(0, {"role": "system", "content": self.system_prompt})

                # 注入记忆
                memory_context = self._get_memory_context(query)
                user_content = query
                if memory_context:
                    user_content = f"{memory_context}\n\n## Task\n{query}"
                messages.append({"role": "user", "content": user_content})

                # ── 2. 工具 schema ──
                tool_schemas = [tool.to_schema() for tool in self.tools.values()]

                # ── 3. 主循环 ──
                for i in range(self.max_iterations):
                    if self.verbose:
                        logger.info(f"🔄 Iteration {i+1}/{self.max_iterations}")
                    self.state = AgentState.THINKING

                    # ══════════════════════════════════════════════
                    # ★ 上下文压缩
                    # ══════════════════════════════════════════════
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
                                    saved, comp.dropped_messages,
                                    comp.truncated_tool_results,
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
                            logger.exception(
                                "context compression failed; continuing with full context"
                            )
                    # ── LLM 调用埋点 ──
                    t0 = time.time()
                    self.tracer.emit(
                        "llm_call",
                        step_index=i,
                        model=getattr(self.llm, "model", "unknown"),
                        n_messages=len(messages),
                        n_tools=len(tool_schemas),
                    )
                    # 带重试llm调用
        
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
                           continue   # 重试本轮

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

                             self.tracer.emit(
                                "run_end",
                                step_index=i,
                                success=False,
                                iterations=i + 1,
                                error=f"llm_error: {e}",
                              )
                             return result
                   # 先检查"熔断器"是否处于打开（open）状态。
                   # 如果打开，说明系统已经判定当前服务/依赖不可用，直接快速失败返回，不再继续执行
                        # 熔断检查
                         if self._circuit.is_open():
                             result =  AgentResult(
                                   success=False,
                                   answer=self._partial_answer(),
                                   steps=self.steps,
                                   iterations=i + 1,
                                   state=AgentState.ERROR,
                                   error=f"circuit_breaker_open: {e}",
                                   metadata={"circuit": self._circuit.snapshot()},
                               )
                             self._finalize(result, query, session_id)
                             self.tracer.emit(
                                   "run_end",
                                   step_index=i,
                                   success=False,
                                   iterations=i + 1,
                                   error=f"circuit_breaker_open: {e}",
                                  )
                             return result
                         result = AgentResult(
                                    success=False, answer=f"LLM error: {e}",
                                    steps=self.steps, iterations=i + 1,
                                    state=AgentState.ERROR, error=str(e),
                                    )
                         self._finalize(result, query, session_id)
                         self.tracer.emit(
                                 "run_end",
                                  step_index=i,
                                  success=False,
                                  iterations=i + 1,
                                  error=f"llm_error: {e}",
                                 )
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
                    tool_calls = resp.get("tool_calls", [])

                    if self.verbose and content:
                        logger.info(f"💭 Thought: {content[:100]}...")

                    # ── 4. 无工具调用 → 完成 ──
                    if not tool_calls:
                        self.state = AgentState.DONE
                        messages.append({"role": "assistant", "content": content or ""})

                        step = AgentStep(
                            thought=content or "",
                            observation=content or "",
                        )
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
                            } )

                        # ★ trace: 完成
                        self.tracer.emit(
                                 "run_end",
                                step_index=i,
                                 success=True,
                               iterations=i + 1,
                               reason="completed",
                           )

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
                                    "arguments": json.dumps(
                                        tc["arguments"], ensure_ascii=False
                                    ),
                                },
                            }
                            for tc in tool_calls
                        ],
                    })

                    # 逐个执行 tool call
                    for tc in tool_calls:
                        action_name = tc["name"]
                        action_input = tc["arguments"]
                        call_id = tc["id"]

                        t0 = time.time()
                        self.tracer.emit(
                            "tool_call",
                            step_index=i,
                            tool=action_name,
                            arguments=action_input,
                            call_id=call_id,
                        )
                        if self.verbose:
                            logger.info(
                                f"🔧 Action: {action_name} "
                                f"{json.dumps(action_input, ensure_ascii=False)[:200]}"
                            )

                        observation = self._execute_tool(action_name, action_input) 
                           #是最原始的、工具直接返回的内容
                        self.state = AgentState.OBSERVING
                        tool_ms = (time.time() - t0) * 1000
                        success = not str(observation).startswith("Error:")

                        # ★ Layer 2：失败时反馈给 circuit，并记录提示
                        if success:
                            self._circuit.record_success()
                        else:
                            self._circuit.record_failure()
                            observation = (
                                f"{observation}\n\n"
                                f"[System] This is consecutive failure #{self._circuit.failure_count}. "
                                f"Consider a different approach or tool."
                              )

                         # 把 tool 结果加进 messages  并且注意把"记录"动作提前到熔断判断之前
                        step = AgentStep(
                              thought=content or "",
                              action=action_name,
                              action_input=action_input,
                              observation=str(observation),
                              success=success,
                              error=str(observation) if not success else None,)
                        self.steps.append(step)
                        
                        messages.append({
                             "role": "tool",
                             "tool_call_id": call_id,
                             "content": str(observation)   })
                        
                        
                         # ★ tool_result 事件
                        self.tracer.emit(
                                "tool_result",
                                step_index=i,
                                duration_ms=tool_ms,
                                tool=action_name,
                                call_id=call_id,
                                success=success,
                                result_preview=str(observation)[:200],
                            )
                        if not success:
                            self.tracer.emit(
                                "error",
                                step_index=i,
                                phase="tool_call",
                                tool=action_name,
                                call_id=call_id,
                                error=str(observation)[:500])
                        

                        # ★ Layer 3：熔断检查
                        if self._circuit.is_open():
                            logger.error(
                               "circuit breaker open after %d consecutive failures",
                               self._circuit.failure_count,
                            )
                          
                            result= AgentResult(
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
                            obs_preview = str(observation)[:100]
                            logger.info(f"👁️ Observation: {obs_preview}")

                
                        # 在线反思（不确定性驱动）
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

                    # 每轮工具执行完，落一次 session
                    if session_id:
                        self._save_session_messages(session_id, messages)

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
                    metadata={ "compression_stats": dict(self._compression_stats)},
                )
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


    def _finalize(self, result: AgentResult, query: str, session_id: Optional[str]):
        """任务结束时，写回记忆"""
        if not self.memory_manager:
            return

    # Layer 2: 不在这里写，由 Orchestrator 在平仓时写
    # Layer 3 (semantic): 写"任务级摘要"
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

    # Layer 3 (semantic): 写反思
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
    # 反思相关
    # ─────────────────────────────────────────────
    def _should_reflect(self, context: Dict[str, Any]) -> bool:
        """
        判断当前是否值得反思（不确定性驱动，而不是看步数）。
        """
        if not getattr(self, "enable_reflection", True):
            return False

        # 1. 失败 → 必反思
        if not context.get("success", True):
            return True

        tool_name = context.get("tool_name")
        history = context.get("history") or []

        # 2. 同一工具重复调用 → 可能在试错
        if tool_name:
            same_tool_calls = [
                s for s in history
                if isinstance(s, dict) and s.get("tool") == tool_name
            ]
            if len(same_tool_calls) >= 2:
                return True

        # 3. 策略切换：上一步失败，且这一步换了工具
        if tool_name and history:
            last = history[-1] if isinstance(history[-1], dict) else None
            if last and last.get("tool") != tool_name and not last.get("success", True):
                return True

        # 4. 高风险操作
        if tool_name and tool_name in getattr(self, "high_risk_tools", set()):
            return True

        return False

    def _reflect_on_step(
        self,
        query: str,
        tool_name: str,
        tool_args: Any,
        tool_result: Any,
        error: Optional[str],
        history: List[Any],
    ) -> str:
        """
        针对单步失败的反思，产出"下一步怎么改"的可执行策略。
        失败不影响主流程，返回空字符串。
        """
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
                tools=None,
                tool_choice=None,
                temperature=0.2,
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

    # ─────────────────────────────────────────────
    # session 存储
    # ─────────────────────────────────────────────
    def _load_session_messages(self, session_id: str) -> List[Dict[str, Any]]:
        """从存储加载某 session 的历史 messages（副本）"""
        if not hasattr(self, "_session_store"):
            self._session_store: Dict[str, List[Dict[str, Any]]] = {}
        return list(self._session_store.get(session_id, []))

    def _save_session_messages(
        self, session_id: str, messages: List[Dict[str, Any]]
    ) -> None:
        """把 messages 写回存储"""
        if not hasattr(self, "_session_store"):
            self._session_store = {}
        self._session_store[session_id] = list(messages)

    def _store_successful_step(self, step: AgentStep, query: str):
        """存储成功的步骤到记忆"""
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
    # 其他工具方法
    # ─────────────────────────────────────────────
    def get_response_content(self, response) -> str:
        """获取响应内容"""
        if isinstance(response, AgentResult):
            return response.answer
        if hasattr(response, "answer"):
            return response.answer
        return str(response)

    def add_tool(self, name: str, tool: Any):
        """添加工具"""
        self.tools[name] = tool

    def reset(self):
        """重置Agent"""
        self.state = AgentState.IDLE
        self.steps = []
        self.context = {}
        self._compression_stats = {
            "total_compressions": 0,
            "total_tokens_saved": 0,
        }

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "name": self.name,
            "state": self.state.value,
            "total_steps": len(self.steps),
            "tools": list(self.tools.keys()),
            "max_iterations": self.max_iterations,
            "enable_reflection": self.enable_reflection,
        }

    def _partial_answer(self) -> str:
        """熔断时返回部分答案：取最后一个非空 observation 或 thought"""
        for step in reversed(self.steps):
            if step.observation and not step.observation.startswith("Error:"):
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
    """创建Agent"""
    return CoreAgent(
        name=name,
        llm=llm,
        tools=tools,
        **kwargs,
    )
