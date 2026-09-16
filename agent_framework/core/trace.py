# core/trace.py
"""
Trace - 基于 OpenTelemetry 的结构化执行追踪（写法 C 版本）

设计要点:
1. 一次 run = 一棵 trace 树：root span 由 `Tracer.trace()` 上下文管理器创建/结束
2. `emit` / `span` 内部用 `start_as_current_span`，自动认"当前 span"为父
3. 保留原有 JSON events 导出，向后兼容
4. GenAI semantic conventions 自动映射

用法:
    tracer = Tracer(service_name="test-agent")

    with tracer.trace("agent.run", **{"run.query": query}) as root:
        tracer.emit("llm_call", step_index=0, model="deepseek-chat")
        with tracer.span("tool_call", step_id="0") as s:
            s.payload["tool"] = "calculate"
        tracer.emit("run_end", success=True)
"""
from __future__ import annotations

import json
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from opentelemetry import trace as otel_trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
    SimpleSpanProcessor,
)
from opentelemetry.sdk.resources import Resource
from opentelemetry.trace import Status, StatusCode

__all__ = ["TraceEvent", "Tracer", "init_tracer_provider"]


# ══════════════════════════════════════════════════════════
# Provider 初始化（全局一次）
# ══════════════════════════════════════════════════════════
_provider_initialized = False


def init_tracer_provider(
    service_name: str = "agent-harness",
    exporter: str = "console",       # "console" | "otlp" | "none"
    otlp_endpoint: Optional[str] = None,
):
    global _provider_initialized
    if _provider_initialized:
        return otel_trace.get_tracer_provider()

    resource = Resource.create({
        "service.name": service_name,
        "service.version": "0.1.0",
    })
    provider = TracerProvider(resource=resource)

    if exporter == "console":
        provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
    elif exporter == "otlp":
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )
        endpoint = otlp_endpoint or "http://localhost:4318/v1/traces"
        provider.add_span_processor(
            BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint))
        )

    otel_trace.set_tracer_provider(provider)
    _provider_initialized = True
    return provider


# ══════════════════════════════════════════════════════════
# 向后兼容：TraceEvent
# ══════════════════════════════════════════════════════════
@dataclass
class TraceEvent:
    trace_id: str
    step_id: str
    event_type: str
    timestamp: float
    payload: Dict[str, Any] = field(default_factory=dict)
    duration_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ══════════════════════════════════════════════════════════
# Tracer（写法 C）
# ══════════════════════════════════════════════════════════
class Tracer:
    """
    一次 run 一棵 trace 树。

    - `trace(name, **attrs)` 是根上下文管理器，进入时创建 root span，
      退出时自动 end()，内部所有 `emit` / `span` 自动挂到 root 下。
    - `emit()` 是"一次性 span"，用完即 end，用于记录事件点。
    - `span()` 是"代码块 span"，包裹一段逻辑，自动处理异常/结束。
    """

    def __init__(
        self,
        trace_id: Optional[str] = None,
        sink: Optional[Callable[[TraceEvent], None]] = None,
        verbose: bool = False,
        service_name: str = "agent-harness",
        exporter: str = "console",
        otlp_endpoint: Optional[str] = None,
        enable_otel: bool = True,
        print_spans: bool = True,
    ):
        self.trace_id = trace_id or uuid.uuid4().hex
        self.sink = sink
        self.verbose = verbose
        self.print_spans = print_spans
        self.enable_otel = enable_otel
        self.events: List[TraceEvent] = []

        if enable_otel:
            init_tracer_provider(
                service_name=service_name,
                exporter=exporter,
                otlp_endpoint=otlp_endpoint,
            )
            self._otel_tracer = otel_trace.get_tracer(__name__)
        else:
            self._otel_tracer = None

        # 当前活跃的 root span（进入 trace() 时被赋值，退出时清空）
        self._root_span = None

    # ─────────────────────────────────────────────
    # 根上下文管理器：一次 run 一棵 trace
    # ─────────────────────────────────────────────
    @contextmanager
    def trace(self, name: str = "agent.run", **attrs):
        """
        用法:
            with tracer.trace("agent.run", **{"run.query": q}) as root:
                tracer.emit("llm_call", ...)
        """
        if self._otel_tracer is None:
            yield None
            return

        with self._otel_tracer.start_as_current_span(name) as root:
            root.set_attribute("agent.trace_id", self.trace_id)
            for k, v in attrs.items():
                self._safe_set_attr(root, k, v)

            self._root_span = root
            try:
                yield root
            except Exception as e:
                root.set_status(Status(StatusCode.ERROR, str(e)))
                root.record_exception(e)
                raise
            else:
                # 正常结束时，默认设 OK；如果内部已经设过 ERROR，不覆盖
                if root.status.status_code == StatusCode.UNSET:
                    root.set_status(Status(StatusCode.OK))
            finally:
                self._root_span = None

    # ─────────────────────────────────────────────
    # emit：一次性 span（用完即 end）
    # ─────────────────────────────────────────────
    def emit(self, event_type: str, step_index: int = -1, **payload):
        # 1) 保留原有 JSON events
        self._append_event(event_type, step_id=str(step_index), payload=payload, duration_ms=0.0)

        if self._otel_tracer is None:
            return

        attrs = self._build_attributes(event_type, step_index, payload)

        # 用 start_as_current_span：当前 context 里是谁（root 或父 span）
        # 就是它的爹
        with self._otel_tracer.start_as_current_span(event_type) as span:
            for k, v in attrs.items():
                self._safe_set_attr(span, k, v)

            if payload.get("success") is True:
                span.set_status(Status(StatusCode.OK))
            elif payload.get("success") is False:
                span.set_status(Status(StatusCode.ERROR, str(payload.get("error", ""))))

            if self.print_spans:
                self._print_span_json(span, event_type, attrs)

    # ─────────────────────────────────────────────
    # span：代码块 span
    # ─────────────────────────────────────────────
    @contextmanager
    def span(self, event_type: str, step_id: str, **initial_payload):
        """
        用法:
            with tracer.span("tool_call", step_id="0", tool="calculate") as s:
                result = tool(...)
                s.payload["result_preview"] = str(result)[:200]
        """
        t0 = time.time()
        span_obj = _Span(self, event_type, step_id, initial_payload)

        if self._otel_tracer is None:
            try:
                yield span_obj
            except Exception as e:
                span_obj.payload["exception"] = f"{type(e).__name__}: {e}"
                self._append_event(event_type, step_id, span_obj.payload, (time.time() - t0) * 1000)
                raise
            else:
                self._append_event(event_type, step_id, span_obj.payload, (time.time() - t0) * 1000)
            return

        with self._otel_tracer.start_as_current_span(event_type) as otel_span:
            otel_span.set_attribute("step.id", step_id)
            try:
                yield span_obj
            except Exception as e:
                span_obj.payload["exception"] = f"{type(e).__name__}: {e}"
                otel_span.set_status(Status(StatusCode.ERROR, str(e)))
                otel_span.record_exception(e)
                self._append_event(event_type, step_id, span_obj.payload, (time.time() - t0) * 1000)
                raise
            else:
                duration_ms = (time.time() - t0) * 1000
                for k, v in span_obj.payload.items():
                    self._safe_set_attr(otel_span, f"payload.{k}", v)
                if otel_span.status.status_code == StatusCode.UNSET:
                    otel_span.set_status(StatusCode.OK)
                self._append_event(event_type, step_id, span_obj.payload, duration_ms)

    # ─────────────────────────────────────────────
    # 属性映射
    # ─────────────────────────────────────────────
    def _build_attributes(self, event_type: str, step_index: int, payload: Dict) -> Dict:
        attrs: Dict[str, Any] = {}
        if step_index != -1:
            attrs["step.index"] = step_index

        if event_type == "llm_call":
            attrs.update({
                "gen_ai.operation.name": "chat",
                "gen_ai.system": "openai",
                "gen_ai.request.model": payload.get("model", "unknown"),
                "agent.n_messages": payload.get("n_messages", 0),
                "agent.n_tools": payload.get("n_tools", 0),
            })
        elif event_type == "tool_call":
            attrs.update({
                "gen_ai.operation.name": "execute_tool",
                "gen_ai.tool.name": payload.get("tool", "unknown"),
                "gen_ai.tool.call.id": payload.get("call_id", ""),
                "gen_ai.tool.arguments": json.dumps(
                    payload.get("arguments", {}), ensure_ascii=False
                )[:1000],
            })
        return {k: v for k, v in attrs.items() if v is not None}

    def _safe_set_attr(self, span, key: str, value: Any) -> None:
        """OTel 只接受基础类型，dict/list 要序列化；None 转空串"""
        if value is None:
            value = ""
        elif isinstance(value, (dict, list, tuple)):
            try:
                value = json.dumps(value, ensure_ascii=False, default=str)
            except Exception:
                value = str(value)
        try:
            span.set_attribute(key, value)
        except Exception:
            pass

    def _print_span_json(self, span, name: str, attrs: Dict[str, Any]) -> None:
        sc = span.get_span_context()
        parent_id = None
        if span.parent is not None:
            parent_id = f"0x{span.parent.span_id:016x}"
        print(json.dumps({
            "name": name,
            "context": {
                "trace_id": f"0x{sc.trace_id:032x}",
                "span_id":  f"0x{sc.span_id:016x}",
                "trace_state": "[]",
            },
            "kind": "SpanKind.INTERNAL",
            "parent_id": parent_id,
            "attributes": attrs,
            "status": {
                "status_code": span.status.status_code.name,
                "description": span.status.description or "",
            },
        }, indent=4, ensure_ascii=False))

    # ─────────────────────────────────────────────
    # JSON events（向后兼容）
    # ─────────────────────────────────────────────
    def _append_event(self, event_type: str, step_id: str, payload: Dict, duration_ms: float):
        evt = TraceEvent(
            trace_id=self.trace_id,
            step_id=step_id,
            event_type=event_type,
            timestamp=time.time(),
            payload=payload,
            duration_ms=duration_ms,
        )
        self.events.append(evt)
        if self.sink is not None:
            try:
                self.sink(evt)
            except Exception:
                pass

    def export_json(self, path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "trace_id": self.trace_id,
                    "event_count": len(self.events),
                    "events": [e.to_dict() for e in self.events],
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

    def summary(self) -> Dict[str, Any]:
        by_type: Dict[str, int] = {}
        total_ms = 0.0
        for e in self.events:
            by_type[e.event_type] = by_type.get(e.event_type, 0) + 1
            total_ms += e.duration_ms
        return {
            "trace_id": self.trace_id,
            "total_events": len(self.events),
            "by_type": by_type,
            "total_duration_ms": round(total_ms, 2),
        }

    def shutdown(self) -> None:
        """flush OTel provider（root 的 end 由 trace() 负责，这里不再 end）"""
        try:
            provider = otel_trace.get_tracer_provider()
            if hasattr(provider, "shutdown"):
                provider.shutdown()
        except Exception:
            pass


class _Span:
    """span 上下文管理器里 yield 出去的对象"""
    def __init__(self, tracer: Tracer, event_type: str, step_id: str, initial: Dict):
        self._tracer = tracer
        self.event_type = event_type
        self.step_id = step_id
        self.payload = dict(initial)