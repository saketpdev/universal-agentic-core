import json
from typing import Any, cast
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

from core.infrastructure import redis_client
from models.telemetry import (
    DecisionEvent, ActionEvent, StateEvent, MetricEvent, ActionStatus
)

if not trace._TRACER_PROVIDER:
    provider = TracerProvider()
    processor = BatchSpanProcessor(ConsoleSpanExporter())
    provider.add_span_processor(processor)
    trace.set_tracer_provider(provider)

tracer = trace.get_tracer("agentic.core.telemetry")

class TelemetryLogger:
    def __init__(self, trace_id: str):
        self.trace_id = trace_id

    # 🚀 AWAIT THE REDIS INCREMENT
    async def _get_sequence_id(self) -> int:
        try:
            raw_val = await redis_client.incr("telemetry:global_sequence_id")
            return int(cast(Any, raw_val) or 0)
        except Exception:
            return 0

    def _record_otel_event(self, span_name: str, event_model, attributes: dict):
        otel_attributes = {"app.thread_id": self.trace_id, **attributes}
        with tracer.start_as_current_span(span_name, attributes=otel_attributes) as span:
            span.add_event(
                name=event_model.event_type.value,
                attributes={"payload": event_model.model_dump_json()}
            )

    async def log_decision(self, agent_id: str, reasoning: str, context: str = ''):
        seq_id = await self._get_sequence_id()
        event = DecisionEvent(
            trace_id=self.trace_id,
            sequence_id=seq_id,
            agent_id=agent_id,
            reasoning=reasoning,
            context=context
        )
        self._record_otel_event(f"{agent_id}.decision", event, {"agent.id": agent_id})

    async def log_action(self, agent_id: str, correlation_id: str, tool_name: str, 
                   arguments: str, status: ActionStatus, latency_ms: float = 0.0, 
                   result_summary: str = ''):
        seq_id = await self._get_sequence_id()
        event = ActionEvent(
            trace_id=self.trace_id,
            sequence_id=seq_id,
            agent_id=agent_id,
            action_correlation_id=correlation_id,
            tool_name=tool_name,
            arguments=arguments,
            status=status,
            latency_ms=latency_ms,
            result_summary=result_summary
        )
        self._record_otel_event(f"{agent_id}.tool_call", event, {
            "tool.name": tool_name,
            "tool.status": status.value
        })

    async def log_state(self, agent_id: str, step_index: int, domain_update: dict):
        seq_id = await self._get_sequence_id()
        event = StateEvent(
            trace_id=self.trace_id,
            sequence_id=seq_id,
            agent_id=agent_id,
            current_step_index=step_index,
            domain_update=domain_update
        )
        self._record_otel_event(f"state.checkpoint", event, {"step.index": step_index})

    async def log_metric(self, agent_id: str, tier: str, prompt_tokens: int, completion_tokens: int, cost_usd: float = 0.0):
        total = prompt_tokens + completion_tokens
        seq_id = await self._get_sequence_id()
        event = MetricEvent(
            trace_id=self.trace_id,
            sequence_id=seq_id,
            agent_id=agent_id,
            llm_tier=tier,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total,
            cost_usd=cost_usd
        )
        self._record_otel_event(f"{agent_id}.llm_metrics", event, {
            "llm.tier": tier,
            "llm.usage.total_tokens": total,
            "llm.usage.cost_usd": cost_usd
        })