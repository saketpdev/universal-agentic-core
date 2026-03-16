import json
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
# Note: For production, swap ConsoleSpanExporter with OTLPSpanExporter (for Datadog/Jaeger/New Relic)

from core.infrastructure import redis_client
from models.telemetry import (
    DecisionEvent, ActionEvent, StateEvent, MetricEvent, ActionStatus
)

# Initialize OpenTelemetry Provider (Run once per application lifecycle)
if not trace._TRACER_PROVIDER:
    provider = TracerProvider()
    # Using ConsoleSpanExporter for local dev. This outputs standard OTel structured data.
    processor = BatchSpanProcessor(ConsoleSpanExporter())
    provider.add_span_processor(processor)
    trace.set_tracer_provider(provider)

tracer = trace.get_tracer("agentic.core.telemetry")

class TelemetryLogger:
    def __init__(self, trace_id: str):
        self.trace_id = trace_id

    def _get_sequence_id(self) -> int:
        try:
            return redis_client.incr("telemetry:global_sequence_id")
        except Exception:
            return 0

    def _record_otel_event(self, span_name: str, event_model, attributes: dict):
        """Creates a distinct OTel Span and attaches the Pydantic JSON as an event."""
        # We link the OTel trace to our internal thread_id for cross-system querying
        otel_attributes = {"app.thread_id": self.trace_id, **attributes}
        
        with tracer.start_as_current_span(span_name, attributes=otel_attributes) as span:
            span.add_event(
                name=event_model.event_type.value,
                attributes={"payload": event_model.model_dump_json()}
            )

    def log_decision(self, agent_id: str, reasoning: str, context: str = None):
        event = DecisionEvent(
            trace_id=self.trace_id,
            sequence_id=self._get_sequence_id(),
            agent_id=agent_id,
            reasoning=reasoning,
            context=context
        )
        self._record_otel_event(f"{agent_id}.decision", event, {"agent.id": agent_id})

    def log_action(self, agent_id: str, correlation_id: str, tool_name: str, 
                   arguments: str, status: ActionStatus, latency_ms: float = None, 
                   result_summary: str = None):
        event = ActionEvent(
            trace_id=self.trace_id,
            sequence_id=self._get_sequence_id(),
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

    def log_state(self, agent_id: str, step_index: int, domain_update: dict):
        event = StateEvent(
            trace_id=self.trace_id,
            sequence_id=self._get_sequence_id(),
            agent_id=agent_id,
            current_step_index=step_index,
            domain_update=domain_update
        )
        self._record_otel_event(f"state.checkpoint", event, {"step.index": step_index})

    def log_metric(self, agent_id: str, tier: str, prompt_tokens: int, completion_tokens: int):
        total = prompt_tokens + completion_tokens
        event = MetricEvent(
            trace_id=self.trace_id,
            sequence_id=self._get_sequence_id(),
            agent_id=agent_id,
            model_tier=tier,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total
        )
        self._record_otel_event(f"{agent_id}.llm_metrics", event, {
            "llm.tier": tier,
            "llm.usage.total_tokens": total
        })