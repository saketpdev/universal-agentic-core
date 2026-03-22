from pydantic import BaseModel, Field
from typing import Optional, Any, Dict
from datetime import datetime, timezone
from enum import Enum

class EventType(str, Enum):
    DECISION = "decision"
    ACTION = "action"
    STATE = "state"
    METRIC = "metric"

class ActionStatus(str, Enum):
    PENDING = "pending"
    SUCCESS = "success"
    FAILED = "failed"
    YIELDED = "yielded" # status for dynamic handoffs

class BaseEvent(BaseModel):
    # Enforces ISO 8601 UTC timestamps
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    trace_id: str
    sequence_id: int  # The globally interlocked integer
    agent_id: str
    event_type: EventType

class DecisionEvent(BaseEvent):
    event_type: EventType = EventType.DECISION
    reasoning: str
    context: Optional[str] = None  # E.g., "Judge Critique" or "Planner Routing"

class ActionEvent(BaseEvent):
    event_type: EventType = EventType.ACTION
    action_correlation_id: str     # Links the PENDING request to the SUCCESS/FAILED result
    tool_name: str
    arguments: str
    status: ActionStatus
    latency_ms: Optional[float] = None
    result_summary: Optional[str] = None

class StateEvent(BaseEvent):
    event_type: EventType = EventType.STATE
    stage_index: int
    domain_update: Dict[str, Any]

class MetricEvent(BaseEvent):
    event_type: EventType = EventType.METRIC
    llm_tier: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: float = 0.0