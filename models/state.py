import uuid
from typing import List, Literal, Optional, Any, Dict
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum

class FailurePolicy(str, Enum):
    TERMINATE = "TERMINATE" # Critical failure: Halts the entire DAG.
    IGNORE = "IGNORE"       # Non-critical: Marks task as failed but proceeds.
    PAUSE = "PAUSE"         # Sensitive: Pauses DAG for human review ticket.

class UserProfile(BaseModel):
    user_id: str
    tier: Literal["free", "premium"] = "free"
    hard_constraints: List[str] = Field(default_factory=list, description="Immutable rules the agent MUST follow.")

class ActiveTaskState(BaseModel):
    objective: str
    milestones_completed: List[str] = Field(default_factory=list)
    pending_requirements: List[str] = Field(default_factory=list)

class LoopState(BaseModel):
    iteration: int = 0
    max_iters: int = 5
    budget_lock_id: str
    trace_id: str = Field(default_factory=lambda: str(uuid.uuid4()))

class MockStateDB:
    def load_profile(self, user_id: str) -> UserProfile:
        return UserProfile(
            user_id=user_id,
            tier="premium",
            hard_constraints=["Do not execute database writes without asking for confirmation.", "Always use concise verbosity."]
        )

    def load_task(self, user_id: str) -> ActiveTaskState:
        return ActiveTaskState(
            objective="Audit server logs for latency spikes.",
            milestones_completed=["Fetched API logs", "Identified 429 errors"],
            pending_requirements=["Query database connection pool metrics"]
        )

state_db = MockStateDB()

class EvaluationRubric(BaseModel):
    policy_compliance: float = Field(description="Score from 1.0 to 5.0 indicating adherence to company policy.")
    math_accuracy: bool = Field(description="True if all calculations are strictly correct.")
    critique: str = Field(description="Specific feedback for the generator to improve on the next iteration.")

class Task(BaseModel):
    """Represents a single executable unit of work."""
    agent_target: str = Field(description="The name of the specialized agent (e.g., 'finance_agent')")
    instruction: str = Field(description="Specific, narrow instruction for this agent.")
    status: str = Field(default="pending", description="'pending', 'in_progress', 'completed', 'failed', 'yielded'")
    result_summary: Optional[str] = Field(default=None, description="The worker's synthesized summary.")
    # Default to TERMINATE to ensure strictness unless overridden
    on_failure: FailurePolicy = Field(default=FailurePolicy.TERMINATE)

class Stage(BaseModel):
    """Represents a phase of execution containing parallel tasks."""
    stage_id: int
    stage_name: str = "Unnamed Stage"
    tasks: List[Task] = Field(default_factory=list)

class ExecutionPlan(BaseModel):
    """The full hierarchical tree representing the DAG."""
    planned_stages: List[Stage] = Field(default_factory=list)

class SharedBriefcase(BaseModel):
    """THE MASTER STATE OBJECT (100% Core Logic)."""
    thread_id: str = Field(description="The unique session ID for SQLite checkpointing.")
    original_user_prompt: str = Field(description="The immutable original request.")
    current_stage_index: int = Field(default=0)
    execution_plan: Optional[ExecutionPlan] = None
    domain_state: Dict[str, Dict[str, Any]] = Field(default_factory=dict, description="Agnostic vault for agent-specific data.")
    final_resolution: Optional[str] = Field(default=None)
    has_critical_error: bool = Field(default=False)

class AgentRequest(BaseModel):
    user_prompt: str
    system_prompt: str = "You are a deterministic backend agent."
    user_id: str
    thread_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Pass this to continue a past conversation.")
    workflow_name: Optional[str] = Field(default=None, description="The name of the YAML workflow template to load (e.g., 'financial_audit')")

class AgentResponse(BaseModel):
    status: str
    trace_id: str
    output: str
    iterations: int