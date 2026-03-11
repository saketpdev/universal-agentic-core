import uuid
from typing import List, Literal, Optional
from pydantic import BaseModel, Field

class UserProfile(BaseModel):
    user_id: str
    tier: Literal["free", "premium"] = "free"
    hard_constraints: List[str] = Field(default_factory=list, description="Immutable rules the agent MUST follow.")

class ActiveTaskState(BaseModel):
    objective: str
    milestones_completed: List[str] = Field(default_factory=list)
    pending_requirements: List[str] = Field(default_factory=list)

class AgentRequest(BaseModel):
    user_prompt: str
    system_prompt: str = "You are a deterministic backend agent."
    user_id: str

class AgentResponse(BaseModel):
    status: str
    trace_id: str
    output: str
    iterations: int

class LoopState(BaseModel):
    iteration: int = 0
    max_iters: int = 5
    budget_lock_id: str
    trace_id: str = Field(default_factory=lambda: str(uuid.uuid4()))

# --- MOCKED STATE DB (Replace with actual database later) ---
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

# --- PILLAR 3: EVALUATOR-OPTIMIZER SCHEMAS ---
class EvaluationRubric(BaseModel):
    policy_compliance: float = Field(description="Score from 1.0 to 5.0 indicating adherence to company policy.")
    math_accuracy: bool = Field(description="True if all calculations are strictly correct.")
    critique: str = Field(description="Specific feedback for the generator to improve on the next iteration.")