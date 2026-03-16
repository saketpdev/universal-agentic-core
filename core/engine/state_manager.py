import json
import logging
from models.state import SharedBriefcase, AgentRequest
from core.memory import load_history, save_history

logger = logging.getLogger("AgenticCore.StateManager")

def initialize_or_resume_state(request: AgentRequest) -> SharedBriefcase:
    """Strictly loads existing Briefcase or creates an empty one."""
    raw_history = load_history(request.thread_id)
    if raw_history:
        briefcase = SharedBriefcase(**raw_history)
        logger.info(f"[{request.thread_id}] Resuming at Step {briefcase.current_step_index}")
        return briefcase

    logger.info(f"[{request.thread_id}] Creating fresh Briefcase for new session.")
    return SharedBriefcase(thread_id=request.thread_id, original_user_prompt=request.user_prompt)

def checkpoint_state(briefcase: SharedBriefcase, request: AgentRequest):
    """Safely commits the current Briefcase JSON to SQLite."""
    save_history(request.thread_id, request.user_id, json.loads(briefcase.model_dump_json()))