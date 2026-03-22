import logging
from models.state import SharedBriefcase, AgentRequest
from core.memory import session_manager

logger = logging.getLogger("AgenticCore.StateManager")

def initialize_or_resume_state(request: AgentRequest) -> SharedBriefcase:
    """Strictly loads existing Briefcase or creates an empty one."""

    briefcase = session_manager.get_briefcase(request.thread_id)

    if briefcase:
        logger.info(f"[{request.thread_id}] Resuming at Stage {briefcase.current_stage_index}")
        return briefcase

    logger.info(f"[{request.thread_id}] Creating fresh Briefcase for new session.")
    return SharedBriefcase(thread_id=request.thread_id, original_user_prompt=request.user_prompt)

def checkpoint_state(briefcase: SharedBriefcase, request: AgentRequest):
    """Safely commits the current Briefcase Pydantic object to SQLite."""
    session_manager.save_briefcase(request.thread_id, request.user_id, briefcase)