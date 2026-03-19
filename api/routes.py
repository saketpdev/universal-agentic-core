from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from models.state import AgentRequest, SharedBriefcase
from core.memory import session_manager
from core.infrastructure import task_queue
from core.telemetry import TelemetryLogger
from models.telemetry import ActionStatus

router = APIRouter()

# New response model since we no longer wait for the LLM to finish
class AsyncAcceptedResponse(BaseModel):
    status: str
    thread_id: str
    message: str

@router.post("/execute", response_model=AsyncAcceptedResponse)
def execute_agent(request: AgentRequest):
    # 1. Start the OpenTelemetry Trace right at the Gateway!
    telemetry = TelemetryLogger(trace_id=request.thread_id)
    
    try:
        telemetry.log_decision(
            agent_id="api_gateway",
            reasoning=f"Received asynchronous workflow request from user {request.user_id}",
            context="API Ingestion"
        )

        # 2. Check if resuming, or starting fresh
        briefcase = session_manager.get_briefcase(request.thread_id)
        if not briefcase:
            # Ground Zero Initialization
            briefcase = SharedBriefcase(
                thread_id=request.thread_id, 
                original_user_prompt=request.user_prompt
            )
            # Save the initial empty state to SQLite State Machine
            session_manager.save_briefcase(request.thread_id, request.user_id, briefcase, status="QUEUED")
        
        # 3. Drop it in the Redis Queue
        task_queue.enqueue(request.thread_id)
        
        # 4. Log the successful handoff
        telemetry.log_action(
            agent_id="api_gateway",
            correlation_id=request.thread_id,
            tool_name="redis_enqueue",
            arguments=f'{{"queue": "agentic:task_queue"}}',
            status=ActionStatus.SUCCESS
        )

        # 5. Instantly return to the client (Zero LLM Latency)
        return AsyncAcceptedResponse(
            status="accepted",
            thread_id=request.thread_id,
            message="Workflow queued successfully. Poll the database for updates."
        )
        
    except Exception as e:
        telemetry.log_decision(
            agent_id="api_gateway",
            reasoning=f"Failed to enqueue task: {str(e)}",
            context="API Crash"
        )
        raise HTTPException(status_code=500, detail=str(e))