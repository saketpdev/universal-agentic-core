import asyncio
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from models.state import AgentRequest, SharedBriefcase
from core.memory import session_manager
from core.infrastructure import task_queue
from core.telemetry import TelemetryLogger
from models.telemetry import ActionStatus

router = APIRouter()

class AsyncAcceptedResponse(BaseModel):
    status: str
    thread_id: str
    message: str
    execution_type: str

# ==========================================
# ROUTE 1: INDEFINITE TASKS (The Swarm)
# ==========================================
@router.post("/executeTask", response_model=AsyncAcceptedResponse)
async def execute_ad_hoc_task(request: AgentRequest):
    """
    Handles natural language requests.
    Relies on the LLM Planner to dynamically generate the DAG.
    """
    telemetry = TelemetryLogger(trace_id=request.thread_id)

    try:
        # 🚀 AWAIT the async telemetry
        await telemetry.log_decision(
            agent_id="api_gateway",
            reasoning=f"Received unstructured task from {request.user_id}",
            context="Ad-Hoc Ingestion"
        )

        # 🚀 NON-BLOCKING SQLITE FETCH
        briefcase = await asyncio.to_thread(session_manager.get_briefcase, request.thread_id)
        if not briefcase:
            briefcase = SharedBriefcase(
                thread_id=request.thread_id,
                original_user_prompt=request.user_prompt
            )
            # 🚀 NON-BLOCKING SQLITE WRITE
            await asyncio.to_thread(session_manager.save_briefcase, request.thread_id, request.user_id, briefcase, "QUEUED")

        # 🚀 AWAIT the async Redis Queue
        await task_queue.enqueue(request.thread_id)

        return AsyncAcceptedResponse(
            status="accepted",
            thread_id=request.thread_id,
            message="Ad-Hoc Task queued. The Master Planner is decomposing the request.",
            execution_type="GENERATIVE_DAG"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==========================================
# ROUTE 2: DEFINITE TASKS (SOP Workflows)
# ==========================================
@router.post("/executeWorkflow", response_model=AsyncAcceptedResponse)
async def execute_strict_workflow(request: AgentRequest):
    """
    Handles predefined Standard Operating Procedures.
    Bypasses the LLM Planner completely and loads the requested YAML graph.
    """
    # 🚀 FIXED: Checking for the correct workflow_name variable!
    if not request.workflow_name:
        raise HTTPException(status_code=400, detail="workflow_name is required for this endpoint.")

    telemetry = TelemetryLogger(trace_id=request.thread_id)
    try:
        # 🚀 AWAIT the async telemetry
        await telemetry.log_decision(
            agent_id="api_gateway",
            reasoning=f"Received strict workflow [{request.workflow_name}] from {request.user_id}",
            context="SOP Ingestion"
        )

        # 🚀 NON-BLOCKING SQLITE FETCH
        briefcase = await asyncio.to_thread(session_manager.get_briefcase, request.thread_id)
        if not briefcase:
            briefcase = SharedBriefcase(
                thread_id=request.thread_id,
                original_user_prompt=request.user_prompt
            )
            # We save the requested workflow_name into the briefcase so the Runner knows to load the YAML
            await asyncio.to_thread(session_manager.save_briefcase, request.thread_id, request.user_id, briefcase, "QUEUED")

        await task_queue.enqueue(request.thread_id)

        return AsyncAcceptedResponse(
            status="accepted",
            thread_id=request.thread_id,
            message=f"Workflow '{request.workflow_name}' queued for parallel execution.",
            execution_type="DECLARATIVE_DAG"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))