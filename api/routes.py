from fastapi import APIRouter, HTTPException
from models.state import AgentRequest, AgentResponse
from core.engine import run_agentic_loop

router = APIRouter()

@router.post("/execute", response_model=AgentResponse)
def execute_agent(request: AgentRequest):
    try:
        return run_agentic_loop(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))