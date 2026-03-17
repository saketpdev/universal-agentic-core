import json
import logging
import traceback
from models.state import AgentRequest, AgentResponse
from core.infrastructure import budget_manager, BudgetExceededException
from core.engine.state_manager import initialize_or_resume_state, checkpoint_state
from core.engine.node_executor import execute_worker_node
from core.planner import generate_execution_plan
from core.telemetry import TelemetryLogger

logger = logging.getLogger("AgenticCore.DAGRunner")

WORKFLOW_RESERVE_USD = 0.50  # We lock 50 cents per workflow run

def run_agentic_loop(request: AgentRequest) -> AgentResponse:
    telemetry = TelemetryLogger(trace_id=request.thread_id)
    
    try:
        # 1. THE PRE-FLIGHT DEPOSIT
        budget_manager.reserve_deposit(trace_id=request.thread_id, max_budget_usd=WORKFLOW_RESERVE_USD)
        
        briefcase = initialize_or_resume_state(request)

        if not briefcase.execution_plan:
            briefcase.execution_plan = generate_execution_plan(request.user_prompt)
            telemetry.log_decision("planner", f"Generated {len(briefcase.execution_plan)} SubTasks.", context="DAG Generation")
            checkpoint_state(briefcase, request)

        while briefcase.current_step_index < len(briefcase.execution_plan):
            current_task = briefcase.execution_plan[briefcase.current_step_index]
            
            if current_task.status == "completed":
                briefcase.current_step_index += 1
                continue
                
            current_task.status = "in_progress"
            checkpoint_state(briefcase, request)
            
            success, output = execute_worker_node(
                task=current_task, 
                briefcase=briefcase, 
                thread_id=request.thread_id
            )
            
            if not success:
                current_task.status = "failed"
                briefcase.has_critical_error = True
                checkpoint_state(briefcase, request)
                return AgentResponse(status="error", trace_id=request.thread_id, output=output, iterations=briefcase.current_step_index)

            briefcase.domain_state[current_task.agent_target] = {"latest_output": output}
            current_task.status = "completed"
            telemetry.log_state(current_task.agent_target, briefcase.current_step_index, briefcase.domain_state[current_task.agent_target])
            
            briefcase.current_step_index += 1
            checkpoint_state(briefcase, request)

        return AgentResponse(
            status="success",
            trace_id=request.thread_id,
            output=json.dumps(briefcase.domain_state),
            iterations=briefcase.current_step_index
        )
        
    except BudgetExceededException as be:
        logger.error(f"[{request.thread_id}] Workflow terminated due to FinOps Guardrail: {str(be)}")
        return AgentResponse(status="budget_exceeded", trace_id=request.thread_id, output=str(be), iterations=0)
        
    except Exception as e:
        logger.error(f"[{request.thread_id}] Fatal graph error:\n{traceback.format_exc()}")
        raise e
        
    finally:
        # 2. THE RECONCILIATION (Always refund unused money)
        budget_manager.release_deposit(trace_id=request.thread_id)