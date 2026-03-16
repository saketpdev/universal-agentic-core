import json
import logging
import traceback
from models.state import AgentRequest, AgentResponse
from core.infrastructure import budget_manager
from core.engine.state_manager import initialize_or_resume_state, checkpoint_state
from core.engine.node_executor import execute_worker_node
from core.planner import generate_execution_plan

logger = logging.getLogger("AgenticCore.DAGRunner")

def run_agentic_loop(request: AgentRequest) -> AgentResponse:
    budget_lock_id = budget_manager.reserve(amount=5.00, ttl=300)
    logger.info(f"[{request.thread_id}] Starting DAG Execution.")
    
    try:
        briefcase = initialize_or_resume_state(request)

        # Explicit Planner Execution
        if not briefcase.execution_plan:
            logger.info(f"[{request.thread_id}] Waking up Planner Agent...")
            briefcase.execution_plan = generate_execution_plan(request.user_prompt)
            checkpoint_state(briefcase, request)

        while briefcase.current_step_index < len(briefcase.execution_plan):
            current_task = briefcase.execution_plan[briefcase.current_step_index]
            
            if current_task.status == "completed":
                briefcase.current_step_index += 1
                continue
                
            logger.info(f"[{request.thread_id}] Executing Step {briefcase.current_step_index + 1}: {current_task.agent_target}")
            current_task.status = "in_progress"
            
            # Aggressive Checkpointing to prevent double-execution on crash
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
                budget_manager.release(budget_lock_id)
                return AgentResponse(status="error", trace_id=request.thread_id, output=output, iterations=briefcase.current_step_index)

            briefcase.domain_state[current_task.agent_target] = {"latest_output": output}
            current_task.status = "completed"
            briefcase.current_step_index += 1
            checkpoint_state(briefcase, request)

        budget_manager.release(budget_lock_id)
        return AgentResponse(
            status="success",
            trace_id=request.thread_id,
            output=json.dumps(briefcase.domain_state),
            iterations=briefcase.current_step_index
        )
        
    except Exception as e:
        logger.error(f"[{request.thread_id}] Fatal graph error:\n{traceback.format_exc()}")
        budget_manager.release(budget_lock_id)
        raise e