import json
import logging
import traceback
import asyncio

from models.state import AgentRequest, AgentResponse, Task, Stage, FailurePolicy, SharedBriefcase
from core.infrastructure import budget_manager, BudgetExceededException
from core.engine.state_manager import initialize_or_resume_state, checkpoint_state
from core.engine.node_executor import execute_worker_node
from core.telemetry import TelemetryLogger
from core.planner import get_execution_plan

logger = logging.getLogger("AgenticCore.DAGRunner")

WORKFLOW_RESERVE_USD = 0.50

async def _execute_and_checkpoint(task: Task, briefcase: SharedBriefcase, request: AgentRequest, telemetry: TelemetryLogger):
    """
    PER-TASK CHECKPOINTING: 
    Ensures that if the server crashes mid-stage, completed tasks are not re-run.
    """
    task.status = "in_progress"
    checkpoint_state(briefcase, request)

    success, output = await execute_worker_node(
        task=task,
        briefcase=briefcase,
        thread_id=request.thread_id
    )

    if success and isinstance(output, str) and output.startswith("__YIELD__"):
        task.status = "yielded"
        briefcase.domain_state[task.agent_target] = {"status": "yielded to another agent"}
        checkpoint_state(briefcase, request)
        return task, success, output

    if not success:
        task.status = "failed"
        checkpoint_state(briefcase, request)
        return task, success, output

    briefcase.domain_state[task.agent_target] = {"latest_output": output}
    task.status = "completed"

    # Commit completion to both OpenTelemetry and SQLite instantly
    await telemetry.log_state(task.agent_target, briefcase.current_stage_index, briefcase.domain_state[task.agent_target])
    checkpoint_state(briefcase, request)

    return task, success, output

async def run_agentic_loop(request: AgentRequest) -> AgentResponse:
    telemetry = TelemetryLogger(trace_id=request.thread_id)

    try:
        await budget_manager.reserve_deposit(trace_id=request.thread_id, max_budget_usd=WORKFLOW_RESERVE_USD)

        briefcase = initialize_or_resume_state(request)

        if not briefcase.execution_plan:
            briefcase.execution_plan = await get_execution_plan(request)
            checkpoint_state(briefcase, request)

        # ==========================================
        # PARALLEL STAGE ENGINE
        # ==========================================
        while briefcase.current_stage_index < len(briefcase.execution_plan.planned_stages):
            current_stage = briefcase.execution_plan.planned_stages[briefcase.current_stage_index]

            # Extract tasks that survived a crash or are fresh
            pending_tasks = [t for t in current_stage.tasks if t.status in ["pending", "in_progress"]]

            if not pending_tasks:
                briefcase.current_stage_index += 1
                checkpoint_state(briefcase, request)
                continue

            logger.info(f"[{request.thread_id}] Executing Stage {current_stage.stage_id} '{current_stage.stage_name}' ({len(pending_tasks)} parallel tasks)")

            # Fire all tasks in the current stage concurrently
            coroutines = [_execute_and_checkpoint(t, briefcase, request, telemetry) for t in pending_tasks]
            results = await asyncio.gather(*coroutines)

            yielded_tasks_to_insert = []
            critical_failure = False
            error_msg = ""

            # ==========================================
            # STATE RESOLUTION & FAILURE POLICIES
            # ==========================================
            for task, _, output in results:
                if task.status == "yielded":
                    target_agent = output.split("__YIELD__")[1]
                    handoff_data = briefcase.domain_state.get("handoff_context", {})
                    handoff_reason_str = handoff_data.get("reason", "") if isinstance(handoff_data, dict) else ""

                    yielded_tasks_to_insert.append(Task(
                        agent_target=target_agent,
                        instruction=f"CONTINUE WORKFLOW. Context: {handoff_reason_str}",
                        status="pending"
                    ))
                elif task.status == "failed":
                    if task.on_failure == FailurePolicy.TERMINATE:
                        logger.error(f"[{request.thread_id}] Task '{task.agent_target}' triggered TERMINATE policy.")
                        critical_failure = True
                        error_msg = output
                    elif task.on_failure == FailurePolicy.PAUSE:
                        logger.warning(f"[{request.thread_id}] Task '{task.agent_target}' triggered PAUSE policy.")
                        critical_failure = True
                        error_msg = f"PAUSED FOR REVIEW: {output}"
                    elif task.on_failure == FailurePolicy.IGNORE:
                        logger.warning(f"[{request.thread_id}] Task '{task.agent_target}' failed, but policy is IGNORE. Proceeding.")

            if critical_failure:
                briefcase.has_critical_error = True
                checkpoint_state(briefcase, request)
                return AgentResponse(status="error", trace_id=request.thread_id, output=error_msg, iterations=briefcase.current_stage_index)

            # ==========================================
            # DYNAMIC DAG MUTATION (For yielded tasks)
            # ==========================================
            if yielded_tasks_to_insert:
                new_stage = Stage(
                    stage_id=current_stage.stage_id + 1,
                    stage_name=f"Handoff Resolution (Spawned from Stage {current_stage.stage_id})",
                    tasks=yielded_tasks_to_insert
                )

                # Shift all downstream stage IDs up by 1 to maintain array integrity
                for i in range(briefcase.current_stage_index + 1, len(briefcase.execution_plan.planned_stages)):
                    briefcase.execution_plan.planned_stages[i].stage_id += 1

                briefcase.execution_plan.planned_stages.insert(briefcase.current_stage_index + 1, new_stage)

            # Stage cleared! Move pointer forward.
            briefcase.current_stage_index += 1
            checkpoint_state(briefcase, request)

        return AgentResponse(
            status="success",
            trace_id=request.thread_id,
            output=json.dumps(briefcase.domain_state),
            iterations=briefcase.current_stage_index
        )
    except BudgetExceededException as be:
        logger.error(f"[{request.thread_id}] Workflow terminated due to FinOps Guardrail: {str(be)}")
        return AgentResponse(status="budget_exceeded", trace_id=request.thread_id, output=str(be), iterations=0)

    except Exception as e:
        logger.error(f"[{request.thread_id}] Fatal graph error:\n{traceback.format_exc()}")
        raise e

    finally:
        await budget_manager.release_deposit(trace_id=request.thread_id)