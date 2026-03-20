import json
import time
import uuid
import logging
from typing import Tuple

from core.llm import call_llm
from core.infrastructure import db, budget_manager
from core.security import asymmetric_action_gate
from core.agents.agent_registry import swarm_registry
from core.evaluator import run_dynamic_evaluation
from core.telemetry import TelemetryLogger

from models.state import SubTask, SharedBriefcase
from models.telemetry import ActionStatus
from models.llm_schemas import StandardLLMResponse
from models.evaluations.base import BaseEvaluationSchema

from tools.registry import execute_secure_tool, LLM_TOOLS

logger = logging.getLogger("AgenticCore.NodeExecutor")

MAX_EVAL_RETRIES = 3
MAX_TOOL_ITERS = 5

def execute_worker_node(task: SubTask, briefcase: SharedBriefcase, thread_id: str) -> Tuple[bool, str]:
    telemetry = TelemetryLogger(trace_id=thread_id)

    # 1. PULL FROM THE YAML REGISTRY
    agent_def = swarm_registry.get_agent(task.agent_target)

    agent_specific_data = briefcase.domain_state.get(task.agent_target, {})

    # 2. BUILD THE DYNAMIC PROMPT
    system_prompt = agent_def.system_prompt_builder()

    # 3. THE STATIC PREFIX (Highly Cacheable)
    messages = [
        {
            "role": "system",
            "content": system_prompt,
            "cache_control": True
        }
    ]

    # 4. THE DYNAMIC PAYLOAD (The Cache Breakers)
    messages.append({
        "role": "user", 
        "content": f"INSTRUCTION: {task.instruction}\n\nVAULT DATA: {json.dumps(agent_specific_data)}"
    })

    eval_retries = 0
    worker_final_output = ""

    # --- OUTER LOOP: EVALUATOR REFLECTION ---
    while eval_retries < MAX_EVAL_RETRIES:
        tool_iters = 0

        # --- INNER LOOP: REACT & TOOLS ---
        while tool_iters < MAX_TOOL_ITERS:
            # 1. Fetch Provider-Agnostic Response
            response: StandardLLMResponse = call_llm(
                        messages=messages,
                        tier=agent_def.config.llm_tier,
                        temperature=agent_def.config.temperature,
                        tools=LLM_TOOLS,                           # <-- FIXED: Reverted to LLM_TOOLS temporarily
                        trace_id=thread_id
                    )

            # 2. FINOPS: BURN DOWN THE LEDGER
            if response.usage and response.usage.total_cost_usd > 0:
                budget_manager.burn_down(trace_id=thread_id, cost_usd=response.usage.total_cost_usd)

            # 3. Log Metrics
            if response.usage:
                telemetry.log_metric(
                    agent_id=task.agent_target,
                    tier="worker",
                    prompt_tokens=response.usage.prompt_tokens,
                    completion_tokens=response.usage.completion_tokens,
                    cost_usd=response.usage.total_cost_usd
                )
            safe_content = response.content or ""

            # 4. Log Decision
            if safe_content:
                telemetry.log_decision(
                    agent_id=task.agent_target,
                    reasoning=safe_content,
                    context=f"ReAct Loop Iteration {tool_iters + 1}"
                )

            # 5. Clean Tool Array Builder
            assistant_msg = {"role": "assistant", "content": safe_content}
            if response.tool_calls:
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function_name, "arguments": tc.arguments}
                    }
                    for tc in response.tool_calls
                ]

            messages.append(assistant_msg)

            # --- NEW: SWARM INTERCEPTOR ---
            if response.tool_calls:
                for tc in response.tool_calls:
                    if tc.function_name == "transfer_to_agent":
                        try:
                            args = json.loads(tc.arguments) if isinstance(tc.arguments, str) else tc.arguments
                        except json.JSONDecodeError:
                            args = {}

                        target = args.get("target_agent", "unknown")
                        reason = args.get("reason", "No reason provided.")

                        telemetry.log_decision(
                            agent_id=task.agent_target,
                            reasoning=f"Yielding to {target}. Reason: {reason}",
                            context="Swarm Handoff Routing"
                        )

                        briefcase.domain_state["handoff_context"] = reason
                        return True, f"__YIELD__{target}"

            # 6. Break if no tools are requested (Agent is done)
            if not response.tool_calls:
                worker_final_output = safe_content
                break

            # 7. Clean iteration over StandardToolCall objects
            for call in response.tool_calls:
                correlation_id = str(uuid.uuid4())

                telemetry.log_action(
                    agent_id=task.agent_target,
                    correlation_id=correlation_id,
                    tool_name=call.function_name,
                    arguments=call.arguments,
                    status=ActionStatus.PENDING
                )

                start_time = time.time()
                hash_key = db.generate_hash(thread_id, call.function_name, call.arguments)
                action_status = ActionStatus.SUCCESS
                result = ""

                if db.check_idempotency(hash_key):
                    result = db.get_result(hash_key)
                    logger.info(f"[{thread_id}] Idempotency hit: {call.function_name}")
                else:
                    is_safe, sec_msg = asymmetric_action_gate(task.instruction, call.function_name)
                    if not is_safe:
                        result = f'{{"error": "SecurityViolation", "message": "{sec_msg}"}}'
                        action_status = ActionStatus.FAILED
                    else:
                        try:
                            result = execute_secure_tool(call.function_name, call.arguments)
                            db.save_idempotency(hash_key, result)
                        except Exception as e:
                            logger.error(f"Tool {call.function_name} crashed: {str(e)}")
                            result = f'{{"error": "ToolExecutionFailed", "message": "{str(e)}. Please adjust your arguments and try again."}}'
                            action_status = ActionStatus.FAILED

                latency_ms = (time.time() - start_time) * 1000

                telemetry.log_action(
                    agent_id=task.agent_target,
                    correlation_id=correlation_id,
                    tool_name=call.function_name,
                    arguments=call.arguments,
                    status=action_status,
                    latency_ms=latency_ms,
                    result_summary=result[:500] if result else ""
                )

                messages.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": result
                })

            tool_iters += 1

        # --- QA GATE & CRITIQUE INJECTION ---
        agent_evaluator_rubric = agent_def.config.evaluator_rubric
        dynamic_evaluator_rubric = agent_evaluator_rubric or f"You are a strict QA Auditor reviewing the {task.agent_target.upper()}. Ensure the output fully satisfies the objective."
        # Maps to Pydantic Evaluation Model (models/evaluations) speficied in this agent's 'evaluator_schema_name' field in its config.yaml
        dynamic_evaluator_schema = agent_def.get_evaluation_schema

        evaluation = run_dynamic_evaluation(
            output_text=worker_final_output,
            objective=task.instruction,
            evaluator_prompt=dynamic_evaluator_rubric,
            schema_class=dynamic_evaluator_schema,
            trace_id=thread_id
        )

        telemetry.log_decision(
            agent_id=f"{task.agent_target}_judge",
            reasoning=evaluation.critique, 
            context=f"Pass Status: {evaluation.pass_status}"
        )

        if evaluation.pass_status:
            logger.info(f"[{thread_id}] Judge Passed {task.agent_target} on attempt {eval_retries + 1}.")
            return True, worker_final_output

        logger.warning(f"[{thread_id}] Judge Failed {task.agent_target}. Injecting critique. (Attempt {eval_retries + 1}/{MAX_EVAL_RETRIES})")

        messages.append({
            "role": "user",
            "content": f"QA SYSTEM REJECTION: Your previous output failed the objective. REASON: {evaluation.critique}. You must revise your output to fix this exact issue."
        })
        eval_retries += 1

    return False, f"Task Failed after {MAX_EVAL_RETRIES} QA attempts. Final Critique: {evaluation.critique}"