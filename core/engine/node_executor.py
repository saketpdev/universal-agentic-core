import json
import time
import uuid
import logging
import asyncio
from typing import Any, Dict, Optional, Tuple, cast

from core.mcp.manager import mcp_manager
from core.mcp.router import route_and_execute_tool
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

from core.engine.system_tools import SYSTEM_TOOLS

logger = logging.getLogger("AgenticCore.NodeExecutor")

MAX_EVAL_RETRIES = 3
MAX_TOOL_ITERS = 5

async def _execute_single_tool(call, task, thread_id, telemetry) -> Dict[str, Any]:
    correlation_id = str(uuid.uuid4())
    await telemetry.log_action(task.agent_target, correlation_id, call.function_name, call.arguments, ActionStatus.PENDING)

    start_time = time.time()
    hash_key = db.generate_hash(thread_id, call.function_name, call.arguments)
    action_status = ActionStatus.SUCCESS
    result = ""

    if await db.check_idempotency(hash_key):
        result = await db.get_result(hash_key)
        logger.info(f"[{thread_id}] Idempotency hit: {call.function_name}")
    else:
        is_safe, sec_msg = asymmetric_action_gate(task.instruction, call.function_name)
        if not is_safe:
            result = f'{{"error": "SecurityViolation", "message": "{sec_msg}"}}'
            action_status = ActionStatus.FAILED
        else:
            try:
                # 🚀 DELEGATED TO THE UNIVERSAL ROUTER
                result = await route_and_execute_tool(call.function_name, call.arguments)
                await db.save_idempotency(hash_key, result)
            except Exception as e:
                logger.error(f"Tool {call.function_name} crashed: {str(e)}")
                result = f'{{"error": "ToolExecutionFailed", "message": "{str(e)}. Please adjust your arguments and try again."}}'
                action_status = ActionStatus.FAILED

    latency_ms = (time.time() - start_time) * 1000

    await telemetry.log_action(task.agent_target, correlation_id, call.function_name, call.arguments, action_status, latency_ms, result[:500] if result else "")

    return {
        "role": "tool",
        "tool_call_id": call.id,
        "content": result
    }

async def execute_worker_node(task: SubTask, briefcase: SharedBriefcase, thread_id: str) -> Tuple[bool, str]:
    telemetry = TelemetryLogger(trace_id=thread_id)
    agent_def = swarm_registry.get_agent(task.agent_target)
    agent_specific_data = briefcase.domain_state.get(task.agent_target, {})
    system_prompt = agent_def.system_prompt_builder()

    messages = [{"role": "system", "content": system_prompt, "cache_control": True}]
    messages.append({"role": "user", "content": f"INSTRUCTION: {task.instruction}\n\nVAULT DATA: {json.dumps(agent_specific_data)}"})

    # 🚀 START WITH SYSTEM TOOLS
    dynamic_tools = list(SYSTEM_TOOLS) 

    # 🚀 DYNAMICALLY APPEND AUTHORIZED MCP TOOLS
    for server in agent_def.config.allowed_mcp_servers:
        mcp_tools = await mcp_manager.get_tools_for_server(server)
        dynamic_tools.extend(mcp_tools)

    eval_retries = 0
    worker_final_output = ""
    evaluation: Optional[BaseEvaluationSchema] = None

    while eval_retries < MAX_EVAL_RETRIES:
        tool_iters = 0

        while tool_iters < MAX_TOOL_ITERS:

            response: StandardLLMResponse = await call_llm(
                        messages=messages,
                        tier=agent_def.config.llm_tier,
                        temperature=agent_def.config.temperature,
                        tools=dynamic_tools if dynamic_tools else None,
                        trace_id=thread_id
                    )

            if response.usage and response.usage.total_cost_usd > 0:
                await budget_manager.burn_down(trace_id=thread_id, cost_usd=response.usage.total_cost_usd)

            if response.usage:
                await telemetry.log_metric(task.agent_target, "worker", response.usage.prompt_tokens, response.usage.completion_tokens, response.usage.total_cost_usd)

            safe_content = response.content or ""

            if safe_content:
                await telemetry.log_decision(task.agent_target, safe_content, f"ReAct Loop Iteration {tool_iters + 1}")

            assistant_msg: Dict[str, Any] = {"role": "assistant", "content": safe_content}
            if response.tool_calls:
                assistant_msg["tool_calls"] = [{"id": tc.id, "type": "function", "function": {"name": tc.function_name, "arguments": tc.arguments}} for tc in response.tool_calls]

            messages.append(assistant_msg)

            if response.tool_calls:
                handoff_target = None
                handoff_reason = None
                
                # 1. Identify if a handoff was requested
                for tc in response.tool_calls:
                    if tc.function_name == "transfer_to_agent":
                        args = json.loads(tc.arguments) if isinstance(tc.arguments, str) else tc.arguments
                        handoff_target = args.get("target_agent", "unknown")
                        handoff_reason = args.get("reason", "No reason provided.")

                # 2. Execute all OTHER tools concurrently
                standard_calls = [tc for tc in response.tool_calls if tc.function_name != "transfer_to_agent"]
                if standard_calls:
                    tool_tasks = [_execute_single_tool(call, task, thread_id, telemetry) for call in standard_calls]
                    completed_tool_messages = await asyncio.gather(*tool_tasks)
                    messages.extend(completed_tool_messages)

                # 3. NOW yield control to the next agent safely
                if handoff_target:
                    await telemetry.log_decision(task.agent_target, f"Yielding to {handoff_target}. Reason: {handoff_reason}", "Swarm Handoff Routing")
                    briefcase.domain_state["handoff_context"] = {"reason": handoff_reason or "No reason provided."}
                    return True, f"__YIELD__{handoff_target}"

            if not response.tool_calls:
                worker_final_output = safe_content
                break

            # 🚀 PARALLEL EXECUTION OF ROUTED TOOLS
            tool_tasks = [_execute_single_tool(call, task, thread_id, telemetry) for call in response.tool_calls]
            completed_tool_messages = await asyncio.gather(*tool_tasks)

            messages.extend(completed_tool_messages)

            tool_iters += 1

        agent_evaluator_rubric = agent_def.config.evaluator_rubric
        dynamic_evaluator_rubric = agent_evaluator_rubric or f"You are a strict QA Auditor reviewing the {task.agent_target.upper()}. Ensure the output fully satisfies the objective."
        dynamic_evaluator_schema = agent_def.get_evaluation_schema

        evaluation = await run_dynamic_evaluation(
            output_text=worker_final_output,
            objective=task.instruction,
            evaluator_prompt=dynamic_evaluator_rubric,
            schema_class=dynamic_evaluator_schema,
            trace_id=thread_id
        )

        await telemetry.log_decision(f"{task.agent_target}_judge", evaluation.critique, f"Pass Status: {evaluation.pass_status}")

        if evaluation.pass_status:
            logger.info(f"[{thread_id}] Judge Passed {task.agent_target} on attempt {eval_retries + 1}.")
            return True, worker_final_output

        logger.warning(f"[{thread_id}] Judge Failed {task.agent_target}. Injecting critique. (Attempt {eval_retries + 1}/{MAX_EVAL_RETRIES})")

        messages.append({
            "role": "user",
            "content": f"QA SYSTEM REJECTION: Your previous output failed the objective. REASON: {evaluation.critique}. You must revise your output to fix this exact issue."
        })
        eval_retries += 1

    final_critique = evaluation.critique if evaluation else "Maximum evaluation retries exceeded."
    return False, f"Task Failed after {MAX_EVAL_RETRIES} QA attempts. Final Critique: {final_critique}"