import json
import time
import uuid
import logging
from typing import Tuple

from models.state import SubTask, SharedBriefcase
from core.llm import call_llm
from core.infrastructure import db
from core.security import asymmetric_action_gate
from tools.registry import execute_secure_tool, LLM_TOOLS
from skills.orchestrator import get_agent_context
from core.evaluator import run_dynamic_evaluation
from core.telemetry import TelemetryLogger
from models.telemetry import ActionStatus
from models.llm_schemas import StandardLLMResponse

logger = logging.getLogger("AgenticCore.NodeExecutor")

MAX_EVAL_RETRIES = 3
MAX_TOOL_ITERS = 5

def execute_worker_node(task: SubTask, briefcase: SharedBriefcase, thread_id: str) -> Tuple[bool, str]:
    telemetry = TelemetryLogger(trace_id=thread_id)
    current_skill = get_agent_context(task.agent_target)
    agent_specific_data = briefcase.domain_state.get(task.agent_target, {})
    
    messages = [
        {"role": "system", "content": current_skill["generator_prompt"]},
        {"role": "user", "content": f"INSTRUCTION: {task.instruction}\n\nVAULT DATA: {json.dumps(agent_specific_data)}"}
    ]
    
    eval_retries = 0
    worker_final_output = ""
    
    # --- OUTER LOOP: EVALUATOR REFLECTION ---
    while eval_retries < MAX_EVAL_RETRIES:
        tool_iters = 0
        
        # --- INNER LOOP: REACT & TOOLS ---
        while tool_iters < MAX_TOOL_ITERS:
            # 1. Fetch Provider-Agnostic Response
            response: StandardLLMResponse = call_llm(messages, tools=LLM_TOOLS, tier="worker")
            
            # 2. Log Metrics
            if response.usage:
                telemetry.log_metric(
                    agent_id=task.agent_target, 
                    tier="worker", 
                    prompt_tokens=response.usage.prompt_tokens, 
                    completion_tokens=response.usage.completion_tokens
                )

            safe_content = response.content or ""
            
            # 3. Log Decision (The internal monologue)
            if safe_content:
                telemetry.log_decision(
                    agent_id=task.agent_target, 
                    reasoning=safe_content, 
                    context=f"ReAct Loop Iteration {tool_iters + 1}"
                )

            # 4. Clean Tool Array Builder
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
            
            # 5. Break if no tools are requested (Agent is done)
            if not response.tool_calls:
                worker_final_output = safe_content
                break 
                
            # 6. Clean iteration over StandardToolCall objects
            for call in response.tool_calls:
                correlation_id = str(uuid.uuid4())
                
                # Log Action (Pending)
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
                
                # Log Action (Resolved)
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
        evaluation = run_dynamic_evaluation(
            output_text=worker_final_output,
            objective=task.instruction,
            evaluator_prompt=current_skill["evaluator_prompt"],
            schema_class=current_skill["evaluator_schema"]
        )
        
        # Log Evaluator Decision
        telemetry.log_decision(
            agent_id=f"{task.agent_target}_judge", 
            reasoning=evaluation.critique, 
            context=f"Pass Status: {evaluation.pass_status}"
        )
        
        if evaluation.pass_status:
            logger.info(f"[{thread_id}] Judge Passed {task.agent_target} on attempt {eval_retries + 1}.")
            return True, worker_final_output
            
        logger.warning(f"[{thread_id}] Judge Failed {task.agent_target}. Injecting critique. (Attempt {eval_retries + 1}/{MAX_EVAL_RETRIES})")
        
        # Inject the slap on the wrist
        messages.append({
            "role": "user", 
            "content": f"QA SYSTEM REJECTION: Your previous output failed the objective. REASON: {evaluation.critique}. You must revise your output to fix this exact issue."
        })
        eval_retries += 1
        
    return False, f"Task Failed after {MAX_EVAL_RETRIES} QA attempts. Final Critique: {evaluation.critique}"