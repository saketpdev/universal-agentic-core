import json
import logging
from typing import Tuple
from models.state import SubTask, SharedBriefcase
from core.llm import call_llm
from core.infrastructure import db
from core.security import asymmetric_action_gate
from tools.registry import execute_secure_tool, LLM_TOOLS # Imported here for DI
from skills.orchestrator import get_agent_context
from core.evaluator import run_dynamic_evaluation

logger = logging.getLogger("AgenticCore.NodeExecutor")

MAX_EVAL_RETRIES = 3
MAX_TOOL_ITERS = 5

def execute_worker_node(task: SubTask, briefcase: SharedBriefcase, thread_id: str) -> Tuple[bool, str]:
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
            response_msg = call_llm(
                messages=messages, 
                tools=LLM_TOOLS,
                tier="worker"
            )
            safe_content = response_msg.content or ""
            assistant_msg = {"role": "assistant", "content": safe_content}
            
            safe_tool_calls = getattr(response_msg, "tool_calls", None) or []
            if safe_tool_calls:
                assistant_msg["tool_calls"] = [
                    {"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}} 
                    for tc in safe_tool_calls
                ]
            
            messages.append(assistant_msg)
            
            if not safe_tool_calls:
                worker_final_output = safe_content
                break # Finished tool loop
                
            for call in safe_tool_calls:
                args_str = getattr(call.function, "arguments", "{}")
                
                # Deterministic Idempotency Hash
                hash_key = db.generate_hash(thread_id, call.function.name, args_str)
                
                if db.check_idempotency(hash_key):
                    result = db.get_result(hash_key)
                    logger.info(f"[{thread_id}] Idempotency hit: {call.function.name}")
                else:
                    is_safe, sec_msg = asymmetric_action_gate(task.instruction, call.function.name)
                    if not is_safe:
                        result = f'{{"error": "SecurityViolation", "message": "{sec_msg}"}}'
                    else:
                        # Self-Correcting Try/Catch wrapper
                        try:
                            result = execute_secure_tool(call.function.name, args_str)
                            db.save_idempotency(hash_key, result)
                        except Exception as e:
                            logger.error(f"Tool {call.function.name} crashed: {str(e)}")
                            result = f'{{"error": "ToolExecutionFailed", "message": "{str(e)}. Please adjust your arguments and try again."}}'
                            
                messages.append({"role": "tool", "tool_call_id": call.id, "content": result})
            tool_iters += 1

        # --- QA GATE & CRITIQUE INJECTION ---
        evaluation = run_dynamic_evaluation(
            output_text=worker_final_output,
            objective=task.instruction,
            evaluator_prompt=current_skill["evaluator_prompt"],
            schema_class=current_skill["evaluator_schema"]
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