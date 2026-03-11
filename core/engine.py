import os
import uuid
import logging
import traceback
import json
from typing import List, Dict, Any
from openai import OpenAI
from dotenv import load_dotenv

from models.state import AgentRequest, AgentResponse, LoopState, state_db
from core.infrastructure import budget_manager, db
from core.security import asymmetric_action_gate
from tools.registry import execute_secure_tool, LLM_TOOLS

# --- THE NEW DYNAMIC IMPORTS ---
from skills.orchestrator import load_skill_context
from core.evaluator import run_dynamic_evaluation

logger = logging.getLogger("AgenticCore.Engine")

load_dotenv()

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ.get("GROQ_API_KEY")
)

def call_llm(messages: List[Dict]) -> Any:
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        tools=LLM_TOOLS,
        temperature=0.1
    )
    return response.choices[0].message

def assemble_tiered_context(user_id: str, base_system_prompt: str, history: List[Dict]) -> List[Dict]:
    profile = state_db.load_profile(user_id)
    task = state_db.load_task(user_id)
    
    state_block = {
        "role": "system",
        "content": f"CURRENT_FACTS: {profile.model_dump_json()}\nTASK_PROGRESS: {task.model_dump_json()}"
    }
    
    messages = [{"role": "system", "content": base_system_prompt}, state_block] + history
    return messages

def run_agentic_loop(request: AgentRequest) -> AgentResponse:
    # 1. Load the Paired-Skill Context from JSON Registry
    skill_context = load_skill_context(request.user_prompt)
    
    initial_history = [{"role": "user", "content": request.user_prompt}]
    
    # 2. Inject the specific Generator Prompt
    messages = assemble_tiered_context(request.user_id, skill_context["generator_prompt"], initial_history)
    
    state = LoopState(budget_lock_id=budget_manager.reserve(amount=5.00, ttl=300))
    logger.info(f"[{state.trace_id}] Starting loop. Budget reserved: {state.budget_lock_id}")

    try:
        while state.iteration < state.max_iters:
            logger.info(f"[{state.trace_id}] Iteration {state.iteration + 1}/{state.max_iters}")
            
            msg = call_llm(messages)
            
            safe_content = msg.content or ""
            assistant_msg = {"role": "assistant", "content": safe_content}
            safe_tool_calls = getattr(msg, "tool_calls", None) or []
            
            if safe_tool_calls:
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": getattr(tc, "type", "function"),
                        "function": {
                            "name": tc.function.name, 
                            "arguments": getattr(tc.function, "arguments", "{}")
                        }
                    } for tc in safe_tool_calls
                ]
            
            messages.append(assistant_msg)

            # Exit condition: LLM chose not to use any tools
            if not safe_tool_calls:
                
                # 3. Trigger the Dynamic Evaluator
                evaluation = run_dynamic_evaluation(
                    output_text=safe_content, 
                    objective=request.user_prompt, 
                    evaluator_prompt=skill_context["evaluator_prompt"], 
                    schema_class=skill_context["evaluator_schema"]
                )
                
                logger.info(f"[{state.trace_id}] Eval Pass Status: {evaluation.pass_status}")
                
                # If the Judge rejects it, force a rewrite
                if not evaluation.pass_status:
                    logger.warning(f"[{state.trace_id}] Evaluation Failed. Forcing revision.")
                    messages.append({
                        "role": "user", 
                        "content": f"Your output failed QA validation. Critique: {evaluation.critique}. Rewrite your final response to fix these issues. DO NOT output any tool calls, only the corrected text/JSON."
                    })
                    state.iteration += 1
                    continue
                
                # Save trace and exit safely
                db.save_idempotency(f"trace_{state.trace_id}", json.dumps(messages))
                logger.info(f"[{state.trace_id}] Final trace artifact saved to Redis.")

                budget_manager.release(state.budget_lock_id)
                return AgentResponse(
                    status="success",
                    trace_id=state.trace_id,
                    output=safe_content,
                    iterations=state.iteration + 1
                )

            # Idempotent Execution & Security Checks
            for call in safe_tool_calls:
                if db.check_idempotency(call.id):
                    result = db.get_result(call.id)
                else:
                    is_safe, sec_msg = asymmetric_action_gate(request.user_prompt, call.function.name)
                    
                    if not is_safe:
                        result = f'{{"error": "SecurityViolation", "message": "{sec_msg}"}}'
                    else:
                        result = execute_secure_tool(call.function.name, getattr(call.function, "arguments", "{}"))
                        db.save_idempotency(call.id, result)

                messages.append({"role": "tool", "tool_call_id": call.id, "content": result})
            
            state.iteration += 1

        budget_manager.release(state.budget_lock_id)
        return AgentResponse(
            status="error",
            trace_id=state.trace_id,
            output="Error: Guardrail Failure - Max iterations reached.",
            iterations=state.iteration
        )
        
    except Exception as e:
        logger.error(f"[{state.trace_id}] Fatal error in loop:\n{traceback.format_exc()}")
        budget_manager.release(state.budget_lock_id)
        raise e