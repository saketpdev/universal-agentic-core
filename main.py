import json
import uuid
import logging
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn
import redis

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AgenticCore")

app = FastAPI(title="Universal Agentic Core", version="1.0.0")

# --- PRODUCTION REDIS INFRASTRUCTURE ---
# Connect to the local Redis Docker container
redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)

class RedisBudgetManager:
    def reserve(self, amount: float, ttl: int) -> str:
        lock_id = f"budget_lock_{uuid.uuid4().hex[:8]}"
        # Reserve an Economic Lock with a Time-To-Live. If the process crashes, the budget must auto-release. 
        redis_client.set(lock_id, str(amount), ex=ttl)
        return lock_id
        
    def release(self, lock_id: str):
        redis_client.delete(lock_id)
        logger.info(f"Budget lock {lock_id} released in Redis.")

class RedisIdempotencyRegistry:
    def check_idempotency(self, call_id: str) -> bool:
        # Every tool call MUST include a tool_call_id. The executor must check a database/cache before running.
        return redis_client.exists(f"idem:{call_id}") > 0
        
    def get_result(self, call_id: str) -> str:
        # If a result exists for that ID, return it immediately. 
        return redis_client.get(f"idem:{call_id}")
        
    def save_idempotency(self, call_id: str, result: str):
        # Cache the tool execution result for 24 hours
        redis_client.set(f"idem:{call_id}", result, ex=86400)

budget_manager = RedisBudgetManager()
db = RedisIdempotencyRegistry()

# --- THE SECURE TOOL REGISTRY ---
# PILLAR 6: Define tools that have a blast radius
IRREVERSIBLE_TOOLS = ["send_notification", "delete_record"]

def fetch_user_data(args_dict: dict) -> str:
    """Mock reversible tool: Fetches user profile."""
    user_id = args_dict.get("user_id", "unknown")
    logger.info(f"TOOL EXECUTED: fetch_user_data for {user_id}")
    return json.dumps({"user_id": user_id, "status": "active", "tier": "premium"})

def send_notification(args_dict: dict) -> str:
    """Mock irreversible tool: Sends an alert."""
    message = args_dict.get("message", "empty")
    logger.info(f"TOOL EXECUTED: send_notification - {message}")
    return json.dumps({"status": "success", "delivered_at": "2026-03-11T12:00:00Z"})

# Map the LLM's requested string to the actual Python function
TOOL_REGISTRY = {
    "fetch_user_data": fetch_user_data,
    "send_notification": send_notification
}

def execute_secure_tool(name: str, args: str) -> str:
    # 1. Boundary Check: Is the tool recognized?
    if name not in TOOL_REGISTRY:
        return json.dumps({"error": "ToolNotFound", "message": f"Tool '{name}' is not authorized."})
        
    # 2. Parse the LLM's JSON arguments safely
    try:
        args_dict = json.loads(args)
    except json.JSONDecodeError:
        return json.dumps({"error": "InvalidArguments", "message": "Failed to parse JSON arguments."})

    # 3. Asymmetric Action Gate (SPEC_03)
    if name in IRREVERSIBLE_TOOLS:
        logger.warning(f"ACTION GATE TRIGGERED: Tool '{name}' requires human approval.")
        return json.dumps({"status": "PENDING_APPROVAL", "message": "Human review required for this action."})

    # 4. Execute the mapped Python function safely
    try:
        return TOOL_REGISTRY[name](args_dict)
    except Exception as e:
        return json.dumps({"error": "ExecutionFailed", "message": str(e)})

# --- PYDANTIC MODELS ---
class AgentRequest(BaseModel):
    user_prompt: str
    system_prompt: str = "You are a deterministic backend agent."
    user_id: str

class AgentResponse(BaseModel):
    status: str
    trace_id: str
    output: str
    iterations: int

class LoopState(BaseModel):
    iteration: int = 0
    max_iters: int = 5
    budget_lock_id: str
    trace_id: str = Field(default_factory=lambda: str(uuid.uuid4()))

# --- THE MOCKED LLM ROUTER ---
def mock_llm_call(messages: List[Dict]) -> Any:
    """Mocks the OpenAI SDK response to prevent requiring an API key right now."""
    class MockMessage:
        def __init__(self, content, tool_calls):
            self.content = content
            self.tool_calls = tool_calls
            self.role = "assistant"
            
    class MockToolCall:
        def __init__(self, id, name, arguments):
            self.id = id
            class Func: pass
            self.function = Func()
            self.function.name = name
            self.function.arguments = arguments

    iteration_count = len([m for m in messages if m.get("role") == "assistant"])
    
    if iteration_count == 0:
        # First pass: request a tool
        return MockMessage(None, [MockToolCall(f"call_{uuid.uuid4().hex[:4]}", "fetch_data", '{"query": "test"}')])
    else:
        # Second pass: return final answer
        return MockMessage("Task completed successfully based on tool data.", None)

# --- THE API ENDPOINT ---
@app.post("/execute", response_model=AgentResponse)
def run_production_loop(request: AgentRequest):
    messages = [
        {"role": "system", "content": request.system_prompt},
        {"role": "user", "content": request.user_prompt}
    ]
    
    state = LoopState(budget_lock_id=budget_manager.reserve(amount=5.00, ttl=300))
    logger.info(f"[{state.trace_id}] Starting loop. Budget reserved: {state.budget_lock_id}")

    try:
        while state.iteration < state.max_iters:
            logger.info(f"[{state.trace_id}] Iteration {state.iteration + 1}/{state.max_iters}")
            
            # Step 1: LLM Routing
            msg = mock_llm_call(messages)
            messages.append({"role": "assistant", "content": msg.content, "tool_calls": getattr(msg, "tool_calls", None)})

            if not getattr(msg, "tool_calls", None):
                budget_manager.release(state.budget_lock_id)
                return AgentResponse(
                    status="success",
                    trace_id=state.trace_id,
                    output=msg.content,
                    iterations=state.iteration + 1
                )

            # Step 2: Idempotent Execution
            for call in msg.tool_calls:
                if db.check_idempotency(call.id):
                    result = db.get_result(call.id)
                else:
                    try:
                        result = execute_secure_tool(call.function.name, call.function.arguments)
                        db.save_idempotency(call.id, result)
                    except Exception as e:
                        result = json.dumps({"error": type(e).__name__, "message": str(e)})

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
        budget_manager.release(state.budget_lock_id)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)