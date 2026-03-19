import json
import logging

logger = logging.getLogger("AgenticCore.Tools")

IRREVERSIBLE_TOOLS = ["send_notification", "delete_record"]

def fetch_user_data(args_dict: dict) -> str:
    user_id = args_dict.get("user_id", "unknown")
    logger.info(f"TOOL EXECUTED: fetch_user_data for {user_id}")
    return json.dumps({"user_id": user_id, "status": "active", "tier": "premium"})

def send_notification(args_dict: dict) -> str:
    message = args_dict.get("message", "empty")
    logger.info(f"TOOL EXECUTED: send_notification - {message}")
    return json.dumps({"status": "success", "delivered_at": "2026-03-11T12:00:00Z"})

# --- PILLAR 4: Actionable Tool Specs (The LLM Interface) ---
LLM_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "transfer_to_agent",
            "description": "Hands off the conversation to a specialized agent.",
            "parameters": {
                "type": "object",
                "properties": {
                    "target_agent": {
                        "type": "string", 
                        "description": "The name of the agent to transfer to. Options: 'supervisor', 'customer_support', 'invoice_processor'"
                    },
                    "reason": {
                        "type": "string",
                        "description": "Brief explanation of why you are transferring to this agent."
                    }
                },
                "required": ["target_agent", "reason"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_user_data",
            "description": "Fetches user profile. Use this to check tier status.",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {"type": "string", "description": "The target user ID"}
                },
                "required": ["user_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "send_notification",
            "description": "Irreversible tool. Sends an alert to the user.",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {"type": "string", "description": "The alert text"}
                },
                "required": ["message"]
            }
        }
    }
]

TOOL_REGISTRY = {
    "fetch_user_data": fetch_user_data,
    "send_notification": send_notification
}

def execute_secure_tool(name: str, args: str) -> str:
    if name not in TOOL_REGISTRY:
        return json.dumps({"error": "ToolNotFound", "message": f"Tool '{name}' is not authorized."})

    try:
        args_dict = json.loads(args)
    except json.JSONDecodeError:
        return json.dumps({"error": "InvalidArguments", "message": "Failed to parse JSON arguments."})

    if name in IRREVERSIBLE_TOOLS:
        logger.warning(f"ACTION GATE TRIGGERED: Tool '{name}' requires human approval.")
        return json.dumps({"status": "PENDING_APPROVAL", "message": "Human review required for this action."})

    try:
        return TOOL_REGISTRY[name](args_dict)
    except Exception as e:
        return json.dumps({"error": "ExecutionFailed", "message": str(e)})

def transfer_to_agent(target_agent: str, reason: str) -> str:
    """
    Use this tool to yield execution to a different specialized agent if the current task 
    is outside your domain of expertise.
    
    Args:
        target_agent: The exact string name of the agent to route to (e.g., "legal_agent", "database_agent").
        reason: A brief explanation of why you are transferring this task and what the next agent should do.
    """
    # This string is technically never returned to the LLM, because the orchestrator 
    # intercepts the call and terminates the loop first.
    return f"TRANSFER_SIGNAL_INITIATED:{target_agent}:{reason}"