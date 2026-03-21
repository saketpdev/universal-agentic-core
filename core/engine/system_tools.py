import json

# These are CORE tools that belong to the Swarm, not external APIs.
SYSTEM_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "transfer_to_agent",
            "description": "Hands off the conversation to a specialized agent.",
            "parameters": {
                "type": "object",
                "properties": {
                    "target_agent": {"type": "string", "description": "Name of the agent."},
                    "reason": {"type": "string", "description": "Why you are transferring."}
                },
                "required": ["target_agent", "reason"]
            }
        }
    }
]

def execute_system_tool(name: str, args: str) -> str:
    """Executes internal Swarm routing logic."""
    if name == "transfer_to_agent":
        # Handled inherently by the interceptor in node_executor.py
        return "TRANSFER_SIGNAL_INITIATED"
        
    return json.dumps({"error": "SystemToolNotFound"})