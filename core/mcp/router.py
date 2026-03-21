import json
import logging
from core.mcp.manager import mcp_manager

from core.engine.system_tools import execute_system_tool

logger = logging.getLogger("AgenticCore.MCPRouter")

async def route_and_execute_tool(function_name: str, arguments: str) -> str:
    """
    The Universal Interceptor.
    Routes tool calls to either local Python functions or external MCP servers.
    """
    try:
        # 1. MCP Network Route
        if "__" in function_name:
            server_name = function_name.split("__")[0]
            # Ensure arguments are a dictionary for the MCP SDK
            args_dict = json.loads(arguments) if isinstance(arguments, str) else arguments

            logger.info(f"Router: Forwarding '{function_name}' to MCP network...")
            return await mcp_manager.execute_tool(server_name, function_name, args_dict)

        # 2. Local Python Route (System Tools / Handoffs)
        logger.info(f"Router: Executing '{function_name}' locally...")

        return execute_system_tool(function_name, arguments)

    except Exception as e:
        logger.error(f"Tool Router crashed on '{function_name}': {e}")
        return json.dumps({"error": "RoutingFailure", "message": str(e)})