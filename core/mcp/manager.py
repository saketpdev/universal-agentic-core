import os
import yaml
import json
import logging
from typing import Dict, Any, List
from contextlib import AsyncExitStack

# Official MCP SDK Imports
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.sse import sse_client

logger = logging.getLogger("AgenticCore.MCPManager")

class MCPConnectionManager:
    def __init__(self):
        self.registry_path = "core/mcp/servers.yaml"
        self.servers_config = self._load_registry()
        self.sessions: Dict[str, ClientSession] = {}
        self.exit_stack = AsyncExitStack()
        
        # In-memory cache to prevent network spam during ReAct loops
        self._tool_cache: Dict[str, List[Dict[str, Any]]] = {}

    def _load_registry(self) -> Dict[str, Any]:
        if not os.path.exists(self.registry_path):
            return {}
        with open(self.registry_path, 'r') as f:
            return yaml.safe_load(f).get("mcpServers", {})

    async def connect_all(self):
        """Establishes streaming connections to all registered MCP servers."""
        for name, config in self.servers_config.items():
            try:
                transport_type = config.get("transport", "stdio")
                if transport_type == "stdio":
                    server_params = StdioServerParameters(
                        command=config["command"],
                        args=config.get("args", []),
                        env={**os.environ, **config.get("env", {})}
                    )
                    transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
                elif transport_type == "sse":
                    transport = await self.exit_stack.enter_async_context(sse_client(url=config["url"]))
                else:
                    logger.error(f"Unknown transport {transport_type} for MCP server {name}")
                    continue

                read, write = transport
                session = await self.exit_stack.enter_async_context(ClientSession(read, write))
                await session.initialize()
                
                self.sessions[name] = session
                logger.info(f"✅ Connected to MCP Server: {name} via {transport_type.upper()}")
                
            except Exception as e:
                logger.error(f"❌ Failed to connect to MCP Server '{name}': {e}")

    async def get_tools_for_server(self, server_name: str) -> List[Dict[str, Any]]:
        """Fetches tools from the remote server and translates them to OpenAI Schema."""
        if server_name not in self.sessions:
            return []
        if server_name in self._tool_cache:
            return self._tool_cache[server_name]

        session = self.sessions[server_name]
        try:
            mcp_tools = await session.list_tools()
            formatted_tools = []
            
            for tool in mcp_tools.tools:
                formatted_tools.append({
                    "type": "function",
                    "function": {
                        # 🚀 NAMESPACING: Prevents collisions if 2 servers have a "search" tool
                        "name": f"{server_name}__{tool.name}",
                        "description": tool.description,
                        "parameters": tool.inputSchema
                    }
                })
            
            self._tool_cache[server_name] = formatted_tools
            return formatted_tools
        except Exception as e:
            logger.error(f"Failed to fetch tools from {server_name}: {e}")
            return []

    async def execute_tool(self, server_name: str, namespaced_tool_name: str, arguments: dict) -> str:
        """Sends the payload over the wire to the target MCP server."""
        if server_name not in self.sessions:
            return json.dumps({"error": f"MCP server '{server_name}' is disconnected."})

        session = self.sessions[server_name]
        try:
            # Strip the namespace prefix so the target server recognizes its own tool
            original_tool_name = namespaced_tool_name.replace(f"{server_name}__", "", 1)
            
            result = await session.call_tool(original_tool_name, arguments=arguments)
            
            if not result.content:
                return "Tool executed successfully but returned no output."
            
            # Combine all text blocks returned by the server
            return "\n".join([block.text for block in result.content if block.type == "text"])
            
        except Exception as e:
            logger.error(f"MCP Execution failed on {server_name}: {e}")
            return json.dumps({"error": "MCP Network Failure", "message": str(e)})

    async def disconnect_all(self):
        await self.exit_stack.aclose()

mcp_manager = MCPConnectionManager()