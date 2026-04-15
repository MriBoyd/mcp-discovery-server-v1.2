# mcp_server.py
import asyncio
import logging
import json
from pathlib import Path
from contextlib import AsyncExitStack
from typing import Any, AsyncIterator, Dict, Optional
from contextlib import asynccontextmanager
from dataclasses import dataclass
import sys
import contextlib
from mcp.server.fastmcp import FastMCP, Context
from mcp.server.session import ServerSession
from mcp.server.fastmcp.exceptions import ToolError
from hybrid_searcher import HybridToolSearcher
import traceback
import warnings
import types

# MCP Client imports
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.sse import sse_client

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stderr
)

# Module-level singleton for hybrid searcher
_shared_searcher: Optional[HybridToolSearcher] = None

def get_shared_searcher() -> HybridToolSearcher:
    """Lazily initialize and return a single shared HybridToolSearcher instance."""
    global _shared_searcher
    if _shared_searcher is None:
        with contextlib.redirect_stdout(sys.stderr):
            _shared_searcher = HybridToolSearcher()
            _shared_searcher.is_indexed = True
    return _shared_searcher


class MCPToolRegistry:
    """Registry of all MCP servers and their tools"""
    
    def __init__(self, config_path: Path):
        self.servers: Dict[str, Dict] = {}  # server_name -> {config, client, tools}
        self.exit_stack = AsyncExitStack()
        self._load_config(config_path)
    
    def _load_config(self, config_path: Path):
        """Load server configurations from JSON file"""
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path) as f:
            data = json.load(f)
        
        for server_config in data.get("servers", []):
            server_name = server_config["name"]
            
            # Build tools map for quick lookup
            tools = {}
            for tool in server_config.get("tools", []):
                tool_name = tool["name"]
                tools[tool_name] = {
                    "description": tool.get("description", ""),
                    "parameters": tool.get("parameters", {}),
                    "server_name": server_name
                }
            
            self.servers[server_name] = {
                "config": server_config,
                "tools": tools,
                "client": None  # Will be set when connected
            }
        
        logging.info(f"Loaded {len(self.servers)} MCP servers with {self.total_tools()} total tools")
    
    def total_tools(self) -> int:
        """Return total number of tools across all servers"""
        return sum(len(server["tools"]) for server in self.servers.values())
    
    def get_server_for_tool(self, tool_name: str) -> Optional[str]:
        """Find which server hosts the given tool"""
        for server_name, server_info in self.servers.items():
            if tool_name in server_info["tools"]:
                return server_name
        return None
    
    async def connect_to_server(self, server_name: str) -> ClientSession:
        """Connect to an MCP server and return the client session"""
        server_info = self.servers[server_name]
        
        if server_info["client"]:
            return server_info["client"]
        
        config = server_info["config"]
        transport = config["transport"]
        
        logging.info(f"Connecting to server '{server_name}' via {transport}")
        
        try:
            if transport == "stdio":
                server_params = StdioServerParameters(
                    command=config["command"],
                    args=config.get("args", []),
                    env=config.get("env")
                )
                # Enter the stdio client context manually and keep it open
                stdio_cm = stdio_client(server_params)
                read_stream, write_stream = await self.exit_stack.enter_async_context(stdio_client(server_params))

                client = await self.exit_stack.enter_async_context(ClientSession(read_stream, write_stream))
                # Enter the ClientSession context to start background tasks
                await client.initialize()
                # Store the transport context manager so we can close it later
                server_info["_transport_cm"] = stdio_cm
                
            elif transport == "sse":
                url = config["url"]
                sse_cm = sse_client(url)
                read_stream, write_stream = await self.exit_stack.enter_async_context(sse_client(url))

                client = await self.exit_stack.enter_async_context(ClientSession(read_stream, write_stream))
                await client.initialize()
                server_info["_transport_cm"] = sse_cm
                
            else:
                raise ValueError(f"Unknown transport: {transport}")
            
            server_info["client"] = client
            return client
            
        except Exception as e:
            logging.error(f"Failed to connect to {server_name}: {e}")
            raise ToolError(f"Cannot connect to MCP server '{server_name}': {str(e)}")
    
    async def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Execute a tool on its server"""
        server_name = self.get_server_for_tool(tool_name)
        if not server_name:
            raise ToolError(f"Tool '{tool_name}' not found in any configured MCP server")
        
        # Connect to the server
        client = await self.connect_to_server(server_name)
        
        # Execute the tool
        try:
            result = await client.call_tool(tool_name, arguments=arguments)
            # Format result
            if hasattr(result, 'content'):
                texts = [c.text for c in result.content if hasattr(c, 'text')]
                return "\n".join(texts) if texts else str(result)
            return str(result)
        except Exception as e:
            raise ToolError(f"Failed to execute '{tool_name}' on '{server_name}': {str(e)}")
    
    async def cleanup(self):
        """Close all server connections"""
        await self.exit_stack.aclose()


@dataclass
class AppContext:
    """Application context with shared resources"""
    searcher: HybridToolSearcher
    registry: MCPToolRegistry


@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
    """Manage application lifecycle"""
    searcher = get_shared_searcher()
    config_path = Path(__file__).parent / "mcp_servers.json"
    registry = MCPToolRegistry(config_path)
    
    yield AppContext(searcher=searcher, registry=registry)
    
    # Cleanup
    await registry.cleanup()


# ============= Create MCP Server =============

mcp_stdio = FastMCP(
    name="mcp-tool-search-server",
    instructions="Search and execute tools across multiple MCP servers",
    lifespan=app_lifespan
)

mcp_sse = FastMCP(
    name="mcp-tool-search-server",
    instructions="Search and execute tools across multiple MCP servers",
    lifespan=app_lifespan,
    port=8000,
    host="0.0.0.0"
)


# ============= Tool: Search for Tools (using hybrid search) =============

@mcp_stdio.tool(
    name="search_tools",
    description="Search for relevant MCP tools based on a natural language query. Returns tool names, descriptions, and parameter schemas."
)
@mcp_sse.tool(
    name="search_tools",
    description="Search for relevant MCP tools based on a natural language query. Returns tool names, descriptions, and parameter schemas."
)
async def search_tools(
    query: str,
    ctx: Context[ServerSession, AppContext]
) -> str:
    """Search for tools using hybrid retrieval."""
    try:
        searcher = ctx.request_context.lifespan_context.searcher
        
        await ctx.report_progress(0.3, 1.0, "Searching for relevant tools...")
        
        # Perform hybrid search
        results = await asyncio.to_thread(searcher.search, query)
        
        await ctx.report_progress(0.8, 1.0, "Formatting results...")
        
        if not results:
            return "No tools found matching your query."
        
        output = f"🔍 Found {len(results)} relevant tools for: '{query}'\n\n"
        
        for i, result in enumerate(results, 1):
            tool_schema = result.get('tool_schema', {})
            tool_name = tool_schema.get('name', result.get('tool_name', 'unknown'))
            tool_description = tool_schema.get('description', result.get('tool_description', 'No description'))
            input_schema = tool_schema.get('inputSchema', {})
            relevance_score = result.get('relevance_score', 1.0)
            
            output += f"{i}. **Tool Name:** `{tool_name}`\n"
            output += f"   **Relevance:** {relevance_score:.3f}\n"
            output += f"   **Description:** {tool_description}\n"
            
            # Format parameters
            if input_schema and input_schema.get('properties'):
                properties = input_schema.get('properties', {})
                required = input_schema.get('required', [])
                
                output += "   **Parameters:**\n"
                for param_name, param_info in properties.items():
                    param_type = param_info.get('type', 'any')
                    param_desc = param_info.get('description', 'No description')
                    is_required = param_name in required
                    required_marker = " (required)" if is_required else " (optional)"
                    
                    output += f"     - `{param_name}`{required_marker}: {param_type} - {param_desc}\n"
            else:
                output += "   **Parameters:** None\n"
            
            output += "\n"
        
        output += "---\n"
        output += "💡 **To use a tool:** Call `call_tool` with the exact tool name and arguments as a JSON string.\n"
        
        return output
        
    except Exception as e:
        raise ToolError(f"Search failed: {str(e)}")


# ============= Tool: Call Tool Proxy =============

@mcp_stdio.tool(
    name="call_tool",
    description="Execute a tool on its original MCP server. Provide the exact tool name and arguments as a JSON string."
)
@mcp_sse.tool(
    name="call_tool",
    description="Execute a tool on its original MCP server. Provide the exact tool name and arguments as a JSON string."
)
async def call_tool(
    tool_name: str,
    arguments: str,
    ctx: Context[ServerSession, AppContext]
) -> str:
    """
    Call a tool on its original MCP server.
    
    Args:
        tool_name: The exact unique identifier of the tool (e.g., "Demo--Everything-__echo")
        arguments: JSON string of arguments to pass to the tool (e.g., '{"message": "Hello"}')
    """
    try:
        # Parse arguments
        try:
            args_dict = json.loads(arguments) if arguments else {}
        except json.JSONDecodeError as e:
            raise ToolError(f"Invalid JSON arguments: {str(e)}. Arguments must be a valid JSON string.")
        
        await ctx.report_progress(0.3, 1.0, f"Looking up server for '{tool_name}'...")
        
        # Execute the tool using registry
        registry = ctx.request_context.lifespan_context.registry
        result = await registry.execute_tool(tool_name, args_dict)
        
        await ctx.report_progress(0.9, 1.0, "Tool execution completed")
        
        return f"✅ **Tool executed successfully:** `{tool_name}`\n\n{result}"
        
    except ToolError:
        raise
    except Exception as e:
        raise ToolError(f"Tool execution failed: {str(e)}")


# ============= Main Entry Points =============

async def run_stdio():
    await mcp_stdio.run_stdio_async()

async def run_sse():
    await mcp_sse.run_sse_async()

if __name__ == "__main__":
    import sys
    # Show ResourceWarnings with tracebacks to help diagnose unclosed streams
    warnings.simplefilter("default", ResourceWarning)

    def _log_exception_group(exc: BaseException):
        """Log ExceptionGroup/BaseExceptionGroup inner exceptions with tracebacks."""
        logging.error("Top-level exception: %s", exc)
        inner = getattr(exc, 'exceptions', None)
        if inner and isinstance(inner, (list, tuple)):
            for i, sub in enumerate(inner, start=1):
                try:
                    if isinstance(sub, BaseException):
                        logging.error("--- Sub-exception %d: %s", i, sub)
                        tb = ''.join(traceback.format_exception(type(sub), sub, sub.__traceback__))
                        logging.error(tb)
                    else:
                        logging.error("--- Sub-exception %d (non-exception): %r", i, sub)
                except Exception:
                    logging.exception("Failed to log sub-exception %d", i)

    try:
        if len(sys.argv) > 1 and sys.argv[1] == "--sse":
            asyncio.run(run_sse())
        else:
            asyncio.run(run_stdio())
    except BaseException as e:
        # anyio/asyncio may raise ExceptionGroup/BaseExceptionGroup. Log details.
        if hasattr(e, 'exceptions'):
            _log_exception_group(e)
        else:
            logging.exception("Unhandled exception in main: %s", e)
        raise