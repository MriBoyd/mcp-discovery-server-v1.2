# mcp_server.py
import asyncio
import logging
import json
from pathlib import Path
from contextlib import AsyncExitStack
from typing import Any, AsyncIterator, Dict, List, Optional, Union
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

from src.config import Config
from src.auth_manager import AuthManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stderr
)

# Module-level singleton for hybrid searcher and auth manager
_shared_searcher: Optional[HybridToolSearcher] = None
_auth_manager: Optional[AuthManager] = None

def get_shared_searcher() -> HybridToolSearcher:
    """Lazily initialize and return a single shared HybridToolSearcher instance."""
    global _shared_searcher
    if _shared_searcher is None:
        with contextlib.redirect_stdout(sys.stderr):
            _shared_searcher = HybridToolSearcher()
            _shared_searcher.is_indexed = True
    return _shared_searcher

def get_auth_manager() -> AuthManager:
    """Lazily initialize and return a single shared AuthManager instance."""
    global _auth_manager
    if _auth_manager is None:
        _auth_manager = AuthManager()
    return _auth_manager


class ServerConnection:
    """Manages the lifecycle of a single MCP server connection in its own dedicated task."""
    def __init__(self, server_name: str, config: Dict, auth_manager: AuthManager):
        self.server_name = server_name
        self.config = config
        self.auth_manager = auth_manager
        self.client: Optional[ClientSession] = None
        self.error: Optional[Exception] = None
        self._connected_event = asyncio.Event()
        self._stop_event = asyncio.Event()
        self._task: Optional[asyncio.Task] = None
        self._stack: Optional[AsyncExitStack] = None
        self._lock = asyncio.Lock()

    async def start(self):
        """Start the connection task and wait for it to be ready."""
        async with self._lock:
            if self._task and not self._task.done():
                await self._connected_event.wait()
                if self.error:
                    raise self.error
                return self.client

            self._connected_event.clear()
            self._stop_event.clear()
            self.error = None
            self._task = asyncio.create_task(self._run())
            
            try:
                async with asyncio.timeout(35.0): # Slightly longer than inner timeout
                    await self._connected_event.wait()
            except asyncio.TimeoutError:
                self.error = ToolError(f"Connection to '{self.server_name}' timed out during startup")
            
            if self.error:
                # Try to cleanup if task failed
                await self.stop()
                if isinstance(self.error, ToolError):
                    raise self.error
                raise ToolError(f"Failed to connect to '{self.server_name}': {str(self.error)}")
                
            return self.client

    async def _run(self):
        """Background task that owns the connection and its contexts."""
        self._stack = AsyncExitStack()
        try:
            transport = self.config["transport"]
            headers = self.config.get("headers", {})
            
            if "auth" in self.config:
                token = self.auth_manager.get_token(self.server_name)
                if not token:
                    token = await self.auth_manager.refresh_token(self.server_name, self.config)
                if token:
                    headers["Authorization"] = f"Bearer {token}"

            async with asyncio.timeout(30.0):
                if transport == "stdio":
                    server_params = StdioServerParameters(
                        command=self.config["command"],
                        args=self.config.get("args", []),
                        env=self.config.get("env")
                    )
                    read_stream, write_stream = await self._stack.enter_async_context(stdio_client(server_params))
                    self.client = await self._stack.enter_async_context(ClientSession(read_stream, write_stream))
                    await self.client.initialize()
                elif transport in ["sse", "http"]:
                    url = self.config["url"]
                    read_stream, write_stream = await self._stack.enter_async_context(sse_client(url, headers=headers))
                    self.client = await self._stack.enter_async_context(ClientSession(read_stream, write_stream))
                    await self.client.initialize()
                else:
                    raise ValueError(f"Unknown transport: {transport}")
                
            # Successfully connected
            self._connected_event.set()
            
            # Keep the task alive until stopped
            await self._stop_event.wait()
            
        except Exception as e:
            self.error = e
            self._connected_event.set()
        finally:
            if self._stack:
                await self._stack.aclose()
            self.client = None
            self._stack = None

    async def stop(self):
        """Stop the connection task and wait for cleanup."""
        self._stop_event.set()
        if self._task:
            if not self._task.done():
                try:
                    await asyncio.wait_for(self._task, timeout=5.0)
                except (asyncio.TimeoutError, Exception):
                    self._task.cancel()
            self._task = None

class MCPToolRegistry:
    """Registry of all MCP servers and their tools with LRU connection pooling"""
    
    def __init__(self, config_path: Path, auth_manager: AuthManager):
        self.servers: Dict[str, Dict] = {}  # server_name -> {config, tools, connection}
        self.tool_to_server: Dict[str, str] = {} # tool_name -> server_name
        self._active_connections: List[str] = [] # LRU list of server names
        self.auth_manager = auth_manager
        self._load_config(config_path)
    
    def _load_config(self, config_path: Path):
        """Load server configurations from JSON file"""
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path) as f:
            data = json.load(f)
        
        self.tool_to_server.clear()
        for server_config in data.get("servers", []):
            server_name = server_config["name"]
            
            tools = {}
            for tool in server_config.get("tools", []):
                t_name = tool["name"]
                tools[t_name] = tool
                self.tool_to_server[t_name] = server_name
            
            self.servers[server_name] = {
                "config": server_config,
                "tools": tools,
                "connection": ServerConnection(server_name, server_config, self.auth_manager)
            }
        
        logging.info(f"Loaded {len(self.servers)} MCP servers with {len(self.tool_to_server)} total tools")

    def total_tools(self) -> int:
        """Return total number of tools across all servers"""
        return len(self.tool_to_server)
    
    def get_server_for_tool(self, tool_name: str) -> Optional[str]:
        """Find which server hosts the given tool (O(1) lookup)"""
        return self.tool_to_server.get(tool_name)

    async def _close_connection(self, server_name: str):
        """Close a specific server connection and clean up its resources"""
        server_info = self.servers[server_name]
        conn = server_info["connection"]
        logging.info(f"Closing connection to server: {server_name}")
        await conn.stop()
        
        if server_name in self._active_connections:
            self._active_connections.remove(server_name)

    async def connect_to_server(self, server_name: str) -> ClientSession:
        """Connect to an MCP server with LRU pool management"""
        server_info = self.servers[server_name]
        conn = server_info["connection"]
        
        # If already connected, move to end of LRU (most recently used)
        if conn.client:
            if server_name in self._active_connections:
                self._active_connections.remove(server_name)
            self._active_connections.append(server_name)
            return conn.client
        
        # Manage pool size: if we hit the limit, close the oldest connection
        while len(self._active_connections) >= Config.MAX_OPEN_CONNECTIONS:
            oldest_server = self._active_connections.pop(0)
            await self._close_connection(oldest_server)

        client = await conn.start()
        # Small grace period for server to fully initialize its session
        await asyncio.sleep(1.0)
        self._active_connections.append(server_name)
        return client

    async def warmup(self):
        """
        Robust Warmup:
        1. Staggered: Wait 500ms between each server to avoid CPU spikes.
        2. Readiness Probing: Verify the connection is actually responsive.
        """
        # Initial 1s delay to let the main process bind to its port
        await asyncio.sleep(1.0)
        
        limit = Config.WARMUP_LIMIT
        servers_to_warm = list(self.servers.keys())[:limit]
        
        if not servers_to_warm:
            return

        logging.info(f"Starting staggered warmup for {len(servers_to_warm)} servers...")
        
        for i, name in enumerate(servers_to_warm):
            try:
                # 1. Staggered delay (except for the first one)
                if i > 0:
                    await asyncio.sleep(0.5)
                
                # 2. Connect
                client = await self.connect_to_server(name)
                
                # 3. Readiness Probe: Send a simple request to ensure session is alive
                async with asyncio.timeout(5.0):
                    await client.list_tools()
                
                logging.info(f"  [✓] {name} is ready")
                
            except Exception as e:
                logging.warning(f"  [✗] Warmup/Probe failed for {name}: {e}")
                # We don't call _close_connection here to allow it to retry on the first real call
                
        logging.info(f"Warmup phase complete. {len(self._active_connections)} servers active.")

    async def cleanup(self):
        """Close all active server connections"""
        active_copy = list(self._active_connections)
        for name in active_copy:
            await self._close_connection(name)

    def _format_result(self, result: Any) -> str:
        """Helper to extract text from an MCP CallToolResult object."""
        try:
            if hasattr(result, 'content'):
                # Extract text from all content blocks that have it
                texts = [c.text for c in result.content if hasattr(c, 'text')]
                return "\n".join(texts) if texts else str(result)
            return str(result)
        except Exception:
            return str(result)
    
    async def execute_tool(self, tool_name: str, arguments: Dict[str, Any], auth_manager: AuthManager) -> str:
        server_name = self.get_server_for_tool(tool_name)
        if not server_name:
            raise ToolError(f"Tool '{tool_name}' not found in any configured MCP server")
        
        server_config = self.servers[server_name]["config"]
        
        # Check if auth is required but missing/stale
        if "auth" in server_config:
            token = auth_manager.get_token(server_name)
            if not token:
                token = await auth_manager.refresh_token(server_name, server_config)
            
            if not token:
                raise ToolError(f"Authentication required for '{server_name}'. Please run setup_server.py to authenticate.")

        try:
            # Stage 1: Get or create persistent connection
            client = await self.connect_to_server(server_name)
            
            # Stage 2: Call tool with a sensible timeout (e.g. 60s)
            try:
                async with asyncio.timeout(60.0):
                    result = await client.call_tool(tool_name, arguments=arguments)
                    
                    if hasattr(result, 'isError') and result.isError:
                        return f"❌ **Tool returned an error:**\n\n{self._format_result(result)}"
                        
                    return self._format_result(result)
            except asyncio.TimeoutError:
                raise ToolError(f"Tool execution timed out after 60s: {tool_name}")
            except Exception as e:
                # If the session is dead, we close it so it can be re-opened fresh next time
                logging.error(f"Error calling tool {tool_name} on {server_name}, closing connection: {e}")
                await self._close_connection(server_name)
                raise ToolError(f"Error calling tool on {server_name}: {str(e)}")
                
        except Exception as e:
            if isinstance(e, ToolError):
                raise
            logging.error(f"Error executing {tool_name}: {traceback.format_exc()}")
            raise ToolError(f"Failed to execute '{tool_name}': {str(e)}")


@dataclass
class AppContext:
    """Application context with shared resources"""
    searcher: HybridToolSearcher
    registry: MCPToolRegistry
    auth_manager: AuthManager


@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
    """Manage application lifecycle"""
    searcher = get_shared_searcher()
    auth_manager = get_auth_manager()
    config_path = Path(__file__).parent / "mcp_servers.json"
    registry = MCPToolRegistry(config_path, auth_manager)
    
    # Warm up in background to avoid blocking server start
    # but still get them ready as soon as possible
    asyncio.create_task(registry.warmup())
    
    yield AppContext(searcher=searcher, registry=registry, auth_manager=auth_manager)
    
    # Cleanup
    await registry.cleanup()


# ============= Create MCP Server =============

mcp_stdio = FastMCP(
    name="Ansam",
    instructions="Search and execute tools across multiple MCP servers",
    lifespan=app_lifespan
)

mcp_sse = FastMCP(
    name="Ansam",
    instructions="Search and execute tools across multiple MCP servers",
    lifespan=app_lifespan,
    port=8000,
    host="0.0.0.0"
)


# ============= Tool: Search for Tools (using hybrid search) =============

@mcp_stdio.tool(
    name="search_tools",
    description="Search for relevant MCP tools based on a short key descriptive words. Returns tool names, descriptions, and parameter schemas."
)
@mcp_sse.tool(
    name="search_tools",
    description="Search for relevant MCP tools based on a short key descriptive words. Returns tool names, descriptions, and parameter schemas."
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
        
        # Filter results based on authentication status
        authenticated_results = []
        app_ctx = ctx.request_context.lifespan_context
        for res in results:
            server_name = app_ctx.registry.get_server_for_tool(res['tool_name'])
            if server_name:
                server_config = app_ctx.registry.servers[server_name]["config"]
                if "auth" in server_config:
                    # Check if we have a token or can refresh
                    token = app_ctx.auth_manager.get_token(server_name)
                    if not token:
                        token = await app_ctx.auth_manager.refresh_token(server_name, server_config)
                    
                    if not token:
                        # Skip this tool, user hasn't authenticated yet
                        continue
            authenticated_results.append(res)
        
        results = authenticated_results
        
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
    arguments: Union[str, Dict[str, Any]],
    ctx: Context[ServerSession, AppContext]
) -> str:
    """
    Call a tool on its original MCP server.
    
    Args:
        tool_name: The exact unique identifier of the tool (e.g., "send-email")
        arguments: JSON string or dictionary of arguments to pass to the tool.
    """
    try:
        # Parse arguments if string, else use as-is
        if isinstance(arguments, str):
            try:
                args_dict = json.loads(arguments) if arguments else {}
            except json.JSONDecodeError as e:
                raise ToolError(f"Invalid JSON arguments: {str(e)}. Arguments must be a valid JSON string or a JSON object.")
        else:
            args_dict = arguments
        
        await ctx.report_progress(0.3, 1.0, f"Looking up server for '{tool_name}'...")
        
        # Execute the tool using registry
        app_ctx = ctx.request_context.lifespan_context
        registry = app_ctx.registry
        result = await registry.execute_tool(tool_name, args_dict, app_ctx.auth_manager)
        
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