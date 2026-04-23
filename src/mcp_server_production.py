import asyncio
from collections import OrderedDict
from typing import Dict, Optional
from fastmcp.server import create_proxy
import json
from fastmcp import FastMCP, Context
from fastmcp.dependencies import Progress
from fastmcp.exceptions import ToolError 
from fastmcp.server.lifespan import lifespan
from pathlib import Path
import logging
from fastmcp.client.transports.stdio import StdioTransport
import sys
import argparse

from auth_manager import AuthManager
from hybrid_searcher import HybridToolSearcher
from rate_limiter import RateLimiterManager
from circuit_breaker import CircuitBreakerManager, CircuitOpenError
from config import Config


logger = logging.getLogger(__name__)



# Module-level singleton for hybrid searcher and auth manager
searcher: HybridToolSearcher = HybridToolSearcher()
auth_manager: AuthManager = AuthManager()

# Rate limiters
global_rate_limiter = RateLimiterManager(rate=Config.GLOBAL_RATE, capacity=Config.GLOBAL_CAPACITY)
server_rate_limiter = RateLimiterManager(rate=Config.PER_SERVER_RATE, capacity=Config.PER_SERVER_CAPACITY)

# Circuit breaker manager
cb_manager = CircuitBreakerManager(
    failure_threshold=Config.CB_FAILURE_THRESHOLD,
    recovery_timeout=Config.CB_RECOVERY_TIMEOUT,
    half_open_success=Config.CB_HALF_OPEN_SUCCESS
)

tool_to_server_map = {}

# Note: server configs are loaded asynchronously in `app_lifespan`



class ConcurrentLRUCache:
    """Async-safe LRU cache for managing server connections using asyncio.Lock.

    Methods are async and must be awaited. The internal lock is created
    lazily to avoid requiring a running event loop at import time.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = OrderedDict()
        self._lock = None  # type: Optional[asyncio.Lock]

    async def _ensure_lock(self):
        if self._lock is None:
            self._lock = asyncio.Lock()

    async def get(self, key):
        await self._ensure_lock()
        async with self._lock:
            if key not in self.cache:
                return None
            # Move to end to mark as most recently used
            self.cache.move_to_end(key)
            return self.cache[key]

    async def put(self, key, value):
        await self._ensure_lock()
        async with self._lock:
            if key in self.cache:
                self.cache.move_to_end(key)
            self.cache[key] = value
            if len(self.cache) > self.capacity:
                # Remove the LRU item (first item)
                evicted_key, evicted_value = self.cache.popitem(last=False)
                logger.info(f"Evicted server '{evicted_key}' from cache (capacity={self.capacity})")
                return evicted_key, evicted_value
        return None, None

    async def remove(self, key):
        await self._ensure_lock()
        async with self._lock:
            if key in self.cache:
                del self.cache[key]
                logger.debug(f"Removed server '{key}' from cache")

    async def size(self):
        await self._ensure_lock()
        async with self._lock:
            return len(self.cache)

    async def keys(self):
        await self._ensure_lock()
        async with self._lock:
            return list(self.cache.keys())

    async def popitem(self, last: bool = False):
        """Async-safe popitem compatible with OrderedDict.popitem.

        If last is False, pop the least-recently-used item (first item).
        Returns a tuple (key, value).
        """
        await self._ensure_lock()
        async with self._lock:
            if not self.cache:
                raise KeyError("popitem(): cache is empty")
            key, value = self.cache.popitem(last=last)
            logger.info(f"Evicted server '{key}' from cache via popitem()")
            return key, value


class DynamicProxyManager:
    """Manages MCP server proxies with LRU caching and lazy loading"""
    
    def __init__(self, max_connections: int = 50, default_servers: list = None):
        """
        Args:
            max_connections: Maximum number of concurrent server connections
            default_servers: List of server configs to pre-initialize (hot tier)
        """
        self.max_connections = max_connections
        self.cache = ConcurrentLRUCache(max_connections)
        self.server_configs: Dict[str, dict] = {}
        self._cleanup_tasks: Dict[str, asyncio.Task] = {}
        self._lock = asyncio.Lock()
        
        # Initialize default servers (hot tier)
        if default_servers:
            asyncio.create_task(self._initialize_default_servers(default_servers))
    
    async def _initialize_default_servers(self, default_servers: list):
        """Pre-initialize critical servers (hot tier)"""
        logger.info(f"Pre-initializing {len(default_servers)} default servers...")
        
        for server_config in default_servers:
            name = server_config["name"]
            try:
                proxy = await self._create_proxy(server_config)
                await self.cache.put(name, proxy)
                logger.info(f"✓ Default server '{name}' initialized")
            except Exception as e:
                logger.error(f"✗ Failed to initialize default server '{name}': {e}")
    
    def register_server(self, config: dict):
        """Register a server configuration (doesn't connect yet)"""
        self.server_configs[config["name"]] = config
        logger.debug(f"Registered server '{config['name']}' (total: {len(self.server_configs)})")
    
    async def get_proxy(self, server_name: str) -> Optional[FastMCP]:
        """
        Get or create a proxy for the server.
        Implements lazy loading - only connects when first requested.
        """
        # Check cache first
        cached = await self.cache.get(server_name)
        if cached:
            logger.debug(f"Cache hit for '{server_name}'")
            return cached
        
        # Need to create new connection
        async with self._lock:
            # Double-check after acquiring lock (could have been created while waiting)
            cached = await self.cache.get(server_name)
            if cached:
                return cached
            
            # Check if server is registered
            if server_name not in self.server_configs:
                logger.warning(f"Server '{server_name}' not registered")
                return None
            
            # Create new proxy
            config = self.server_configs[server_name]
            size = await self.cache.size()
            logger.info(f"Creating new proxy for '{server_name}' (active: {size}/{self.max_connections})")
            
            try:
                proxy = await self._create_proxy(config)

                # Put in cache - this may evict an old connection
                evicted_key, evicted_proxy = await self.cache.put(server_name, proxy)
                
                # Cleanup evicted connection if any
                if evicted_key:
                    await self._cleanup_proxy(evicted_key, evicted_proxy)
                
                return proxy
                
            except Exception as e:
                logger.error(f"Failed to create proxy for '{server_name}': {e}")
                return None
    
    async def _create_proxy(self, config: dict) -> FastMCP:
        """Create a proxy for a single server"""
        transport_type = config.get("transport", "http")
        name = config["name"]
        
        try:
            if transport_type == "http":
                # Simple URL-based proxy (recommended)
                proxy = create_proxy(
                    config["url"],
                    name=name
                )
            elif transport_type == "stdio":
                # Use command if available (executable), fallback to target
                executable = config.get("command", config.get("target"))
                args = config.get("args", [])
                env = config.get("env")
                
                # Create explicit stdio transport to avoid inference issues with commands like 'uv'
                stdio_transport = StdioTransport(
                    command=executable,
                    args=args,
                    env=env
                )
                
                proxy = create_proxy(
                    stdio_transport,
                    name=name
                )
                logger.info(f"Creating proxy for '{name}' with transport 'stdio', executable: {executable}, args: {args}")
            else:
                raise ValueError(f"Unknown transport: {transport_type}")
            
            # Initialize the proxy (warm it up)
            # Note: create_proxy returns a server that needs to be "mounted" or accessed
            # For FastMCP, we need to ensure it's ready
            if hasattr(proxy, 'initialize'):
                await proxy.initialize()
            
            return proxy
            
        except Exception as e:
            logger.error(f"Error creating proxy for '{name}': {e}")
            raise
    
    async def _cleanup_proxy(self, server_name: str, proxy: FastMCP):
        """Properly cleanup a proxy connection"""
        logger.info(f"Cleaning up proxy for '{server_name}'")
        try:
            # Attempt graceful shutdown
            if hasattr(proxy, 'close'):
                await proxy.close()
            elif hasattr(proxy, 'cleanup'):
                await proxy.cleanup()
        except Exception as e:
            logger.warning(f"Error during cleanup of '{server_name}': {e}")
    
    async def warmup_popular_servers(self, popularity_list: list, limit: int = 10):
        """
        Pre-warm the most popular servers based on historical usage
        
        Args:
            popularity_list: List of server names in order of popularity
            limit: Maximum number to warm up
        """
        logger.info(f"Warming up top {limit} popular servers...")
        tasks = []
        for server_name in popularity_list[:limit]:
            if server_name in self.server_configs:
                tasks.append(self.get_proxy(server_name))
        
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            success_count = sum(1 for r in results if r and not isinstance(r, Exception))
            logger.info(f"Warmed up {success_count}/{len(tasks)} servers")
    
    async def get_active_servers(self) -> list:
        """Get list of currently active server names"""
        return await self.cache.keys()
    
    async def cleanup_all(self):
        """Cleanup all active connections"""
        total = await self.cache.size()
        logger.info(f"Cleaning up all {total} active connections...")

        keys = await self.cache.keys()
        for server_name in keys:
            proxy = await self.cache.get(server_name)
            if proxy:
                await self._cleanup_proxy(server_name, proxy)

        # Clear cache
        while await self.cache.size() > 0:
            await self.cache.popitem(last=False)


# ============= Integration with FastMCP =============

# Load your 1000 server configs
def load_all_server_configs() -> list:
    """Load your 1000 server configurations from JSON/database"""
    # Example: load from JSON file
    
    config_path = Path(__file__).parent / "mcp_servers.json"
    with open(config_path) as f:
        data = json.load(f)
    
    # You can have 1000+ servers here
    return data.get("servers", [])


# Create the dynamic manager (configs will be loaded asynchronously in app_lifespan)
proxy_manager = DynamicProxyManager(
    max_connections=50,
    default_servers=None
)


# ============= FastMCP Server Setup =============

@lifespan
async def app_lifespan(server: FastMCP):
    """Application lifespan with background maintenance"""    
    # Load server configurations off the event loop to avoid blocking
    try:
        server_configs = await asyncio.to_thread(load_all_server_configs)
    except Exception as e:
        logger.error(f"Failed to load server configs: {e}")
        server_configs = []

    # Build tool->server map and register server configs
    all_tools = []
    for cfg in server_configs:
        name = cfg.get("name")
        for tool in cfg.get("tools", []):
            tool_to_server_map[tool.get("name")] = name
            all_tools.append(tool)
        proxy_manager.register_server(cfg)

    # Index searcher with all tools found in config
    # This avoids the Qdrant scroll bottleneck on startup
    if all_tools:
        logger.info(f"Indexing {len(all_tools)} tools into hybrid searcher...")
        await asyncio.to_thread(searcher.index, all_tools)

    # Initialize hot-tier servers (run synchronously here to ensure availability)
    hot_tier_servers = [c for c in server_configs if c.get("priority") == "high"][:10]
    if hot_tier_servers:
        await proxy_manager._initialize_default_servers(hot_tier_servers)

    # Start background task to log cache stats periodically
    async def log_stats():
        while True:
            await asyncio.sleep(60)  # Every minute
            active = await proxy_manager.get_active_servers()
            logger.info(f"Active connections: {len(active)}/{proxy_manager.max_connections}")
    
    stats_task = asyncio.create_task(log_stats())
    
    try:
        yield {"proxy_manager": proxy_manager, "searcher": searcher, "tool_to_server": tool_to_server_map, "auth_manager": auth_manager}
    finally:
        stats_task.cancel()
        await proxy_manager.cleanup_all()


# Create main FastMCP server
mcp = FastMCP(
    name="Ansam-Scaled",
    instructions="Dynamic proxy manager for 1000+ MCP servers",
    lifespan=app_lifespan
)







# ------------
@mcp.tool(task=True)
async def search_tools(
    query: str,
    ctx: Context,
    progress: Progress = Progress()
) -> str:   
    """Search for tools using hybrid retrieval."""
    try:
        # Global Rate Limiting
        if not await global_rate_limiter.try_acquire("global"):
            return "Too many requests. Please try again later."
        
        await progress.set_total(4)  # 4 main steps
        await progress.set_message("Initializing search...")
        await progress.increment()
        
        
        # Perform hybrid search
        await progress.set_message("Searching for relevant tools...")
        results = await asyncio.to_thread(searcher.search, query)
        await progress.increment()
        
        # Filter results based on authentication status
        await progress.set_message("Checking authentication status...")
        authenticated_results = []
        for res in results:
            server_name = get_server_by_tool(res['tool_name'])
            if server_name is None:
                logger.debug(f"No server mapping for tool '{res.get('tool_name')}', skipping result")
                continue

            server_config = proxy_manager.server_configs.get(server_name, {})
            if "auth" in server_config:
                # Check if we have a token or can refresh
                token = auth_manager.get_token(server_name)
                if not token:
                    token = await auth_manager.refresh_token(server_name, server_config)

                if not token:
                    # Skip this tool, user hasn't authenticated yet
                    continue

            authenticated_results.append(res)
        
        results = authenticated_results
        
        await ctx.report_progress(0.8, 1.0, "Formatting results...")
        
        if not results:
            return "No tools found matching your query."
        
        output = ""
        
        for i, result in enumerate(results, 1):
            tool_schema = result.get('tool_schema', {})
            tool_name = tool_schema.get('name', result.get('tool_name', 'unknown'))
            tool_description = tool_schema.get('description', result.get('tool_description', 'No description'))
            input_schema = tool_schema.get('inputSchema', {})
            relevance_score = result.get('relevance_score', 1.0)
            
            output += f"{i}. **Tool Name:** `{tool_name}`\n"
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




















def get_server_by_tool(tool_name: str) -> str:
    """
    Retrieves the server name in O(1) time using the pre-processed map.
    """
    return tool_to_server_map.get(tool_name)


@mcp.tool(task=True)
async def call_tool(
    tool_name: str,
    arguments: dict,
    ctx: Context
):
    """
    Call a tool on any registered server.
    Server is loaded on-demand and cached via LRU.
    """
    
    # Get server mapping
    server_name = get_server_by_tool(tool_name)
    if server_name is None:
        return f"Tool '{tool_name}' is not registered to any server"

    # Global Rate Limiting
    if not await global_rate_limiter.try_acquire("global"):
        return "Too many requests. Please try again later."

    # Per-Server Rate Limiting
    if not await server_rate_limiter.try_acquire(server_name):
        return f"Too many requests for server '{server_name}'. Please try again later."

    ctx.info(f"Received request to call tool '{tool_name}' on server '{server_name}' with arguments: {arguments}")

    async def _execute_tool_call():
        proxy = await proxy_manager.get_proxy(server_name)
        if not proxy:
            raise Exception(f"Server '{server_name}' not found or failed to connect")
        return await proxy.call_tool(tool_name, arguments)
    
    try:
        # Call the tool with circuit breaker covering both connection and execution
        result = await cb_manager.call(server_name, _execute_tool_call)
        return result
        
    except CircuitOpenError as e:
        return f"Circuit breaker is OPEN for server '{server_name}': {e}"
    except Exception as e:
        return f"Error calling {tool_name} on {server_name}: {e}"



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the MCP server")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--sse", action="store_true", help="Run with SSE transport")
    group.add_argument("--http", action="store_true", help="Run with HTTP transport")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--transport", choices=["stdio", "sse", "http"],
                        help="Explicit transport to use (overrides flags)")

    args = parser.parse_args()

    if args.transport == "sse" or args.sse:
        mcp.run(transport="sse", host=args.host, port=args.port)
        logger.info("Starting MCP server with SSE transport")
    elif args.transport == "http" or args.http:
        mcp.run(transport="http", host=args.host, port=args.port)
        logger.info("Starting MCP server with HTTP transport")
    else:
        logger.info("Starting MCP server with stdio transport")
        mcp.run(transport="stdio")