import asyncio
import threading
from collections import OrderedDict
from typing import Dict, Optional, Any
from fastmcp import FastMCP
from fastmcp.server import create_proxy
from fastmcp import FastMCP, Context
from fastmcp.dependencies import Progress
from fastmcp.server.lifespan import lifespan
import json
from pathlib import Path
import logging
from fastmcp.client.transports.stdio import StdioTransport
import sys

from auth_manager import AuthManager
from hybrid_searcher import HybridToolSearcher



logger = logging.getLogger(__name__)



# Module-level singleton for hybrid searcher and auth manager
searcher: HybridToolSearcher = HybridToolSearcher()
auth_manager: AuthManager = AuthManager()
tool_to_server_map = {}

config_path = Path(__file__).parent / "mcp_servers.json"
with open(config_path) as f:
    data = json.load(f)

for server in data.get("servers", []):
    server_name = server["name"]
    for tool in server.get("tools", []):
        tool_to_server_map[tool["name"]] = server_name


class ConcurrentLRUCache:
    """Thread-safe LRU cache for managing server connections"""
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = OrderedDict()
        self.lock = threading.Lock()

    def get(self, key):
        with self.lock:
            if key not in self.cache:
                return None
            # Move to end to mark as most recently used
            self.cache.move_to_end(key)
            return self.cache[key]

    def put(self, key, value):
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
            self.cache[key] = value
            if len(self.cache) > self.capacity:
                # Remove the LRU item (first item)
                evicted_key, evicted_value = self.cache.popitem(last=False)
                logger.info(f"Evicted server '{evicted_key}' from cache (capacity={self.capacity})")
                return evicted_key, evicted_value
        return None, None

    def remove(self, key):
        with self.lock:
            if key in self.cache:
                del self.cache[key]
                logger.debug(f"Removed server '{key}' from cache")

    def size(self):
        with self.lock:
            return len(self.cache)

    def keys(self):
        with self.lock:
            return list(self.cache.keys())


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
                self.cache.put(name, proxy)
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
        cached = self.cache.get(server_name)
        if cached:
            logger.debug(f"Cache hit for '{server_name}'")
            return cached
        
        # Need to create new connection
        async with self._lock:
            # Double-check after acquiring lock (could have been created while waiting)
            cached = self.cache.get(server_name)
            if cached:
                return cached
            
            # Check if server is registered
            if server_name not in self.server_configs:
                logger.warning(f"Server '{server_name}' not registered")
                return None
            
            # Create new proxy
            config = self.server_configs[server_name]
            logger.info(f"Creating new proxy for '{server_name}' (active: {self.cache.size()}/{self.max_connections})")
            
            try:
                proxy = await self._create_proxy(config)
                
                # Put in cache - this may evict an old connection
                evicted_key, evicted_proxy = self.cache.put(server_name, proxy)
                
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
        return self.cache.keys()
    
    async def cleanup_all(self):
        """Cleanup all active connections"""
        logger.info(f"Cleaning up all {self.cache.size()} active connections...")
        for server_name in self.cache.keys():
            proxy = self.cache.get(server_name)
            if proxy:
                await self._cleanup_proxy(server_name, proxy)
        # Clear cache
        while self.cache.size() > 0:
            self.cache.popitem(last=False)


# ============= Integration with FastMCP =============

# Load your 1000 server configs
def load_all_server_configs() -> list:
    """Load your 1000 server configurations from JSON/database"""
    # Example: load from JSON file
    import json
    from pathlib import Path
    
    config_path = Path(__file__).parent / "mcp_servers.json"
    with open(config_path) as f:
        data = json.load(f)
    
    # You can have 1000+ servers here
    return data.get("servers", [])


# Create the dynamic manager
server_configs = load_all_server_configs()

# Define your hot tier (servers that should always stay connected)
hot_tier_servers = [
    cfg for cfg in server_configs 
    if cfg.get("priority") == "high"  # Mark high-priority servers in your config
][:10]  # Keep top 10 always connected

# Initialize the manager
proxy_manager = DynamicProxyManager(
    max_connections=50,  # Keep 50 servers active at once
    default_servers=hot_tier_servers  # Pre-initialize these
)

# Register all 1000 servers (configs only, no connections yet)
for config in server_configs:
    proxy_manager.register_server(config)


# ============= FastMCP Server Setup =============

@lifespan
async def app_lifespan(server: FastMCP):
    """Application lifespan with background maintenance"""    
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
            if server_name:
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
    return tool_to_server_map.get(tool_name, "Tool not found")


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
    
    # Get or create proxy for this server
    server_name = get_server_by_tool(tool_name)
    
    ctx.info(f"Received request to call tool '{tool_name}' on server '{server_name}' with arguments: {arguments}")

    proxy = await proxy_manager.get_proxy(server_name)
    
    # return f"Calling '{tool_name}' on server '{server_name}' with arguments: {arguments}"

    if not proxy:
        return f"Server '{server_name}' not found or failed to connect"
    
    try:
        # Call the tool on the proxy
        result = await proxy.call_tool(tool_name, arguments)
        return result
        
    except Exception as e:
        return f"Error calling {tool_name} on {server_name}: {e}"



if __name__ == "__main__":    
    match sys.argv[1]:
        case "--sse":
            mcp.run(transport="sse", host="0.0.0.0", port=8000)
            logger.info("Starting MCP server with HTTP transport (SSE)")
        case "--http":
            mcp.run(transport="http", host="0.0.0.0", port=8000)
        case _:
            logger.info("Starting MCP server with stdio transport")
            mcp.run(transport="stdio")