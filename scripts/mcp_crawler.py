import asyncio
import json
import logging
import argparse
from pathlib import Path
from contextlib import AsyncExitStack
from typing import List, Dict, Any
import httpx

# Official 2026 MCP SDK imports
from mcp import ClientSession
from mcp.client.sse import sse_client

logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("mcp_crawler")


async def fetch_from_sse_server(name: str, url: str, timeout: int = 30) -> List[Dict[str, Any]]:
    """Connect to an SSE MCP server and fetch its tools."""
    tools = []
    
    try:
        logger.info(f"🌍 [SSE] Connecting to {name} at {url}")
        
        async with asyncio.timeout(timeout):
            async with AsyncExitStack() as stack:
                # Create SSE client connection
                logger.debug(f"Establishing SSE connection to {url}")
                sse_transport = await stack.enter_async_context(sse_client(url))
                
                # sse_client returns (read_stream, write_stream) tuple
                read_stream, write_stream = sse_transport
                
                logger.debug(f"Creating client session")
                async with ClientSession(read_stream, write_stream) as session:
                    # Initialize the session
                    logger.debug(f"Initializing session")
                    await session.initialize()
                    
                    # List all available tools
                    logger.debug(f"Requesting tool list")
                    response = await session.list_tools()
                    
                    logger.info(f"Found {len(response.tools)} tools from {name}")
                    
                    # Process each tool
                    for tool in response.tools:
                        tool_dict = tool.model_dump()
                        tool_dict["server_origin"] = name
                        tool_dict["transport_used"] = "sse"
                        tools.append(tool_dict)
                        
                        logger.debug(f"  - {tool.name}: {tool.description[:50]}...")
                    
    except asyncio.TimeoutError:
        logger.error(f"⏰ Timeout connecting to '{name}' after {timeout} seconds")
    except httpx.ConnectError as e:
        logger.error(f"🔌 Connection error for '{name}': {e}")
    except Exception as e:
        logger.error(f"❌ Failed '{name}': {str(e)}", exc_info=True)
    
    return tools


async def fetch_from_stdio_server(name: str, command: str, args: List[str], env: Dict = None, timeout: int = 30) -> List[Dict[str, Any]]:
    """Connect to a stdio MCP server and fetch its tools."""
    from mcp.client.stdio import stdio_client
    from mcp import StdioServerParameters
    
    tools = []
    
    try:
        logger.info(f"💻 [Stdio] Connecting to {name}: {command} {' '.join(args)}")
        
        async with asyncio.timeout(timeout):
            async with AsyncExitStack() as stack:
                server_params = StdioServerParameters(
                    command=command,
                    args=args,
                    env=env or {}
                )
                
                stdio_transport = await stack.enter_async_context(stdio_client(server_params))
                read_stream, write_stream = stdio_transport
                
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    response = await session.list_tools()
                    
                    logger.info(f"Found {len(response.tools)} tools from {name}")
                    
                    for tool in response.tools:
                        tool_dict = tool.model_dump()
                        tool_dict["server_origin"] = name
                        tool_dict["transport_used"] = "stdio"
                        tools.append(tool_dict)
                        
    except asyncio.TimeoutError:
        logger.error(f"⏰ Timeout connecting to '{name}' after {timeout} seconds")
    except Exception as e:
        logger.error(f"❌ Failed '{name}': {str(e)}", exc_info=True)
    
    return tools


async def fetch_from_server(name: str, config: Dict[str, Any], timeout: int = 30) -> List[Dict[str, Any]]:
    """Route to appropriate connection handler based on transport type."""
    transport_type = config.get("transport", "stdio").lower()
    
    if transport_type == "sse":
        url = config.get("url")
        if not url:
            logger.error(f"No URL provided for SSE server '{name}'")
            return []
        return await fetch_from_sse_server(name, url, timeout)
    
    elif transport_type in ["http", "streamable-http"]:
        # For HTTP transport (if needed)
        from mcp.client.streamable_http import streamable_http_client
        url = config.get("url")
        if not url:
            logger.error(f"No URL provided for HTTP server '{name}'")
            return []
        
        try:
            async with asyncio.timeout(timeout):
                async with AsyncExitStack() as stack:
                    transport = await stack.enter_async_context(streamable_http_client(url))
                    read_stream, write_stream = transport
                    
                    async with ClientSession(read_stream, write_stream) as session:
                        await session.initialize()
                        response = await session.list_tools()
                        
                        tools = []
                        for tool in response.tools:
                            tool_dict = tool.model_dump()
                            tool_dict["server_origin"] = name
                            tool_dict["transport_used"] = transport_type
                            tools.append(tool_dict)
                        
                        logger.info(f"Found {len(tools)} tools from {name}")
                        return tools
        except Exception as e:
            logger.error(f"❌ Failed HTTP server '{name}': {e}")
            return []
    
    else:  # stdio
        command = config.get("command")
        if not command:
            logger.error(f"No command provided for stdio server '{name}'")
            return []
        args = config.get("args", [])
        env = config.get("env")
        return await fetch_from_stdio_server(name, command, args, env, timeout)


async def main():
    parser = argparse.ArgumentParser(description="MCP Crawler - Discover tools from MCP servers")
    
    # Single server mode
    parser.add_argument("--transport", choices=["stdio", "sse", "http"], help="Transport type")
    parser.add_argument("--command", help="Command for stdio transport")
    parser.add_argument("--args", nargs="*", help="Arguments for stdio command")
    parser.add_argument("--url", help="URL for SSE/HTTP transport")
    
    # Batch mode
    parser.add_argument("--config", help="Path to servers.json config file")
    
    # Output options
    parser.add_argument("--output", default="registry_dump.json", help="Output file path")
    parser.add_argument("--timeout", type=int, default=30, help="Connection timeout in seconds")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Build server configurations
    servers_to_crawl = {}
    
    if args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            logger.error(f"Config file not found: {args.config}")
            return
        
        with open(args.config, "r") as f:
            servers_to_crawl = json.load(f)
            
    elif args.transport:
        cfg = {"transport": args.transport}
        
        if args.transport in ["sse", "http"]:
            if not args.url:
                logger.error("--url is required for SSE/HTTP transport")
                return
            cfg["url"] = args.url
        else:  # stdio
            if not args.command:
                logger.error("--command is required for stdio transport")
                return
            cfg["command"] = args.command
            cfg["args"] = args.args or []
        
        servers_to_crawl = {"single_server": cfg}
    else:
        # Default: try to connect to local SSE server
        logger.info("No config provided, attempting to connect to default local SSE server...")
        servers_to_crawl = {
            "local_sse": {
                "transport": "sse",
                "url": "http://localhost:8000"
            }
        }
    
    # Validate configurations
    for name, cfg in servers_to_crawl.items():
        transport = cfg.get("transport", "stdio")
        if transport in ["sse", "http"] and "url" not in cfg:
            logger.error(f"Server '{name}' uses {transport} but missing 'url' field")
            return
        if transport == "stdio" and "command" not in cfg:
            logger.error(f"Server '{name}' uses stdio but missing 'command' field")
            return
    
    # Crawl servers (one at a time to avoid overloading)
    logger.info(f"Starting crawl of {len(servers_to_crawl)} server(s)...")
    all_tools = []
    
    for name, cfg in servers_to_crawl.items():
        logger.info(f"\n{'='*50}")
        logger.info(f"Processing server: {name}")
        logger.info(f"{'='*50}")
        
        tools = await fetch_from_server(name, cfg, args.timeout)
        all_tools.extend(tools)
        
        # Small delay between servers
        await asyncio.sleep(1)
    
    # Save results
    output_path = Path(args.output)
    with open(output_path, "w") as f:
        json.dump(all_tools, f, indent=2)
    
    logger.info(f"\n{'='*50}")
    logger.info(f"🎉 Crawl finished!")
    logger.info(f"📊 Total tools discovered: {len(all_tools)}")
    logger.info(f"💾 Results saved to: {output_path}")
    logger.info(f"{'='*50}")
    
    if len(all_tools) == 0:
        logger.warning("\n⚠️  No tools were discovered. Possible issues:")
        logger.warning("  1. Server not running (check with: curl http://localhost:8000)")
        logger.warning("  2. Wrong URL (try http://localhost:8000 or http://127.0.0.1:8000)")
        logger.warning("  3. Server not properly implementing MCP SSE protocol")
        logger.warning("  4. Firewall blocking the connection")


if __name__ == "__main__":
    asyncio.run(main())