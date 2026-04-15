#!/usr/bin/env python3
"""
MCP Server Registry Extractor
Extracts tool schemas from MCP servers and generates the mcp_servers.json config file
that the call_tool function uses to route tool executions.
"""

import asyncio
import json
import logging
import sys
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import argparse

# MCP imports
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.sse import sse_client

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stderr
)
logger = logging.getLogger(__name__)


class MCPRegistryExtractor:
    """Extracts tool information from MCP servers and builds registry config"""
    
    def __init__(self, timeout: int = 30, debug: bool = False):
        self.timeout = timeout
        self.debug = debug
        if debug:
            logger.setLevel(logging.DEBUG)
    
    def parse_tool_schema(self, tool) -> Dict[str, Any]:
        """Parse MCP tool into the format needed for registry"""
        # Extract parameters from inputSchema
        parameters = {}
        input_schema = tool.inputSchema if hasattr(tool, 'inputSchema') else {}
        
        if input_schema and 'properties' in input_schema:
            properties = input_schema.get('properties', {})
            required = input_schema.get('required', [])
            
            for param_name, param_info in properties.items():
                parameters[param_name] = {
                    "type": param_info.get('type', 'string'),
                    "description": param_info.get('description', ''),
                    "required": param_name in required
                }
        
        return {
            "name": tool.name,
            "description": tool.description or "",
            "parameters": parameters
        }
    
    async def extract_stdio_server(self, name: str, command: str, args: List[str] = None, env: Dict[str, str] = None) -> Dict[str, Any]:
        """Extract tools from a stdio MCP server"""
        logger.info(f"Connecting to stdio server '{name}': {command} {' '.join(args or [])}")
        
        try:
            server_params = StdioServerParameters(
                command=command,
                args=args or [],
                env=env or os.environ.copy()
            )
            
            async with stdio_client(server_params) as (read_stream, write_stream):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    
                    # List available tools
                    tools_result = await session.list_tools()
                    
                    tools = []
                    for tool in tools_result.tools:
                        tool_info = self.parse_tool_schema(tool)
                        tools.append(tool_info)
                    
                    server_config = {
                        "name": name,
                        "transport": "stdio",
                        "command": command,
                        "args": args or [],
                        "tools": tools
                    }
                    
                    if env:
                        server_config["env"] = env
                    
                    logger.info(f"Extracted {len(tools)} tools from {name}")
                    return server_config
                    
        except Exception as e:
            logger.error(f"Failed to extract from stdio server '{name}': {e}")
            if self.debug:
                logger.exception(e)
            return None
    
    async def extract_sse_server(self, name: str, url: str, headers: Dict[str, str] = None) -> Dict[str, Any]:
        """Extract tools from an SSE MCP server"""
        logger.info(f"Connecting to SSE server '{name}': {url}")
        
        try:
            async with sse_client(url) as (read_stream, write_stream):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    
                    # List available tools
                    tools_result = await session.list_tools()
                    
                    tools = []
                    for tool in tools_result.tools:
                        tool_info = self.parse_tool_schema(tool)
                        tools.append(tool_info)
                    
                    server_config = {
                        "name": name,
                        "transport": "sse",
                        "url": url,
                        "tools": tools
                    }
                    
                    if headers:
                        server_config["headers"] = headers
                    
                    logger.info(f"Extracted {len(tools)} tools from {name}")
                    return server_config
                    
        except Exception as e:
            logger.error(f"Failed to extract from SSE server '{name}': {e}")
            if self.debug:
                logger.exception(e)
            return None
    
    async def extract_http_server(self, name: str, url: str, headers: Dict[str, str] = None) -> Dict[str, Any]:
        """Extract tools from an HTTP MCP server"""
        logger.info(f"Connecting to HTTP server '{name}': {url}")
        
        try:
            # Try to use SSE client for HTTP as well (MCP typically uses SSE for HTTP)
            async with sse_client(url) as (read_stream, write_stream):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    
                    # List available tools
                    tools_result = await session.list_tools()
                    
                    tools = []
                    for tool in tools_result.tools:
                        tool_info = self.parse_tool_schema(tool)
                        tools.append(tool_info)
                    
                    server_config = {
                        "name": name,
                        "transport": "http",
                        "url": url,
                        "tools": tools
                    }
                    
                    if headers:
                        server_config["headers"] = headers
                    
                    logger.info(f"Extracted {len(tools)} tools from {name}")
                    return server_config
                    
        except Exception as e:
            logger.error(f"Failed to extract from HTTP server '{name}': {e}")
            if self.debug:
                logger.exception(e)
            return None
    
    async def extract_from_config_file(self, config_path: str) -> List[Dict[str, Any]]:
        """Extract tools from multiple servers defined in a config file"""
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        servers_config = []
        servers = config.get('servers', [])
        
        for server in servers:
            name = server.get('name')
            transport = server.get('transport')
            
            if not name or not transport:
                logger.warning(f"Skipping server with missing name or transport: {server}")
                continue
            
            result = None
            if transport == 'stdio':
                command = server.get('command')
                if not command:
                    logger.warning(f"Skipping stdio server '{name}': missing command")
                    continue
                result = await self.extract_stdio_server(
                    name=name,
                    command=command,
                    args=server.get('args', []),
                    env=server.get('env')
                )
            elif transport == 'sse':
                url = server.get('url')
                if not url:
                    logger.warning(f"Skipping SSE server '{name}': missing url")
                    continue
                result = await self.extract_sse_server(
                    name=name,
                    url=url,
                    headers=server.get('headers')
                )
            elif transport == 'http':
                url = server.get('url')
                if not url:
                    logger.warning(f"Skipping HTTP server '{name}': missing url")
                    continue
                result = await self.extract_http_server(
                    name=name,
                    url=url,
                    headers=server.get('headers')
                )
            else:
                logger.warning(f"Unknown transport '{transport}' for server '{name}'")
                continue
            
            if result:
                servers_config.append(result)
        
        return servers_config


def generate_mcp_servers_json(servers_config: List[Dict[str, Any]], output_file: str, merge: bool = False):
    """Generate the mcp_servers.json file for the call_tool function"""
    
    if merge and Path(output_file).exists():
        # Merge with existing config
        with open(output_file, 'r') as f:
            existing = json.load(f)
        existing_servers = {s['name']: s for s in existing.get('servers', [])}
        
        for new_server in servers_config:
            existing_servers[new_server['name']] = new_server
        
        final_config = {"servers": list(existing_servers.values())}
    else:
        final_config = {"servers": servers_config}
    
    with open(output_file, 'w') as f:
        json.dump(final_config, f, indent=2)
    
    total_tools = sum(len(s.get('tools', [])) for s in final_config['servers'])
    logger.info(f"Generated {output_file} with {len(final_config['servers'])} servers and {total_tools} tools")


async def main():
    parser = argparse.ArgumentParser(
        description="Extract MCP server tools and generate mcp_servers.json for call_tool function",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract from a single stdio server
  %(prog)s --name filesystem --transport stdio --command npx --args -y @modelcontextprotocol/server-filesystem /tmp
  
  # Extract from a single SSE server
  %(prog)s --name demo --transport sse --url http://localhost:8000/sse
  
  # Extract from multiple servers using a source config
  %(prog)s --source-config servers.json --output mcp_servers.json
  
  # Merge with existing mcp_servers.json
  %(prog)s --source-config new_server.json --output mcp_servers.json --merge
        """
    )
    
    # Single server mode
    parser.add_argument("--name", help="Server name for single server extraction")
    parser.add_argument("--transport", choices=["stdio", "sse", "http"], help="Transport type")
    parser.add_argument("--command", help="Command for stdio transport")
    parser.add_argument("--args", nargs="*", help="Arguments for stdio command")
    parser.add_argument("--url", help="URL for SSE/HTTP transport")
    parser.add_argument("--env", nargs="*", help="Environment variables for stdio (KEY=value format)")
    
    # Batch mode
    parser.add_argument("--source-config", help="Path to source servers.json config file (NOT the output)")
    
    # Output options
    parser.add_argument("--output", default="mcp_servers.json", help="Output file path (default: mcp_servers.json)")
    parser.add_argument("--merge", action="store_true", help="Merge with existing output file instead of overwriting")
    parser.add_argument("--timeout", type=int, default=30, help="Connection timeout in seconds")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("-H", "--header", action="append", help='Custom header, e.g. "Header: value". Can be used multiple times.')
    
    args = parser.parse_args()
    
    # Set debug logging
    if args.debug:
        logger.setLevel(logging.DEBUG)
    
    extractor = MCPRegistryExtractor(timeout=args.timeout, debug=args.debug)
    
    # Parse headers if provided
    headers = {}
    if args.header:
        for header in args.header:
            if ':' in header:
                key, value = header.split(':', 1)
                headers[key.strip()] = value.strip()
            else:
                logger.warning(f"Ignoring malformed header: {header}")
    
    # Parse env vars if provided
    env_vars = {}
    if args.env:
        for env in args.env:
            if '=' in env:
                key, value = env.split('=', 1)
                env_vars[key] = value
            else:
                logger.warning(f"Ignoring malformed env var: {env}")
    
    servers_config = []
    
    # Extract from source config or single server
    if args.source_config:
        # Batch mode - read source config and extract from each server
        logger.info(f"Extracting from source config: {args.source_config}")
        servers_config = await extractor.extract_from_config_file(args.source_config)
        
    elif args.name and args.transport:
        # Single server mode
        logger.info(f"Extracting from single server: {args.name}")
        
        result = None
        if args.transport == "stdio":
            if not args.command:
                logger.error("stdio transport requires --command")
                sys.exit(1)
            result = await extractor.extract_stdio_server(
                name=args.name,
                command=args.command,
                args=args.args or [],
                env=env_vars if env_vars else None
            )
        elif args.transport == "sse":
            if not args.url:
                logger.error("sse transport requires --url")
                sys.exit(1)
            result = await extractor.extract_sse_server(
                name=args.name,
                url=args.url,
                headers=headers if headers else None
            )
        elif args.transport == "http":
            if not args.url:
                logger.error("http transport requires --url")
                sys.exit(1)
            result = await extractor.extract_http_server(
                name=args.name,
                url=args.url,
                headers=headers if headers else None
            )
        
        if result:
            servers_config.append(result)
        else:
            logger.error(f"Failed to extract from server '{args.name}'")
            sys.exit(1)
    else:
        parser.print_help()
        logger.error("\nError: Either provide --source-config OR (--name and --transport)")
        sys.exit(1)
    
    # Generate output file
    if servers_config:
        generate_mcp_servers_json(servers_config, args.output, args.merge)
        
        # Print summary
        print(f"\n✅ Successfully generated {args.output}", file=sys.stderr)
        print(f"\n📊 Summary:", file=sys.stderr)
        for server in servers_config:
            print(f"  - {server['name']} ({server['transport']}): {len(server.get('tools', []))} tools", file=sys.stderr)
        
        print(f"\n💡 This file can now be used by your call_tool function!", file=sys.stderr)
    else:
        logger.error("No servers extracted")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())