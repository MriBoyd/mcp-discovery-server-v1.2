#!/usr/bin/env python3
"""
MCP Registry Manager
Merged tool for discovering, extracting, and managing MCP tool registries.
Combines functionality from mcp_crawler.py and extract_registry.py.
"""

import asyncio
import json
import logging
import sys
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import argparse
import shlex
import httpx
import anyio

# MCP imports
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.sse import sse_client
from mcp.shared._httpx_utils import create_mcp_http_client

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stderr
)
logger = logging.getLogger(__name__)

class MCPRegistryManager:
    """Manages MCP tool extraction and registry maintenance"""
    
    def __init__(self, timeout: int = 30, debug: bool = False):
        self.timeout = timeout
        self.debug = debug
        if debug:
            logger.setLevel(logging.DEBUG)

    def parse_header_entries(self, header_entries: List[str]) -> Dict[str, str]:
        """Parse list of header strings like 'Key: value' into a dict."""
        headers: Dict[str, str] = {}
        if not header_entries:
            return headers
        for entry in header_entries:
            if not entry or ":" not in entry:
                continue
            k, v = entry.split(":", 1)
            headers[k.strip()] = v.strip()
        return headers

    def parse_tool_for_registry(self, tool, server_name: str, transport: str) -> Dict[str, Any]:
        """Convert MCP tool into flat registry format (mcp_crawler style)"""
        tool_dict = tool.model_dump()
        tool_dict["server_origin"] = server_name
        tool_dict["transport_used"] = transport
        
        return tool_dict

    def parse_tool_for_config(self, tool) -> Dict[str, Any]:
        """Convert MCP tool into server config format (extract_registry style)"""
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
            "parameters": parameters,
            "inputSchema": input_schema # Keep original for compatibility
        }

    async def extract_from_server(self, name: str, config: Dict[str, Any]) -> List[Any]:
        """Universal extraction from any MCP server type"""
        transport_type = config.get("transport", "stdio").lower()
        tools = []
        
        try:
            async with asyncio.timeout(self.timeout):
                if transport_type == "stdio":
                    command = config.get("command")
                    args = config.get("args", [])
                    env = config.get("env") or os.environ.copy()
                    
                    server_params = StdioServerParameters(command=command, args=args, env=env)
                    async with stdio_client(server_params) as (read, write):
                        async with ClientSession(read, write) as session:
                            await session.initialize()
                            response = await session.list_tools()
                            tools = response.tools
                            
                elif transport_type in ["sse", "http"]:
                    url = config.get("url")
                    headers = config.get("headers")
                    
                    async with AsyncExitStack() as stack:
                        if transport_type == "http":
                            # Use streamable_http_client if explicit, else sse
                            from mcp.client.streamable_http import streamable_http_client
                            http_client = await stack.enter_async_context(create_mcp_http_client(headers)) if headers else None
                            transport = await stack.enter_async_context(streamable_http_client(url, http_client=http_client))
                            read, write, _ = transport
                        else:
                            # SSE
                            read, write = await stack.enter_async_context(sse_client(url, headers=headers))
                            
                        async with ClientSession(read, write) as session:
                            await session.initialize()
                            response = await session.list_tools()
                            tools = response.tools
                
                logger.info(f"Successfully extracted {len(tools)} tools from {name}")
                return tools
                
        except Exception as e:
            logger.error(f"Failed to extract from {transport_type} server '{name}': {e}")
            if self.debug:
                logger.exception(e)
            return []

    def save_as_flat_list(self, tools_data: List[Dict[str, Any]], output_file: str, append: bool = False):
        """Save results as a flat list of tools (registry_dump.json style)"""
        output_path = Path(output_file)
        all_tools = tools_data

        if append and output_path.exists():
            try:
                with open(output_path, "r") as f:
                    existing = json.load(f)
                    if isinstance(existing, list):
                        seen = {(t.get("server_origin"), t.get("name")) for t in existing}
                        for tool in tools_data:
                            if (tool.get("server_origin"), tool.get("name")) not in seen:
                                existing.append(tool)
                        all_tools = existing
            except Exception as e:
                logger.error(f"Failed to append to {output_file}: {e}")

        with open(output_path, "w") as f:
            json.dump(all_tools, f, indent=2)
        logger.info(f"Saved {len(all_tools)} tools to {output_file}")

    def save_as_server_config(self, servers_data: List[Dict[str, Any]], output_file: str, merge: bool = False):
        """Save results as grouped server configurations (mcp_servers.json style)"""
        output_path = Path(output_file)
        final_servers = {s['name']: s for s in servers_data}

        if merge and output_path.exists():
            try:
                with open(output_path, "r") as f:
                    existing = json.load(f)
                    existing_servers = {s['name']: s for s in existing.get('servers', [])}
                    
                    for name, new_server in final_servers.items():
                        if name in existing_servers:
                            # Merge tools
                            tools_map = {t['name']: t for t in existing_servers[name].get('tools', [])}
                            for tool in new_server.get('tools', []):
                                tools_map[tool['name']] = tool
                            
                            updated = dict(existing_servers[name])
                            updated.update({k: v for k, v in new_server.items() if k != 'tools'})
                            updated['tools'] = list(tools_map.values())
                            existing_servers[name] = updated
                        else:
                            existing_servers[name] = new_server
                    final_servers = existing_servers
            except Exception as e:
                logger.error(f"Failed to merge with {output_file}: {e}")

        with open(output_path, "w") as f:
            json.dump({"servers": list(final_servers.values())}, f, indent=2)
        
        total_tools = sum(len(s.get('tools', [])) for s in final_servers.values())
        logger.info(f"Saved {len(final_servers)} servers ({total_tools} tools) to {output_file}")

from contextlib import AsyncExitStack

async def main():
    parser = argparse.ArgumentParser(description="MCP Registry Manager - Universal Tool Discovery & Extraction")
    
    # Modes
    parser.add_argument("--mode", choices=["flat", "config"], default="config", 
                        help="Output format: 'flat' for registry_dump.json, 'config' for mcp_servers.json")
    
    # Single server mode
    parser.add_argument("--name", help="Server name")
    parser.add_argument("--transport", choices=["stdio", "sse", "http"], help="Transport type")
    parser.add_argument("--command", help="Command for stdio")
    parser.add_argument("--args", nargs="*", help="Arguments for stdio")
    parser.add_argument("--url", help="URL for SSE/HTTP")
    
    # Batch mode
    parser.add_argument("--source-config", help="Path to source servers.json config file")
    
    # Output options
    parser.add_argument("--output", help="Output file path")
    parser.add_argument("--merge", "--append", action="store_true", dest="merge", 
                        help="Non-destructive update (merge for config, append for flat)")
    parser.add_argument("--timeout", type=int, default=30, help="Timeout in seconds")
    parser.add_argument("--debug", action="store_true", help="Debug logging")
    parser.add_argument("-H", "--header", action="append", help="Custom headers")
    
    args = parser.parse_args()
    manager = MCPRegistryManager(timeout=args.timeout, debug=args.debug)
    headers = manager.parse_header_entries(args.header)
    
    servers_to_process = {}
    if args.source_config:
        with open(args.source_config, 'r') as f:
            src = json.load(f)
            # Support both flat map and {"servers": [...]} format
            if isinstance(src, dict) and "servers" in src:
                servers_to_process = {s['name']: s for s in src['servers']}
            else:
                servers_to_process = src
    elif args.name and args.transport:
        cfg = {"name": args.name, "transport": args.transport}
        if args.transport == "stdio":
            cfg["command"] = args.command
            # Sanitize args
            raw_args = args.args or []
            san_args = []
            for a in raw_args:
                if " " in a: san_args.extend(shlex.split(a))
                else: san_args.append(a)
            cfg["args"] = san_args
            # Record the full command (command + sanitized args) as the target.
            cmd_tokens = []
            if args.command:
                try:
                    cmd_tokens = shlex.split(args.command)
                except Exception:
                    cmd_tokens = [args.command]

            full_tokens = cmd_tokens + san_args
            if full_tokens:
                try:
                    cfg['target'] = shlex.join(full_tokens)
                except AttributeError:
                    # Fallback for Python versions without shlex.join
                    cfg['target'] = ' '.join(full_tokens)
        else:
            cfg["url"] = args.url
            if headers: cfg["headers"] = headers
        servers_to_process = {args.name: cfg}
    else:
        logger.error("Must provide --source-config or --name/--transport")
        sys.exit(1)

    all_flat_tools = []
    all_server_configs = []

    for s_name, s_cfg in servers_to_process.items():
        raw_tools = await manager.extract_from_server(s_name, s_cfg)
        
        if args.mode == "flat":
            for t in raw_tools:
                all_flat_tools.append(manager.parse_tool_for_registry(t, s_name, s_cfg.get("transport", "stdio")))
        else:
            server_entry = dict(s_cfg)
            server_entry["name"] = s_name
            server_entry["tools"] = [manager.parse_tool_for_config(t) for t in raw_tools]
            all_server_configs.append(server_entry)

    # Default output names if not provided
    output = args.output
    if not output:
        output = "registry_dump.json" if args.mode == "flat" else "mcp_servers.json"

    if args.mode == "flat":
        manager.save_as_flat_list(all_flat_tools, output, append=args.merge)
    else:
        manager.save_as_server_config(all_server_configs, output, merge=args.merge)

if __name__ == "__main__":
    asyncio.run(main())
