#!/usr/bin/env python3
"""Test script for MCP proxy server."""

import asyncio
import json
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def test_proxy():
    """Test the proxy server by listing and calling tools."""
    
    server_params = StdioServerParameters(
        command="python",
        args=["proxy_server.py", "config.yaml"]
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            # List all tools
            tools = await session.list_tools()
            print(f"\n📦 Available tools ({len(tools.tools)}):")
            for tool in tools.tools[:5]:  # Show first 5
                print(f"  - {tool.name}: {tool.description}")
            
            if len(tools.tools) > 5:
                print(f"  ... and {len(tools.tools) - 5} more")
            
            # Test a specific tool if available
            if tools.tools:
                test_tool = tools.tools[0]
                print(f"\n🧪 Testing tool: {test_tool.name}")
                
                # Provide minimal test arguments
                test_args = {}
                if "path" in test_tool.inputSchema.get("properties", {}):
                    test_args["path"] = "/tmp"
                
                result = await session.call_tool(test_tool.name, test_args)
                print(f"✅ Result: {result.content[0].text[:200]}...")


if __name__ == "__main__":
    asyncio.run(test_proxy())