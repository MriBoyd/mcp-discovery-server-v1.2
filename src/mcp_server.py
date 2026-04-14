# mcp_server.py
import asyncio
import logging
from typing import Any, AsyncIterator, Dict, List, Optional
from contextlib import asynccontextmanager
from dataclasses import dataclass

from mcp.server.fastmcp import FastMCP, Context
from mcp.server.session import ServerSession
from mcp.types import TextContent, CallToolResult
from mcp.server.fastmcp.exceptions import ToolError

from src.hybrid_searcher import HybridToolSearcher
from src.config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============= Application Context =============

@dataclass
class AppContext:
    """Type-safe application context with shared resources"""
    searcher: HybridToolSearcher
    all_tools: List[Dict[str, Any]]

@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
    """
    Manage application lifecycle.
    Initializes the hybrid search engine on startup.
    """
    logger.info("Starting MCP Tool Search Server...")
    
    # Load your 8,000 tools
    import json
    with open("tools/all_tools.json", "r") as f:
        tools_data = json.load(f)
        all_tools = tools_data if isinstance(tools_data, list) else tools_data.get("tools", [])
    
    logger.info(f"Loaded {len(all_tools)} tools")
    
    # Initialize hybrid searcher
    searcher = HybridToolSearcher()
    searcher.index(all_tools)
    
    logger.info("Server ready")
    
    yield AppContext(
        searcher=searcher,
        all_tools=all_tools
    )
    
    logger.info("Server shutdown complete")

# ============= Create MCP Server =============

mcp = FastMCP(
    name="mcp-tool-search-server",
    instructions="A semantic tool search server with hybrid BM25 + dense retrieval + reranking",
    lifespan=app_lifespan
)

# ============= Tool: Search for Tools =============

@mcp.tool(
    name="search_tools",
    description="Search for relevant MCP tools based on a natural language query. Returns the top 5 most relevant tools with their schemas."
)
async def search_tools(
    query: str,
    ctx: Context[ServerSession, AppContext]
) -> str:
    """
    Search for tools using hybrid retrieval.
    
    Args:
        query: Natural language description of what you want to do
              (e.g., "create a GitHub issue", "send a Slack message")
    
    Returns:
        Formatted string with top matching tools and their schemas
    """
    try:
        # Access the searcher from lifespan context
        searcher = ctx.request_context.lifespan_context.searcher
        
        # Report progress
        await ctx.report_progress(
            progress=0.3,
            total=1.0,
            message="Searching for relevant tools..."
        )
        
        # Perform hybrid search
        results = await asyncio.to_thread(searcher.search, query)
        
        await ctx.report_progress(
            progress=0.8,
            total=1.0,
            message="Formatting results..."
        )
        
        if not results:
            return "No tools found matching your query."
        
        # Format results
        output = f"🔍 Found {len(results)} relevant tools for: '{query}'\n\n"
        
        for i, result in enumerate(results, 1):
            tool = result['tool_schema']
            score = result['relevance_score']
            
            output += f"{i}. **{tool['name']}** (relevance: {score:.3f})\n"
            output += f"   Description: {tool['description'][:150]}...\n"
            
            # Show parameters
            if tool.get('parameters'):
                output += "   Parameters:\n"
                for param_name, param_info in tool['parameters'].items():
                    param_type = param_info.get('type', 'any')
                    param_desc = param_info.get('description', '')
                    required = param_name in tool.get('required', [])
                    req_mark = " (required)" if required else ""
                    output += f"     - {param_name} ({param_type}){req_mark}: {param_desc}\n"
            output += "\n"
        
        return output
        
    except Exception as e:
        raise ToolError(f"Search failed: {str(e)}")

# ============= Tool: Get Tool Schema =============

@mcp.tool(
    name="get_tool_schema",
    description="Get the complete JSON schema for a specific tool by name."
)
async def get_tool_schema(
    tool_name: str,
    ctx: Context[ServerSession, AppContext]
) -> str:
    """
    Retrieve the full schema of a specific tool.
    
    Args:
        tool_name: Exact name of the tool (e.g., "create_github_issue")
    """
    searcher = ctx.request_context.lifespan_context.searcher
    
    # Search for exact match
    results = await asyncio.to_thread(searcher.search, tool_name, top_k=5)
    
    for result in results:
        if result['tool_name'] == tool_name:
            import json
            schema = result['tool_schema']
            return json.dumps(schema, indent=2)
    
    return f"Tool '{tool_name}' not found."

# ============= Tool: Execute Tool (Placeholder) =============

@mcp.tool(
    name="execute_tool",
    description="Execute a discovered tool with given arguments. Use search_tools first to find the right tool."
)
async def execute_tool(
    tool_name: str,
    arguments: str,
    ctx: Context[ServerSession, AppContext]
) -> str:
    """
    Execute a tool with provided arguments.
    
    Args:
        tool_name: Name of the tool to execute
        arguments: JSON string of arguments for the tool
    """
    searcher = ctx.request_context.lifespan_context.searcher
    
    # Find the tool
    results = await asyncio.to_thread(searcher.search, tool_name, top_k=3)
    tool = None
    
    for result in results:
        if result['tool_name'] == tool_name:
            tool = result
            break
    
    if not tool:
        raise ToolError(f"Tool '{tool_name}' not found. Use search_tools to find available tools.")
    
    # Parse arguments
    import json
    try:
        args = json.loads(arguments)
    except json.JSONDecodeError:
        raise ToolError(f"Invalid JSON arguments: {arguments}")
    
    # Validate required parameters
    schema = tool['tool_schema']
    required = schema.get('required', [])
    missing = [p for p in required if p not in args]
    if missing:
        raise ToolError(f"Missing required parameters: {', '.join(missing)}")
    
    # Execute the actual tool logic here
    # This is where you'd integrate your actual tool execution
    result = await _execute_actual_tool(tool_name, args)
    
    return result

async def _execute_actual_tool(tool_name: str, args: Dict) -> str:
    """
    Placeholder for actual tool execution.
    Replace with your tool implementation.
    """
    # This is where you'd route to your actual tool functions
    # For now, return a mock response
    return f"Executed {tool_name} with args: {args}"

# ============= Resource: List All Tools =============

@mcp.resource("mcp://tools")
async def list_all_tools(ctx: Context[ServerSession, AppContext]) -> str:
    """Get a formatted list of all available tools."""
    searcher = ctx.request_context.lifespan_context.searcher
    all_tools = ctx.request_context.lifespan_context.all_tools
    
    output = f"# Available Tools ({len(all_tools)} total)\n\n"
    
    for tool in all_tools[:50]:  # Limit to 50 for readability
        output += f"- **{tool['name']}**: {tool['description'][:100]}...\n"
    
    if len(all_tools) > 50:
        output += f"\n... and {len(all_tools) - 50} more tools. Use search_tools to find specific tools."
    
    return output

# ============= Resource: Tool by Name Template =============

@mcp.resource("mcp://tool/{name}")
async def get_tool_by_name(name: str, ctx: Context[ServerSession, AppContext]) -> str:
    """Get details of a specific tool by name."""
    searcher = ctx.request_context.lifespan_context.searcher
    results = await asyncio.to_thread(searcher.search, name, top_k=5)
    
    for result in results:
        if result['tool_name'] == name:
            import json
            return json.dumps(result['tool_schema'], indent=2)
    
    return f"Tool '{name}' not found."

# ============= Prompt: Tool Selection Guidance =============

@mcp.prompt(name="select_tool")
async def select_tool_prompt(user_goal: str) -> str:
    """
    Prompt template for helping users select the right tool.
    """
    return f"""
User Goal: {user_goal}

Please search for relevant tools using the search_tools function with this goal as the query.

Based on the search results:
1. Identify the most appropriate tool for this task
2. List the required parameters
3. Provide an example of how to use it

Tool search query: search_tools(query="{user_goal}")
"""

# ============= Main Entry Point =============

async def main():
    """Run the MCP server using stdio transport."""
    await mcp.run_stdio_async()

if __name__ == "__main__":
    asyncio.run(main())