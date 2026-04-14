# mcp_server.py
import asyncio
import logging
import json
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional
from contextlib import asynccontextmanager
from dataclasses import dataclass

from mcp.server.fastmcp import FastMCP, Context
from mcp.server.session import ServerSession
from mcp.types import TextContent, CallToolResult
from mcp.server.fastmcp.exceptions import ToolError
from hybrid_searcher import HybridToolSearcher

# Comment out if hybrid_searcher isn't ready yet
# from hybrid_searcher import HybridToolSearcher
# from config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============= Mock searcher for testing =============

@dataclass
class AppContext:
    """Type-safe application context with shared resources"""
    searcher: Any  # HybridToolSearcher or Mock
    all_tools: List[Dict[str, Any]]

@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
    """Manage application lifecycle."""
    logger.info("Starting MCP Tool Search Server...")
    
    # Create tools data directory if it doesn't exist
    tools_dir = Path("tools")
    tools_dir.mkdir(exist_ok=True)
    
    # Load or create sample tools data
    tools_file = tools_dir / "all_tools.json"
    
    if tools_file.exists():
        with open(tools_file, "r") as f:
            tools_data = json.load(f)
            all_tools = tools_data if isinstance(tools_data, list) else tools_data.get("tools", [])
    else:
        # Create sample tools data for testing
        all_tools = [
            {
                "name": "create_github_issue",
                "description": "Create a new issue in a GitHub repository",
                "parameters": {
                    "repo": {"type": "string", "description": "Repository name"},
                    "title": {"type": "string", "description": "Issue title"},
                    "body": {"type": "string", "description": "Issue body"}
                },
                "required": ["repo", "title"]
            },
            {
                "name": "send_slack_message",
                "description": "Send a message to a Slack channel",
                "parameters": {
                    "channel": {"type": "string", "description": "Channel name"},
                    "message": {"type": "string", "description": "Message text"}
                },
                "required": ["channel", "message"]
            },
            {
                "name": "search_files",
                "description": "Search for files in a directory",
                "parameters": {
                    "path": {"type": "string", "description": "Directory path"},
                    "pattern": {"type": "string", "description": "Search pattern"}
                },
                "required": ["path"]
            }
        ]
        # Save sample data
        with open(tools_file, "w") as f:
            json.dump(all_tools, f, indent=2)
        logger.info(f"Created sample tools file with {len(all_tools)} tools")
    
    logger.info(f"Loaded {len(all_tools)} tools")
    
    # Initialize searcher (use mock for now)
    searcher = HybridToolSearcher()
    searcher.is_indexed = True
        
    
    logger.info("Server ready")
    
    yield AppContext(
        searcher=searcher,
        all_tools=all_tools
    )
    
    logger.info("Server shutdown complete")

# ============= Create MCP Server =============

# For stdio transport (default)
mcp_stdio = FastMCP(
    name="mcp-tool-search-server",
    instructions="A semantic tool search server with hybrid retrieval",
    lifespan=app_lifespan
)

# For SSE transport (needs separate instance)
mcp_sse = FastMCP(
    name="mcp-tool-search-server",
    instructions="A semantic tool search server with hybrid retrieval",
    lifespan=app_lifespan,
    port=8000,
    host="0.0.0.0"  # Use 0.0.0.0 to allow external connections
)

# ============= Tool: Search for Tools =============

@mcp_stdio.tool(
    name="search_tools",
    description="Search for relevant MCP tools based on a natural language query."
)
@mcp_sse.tool(
    name="search_tools", 
    description="Search for relevant MCP tools based on a natural language query."
)
async def search_tools(
    query: str,
    ctx: Context[ServerSession, AppContext]
) -> str:
    """Search for tools using hybrid retrieval."""
    try:
        searcher = ctx.request_context.lifespan_context.searcher
        
        await ctx.report_progress(
            progress=0.3,
            total=1.0,
            message="Searching for relevant tools..."
        )
        
        # Perform search (sync to async conversion)
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
            tool = result.get('tool_schema', result)  # Handle both formats
            score = result.get('relevance_score', 1.0)
            
            output += f"{i}. **{tool.get('name', 'unknown')}** (relevance: {score:.3f})\n"
            output += f"   Description: {tool.get('description', 'No description')[:150]}...\n"
            
            # Show parameters if available
            params = tool.get('parameters', {})
            if params:
                output += "   Parameters:\n"
                for param_name, param_info in params.items():
                    if isinstance(param_info, dict):
                        param_type = param_info.get('type', 'any')
                        param_desc = param_info.get('description', '')
                    else:
                        param_type = 'any'
                        param_desc = str(param_info)
                    output += f"     - {param_name} ({param_type}): {param_desc}\n"
            output += "\n"
        
        return output
        
    except Exception as e:
        logger.error(f"Search failed: {e}", exc_info=True)
        raise ToolError(f"Search failed: {str(e)}")

# ============= Main Entry Points =============

async def run_stdio():
    """Run the MCP server using stdio transport."""
    logger.info("Starting MCP server in stdio mode...")
    await mcp_stdio.run_stdio_async()

async def run_sse():
    """Run the MCP server using SSE transport."""
    logger.info(f"Starting MCP server in SSE mode on http://0.0.0.0:8000")
    await mcp_sse.run_sse_async()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--sse":
        # Run with SSE transport
        asyncio.run(run_sse())
    else:
        # Run with stdio transport (default)
        asyncio.run(run_stdio())