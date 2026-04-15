# mcp_server.py
import asyncio
import logging
import json
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional
from contextlib import asynccontextmanager
from dataclasses import dataclass
import sys
import contextlib
from mcp.server.fastmcp import FastMCP, Context
from mcp.server.session import ServerSession
from mcp.server.fastmcp.exceptions import ToolError
from hybrid_searcher import HybridToolSearcher

# Module-level singleton for the heavy searcher to avoid repeated weight loads
_shared_searcher: Optional[HybridToolSearcher] = None


def get_shared_searcher() -> HybridToolSearcher:
    """Lazily initialize and return a single shared HybridToolSearcher instance."""
    global _shared_searcher
    if _shared_searcher is None:
        with contextlib.redirect_stdout(sys.stderr):
            _shared_searcher = HybridToolSearcher()
            _shared_searcher.is_indexed = True
    return _shared_searcher

# Comment out if hybrid_searcher isn't ready yet
# from hybrid_searcher import HybridToolSearcher
# from config import Config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stderr  # CRITICAL: Force logs to stderr
)
# ============= Mock searcher for testing =============

@dataclass
class AppContext:
    """Type-safe application context with shared resources"""
    searcher: Any  # HybridToolSearcher or Mock
    

@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
    """Manage application lifecycle."""
      
    # Use shared searcher singleton to prevent multiple heavy initializations
    searcher = get_shared_searcher()

    yield AppContext(
        searcher=searcher,
    )
    

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
        raise ToolError(f"Search failed: {str(e)}")

# ============= Main Entry Points =============

async def run_stdio():
    """Run the MCP server using stdio transport."""
    await mcp_stdio.run_stdio_async()

async def run_sse():
    """Run the MCP server using SSE transport."""
    await mcp_sse.run_sse_async()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--sse":
        # Run with SSE transport
        asyncio.run(run_sse())
    else:
        # Run with stdio transport (default)
        asyncio.run(run_stdio())