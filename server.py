from mcp.server.fastmcp import FastMCP
from index import ToolIndex
from mcp_clients import MCPClientManager
from reranker import simple_rerank
import asyncio
import json

# Setup
mcp = FastMCP("Discovery Server", json_response=True)
client_manager = MCPClientManager()
index = ToolIndex()

# =========================
# 🔍 1. BM25 Search (Keyword)
# =========================

@mcp.tool()
def search_tools_bm25(query: str, top_k: int = 10):
    """Keyword search using BM25. Good for exact matches."""
    results = index.bm25_search(query, top_k=top_k)
    return results

# =========================
# 🔍 2. Vector Search (Semantic)
# =========================

@mcp.tool()
def search_tools_vector(query: str, top_k: int = 10):
    """Semantic search using embeddings. Good for conceptual matches."""
    results = index.vector_search(query, top_k=top_k)
    return results

# =========================
# 🔍 3. Hybrid Search (BM25 + Vector)
# =========================

@mcp.tool()
def search_tools_hybrid(query: str, top_k: int = 10):
    """Hybrid search combining keyword and semantic scores."""
    results = index.hybrid_search(query, top_k=top_k)
    return results

# =========================
# 🔍 4. Rerank Search (Recall + Precision)
# =========================

@mcp.tool()
def search_tools_rerank(query: str, top_k: int = 5):
    """Rerank candidates for high precision. Two-stage recall + rerank."""
    candidates = index.hybrid_search(query, top_k=20)
    results = simple_rerank(query, candidates, top_k=top_k)
    return results

# =========================
# 🔍 5. HyDE Search (Hypothetical)
# =========================

@mcp.tool()
def search_tools_hyde(query: str, top_k: int = 5):
    """Search using a generated hypothetical query."""
    hyde_query = index.generate_hyde_query(query)
    # Search with HyDE using rerank for precision
    candidates = index.hybrid_search(hyde_query, top_k=20)
    results = simple_rerank(query, candidates, top_k=top_k)
    return results

# =========================
# 🔍 6. Hierarchical Search (Server -> Tool)
# =========================

@mcp.tool()
def search_servers(query: str):
    """Step 1: Search for relevant servers."""
    servers = list(client_manager.servers.keys())
    return [s for s in servers if query.lower() in s.lower()]

@mcp.tool()
def search_tools_by_server(server_name: str, query: str = None):
    """Step 2: List or search within a specific server."""
    all_tools = index.tools
    filtered = [t for t in all_tools if t["server"] == server_name]
    if query:
        return simple_rerank(query, filtered, top_k=10)
    return filtered

# =========================
# 📦 Lazy Load Tool Definition
# =========================

@mcp.tool()
def get_tool_definition(tool_id: str):
    """Load the full schema for a tool."""
    for t in index.tools:
        if t["name"] == tool_id:
            return t
    return {"error": "Tool not found"}

# =========================
# ⚙️ Execution
# =========================

@mcp.tool()
async def call_mcp_tool(tool_id: str, arguments: dict):
    """Execute a tool via MCP client."""
    tool = next((t for t in index.tools if t["name"] == tool_id), None)
    if not tool: return {"error": f"Tool '{tool_id}' not found."}
    
    server = tool["server"]
    raw_name = tool.get("raw_name", tool_id)
    
    try:
        result = await client_manager.call_tool(server, raw_name, arguments)
        return {"status": "success", "result": result}
    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    mcp.run()
