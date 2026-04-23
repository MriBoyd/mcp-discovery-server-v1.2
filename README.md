# 🔍 Ansam: MCP Discovery & Proxy Server

**The "Global Tool Shed" for AI Agents.** 

Ansam is a high-performance Model Context Protocol (MCP) "Super-Server" designed to index, search, and proxy thousands of MCP tools. While standard LLMs struggle with "tool-selection fatigue" when faced with more than 20-30 tools, Ansam scales to **8,000+ tools** using an industrial-grade hybrid RAG pipeline.

---

## 🚀 Key Features

*   **Industrial Hybrid Search:** Combines BM25 (lexical), Dense Vectors (semantic), and Cross-Encoder Reranking for 99%+ tool retrieval accuracy.
*   **MCP Proxy Gateway:** Acts as a single entry point for hundreds of downstream MCP servers (stdio & SSE).
*   **LRU Connection Pooling:** Dynamically manages server connections to save system resources while maintaining low latency.
*   **Massive Scalability:** Specifically engineered to handle tool registries that exceed the context window of any modern LLM.
*   **Just-in-Time Tooling:** Allows agents to "Search -> Discover -> Execute" tools on the fly without pre-configuring every server.

---

## 🛠 Architecture: The Three-Stage Pipeline

To ensure the agent finds the *exact* tool it needs among thousands, we use a sophisticated retrieval strategy:

1.  **Lexical Search (BM25):** Optimized for matching technical terms, parameter names, and specific function signatures.
2.  **Dense Semantic Search (Qdrant + Jina):** Uses `jina-code-0.5b` to capture the "intent" behind a query (e.g., "help me with my calendar" vs. "list_google_calendar_events").
3.  **Cross-Encoder Reranking:** A final high-precision pass that compares the top 20-30 candidates against the query to ensure the highest relevance score.

---

## ⚔️ Comparison: Ansam vs. Alternatives

| Feature | Ansam (This Repo) | Anthropic Native Tool Search | Anthropic Claude Code |
| :--- | :--- | :--- | :--- |
| **Role** | **Infrastructure/Hub**: A "Global Registry" for tools. | **API Feature**: Native tool discovery in Claude. | **Agent/Client**: A "Hand" that uses tools to write code. |
| **Tool Capacity** | **Massive (8,000+ tools)**. | **Moderate**: Designed for hundreds/low thousands. | **Limited**: Performance degrades as context fills up. |
| **Search Tech** | **Hybrid RAG**: BM25 + Dense Vector + Cross-Encoder Reranking. | **Basic**: BM25 & Regex. | LLM-based reasoning (context-heavy). |
| **Retrieval Accuracy** | **Highest**: Reranking ensures top-tier relevance. | **Good**: Standard keyword/pattern matching. | **Context Dependent**: High but token-intensive. |
| **Connectivity** | **Proxy**: Forwards calls to remote servers. | **Client-side**: You manage the MCP servers. | **Direct**: Connects to local/configured servers. |
| **Platform** | **Agnostic**: Works with any MCP client. | **Claude-Specific**: Requires `advanced-tool-use` beta. | **Claude-Specific**: Built-in to the CLI. |

---

## 📦 Installation & Setup

### 1. Prerequisites
*   Docker & Docker Compose
*   Python 3.10+
*   Jina AI API Key (for high-quality embeddings)

### 2. Start the Vector Database
```bash
docker-compose up -d
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Index Your Tools
Add your MCP servers to `src/mcp_servers.json` and run the indexer:
```bash
python scripts/index_tools.py
```

---

## 🔌 Usage

### As a Standalone MCP Server
Launch the server to expose the `search_tools` and `call_tool` capabilities to any MCP-compatible client (like Claude Desktop, Goose, or your own agent).

```bash
# Start via stdio
python src/mcp_server.py

# Start via SSE (Web Server)
python src/mcp_server.py --sse
```

### Example Agent Flow
1.  **Search:** `search_tools(query="convert csv to markdown")`
2.  **Discover:** The server returns the `csv_transformer` tool schema and its host server.
3.  **Execute:** `call_tool(tool_name="csv_transformer", arguments='{"file": "data.csv"}')`

---

## ⚙️ Configuration

Edit `src/config.py` to tune the search parameters:
*   `BM25_WEIGHT`: Adjust the importance of lexical matching.
*   `DENSE_WEIGHT`: Adjust the importance of semantic matching.
*   `RERANKER_MODEL`: Swap out the cross-encoder for higher speed or higher precision.
*   `MAX_OPEN_CONNECTIONS`: Limit the number of concurrent MCP server connections.

---

## 📄 License
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0). See [LICENSE](LICENSE) for details.
