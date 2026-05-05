# Ansam: MCP Discovery Server (v1.2)

Ansam is a high-performance **Model Context Protocol (MCP) Discovery Server** designed to manage, index, and proxy requests across a massive ecosystem of 1000+ MCP servers. It provides AI models with a unified interface to search for specific tools using advanced hybrid retrieval and execute them safely via a dynamic, LRU-cached proxy system.

## 🚀 Key Features

-   **Dynamic Proxy Management:** Seamlessly handle connections to hundreds of MCP servers. Uses an **LRU Cache** (default: 50 active connections) to optimize resource usage and prevent socket exhaustion.
-   **Advanced Hybrid Tool Search:** A three-stage retrieval pipeline using **Qdrant**:
    1.  **Retrieval:** Combines Dense (Semantic) and Sparse (Lexical) vectors with BM25.
    2.  **Reranking:** In-database **ColBERT** late interaction for high-precision results.
    3.  **Fusion:** Optimized weighted fusion of multiple scoring signals.
-   **Universal Tool Extraction:** Built-in `MCPRegistryManager` to discover and extract tool definitions from **stdio**, **SSE**, and **HTTP** transports.
-   **Production-Grade Reliability:**
    -   **Circuit Breakers:** Prevent cascading failures when downstream servers are unresponsive.
    -   **Rate Limiting:** Granular global and per-server rate control.
    -   **Authentication:** Integrated `AuthManager` for handling token-based security for proxied servers.
-   **Flexible Deployment:** Supports **stdio**, **SSE**, and **HTTP** server transports for the discovery interface itself.

---

## 🏗️ Architecture Overview

Ansam acts as an intelligent gateway between an AI model and a vast collection of MCP tools:

1.  **Discovery Layer:** Scans known MCP servers and builds a comprehensive registry (`mcp_servers.json`).
2.  **Indexing Layer:** Processes the registry and populates a **Qdrant** vector database with semantic embeddings and lexical tokens.
3.  **Search Layer:** When a user asks for a tool, Ansam performs a hybrid search to find the most relevant tool across all registered servers.
4.  **Execution Layer:** When a tool is called, Ansam dynamically creates a proxy connection to the target server, executes the command, and returns the result, maintaining the connection in a "hot" cache for subsequent calls.

---

## 🛠️ Installation & Setup

### Prerequisites

-   Python 3.10+
-   **Qdrant** Vector Database (Running on port 6333)
    ```bash
    docker run -p 6333:6333 qdrant/qdrant
    ```

### 1. Clone & Install Dependencies

```bash
git clone <repository-url>
cd mcp-discovery-server-v1.2
pip install -r requirements.txt
```

---

## 📖 Quick Start

### 1. Extract Tools (Optional)
If you have new servers to add, use the registry manager to extract their tool definitions:
```bash
python src/mcp_registry_manager.py --name my-server --transport stdio --command "python" --args "path/to/server.py" --output src/mcp_servers.json --merge
```

### 2. Index Tools
Populate the Qdrant database with the tools defined in your configuration:
```bash
python src/scripts/index_tools.py --file src/mcp_servers.json
```

### 3. Run the Discovery Server
Start the Ansam MCP server:
```bash
# Default: stdio transport
python src/mcp_server_production.py

# Or use SSE/HTTP transport
python src/mcp_server_production.py --sse --port 8000
```

---

## 🔍 Search Capabilities

Ansam exposes a `search_tools` tool that allows models to find capabilities using natural language.

**Example Query:** *"I need a tool to fetch weather data and send an email."*

**Retrieval Pipeline:**
-   **Dense:** `sentence-transformers/all-MiniLM-L6-v2`
-   **Sparse:** `Splade_PP_en_v1`
-   **Reranker:** `colbert-ir/colbertv2.0`

---

## ⚙️ Configuration

Key settings can be adjusted in `src/config.py` or via Environment Variables:

| Variable | Default | Description |
| :--- | :--- | :--- |
| `QDRANT_URL` | `http://localhost:6333` | URL of the Qdrant instance |
| `MAX_OPEN_CONNECTIONS` | `32` | Max concurrent proxy connections in LRU cache |
| `GLOBAL_RATE` | `10.0` | Global requests per second limit |
| `PER_SERVER_RATE` | `2.0` | Per-server requests per second limit |
| `CB_FAILURE_THRESHOLD` | `5` | Failures before circuit breaker opens |
| `DEVICE` | `cuda`/`cpu` | Processing device for embeddings |

---

## 📂 Project Structure

-   `src/mcp_server_production.py`: Main entry point for the discovery server.
-   `src/mcp_registry_manager.py`: Universal tool discovery and registry maintenance.
-   `src/hybrid_searcher.py`: Core search engine logic.
-   `src/auth_manager.py`: Authentication handler for proxied servers.
-   `src/circuit_breaker.py`: Fault tolerance implementation.
-   `src/scripts/index_tools.py`: Data ingestion script for Qdrant.
-   `src/mcp_servers.json`: The source-of-truth configuration for all proxied servers.

---

## 🧪 Testing

Run the test suite to ensure system integrity:
```bash
# Unit tests
pytest tests/

# Benchmarks
python tests/benchmark_search.py
```

---

## 📜 License

This project is licensed under the terms of the **MIT License**. See `LICENSE` for details.
