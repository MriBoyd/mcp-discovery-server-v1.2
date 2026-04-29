import json
import logging
import os
import argparse
from  src.hybrid_searcher import HybridToolSearcher

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Index tools from JSON into Qdrant")
    parser.add_argument("--file", help="Path to tools JSON file (registry_dump.json or mcp_servers.json)")
    parser.add_argument("--force", action="store_true", help="Force re-indexing even if data exists")
    args = parser.parse_args()

    logger.info("Starting index maintenance script...")
    base = os.path.dirname(os.path.abspath(__file__))
    
    # Determine which file to load
    file_path = args.file
    if not file_path:
        # Default fallback logic
        dump_path = os.path.join(base, "../registry_dump.json")
        config_path = os.path.join(base, "../mcp_servers.json")
        
        if os.path.exists(config_path):
            file_path = config_path
        elif os.path.exists(dump_path):
            file_path = dump_path
        else:
            logger.error("No tools file found (registry_dump.json or mcp_servers.json)")
            return

    # 1. Load tools
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
            
            # Detect format
            if isinstance(data, list):
                # Flat list format (registry_dump.json)
                tools = data
            elif isinstance(data, dict):
                if "servers" in data:
                    # Config format (mcp_servers.json)
                    tools = []
                    for server in data["servers"]:
                        server_name = server.get("name", "unknown")
                        for tool in server.get("tools", []):
                            # Ensure tool has server origin for indexing if not present
                            if "server_origin" not in tool:
                                tool["server_origin"] = server_name
                            tools.append(tool)
                elif "tools" in data:
                    tools = data["tools"]
                else:
                    logger.error(f"Unrecognized JSON format in {file_path}")
                    return
            else:
                logger.error(f"Invalid JSON data in {file_path}")
                return
                
        logger.info(f"Loaded {len(tools)} tools from {file_path}")
    except Exception as e:
        logger.error(f"Failed to load tools from {file_path}: {e}")
        return

    # 2. Initialize searcher
    searcher = HybridToolSearcher()
    
    # 3. Index tools
    logger.info("Indexing tools into Qdrant...")
    searcher.index(tools, force_reindex=args.force)
    
    logger.info("Indexing complete. Data is now persisted in Qdrant.")

if __name__ == "__main__":
    main()
