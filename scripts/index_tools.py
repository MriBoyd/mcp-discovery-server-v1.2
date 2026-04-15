
import json
import logging

import os
from src.hybrid_searcher import HybridToolSearcher

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting index maintenance script...")
    base = os.path.dirname(os.path.abspath(__file__))

    # 1. Load tools
    try:
        with open(os.path.join(base, "./registry_dump.json"), "r") as f:
            data = json.load(f)
            tools = data if isinstance(data, list) else data.get('tools', [])
        logger.info(f"Loaded {len(tools)} tools from file.")
    except Exception as e:
        logger.error(f"Failed to load tools: {e}")
        return

    # 2. Initialize searcher
    # This will connect to Qdrant and check index state
    searcher = HybridToolSearcher()
    
    # 3. Index tools
    # By default index() checks if Qdrant already has data.
    # To force re-indexing if tools changed, use force_reindex=True
    logger.info("Indexing tools into Qdrant...")
    searcher.index(tools, force_reindex=True)
    
    logger.info("Indexing complete. Data is now persisted in Qdrant.")

if __name__ == "__main__":
    main()
