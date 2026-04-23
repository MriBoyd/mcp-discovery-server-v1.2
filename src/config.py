# src/config.py
import os
import torch 

class Config:
    
    # Model paths (local cache)
    MODEL_CACHE_DIR = os.getenv("MODEL_CACHE_DIR", ".")
    
    # Jina Code Embeddings (local)
    CODE_EMBEDDING_MODEL = "./jina-code-0.5b"
    EMBEDDING_DIM = 896
    
    # Reranker (local)
    RERANKER_MODEL = "./re-rank"  # or GGUF version
    
    # Search settings
    BM25_CANDIDATES = 20      # Initial BM25 retrieval
    DENSE_CANDIDATES = 20     # Initial dense retrieval
    FUSION_CANDIDATES = 4      # Candidates after fusion (reduced from 7 for speed)
    FINAL_RESULTS = 1          # Top results after reranking
    SEARCH_CACHE_SIZE = 100    # Number of queries to cache
    SEARCH_CACHE_TTL = 3600    # Cache TTL in seconds (1 hour)
    
    # Fusion weights (BM25 : Dense)
    BM25_WEIGHT = 0.3
    DENSE_WEIGHT = 0.7
    FUSION_METHOD = "rrf"  # "weighted", "rrf", or "weighted_rank"
    RRF_K = 60  # RRF constant (typical values: 60)
        
    # Qdrant Vector DB
    QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
    QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "mcp_tools")
    
    # Connection Pool settings
    MAX_OPEN_CONNECTIONS = int(os.getenv("MAX_OPEN_CONNECTIONS", "32"))
    WARMUP_LIMIT = int(os.getenv("WARMUP_LIMIT", "5"))
    
    # Rate Limiting
    GLOBAL_RATE = float(os.getenv("GLOBAL_RATE", "10.0"))
    GLOBAL_CAPACITY = int(os.getenv("GLOBAL_CAPACITY", "20"))
    PER_SERVER_RATE = float(os.getenv("PER_SERVER_RATE", "2.0"))
    PER_SERVER_CAPACITY = int(os.getenv("PER_SERVER_CAPACITY", "5"))
    
    # Circuit Breaker
    CB_FAILURE_THRESHOLD = int(os.getenv("CB_FAILURE_THRESHOLD", "5"))
    CB_RECOVERY_TIMEOUT = float(os.getenv("CB_RECOVERY_TIMEOUT", "30.0"))
    CB_HALF_OPEN_SUCCESS = int(os.getenv("CB_HALF_OPEN_SUCCESS", "2"))
    
    # Device
    DEVICE = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    
    # Offline mode - prevent any network requests
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["SENTENCE_TRANSFORMERS_HOME"] = MODEL_CACHE_DIR
    
    