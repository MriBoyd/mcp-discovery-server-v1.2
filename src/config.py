# src/config.py
import os
import torch 

class Config:
    
    # Model paths (local cache)
    MODEL_CACHE_DIR = os.getenv("MODEL_CACHE_DIR", ".")
    
    # Jina Code Embeddings (local)
    CODE_EMBEDDING_MODEL = "./embed-models"
    EMBEDDING_DIM = 896
    
    # Reranker (local)
    RERANKER_MODEL = "./re-rank"  # or GGUF version
    
    # Search settings
    BM25_CANDIDATES = 20      # Initial BM25 retrieval
    DENSE_CANDIDATES = 20     # Initial dense retrieval
    FUSION_CANDIDATES = 7      # Candidates after fusion (reduced from 10)
    FINAL_RESULTS = 1          # Top results after reranking
    
    # Fusion weights (BM25 : Dense)
    BM25_WEIGHT = 0.3
    DENSE_WEIGHT = 0.7
    FUSION_METHOD = "rrf"  # "weighted", "rrf", or "weighted_rank"
    RRF_K = 60  # RRF constant (typical values: 60)
        
    # Qdrant Vector DB
    QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
    QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "mcp_tools")
    
    # Device
    DEVICE = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    
    # Offline mode - prevent any network requests
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["SENTENCE_TRANSFORMERS_HOME"] = MODEL_CACHE_DIR
    
    