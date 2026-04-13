# src/config.py
import os
import torch 

class Config:
    # Model paths (local cache)
    MODEL_CACHE_DIR = os.getenv("MODEL_CACHE_DIR", "./models")
    
    # Jina Code Embeddings (local)
    CODE_EMBEDDING_MODEL = "jinaai/jina-code-embeddings-1.5b"
    EMBEDDING_DIM = 1536
    
    # Reranker (local)
    RERANKER_MODEL = "jinaai/jina-reranker-v3"  # or GGUF version
    
    # Search settings
    BM25_CANDIDATES = 100      # Initial BM25 retrieval
    DENSE_CANDIDATES = 100     # Initial dense retrieval
    FUSION_CANDIDATES = 50     # Candidates after fusion
    FINAL_RESULTS = 5          # Top results after reranking
    
    # Fusion weights (BM25 : Dense)
    BM25_WEIGHT = 0.3
    DENSE_WEIGHT = 0.7
    FUSION_METHOD = "rrf"  # "weighted", "rrf", or "weighted_rank"
    RRF_K = 60  # RRF constant (typical values: 60)
        
    # Device
    DEVICE = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    
    # Offline mode - prevent any network requests
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["SENTENCE_TRANSFORMERS_HOME"] = MODEL_CACHE_DIR