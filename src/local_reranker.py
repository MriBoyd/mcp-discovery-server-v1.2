# src/local_reranker.py
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from typing import List, Dict, Any, Optional
import logging
from config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LocalReranker:
    """
    Local Jina Reranker v3 for final precision ranking.
    Uses custom JinaForRanking architecture from the model directory.
    """
    
    def __init__(self, model_path: str = "./re-rank", 
                 cache_dir: str = "."):
        self.model_path = model_path
        self.cache_dir = cache_dir
        
        logger.info(f"Loading Reranker from {model_path}...")
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            cache_dir=cache_dir,
            local_files_only=True,
            trust_remote_code=True
        )
        
        # Use AutoModel instead of AutoModelForSequenceClassification
        # This will use the JinaForRanking class defined in modeling.py
        self.model = AutoModel.from_pretrained(
            model_path,
            cache_dir=cache_dir,
            trust_remote_code=True,
            dtype=torch.float32, 
            local_files_only=True
        )
        self.model.eval()
        self.model.to(Config.DEVICE)
        
        # Inject tokenizer into model to avoid redundant loading in its internal methods
        self.model._tokenizer = self.tokenizer
        
        logger.info(f"Reranker loaded on {Config.DEVICE}")
    
    def rerank(self, query: str, documents: List[str], top_n: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Rerank documents based on relevance to query using the model's built-in rerank logic.
        
        Args:
            query: Search query
            documents: List of document texts to rerank
            top_n: Number of top results to return
        
        Returns:
            List of dicts with 'index', 'relevance_score', 'document'
        """
        if top_n is None:
            top_n = Config.FINAL_RESULTS
        
        if not documents:
            return []
            
        # Use the model's native rerank method which handles the specific prompt format
        # and cosine similarity logic for Jina Reranker v3.
        results = self.model.rerank(
            query=query,
            documents=documents,
            top_n=top_n
        )
        
        # Ensure the results match the expected format: {'index', 'relevance_score', 'document'}
        # JinaForRanking.rerank returns {'index', 'relevance_score', 'document', 'embedding'}
        return [
            {
                'index': item['index'],
                'relevance_score': float(item['relevance_score']),
                'document': item['document']
            }
            for item in results
        ]