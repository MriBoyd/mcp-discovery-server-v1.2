# src/local_reranker.py
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from typing import List, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LocalReranker:
    """
    Local Jina Reranker v3 for final precision ranking
    Cross-encoder that scores (query, document) pairs jointly
    """
    
    def __init__(self, model_path: str = "jinaai/jina-reranker-v3", 
                 cache_dir: str = "./models"):
        self.model_path = model_path
        self.cache_dir = cache_dir
        
        logger.info(f"Loading Reranker from {model_path}...")
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            cache_dir=cache_dir,
            local_files_only=True
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            cache_dir=cache_dir,
            torch_dtype=torch.bfloat16,
            local_files_only=True
        )
        self.model.eval()
        self.model.to(Config.DEVICE)
        
        logger.info(f"Reranker loaded on {Config.DEVICE}")
    
    def rerank(self, query: str, documents: List[str], top_n: int = None) -> List[Dict[str, Any]]:
        """
        Rerank documents based on relevance to query
        
        Args:
            query: Search query
            documents: List of document texts to rerank
            top_n: Number of top results to return
        
        Returns:
            List of dicts with 'index', 'relevance_score', 'document'
        """
        if top_n is None:
            top_n = Config.FINAL_RESULTS
        
        # Format pairs for cross-encoder
        pairs = [[query, doc] for doc in documents]
        
        # Tokenize
        inputs = self.tokenizer(
            pairs,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Score
        with torch.no_grad():
            scores = self.model(**inputs).logits.squeeze(-1)
            scores = torch.sigmoid(scores)  # Convert to [0,1] range
        
        # Convert to list
        scores = scores.cpu().numpy().tolist()
        
        # Sort by score
        indexed_scores = list(enumerate(scores))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return top_n
        results = []
        for idx, score in indexed_scores[:top_n]:
            results.append({
                'index': idx,
                'relevance_score': score,
                'document': documents[idx]
            })
        
        return results