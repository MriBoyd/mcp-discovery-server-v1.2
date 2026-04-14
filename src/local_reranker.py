# src/local_reranker.py
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from typing import List, Dict, Any
import logging
from src.config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LocalReranker:
    """
    Local Jina Reranker v3 for final precision ranking
    Cross-encoder that scores (query, document) pairs jointly
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
            local_files_only=True
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            cache_dir=cache_dir,
            trust_remote_code=True,
            torch_dtype=torch.float32, 
            local_files_only=True
        )
        self.model.eval()
        self.model.to(Config.DEVICE)
        
        logger.info(f"Reranker loaded on {Config.DEVICE}")
    
    def rerank(self, query: str, documents: List[str], top_n: int = 5) -> List[Dict[str, Any]]:
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
        ).to(self.model.device)
        
        # Score
        with torch.no_grad():
            outputs = self.model(**inputs)
            scores = outputs.logits.squeeze(-1).cpu().numpy().tolist()
        
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
                'relevance_score': float(score),
                'document': documents[idx]
            })
        
        return results