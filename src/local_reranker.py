# src/local_reranker.py
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from typing import List, Dict, Any, Optional
from src.config import Config


class LocalReranker:
    """
    Local Jina Reranker v3 for final precision ranking.
    Uses custom JinaForRanking architecture from the model directory.
    """
    
    def __init__(self, model_path: str = "./re-rank", 
                 cache_dir: str = "."):
        self.model_path = model_path
        self.cache_dir = cache_dir
                
        # Optimization: Set threads for CPU inference
        if Config.DEVICE == "cpu":
            import os
            # Use a reasonable number of threads, not all to avoid contention
            num_threads = min(os.cpu_count() or 4, 8)
            torch.set_num_threads(num_threads)

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
        
        # CPU Optimization: Dynamic Quantization
        if Config.DEVICE == "cpu":
            try:
                # Quantize Linear layers to int8 for 2-4x speedup on CPU
                self.model = torch.quantization.quantize_dynamic(
                    self.model, {torch.nn.Linear}, dtype=torch.qint8
                )
            except Exception as e:
                import logging
                logging.warning(f"Failed to quantize reranker: {e}")

        # Inject tokenizer into model to avoid redundant loading in its internal methods
        self.model._tokenizer = self.tokenizer
            
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

        # Use the model's native rerank method with inference_mode
        with torch.inference_mode():
            results = self.model.rerank(
                query=query,
                documents=documents,
                top_n=top_n
            )

        # Ensure the results match the expected format: {'index', 'relevance_score', 'document'}
        return [
            {
                'index': item['index'],
                'relevance_score': float(item['relevance_score']),
                'document': item['document']
            }
            for item in results
        ]