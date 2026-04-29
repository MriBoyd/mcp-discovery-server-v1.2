# src/local_reranker.py
import os
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoModel, AutoTokenizer
from typing import List, Dict, Any, Optional
from config import Config
import logging

logging.basicConfig(level=logging.INFO)

class LocalReranker:
    """
    Local Cross-Encoder reranker for final precision ranking.
    Compatible with standard HuggingFace cross-encoders (e.g., MiniLM, BERT).
    """
    
    def __init__(self, model_path: str = "./re-rank", 
                 cache_dir: str = "."):
        self.model_path = model_path
        self.cache_dir = cache_dir
                
        # Optimization: Set threads for CPU inference
        if Config.DEVICE == "cpu":
            # Use a reasonable number of threads, not all to avoid contention
            num_threads = min(os.cpu_count() or 4, 8)
            torch.set_num_threads(num_threads)

        # Load model and tokenizer
        logging.info(f"Loading reranker tokenizer from {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            cache_dir=cache_dir,
            local_files_only=True
        )
        
        logging.info(f"Loading reranker model from {model_path} (device={Config.DEVICE})")
        self.model = AutoModel.from_pretrained(
                model_path,
                cache_dir=cache_dir,
                dtype=torch.float32,
                local_files_only=True
            )
        self._is_cross_encoder = False
        self.model.eval()
        self.model.to(Config.DEVICE)
        
        # CPU Optimization: Dynamic Quantization
        if Config.DEVICE == "cpu":
            try:
                # Quantize Linear layers to int8 for 2-4x speedup on CPU
                self.model = torch.quantization.quantize_dynamic(
                    self.model, {torch.nn.Linear}, dtype=torch.qint8
                )
                logging.info("Reranker model quantized for CPU")
            except Exception as e:
                logging.warning(f"Failed to quantize reranker: {e}")

    def rerank(self, query: str, documents: List[str], top_n: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Rerank documents based on relevance to query.

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

        # Construct pairs for cross-encoder
        pairs = [[query, doc] for doc in documents]
        
        # If model supports sequence classification (logits), use cross-encoder path.
        try:
            if self._is_cross_encoder:
                with torch.inference_mode():
                    # Tokenize all pairs
                    features = self.tokenizer(
                        pairs,
                        padding=True,
                        truncation=True,
                        return_tensors="pt",
                        max_length=512
                    ).to(self.model.device)

                    # Get logits
                    outputs = self.model(**features)
                    logits = getattr(outputs, 'logits', None)

                    if logits is None:
                        raise RuntimeError("Model did not return logits; falling back")

                    # Handle different output shapes (binary classification vs single logit)
                    if logits.shape[1] > 1:
                        # Assume classification (label 1 is relevant)
                        scores = torch.softmax(logits, dim=1)[:, 1]
                    else:
                        # Assume regression/single logit
                        scores = logits.view(-1)

                    # Move scores to CPU
                    scores = scores.cpu().float().numpy()
            else:
                raise RuntimeError("Cross-encoder not available; using bi-encoder fallback")
        except Exception:
            # Fallback: bi-encoder style scoring using pooled embeddings + cosine similarity
            logging.info("Using bi-encoder fallback for reranking (pooled embeddings + cosine similarity)")

            def _pool_embeddings(model, inputs):
                out = model(**inputs)
                last_hidden = getattr(out, 'last_hidden_state', None)
                pooled = getattr(out, 'pooler_output', None)
                if pooled is not None:
                    return pooled
                if last_hidden is None:
                    raise RuntimeError('Model did not return usable hidden states')
                # Mean pooling with attention
                mask = inputs['attention_mask'].unsqueeze(-1).expand(last_hidden.size()).float()
                summed = (last_hidden * mask).sum(1)
                counts = mask.sum(1).clamp(min=1e-9)
                return summed / counts

            # Encode query
            with torch.inference_mode():
                q_inputs = self.tokenizer([query], padding=True, truncation=True, return_tensors='pt', max_length=512).to(self.model.device)
                q_vec = _pool_embeddings(self.model, q_inputs)
                q_vec = F.normalize(q_vec, p=2, dim=1)

                # Encode documents in a batch
                doc_inputs = self.tokenizer(documents, padding=True, truncation=True, return_tensors='pt', max_length=512).to(self.model.device)
                d_vecs = _pool_embeddings(self.model, doc_inputs)
                d_vecs = F.normalize(d_vecs, p=2, dim=1)

                # Cosine similarities
                scores_tensor = torch.matmul(d_vecs, q_vec.t()).view(-1)
                scores = scores_tensor.cpu().float().numpy()

        # Build results and sort
        results = []
        for i, score in enumerate(scores):
            results.append({
                'index': i,
                'relevance_score': float(score),
                'document': documents[i]
            })
            
        # Sort by score descending
        results.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        return results[:top_n]