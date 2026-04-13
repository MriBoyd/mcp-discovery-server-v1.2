import time
# src/hybrid_searcher.py
import numpy as np
from rank_bm25 import BM25Okapi
from typing import List, Dict, Any, Tuple, Optional
from tqdm import tqdm
import logging

from src.code_embedder import CodeEmbedder
from src.local_reranker import LocalReranker
from src.config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HybridToolSearcher:
    """
    Three-stage hybrid search:
    1. BM25 for exact lexical matching (tool names, parameters)
    2. Dense semantic search with Jina Code Embeddings
    3. Cross-encoder reranking for final precision
    """
    
    def __init__(self):
        self.bm25 = None
        self.tools = []
        self.tool_texts = []
        self.tool_embeddings = None
        
        # ← Initialize embedder immediately
        self.embedder = CodeEmbedder(
            model_path=Config.CODE_EMBEDDING_MODEL,
            cache_dir=Config.MODEL_CACHE_DIR
        )
        self.reranker = LocalReranker(
            model_path=Config.RERANKER_MODEL,
            cache_dir=Config.MODEL_CACHE_DIR
        )
        
    
    def _prepare_tool_text(self, tool: Dict[str, Any]) -> str:
        """
        Prepare rich text representation for a tool
        Optimized for both BM25 and dense retrieval
        """
        parts = [
            f"Function: {tool['name']}",
            f"Description: {tool['description']}"
        ]
        
        # Add parameters with their types (critical for exact matching)
        if tool.get('parameters'):
            param_lines = ["Parameters:"]
            for param_name, param_info in tool['parameters'].items():
                param_type = param_info.get('type', 'unknown')
                param_desc = param_info.get('description', '')
                param_lines.append(f"  - {param_name} ({param_type}): {param_desc}")
            parts.extend(param_lines)
        
        # Add required parameters
        if tool.get('required'):
            parts.append(f"Required parameters: {', '.join(tool['required'])}")
        
        return "\n".join(parts)
    
    def index(self, tools: List[Dict[str, Any]]):
        """
        Index all tools with BM25 and dense embeddings
        """
        self.tools = tools
        
        # Prepare searchable texts
        self.tool_texts = [self._prepare_tool_text(tool) for tool in tools]
        
        # Build BM25 index (lexical)
        tokenized_corpus = [text.lower().split() for text in self.tool_texts]
        self.bm25 = BM25Okapi(tokenized_corpus)

        embeddings_list = self.embedder.batch_encode_tools(self.tool_texts)
        
        # Initialize embedder and generate dense vectors
        
        self.tool_embeddings = np.array(embeddings_list, dtype=np.float32)        
        self.is_indexed = True
        # Initialize reranker
        
            
    def _bm25_search(self, query: str, top_k: Optional[int] = None) -> List[Tuple[float, int]]:
        """
        BM25 lexical search
        Returns list of (score, index) tuples
        """
        if top_k is None:
            top_k = Config.BM25_CANDIDATES
        
        if self.bm25 is None:
            logger.warning("BM25 not initialized. Call index() first.")
            return []
        
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top-k indices
        top_indices = np.argpartition(scores, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
        
        return [(float(scores[idx]), int(idx)) for idx in top_indices if scores[idx] > 0]
    
    def _dense_search(self, query: str, top_k: int = 5) -> List[Tuple[float, int]]:
        """
        Dense semantic search with Jina Code Embeddings
        Returns list of (score, index) tuples
        """
        
        """Dense semantic search"""

        if self.tool_embeddings is None:
            raise RuntimeError("Must call index() before search()")
        
        if not isinstance(self.tool_embeddings, np.ndarray):
            logger.warning("Converting tool_embeddings to numpy array...")
            self.tool_embeddings = np.array(self.tool_embeddings, dtype=np.float32)
    
        
        if top_k is None:
            top_k = Config.DENSE_CANDIDATES
        
        # Encode query
        query_embedding = self.embedder.encode_query(query)
        query_vec = np.array(query_embedding)
        
        # Compute cosine similarities
        similarities = np.dot(self.tool_embeddings, query_vec)
        if len(similarities) > top_k:
            top_indices = np.argpartition(similarities, -top_k)[-top_k:]
            top_indices = top_indices[np.argsort(similarities[top_indices])[::-1]]
        else:
            top_indices = np.argsort(similarities)[::-1]
        
        return [(float(similarities[idx]), int(idx)) for idx in top_indices]
    
    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """
        Normalize scores to [0, 1] range using min-max scaling.
        Preserves ranking while making scores comparable across methods.
        """
        if not scores:
            return []
        
        scores_array = np.array(scores)
        min_score = np.min(scores_array)
        max_score = np.max(scores_array)
        
        if max_score == min_score:
            return [0.5] * len(scores)  # Default to neutral if all equal
        
        normalized = (scores_array - min_score) / (max_score - min_score)
        return normalized.tolist()
    
    def _normalize_scores_softmax(self, scores: List[float], temperature: float = 1.0) -> List[float]:
        """
        Alternative: Softmax normalization for probability distribution.
        Better when you want to emphasize top results.
        """
        if not scores:
            return []
        
        scores_array = np.array(scores)
        exp_scores = np.exp(scores_array / temperature)
        normalized = exp_scores / np.sum(exp_scores)
        return normalized.tolist()
    
    def _normalize_scores_rank(self, scores: List[float]) -> List[float]:
        """
        Alternative: Rank-based normalization (1 - rank/total).
        Robust to outliers but loses score magnitude information.
        """
        if not scores:
            return []
        
        # Get ranks (1 = highest score)
        sorted_indices = np.argsort(scores)[::-1]
        ranks = np.zeros(len(scores))
        for rank, idx in enumerate(sorted_indices):
            ranks[idx] = rank + 1
        
        # Convert rank to score (1 - normalized rank)
        normalized = 1 - (ranks - 1) / len(scores)
        return normalized.tolist()
    
    
    def _fuse_scores(self, bm25_results: List[Tuple[float, int]], 
                     dense_results: List[Tuple[float, int]],
                     normalization: str = "minmax"
                     ) -> List[Tuple[float, int]]:
        """
        Fuse BM25 and dense scores using weighted sum
        Returns list of (fused_score, index) sorted descending
        """
        all_indices = set()
        score_map = {}
        
        
        for _, idx in bm25_results:
            all_indices.add(idx)
        for _, idx in dense_results:
            all_indices.add(idx)
        
        # Create score vectors for normalization
        bm25_scores_by_idx = {}
        dense_scores_by_idx = {}
        
        # Add BM25 scores
        for score, idx in bm25_results:
            bm25_scores_by_idx[idx] = score
        
        for score, idx in dense_results:
            dense_scores_by_idx[idx] = score
        
        # Build aligned score lists for normalization
        indices_list = list(all_indices)
        bm25_vector = [bm25_scores_by_idx.get(idx, 0) for idx in indices_list]
        dense_vector = [dense_scores_by_idx.get(idx, 0) for idx in indices_list]
        
        # Normalize scores
        if normalization == "minmax":
            bm25_normalized = self._normalize_scores(bm25_vector)
            dense_normalized = self._normalize_scores(dense_vector)
        elif normalization == "softmax":
            bm25_normalized = self._normalize_scores_softmax(bm25_vector, temperature=0.5)
            dense_normalized = self._normalize_scores_softmax(dense_vector, temperature=0.5)
        elif normalization == "rank":
            bm25_normalized = self._normalize_scores_rank(bm25_vector)
            dense_normalized = self._normalize_scores_rank(dense_vector)
        else:
            raise ValueError(f"Unknown normalization: {normalization}")
        
        # Weighted fusion with normalized scores
        fused_results = []
        for i, idx in enumerate(indices_list):
            fused_score = (Config.BM25_WEIGHT * bm25_normalized[i] + 
                          Config.DENSE_WEIGHT * dense_normalized[i])
            fused_results.append((fused_score, idx))
        
        # Sort by fused score descending
        fused_results.sort(key=lambda x: x[0], reverse=True)
        
        return fused_results[:Config.FUSION_CANDIDATES]
    
    def _fuse_scores_reciprocal_rank(self, 
                                     bm25_results: List[Tuple[float, int]], 
                                     dense_results: List[Tuple[float, int]],
                                     k: int = 60) -> List[Tuple[float, int]]:
        """
        Alternative: Reciprocal Rank Fusion (RRF).
        Doesn't require score normalization, only ranks.
        Often works better than weighted fusion for search.
        """
        rrf_scores = {}
        
        # Process BM25 results
        for rank, (_, idx) in enumerate(bm25_results, start=1):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (k + rank)
        
        # Process dense results
        for rank, (_, idx) in enumerate(dense_results, start=1):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (k + rank)
        
        # Convert to list and sort
        fused = [(score, idx) for idx, score in rrf_scores.items()]
        fused.sort(key=lambda x: x[0], reverse=True)
        
        return fused[:Config.FUSION_CANDIDATES]
    
    def search(self, query: str, top_k: Optional[int] = None, fusion_method: str = "rrf") -> List[Dict[str, Any]]:
        """
        Complete hybrid search pipeline.
        
        Args:
            query: Search query
            top_k: Number of results to return (default 5)
        """
        start_time = time.time()
        
        if not self.is_indexed:
            raise RuntimeError("Must call index() before search()")
    
        
        if top_k is None:
            top_k = Config.FINAL_RESULTS
        
        # Stage 1: BM25 lexical search
        bm25_results = self._bm25_search(query)
           
        # Stage 2: Dense semantic search
        dense_results = self._dense_search(query)
       
        # Stage 3: Score fusion
        if fusion_method == "rrf":
            # Reciprocal Rank Fusion (recommended)
            fused_results = self._fuse_scores_reciprocal_rank(bm25_results, dense_results)
        elif fusion_method == "weighted":
            # Weighted with min-max normalization
            fused_results = self._fuse_scores(bm25_results, dense_results, normalization="minmax")
        elif fusion_method == "weighted_rank":
            # Weighted with rank normalization
            fused_results = self._fuse_scores(bm25_results, dense_results, normalization="rank")
        else:
            raise ValueError(f"Unknown fusion_method: {fusion_method}")
        
          
        if not fused_results:
            return []
        
        # Stage 4: Rerank with cross-encoder
        candidate_indices = [idx for _, idx in fused_results]
        candidate_documents = [self.tool_texts[idx] for idx in candidate_indices]
        reranked = self.reranker.rerank(
            query, 
            candidate_documents, 
            top_n=top_k  # Use parameter here
        )
                
        # Build final results
        # Build final results
        final_results = []
        for rerank_item in reranked:
            original_idx = candidate_indices[rerank_item['index']]
            tool = self.tools[original_idx]
            
            final_results.append({
                'tool_name': tool['name'],
                'tool_description': tool['description'],
                'tool_schema': tool,
                'relevance_score': rerank_item['relevance_score']
            })
        
        logger.debug(f"Search completed in {time.time() - start_time:.3f}s")
        
        return final_results[:top_k]