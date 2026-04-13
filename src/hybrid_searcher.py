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
        logger.info(f"Indexing {len(tools)} tools...")
        self.tools = tools
        
        # Prepare searchable texts
        logger.info("Preparing tool texts...")
        self.tool_texts = [self._prepare_tool_text(tool) for tool in tools]
        
        # Build BM25 index (lexical)
        logger.info("Building BM25 index...")
        tokenized_corpus = [text.lower().split() for text in self.tool_texts]
        self.bm25 = BM25Okapi(tokenized_corpus)
        
        # Initialize embedder and generate dense vectors
        logger.info("Initializing Code Embedder...")
        
        logger.info("Generating dense embeddings...")
        self.tool_embeddings = self.embedder.batch_encode_tools(self.tool_texts)
        
        # Initialize reranker
        logger.info("Initializing Reranker...")
        
        
        logger.info(f"Indexing complete! {len(self.tools)} tools ready.")
    
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
    
    def _fuse_scores(self, bm25_results: List[Tuple[float, int]], 
                     dense_results: List[Tuple[float, int]]) -> List[Tuple[float, int]]:
        """
        Fuse BM25 and dense scores using weighted sum
        Returns list of (fused_score, index) sorted descending
        """
        score_map = {}
        
        # Add BM25 scores
        for score, idx in bm25_results:
            score_map[idx] = Config.BM25_WEIGHT * score
        
        # Add dense scores
        for score, idx in dense_results:
            if idx in score_map:
                score_map[idx] += Config.DENSE_WEIGHT * score
            else:
                score_map[idx] = Config.DENSE_WEIGHT * score
        
        # Convert to list and sort
        fused = [(score, idx) for idx, score in score_map.items()]
        fused.sort(key=lambda x: x[0], reverse=True)
        
        return fused[:Config.FUSION_CANDIDATES]
    
    def search(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Complete hybrid search pipeline.
        
        Args:
            query: Search query
            top_k: Number of results to return (default 5)
        """
        start_time = time.time()
        
        
        if top_k is None:
            top_k = Config.FINAL_RESULTS
        
        # Stage 1: BM25 lexical search
        bm25_start = time.time()
        bm25_results = self._bm25_search(query)
           
        # Stage 2: Dense semantic search
        dense_start = time.time()
        dense_results = self._dense_search(query)
       
        # Stage 3: Score fusion
        fused_results = self._fuse_scores(bm25_results, dense_results)
          
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