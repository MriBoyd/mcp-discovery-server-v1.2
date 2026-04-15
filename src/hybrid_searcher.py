import time
# src/hybrid_searcher.py
import numpy as np
from rank_bm25 import BM25Okapi
from typing import List, Dict, Any, Tuple, Optional
from tqdm import tqdm
import logging

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from code_embedder import CodeEmbedder
from local_reranker import LocalReranker
from config import Config

class HybridToolSearcher:
    """
    Three-stage hybrid search:
    1. BM25 for exact lexical matching (tool names, parameters)
    2. Dense semantic search with Qdrant + Jina Code Embeddings
    3. Cross-encoder reranking for final precision
    """
    
    def __init__(self):
        self.bm25 = None
        self.tools = []
        self.tool_texts = []
        self.is_indexed = False
        
        # Initialize embedder and reranker
        self.embedder = CodeEmbedder(
            model_path=Config.CODE_EMBEDDING_MODEL,
            cache_dir=Config.MODEL_CACHE_DIR
        )
        self.reranker = LocalReranker(
            model_path=Config.RERANKER_MODEL,
            cache_dir=Config.MODEL_CACHE_DIR
        )
        
        # Initialize Qdrant Client
        self.qdrant = QdrantClient(url=Config.QDRANT_URL)
        self._ensure_collection()
        
    def _ensure_collection(self):
        """Ensure Qdrant collection exists and load tool data if needed"""
        collections = self.qdrant.get_collections().collections
        exists = any(c.name == Config.QDRANT_COLLECTION for c in collections)
        
        if not exists:
            self.qdrant.create_collection(
                collection_name=Config.QDRANT_COLLECTION,
                vectors_config=VectorParams(
                    size=Config.EMBEDDING_DIM,
                    distance=Distance.COSINE
                )
            )
        else:
            # Load tools from existing collection to populate self.tools and self.tool_texts
            # This is a simple approach: scroll all points
            points, _ = self.qdrant.scroll(collection_name=Config.QDRANT_COLLECTION, limit=1000)
            
            # Sort points by ID
            points.sort(key=lambda p: p.id)
            
            
            
            # replace current line that sets self.tools
            self.tools = [
                p.payload.get("tool", {"name": p.payload.get("name", "Unknown"),
                                    "description": p.payload.get("text", "")})
                for p in points if p.payload
            ]
            self.tool_texts = [p.payload.get("text", "") for p in points if p.payload]

            # Rebuild BM25
            tokenized_corpus = [text.lower().split() for text in self.tool_texts]
            self.bm25 = BM25Okapi(tokenized_corpus)
            self.is_indexed = True
    
    def _prepare_tool_text(self, tool: Dict[str, Any]) -> str:
        """
        Prepare a compressed, signature-like representation for a tool.
        Emphasizes parameter names and types for better capability matching.
        """
        name = tool.get('name', 'unknown')
        desc = tool.get('description', '')
        
        # Build parameter signature: param:type
        params = tool.get('parameters', {})
        required = tool.get('required', [])
        
        param_parts = []
        for p_name, p_info in params.items():
            p_type = p_info.get('type', 'any')
            is_req = "*" if p_name in required else ""
            param_parts.append(f"{p_name}{is_req}:{p_type}")
        
        sig = f"{name}({', '.join(param_parts)})"
        
        # Combine signature with a concise description
        return f"Tool: {sig}\nDescription: {desc}"
    
    def index(self, tools: List[Dict[str, Any]], force_reindex: bool = False):
        """
        Index all tools with BM25 and Qdrant
        """
        self.tools = tools
        self.tool_texts = [self._prepare_tool_text(tool) for tool in tools]
        
        # 1. Build BM25 index (lexical)
        tokenized_corpus = [text.lower().split() for text in self.tool_texts]
        self.bm25 = BM25Okapi(tokenized_corpus)
        
        # 2. Check if Qdrant already has data
        collection_info = self.qdrant.get_collection(Config.QDRANT_COLLECTION)
        points_count = collection_info.points_count or 0
        if points_count > 0 and not force_reindex:
            self.is_indexed = True
            return

        # 3. Generate embeddings and upload to Qdrant
        embeddings_list = self.embedder.batch_encode_tools(self.tool_texts)
        
        # when building PointStruct points
        points = [
            PointStruct(
                id=i,
                vector=embeddings_list[i],
                payload={
                    "name": tools[i]['name'],
                    "text": self.tool_texts[i],
                    "tool": tools[i]   # << include full tool
                }
            )
            for i in range(len(tools))
        ]
        
        # Batch upload to Qdrant
        self.qdrant.upsert(
            collection_name=Config.QDRANT_COLLECTION,
            points=points
        )
        
        self.is_indexed = True
            
    def _bm25_search(self, query: str, top_k: Optional[int] = None) -> List[Tuple[float, int]]:
        """BM25 lexical search with bounds checking"""
        if self.bm25 is None:
            return []
        
        if top_k is None:
            top_k = Config.BM25_CANDIDATES
        
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        
        num_scores = len(scores)
        if num_scores == 0:
            return []
        
        actual_k = max(1, min(top_k, num_scores))
        
        if actual_k < num_scores:
            top_indices = np.argpartition(scores, -actual_k)[-actual_k:]
            top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
        else:
            top_indices = np.argsort(scores)[::-1]
        
        return [(float(scores[idx]), int(idx)) for idx in top_indices if scores[idx] > 0]
    
    def _dense_search(self, query: str, top_k: Optional[int] = None) -> List[Tuple[float, int]]:
        """Dense semantic search using Qdrant"""
        if top_k is None:
            top_k = Config.DENSE_CANDIDATES
        
        # Encode query
        query_embedding = self.embedder.encode_query(query)
        
        # Search Qdrant
        search_result = self.qdrant.query_points(
            collection_name=Config.QDRANT_COLLECTION,
            query=query_embedding,
            limit=top_k
        )
        
        return [(hit.score, int(hit.id)) for hit in search_result.points]
    
    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """Normalize scores to [0, 1] range using min-max scaling."""
        if not scores:
            return []
        
        scores_array = np.array(scores)
        min_score = np.min(scores_array)
        max_score = np.max(scores_array)
        
        if max_score == min_score:
            return [0.5] * len(scores)
        
        normalized = (scores_array - min_score) / (max_score - min_score)
        return normalized.tolist()
    
    def _fuse_scores(self, bm25_results: List[Tuple[float, int]], 
                     dense_results: List[Tuple[float, int]],
                     normalization: str = "minmax"
                     ) -> List[Tuple[float, int]]:
        """
        Fuse BM25 and dense scores using weighted sum
        """
        all_indices = set()
        for _, idx in bm25_results:
            all_indices.add(idx)
        for _, idx in dense_results:
            all_indices.add(idx)
        
        if not all_indices:
            return []

        # Create score maps
        bm25_scores_by_idx = {idx: score for score, idx in bm25_results}
        dense_scores_by_idx = {idx: score for score, idx in dense_results}
        
        indices_list = list(all_indices)
        bm25_vector = [bm25_scores_by_idx.get(idx, 0) for idx in indices_list]
        dense_vector = [dense_scores_by_idx.get(idx, 0) for idx in indices_list]
        
        # Normalize
        bm25_normalized = self._normalize_scores(bm25_vector)
        dense_normalized = self._normalize_scores(dense_vector)
        
        # Weighted fusion
        fused_results = []
        for i, idx in enumerate(indices_list):
            fused_score = (Config.BM25_WEIGHT * bm25_normalized[i] + 
                          Config.DENSE_WEIGHT * dense_normalized[i])
            fused_results.append((fused_score, idx))
        
        fused_results.sort(key=lambda x: x[0], reverse=True)
        return fused_results[:Config.FUSION_CANDIDATES]
    
    def search(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """Complete hybrid search pipeline"""
        if not self.is_indexed:
            raise RuntimeError("Must call index() before search()")
        
        if top_k is None:
            top_k = Config.FINAL_RESULTS
        
        start_time = time.time()
        
        # Stage 1: BM25 lexical search
        bm25_results = self._bm25_search(query)
        
        # Stage 2: Dense semantic search (Qdrant)
        dense_results = self._dense_search(query)
        
        # Stage 3: Score fusion
        fused_results = self._fuse_scores(bm25_results, dense_results)
        
        if not fused_results:
            return []
        
        # Stage 4: Rerank with cross-encoder
        candidate_indices = [idx for _, idx in fused_results if idx < len(self.tool_texts)]
        candidate_documents = [self.tool_texts[idx] for idx in candidate_indices]
        
        if not candidate_documents:
            return []
            
        actual_top_k = min(top_k, len(candidate_documents))
        reranked = self.reranker.rerank(query, candidate_documents, top_n=actual_top_k)
        
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
        
        return final_results
