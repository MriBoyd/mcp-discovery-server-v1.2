import time
# src/hybrid_searcher.py
import numpy as np
from rank_bm25 import BM25Okapi
from typing import List, Dict, Any, Tuple, Optional
from tqdm import tqdm
import logging

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from src.code_embedder import CodeEmbedder
from src.local_reranker import LocalReranker
from src.config import Config

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
        self._search_cache = {}
        self._max_cache_size = 100
        
        # Initialize regex for tokenization
        import re
        self.token_pattern = re.compile(r'[^a-zA-Z0-9]')
        self.camel_pattern = re.compile(r'([a-z])([A-Z])')
        
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
        
    def _tokenize(self, text: str) -> List[str]:
        """Robust tokenization for lexical search"""
        if not text:
            return []
        # Handle camelCase
        text = self.camel_pattern.sub(r'\1 \2', text)
        # Split on non-alphanumeric characters
        tokens = self.token_pattern.split(text)
        return [t.lower() for t in tokens if t]

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
            # Use scrolling to retrieve all points
            all_points = []
            next_page_offset = None
            
            while True:
                points, next_page_offset = self.qdrant.scroll(
                    collection_name=Config.QDRANT_COLLECTION,
                    limit=1000,
                    offset=next_page_offset,
                    with_payload=True,
                    with_vectors=False
                )
                all_points.extend(points)
                if next_page_offset is None:
                    break
            
            # Sort points by ID if they were indexed sequentially
            all_points.sort(key=lambda p: p.id if isinstance(p.id, int) else 0)
            
            self.tools = [
                p.payload.get("tool", {"name": p.payload.get("name", "Unknown"),
                                    "description": p.payload.get("text", "")})
                for p in all_points if p.payload
            ]
            self.tool_texts = [p.payload.get("text", "") for p in all_points if p.payload]

            if self.tool_texts:
                # Rebuild BM25
                tokenized_corpus = [self._tokenize(text) for text in self.tool_texts]
                self.bm25 = BM25Okapi(tokenized_corpus)
                self.is_indexed = True
                self._search_cache = {} # Clear cache on reload
    
    def _prepare_tool_text(self, tool: Dict[str, Any]) -> str:
        """
        Prepare a compressed, signature-like representation for a tool.
        Emphasizes parameter names and types for better capability matching.
        """
        name = tool.get('name', 'unknown')
        desc = tool.get('description', '')
        
        # Limit description length for retrieval quality/speed balance
        if len(desc) > 1000:
            desc = desc[:997] + "..."
            
        # Build parameter signature: param:type
        params = tool.get('parameters', {})
        if not params and 'inputSchema' in tool:
            params = tool['inputSchema'].get('properties', {})
            
        required = tool.get('required', [])
        if not required and 'inputSchema' in tool:
            required = tool['inputSchema'].get('required', [])
        
        param_parts = []
        # Limit number of parameters in the signature to prevent 8K+ tool bloat
        for i, (p_name, p_info) in enumerate(params.items()):
            if i >= 15: # Only show first 15 params in signature
                param_parts.append("...")
                break
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
        tokenized_corpus = [self._tokenize(text) for text in self.tool_texts]
        self.bm25 = BM25Okapi(tokenized_corpus)
        
        # 2. Check if Qdrant already has data
        try:
            collection_info = self.qdrant.get_collection(Config.QDRANT_COLLECTION)
            points_count = collection_info.points_count or 0
        except Exception:
            points_count = 0
            
        if points_count > 0 and not force_reindex:
            self.is_indexed = True
            return

        # 3. Generate embeddings and upload to Qdrant
        # We process in batches to save memory for 8K+ tools
        embeddings_list = self.embedder.batch_encode_tools(self.tool_texts, batch_size=16)
        
        points = [
            PointStruct(
                id=i,
                vector=embeddings_list[i],
                payload={
                    "name": tools[i]['name'],
                    "text": self.tool_texts[i],
                    "tool": tools[i]
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
        self._search_cache = {} # Clear cache on reindex
            
    def _bm25_search(self, query: str, top_k: Optional[int] = None) -> List[Tuple[float, int]]:
        """BM25 lexical search with bounds checking"""
        if self.bm25 is None or not self.tool_texts:
            return []
        
        if top_k is None:
            top_k = Config.BM25_CANDIDATES
        
        tokenized_query = self._tokenize(query)
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
        if not self.is_indexed:
            return []
            
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
            return [1.0] * len(scores) if max_score > 0 else [0.0] * len(scores)
        
        normalized = (scores_array - min_score) / (max_score - min_score)
        return normalized.tolist()
    
    def _fuse_scores(self, bm25_results: List[Tuple[float, int]], 
                     dense_results: List[Tuple[float, int]],
                     method: Optional[str] = None
                     ) -> List[Tuple[float, int]]:
        """
        Fuse BM25 and dense scores using configured method
        """
        if method is None:
            method = Config.FUSION_METHOD
            
        if method == "rrf":
            return self._rrf_fuse(bm25_results, dense_results)
        else:
            return self._weighted_fuse(bm25_results, dense_results)

    def _rrf_fuse(self, bm25_results: List[Tuple[float, int]], 
                  dense_results: List[Tuple[float, int]]) -> List[Tuple[float, int]]:
        """Reciprocal Rank Fusion"""
        k = getattr(Config, 'RRF_K', 60)
        scores = {}
        
        for rank, (_, idx) in enumerate(bm25_results):
            scores[idx] = scores.get(idx, 0) + 1.0 / (k + rank)
            
        for rank, (_, idx) in enumerate(dense_results):
            scores[idx] = scores.get(idx, 0) + 1.0 / (k + rank)
            
        fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [(score, idx) for idx, score in fused][:Config.FUSION_CANDIDATES]

    def _weighted_fuse(self, bm25_results: List[Tuple[float, int]], 
                      dense_results: List[Tuple[float, int]]) -> List[Tuple[float, int]]:
        """Weighted sum of normalized scores"""
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
        
        query_stripped = query.lower().strip()
        
        # 0. Check Cache
        cache_key = f"{query_stripped}:{top_k}"
        if cache_key in self._search_cache:
            return self._search_cache[cache_key]

        # 1. Fast Path: Exact name match short-circuit
        for i, tool in enumerate(self.tools):
            if tool['name'].lower() == query_stripped:
                res = [{
                    'tool_name': tool['name'],
                    'tool_description': tool.get('description', ''),
                    'tool_schema': tool,
                    'relevance_score': 1.0
                }]
                self._search_cache[cache_key] = res
                return res

        from concurrent.futures import ThreadPoolExecutor
        
        start_time = time.time()
        
        # Stage 1 & 2: Parallel BM25 and Dense search
        with ThreadPoolExecutor(max_workers=2) as executor:
            bm25_future = executor.submit(self._bm25_search, query)
            dense_future = executor.submit(self._dense_search, query)
            
            bm25_results = bm25_future.result()
            dense_results = dense_future.result()
        
        # print(f"Hybrid retrieval found {len(bm25_results)} lexical and {len(dense_results)} dense candidates in {time.time() - start_time:.2f}s")
        

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
        
        # Limit reranking if retrieval score is extremely high for top item
        if fused_results[0][0] > 0.95 and len(fused_results) > 1:
            # We still rerank but could potentially optimize here
            pass

        reranked = self.reranker.rerank(query, candidate_documents, top_n=actual_top_k)
        
        # Build final results
        final_results = []
        for rerank_item in reranked:
            original_idx = candidate_indices[rerank_item['index']]
            tool = self.tools[original_idx]
            
            final_results.append({
                'tool_name': tool['name'],
                'tool_description': tool.get('description', ''),
                'tool_schema': tool,
                'relevance_score': rerank_item['relevance_score']
            })
        
        # Manage cache size
        if len(self._search_cache) >= self._max_cache_size:
            # Remove oldest item (FIFO)
            try:
                self._search_cache.pop(next(iter(self._search_cache)))
            except (StopIteration, KeyError):
                pass
        self._search_cache[cache_key] = final_results
        
        return final_results
