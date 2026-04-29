import time
# src/hybrid_searcher.py
import numpy as np
from rank_bm25 import BM25Okapi
from typing import List, Dict, Any, Tuple, Optional
from tqdm import tqdm
import uuid
import logging

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from concurrent.futures import ThreadPoolExecutor
import re
from code_embedder import CodeEmbedder
from local_reranker import LocalReranker
from config import Config

from collections import OrderedDict

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
        
        # LRU Cache settings
        self._search_cache = OrderedDict()
        self._max_cache_size = Config.SEARCH_CACHE_SIZE if hasattr(Config, 'SEARCH_CACHE_SIZE') else 100
        self._cache_ttl = Config.SEARCH_CACHE_TTL if hasattr(Config, 'SEARCH_CACHE_TTL') else 3600
        
        # Initialize regex for tokenization
        self.token_pattern = re.compile(r'[^a-zA-Z0-9]')
        self.camel_pattern = re.compile(r'([a-z])([A-Z])')
        
        # Initialize embedder and reranker
        self.embedder = CodeEmbedder(
            model_path=Config.CODE_EMBEDDING_MODEL,
            cache_dir=Config.MODEL_CACHE_DIR
        )
        # Ensure Config.EMBEDDING_DIM matches the loaded model's output dimension
        try:
            model_conf = getattr(self.embedder.model, 'config', None)
            model_dim = None
            if model_conf is not None:
                model_dim = getattr(model_conf, 'hidden_size', None) or getattr(model_conf, 'dim', None) or getattr(model_conf, 'n_embd', None)
            if model_dim:
                if getattr(Config, 'EMBEDDING_DIM', None) != model_dim:
                    logging.warning(f"Config.EMBEDDING_DIM ({getattr(Config, 'EMBEDDING_DIM', None)}) != model dim ({model_dim}); updating Config.EMBEDDING_DIM to model dim")
                    Config.EMBEDDING_DIM = model_dim
        except Exception as e:
            logging.warning(f"Could not determine embedder model dimension: {e}")
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
        """Ensure Qdrant collection exists and matches the required dimension.
        If a dimension mismatch is detected, the collection is recreated.
        """
        try:
            collections = self.qdrant.get_collections().collections
            exists = any(c.name == Config.QDRANT_COLLECTION for c in collections)
            
            if exists:
                # Validate dimension
                collection_info = self.qdrant.get_collection(Config.QDRANT_COLLECTION)
                # Some Qdrant versions might have different structures; handle with care
                current_size = getattr(collection_info.config.params.vectors, 'size', None)
                if current_size is None and hasattr(collection_info.config.params.vectors, '__dict__'):
                     # Fallback for different client versions
                     current_size = collection_info.config.params.vectors.size
                
                if current_size and current_size != Config.EMBEDDING_DIM:
                    logging.warning(
                        f"Collection {Config.QDRANT_COLLECTION} dimension mismatch: "
                        f"existing={current_size}, model={Config.EMBEDDING_DIM}. Recreating."
                    )
                    self.qdrant.delete_collection(Config.QDRANT_COLLECTION)
                    exists = False
            
            if not exists:
                logging.info(f"Creating Qdrant collection '{Config.QDRANT_COLLECTION}' with size {Config.EMBEDDING_DIM}")
                self.qdrant.create_collection(
                    collection_name=Config.QDRANT_COLLECTION,
                    vectors_config=VectorParams(
                        size=Config.EMBEDDING_DIM,
                        distance=Distance.COSINE
                    )
                )
        except Exception as e:
            logging.error(f"Failed to ensure Qdrant collection: {e}")

    def _get_from_cache(self, key: str) -> Optional[List[Dict[str, Any]]]:
        """Get item from LRU cache with TTL check"""
        if key not in self._search_cache:
            return None
        
        timestamp, result = self._search_cache[key]
        if time.time() - timestamp > self._cache_ttl:
            del self._search_cache[key]
            return None
            
        self._search_cache.move_to_end(key)
        return result

    def _save_to_cache(self, key: str, value: List[Dict[str, Any]]):
        """Save item to LRU cache"""
        if key in self._search_cache:
            self._search_cache.move_to_end(key)
        
        self._search_cache[key] = (time.time(), value)
        
        if len(self._search_cache) > self._max_cache_size:
            self._search_cache.popitem(last=False)

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
        Index tools incrementally. Only embeds tools not already in Qdrant.
        """        
        self.tools = tools
        self.tool_texts = [self._prepare_tool_text(tool) for tool in tools]
        
        # 1. Build BM25 index (lexical) - must be in memory
        tokenized_corpus = [self._tokenize(text) for text in self.tool_texts]
        self.bm25 = BM25Okapi(tokenized_corpus)
        
        # 2. Identify missing tools in Qdrant
        indexed_names = set()
        if not force_reindex:
            try:
                # Fetch only names to check existence efficiently
                offset = None
                while True:
                    points, offset = self.qdrant.scroll(
                        collection_name=Config.QDRANT_COLLECTION,
                        limit=1000,
                        offset=offset,
                        with_payload=["name"],
                        with_vectors=False
                    )
                    for p in points:
                        if p.payload and "name" in p.payload:
                            indexed_names.add(p.payload["name"])
                    if offset is None:
                        break
            except Exception as e:
                logging.warning(f"Incremental check failed, may re-embed some tools: {e}")

        # 3. Filter tools that need embedding
        to_index_indices = []
        for i, tool in enumerate(tools):
            if force_reindex or tool['name'] not in indexed_names:
                to_index_indices.append(i)
        
        if not to_index_indices:
            logging.info("All tools already indexed in Qdrant.")
            self.is_indexed = True
            self._search_cache.clear()
            return

        logging.info(f"Embedding {len(to_index_indices)} new/missing tools...")
        
        # 4. Generate embeddings for ONLY the new tools
        texts_to_embed = [self.tool_texts[i] for i in to_index_indices]
        # batch_encode_tools handles batching internally
        embeddings_list = self.embedder.batch_encode_tools(texts_to_embed, batch_size=16)
        
        def str_to_uuid(s):
            # Deterministic UUID based on tool name
            return str(uuid.uuid5(uuid.NAMESPACE_DNS, s))

        points = []
        for j, i in enumerate(to_index_indices):
            points.append(PointStruct(
                id=str_to_uuid(tools[i]['name']),
                vector=embeddings_list[j],
                payload={
                    "name": tools[i]['name'],
                    "index": i,
                    "text": self.tool_texts[i],
                    "tool": tools[i]
                }
            ))
        
        # 5. Batch upload to Qdrant
        for i in range(0, len(points), 100):
            self.qdrant.upsert(
                collection_name=Config.QDRANT_COLLECTION,
                points=points[i:i+100]
            )
        
        self.is_indexed = True
        self._search_cache.clear()

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
            limit=top_k,
            with_payload=True
        )

        results = []
        for hit in search_result.points:
            idx = None
            # Prefer explicit payload index
            if hasattr(hit, 'payload') and hit.payload:
                if 'index' in hit.payload:
                    try:
                        idx = int(hit.payload['index'])
                    except Exception:
                        idx = None
                elif 'name' in hit.payload:
                    # fallback: map name to index
                    try:
                        name = hit.payload['name']
                        for i, t in enumerate(self.tools):
                            if t.get('name') == name:
                                idx = i
                                break
                    except Exception:
                        idx = None

            # Last resort: try converting id to int (some collections may use numeric ids)
            if idx is None:
                try:
                    idx = int(hit.id)
                except Exception:
                    # cannot determine index; skip this hit
                    continue

            results.append((hit.score, int(idx)))

        return results
    
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
        cached_res = self._get_from_cache(cache_key)
        if cached_res:
            return cached_res

        # 1. Fast Path: Exact name match short-circuit
        for i, tool in enumerate(self.tools):
            if tool['name'].lower() == query_stripped:
                res = [{
                    'tool_name': tool['name'],
                    'tool_description': tool.get('description', ''),
                    'tool_schema': tool,
                    'relevance_score': 1.0
                }]
                self._save_to_cache(cache_key, res)
                return res

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
        
        self._save_to_cache(cache_key, final_results)
        
        return final_results
