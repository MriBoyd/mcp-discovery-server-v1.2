import time
import numpy as np
import json
import itertools
import re
import logging
import uuid
from typing import List, Dict, Any, Tuple, Optional
from collections import OrderedDict

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, models, Document, PointStruct
)
from rank_bm25 import BM25Okapi

class HybridToolSearcher:
    """
    Advanced Three-Stage Hybrid Searcher:
    1. Retrieval: Qdrant Hybrid (Dense + Sparse) + Optional BM25
    2. Rerank: In-database ColBERT Late Interaction
    3. Fusion: 3-way normalized weighted sum (ColBERT + RRF + BM25)
    """

    def __init__(
        self,
        qdrant_url: str = "http://0.0.0.0:6333",
        collection_name: str = "tool-search",
        dense_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        sparse_model: str = "prithivida/Splade_PP_en_v1",
        rerank_model: str = "colbert-ir/colbertv2.0"
    ):
        self.client = QdrantClient(url=qdrant_url)
        self.collection_name = collection_name
        self.dense_model = dense_model
        self.sparse_model = sparse_model
        self.rerank_model = rerank_model

        # State
        self.tools = []
        self.bm25 = None
        self.is_indexed = False

        # Caching
        self._cache = OrderedDict()
        self._max_cache_size = 200
        self._cache_ttl = 3600  # 1 hour

        # Regex for tokenization
        self.token_pattern = re.compile(r'[^a-zA-Z0-9]')
        self.camel_pattern = re.compile(r'([a-z])([A-Z])')

    # =========================
    # CORE UTILS
    # =========================

    def _str_to_uuid(self, s: str) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, s))

    def _normalize_text(self, text: str) -> str:
        return " ".join(text.strip().lower().split())

    def _tokenize(self, text: str) -> List[str]:
        """Robust tokenization for lexical search"""
        if not text:
            return []
        # Handle camelCase
        text = self.camel_pattern.sub(r'\1 \2', text)
        # Split on non-alphanumeric characters
        tokens = self.token_pattern.split(text)
        return [t.lower() for t in tokens if t]

    def _min_max_scale(self, scores: Dict[Any, float]) -> Dict[Any, float]:
        """Normalize scores to [0, 1] range using numpy."""
        if not scores:
            return {}
        if len(scores) == 1:
            return {k: 1.0 for k in scores.keys()}
            
        vals = np.array(list(scores.values()))
        min_s = vals.min()
        max_s = vals.max()
        
        if max_s == min_s:
            return {k: 1.0 for k in scores.keys()}
        
        norm_vals = (vals - min_s) / (max_s - min_s)
        return dict(zip(scores.keys(), norm_vals))

    def _tool_to_text(self, tool: Dict[str, Any]) -> str:
        params = tool.get("parameters", {})
        # Handle production schema keys if present
        if not params and "inputSchema" in tool:
            params = tool["inputSchema"].get("properties", {})

        param_text = ", ".join(
            f"{k}: {v.get('description','')}"
            for k, v in params.items()
        ) if params else "none"

        required = tool.get("required", [])
        if not required and "inputSchema" in tool:
            required = tool["inputSchema"].get("required", [])
        
        required_text = ", ".join(required) if required else "none"

        return self._normalize_text(f"""
        Tool: {tool['name']}
        Server: {tool.get('server', 'unknown')}
        Description: {tool['description']}
        Parameters: {param_text}
        Required parameters: {required_text}
        """)

    # =========================
    # CACHING
    # =========================

    def _get_from_cache(self, key: str) -> Optional[List[Any]]:
        if key not in self._cache:
            return None
        timestamp, value = self._cache[key]
        if time.time() - timestamp > self._cache_ttl:
            del self._cache[key]
            return None
        self._cache.move_to_end(key)
        return value

    def _set_to_cache(self, key: str, value: List[Any]):
        if key in self._cache:
            del self._cache[key]
        self._cache[key] = (time.time(), value)
        while len(self._cache) > self._max_cache_size:
            self._cache.popitem(last=False)

    # =========================
    # INDEXING
    # =========================

    def index(self, tools_data: Dict[str, Any], recreate: bool = False, batch_size: int = 32):
        """
        Main indexing entry point.
        """
        # 1. Flatten tools
        self.tools = []
        for server in tools_data.get("servers", []):
            for tool in server.get("tools", []):
                tool["server"] = server["name"]
                self.tools.append(tool)
        
        if not self.tools:
            logging.warning("No tools found to index.")
            return

        # 2. Setup Qdrant
        if recreate and self.client.collection_exists(self.collection_name):
            self.client.delete_collection(self.collection_name)
        
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                self.collection_name,
                vectors_config={
                    "dense": VectorParams(size=384, distance=Distance.COSINE),
                    "multi": VectorParams(
                        size=128,
                        distance=Distance.COSINE,
                        multivector_config=models.MultiVectorConfig(
                            comparator=models.MultiVectorComparator.MAX_SIM
                        ),
                        hnsw_config=models.HnswConfigDiff(m=0),
                    ),
                },
                sparse_vectors_config={
                    "sparse": models.SparseVectorParams(
                        modifier=models.Modifier.IDF
                    )
                }
            )
            indexed_ids = set()
        else:
            # Incremental scroll
            indexed_ids = set()
            offset = None
            while True:
                points, offset = self.client.scroll(
                    collection_name=self.collection_name,
                    limit=1000,
                    offset=offset,
                    with_payload=False,
                    with_vectors=False
                )
                for p in points:
                    indexed_ids.add(str(p.id))
                if offset is None:
                    break

        # 3. Build BM25
        corpus = [self._tool_to_text(t) for t in self.tools]
        tokenized_corpus = [self._tokenize(doc) for doc in corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)

        # 4. Ingest new tools
        to_index = [t for t in self.tools if self._str_to_uuid(t["name"]) not in indexed_ids]
        
        if to_index:
            logging.info(f"Indexing {len(to_index)} new tools...")
            def batch(iterable, size):
                it = iter(iterable)
                while chunk := list(itertools.islice(it, size)):
                    yield chunk

            def generate_points():
                for tool in to_index:
                    text = self._tool_to_text(tool)
                    yield PointStruct(
                        id=self._str_to_uuid(tool["name"]),
                        vector={
                            "dense": Document(text=text, model=self.dense_model),
                            "sparse": Document(text=text, model=self.sparse_model),
                            "multi": Document(text=text, model=self.rerank_model),
                        },
                        payload=tool
                    )
            
            for chunk in batch(generate_points(), batch_size):
                self.client.upload_points(self.collection_name, points=chunk)
        
        self.is_indexed = True
        self._cache.clear()

    # =========================
    # SEARCH
    # =========================

    async def search(
        self,
        query: str,
        limit: int = 10,
        prefetch_limit: int = 100,
        fusion_limit: int = 100,
        weights: Dict[str, float] = {
            "colbert": 0.6,
            "rrf": 0.25,
            "bm25": 0.15
        }
    ) -> List[Dict[str, Any]]:
        if not self.is_indexed:
            logging.warning("Search called before indexing.")
            return []

        # 0. Cache check
        cache_key = f"{query}:{limit}:{weights}"
        cached = self._get_from_cache(cache_key)
        if cached:
            return cached

        # 1. Qdrant Retrieval (Hybrid: Dense + Sparse)
        dense_q = Document(text=query, model=self.dense_model)
        sparse_q = Document(text=query, model=self.sparse_model)
        rerank_q = Document(text=query, model=self.rerank_model)

        prefetch = [
            models.Prefetch(query=dense_q, using="dense", limit=prefetch_limit),
            models.Prefetch(query=sparse_q, using="sparse", limit=prefetch_limit),
        ]
        
        fused = self.client.query_points(
            self.collection_name,
            prefetch=prefetch,
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=fusion_limit,
        )
        
        qdrant_ids = [p.id for p in fused.points]
        fusion_raw_map = {p.id: p.score for p in fused.points}

        # 2. BM25 Lexical Signal
        bm25_raw_map = {}
        bm25_ids = []
        if self.bm25:
            tokenized_query = self._tokenize(query)
            bm25_scores = self.bm25.get_scores(tokenized_query)
            top_indices = sorted(
                range(len(bm25_scores)),
                key=lambda i: bm25_scores[i],
                reverse=True
            )[:50]
            
            for idx in top_indices:
                tid = self._str_to_uuid(self.tools[idx]["name"])
                bm25_ids.append(tid)
                bm25_raw_map[tid] = bm25_scores[idx]

        # 3. Merge & Rerank (ColBERT)
        candidate_ids = list(set(qdrant_ids + bm25_ids))[:200]
        if not candidate_ids:
            return []

        reranked = self.client.query_points(
            collection_name=self.collection_name,
            query=rerank_q,
            using="multi",
            limit=fusion_limit,
            with_payload=True,
            query_filter=models.Filter(
                must=[models.HasIdCondition(has_id=candidate_ids)]
            )
        )
        
        colbert_raw_map = {r.id: r.score for r in reranked.points}

        # 4. Normalized Fusion
        norm_fusion = self._min_max_scale(fusion_raw_map)
        norm_colbert = self._min_max_scale(colbert_raw_map)
        norm_bm25 = self._min_max_scale(bm25_raw_map) if bm25_raw_map else {}
        
        final_results = []
        for r in reranked.points:
            c_score = norm_colbert.get(r.id, 0)
            f_score = norm_fusion.get(r.id, 0)
            b_score = norm_bm25.get(r.id, 0)
            
            combined_score = (
                weights["colbert"] * c_score +
                weights["rrf"] * f_score +
                (weights["bm25"] * b_score if bm25_raw_map else 0)
            )
            
            # Recalibrate weight if BM25 is empty
            if not bm25_raw_map:
                total_weight = weights["colbert"] + weights["rrf"]
                combined_score /= total_weight
                
            r.score = combined_score
            final_results.append(r)

        final_results.sort(key=lambda x: x.score, reverse=True)
        top_k = final_results[:limit]
        
        # Format output similar to production
        formatted = []
        for p in top_k:
            formatted.append({
                "tool_name": p.payload["name"],
                "tool_description": p.payload.get("description", ""),
                "tool_schema": p.payload,
                "relevance_score": p.score
            })

        self._set_to_cache(cache_key, formatted)
        return formatted

# Integration Demo
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    searcher = HybridToolSearcher()
    searcher.is_indexed = True  

    # Load data
    with open("mcp_servers.json", "r") as f:
        data = json.load(f)

    # Index (Incremental)
    # searcher.index(data)

    # Search
    for query in ["send an email", "schedule a meeting", "fetch user data", "create a calendar event", "get weather forecast"]:
        results = searcher.search(query, limit=1)
        logging.info(f"Query: {query}")
        for r in results:
            print(f"{r['tool_name']} -> {r['relevance_score']:.4f}")
