from qdrant_client import QdrantClient
from qdrant_client.models import (Distance, VectorParams, models, Document, PointStruct)
import json
import itertools
from rank_bm25 import BM25Okapi
import re
import logging
import uuid
import time
from collections import OrderedDict
import numpy as np

logging.basicConfig(level=logging.INFO)

# =========================
# CONFIG
# =========================
collection_name = "hybrid-search"


# =========================
# CLIENT
# =========================
client = QdrantClient(url="http://0.0.0.0:6333")


logging.info(f"Collection name: {collection_name}")

dense_embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
sparse_embedding_model = "prithivida/Splade_PP_en_v1"
late_interaction_embedding_model = "colbert-ir/colbertv2.0"

logging.info(f"Using dense embedding model: {dense_embedding_model}")
logging.info(f"Using sparse embedding model: {sparse_embedding_model}")
logging.info(f"Using late interaction embedding model: {late_interaction_embedding_model}")


BATCH_SIZE = 32
PREFETCH_LIMIT = 100   # 🔥 increased (oversampling)
FINAL_LIMIT = 10
FUSION_LIMIT = 100  # candidates for reranking
ALPHA = 0.7  # weight for reranker
BETA = 0.3
TOKEN_PATTERN = re.compile(r'[^a-zA-Z0-9]')
CAMEL_PATTERN = re.compile(r'([a-z])([A-Z])')

# Caching settings
CACHE_MAX_SIZE = 200
CACHE_TTL = 60 * 60  # 1 hour


class LRUCacheTTL:
    def __init__(self, max_size: int = 100, ttl: int = 3600):
        self.max_size = max_size
        self.ttl = ttl
        self._store = OrderedDict()

    def get(self, key):
        item = self._store.get(key)
        if item is None:
            return None
        timestamp, value = item
        if time.time() - timestamp > self.ttl:
            try:
                del self._store[key]
            except KeyError:
                pass
            return None
        # move to end as most recently used
        self._store.move_to_end(key)
        return value

    def set(self, key, value):
        if key in self._store:
            del self._store[key]
        self._store[key] = (time.time(), value)
        # Evict oldest if over capacity
        while len(self._store) > self.max_size:
            self._store.popitem(last=False)


# in-memory search cache
search_cache = LRUCacheTTL(max_size=CACHE_MAX_SIZE, ttl=CACHE_TTL)

# =========================
# UTILS
# =========================

def str_to_uuid(s):
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, s))

def normalize_text(text: str) -> str:
    return " ".join(text.strip().lower().split())

def min_max_scale(scores: dict) -> dict:
    """Normalize scores to [0, 1] range using numpy for efficiency."""
    if not scores:
        return {}
    if len(scores) == 1:
        return {k: 1.0 for k in scores.keys()}
        
    vals = np.array(list(scores.values()))
    min_s = vals.min()
    max_s = vals.max()
    
    if max_s == min_s:
        return {k: 1.0 for k in scores.keys()}
    
    # Vectorized normalization
    norm_vals = (vals - min_s) / (max_s - min_s)
    return dict(zip(scores.keys(), norm_vals))

def tool_to_text(tool):
    params = tool.get("parameters", {})
    param_text = ", ".join(
        f"{k}: {v.get('description','')}"
        for k, v in params.items()
    ) if params else "none"
    required = [
        k for k, v in params.items()
        if v.get("required")
    ]
    required_text = ", ".join(required) if required else "none"

    return normalize_text(f"""
    Tool: {tool['name']}
    Server: {tool.get('server', 'unknown')}
    Description: {tool['description']}
    Parameters: {param_text}
    Required parameters: {required_text}
    """)

# Initialize regex for tokenization
TOKEN_PATTERN = re.compile(r'[^a-zA-Z0-9]')
CAMEL_PATTERN = re.compile(r'([a-z])([A-Z])')

def tokenize(text: str) -> list[str]:
    """Robust tokenization for lexical search (from hybrid_searcher.py)"""
    if not text:
        return []
    # Handle camelCase
    text = CAMEL_PATTERN.sub(r'\1 \2', text)
    # Split on non-alphanumeric characters
    tokens = TOKEN_PATTERN.split(text)
    return [t.lower() for t in tokens if t]

# =========================
# LOAD JSON
# =========================
with open("mcp_servers.json", "r") as f:
    data = json.load(f)

def index_tools(
    client,
    collection_name,
    tools_data,
    dense_model,
    sparse_model,
    rerank_model,
    batch_size=32,
    recreate=False
):
    """
    Unified indexing strategy:
    1. Flattens tools from servers
    2. Ensures collection exists with Dense, Sparse, and Multi-vector configs
    3. Performs incremental indexing by default (skips existing IDs)
    """
    # 1. Flatten tools
    tools = []
    for server in tools_data.get("servers", []):
        for tool in server.get("tools", []):
            tool["server"] = server["name"]
            tools.append(tool)
    
    if not tools:
        logging.warning("No tools found to index.")
        return {"status": "skipped", "count": 0}

    # 2. Collection Setup
    if recreate and client.collection_exists(collection_name):
        logging.info(f"Recreating collection: {collection_name}")
        client.delete_collection(collection_name)
    
    if not client.collection_exists(collection_name):
        logging.info(f"Creating collection: {collection_name}")
        client.create_collection(
            collection_name,
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
        # Incremental: Fetch existing IDs to skip them
        logging.info(f"Checking existing points in {collection_name}...")
        indexed_ids = set()
        try:
            offset = None
            while True:
                points, offset = client.scroll(
                    collection_name=collection_name,
                    limit=1000,
                    offset=offset,
                    with_payload=False,
                    with_vectors=False,
                )
                for p in points:
                    indexed_ids.add(str(p.id))
                if offset is None:
                    break
        except Exception as e:
            logging.warning(f"Scroll failed, proceeding with full ingest: {e}")

    # 3. Filter new tools
    to_index = []
    for tool in tools:
        if str_to_uuid(tool["name"]) not in indexed_ids:
            to_index.append(tool)

    if not to_index:
        logging.info("All tools already indexed.")
        return {"status": "unchanged", "count": len(tools)}

    logging.info(f"Indexing {len(to_index)} new tools...")

    # 4. Batch Ingestion
    def batch(iterable, size):
        it = iter(iterable)
        while chunk := list(itertools.islice(it, size)):
            yield chunk

    def generate_points():
        for tool in to_index:
            text = tool_to_text(tool)
            yield PointStruct(
                id=str_to_uuid(tool["name"]),
                vector={
                    "dense": Document(text=text, model=dense_model),
                    "sparse": Document(text=text, model=sparse_model),
                    "multi": Document(text=text, model=rerank_model),
                },
                payload=tool
            )
    
    for chunk in batch(generate_points(), batch_size):
        client.upload_points(
            collection_name=collection_name,
            points=chunk
        )
        
    return {"status": "indexed", "count": len(to_index)}

def build_bm25_index(tools, text_func):
    corpus = [text_func(t) for t in tools]
    tokenized = [tokenize(doc) for doc in corpus]
    return BM25Okapi(tokenized), corpus


def search_tools(
    client,
    collection_name,
    query,
    dense_model,
    sparse_model,
    rerank_model,
    bm25=None,           # Optional: enables 3-way fusion if provided
    tools=None,          # Required if bm25 is provided
    prefetch_limit=100,
    fusion_limit=100,
    final_limit=10,
    weights={
        "colbert": 0.6,
        "rrf": 0.25,
        "bm25": 0.15
    },
    cache=None
):
    """
    Unified search strategy:
    1. Retrieval: Qdrant Hybrid (Dense + Sparse) + Optional BM25
    2. Rerank: ColBERT late interaction
    3. Fusion: 3-way normalized weighted sum
    """
    cache_key = f"unified:{query}:{prefetch_limit}:{fusion_limit}:{final_limit}:{bm25 is not None}"
    if cache:
        cached = cache.get(cache_key)
        if cached:
            return cached
        
    # =========================
    # 1. QDRANT RETRIEVAL
    # =========================
    dense_query = Document(text=query, model=dense_model)
    sparse_query = Document(text=query, model=sparse_model)
    rerank_query = Document(text=query, model=rerank_model)

    prefetch = [
        models.Prefetch(query=dense_query, using="dense", limit=prefetch_limit),
        models.Prefetch(query=sparse_query, using="sparse", limit=prefetch_limit),
    ]
    
    fused = client.query_points(
        collection_name,
        prefetch=prefetch,
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=fusion_limit,
    )
    
    qdrant_ids = [p.id for p in fused.points]
    fusion_raw_map = {p.id: p.score for p in fused.points}

    # =========================
    # 2. OPTIONAL BM25
    # =========================
    bm25_raw_map = {}
    bm25_ids = []
    
    if bm25 and tools:
        tokenized_query = tokenize(query)
        bm25_scores = bm25.get_scores(tokenized_query)
        
        # Get top candidates for lexical signal
        top_indices = sorted(
            range(len(bm25_scores)),
            key=lambda i: bm25_scores[i],
            reverse=True
        )[:50]
        
        for idx in top_indices:
            tid = str_to_uuid(tools[idx]["name"])
            bm25_ids.append(tid)
            bm25_raw_map[tid] = bm25_scores[idx]

    # =========================
    # 3. MERGE & RERANK
    # =========================
    candidate_ids = list(set(qdrant_ids + bm25_ids))[:200]

    if not candidate_ids:
        return []

    reranked = client.query_points(
        collection_name=collection_name,
        query=rerank_query,
        using="multi",
        limit=fusion_limit,  # Rerank all candidates for stable normalization
        with_payload=True,
        query_filter=models.Filter(
            must=[models.HasIdCondition(has_id=candidate_ids)]
        )
    )
    
    colbert_raw_map = {r.id: r.score for r in reranked.points}

    # =========================
    # 4. NORMALIZED FUSION
    # =========================
    norm_fusion = min_max_scale(fusion_raw_map)
    norm_colbert = min_max_scale(colbert_raw_map)
    norm_bm25 = min_max_scale(bm25_raw_map) if bm25_raw_map else {}
    
    final_results = []
    for r in reranked.points:
        c_score = norm_colbert.get(r.id, 0)
        f_score = norm_fusion.get(r.id, 0)
        b_score = norm_bm25.get(r.id, 0)
        
        # Calculate weighted combined score
        combined_score = (
            weights["colbert"] * c_score +
            weights["rrf"] * f_score +
            (weights["bm25"] * b_score if bm25_raw_map else 0)
        )
        
        # Re-adjust total weight if BM25 was missing to maintain range
        if not bm25_raw_map:
            total_active_weight = weights["colbert"] + weights["rrf"]
            combined_score = combined_score / total_active_weight
            
        r.score = combined_score
        final_results.append(r)

    # Sort and slice to final_limit after fusion
    final_results.sort(key=lambda x: x.score, reverse=True)
    final_results = final_results[:final_limit]
    
    if cache:
        cache.set(cache_key, final_results)

    return final_results

if __name__ == "__main__":
    # 🔥 index once (incremental by default)
    index_tools(
        client,
        "tool-search",
        data,
        dense_embedding_model,
        sparse_embedding_model,
        late_interaction_embedding_model,
        batch_size=32,
        recreate=True
    )

    # Prepare local BM25 for the unified searcher
    # Flatten data for BM25 (same as index_tools does internally)
    all_tools = []
    for server in data.get("servers", []):
        for tool in server.get("tools", []):
            tool["server"] = server["name"]
            all_tools.append(tool)
            
    bm25, corpus = build_bm25_index(all_tools, tool_to_text)

    # 🔥 reuse everywhere
    results = search_tools(
        client,
        "tool-search",
        "send email to user",
        dense_embedding_model,
        sparse_embedding_model,
        late_interaction_embedding_model,
        bm25=bm25,
        tools=all_tools,
        cache=search_cache
    )

    for r in results:
        print(r.payload["name"], "->", r.score)
