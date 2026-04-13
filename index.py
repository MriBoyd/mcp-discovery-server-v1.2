import json
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

class ToolIndex:
    def __init__(self, path="data/tools.json"):
        with open(path, "r") as f:
            self.tools = json.load(f)

        # Multi-Vector approximation: Index name, description, and raw_name
        self.docs = []
        for t in self.tools:
            self.docs.append(f"{t['name']} {t['server']} {t['description']}")

        # BM25 for keyword recall
        tokenized = [doc.lower().split() for doc in self.docs]
        self.bm25 = BM25Okapi(tokenized)

        # Vector model for semantic recall
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.embeddings = self.model.encode(self.docs)

    def bm25_search(self, query, top_k=50):
        """Pure keyword search using BM25."""
        scores = self.bm25.get_scores(query.lower().split())
        ranked = sorted(list(enumerate(scores)), key=lambda x: x[1], reverse=True)
        return [self.tools[i] for i, score in ranked[:top_k] if score > 0]

    def vector_search(self, query, top_k=50):
        """Pure semantic search using embeddings."""
        q_vec = self.model.encode([query])[0]
        scores = np.dot(self.embeddings, q_vec)
        ranked = sorted(list(enumerate(scores)), key=lambda x: x[1], reverse=True)
        return [self.tools[i] for i, _ in ranked[:top_k]]

    def hybrid_search(self, query, top_k=50, alpha=0.3):
        """Hybrid search combining BM25 and Vector scores."""
        bm25_scores = self.bm25.get_scores(query.lower().split())
        if np.max(bm25_scores) > 0:
            bm25_scores = bm25_scores / np.max(bm25_scores)

        q_vec = self.model.encode([query])[0]
        vec_scores = np.dot(self.embeddings, q_vec)
        if np.max(vec_scores) > 0:
             vec_scores = vec_scores / np.max(vec_scores)

        # Combined score
        scores = (alpha * bm25_scores) + ((1 - alpha) * vec_scores)

        ranked = sorted(list(enumerate(scores)), key=lambda x: x[1], reverse=True)
        return [self.tools[i] for i, _ in ranked[:top_k]]

    def generate_hyde_query(self, query):
        """
        Simulate HyDE by expanding the query with hypothetical context.
        In a real app, this would call an LLM.
        """
        hypothetical_context = f"This tool helps with {query} by providing automated functions."
        return f"{query} {hypothetical_context}"
