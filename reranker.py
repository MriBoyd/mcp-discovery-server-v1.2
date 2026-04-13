from sentence_transformers import CrossEncoder

# Optional Cross-Encoder for higher precision
# BGE-Reranker is high quality (BAAI/bge-reranker-base)
# For now, let's use a smaller model or keep it flexible
try:
    _reranker = CrossEncoder("BAAI/bge-reranker-base", max_length=512)
except Exception:
    _reranker = None

def simple_rerank(query, tools, top_k=5):
    if not tools:
        return []

    if _reranker:
        # Cross-Encoder (Query + Doc interaction)
        pairs = [[query, f"{t['name']} {t['description']}"] for t in tools]
        scores = _reranker.predict(pairs)

        ranked = sorted(
            zip(scores, tools),
            key=lambda x: x[0],
            reverse=True
        )
        return [t for _, t in ranked[:top_k]]

    # Fallback to current simple rank
    scored = []
    for t in tools:
        score = 0
        if query.lower() in t["name"].lower():
            score += 3
        if query.lower() in t["description"].lower():
            score += 2
        scored.append((score, t))

    ranked = sorted(scored, key=lambda x: x[0], reverse=True)
    return [t for _, t in ranked[:top_k]]
