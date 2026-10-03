from FlagEmbedding import FlagReranker


def reranker(
    query: str,
    candidates: list,
    reranker_model: FlagReranker,
    top_k: int,
    batch_size: int,
):
    if not candidates:
        return []
    pairs = []
    for candidate in candidates:
        pairs.append([query, candidate["chunk"]["text"]])
    scores = reranker_model.compute_score(
        sentence_pairs=pairs, batch_size=batch_size, max_length=512
    )
    reranked_results = []
    for index, candidate in enumerate(candidates):
        entry = candidate.copy()
        entry["rerank_score"] = float(scores[index])
        reranked_results.append(entry)
    reranked_results.sort(key=lambda x: x["rerank_score"], reverse=True)
    return reranked_results[:top_k]
