def reciprocal_rank_fusion(
    graph_chunks: list,
    hype_chunks: list,
    top_k: int,
    k: int,
):
    fused_scores = {}
    chunk_dictionary = {}
    for rank, chunk in enumerate(graph_chunks, start=1):
        index = chunk["chunk_index"]
        if index is None:
            continue
        fused_scores[index] = fused_scores.get(index, 0.0) + 1.0 / (k + rank)
        if index not in chunk_dictionary:
            chunk_dictionary[index] = chunk.copy()
        chunk_dictionary[index]["graph_rank"] = rank
    for rank, chunk in enumerate(hype_chunks, start=1):
        index = chunk["chunk_index"]
        if index is None:
            continue
        fused_scores[index] = fused_scores.get(index, 0.0) + 1.0 / (k + rank)

        if index not in chunk_dictionary:
            chunk_dictionary[index] = chunk.copy()
        else:
            if "questions" in chunk:
                chunk_dictionary[index]["question"] = chunk["question"]
            if "dense_score" in chunk:
                chunk_dictionary[index]["dense_score"] = chunk["dense_score"]
                chunk_dictionary[index]["sparse_score"] = chunk["sparse_score"]

        chunk_dictionary[index]["hype_rank"] = rank
    fused_list = []
    for cid, score in fused_scores.items():
        doc = chunk_dictionary[cid]
        doc["rrf_score"] = score
        fused_list.append(doc)

    fused_list = sorted(fused_list, key=lambda x: x["rrf_score"], reverse=True)
    return fused_list[:top_k]
