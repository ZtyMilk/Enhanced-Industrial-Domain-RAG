from FlagEmbedding import BGEM3FlagModel, FlagReranker

from rerank import reranker
from retrieve import GraphRetriever, HypotheticalRetriever, reciprocal_rank_fusion


def main():
    embedding_model_client = BGEM3FlagModel(
        model_name_or_path=r"./bge_m3",
        normalize_embeddings=True,
        use_fp16=True,
        devices="cuda:0",
    )
    reranker_model_client = FlagReranker(
        model_name_or_path=r"./bge_reranker_large",
        normalize=True,
        use_fp16=True,
        devices="cuda:0",
    )

    query = "高压压铸镁合金中偏析带（Defect band）与气孔缺陷的形成机理是什么？"
    # "真空压铸中如何将型腔真空度控制在5kPa以下以消除气孔？"
    # "金属模具多点温度传感器布置方式对注塑成型周期温度场的影响"
    hype_retriever = HypotheticalRetriever(
        embedding_model=embedding_model_client,
        database_direction=r"./data/store",
        dense_weight=0.7,
    )
    hype_retriever.index_read()
    graph_retriever = GraphRetriever(
        embedding_model=embedding_model_client,
        database_direction=r"./data/store",
        decay_factor=0.82,
        seed_bonus=0.2,
        max_depth=2,
    )
    graph_retriever.index_read()

    hype_retrieved_chunks = hype_retriever.retriever(query=query, top_k=40)
    graph_retrieved_chunks = graph_retriever.retriever(query=query, top_k=40)
    raw_candidates = reciprocal_rank_fusion(
        graph_chunks=graph_retrieved_chunks,
        hype_chunks=hype_retrieved_chunks,
        top_k=40,
        k=100,
    )
    reranked_candidates = reranker(
        query=query,
        candidates=raw_candidates,
        reranker_model=reranker_model_client,
        top_k=20,
        batch_size=16,
    )
    for i in reranked_candidates:
        print(f"{i['chunk']}\n")


if __name__ == "__main__":
    main()
