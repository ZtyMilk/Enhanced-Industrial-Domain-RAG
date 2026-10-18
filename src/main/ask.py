from typing import Any

from FlagEmbedding import BGEM3FlagModel, FlagReranker
from funasr.auto.auto_model import AutoModel
from openai import OpenAI

from ragengine import (
    GraphRetriever,
    HypotheticalRetriever,
    reciprocal_rank_fusion,
    reranker,
)
from speechengine import AudioRecorder, SenseVoiceTranscriber


def single_request(
    hype_retriever: HypotheticalRetriever,
    graph_retriever: GraphRetriever,
    voice_model: AutoModel,
    reranker_model: FlagReranker,
    language_model: OpenAI,
    language_model_name: str,
    top_k: int,
):
    query = SenseVoiceTranscriber(voice_model=voice_model).transcribe(
        audio_data=AudioRecorder(sample_rate=16000, channels=1).record()
    )
    print(f"\n\n\nQUERY: {query}")
    hype_retrieved_chunks = hype_retriever.retriever(query=query, top_k=2 * top_k)
    graph_retrieved_chunks = graph_retriever.retriever(query=query, top_k=2 * top_k)
    raw_candidates = reciprocal_rank_fusion(
        graph_chunks=graph_retrieved_chunks,
        hype_chunks=hype_retrieved_chunks,
        top_k=2 * top_k,
        rank_constant=100,
    )
    reranked_candidates = reranker(
        query=query,
        candidates=raw_candidates,
        reranker_model=reranker_model,
        top_k=top_k,
        batch_size=16,
    )
    chunk_text = "\n\n".join(
        [
            chunk["chunk"]["text"]
            for chunk in reranked_candidates
            if "chunk" in chunk and "text" in chunk["chunk"]
        ]
    )
    prompt = (
        f"Answer the question: {query}\n"
        f"Here the retrived chunks:\n{chunk_text}\n"
        f"Output in markdown format\n"
    )
    messages: list[Any] = [
        {
            "role": "system",
            "content": (
                "You are an injection molding expert\n<rules>\n"
                "1.All answer must be based on retrieved chunks\n"
                "2.Should challenge user's premise of the question\n"
                "3.Only output actionable recommendations and concrete results."
            ),
        },
        {"role": "user", "content": f"{prompt}"},
    ]
    response_message = None
    try:
        completion = language_model.chat.completions.create(
            model=language_model_name,
            messages=messages,
            temperature=0.00,
            max_tokens=5000,
            stream=False,
        )
        response_message = completion.choices[0].message.content
    except Exception as e:  # noqa: BLE001
        print(f"[WARNING]One chunk's hype failed.{e}")

    return f"\nANWSER:\n{response_message}"


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
language_model_client = OpenAI(
    api_key="sk-ws-H.PMHMIHP.2Efh.MEUCIF414IP92KqtbsMJJV93qsB-SfqmhaLHCU3YPSG4Hkq9AiEAxB7Oole2OoYcbEUMyX1z-40P4F5xtqrf8Y7O3qV2R28",
    base_url="https://ws-ozkoniwn00sypl57.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
    timeout=240,
    max_retries=3,
)
voice_model_client = AutoModel(
    model=r"./seaco_paraformer",
    device="cuda:0",
    disable_update=True,
    disable_pbar=True,
)
hype_retrieve_client = HypotheticalRetriever(
    embedding_model=embedding_model_client,
    database_direction=r"./data/store",
    dense_weight=0.7,
)
graph_retrieve_client = GraphRetriever(
    embedding_model=embedding_model_client,
    database_direction=r"./data/store",
    decay_factor=0.82,
    seed_bonus=0.2,
    max_depth=2,
)

hype_retrieve_client.index_read()
graph_retrieve_client.index_read()
while True:
    print(
        f"{
            single_request(
                hype_retriever=hype_retrieve_client,
                graph_retriever=graph_retrieve_client,
                voice_model=voice_model_client,
                reranker_model=reranker_model_client,
                language_model=language_model_client,
                language_model_name='qwen-flash',
                top_k=20,
            )
        }"
    )

# 在AE44镁合金高压压铸中,为什么大幅提高压射速度到50m/s就能彻底消除偏析带(Defect band)?
# 高压压铸镁合金中偏析带(Defect band)与气孔缺陷的形成机理是什么?
# 真空压铸中如何将型腔真空度控制在5kPa以下以消除气孔?
# 金属模具多点温度传感器布置方式对注塑成型周期温度场的影响
