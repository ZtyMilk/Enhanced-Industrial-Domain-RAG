import time

from FlagEmbedding import BGEM3FlagModel
from openai import OpenAI

from embed import Embed
from hypotheticalprompt import Generator
from knowledgegraph import EntityAlign, Graphing, RelationAlign
from multimodelprocess import MutiModelClient
from prepareprocess import Chunker


def main():
    embedding_model_client = BGEM3FlagModel(
        model_name_or_path=r"./bge_m3",
        normalize_embeddings=True,
        use_fp16=True,
        devices="cuda:0",
    )
    large_language_model_client = OpenAI(
        api_key="sk-ws-H.PMHMIHP.2Efh.MEUCIF414IP92KqtbsMJJV93qsB-SfqmhaLHCU3YPSG4Hkq9AiEAxB7Oole2OoYcbEUMyX1z-40P4F5xtqrf8Y7O3qV2R28",
        base_url="https://ws-ozkoniwn00sypl57.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
        timeout=240,
        max_retries=3,
    )

    t0 = time.perf_counter()

    chunks = Chunker(
        input_direction="./data/pdf",
        output_figure_direction="./data/store/image",
        chunk_size=400,
        chunk_overlap=50,
    ).pdf_chunker()
    print("mutimodel_process beginning...")
    chunks = MutiModelClient(
        language_model_name="qwen-flash",
        vision_model_name="qwen3-vl-flash",
        workers=32,
        client=large_language_model_client,
    ).all_modality_process(chunks=chunks)

    print(f"{time.perf_counter() - t0:.2f} seconds\nGraphing begining...")

    entity_aligner = EntityAlign(
        embedding_model=embedding_model_client,
        threshold=0.525,
        entity_direction="./data/aligner",
        batch_size=16,
    )
    relation_aligner = RelationAlign(
        embedding_model=embedding_model_client,
        threshold=0.55,
        relation_direction="./data/aligner",
        batch_size=16,
    )
    graph, graph_chunks = Graphing(
        client=large_language_model_client,
        language_model_name="qwen-flash",
        workers=64,
        entity_aligner=entity_aligner,
        relation_aligner=relation_aligner,
    ).knowledge_graph_generator(chunks=chunks)
    print(f"{time.perf_counter() - t0:.2f} seconds\nHypothesizing begining...")

    hype_chunks = Generator(
        client=large_language_model_client,
        language_model_name="qwen-flash",
        workers=32,
        question_number=2,
    ).hypothetical_prompts_generate(chunks=chunks)
    print(f"{time.perf_counter() - t0:.2f} seconds\nEmbedding begining...")

    embed = Embed(
        embedding_model=embedding_model_client,
        database_direction=r"./data/store",
    )
    embed.graph_embed(chunks=graph_chunks, graph=graph, batch_size=16)
    embed.hypothetical_embed(chunks=hype_chunks, batch_size=16)
    print(f"{time.perf_counter() - t0:.2f} seconds\nFinished.")
    return 0


if __name__ == "__main__":
    main()
