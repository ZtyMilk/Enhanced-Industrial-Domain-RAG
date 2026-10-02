import concurrent.futures
from typing import Any

import networkx
from openai import OpenAI

from knowledgegraph.entity_align import EntityAlign
from knowledgegraph.extract_triples import extract_triples
from knowledgegraph.relation_align import RelationAlign


class Graphing:
    def __init__(
        self,
        client: OpenAI,
        language_model_name: str,
        workers: int,
        entity_aligner: EntityAlign,
        relation_aligner: RelationAlign,
    ):
        self.language_model_name = language_model_name
        self.client = client
        self.workers = workers
        self.entity_aligner = entity_aligner
        self.relation_aligner = relation_aligner

    def knowledge_graph_generator(self, chunks: list[dict[str, Any]]):
        for index, chunk in enumerate(chunks):
            chunk.setdefault("index", index)
        raw_chunks = self._parallel_extract_triples(chunks=chunks)
        entities, relations = [], []
        for raw_chunk in raw_chunks:
            entities.append(raw_chunk["head"])
            entities.append(raw_chunk["tail"])
            relations.append(raw_chunk["relation"])
        self.entity_aligner.index_read()
        self.relation_aligner.index_read()
        entity_dictionary = self.entity_aligner.aligner(entities)
        relation_dictionary = self.relation_aligner.aligner(relations)

        result_chunks, graph = [], networkx.MultiDiGraph()
        for index, raw_chunk in enumerate(raw_chunks):
            head = entity_dictionary.get(raw_chunk["head"], raw_chunk["head"])
            tail = entity_dictionary.get(raw_chunk["tail"], raw_chunk["tail"])
            relation = relation_dictionary.get(
                raw_chunk["relation"], raw_chunk["relation"]
            )
            if not (head and tail and relation) or head == tail:
                continue
            graph.add_edge(
                head,
                tail,
                key=index,
                relation=relation,
                chunk_index=raw_chunk["index"],
                text=raw_chunk["text"],
            )
            result_chunks.append(
                {
                    "index": raw_chunk["index"],
                    "head": head,
                    "relation": relation,
                    "tail": tail,
                    "text": raw_chunk["text"],
                }
            )
        return graph, result_chunks

    def _parallel_extract_triples(self, chunks: list[dict[str, Any]]):
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.workers)
        result_chunks = []
        try:
            response_chunks = executor.map(self._single_extract_triples, chunks)
            for response_chunk in response_chunks:
                result_chunks.extend(response_chunk)
        finally:
            executor.shutdown(wait=True)
        return result_chunks

    def _single_extract_triples(self, chunk: dict[str, Any]):
        try:
            raw_triples = extract_triples(
                client=self.client,
                input_text=chunk.get("text", ""),
                model_name=self.language_model_name,
            )
            cleaned_chunks = []
            for raw_triple in raw_triples:
                raw_head = raw_triple.get("head", "")
                raw_relation = raw_triple.get("relation", "")
                raw_tail = raw_triple.get("tail", "")

                if not (raw_head and raw_tail and raw_relation):
                    continue
                if raw_head == raw_tail:
                    continue
                cleaned_chunks.append(
                    {
                        "index": chunk.get("index"),
                        "head": raw_head,
                        "relation": raw_relation,
                        "tail": raw_tail,
                        "text": chunk.get("text", ""),
                    }
                )
            return cleaned_chunks
        except Exception as e:  # noqa: BLE001
            print(f"One chunk's graphy generator failed.{e}")
            return []
