import pickle
from pathlib import Path
from typing import Any

import faiss
import networkx
import numpy
from FlagEmbedding import BGEM3FlagModel


class Embed:
    def __init__(self, embedding_model: BGEM3FlagModel, database_direction: str):
        self.embedding_model = embedding_model
        self.database_path = Path(database_direction)

    def hypothetical_embed(self, chunks: list[dict[str, Any]], batch_size: int):
        texts = [
            f"[prompt question]:\n"
            f"{
                '\n'.join(
                    f'{question_index + 1}.{question}'
                    for question_index, question in enumerate(
                        chunk.get('questions', [])
                    )
                )
            }\n"
            f"[text]:{chunk['text']}\n"
            for chunk in chunks
        ]
        output = self.embedding_model.encode(
            sentences=texts,
            batch_size=batch_size,
            max_length=1024,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        dense_vectors = output["dense_vecs"].astype(numpy.float32)
        lexical_weights = output["lexical_weights"]
        index = faiss.IndexFlatIP(dense_vectors.shape[1])
        index.add(dense_vectors)
        faiss.write_index(index, str(self.database_path / "hype" / "index.faiss"))
        with open(self.database_path / "hype" / "sparse_index.pkl", "wb") as f:
            pickle.dump(lexical_weights, f)
        with open(self.database_path / "hype" / "metadata.pkl", "wb") as f:
            pickle.dump(chunks, f)

        return 0

    def graph_embed(
        self,
        chunks: list[dict[str, Any]],
        graph: networkx.MultiDiGraph,
        batch_size: int,
    ):
        if not chunks:
            return 0
        texts = [
            f"({chunk['head']}) --[{chunk['relation']}]--> ({chunk['tail']})"
            for chunk in chunks
        ]
        output = self.embedding_model.encode(
            sentences=texts,
            batch_size=batch_size,
            max_length=256,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        dense_vectors = output["dense_vecs"].astype(numpy.float32)
        lexical_weights = output["lexical_weights"]
        index = faiss.IndexFlatIP(dense_vectors.shape[1])
        index.add(dense_vectors)
        faiss.write_index(index, str(self.database_path / "graph" / "index.faiss"))
        with open(self.database_path / "graph" / "sparse_index.pkl", "wb") as f:
            pickle.dump(lexical_weights, f)
        with open(self.database_path / "graph" / "metadata.pkl", "wb") as f:
            pickle.dump(chunks, f)
        with open(self.database_path / "graph" / "graph.pkl", "wb") as f:
            pickle.dump(graph, f)
        return 0
