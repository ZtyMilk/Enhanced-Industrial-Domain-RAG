import pickle
from pathlib import Path

import faiss
import numpy
from FlagEmbedding import BGEM3FlagModel


class HypotheticalRetriever:
    def __init__(
        self,
        embedding_model: BGEM3FlagModel,
        database_direction: str,
        dense_weight: float,
    ):
        self.embedding_model = embedding_model
        self.database_path = Path(database_direction) / "hype"
        self.dense_weight = dense_weight
        self.sparse_weight = 1 - dense_weight

    def index_read(self):
        index_file = self.database_path / "index.faiss"
        sparse_file = self.database_path / "sparse_index.pkl"
        metadata_file = self.database_path / "metadata.pkl"
        self.faiss_index = faiss.read_index(str(index_file))
        with open(sparse_file, "rb") as f:
            self.sparse_index = pickle.load(f)
        with open(metadata_file, "rb") as f:
            self.metadata = pickle.load(f)
        return 0

    def retriever(self, query: str, top_k: int):
        query_dense_vectors, query_lexical_weights = self._query_embed(query=query)
        candidate_size = min(top_k * 2, self.faiss_index.ntotal)
        dense_scores, dense_indices = self.faiss_index.search(
            query_dense_vectors, k=candidate_size
        )

        dense_map = {
            int(index): float(score)
            for score, index in zip(dense_scores[0], dense_indices[0])
            if index != -1
        }
        sparse_scores = self.embedding_model.compute_lexical_matching_score(
            [query_lexical_weights], self.sparse_index
        )[0]
        sparse_top = numpy.argsort(sparse_scores, descending=True)[:candidate_size]
        sparse_map = {
            int(index): float(sparse_scores[index])
            for index in sparse_top
            if sparse_scores[index] > 0
        }

        candidates = set(dense_map.keys()) | set(sparse_map.keys())
        results = []
        for i in candidates:
            dense_score = dense_map.get(i, 0.0)
            sparse_score = sparse_map.get(i, 0.0)
            score = self.dense_weight * dense_score + self.sparse_weight * sparse_score
            chunk = self.metadata[i]
            results.append(
                {
                    "chunk": chunk,
                    "chunk_index": i,
                    "score": score,
                    "dense_score": dense_score,
                    "sparse_score": sparse_score,
                }
            )
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def _query_embed(self, query: str):
        query_output = self.embedding_model.encode(
            sentences=[query],
            max_length=64,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        query_dense_vectors = query_output["dense_vecs"].astype(numpy.float32)
        query_lexical_weights = query_output["lexical_weights"][0]
        return query_dense_vectors, query_lexical_weights
