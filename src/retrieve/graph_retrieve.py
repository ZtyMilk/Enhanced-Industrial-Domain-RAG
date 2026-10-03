import pickle
import re
from collections import defaultdict
from pathlib import Path

import faiss
import networkx
import numpy
from FlagEmbedding import BGEM3FlagModel


class GraphRetriever:
    def __init__(
        self,
        embedding_model: BGEM3FlagModel,
        database_direction: str,
        decay_factor: float,
        seed_bonus: float,
        max_depth: int,
    ):
        self.embedding_model = embedding_model
        self.database_path = Path(database_direction)
        self.decay_factor = decay_factor
        self.seed_bonus = seed_bonus
        self.max_depth = max_depth

    def index_read(self):
        index_file = self.database_path / "graph" / "index.faiss"
        sparse_file = self.database_path / "graph" / "sparse_index.pkl"
        metadata_file = self.database_path / "graph" / "metadata.pkl"
        graph_file = self.database_path / "graph" / "graph.pkl"
        document_file = self.database_path / "hype" / "metadata.pkl"
        self.faiss_index = faiss.read_index(str(index_file))
        with open(sparse_file, "rb") as f:
            self.sparse_index = pickle.load(f)
        with open(metadata_file, "rb") as f:
            self.metadata = pickle.load(f)
        with open(graph_file, "rb") as f:
            self.graph: networkx.MultiDiGraph = pickle.load(f)
        with open(document_file, "rb") as f:
            self.document = pickle.load(f)
        return 0

    def retriever(self, query: str, top_k: int):
        query_dense_vectors, _ = self._query_embed(query=query)
        seed_nodes: dict[str, float] = {}
        seed_edges: set[tuple[str, str, str, int, float]] = set()
        candidate_size = min(top_k * 2, self.faiss_index.ntotal)
        dense_scores, dense_indices = self.faiss_index.search(
            query_dense_vectors, k=candidate_size
        )

        for score, index in zip(dense_scores[0], dense_indices[0]):
            if index == -1:
                continue
            triple = self.metadata[index]
            head = triple["head"]
            relation = triple["relation"]
            tail = triple["tail"]
            chunk_index = triple["index"]
            score_value = float(score)
            seed_edges.add((head, tail, relation, chunk_index, score_value))
            seed_nodes[head] = max(seed_nodes.get(head, 0.0), score_value)
            seed_nodes[tail] = max(seed_nodes.get(tail, 0.0), score_value)
        # [Clean the graph nodes] & [Load it into the seed nodes]
        for node in self.graph.nodes():
            s = str(node).strip()
            if len(s) < 2:
                continue
            if re.search(r"[a-zA-Z]", s):
                if re.search(rf"\b{re.escape(s)}\b", query, re.IGNORECASE):
                    seed_nodes[node] = (
                        max(seed_nodes.get(node, 0.0), 0.8) + self.seed_bonus
                    )
            else:
                if s in query:
                    seed_nodes[node] = (
                        max(seed_nodes.get(node, 0.0), 0.8) + self.seed_bonus
                    )

        if not seed_nodes and not seed_edges:
            return []
        chunk_edge_weights = defaultdict(list)
        chunk_chains = defaultdict(list)
        for head, tail, relation, chunk_index, score_value in seed_edges:
            chunk_edge_weights[chunk_index].append(score_value)
            chunk_chains[chunk_index].append(
                f"SeedTriple: ({head}) -[{relation}]-> ({tail}) [score={score_value:.3f}]"
            )
        visited = dict(seed_nodes)
        queue = [(node, score, 0) for node, score in seed_nodes.items()]
        for curr_node, curr_score, depth in queue:
            if depth >= self.max_depth:
                continue
            # 出度：正向推演 (What-If)
            if self.graph.has_node(curr_node):
                for _, neighbor, data in self.graph.out_edges(curr_node, data=True):
                    chunk_index = data.get("chunk_index")
                    rel = data.get("relation", "")
                    edge_score = curr_score * self.decay_factor
                    if chunk_index is not None:
                        chunk_edge_weights[chunk_index].append(edge_score)
                        chunk_chains[chunk_index].append(
                            f"Out: ({curr_node}) -[{rel}]-> ({neighbor}) [hop={depth + 1}, score={edge_score:.3f}]"
                        )
                    if edge_score > visited.get(neighbor, 0.0):
                        visited[neighbor] = edge_score
                        queue.append((neighbor, edge_score, depth + 1))
                # 入度：根因追溯 (Root Cause)
                for neighbor, _, data in self.graph.in_edges(curr_node, data=True):
                    chunk_index = data.get("chunk_index")
                    rel = data.get("relation", "")
                    edge_score = curr_score * self.decay_factor
                    if chunk_index is not None:
                        chunk_edge_weights[chunk_index].append(edge_score)
                        chunk_chains[chunk_index].append(
                            f"In: ({neighbor}) -[{rel}]-> ({curr_node}) [hop={depth + 1}, score={edge_score:.3f}]"
                        )
                    if edge_score > visited.get(neighbor, 0.0):
                        visited[neighbor] = edge_score
                        queue.append((neighbor, edge_score, depth + 1))
        # 3. 切片边际递减聚合与组装
        results = []
        for chunk_index, weights in chunk_edge_weights.items():
            weights.sort(reverse=True)
            # 首项满额加权 + 剩余项 0.1 边际加权，打破平局
            final_score = weights[0] + 0.1 * sum(weights[1:])
            if self.document and chunk_index < len(self.document):
                chunk_obj = self.document[chunk_index]
            else:
                chunk_obj = {
                    "index": chunk_index,
                    "text": chunk_chains[chunk_index][0]
                    if chunk_chains[chunk_index]
                    else "",
                }
            results.append(
                {
                    "chunk": chunk_obj,
                    "chunk_index": chunk_index,
                    "score": final_score,
                    "reasoning_chains": chunk_chains[chunk_index],
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
