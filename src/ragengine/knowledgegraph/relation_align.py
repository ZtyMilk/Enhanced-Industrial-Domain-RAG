import json
from pathlib import Path

import faiss
import numpy
from FlagEmbedding import BGEM3FlagModel


class RelationAlign:
    def __init__(
        self,
        embedding_model: BGEM3FlagModel,
        threshold: float,  # 0.55
        relation_direction: str,
        batch_size: int,
    ):
        self.embedding_model = embedding_model
        self.threshold = threshold
        self.relation_path = Path(relation_direction)

        self.inverted_canonical_relations: dict[str, str] = {}
        self.canonical_index: faiss.Index | None = None
        self.phrase_list: list = []
        self.relation_cache: dict[str, str] = {}
        self.batch_size = batch_size

    def canonical_relation_embed(
        self,
        input_canonical_relations: dict[str, list[str]],  # canonical_relation to phrase
    ):
        phrases: list[str] = []
        inverted_canonical_relations: dict[
            str, str
        ] = {}  # phrase to canonical_relation
        for canonical_relation, phrase_list in input_canonical_relations.items():
            for phrase in phrase_list:
                if phrase not in inverted_canonical_relations:
                    phrases.append(phrase)
                    inverted_canonical_relations[phrase] = canonical_relation
        output = self.embedding_model.encode(
            sentences=phrases,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
            max_length=32,
            batch_size=8,
        )
        dense_vectors = output["dense_vecs"].astype(numpy.float32)
        index = faiss.IndexFlatIP(dense_vectors.shape[1])
        index.add(dense_vectors)
        faiss.write_index(index, str(self.relation_path / "relation" / "index.faiss"))
        with open(
            self.relation_path / "relation" / "relations.json", "w", encoding="utf-8"
        ) as f:
            json.dump(inverted_canonical_relations, f, ensure_ascii=False, indent=2)
        return 0

    def index_read(self):
        index_file = self.relation_path / "relation" / "index.faiss"
        relations_file = self.relation_path / "relation" / "relations.json"
        if not (index_file.exists() and relations_file.exists()):
            raise FileNotFoundError(
                f"Cannot find the canonical relation file "
                f"in the folder .{self.relation_path / 'canonical'!s}\n"
            )
        self.canonical_index = faiss.read_index(str(index_file))
        with open(relations_file, "r", encoding="utf-8") as f:
            self.inverted_canonical_relations = json.load(f)
        self.phrase_list = list(self.inverted_canonical_relations.keys())
        return 0

    def aligner(self, raw_relations: list[str]):
        raw_relations = list(
            {
                raw_relation.strip()
                for raw_relation in raw_relations
                if raw_relation and raw_relation.strip()
            }
        )
        if not raw_relations:
            return {}
        output = self.embedding_model.encode(
            sentences=raw_relations,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
            max_length=32,
            batch_size=self.batch_size,
        )
        raw_dense_vector = output["dense_vecs"].astype(numpy.float32)
        scores, indices = self.canonical_index.search(raw_dense_vector, k=1)
        for index, raw_relation in enumerate(raw_relations):
            best_score = float(scores[index][0])
            if best_score >= self.threshold:
                canonical_relation = self.inverted_canonical_relations[
                    self.phrase_list[int(indices[index][0])]
                ]
                self.relation_cache[raw_relation] = canonical_relation
            else:
                self.relation_cache[raw_relation] = raw_relation
        return self.relation_cache
