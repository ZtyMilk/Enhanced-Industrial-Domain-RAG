import json
from pathlib import Path

import faiss
import numpy
from FlagEmbedding import BGEM3FlagModel


class EntityAlign:
    def __init__(
        self,
        embedding_model: BGEM3FlagModel,
        threshold: float,  # 0.525
        entity_direction: str,
        batch_size: int,
    ):
        self.embedding_model = embedding_model
        self.threshold = threshold
        self.entity_path = Path(entity_direction)
        self.batch_size = batch_size

        self.entity_cache: dict[str, str] = {}
        self.canonical_entities: list[str] = []
        self.canonical_index: faiss.Index | None = None

    def canonical_entity_embed(self, input_canonical_entities: list[str]):
        output = self.embedding_model.encode(
            sentences=input_canonical_entities,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
            max_length=32,
            batch_size=8,
        )
        dense_vectors = output["dense_vecs"].astype(numpy.float32)
        index = faiss.IndexFlatIP(dense_vectors.shape[1])
        index.add(dense_vectors)
        faiss.write_index(index, str(self.entity_path / "entity" / "index.faiss"))
        with open(
            self.entity_path / "entity" / "entities.json", "w", encoding="utf-8"
        ) as f:
            json.dump(input_canonical_entities, f, ensure_ascii=False, indent=2)
        return 0

    def index_read(self):
        index_file = self.entity_path / "entity" / "index.faiss"
        entities_file = self.entity_path / "entity" / "entities.json"
        if not (index_file.exists() and entities_file.exists()):
            raise FileNotFoundError(
                f"Cannot find the canonical entity file "
                f"in the folder .{self.entity_path / 'canonical'!s}\n"
            )
        self.canonical_index = faiss.read_index(str(index_file))
        with open(entities_file, "r", encoding="utf-8") as f:
            self.canonical_entities = json.load(f)
        return 0

    def aligner(self, raw_entities: list[str]):
        raw_entities = list(
            {
                raw_entity.strip()
                for raw_entity in raw_entities
                if raw_entity and raw_entity.strip()
            }
        )
        if not raw_entities:
            return {}
        output = self.embedding_model.encode(
            sentences=raw_entities,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
            max_length=32,
            batch_size=self.batch_size,
        )
        raw_dense_vector = output["dense_vecs"].astype(numpy.float32)
        scores, indices = self.canonical_index.search(raw_dense_vector, k=1)
        for index,raw_entity in enumerate(raw_entities):
            best_score = float(scores[index][0])
            if best_score >= self.threshold:
                canonical_entity = self.canonical_entities[int(indices[index][0])]
                self.entity_cache[raw_entity] = canonical_entity
            else:
                self.entity_cache[raw_entity] = raw_entity
        return self.entity_cache
