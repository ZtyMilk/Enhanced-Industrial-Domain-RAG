import concurrent.futures
from typing import Any

from openai import OpenAI

from .figure_modality import figure_understand
from .table_modality import table_understand


class MutiModelClient:
    def __init__(
        self,
        language_model_name: str,
        vision_model_name: str,
        client: OpenAI,
        workers: int,
    ):
        self.language_model_name = language_model_name
        self.vision_model_name = vision_model_name
        self.workers = workers
        self.client = client

    def all_modality_process(self, chunks: list[dict[Any, Any]]):
        raw_chunks = [c for c in chunks if c.get("modality") in ["table", "figure"]]
        result_chunks = [
            c for c in chunks if c.get("modality") not in ["table", "figure"]
        ]
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.workers)
        try:
            result_chunks.extend(
                executor.map(self._single_all_modality_process, raw_chunks)
            )
        except Exception as e:  # noqa: BLE001
            print(f"One chunk's modality processing failed.{e}")
        finally:
            executor.shutdown(wait=True)
        return result_chunks

    def _single_all_modality_process(self, chunk: dict[str, Any]):
        modality = chunk.get("modality")
        if modality == "table":
            chunk["text"] = self._table_process(chunk=chunk)
        elif modality == "figure":
            chunk["text"] = self._figure_process(chunk=chunk)
        return chunk

    def _table_process(self, chunk: dict[str, Any]):
        processed_text = table_understand(
            chunk=chunk, model_name=self.language_model_name, client=self.client
        )
        return processed_text

    def _figure_process(self, chunk: dict[str, Any]):
        processed_text = figure_understand(
            chunk=chunk, model_name=self.vision_model_name, client=self.client
        )
        return processed_text
