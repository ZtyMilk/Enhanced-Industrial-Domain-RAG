import concurrent.futures
from typing import Any

from openai import OpenAI

from .question_generate import question_generate


class Generator:
    def __init__(
        self, client: OpenAI, language_model_name, workers: int, question_number: int
    ):
        self.client = client
        self.language_model_name = language_model_name
        self.workers = workers
        self.question_number = question_number

    def hypothetical_prompts_generate(self, chunks: list[dict[str, Any]]):
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.workers)
        result_chunks = []
        try:
            result_chunks.extend(
                executor.map(self._single_hypothetical_prompts_generate, chunks)
            )
        finally:
            executor.shutdown(wait=True)
        return result_chunks

    def _single_hypothetical_prompts_generate(self, chunk: dict[str, Any]):
        try:
            questions = question_generate(
                input_text=chunk["text"],
                model_name=self.language_model_name,
                client=self.client,
                question_number=self.question_number,
            )
            chunk["questions"] = questions
            return chunk
        except Exception as e:  # noqa: BLE001
            print(f"One chunk's hype generator failed.{e}")
            return chunk
