from collections.abc import Iterable
from typing import Any, cast

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam


def table_understand(
        chunk: dict[str, Any],
        model_name: str,
        client: OpenAI,
):
    prompt = (
        f"Please analyze the Markdown-formatted table following:\n{chunk['text']}\n\n"
        "Generate a comprehensive analysis of 100 words that "
        "incorporates all critical process parameters and their "
        "detailed technical interpretations.\nWarn:output single plain paragraph only."
    )
    messages: Iterable[ChatCompletionMessageParam] = cast(Any,[
        {"role": "system", "content": "You are an injection molding expert"},
        {"role": "user", "content": f"{prompt}"},
    ])
    try:
        response = client.chat.completions.create(
            model = model_name,
            messages = messages,
            temperature = 0.3,
            max_tokens = 500,
            stream = False,
        )
        understand_text = response.choices[0].message.content
        if understand_text and understand_text.strip():
            return f"{understand_text.strip()}"
        return chunk["text"]
    except Exception as e:# noqa: BLE001
        print(f"[WARNING]One table's understanding failed, back to Markdown.{e}")
        return chunk["text"]