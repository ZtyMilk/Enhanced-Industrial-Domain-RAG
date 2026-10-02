import base64
from pathlib import Path
from typing import Any

from openai import OpenAI


def figure_understand(
    chunk: dict[str, Any],
    model_name: str,
    client: OpenAI,
):
    figure_extension = Path(chunk["figure_save_path"]).suffix.replace(".", "").lower()
    if figure_extension == "jpg":
        figure_extension = "jpeg"
    with open(chunk["figure_save_path"], "rb") as f:
        figure_base64 = base64.b64encode(f.read()).decode("utf-8")
    prompt = (
        f"The academic figure has the caption: {chunk['text']}\n"
        f"Generate a comprehensive analysis of approximately 100 words that "
        f"incorporates all critical process parameters and their detailed "
        f"technical interpretations.\nWarn:output single plain paragraph only."
    )
    messages = [
        {"role": "system", "content": "You are an injection molding expert"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"{prompt}"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/{figure_extension};base64,{figure_base64}",
                        "detail": "high",
                    },
                },
            ],
        },
    ]
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.4,
            max_tokens=600,
            stream=False,
        )
        understand_text = response.choices[0].message.content
        if understand_text and understand_text.strip():
            return f"{understand_text.strip()}"
        return chunk["text"]
    except Exception as e:  # noqa: BLE001
        print(f"[WARNING]One figure's understanding failed, back to it's capture.{e}")
        return chunk["text"]
