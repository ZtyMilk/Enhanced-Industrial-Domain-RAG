import json

from openai import OpenAI


def extract_triples(
    client: OpenAI,
    input_text: str,
    model_name: str,
):
    prompt = (
        f"Extract key knowledge graph triples (process causality, "
        f"defects, and parameters) from the following text:\n\n"
        f":{input_text}\n"
    )
    messages = [
        {"role": "system", "content": "You are an injection molding expert"},
        {"role": "user", "content": f"{prompt}"},
    ]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "extract_triples_schema",
                "description": "Submit extracted knowledge graph triples from input text",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "triples": {
                            "type": "array",
                            "description": "List of extracted triple dictionary objects with keys: (head,relation,tail)",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "head": {
                                        "type": "string",
                                        "description": "Head entity: cause, process parameter, material grade, or component",
                                    },
                                    "relation": {
                                        "type": "string",
                                        "description": "Predicate: causal impact, mechanism, or structural relation",
                                    },
                                    "tail": {
                                        "type": "string",
                                        "description": "Tail entity: defect outcome, physical state, or target parameter",
                                    },
                                },
                                "required": ["head", "relation", "tail"],
                            },
                        }
                    },
                    "required": ["triples"],
                },
            },
        }
    ]

    output_triples = []
    try:
        completion = client.chat.completions.create(
            model=model_name,
            messages=messages,
            tools=tools,
            tool_choice={
                "type": "function",
                "function": {"name": "extract_triples_schema"},
            },
            temperature=0.3,
            max_tokens=1500,
            stream=False,
        )
        response_message = completion.choices[0].message
        if response_message.tool_calls:
            arguments_string = response_message.tool_calls[0].function.arguments
            arguments = json.loads(arguments_string)
            output_triples = arguments.get("triples", [])
    except Exception as e:  # noqa: BLE001
        print(f"[WARNING]One text's triple extract failed.{e}")

    return output_triples
