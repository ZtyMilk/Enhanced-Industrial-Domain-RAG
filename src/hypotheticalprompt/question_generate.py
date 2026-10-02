import json

from openai import OpenAI


def question_generate(
    input_text: str,
    model_name: str,
    client: OpenAI,
    question_number: int,
):
    prompt = (
        f"Please analyze the following text chunk and generate {question_number} "
        f"specific, realistic user queries or hypothetical questions\n\n"
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
                "name": "hypothetical_question_schema",
                "description": "Submit generated hypothetical questions for a chunk",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "questions": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "The list of generated questions",
                        }
                    },
                    "required": ["questions"],
                },
            },
        }
    ]
    
    output_questions = []
    try:
        completion = client.chat.completions.create(
            model=model_name,
            messages=messages,
            tools=tools,
            tool_choice={
                "type": "function",
                "function": {"name": "hypothetical_question_schema"},
            },
            temperature=0.3,
            max_tokens=500,
            stream=False,
        )
        response_message = completion.choices[0].message
        if response_message.tool_calls:
            arguments_string = response_message.tool_calls[0].function.arguments
            arguments = json.loads(arguments_string)
            output_questions = arguments.get("questions", [])
    except Exception as e:  # noqa: BLE001
        print(f"[WARNING]One chunk's hype failed.{e}")

    return output_questions