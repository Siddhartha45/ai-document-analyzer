import os

from google import genai
from google.genai import types
from dotenv import load_dotenv
from pydantic import ValidationError
from functools import partial
from sqlalchemy.orm import Session

from tools import calculator_tool, rag_tool
from helpers import calculate
from schemas import CalculatorArgs, QuestionRequest
from rag import answer_question

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


def tool_calling(user_message: str, db: Session):
    system_instruction = """
    only call calculator tool if 2 numbers and a operation exists in a message.
    Similarly only call rag tool if the question is related to official documents and
    policies.
    """
    config = types.GenerateContentConfig(
        system_instruction=system_instruction, tools=[calculator_tool, rag_tool]
    )
    contents = [types.Content(role="user", parts=[types.Part(text=user_message)])]

    # First llm call for deciding tool use
    response = client.models.generate_content(
        model="gemini-3.6-flash",
        config=config,
        contents=contents,
    )

    candidate_part = response.candidates[0].content.parts[0]

    if candidate_part.function_call is None:
        return response.text  # No tool needed, return directly

    function_call = candidate_part.function_call

    tool_registry = {
        "calculator": (calculate, CalculatorArgs),
        "rag": (partial(answer_question, db=db), QuestionRequest),
    }

    try:
        args = tool_registry[function_call.name][1].model_validate(function_call.args)
        result = tool_registry[function_call.name][0](**args.model_dump())
    except ValidationError as e:
        result = f"Validation Error: {e}"
    except ValueError as e:
        result = f"Value Error: {e}"

    contents.append(response.candidates[0].content)
    contents.append(
        types.Content(
            role="user",
            parts=[
                types.Part.from_function_response(
                    name=function_call.name, response={"result": result}
                )
            ],
        )
    )
    # print("CONTENTS:", contents)

    final_response = client.models.generate_content(
        model="gemini-3.6-flash", config=config, contents=contents
    )

    return final_response.text
