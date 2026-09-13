import os

from google import genai
from google.genai import types
from dotenv import load_dotenv
from pydantic import ValidationError

from tools import calculator_tool
from helpers import calculate
from schemas import CalculatorArgs

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


def tool_calling(user_message: str):
    config = types.GenerateContentConfig(tools=[calculator_tool])
    contents = [types.Content(role="user", parts=[types.Part(text=user_message)])]
    print("FIRST CONTENT:", contents)

    # First llm call for deciding tool use
    response = client.models.generate_content(
        model="gemini-3.6-flash",
        config=config,
        contents=contents,
    )

    candidate_part = response.candidates[0].content.parts[0]
    print("CANDIDATE PART:", candidate_part)

    if candidate_part.function_call is None:
        return response.text  # No tool needed, return directly

    function_call = candidate_part.function_call
    print("FUNCTION CALL:", function_call)

    tool_registry = {"calculator": (calculate, CalculatorArgs)}

    try:
        args = tool_registry[function_call.name][1].model_validate(function_call.args)
        result = tool_registry[function_call.name][0](**args.model_dump())
    except ValidationError as e:
        result = f"Validation Error: {e}"
    except ValueError as e:
        result = f"Value Error: {e}"

    print("MODEL TURN:", response.candidates[0].content)
    contents.append(response.candidates[0].content)
    print("SECOND CONTENT:", contents)
    contents.append(
        types.Content(
            role="user",
            parts=[
                types.Part.from_function_response(
                    name="calculator", response={"result": result}
                )
            ],
        )
    )
    print("THIRD & FINAL CONTENT:", contents)

    final_response = client.models.generate_content(
        model="gemini-3.6-flash", config=config, contents=contents
    )

    return final_response.text
