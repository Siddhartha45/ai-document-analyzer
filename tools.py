from google.genai import types

from schemas import Operation

calculator_tool = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name="calculator",
            description="Performs a basic arithmetic operation (add, subtract, multiply, divide) on two numbers.",
            parameters={
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "First operand"},
                    "b": {"type": "number", "description": "Second operand"},
                    "operation": {
                        "type": "string",
                        "enum": [op.value for op in Operation],
                        "description": "Which arithmetic operation to perform",
                    },
                },
                "required": ["a", "b", "operation"],
            },
        )
    ]
)


rag_tool = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name="rag",
            description="Answers a user's question using information found in the documents that have been uploaded. Use this for factual or content-related questions about stored documents.",
            parameters={
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The user's question, in natural language, to search for within the uploaded documents.",
                    }
                },
                "required": ["question"],
            },
        )
    ]
)
