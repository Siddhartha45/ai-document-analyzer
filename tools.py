from google.genai import types

from schemas import Operation


calculator_tool = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name = "calculator",
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