from schemas import Operation


def chunk_text(text: str, chunk_size: int = 200, overlap: int = 50) -> list[str]:
    """Split text into overlapping chunks of a specified size."""
    start = 0
    chunks = []

    while start < len(text):
        chunk = text[start : start + chunk_size]
        chunks.append(chunk)
        start = start + (chunk_size - overlap)

    return chunks


def calculate(a: float, b: float, operation: Operation) -> float:
    if operation == Operation.ADD:
        return a + b
    elif operation == Operation.SUBTRACT:
        return a - b
    elif operation == Operation.MULTIPLY:
        return a * b
    elif operation == Operation.DIVIDE:
        if b == 0:
            raise ValueError("Division by zero")
        return a / b
