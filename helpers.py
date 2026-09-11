def chunk_text(text: str, chunk_size: int = 200, overlap: int = 50) -> list[str]:
    start = 0
    chunks = []

    while start < len(text):
        chunk = text[start : start + chunk_size]
        chunks.append(chunk)
        start = start + (chunk_size - overlap)

    return chunks
