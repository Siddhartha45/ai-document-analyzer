import os

from google import genai
from dotenv import load_dotenv
from sqlalchemy import text
from sqlalchemy.orm import Session

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


def answer_question(question: str, db: Session):
    question_result = client.models.embed_content(
        model="gemini-embedding-001", contents=question
    )
    question_embedding = question_result.embeddings[0].values

    results = db.execute(
        text("""
        SELECT chunks.chunk_text, documents.title, chunks.embedding <=> :question_embedding AS distance
        FROM chunks
        JOIN documents ON chunks.document_id = documents.id
        ORDER BY distance ASC
        LIMIT 2
    """),
        {"question_embedding": str(question_embedding)},
    ).fetchall()

    retrieved_chunks = []
    for row in results:
        if row.distance < 0.35:
            retrieved_chunks.append((row.title, row.chunk_text))
    if len(retrieved_chunks) == 0:
        return "I don't have enough information to answer that question."

    sources = set()
    context_pieces = []
    for title, chunk_text in retrieved_chunks:
        context_pieces.append(f"[Source: {title}]\n{chunk_text}")
        sources.add(title)

    context_text = "\n\n".join(context_pieces)
    user_message = f"""
    Context:
    {context_text}
    
    Question: {question}
    """
    system_instruction = """
    You are a helpful assistant that answers questions using ONLY the provided context.
    If the context does not contain enough information to answer the question, say so 
    explicitly rather than guessing or using outside knowledge.
    """

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        config=genai.types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0,
        ),
        contents=user_message,
    )

    return {"answer": response.text, "sources": list(sources)}
