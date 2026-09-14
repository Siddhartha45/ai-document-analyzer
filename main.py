import os

from fastapi import FastAPI, HTTPException, Depends, status
from sqlalchemy.orm import Session
from google import genai

from document_analyzer import (
    analyze_document,
    InvalidAPIKeyError,
    RateLimitExceededError,
    ValidationExhaustedError,
)
from schemas import DocumentRequest, QuestionRequest
from models import DocumentModel, ChunkModel
from database import Base, engine, get_db
from helpers import chunk_text
from rag import answer_question
from tool_calling import tool_calling

Base.metadata.create_all(bind=engine)
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

app = FastAPI()


@app.post("/documents")
def create_document(request: DocumentRequest, db: Session = Depends(get_db)):
    try:
        analysis = analyze_document(request.text)
    except InvalidAPIKeyError as e:
        raise HTTPException(
            status_code=503, detail="Service is temporarily misconfigured."
        ) from e
    except RateLimitExceededError as e:
        raise HTTPException(
            status_code=503, detail="Service is temporarily overloaded."
        ) from e
    except ValidationExhaustedError as e:
        raise HTTPException(
            status_code=502, detail="Unable to process this document reliably."
        ) from e

    db_document = DocumentModel(
        title=request.title,
        text=request.text,
        summary=analysis.summary,
        key_points=analysis.key_points,
        entities=[entity.model_dump() for entity in analysis.entities],
        action_items=[item.model_dump() for item in analysis.action_items],
    )
    db.add(db_document)
    db.commit()
    db.refresh(db_document)

    chunks = chunk_text(request.text, chunk_size=200, overlap=50)
    chunk_embeddings = client.models.embed_content(
        model="gemini-embedding-001", contents=chunks
    )

    for chunk, embedding in zip(chunks, chunk_embeddings.embeddings):
        db_chunk = ChunkModel(
            document_id=db_document.id,
            chunk_text=chunk,
            embedding=embedding.values,
        )
        db.add(db_chunk)
    db.commit()
    db.refresh(db_document)

    return db_document


@app.post("/documents/ask/rag")
def ask_question(request: QuestionRequest, db: Session = Depends(get_db)):
    """endpoint for rag system only - for asking question"""
    return answer_question(request.question, db)


@app.post("/documents/ask")
def call_tools(request: QuestionRequest, db: Session = Depends(get_db)):
    """endpoint for asking question - multiple tools"""
    return tool_calling(request.question, db)


@app.get("/documents/{document_id}")
def retrieve_document(document_id: int, db: Session = Depends(get_db)):
    document = db.get(DocumentModel, document_id)
    if document:
        return document
    else:
        raise HTTPException(status_code=404, detail="Document not found.")


@app.delete("/documents/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(document_id: int, db: Session = Depends(get_db)):
    document = db.get(DocumentModel, document_id)
    if document:
        db.delete(document)
        db.commit()
    else:
        raise HTTPException(status_code=404, detail="Document not found.")
