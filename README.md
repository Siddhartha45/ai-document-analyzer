# AI Document Analyzer & Knowledge Assistant

A FastAPI backend that ingests documents, extracts structured information from them using an LLM, and answers questions about them using retrieval-augmented generation (RAG) with PostgreSQL + pgvector.

This project was built as part of a hands-on, self-directed curriculum on applied AI/LLM engineering, following a backend development background (Django/DRF). It's a learning project, not a production deployment — see [Known Limitations](#known-limitations) below for an honest breakdown of what's built vs. what's still in progress.

## What it does

1. **Ingest a document** (`POST /documents`) — submit raw text, and the system:
   - Extracts a summary, key points, entities, and action items using a schema-constrained LLM call
   - Chunks the text and generates embeddings for each chunk
   - Stores the document, its analysis, and its chunks (with embeddings) in Postgres

2. **Ask a question about ingested documents** (`POST /documents/ask`) — submit a question, and the system:
   - Embeds the question and runs a vector similarity search (pgvector cosine distance) against stored chunks
   - Filters out low-relevance chunks below a calibrated distance threshold
   - If no chunk is relevant enough, explicitly refuses rather than guessing
   - Otherwise, generates a grounded answer using only the retrieved context, with source document titles attached independently of the LLM's own output

3. **Retrieve or delete a document** (`GET` / `DELETE /documents/{id}`)

## Architecture

```
document_analyze/
├── main.py              # FastAPI app: routes for create/retrieve/delete/ask
├── document_analyzer.py # LLM extraction call: structured output, retries, error handling
├── rag.py                # Retrieval + grounded answer generation
├── helpers.py            # Text chunking
├── models.py              # SQLAlchemy models (DocumentModel, ChunkModel)
├── schemas.py             # Pydantic request/response schemas
├── database.py            # DB engine/session setup
├── eval.py                 # Evaluation harness for the RAG pipeline
├── .env.example
```

**Ingestion and retrieval are decoupled**: `main.py` → `document_analyzer.py` → `helpers.py` handles ingestion (analyze, chunk, embed, store). `rag.py` handles retrieval independently — it has no knowledge of how a chunk got into the database, it only searches what exists. This separation is standard RAG architecture: the ingestion pipeline and the query pipeline are separate concerns that happen to share a data store.

## Tech stack

- **Python**, **FastAPI**, **SQLAlchemy**, **Pydantic**
- **PostgreSQL + pgvector** (Docker) — native vector similarity search via the `<=>` cosine distance operator
- **Google Gemini API** — generation (structured extraction, grounded Q&A) and embeddings (3072-dim)

## Key design decisions

- **Schema-constrained generation**: LLM extraction is validated against a Pydantic schema (`DocumentAnalysis`) rather than parsing free text, so malformed output is caught immediately.
- **Dual retry strategy**: exponential backoff for rate-limit/server errors (429, 504), separate corrective-feedback retries for schema validation failures — the model is re-prompted with its own validation error and asked to fix it.
- **Custom exception hierarchy**: distinct exceptions (`InvalidAPIKeyError`, `RateLimitExceededError`, `ValidationExhaustedError`) map to distinct HTTP status codes (502/503) instead of a single generic error response.
- **Prompt-injection-aware system instruction**: the extraction prompt explicitly tells the model to treat document content as data, not as instructions to follow.
- **Relevance-threshold filtering**: retrieved chunks below a similarity threshold are discarded. The threshold (`0.35`) was calibrated empirically — see [Evaluation](#evaluation) below.
- **Source citation independent of the LLM**: the `sources` returned to the user come from a SQL join on the retrieved chunks, not from anything the model says — so the citation can't be hallucinated.
- **Explicit refusal over guessing**: if no chunk clears the relevance threshold, the system returns "I don't have enough information to answer that question" without ever calling the LLM for that case.

## Evaluation

`eval.py` runs a fixed set of test questions against the pipeline and checks two things per case: whether the answer contains the expected content, and whether the returned sources match expectations — including a case that should trigger refusal (no relevant document exists for the question).

This eval harness caught a real bug during development: the initial relevance threshold (`0.5`) was too loose and let irrelevant chunks through on out-of-scope questions, producing a source citation on an answer that should have refused. Printing actual cosine distances across both relevant and irrelevant questions revealed a clear separation point, and the threshold was recalibrated to `0.35` and reverified against the eval set.

## Known Limitations

This is an actively evolving learning project. Current known gaps:

- **CRUD is Create/Retrieve/Delete only** — no Update endpoint yet.
- **Evaluation uses substring matching**, not semantic/LLM-as-judge scoring.
- **No authentication or multi-tenancy** — not wired in yet.
- **No agents, tool-calling, or deployment work** — planned for later stages of the curriculum, not yet started.
- **Retrieval tested against a single document type so far** (policy-style documents); broader document type coverage is a natural next step.

## Setup

1. Clone the repo and install dependencies.
2. Set up a PostgreSQL database with the `pgvector` extension enabled (via Docker or otherwise).
3. Copy `.env.example` to `.env` and fill in your `GEMINI_API_KEY` and `DATABASE_URL`.
4. Run the app — tables are created automatically on startup.

```bash
uvicorn main:app --reload
```

5. `test_document.txt` contains sample title/text data you can use to try the `POST /documents` endpoint. Example questions to test against `POST /documents/ask` — including a case that should trigger the refusal path — are in `eval.py`.
