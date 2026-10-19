# RAG-Based Document Q&A System

A full-stack Retrieval-Augmented Generation (RAG) pipeline. Upload documents, embed them
into a **FAISS** vector store with **OpenAI embeddings**, and get **grounded answers with
source citations** through a **FastAPI** backend and **Streamlit** UI. Orchestrated with **LangChain**.

## Architecture

```
Upload (PDF/TXT/MD) -> load -> chunk (1000 chars, 150 overlap) -> OpenAI embeddings -> FAISS (persisted)
Question -> embed -> top-k similarity search -> prompt w/ numbered context -> LLM -> answer + [citations]
```

Why it reduces hallucinations: the model is instructed to answer only from retrieved
passages, cite them, and reply "I don't know" when the context lacks the answer.
`eval/evaluate.py` measures this against a vanilla-LLM baseline.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # add your OPENAI_API_KEY

uvicorn app.api:app --reload          # terminal 1 -> http://localhost:8000/docs
streamlit run ui/streamlit_app.py     # terminal 2 -> http://localhost:8501
```

Try it with `data/sample/acme_handbook.txt`, e.g. *"How many vacation days do I get?"*

## API

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Status + number of indexed chunks |
| POST | `/ingest` | Multipart upload of one or more files |
| POST | `/ask` | `{"question": "...", "top_k": 4}` -> grounded answer + sources |
| POST | `/baseline` | Same question with no retrieval (for comparison) |
| DELETE | `/index` | Clear the vector store |

## Evaluation

```bash
python -m eval.evaluate
```
Reports accuracy on answerable questions and hallucination rate on questions whose answers
are *not* in the documents. Add your own cases to `eval/questions.json`. Run it yourself
and record your numbers here; results depend on model and data.

## Tests

```bash
pytest
```

## Project layout

```
app/        config, ingestion, RAG pipeline, FastAPI
ui/         Streamlit app
eval/       RAG-vs-baseline evaluation
tests/      unit tests
data/sample sample document
```
