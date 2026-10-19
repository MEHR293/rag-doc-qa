"""FastAPI backend. Run: uvicorn app.api:app --reload"""
import shutil
from pathlib import Path
from typing import List

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app.config import settings
from app.ingest import SUPPORTED, ingest_path
from app.rag import RAGPipeline

app = FastAPI(title="RAG Document Q&A", version="1.0.0")
_pipeline: RAGPipeline | None = None


def pipeline() -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
        if not settings.openai_api_key:
            raise HTTPException(500, "OPENAI_API_KEY is not set. Copy .env.example to .env.")
        _pipeline = RAGPipeline()
    return _pipeline


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int = Field(default=settings.top_k, ge=1, le=20)


@app.get("/health")
def health():
    return {"status": "ok", "chunks_indexed": pipeline().num_chunks}


@app.post("/ingest")
async def ingest(files: List[UploadFile] = File(...)):
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    total, names = 0, []
    for f in files:
        name = Path(f.filename or "").name
        if Path(name).suffix.lower() not in SUPPORTED:
            raise HTTPException(400, f"Unsupported file: {name}. Allowed: {sorted(SUPPORTED)}")
        dest = settings.upload_dir / name
        with dest.open("wb") as out:
            shutil.copyfileobj(f.file, out)
        total += pipeline().add_chunks(ingest_path(dest))
        names.append(name)
    return {"files": names, "chunks_added": total, "chunks_total": pipeline().num_chunks}


@app.post("/ask")
def ask(req: AskRequest):
    return pipeline().ask(req.question, req.top_k)


@app.post("/baseline")
def baseline(req: AskRequest):
    return pipeline().baseline(req.question)


@app.delete("/index")
def clear_index():
    pipeline().reset()
    return {"status": "cleared"}
