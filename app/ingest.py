"""Document loading and chunking."""
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings

SUPPORTED = {".pdf", ".txt", ".md"}


def load_file(path: Path) -> List[Document]:
    """Load a PDF / TXT / MD file into LangChain Documents (one per page for PDFs)."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED:
        raise ValueError(f"Unsupported file type '{suffix}'. Use: {sorted(SUPPORTED)}")

    if suffix == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        docs = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                docs.append(Document(page_content=text,
                                     metadata={"source": path.name, "page": i + 1}))
        return docs

    text = path.read_text(encoding="utf-8", errors="ignore")
    return [Document(page_content=text, metadata={"source": path.name})]


def split_documents(docs: List[Document],
                    chunk_size: int | None = None,
                    chunk_overlap: int | None = None) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size or settings.chunk_size,
        chunk_overlap=chunk_overlap or settings.chunk_overlap,
    )
    chunks = splitter.split_documents(docs)
    for i, c in enumerate(chunks):
        c.metadata["chunk_id"] = i
    return chunks


def ingest_path(path: Path) -> List[Document]:
    return split_documents(load_file(path))
