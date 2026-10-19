"""RAG pipeline: FAISS vector store + OpenAI embeddings + grounded generation."""
from threading import Lock
from typing import Dict, List

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from app.config import settings

REFUSAL = "I don't know based on the provided documents."

RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a careful assistant that answers questions ONLY from the numbered "
     "context passages below. Cite the passages you use inline like [1], [2]. "
     f"If the context does not contain the answer, reply exactly: \"{REFUSAL}\" "
     "Never use outside knowledge.\n\nContext:\n{context}"),
    ("human", "{question}"),
])

BASELINE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Answer the question."),
    ("human", "{question}"),
])


class RAGPipeline:
    def __init__(self) -> None:
        self.embeddings = OpenAIEmbeddings(model=settings.embedding_model,
                                           api_key=settings.openai_api_key or None)
        self.llm = ChatOpenAI(model=settings.chat_model, temperature=0,
                              api_key=settings.openai_api_key or None)
        self.store: FAISS | None = None
        self._lock = Lock()
        self._load()

    # ---- index management -------------------------------------------------
    def _load(self) -> None:
        if (settings.index_dir / "index.faiss").exists():
            self.store = FAISS.load_local(
                str(settings.index_dir), self.embeddings,
                allow_dangerous_deserialization=True,  # index is created locally by this app
            )

    def _save(self) -> None:
        settings.index_dir.mkdir(parents=True, exist_ok=True)
        self.store.save_local(str(settings.index_dir))

    def add_chunks(self, chunks: List[Document]) -> int:
        if not chunks:
            return 0
        with self._lock:
            if self.store is None:
                self.store = FAISS.from_documents(chunks, self.embeddings)
            else:
                self.store.add_documents(chunks)
            self._save()
        return len(chunks)

    def reset(self) -> None:
        with self._lock:
            self.store = None
            if settings.index_dir.exists():
                for f in settings.index_dir.glob("index.*"):
                    f.unlink()

    @property
    def num_chunks(self) -> int:
        return self.store.index.ntotal if self.store else 0

    # ---- querying ---------------------------------------------------------
    def ask(self, question: str, k: int | None = None) -> Dict:
        if self.store is None:
            return {"answer": REFUSAL, "sources": []}

        hits = self.store.similarity_search_with_score(question, k=k or settings.top_k)
        context = "\n\n".join(f"[{i}] ({d.metadata.get('source')}) {d.page_content}"
                              for i, (d, _) in enumerate(hits, 1))
        msg = (RAG_PROMPT | self.llm).invoke({"context": context, "question": question})
        sources = [{
            "id": i,
            "source": d.metadata.get("source"),
            "page": d.metadata.get("page"),
            "score": float(score),  # L2 distance: lower = more similar
            "snippet": d.page_content[:300],
        } for i, (d, score) in enumerate(hits, 1)]
        return {"answer": msg.content, "sources": sources}

    def baseline(self, question: str) -> Dict:
        """Vanilla LLM with no retrieval, used for comparison."""
        msg = (BASELINE_PROMPT | self.llm).invoke({"question": question})
        return {"answer": msg.content, "sources": []}
