"""Streamlit front end. Run: streamlit run ui/streamlit_app.py"""
import os

import requests
import streamlit as st

API = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="RAG Document Q&A", page_icon="📄", layout="wide")
st.title("📄 RAG Document Q&A")

with st.sidebar:
    st.header("1. Add documents")
    files = st.file_uploader("PDF, TXT or MD", type=["pdf", "txt", "md"],
                             accept_multiple_files=True)
    if st.button("Ingest", disabled=not files):
        with st.spinner("Chunking and embedding..."):
            r = requests.post(f"{API}/ingest",
                              files=[("files", (f.name, f.getvalue())) for f in files])
        if r.ok:
            d = r.json()
            st.success(f"Added {d['chunks_added']} chunks (total {d['chunks_total']}).")
        else:
            st.error(r.text)
    if st.button("Clear index"):
        requests.delete(f"{API}/index")
        st.info("Index cleared.")
    top_k = st.slider("Chunks to retrieve (k)", 1, 10, 4)
    compare = st.checkbox("Compare with vanilla LLM (no retrieval)")

st.header("2. Ask a question")
q = st.text_input("Your question")

if q:
    with st.spinner("Thinking..."):
        rag = requests.post(f"{API}/ask", json={"question": q, "top_k": top_k})
        base = requests.post(f"{API}/baseline", json={"question": q}) if compare else None
    if not rag.ok:
        st.error(rag.text)
        st.stop()

    cols = st.columns(2) if compare else [st.container()]
    with cols[0]:
        st.subheader("Grounded answer (RAG)")
        st.write(rag.json()["answer"])
        st.markdown("**Sources**")
        for s in rag.json()["sources"]:
            page = f", p.{s['page']}" if s.get("page") else ""
            with st.expander(f"[{s['id']}] {s['source']}{page}  (distance {s['score']:.3f})"):
                st.write(s["snippet"])
    if compare and base is not None and base.ok:
        with cols[1]:
            st.subheader("Vanilla LLM (no retrieval)")
            st.write(base.json()["answer"])
            st.caption("No sources: claims cannot be verified.")
