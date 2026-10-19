from pathlib import Path

import pytest

from app.ingest import ingest_path, load_file, split_documents


def test_txt_loading_and_chunking(tmp_path: Path):
    f = tmp_path / "a.txt"
    f.write_text("hello world. " * 400)
    chunks = ingest_path(f)
    assert len(chunks) > 1
    assert all(c.metadata["source"] == "a.txt" for c in chunks)
    assert [c.metadata["chunk_id"] for c in chunks] == list(range(len(chunks)))


def test_chunk_size_respected(tmp_path: Path):
    f = tmp_path / "a.md"
    f.write_text("word " * 2000)
    chunks = split_documents(load_file(f), chunk_size=200, chunk_overlap=20)
    assert max(len(c.page_content) for c in chunks) <= 200


def test_unsupported_type(tmp_path: Path):
    f = tmp_path / "a.exe"
    f.write_text("x")
    with pytest.raises(ValueError):
        load_file(f)
