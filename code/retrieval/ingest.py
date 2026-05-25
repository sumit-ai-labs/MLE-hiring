"""Corpus ingestion."""

from __future__ import annotations

from pathlib import Path

from retrieval.markdown_chunker import DocumentChunk, chunk_markdown


def ingest_markdown(data_dir: Path, repo_root: Path) -> list[DocumentChunk]:
    chunks: list[DocumentChunk] = []
    for path in sorted(data_dir.rglob("*.md")):
        if "api_specs" in path.parts:
            continue
        chunks.extend(chunk_markdown(path, repo_root))
    return chunks
