"""Markdown heading chunker preserving corpus metadata."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class DocumentChunk:
    content: str
    path: str
    company: str
    product: str
    category: str
    heading: str
    bm25_score: float = 0.0
    embedding_score: float = 0.0
    score: float = 0.0
    rerank_score: float = 0.0


HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


def chunk_markdown(path: Path, repo_root: Path) -> list[DocumentChunk]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    rel_path = path.relative_to(repo_root).as_posix()
    company, product, category = infer_metadata(rel_path)
    chunks: list[DocumentChunk] = []
    current_heading = path.stem.replace("-", " ")
    buffer: list[str] = []

    for line in text.splitlines():
        match = HEADING_RE.match(line)
        if match:
            _flush(chunks, buffer, rel_path, company, product, category, current_heading)
            current_heading = match.group(2).strip()
            buffer = [line]
        else:
            buffer.append(line)
    _flush(chunks, buffer, rel_path, company, product, category, current_heading)
    if not chunks and text.strip():
        chunks.append(
            DocumentChunk(
                content=text.strip(),
                path=rel_path,
                company=company,
                product=product,
                category=category,
                heading=current_heading,
            )
        )
    return chunks


def infer_metadata(rel_path: str) -> tuple[str, str, str]:
    parts = rel_path.split("/")
    company = parts[1].lower() if len(parts) > 1 and parts[0] == "data" else "unknown"
    product = parts[2].lower() if len(parts) > 2 else company
    category = parts[3].lower() if len(parts) > 3 else product
    if company == "claude":
        product = parts[2].lower() if len(parts) > 2 else "claude"
    if company == "devplatform" and product in {"general-help", "screen", "interview", "tests", "candidates"}:
        product = product.replace("-", "_")
    return company, product.replace("-", "_"), category.replace("-", "_")


def _flush(
    chunks: list[DocumentChunk],
    buffer: list[str],
    rel_path: str,
    company: str,
    product: str,
    category: str,
    heading: str,
) -> None:
    content = "\n".join(buffer).strip()
    if len(content) < 40:
        return
    chunks.append(
        DocumentChunk(
            content=content,
            path=rel_path,
            company=company,
            product=product,
            category=category,
            heading=heading,
        )
    )
