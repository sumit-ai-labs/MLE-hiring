"""Routed hybrid retriever: BM25 + sentence-transformer semantic retrieval."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import replace
from typing import Iterable

from config import BM25_WEIGHT, CROSS_ENCODER_CANDIDATES, EMBEDDING_WEIGHT, RERANK_TOP_K, TOP_K
from retrieval.markdown_chunker import DocumentChunk
from retrieval.reranker import rerank


class HybridRetriever:
    def __init__(self, chunks: list[DocumentChunk]):
        self.chunks = chunks
        self.tokenized = [_tokens(chunk.content + " " + chunk.path + " " + chunk.heading) for chunk in chunks]
        self.doc_freq = _doc_freq(self.tokenized)
        self.avg_len = sum(len(tokens) for tokens in self.tokenized) / max(1, len(self.tokenized))
        self.semantic_backend = _SemanticBackend(chunks)

    def retrieve(
        self,
        query: str,
        company: str,
        product_area: str = "",
        classifier_confidence: float = 0.0,
        top_k: int = TOP_K,
    ) -> list[DocumentChunk]:
        candidate_indices = self._candidate_indices(company, classifier_confidence)
        if not candidate_indices:
            candidate_indices = list(range(len(self.chunks)))
        bm25 = self._bm25_scores(query, candidate_indices)
        semantic = self.semantic_backend.scores(query, candidate_indices)
        merged: list[DocumentChunk] = []
        for idx in candidate_indices:
            score = BM25_WEIGHT * bm25.get(idx, 0.0) + EMBEDDING_WEIGHT * semantic.get(idx, 0.0)
            if score <= 0:
                continue
            chunk = replace(
                self.chunks[idx],
                bm25_score=round(bm25.get(idx, 0.0), 6),
                embedding_score=round(semantic.get(idx, 0.0), 6),
                score=round(score, 6),
            )
            merged.append(chunk)
        merged = sorted(merged, key=lambda c: (c.score, c.path, c.heading), reverse=True)[: max(CROSS_ENCODER_CANDIDATES, top_k)]
        if not merged and classifier_confidence >= 0.55:
            return self.retrieve(query, "none", product_area, 0.0, top_k)
        return rerank(query, merged, company, product_area, RERANK_TOP_K)

    def _candidate_indices(self, company: str, classifier_confidence: float) -> list[int]:
        if company in {"claude", "devplatform", "visa"} and classifier_confidence >= 0.5:
            return [idx for idx, chunk in enumerate(self.chunks) if chunk.company == company]
        return list(range(len(self.chunks)))

    def _bm25_scores(self, query: str, indices: Iterable[int]) -> dict[int, float]:
        q_tokens = _tokens(query)
        if not q_tokens:
            return {}
        raw: dict[int, float] = {}
        n_docs = len(self.tokenized)
        k1 = 1.5
        b = 0.75
        for idx in indices:
            freqs = Counter(self.tokenized[idx])
            doc_len = len(self.tokenized[idx]) or 1
            score = 0.0
            for token in q_tokens:
                if token not in freqs:
                    continue
                df = self.doc_freq.get(token, 0)
                idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
                tf = freqs[token]
                score += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * doc_len / self.avg_len))
            raw[idx] = score
        max_score = max(raw.values(), default=0.0)
        if max_score <= 0:
            return {}
        return {idx: min(1.0, score / max_score) for idx, score in raw.items()}


class _SemanticBackend:
    def __init__(self, chunks: list[DocumentChunk]):
        self.chunks = chunks
        self.mode = "none"
        self.model = None
        self.embeddings = None
        self.vectorizer = None
        self.matrix = None
        texts = [chunk.content[:2000] for chunk in chunks]
        try:
            from sentence_transformers import SentenceTransformer

            try:
                self.model = SentenceTransformer("all-MiniLM-L6-v2", local_files_only=True)
            except TypeError:
                self.model = SentenceTransformer("all-MiniLM-L6-v2")
            self.embeddings = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
            self.mode = "sentence_transformers"
        except Exception:
            try:
                from sklearn.feature_extraction.text import HashingVectorizer
                from sklearn.preprocessing import normalize

                self.vectorizer = HashingVectorizer(n_features=2**14, alternate_sign=False, norm=None)
                self.matrix = normalize(self.vectorizer.transform(texts))
                self.mode = "hashing"
            except Exception:
                self.mode = "overlap"

    def scores(self, query: str, indices: Iterable[int]) -> dict[int, float]:
        index_list = list(indices)
        if not index_list:
            return {}
        if self.mode == "sentence_transformers" and self.model is not None and self.embeddings is not None:
            q = self.model.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]
            values = {}
            for idx in index_list:
                values[idx] = max(0.0, float(q @ self.embeddings[idx]))
            return _normalize(values)
        if self.mode == "hashing" and self.vectorizer is not None and self.matrix is not None:
            from sklearn.preprocessing import normalize

            qv = normalize(self.vectorizer.transform([query]))
            sims = (self.matrix[index_list] @ qv.T).toarray().ravel()
            return _normalize({idx: max(0.0, float(sim)) for idx, sim in zip(index_list, sims)})
        q_tokens = set(_tokens(query))
        values = {}
        for idx in index_list:
            d_tokens = set(_tokens(self.chunks[idx].content))
            values[idx] = len(q_tokens & d_tokens) / max(1, len(q_tokens | d_tokens))
        return _normalize(values)


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9][a-z0-9_'-]{1,}", (text or "").lower())


def _doc_freq(tokenized: list[list[str]]) -> dict[str, int]:
    df: dict[str, int] = {}
    for tokens in tokenized:
        for token in set(tokens):
            df[token] = df.get(token, 0) + 1
    return df


def _normalize(values: dict[int, float]) -> dict[int, float]:
    max_score = max(values.values(), default=0.0)
    if max_score <= 0:
        return {}
    return {idx: min(1.0, value / max_score) for idx, value in values.items()}
