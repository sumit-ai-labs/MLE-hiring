"""Deterministic confidence calibration via weighted signal fusion."""

from __future__ import annotations

from retrieval.markdown_chunker import DocumentChunk


def calibrate_confidence(
    classifier_confidence: float,
    chunks: list[DocumentChunk],
    tool_valid: bool,
    tool_errors: list[str],
    safety_flag: bool,
    pii_detected: bool,
    missing_docs: bool,
    response: str,
    source_documents: str,
    cross_company_ambiguity: bool = False,
) -> float:
    retrieval_quality = _retrieval_quality(chunks, missing_docs)
    classifier_signal = _clamp(classifier_confidence)
    if cross_company_ambiguity:
        classifier_signal -= 0.18
    tool_signal = 1.0 if tool_valid and not tool_errors else 0.35
    safety_signal = 0.72 if safety_flag else 1.0
    if pii_detected:
        safety_signal -= 0.08
    grounding_signal = _grounding_score(response, source_documents, chunks)

    score = (
        0.35 * retrieval_quality
        + 0.20 * classifier_signal
        + 0.20 * tool_signal
        + 0.15 * safety_signal
        + 0.10 * grounding_signal
    )
    if missing_docs:
        score = min(score, 0.55)
    if cross_company_ambiguity:
        score = min(score, 0.68)
    if safety_flag:
        score = min(score, 0.86)
    if tool_errors:
        score = min(score, 0.62)
    return round(_clamp(score, 0.05, 0.98), 2)


def _retrieval_quality(chunks: list[DocumentChunk], missing_docs: bool) -> float:
    if missing_docs or not chunks:
        return 0.2
    scores = sorted([max(chunk.rerank_score, chunk.score) for chunk in chunks], reverse=True)
    top = _clamp(scores[0])
    gap = _clamp(scores[0] - scores[1], 0.0, 0.35) / 0.35 if len(scores) > 1 else 1.0
    specificity = min(1.0, len({chunk.path for chunk in chunks[:3]}) / 3)
    return _clamp(0.62 * top + 0.23 * gap + 0.15 * specificity)


def _grounding_score(response: str, source_documents: str, chunks: list[DocumentChunk]) -> float:
    if not response.strip():
        return 0.1
    if not source_documents and chunks:
        return 0.35
    if not source_documents:
        return 0.55 if "outside the scope" in response.lower() else 0.25
    cited = source_documents.split("|")
    citation_score = min(1.0, len(cited) / 2)
    response_terms = {term for term in response.lower().split() if len(term) > 4}
    evidence_text = " ".join(chunk.content.lower() for chunk in chunks[:3])
    overlap = sum(1 for term in response_terms if term in evidence_text)
    overlap_score = min(1.0, overlap / max(1, min(8, len(response_terms))))
    return _clamp(0.55 * citation_score + 0.45 * overlap_score)


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))
