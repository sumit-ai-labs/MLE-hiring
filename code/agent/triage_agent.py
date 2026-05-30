"""End-to-end deterministic support triage pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agent.classifier import classify
from agent.conflict_resolver import resolve_conflicts
from agent.decision_engine import decide
from agent.planner import plan_tools
from agent.responder import generate_response
from agent.safety import analyze_safety, clean_support_text
from config import DATA_DIR, DEFAULT_LOG, ENABLE_V2_POLICY_ENGINE, ENABLE_V2_STATE_MACHINE, ENABLE_V2_STRUCTURED_MEMORY, REPO_ROOT, TOOL_SPEC_PATH
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.ingest import ingest_markdown
from tools.executor import execute_actions
from tools.registry import ToolRegistry
from tools.validator import validate_actions
from utils.confidence import calibrate_confidence
from utils.language import detect_language
from utils.logger import TriageLogger
from utils.parser import parse_issue
from utils.pii import detect_pii


class TriageAgent:
    def __init__(self, data_dir: Path = DATA_DIR, log_path: Path | None = DEFAULT_LOG):
        self.chunks = ingest_markdown(data_dir, REPO_ROOT)
        self.retriever = HybridRetriever(self.chunks)
        self.registry = ToolRegistry(TOOL_SPEC_PATH)
        self.logger = TriageLogger(log_path)
        self.memory_store = None
        if ENABLE_V2_STRUCTURED_MEMORY:
            from memory.memory_store import MemoryStore

            self.memory_store = MemoryStore()

    def process_row(self, row: dict[str, Any], ticket_id: int) -> dict[str, str]:
        issue = _get(row, "issue")
        subject = _get(row, "subject")
        input_company = _get(row, "company")

        # CSV row -> conversation parser
        state = parse_issue(issue, subject, input_company)

        # -> safety layer
        safety = analyze_safety(" ".join([state.subject, state.full_text]))
        support_text = clean_support_text(state.full_text)

        # -> PII detection
        pii = detect_pii(" ".join([state.subject, state.full_text]))
        safe_support_text = detect_pii(support_text).masked_text

        # -> language detection
        language = detect_language(state.latest_user_text or state.full_text)

        # -> company classifier -> product classifier
        classification = classify(safe_support_text, subject, input_company)

        # -> conflict resolver
        classification = resolve_conflicts(state, classification)

        # -> retrieval router -> hybrid retriever -> reranker
        retrieval_query = " ".join([subject, safe_support_text, classification.product_area]).strip()
        retrieved = self.retriever.retrieve(
            retrieval_query,
            classification.company,
            classification.product_area,
            classification.confidence,
        )

        # -> decision engine
        decision = decide(state, safety, classification, len(retrieved))
        if ENABLE_V2_POLICY_ENGINE:
            from policy_engine.adapter import decide_with_policy_engine

            decision = decide_with_policy_engine(decision, state, safety, classification, len(retrieved))

        # -> tool planner -> validator -> executor
        proposed_actions = plan_tools(decision, state)
        validation = validate_actions(proposed_actions, self.registry, state, decision.internal_request_type)
        executed_actions = execute_actions(validation.actions)

        # -> grounded responder
        response, justification = generate_response(state, safety, decision, retrieved, executed_actions, language)

        self._run_v2_shadow(ticket_id, state, safety, pii, language, classification, decision, validation.actions)

        source_documents = _source_documents(retrieved, decision)
        confidence = calibrate_confidence(
            classifier_confidence=classification.confidence,
            chunks=retrieved,
            tool_valid=validation.valid,
            tool_errors=validation.errors,
            safety_flag=safety.attack_detected,
            pii_detected=pii.detected,
            missing_docs=not bool(retrieved) and classification.company != "none",
            response=response,
            source_documents=source_documents,
            cross_company_ambiguity=_cross_company_ambiguity(input_company, classification.company, classification.confidence),
        )
        if decision.response_mode == "unsupported_action":
            confidence = min(confidence, 0.82)
        if safety.attack_detected:
            confidence = min(confidence, 0.86)
        if decision.risk_level == "critical":
            confidence = min(confidence, 0.90)
        final_risk_level = _bump_risk_for_pii(decision.risk_level, pii.categories)
        if decision.final_request_type == "invalid" and not source_documents:
            confidence = min(confidence, _invalid_confidence_cap(state.full_text))

        actions_taken = json.dumps(validation.actions, sort_keys=True, separators=(",", ":"))

        output = {
            "issue": issue,
            "subject": subject,
            "company": classification.display_company,
            "response": response,
            "product_area": decision.product_area,
            "status": decision.status,
            "request_type": decision.final_request_type,
            "justification": justification,
            "confidence_score": f"{confidence:.2f}",
            "source_documents": source_documents,
            "risk_level": final_risk_level,
            "pii_detected": "true" if pii.detected else "false",
            "language": language,
            "actions_taken": actions_taken,
        }

        self.logger.log_ticket(
            ticket_id,
            {
                "INPUT": f"subject={subject}; company={input_company}; issue={pii.masked_text[:600]}",
                "SAFETY": f"attack={safety.attack_detected}; categories={safety.categories}; risk={safety.risk_level}",
                "PII": f"detected={pii.detected}; categories={pii.categories}",
                "LANGUAGE": language,
                "CLASSIFICATION": f"company={classification.company}; product={classification.product_area}; request={classification.internal_request_type}; confidence={classification.confidence}",
                "RETRIEVAL": _retrieval_diagnostics(classification.company, classification.confidence, retrieved),
                "DECISION": f"status={decision.status}; decision={decision.decision}; risk={final_risk_level}",
                "TOOLS": actions_taken + (f"; validation_errors={validation.errors}" if validation.errors else ""),
                "RESPONSE": response[:600],
            },
        )
        return output

    def _run_v2_shadow(self, ticket_id, state, safety, pii, language, classification, decision, actions) -> None:
        if ENABLE_V2_STRUCTURED_MEMORY:
            from memory.memory_extractor import extract_memory

            memory = extract_memory(state, classification, safety, pii, language, decision, actions)
            if self.memory_store is not None:
                self.memory_store.upsert(ticket_id, memory)
        if ENABLE_V2_STATE_MACHINE:
            from state_machine.state_machine import build_shadow_lifecycle

            security_signal = None
            if decision.internal_request_type in {"account_compromise", "fraud", "legal"}:
                security_signal = decision.internal_request_type
            elif safety.attack_detected:
                security_signal = "prompt_injection"
            build_shadow_lifecycle(
                ticket_id,
                identity_verified=state.identity_verified,
                security_signal=security_signal,
                risk_level=decision.risk_level,
                actions=actions,
                trace_enabled=False,
            )


def _get(row: dict[str, Any], name: str) -> str:
    for key, value in row.items():
        if key.strip().lower().replace(" ", "_") == name:
            return "" if value is None else str(value)
    return ""


def _source_documents(chunks, decision) -> str:
    if decision.response_mode in {"scope", "escalate"}:
        return ""
    seen = []
    for chunk in chunks:
        if chunk.path not in seen and (REPO_ROOT / chunk.path).exists():
            seen.append(chunk.path)
    return "|".join(seen[:3])


def _bump_risk_for_pii(risk_level: str, categories: list[str]) -> str:
    if not categories:
        return risk_level
    order = {"low": 0, "medium": 1, "high": 2, "critical": 3}
    high_sensitivity = {"payment_card", "ssn", "api_key", "passport"}
    minimum = "medium" if set(categories) & high_sensitivity else risk_level
    return max([risk_level, minimum], key=lambda risk: order.get(risk, 0))


def _cross_company_ambiguity(input_company: str, classified_company: str, confidence: float) -> bool:
    normalized = (input_company or "").strip().lower()
    normalized = {"anthropic": "claude", "hackerrank": "devplatform"}.get(normalized, normalized)
    if normalized in {"", "none"}:
        return confidence < 0.45
    return classified_company not in {normalized, "none"} and confidence < 0.8


def _invalid_confidence_cap(text: str) -> float:
    lower = (text or "").lower()
    if any(token in lower for token in ["thank you", "thanks", "nevermind", "never mind"]):
        return 0.92
    return 0.60


def _retrieval_diagnostics(company_route: str, route_confidence: float, chunks) -> str:
    if not chunks:
        return f"company_route={company_route}; route_confidence={route_confidence}; top_docs=[]"
    parts = [f"company_route={company_route}; route_confidence={route_confidence}; top_docs=["]
    for chunk in chunks[:5]:
        parts.append(
            f"{chunk.path} "
            f"bm25={chunk.bm25_score:.3f} "
            f"embedding={chunk.embedding_score:.3f} "
            f"rerank={chunk.rerank_score:.3f}"
        )
    parts.append("]")
    return "\n".join(parts)
