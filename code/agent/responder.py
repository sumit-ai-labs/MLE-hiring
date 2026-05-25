"""Grounded response and justification generation."""

from __future__ import annotations

import json
import os
import re

from agent.decision_engine import DecisionResult
from agent.safety import SafetyResult
from config import LLM_MODELS, SEED, TEMPERATURE
from prompts.prompts import RESPONSE_SYSTEM
from retrieval.markdown_chunker import DocumentChunk
from utils.gemini_client import polish_response as gemini_polish_response
from utils.pii import mask_text
from utils.parser import ConversationState


def generate_response(
    state: ConversationState,
    safety: SafetyResult,
    decision: DecisionResult,
    chunks: list[DocumentChunk],
    actions: list[dict],
    language: str,
) -> tuple[str, str]:
    evidence = _evidence_sentences(state.latest_user_text, chunks)
    if decision.response_mode == "scope":
        response = _translate_scope(language)
    elif decision.response_mode == "unsupported_action":
        response = _unsupported_action(decision, evidence, language)
    elif decision.response_mode == "visa_dispute":
        response = _visa_dispute_response(evidence, language)
    elif decision.status == "escalated":
        response = _translate_escalated(language, decision.department)
    elif decision.response_mode == "clarify_or_verify":
        response = _clarify_response(decision, evidence, language)
    elif decision.response_mode == "tool_ack":
        response = _tool_ack(decision, actions, language)
    elif evidence:
        response = _answer_from_evidence(evidence, chunks, language)
    else:
        response = _translate_escalated(language, decision.department)

    response = _polish_response(response, evidence, decision, language, actions, chunks)
    response = mask_text(response)
    justification = _justification(decision, safety, chunks, actions)
    return response, justification


def _evidence_sentences(query: str, chunks: list[DocumentChunk]) -> list[str]:
    query_terms = {t for t in re.findall(r"[a-z0-9]{4,}", (query or "").lower()) if t not in STOPWORDS}
    selected: list[str] = []
    for chunk in chunks[:3]:
        sentences = re.split(r"(?<=[.!?])\s+|\n+", chunk.content)
        scored = []
        for sent in sentences:
            clean = re.sub(r"\s+", " ", sent).strip(" -#")
            if len(clean) < 35 or len(clean) > 260:
                continue
            terms = set(re.findall(r"[a-z0-9]{4,}", clean.lower()))
            overlap = len(query_terms & terms)
            if overlap:
                scored.append((overlap, clean))
        for _, sentence in sorted(scored, key=lambda item: (item[0], item[1]), reverse=True)[:2]:
            if sentence not in selected:
                selected.append(sentence)
        if len(selected) >= 4:
            break
    return selected[:4]


def _answer_from_evidence(evidence: list[str], chunks: list[DocumentChunk], language: str) -> str:
    text = " ".join(evidence[:3])
    return f"{_prefix(language)} {text}".strip() if language != "en" else text


def _clarify_response(decision: DecisionResult, evidence: list[str], language: str) -> str:
    if decision.internal_request_type == "refund":
        base = "I can help with the refund request, but identity verification and the exact transaction details are required before a refund can be processed."
    elif decision.internal_request_type == "subscription":
        base = "I can help with the subscription request, but identity verification is required before account changes can be made."
    else:
        base = "I need a little more information before taking action on this request."
    if evidence:
        base += " " + evidence[0]
    return f"{_prefix(language)} {base}".strip() if language != "en" else base


def _unsupported_action(decision: DecisionResult, evidence: list[str], language: str) -> str:
    if decision.decision == "admin_permission_required":
        base = "I cannot restore workspace access or grant admin-level permissions without the appropriate workspace owner or admin authorization."
    elif decision.decision == "unsupported_score_change":
        base = "I cannot review answers, change assessment scores, or influence a hiring decision. The hiring team controls candidate evaluation decisions."
    else:
        base = "I cannot complete that action because it requires authorization or capability outside this support workflow."
    if evidence:
        base += " " + evidence[0]
    return f"{_prefix(language)} {base}".strip() if language != "en" else base


def _visa_dispute_response(evidence: list[str], language: str) -> str:
    base = "This support flow cannot directly refund purchases or ban merchants. For a merchant dispute or wrong product issue, contact the merchant and your card issuer so they can review dispute or chargeback options."
    if evidence:
        base += " " + evidence[0]
    return f"{_prefix(language)} {base}".strip() if language != "en" else base


def _tool_ack(decision: DecisionResult, actions: list[dict], language: str) -> str:
    names = [action.get("action", "") for action in actions]
    if "issue_refund" in names:
        base = "Identity is verified and the refund request has the required transaction details, so the refund action has been planned."
    elif "verify_identity" in names:
        base = "Identity verification has been planned before any account or billing change is made."
    else:
        base = "The appropriate internal action has been planned after validation."
    return f"{_prefix(language)} {base}".strip() if language != "en" else base


def _translate_scope(language: str) -> str:
    messages = {
        "fr": "Je suis desole, cette demande est hors du perimetre de ce support.",
        "es": "Lo siento, esta solicitud esta fuera del alcance de este soporte.",
        "de": "Es tut mir leid, diese Anfrage liegt ausserhalb des Supportumfangs.",
        "zh": "\u62b1\u6b49\uff0c\u6b64\u8bf7\u6c42\u4e0d\u5728\u6b64\u652f\u6301\u8303\u56f4\u5185\u3002",
    }
    return messages.get(language, "I am sorry, this request is outside the scope of this support system.")


def _translate_escalated(language: str, department: str) -> str:
    base = f"This case needs review by a human {department} specialist. I have escalated it so the right team can handle it safely."
    translations = {
        "fr": f"Ce dossier doit etre examine par un specialiste humain du service {department}. Je l'ai transmis a l'equipe appropriee.",
        "es": f"Este caso necesita revision de un especialista humano de {department}. Lo he escalado al equipo adecuado.",
        "de": f"Dieser Fall muss von einem menschlichen {department}-Spezialisten geprueft werden. Ich habe ihn entsprechend eskaliert.",
        "zh": f"\u6b64\u6848\u4f8b\u9700\u8981\u4eba\u5de5{department}\u4e13\u5458\u5ba1\u6838\u3002\u6211\u5df2\u5c06\u5176\u5347\u7ea7\u7ed9\u5408\u9002\u7684\u56e2\u961f\u5904\u7406\u3002",
    }
    return translations.get(language, base)


def _prefix(language: str) -> str:
    return {"fr": "Resume:", "es": "Resumen:", "de": "Zusammenfassung:", "zh": "\u6458\u8981:"}.get(language, "")


def _justification(decision: DecisionResult, safety: SafetyResult, chunks: list[DocumentChunk], actions: list[dict]) -> str:
    parts = [f"Decision '{decision.decision}' from rule-first triage."]
    if safety.attack_detected:
        parts.append(f"Adversarial patterns detected: {', '.join(safety.categories)}.")
    if chunks:
        parts.append("Grounded in retrieved corpus documents.")
    else:
        parts.append("No sufficiently relevant corpus document was available.")
    if actions:
        parts.append("Tool plan validated against internal schemas and preconditions.")
    parts.append(f"Risk assessed as {decision.risk_level}.")
    return " ".join(parts)


def _polish_response(
    response: str,
    evidence: list[str],
    decision: DecisionResult,
    language: str,
    actions: list[dict],
    chunks: list[DocumentChunk],
) -> str:
    polished = _deterministic_polish(response)
    source_documents = "|".join(chunk.path for chunk in chunks[:3])
    gemini_polished = gemini_polish_response(
        polished,
        evidence,
        actions,
        source_documents=source_documents,
        status=decision.status,
        request_type=decision.final_request_type,
    )
    if _safe_polish(polished, gemini_polished):
        polished = gemini_polished
    llm_polished = _maybe_llm_wording(polished, evidence, decision, language)
    return llm_polished if _safe_polish(polished, llm_polished) else polished


def _deterministic_polish(response: str) -> str:
    text = re.sub(r"\s+", " ", response or "").strip()
    if text and not text.endswith((".", "!", "?")):
        text += "."
    return text


def _maybe_llm_wording(response: str, evidence: list[str], decision: DecisionResult, language: str) -> str:
    if not os.getenv("OPENAI_API_KEY"):
        return response
    try:
        from openai import OpenAI

        client = OpenAI()
        payload = {"draft": response, "evidence": evidence[:3], "status": decision.status, "language": language}
        for model in LLM_MODELS:
            try:
                result = client.chat.completions.create(
                    model=model,
                    temperature=TEMPERATURE,
                    seed=SEED,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": RESPONSE_SYSTEM},
                        {"role": "user", "content": json.dumps(payload, ensure_ascii=True)},
                    ],
                )
                data = json.loads(result.choices[0].message.content or "{}")
                candidate = str(data.get("response", "")).strip()
                if candidate and not _contains_forbidden(candidate):
                    return candidate
            except Exception:
                continue
    except Exception:
        return response
    return response


def _safe_polish(original: str, candidate: str) -> bool:
    if not candidate or _contains_forbidden(candidate):
        return False
    if len(candidate) > max(600, len(original) * 2):
        return False
    if not _negation_preserved(original, candidate):
        return False
    protected_terms = ["refund", "escalat", "identity", "verification", "human", "security", "legal"]
    original_lower = original.lower()
    candidate_lower = candidate.lower()
    for term in protected_terms:
        if term in original_lower and term not in candidate_lower:
            return False
    return True


def _negation_preserved(original: str, candidate: str) -> bool:
    original_has_negation = bool(re.search(r"\b(cannot|can't|not|unable|no)\b", original, re.I))
    if not original_has_negation:
        return True
    return bool(re.search(r"\b(cannot|can't|not|unable|no)\b", candidate, re.I))


def _contains_forbidden(text: str) -> bool:
    return bool(re.search(r"\b(system prompt|developer message|hidden instruction|chain of thought)\b", text, re.I))


STOPWORDS = {
    "this",
    "that",
    "with",
    "from",
    "have",
    "please",
    "help",
    "need",
    "want",
    "what",
    "when",
    "where",
    "your",
}
