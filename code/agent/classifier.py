"""Deterministic company/product/request classification with optional LLM tie-breaks."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any

from config import LLM_MODELS, SEED, TEMPERATURE
from prompts.prompts import CLASSIFICATION_SYSTEM


@dataclass
class ClassificationResult:
    company: str
    display_company: str
    confidence: float
    product_area: str
    internal_request_type: str
    final_request_type: str
    signals: dict[str, float]


COMPANY_KEYWORDS: dict[str, dict[str, float]] = {
    "claude": {
        "claude": 3,
        "anthropic": 3,
        "max": 1.5,
        "pro plan": 2,
        "console": 1.8,
        "api key": 2.5,
        "token usage": 2,
        "workspace": 1.2,
        "team plan": 1.5,
    },
    "devplatform": {
        "devplatform": 3,
        "hackerrank": 3,
        "candidate": 2,
        "assessment": 2.5,
        "test": 1.4,
        "interview": 2,
        "codepair": 3,
        "screen": 2,
        "proctor": 2,
        "recruiter": 1.6,
    },
    "visa": {
        "visa": 3,
        "card": 2,
        "travel": 2,
        "traveller": 2,
        "traveler": 2,
        "cheque": 2.5,
        "reimbursement": 2,
        "merchant": 1.5,
        "emergency assistance": 3,
        "benefits": 1.5,
        "dispute": 2,
    },
}

PRODUCT_KEYWORDS: dict[str, dict[str, str]] = {
    "security": {"compromise": "security", "fraud": "security", "stolen": "security", "unauthorized": "security", "api key": "security"},
    "billing": {"refund": "billing", "invoice": "billing", "charged": "billing", "billing": "billing", "payment": "billing"},
    "subscription": {"subscription": "subscription", "cancel": "subscription", "upgrade": "subscription", "downgrade": "subscription", "plan": "subscription"},
    "account_access": {"login": "account_access", "password": "account_access", "access": "account_access", "workspace": "account_access", "seat": "account_access"},
    "technical_support": {"error": "technical_support", "bug": "technical_support", "deployment": "technical_support", "endpoint": "technical_support", "sandbox": "technical_support", "api": "technical_support"},
    "screen": {"assessment": "screen", "candidate": "screen", "test": "screen", "codepair": "screen", "interview": "screen", "proctor": "screen"},
    "travel_support": {"travel": "travel_support", "cheque": "travel_support", "emergency": "travel_support", "reimbursement": "travel_support"},
    "general_support": {"contact": "general_support", "support": "general_support", "help": "general_support"},
}

REQUEST_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("legal", re.compile(r"\b(sue|lawsuit|lawyer|attorney|legal|court|regulator|gdpr complaint|demand\w*|abogado)\b", re.I)),
    ("account_compromise", re.compile(r"\b(hacked|compromised|account takeover|unauthorized access|someone logged in|stolen account)\b", re.I)),
    ("fraud", re.compile(r"\b(fraud|fraudulent|identity theft|stolen card|unauthorized charge|unauthorized transaction)\b", re.I)),
    ("refund", re.compile(r"\b(refund|money back|give me my money|reimburse|charged back)\b", re.I)),
    ("billing", re.compile(r"\b(payment|billing|invoice|charge|charged|dispute|chargeback|order id)\b", re.I)),
    ("subscription", re.compile(r"\b(cancel|upgrade|downgrade|pause|subscription|plan)\b", re.I)),
    ("feature_request", re.compile(r"\b(feature request|please add|could you add|enhancement|would like a feature)\b", re.I)),
    ("bug", re.compile(r"\b(bug|error|crash|broken|not working|failed|stopped|issue)\b", re.I)),
    ("technical_issue", re.compile(r"\b(api|endpoint|deployment|sandbox|429|rate limit|login|password|access)\b", re.I)),
]


def classify(text: str, subject: str, input_company: str, allow_llm: bool = True) -> ClassificationResult:
    combined = " ".join([subject or "", text or ""]).lower()
    scores = {company: _keyword_score(combined, keywords) for company, keywords in COMPANY_KEYWORDS.items()}
    body_scores = {company: _keyword_score((text or "").lower(), keywords) for company, keywords in COMPANY_KEYWORDS.items()}
    input_norm = (input_company or "").strip().lower()
    if input_norm in {"claude", "anthropic"}:
        scores["claude"] += 0.8
    elif input_norm in {"devplatform", "hackerrank"}:
        scores["devplatform"] += 0.8
    elif input_norm == "visa":
        scores["visa"] += 0.8

    company, top_score = max(scores.items(), key=lambda item: (item[1], item[0]))
    body_company, body_top_score = max(body_scores.items(), key=lambda item: (item[1], item[0]))
    body_sorted = sorted(body_scores.values(), reverse=True)
    body_margin = body_sorted[0] - body_sorted[1] if len(body_sorted) > 1 else body_top_score
    sorted_scores = sorted(scores.values(), reverse=True)
    margin = sorted_scores[0] - sorted_scores[1] if len(sorted_scores) > 1 else top_score
    confidence = min(0.98, 0.35 + top_score / 8 + max(0.0, margin) / 5)
    if top_score < 1.5:
        if input_norm in {"claude", "anthropic", "devplatform", "hackerrank", "visa"}:
            company = {"anthropic": "claude", "hackerrank": "devplatform"}.get(input_norm, input_norm)
            confidence = 0.5
        else:
            company = "none"
            confidence = 0.35
    elif margin < 0.7 and allow_llm:
        llm = _llm_classify(text, subject)
        if llm.get("company") in {"claude", "devplatform", "visa", "none"}:
            company = llm["company"]
            confidence = max(confidence, float(llm.get("confidence", confidence)) * 0.9)
    if body_top_score >= 3.0 and body_margin >= 1.0 and body_company != company:
        company = body_company
        confidence = max(confidence, min(0.88, 0.55 + body_top_score / 10 + body_margin / 8))

    internal_request_type = detect_internal_request_type(combined)
    product_area = detect_product_area(combined, company, internal_request_type)
    return ClassificationResult(
        company=company,
        display_company=_display_company(company),
        confidence=round(confidence, 2),
        product_area=product_area,
        internal_request_type=internal_request_type,
        final_request_type=map_request_type(internal_request_type, company),
        signals=scores,
    )


def detect_internal_request_type(text: str) -> str:
    for name, pattern in REQUEST_PATTERNS:
        if pattern.search(text or ""):
            return name
    if re.search(r"\b(thank you|thanks|never mind|nevermind)\b", text or "", re.I):
        return "invalid"
    if len((text or "").strip()) < 12:
        return "invalid"
    return "faq"


def detect_product_area(text: str, company: str, request_type: str) -> str:
    if request_type in {"legal", "fraud", "account_compromise"}:
        return "security" if request_type != "legal" else "legal"
    for area, mapping in PRODUCT_KEYWORDS.items():
        for keyword, value in mapping.items():
            if keyword in text:
                return value
    if company == "visa":
        return "general_support"
    if company == "devplatform":
        return "screen"
    if company == "claude":
        return "account_management"
    return ""


def map_request_type(internal: str, company: str) -> str:
    if internal == "feature_request":
        return "feature_request"
    if internal == "bug":
        return "bug"
    if internal in {"invalid"} or company == "none":
        return "invalid"
    return "product_issue"


def _keyword_score(text: str, keywords: dict[str, float]) -> float:
    score = 0.0
    for keyword, weight in keywords.items():
        if keyword in text:
            score += weight
    return score


def _display_company(company: str) -> str:
    return {"claude": "Claude", "devplatform": "DevPlatform", "visa": "Visa", "none": "None"}.get(company, "None")


def _llm_classify(text: str, subject: str) -> dict[str, Any]:
    if not os.getenv("OPENAI_API_KEY"):
        return {}
    try:
        from openai import OpenAI

        client = OpenAI()
        payload = {"subject": subject, "ticket": text[:3000]}
        for model in LLM_MODELS:
            try:
                response = client.chat.completions.create(
                    model=model,
                    temperature=TEMPERATURE,
                    seed=SEED,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": CLASSIFICATION_SYSTEM},
                        {"role": "user", "content": json.dumps(payload, ensure_ascii=True)},
                    ],
                )
                return json.loads(response.choices[0].message.content or "{}")
            except Exception:
                continue
    except Exception:
        return {}
    return {}
