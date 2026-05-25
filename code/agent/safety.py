"""Deterministic adversarial safety layer."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field

from config import LLM_MODELS, SEED, TEMPERATURE
from prompts.prompts import SAFETY_CLASSIFIER_SYSTEM


@dataclass
class SafetyResult:
    attack_detected: bool
    doc_poisoning_detected: bool = False
    categories: list[str] = field(default_factory=list)
    risk_level: str = "low"
    cleaned_text: str = ""


INJECTION_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("ignore_instructions", re.compile(r"\b(ignore|disregard|forget|bypass|override)\b.{0,80}\b(instructions?|rules?|policy|previous|prior|above)\b", re.I | re.S)),
    ("system_exfiltration", re.compile(r"\b(show|reveal|print|display|dump|extract)\b.{0,80}\b(system prompt|developer message|hidden instructions?|internal prompt|policy)\b", re.I | re.S)),
    ("roleplay_override", re.compile(r"\b(pretend|roleplay|act as|you are now|simulate)\b.{0,80}\b(admin|developer|root|system|unrestricted)\b", re.I | re.S)),
    ("dan", re.compile(r"\b(DAN mode|do anything now|jailbreak|no longer restricted|unfiltered)\b", re.I)),
    ("tool_abuse", re.compile(r"\b(call|execute|run|force)\b.{0,60}\b(tool|api|refund|lock_account|issue_refund)\b.{0,60}\b(without|skip|no)\b.{0,40}\b(verification|approval|checks?)\b", re.I | re.S)),
    ("classification_override", re.compile(r"\b(classify|mark|set|output)\b.{0,60}\b(replied|escalated|low risk|high confidence|valid)\b", re.I | re.S)),
    ("spanish_injection", re.compile(r"\b(ignora|olvida|omite|revela|muestra)\b.{0,80}\b(instrucciones|mensaje del sistema|desarrollador|prompt)\b", re.I | re.S)),
    ("french_injection", re.compile(r"\b(ignorez|ignore|oubliez|oublie|revele|r\u00e9v\u00e8le|montre|affiche)\b.{0,80}\b(instructions|message syst[e\u00e8]me|d[e\u00e9]veloppeur|prompt)\b", re.I | re.S)),
    ("german_injection", re.compile(r"\b(ignoriere|vergiss|zeige|enth[u\u00fc]lle|offenbare)\b.{0,80}\b(anweisungen|systemnachricht|entwickler|system prompt|prompt)\b", re.I | re.S)),
    ("chinese_injection", re.compile(r"(\u5ffd\u7565|\u5fd8\u8bb0|\u6cc4\u9732|\u663e\u793a|\u5c55\u793a).{0,40}(\u6307\u4ee4|\u7cfb\u7edf\u63d0\u793a|\u5f00\u53d1\u8005|\u9690\u85cf)", re.I | re.S)),
]

DOC_POISON_PATTERNS = [
    re.compile(r"\b(ignore|override|disregard)\b.{0,80}\b(user|system|developer|triage|agent)\b", re.I | re.S),
    re.compile(r"\bsecret\s+(policy|instruction|prompt)\b", re.I),
]


def analyze_safety(text: str) -> SafetyResult:
    content = text or ""
    categories = [name for name, pattern in INJECTION_PATTERNS if pattern.search(content)]
    categories.extend(_heuristic_categories(content))
    if _llm_injection_check(content):
        categories.append("llm_injection_classifier")
    cleaned = clean_support_text(content)
    return SafetyResult(
        attack_detected=bool(categories),
        categories=sorted(set(categories)),
        risk_level="high" if categories else "low",
        cleaned_text=cleaned,
    )


def analyze_doc_poisoning(text: str) -> bool:
    return any(pattern.search(text or "") for pattern in DOC_POISON_PATTERNS)


def clean_support_text(text: str) -> str:
    lines = []
    for line in (text or "").splitlines():
        if any(pattern.search(line) for _, pattern in INJECTION_PATTERNS):
            continue
        lines.append(line)
    cleaned = "\n".join(lines).strip()
    return cleaned if cleaned else text


def _heuristic_categories(text: str) -> list[str]:
    lower = text.lower()
    categories: list[str] = []
    override_hits = sum(
        token in lower
        for token in ["ignore", "override", "forget", "disregard", "bypass", "jailbreak", "developer", "system prompt"]
    )
    if override_hits >= 2:
        categories.append("heuristic_override_cluster")
    if re.search(r"\b(output|return|respond)\b.{0,50}\b(exactly|only)\b.{0,50}\b(replied|escalated|low|high)\b", lower, re.S):
        categories.append("heuristic_output_manipulation")
    if re.search(r"\b(without|skip|bypass)\b.{0,50}\b(identity|verification|approval|validator)\b", lower, re.S):
        categories.append("heuristic_precondition_bypass")
    return categories


def _llm_injection_check(text: str) -> str:
    if not os.getenv("OPENAI_API_KEY"):
        return ""
    try:
        from openai import OpenAI

        client = OpenAI()
        payload = {"text": text[:3000]}
        for model in LLM_MODELS:
            try:
                result = client.chat.completions.create(
                    model=model,
                    temperature=TEMPERATURE,
                    seed=SEED,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": SAFETY_CLASSIFIER_SYSTEM},
                        {"role": "user", "content": json.dumps(payload, ensure_ascii=True)},
                    ],
                )
                data = json.loads(result.choices[0].message.content or "{}")
                if data.get("is_prompt_injection") is True:
                    return str(data.get("risk_reason", "llm_detected_injection"))[:120]
            except Exception:
                continue
    except Exception:
        return ""
    return ""
