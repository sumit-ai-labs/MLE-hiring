# Architecture

## System Overview

This is a deterministic, terminal-based support triage pipeline for the MLE hiring evaluation. It processes each CSV row, parses multi-turn support history, applies safety and PII controls, retrieves grounded documentation, validates internal tool actions, and writes the exact required output schema.

The design intentionally avoids chatbot-style behavior, autonomous agents, ReAct loops, recursive planning, and LLM-controlled tool execution. Rules own safety, business constraints, preconditions, and evaluator enum compatibility.

## Pipeline Architecture

```text
CSV row
-> conversation parser
-> safety layer
-> PII detection
-> language detection
-> company classifier
-> product classifier
-> retrieval router
-> hybrid retriever
-> reranker
-> decision engine
-> tool planner
-> validator
-> grounded responder
-> CSV writer
```

Each stage is explicit and observable through deterministic logs. The pipeline keeps richer internal labels for decisions, then maps them to evaluator-safe final enums:

- `status`: `replied` or `escalated`
- `request_type`: `product_issue`, `feature_request`, `bug`, or `invalid`

## Retrieval Strategy

Markdown documentation is chunked by headings. Chunks preserve:

- `path`
- `company`
- `product`
- `category`
- `heading`

The retrieval router selects likely company/product corpora before retrieval. It falls back to global retrieval only for low-confidence routing. Hybrid scoring is deterministic:

```text
0.65 * bm25_score + 0.35 * embedding_score
```

Semantic retrieval uses local `sentence-transformers/all-MiniLM-L6-v2` when available, with deterministic hashing fallback if not. The top merged candidates are reranked with local `cross-encoder/ms-marco-MiniLM-L-6-v2` when available; otherwise a deterministic heuristic reranker is used. The final source list is validated against real retrieved markdown paths.

## Safety & Adversarial Defense

Safety runs before retrieval. It detects prompt injection, jailbreak attempts, role override, system/developer prompt extraction, fake tool execution requests, output-format manipulation, and multilingual bypass attempts.

Detected attacks raise risk without preventing legitimate issue handling. Malicious instructions are ignored and never passed as executable instructions. Support docs are also untrusted: the system extracts factual evidence only and never follows instructions found in retrieved documents.

PII detection is regex-only and deterministic. It detects and masks email, phone, payment card, SSN, DOB, address, account ID, API key, and passport-like values. Responses must not repeat sensitive values.

## Tool Orchestration

Tool orchestration is deterministic and validator-gated:

```text
decision intent -> proposed actions -> schema/precondition validation -> execution record
```

The LLM never executes tools. The validator enforces required fields, refund limits, identity verification, transaction requirements, authorization constraints, and action ordering.

Important rules:

- Refund: `verify_identity` before `issue_refund`.
- Refund over `$500`: escalate.
- Refund older than `90` days: escalate.
- Account compromise or fraud: `lock_account`, never simple `reset_password`.
- Legal threat: `escalate_to_human`.
- Unsupported risky request: escalate safely.

## Determinism Strategy

Default execution is deterministic:

- `temperature=0`
- `seed=42`
- no random sampling
- no retry loops
- no autonomous agent loops
- deterministic rule ordering
- deterministic CSV schema and column order
- deterministic confidence formula

If optional model dependencies or API keys are missing, the system falls back to deterministic local behavior rather than failing.

## Confidence Calibration

Confidence is a weighted fusion of explicit signals:

```text
0.35 * retrieval_quality
+ 0.20 * classifier_confidence
+ 0.20 * tool_path_validity
+ 0.15 * safety_consistency
+ 0.10 * response_grounding_score
```

Retrieval quality considers top score, score gap, and corpus specificity. Confidence is lowered by ambiguity, missing docs, cross-company conflict, prompt injection, unsupported actions, and invalid tool paths.

## Gemini Optional Enhancement

Gemini is optional and disabled by default. It is wired only as a response polish layer after all decisions, tool actions, source attribution, confidence, status, and request type are finalized.

Gemini cannot change meaning, actions, escalation, status, request type, confidence, or policy claims. A validation layer rejects unsafe rewrites, and accepted outputs are cached by SHA256 over response, actions, sources, status, and request type. Missing keys, timeouts, quota errors, malformed JSON, or validation failure return the original response unchanged.

## Tradeoffs

The system is conservative by design. It favors safe escalation over risky automation and grounded brief responses over speculative explanations. This improves hidden-test robustness but can make some low-risk answers less expansive.

Retrieval quality benefits from local MiniLM/cross-encoder models when available. Offline fallback is deterministic and stable but less semantically rich.

## Limitations

- Conservative escalation may over-escalate ambiguous but harmless tickets.
- Multilingual support emphasizes attack detection and safe handling rather than native-level response fluency.
- Gemini polish is optional and disabled by default to preserve default reproducibility.
- The system does not perform fully semantic open-ended reasoning; it relies on rules, retrieval, and constrained helpers.

## Hidden-Test Assumptions

The design assumes hidden tests may include multilingual prompt injection, fake tool requests, misleading company fields, retrieval poisoning inside docs, PII leakage traps, contradictory histories, refund precondition violations, legal threats, account compromise mixed with billing, and missing documentation.

Security, fraud, and legal signals dominate weaker conflicting requests. Unsupported or ambiguous high-risk paths escalate safely.

## Failure Modes

- If relevant documentation is missing, the system may escalate or produce a low-confidence grounded response rather than inventing policy.
- If local semantic models are unavailable, retrieval uses deterministic fallback scoring and may retrieve less precise evidence.
- If a ticket is extremely ambiguous across companies, source precision can drop and confidence should be lower.
- If optional Gemini is enabled but unavailable or invalid, response polish silently falls back to the deterministic response.

## SELF-ASSESSMENT (MANDATORY)

### Strengths

- Deterministic default execution with fixed configuration and no autonomous loops.
- Strong adversarial robustness through rule-first safety before retrieval.
- Safe tool execution with explicit schema and precondition validation.
- Strong retrieval design using routed hybrid retrieval, metadata preservation, reranking, and source validation.
- Reproducible terminal workflow with tests, visible evaluation, validator checks, and deterministic output reproduction.

### Weaknesses

- Conservative escalation can reduce automation on ambiguous tickets.
- Multilingual nuance is limited; handling is strongest for safety detection and basic response routing.
- Optional Gemini polish is disabled by default for reproducibility.
- No fully semantic open-ended reasoning; the system intentionally avoids LLM-owned decisions.

### Future Improvements

- Stronger multilingual support for both classification and localized response quality.
- Richer confidence calibration using more labeled validation data.
- Expanded hidden-test coverage for additional companies, products, and adversarial document patterns.
