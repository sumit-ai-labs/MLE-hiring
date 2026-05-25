# Deterministic Support Triage System

## Challenge Overview

This repository contains a terminal-based support triage system for the MLE hiring evaluation. It reads support tickets from CSV, retrieves grounded evidence from markdown documentation, applies deterministic safety and business rules, plans validated internal tool actions, and writes `support_tickets/output.csv` in the exact evaluator schema.

The system is not a chatbot. It is a reproducible support triage pipeline optimized for correctness, adversarial robustness, explainability, grounded retrieval, and validator compatibility.

## Problem Framing

Each ticket may include multi-turn conversation history, adversarial prompt injection, PII, ambiguous company/product references, unsupported tool requests, or conflicting user intent. The pipeline treats all user text and retrieved documents as untrusted input. Rules own safety, PII handling, tool preconditions, escalation, and output enum mapping. Optional LLM helpers can assist only in constrained wording or ambiguity tasks and must fail closed.

## Architecture Summary

The implementation lives under `code/` and preserves a simple deterministic architecture:

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

There are no autonomous loops, ReAct agents, recursive planners, or multi-agent orchestration.

## Pipeline

- `code/utils/parser.py` parses conversation history and extracts state such as verified identity, previous failures, refund amounts, transaction IDs, and security signals.
- `code/agent/safety.py` detects prompt injection, jailbreaks, override attempts, hidden prompt extraction, fake tool requests, and multilingual attack patterns before retrieval.
- `code/utils/pii.py` performs regex-only PII detection and masking.
- `code/agent/classifier.py` routes company, product, and issue type using deterministic keyword and confidence logic.
- `code/retrieval/` chunks markdown by headings, performs routed hybrid retrieval, and reranks results.
- `code/agent/conflict_resolver.py` makes security, fraud, and legal signals dominate weaker conflicting intents.
- `code/agent/decision_engine.py` applies rule-first business logic.
- `code/agent/planner.py` proposes deterministic tool actions.
- `code/tools/validator.py` enforces schemas and preconditions before execution.
- `code/agent/responder.py` generates brief grounded responses and never exposes PII or internal instructions.

## Retrieval Strategy

Documents are chunked by markdown headings, not arbitrary token windows. Each chunk preserves `path`, `company`, `product`, `category`, and `heading` metadata.

Retrieval routes to likely company/product corpora first. It falls back to global retrieval only when routing confidence is low. Hybrid scoring uses:

```text
0.65 * bm25_score + 0.35 * embedding_score
```

Semantic retrieval uses local `sentence-transformers/all-MiniLM-L6-v2` when available. Reranking uses local `cross-encoder/ms-marco-MiniLM-L-6-v2` when available, with deterministic heuristic fallback if models are unavailable. Source attribution is validated so output paths refer only to real retrieved markdown files.

## Safety System

Safety runs before retrieval and cannot be overridden by LLM output. It detects direct and multilingual prompt injection, role override, system/developer prompt extraction, jailbreak language, fake tool requests, and attempts to bypass identity verification or validators.

Support documents are treated as untrusted factual evidence. Instructions inside documents are never executed.

PII detection is deterministic regex-based and covers emails, phone numbers, payment cards, SSNs, DOBs, addresses, account IDs, API keys, and passport-like identifiers. Responses are masked and must not repeat sensitive values.

## Tool Orchestration

LLMs never execute tools. The flow is:

```text
decision intent -> proposed action list -> schema/precondition validator -> deterministic execution record
```

Important enforced rules include:

- Refunds require identity verification before `issue_refund`.
- Refunds over `$500` or older than `90` days escalate.
- Account compromise and fraud use `lock_account`, never simple password reset.
- Legal threats escalate to a human.
- Unsupported risky requests escalate safely.

`actions_taken` is always valid JSON array text.

## Confidence Scoring

Confidence is deterministic weighted signal fusion:

```text
0.35 * retrieval_quality
+ 0.20 * classifier_confidence
+ 0.20 * tool_path_validity
+ 0.15 * safety_consistency
+ 0.10 * response_grounding_score
```

Ambiguity, missing documents, weak retrieval, cross-company conflict, safety flags, and tool validation failures lower or cap confidence.

## Deterministic Guarantees

The default system is deterministic:

- `temperature=0`
- `seed=42`
- no retry-until-good loops
- no stochastic routing
- no autonomous agents
- deterministic output enum mapping
- deterministic CSV column order

Optional external model calls are disabled by default or guarded by fail-safe fallbacks.

## Gemini Optional Mode

Gemini response polish is optional and disabled by default. To enable it, set:

```text
GEMINI_API_KEY=<secret>
ENABLE_GEMINI_POLISH=1
```

Gemini is used only as a post-processing polish wrapper after decisions are finalized. It cannot change status, request type, tool actions, escalation, confidence, source attribution, or policy meaning. Outputs are validated and cached by a SHA256 key over response, actions, sources, status, and request type. If Gemini is unavailable, times out, returns malformed JSON, or fails validation, the original deterministic response is returned unchanged.

## Runtime Characteristics

The system runs locally from the terminal. Optional local ML models improve retrieval/reranking when installed. If they are unavailable, deterministic fallbacks keep the pipeline runnable for evaluation.

## Setup Instructions

```bash
python -m pip install -r requirements.txt
```

The implementation dependencies are also listed in `code/requirements.txt`.

## Run Instructions

```bash
python code/main.py
```

Explicit form:

```bash
python code/main.py --input support_tickets/support_tickets.csv --data data --output support_tickets/output.csv
```

## Reproducibility

To validate format:

```bash
set PYTHONIOENCODING=utf-8
python code/validate_output.py
```

To run tests:

```bash
python -m unittest discover -s code/tests -p "test_*.py"
python -m unittest discover -s tests/adversarial -p "test_*.py"
```

To run the visible evaluation harness:

```bash
python tests/evaluate_visible_set.py
```

## Evaluation Metrics

Final verified metrics before packaging:

- code tests: PASS, 31 passed
- adversarial tests: PASS, 16 passed
- validator: PASS
- visible evaluation: PASS
- determinism: PASS
- output reproduction: PASS
- tool correctness: 100%
- source attribution: 100%
- retrieval quality: 100%
- PII leakage check: 100%

## Tradeoffs

The system prefers conservative escalation over unsafe automation. This may reduce reply assertiveness for ambiguous tickets but improves hidden-test safety. Retrieval uses local semantic models when available but remains functional with deterministic fallbacks. Optional Gemini polish improves tone but is disabled by default for maximum reproducibility.

## Limitations

- Conservative escalation can over-escalate some harmless ambiguous requests.
- Multilingual handling is safety-oriented rather than fully fluent.
- Optional Gemini polish is disabled by default.
- The system does not perform fully semantic open-ended reasoning; it intentionally relies on rules, retrieval, and constrained helpers.
