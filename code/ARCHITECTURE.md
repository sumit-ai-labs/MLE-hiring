# Architecture

## Data Flow

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

The system is deterministic by default. It uses fixed configuration values (`seed=42`, `temperature=0`) and does not use autonomous loops, ReAct, recursive agents, or multi-agent orchestration.

## Components

- `utils/parser.py`: parses JSON conversation history and extracts multi-turn state such as prior verification, follow-ups, unresolved failures, refund amounts, transaction IDs, and account identifiers.
- `agent/safety.py`: detects prompt injection, system prompt extraction, roleplay overrides, fake tool requests, multilingual jailbreaks, and document poisoning patterns before retrieval.
- `agent/conflict_resolver.py`: resolves contradictory multi-turn intent before decisioning so security, legal, and fraud signals dominate weaker later requests.
- `utils/pii.py`: regex-only PII detection and masking for email, phone, card, SSN, DOB, address, API key, passport, and account IDs.
- `agent/classifier.py`: deterministic company, product, and request routing. Optional LLM tie-breaks are JSON-only and cannot override safety constraints.
- `retrieval/*`: markdown heading chunking, metadata preservation, routed hybrid retrieval, and deterministic reranking.
- `agent/decision_engine.py`: rule-first escalation and reply decisions.
- `tools/*`: schema loading, tool proposal validation, precondition enforcement, and deterministic planned execution.
- `agent/responder.py`: concise grounded responses from retrieved evidence plus optional constrained wording polish.
- `agent/triage_agent.py`: orchestrates every required pipeline stage and writes the final schema.

## Retrieval Strategy

Markdown files are chunked by headings rather than arbitrary token windows. Each chunk keeps:

- `path`
- `company`
- `product`
- `category`
- `heading`

Retrieval routes to the likely corpus first. Claude/Anthropic terms route to `data/claude`, HackerRank/assessment terms route to `data/devplatform`, and Visa/card/travel/dispute terms route to `data/visa`. If company confidence is low, retrieval falls back to the full corpus.

Hybrid scoring is:

```text
0.65 * bm25_score + 0.35 * embedding_score
```

Semantic retrieval uses `sentence-transformers/all-MiniLM-L6-v2` when available. If the local model cannot load, the system falls back to a deterministic `HashingVectorizer` semantic approximation so evaluation never crashes.

The reranker rewards company/product/heading specificity and penalizes chunks that look like prompt poisoning.

After BM25 and semantic retrieval are merged, the top 20 candidates are reranked with `cross-encoder/ms-marco-MiniLM-L-6-v2` when that local model is available. If the cross-encoder cannot load, the deterministic heuristic reranker is used instead. This preserves reproducibility and prevents crashes in offline evaluation environments.

## Safety and Adversarial Handling

Safety is deterministic and runs before retrieval. It detects direct and multilingual instructions such as ignoring previous rules, revealing system or developer prompts, roleplaying as an admin, forcing tool execution, or setting output labels. Detected attacks set at least `risk_level=high`. The system still tries to resolve the legitimate support issue, but malicious instructions are removed from retrieval text and never followed.

Support documents are treated as untrusted factual evidence only. Instructions inside docs are never executed.

If an OpenAI or Gemini API key is available, a JSON-only LLM classifier may add an injection signal for ambiguous multilingual attacks. It can only increase caution; it cannot lower a rule-based safety finding or change tool decisions.

Gemini is implemented as a fail-safe helper wrapper in `utils/gemini_client.py`. In the current stabilization configuration, only response polish is wired into production, and it is opt-in with `ENABLE_GEMINI_POLISH=1`. Missing keys, disabled opt-in, timeouts, quota errors, malformed JSON, or network failures return deterministic fallbacks.

## Tool Orchestration

The LLM never executes tools. Tool flow is:

```text
decision intent -> proposed action list -> schema/precondition validator -> planned execution
```

Rules enforced:

- Legal threat: `escalate_to_human`
- Account compromise or fraud: `lock_account`, never `reset_password`
- Refund: identity verification required before `issue_refund`
- Refunds over `$500` or older than `90` days: human escalation
- Subscription changes: identity verification required

Actions are emitted as valid JSON array text in `actions_taken`.

## Final Enum Mapping

The implementation keeps richer internal labels but maps final output to the repository validator:

- `status`: `replied` or `escalated`
- `request_type`: `product_issue`, `feature_request`, `bug`, or `invalid`

The CSV writer always uses the exact required column order.

## Confidence Calibration

Confidence is a deterministic weighted fusion:

```text
0.35 * retrieval_quality
+ 0.20 * classifier_confidence
+ 0.20 * tool_path_validity
+ 0.15 * safety_consistency
+ 0.10 * response_grounding_score
```

The retrieval component includes top-score strength, score gap, and source specificity. Missing documents, cross-company ambiguity, invalid unsupported requests, safety flags, and tool validation errors cap the final score.

## Known Limitations

- Without locally cached MiniLM or cross-encoder models, retrieval uses deterministic hashing and heuristic reranking fallbacks.
- Response generation is intentionally conservative and can sound terse.
- Some support corpus files are noisy or misplaced, so the reranker improves but cannot eliminate all irrelevant evidence.
- Non-English responses use safe concise templates plus English evidence snippets when no LLM is available.

## Self-Assessment

- Adversarial robustness: 8/10
- Escalation precision: 7/10
- Response quality: 7/10
- Source attribution: 8/10
- Tool calling: 8/10
- PII handling: 8/10
- Code architecture: 8/10
- Confidence calibration: 7/10
- Determinism: 9/10

Hard visible-ticket categories:

1. Unauthorized workspace access restoration: handled by explicit permission rules rather than granting access.
2. Visa merchant refund/ban requests: handled as dispute guidance, not direct refund tooling.
3. Account compromise mixed with billing: security rules take priority over refund/password reset paths.

Likely hidden adversarial categories:

- Multilingual prompt injection mixed with legitimate billing requests.
- Fake tool requests that ask to bypass identity verification.
- Cross-company tickets where the `company` CSV field is misleading.
- Corpus poisoning that tries to instruct the agent.

Known failure mode:

The fallback response extractor can choose a weak sentence from a relevant document. The decision should remain safe, but the answer may be less helpful than an LLM-polished response.
