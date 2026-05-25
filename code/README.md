# Support Triage System

Deterministic terminal pipeline for the MLE Hiring Challenge.

## Setup

```bash
python -m pip install -r requirements.txt
```

If the evaluator installs dependencies from the implementation folder instead:

```bash
python -m pip install -r code\requirements.txt
```

If `python` is unavailable on Windows, use the Python executable installed on the machine. The system also runs in deterministic fallback mode if optional LLM or embedding dependencies are unavailable.

## Run

```bash
python code\main.py
```

Defaults:

- input: `support_tickets/support_tickets.csv`
- output: `support_tickets/output.csv`
- corpus: `data/`
- log: `code/triage.log`

Equivalent explicit command:

```bash
python code\main.py --input support_tickets\support_tickets.csv --data data --output support_tickets\output.csv
```

## Validate

```bash
set PYTHONIOENCODING=utf-8
python code\validate_output.py
```

## Optional LLM Use

If `OPENAI_API_KEY` is set, the pipeline may call `gpt-4.1-mini` or `gpt-4o-mini` with `temperature=0`, `seed=42`, and JSON response format for ambiguous classification, prompt-injection classification, or wording polish. If `GEMINI_API_KEY` is set, Gemini may be used for the same constrained helper roles: multilingual injection classification, low-confidence ambiguity assistance, and response polishing.

Safety, PII detection, tool validation, escalation rules, and final enum mapping remain deterministic and cannot be overridden by any LLM. Gemini response polish is opt-in with `ENABLE_GEMINI_POLISH=1`; leave it unset for deterministic default behavior. If an API key is missing, a request times out, quota is exceeded, or JSON is malformed, the system silently falls back to deterministic behavior.

If no API key is present, the system runs fully offline.

The retriever uses `sentence-transformers/all-MiniLM-L6-v2` and reranks with `cross-encoder/ms-marco-MiniLM-L-6-v2` when those local models are available. If not, deterministic local fallbacks are used.

## Tests

```bash
python -m unittest discover code\tests
```
