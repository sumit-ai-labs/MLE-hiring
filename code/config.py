"""Deterministic configuration for the support triage pipeline."""

from __future__ import annotations

from pathlib import Path


SEED = 42
TEMPERATURE = 0

REPO_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = REPO_ROOT / "code"
DATA_DIR = REPO_ROOT / "data"
TICKETS_DIR = REPO_ROOT / "support_tickets"
DEFAULT_INPUT = TICKETS_DIR / "support_tickets.csv"
DEFAULT_OUTPUT = TICKETS_DIR / "output.csv"
DEFAULT_LOG = CODE_DIR / "triage.log"
TOOL_SPEC_PATH = DATA_DIR / "api_specs" / "internal_tools.json"

OUTPUT_COLUMNS = [
    "issue",
    "subject",
    "company",
    "response",
    "product_area",
    "status",
    "request_type",
    "justification",
    "confidence_score",
    "source_documents",
    "risk_level",
    "pii_detected",
    "language",
    "actions_taken",
]

VALID_STATUS = {"replied", "escalated"}
VALID_REQUEST_TYPES = {"product_issue", "feature_request", "bug", "invalid"}
VALID_RISK_LEVELS = {"low", "medium", "high", "critical"}

BM25_WEIGHT = 0.65
EMBEDDING_WEIGHT = 0.35
TOP_K = 8
RERANK_TOP_K = 5
CROSS_ENCODER_CANDIDATES = 20

LLM_MODELS = ("gpt-4.1-mini", "gpt-4o-mini")
