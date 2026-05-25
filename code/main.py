#!/usr/bin/env python3
"""Entry point for the deterministic support triage system."""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

from agent.triage_agent import TriageAgent
from config import DATA_DIR, DEFAULT_INPUT, DEFAULT_LOG, DEFAULT_OUTPUT, OUTPUT_COLUMNS, SEED


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run support triage on a ticket CSV.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--data", type=Path, default=DATA_DIR)
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--no-log", action="store_true")
    return parser.parse_args()


def main() -> int:
    random.seed(SEED)
    args = parse_args()
    if not args.input.exists():
        print(f"Input CSV not found: {args.input}", file=sys.stderr)
        return 2
    agent = TriageAgent(data_dir=args.data, log_path=None if args.no_log else args.log)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.input.open("r", encoding="utf-8-sig", newline="") as infile, args.output.open("w", encoding="utf-8", newline="") as outfile:
        reader = csv.DictReader(infile)
        writer = csv.DictWriter(outfile, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for idx, row in enumerate(reader, start=1):
            writer.writerow(agent.process_row(row, idx))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
