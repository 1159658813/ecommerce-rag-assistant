#!/usr/bin/env python3
"""Read existing ecommerce-rag-assistant customer_support_trace logs.

No network requests and no LLM/API calls. Reads stdin (e.g. a Docker Compose
logs pipeline) or --input. Outputs only selected numeric trace metrics; it does
not print/store the original logs or identifying fields.

Trace values are server-side stage timing (milliseconds), not client HTTP latency.
The arithmetic gap between total and the displayed stage sum is not a measured
independent stage and might include overlap, work between spans, or missing spans.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import deque
from pathlib import Path

EVENT = "customer_support_trace"
FIELDS = (
    "latency_total",
    "latency_router",
    "latency_retrieval",
    "latency_verifier",
    "latency_tool",
    "latency_composer",
    "verifier_retry_count",
)
NUMBER = r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
PATTERNS = {
    k: re.compile(r"(?<![A-Za-z_])" + re.escape(k) + r"\s*[\"']?\s*[:=]\s*[\"']?\s*" + NUMBER)
    for k in FIELDS
}


def parse_trace(line: str) -> dict[str, float | int | None] | None:
    if EVENT not in line:
        return None
    values: dict[str, float | int | None] = {}
    for field, pattern in PATTERNS.items():
        match = pattern.search(line)
        try:
            values[field] = (int(float(match.group(1))) if field == "verifier_retry_count" else float(match.group(1))) if match else None
        except ValueError:
            values[field] = None
    if values["latency_total"] is None:
        return None
    return values


def self_test() -> None:
    examples = [
        'INFO customer_support_trace latency_total=7875.686 latency_router=3788.912 latency_retrieval=0.0 latency_verifier=0.0 latency_tool=7.734 latency_composer=1450.176 verifier_retry_count=0',
        'INFO customer_support_trace | {"latency_total": 8001.1, "latency_router": 3000.5, "verifier_retry_count": 2}',
    ]
    a, b = [parse_trace(x) for x in examples]
    assert a and a["latency_total"] == 7875.686 and a["latency_tool"] == 7.734
    assert b and b["latency_router"] == 3000.5 and b["verifier_retry_count"] == 2
    assert parse_trace("INFO request_completed status=200") is None
    print("SELF-TEST PASS (no HTTP calls)")


def fmt(value: float | int | None) -> str:
    return "N/A" if value is None else f"{value:.3f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="Read a previously saved log file instead of stdin")
    parser.add_argument("--last", type=int, default=10, help="Only inspect the last N valid traces (default: 10)")
    parser.add_argument("--output", type=Path, help="Optional JSONL output of numeric fields only; no raw logs")
    parser.add_argument("--self-test", action="store_true", help="Verify parser using synthetic data and exit")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    if args.last < 1 or args.last > 10000:
        parser.error("--last must be between 1 and 10000")

    selected = deque(maxlen=args.last)
    seen = 0
    if args.input:
        lines = args.input.open("r", encoding="utf-8-sig", errors="replace")
    else:
        # Native docker command output arrives as bytes on PowerShell and is
        # decoded with replacement so unexpected locale data cannot crash parsing.
        lines = (line.decode("utf-8", errors="replace") for line in sys.stdin.buffer)
    try:
        for line in lines:
            if EVENT in line:
                seen += 1
                result = parse_trace(line)
                if result is not None:
                    selected.append(result)
    finally:
        if args.input:
            lines.close()

    rows = list(selected)
    print(f"Trace event lines={seen}; valid traces selected={len(rows)} (last {args.last}); units=ms")
    if not rows:
        print("No valid traces found. Check whether the log contains one-line customer_support_trace entries.")
        return 1

    for i, row in enumerate(rows, 1):
        print(f"trace {i}:")
        for field in FIELDS:
            print(f"  {field}={fmt(row[field])}")
        stages = [row[k] for k in FIELDS[1:6]]
        if all(x is not None for x in stages):
            gap = float(row["latency_total"]) - sum(float(x) for x in stages)
            print(f"  arithmetic_gap_not_attributed={gap:.3f}")

    if len(rows) >= 2:
        print("Averages across selected traces (not P50/P95):")
        for field in FIELDS[:-1]:
            nums = [float(row[field]) for row in rows if row[field] is not None]
            print(f"  {field}: n={len(nums)} mean={statistics.mean(nums):.3f} ms" if nums else f"  {field}: N/A")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8", newline="\n") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"Numeric-only JSONL saved to {args.output}")
    print("Note: logged server stages may overlap; arithmetic gap is not proof of an independent bottleneck.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
