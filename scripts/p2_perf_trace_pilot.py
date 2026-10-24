#!/usr/bin/env python3
"""P2-B isolated, serial HTTP + Docker Trace pilot.

Place next to p2_perf_baseline.py and p2_trace_summary.py in scripts/.
Requires Python 3.11+, Docker Desktop/Compose running, and no concurrent
traffic (including other browser chat requests) during the experiment.

Default is a dry run: only --execute sends an API request (potentially paid).
Does NOT collect/store questions, answers, IDs, API keys, or raw Docker logs.
A unique trace in a request-time window is *not* a cryptographic/request-ID
match; missing or multiple traces are reported as ambiguous, never guessed.
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from p2_perf_baseline import CASES, one_request, request_json
from p2_trace_summary import parse_trace

TRACE_FIELDS = (
    "latency_total", "latency_router", "latency_retrieval",
    "latency_verifier", "latency_tool", "latency_composer",
    "verifier_retry_count",
)


def parse_window(log_text: str) -> list[dict]:
    """Return only numeric metrics from one-line trace entries."""
    results = []
    for line in log_text.splitlines():
        tr = parse_trace(line)
        if tr is not None:
            results.append({k: tr.get(k) for k in TRACE_FIELDS})
    return results


def read_trace_window(since_utc: datetime, service: str) -> tuple[list[dict], str | None]:
    # Docker timestamps and the host clock must be in sync for the isolation
    # window to be meaningful. RFC3339/UTC avoids timezone interpretation.
    since = since_utc.isoformat(timespec="microseconds").replace("+00:00", "Z")
    args = ["docker", "compose", "logs", "--since", since, "--no-log-prefix", service]
    try:
        p = subprocess.run(args, capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=30, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return [], type(exc).__name__
    if p.returncode != 0:
        return [], "docker_logs_nonzero"
    return parse_window(p.stdout), None


def self_test() -> int:
    sample = ('2026-10-21 INFO customer_support_trace '
              'latency_total=7875.686 latency_router=3788.912 '
              'latency_retrieval=0.0 latency_verifier=0.0 '
              'latency_tool=7.734 latency_composer=1450.176 '
              'verifier_retry_count=0')
    got = parse_window("ordinary log\n" + sample + "\nordinary log\n")
    assert len(got) == 1 and got[0]["latency_total"] == 7875.686
    assert got[0]["latency_tool"] == 7.734
    assert parse_window("ordinary log\n") == []
    assert len(parse_window(sample + "\n" + sample)) == 2
    print("SELF-TEST PASS (no HTTP/Docker calls)")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case", choices=(*CASES.keys(), "all"), default="order")
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--timeout", type=float, default=180)
    p.add_argument("--base-url", default="http://127.0.0.1:8000")
    p.add_argument("--user-id", default="U001")
    p.add_argument("--service", default="rag-app")
    p.add_argument("--execute", action="store_true")
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--output", type=Path, default=Path(tempfile.gettempdir()) /
                   "ecommerce_rag_p2" / "paired_pilot.jsonl")
    args = p.parse_args()
    if args.self_test:
        return self_test()
    if not 1 <= args.repeats <= 20 or args.timeout <= 0:
        p.error("--repeats must be 1..20 and --timeout > 0")
    if args.user_id != "U001":
        p.error("Pilot is restricted to demo user U001")
    if not (args.base_url.startswith("http://127.0.0.1:") or
            args.base_url.startswith("http://localhost:")):
        p.error("Only localhost HTTP is allowed")
    if args.service != "rag-app":
        p.error("Only the local rag-app service is supported")
    cases = list(CASES) if args.case == "all" else [args.case]
    print(f"Plan: {len(cases) * args.repeats} serial calls; cases={','.join(cases)}")
    print("Trace association: one trace within isolated time window; NOT request-ID verified.")
    print("Do not use the chat UI or issue other API calls during sampling.")
    if not args.execute:
        print("DRY RUN ONLY. Add --execute for real (potentially paid) requests.")
        return 0
    try:
        status, health = request_json(args.base_url.rstrip("/") + "/health",
                                      payload=None, timeout=min(args.timeout, 10))
        if status != 200 or health.get("status") != "ok":
            print("Health is not OK. No paid requests sent.", file=sys.stderr)
            return 1
    except Exception as exc:
        print(f"Health request failed ({type(exc).__name__}). No paid requests sent.", file=sys.stderr)
        return 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out = []
    with args.output.open("a", encoding="utf-8", newline="\n") as f:
        for case in cases:
            for i in range(1, args.repeats + 1):
                start = datetime.now(timezone.utc)
                row = one_request(args.base_url, case, args.user_id, args.timeout, i)
                matches, log_error = read_trace_window(start, args.service)
                row["trace_match_status"] = (
                    "docker_logs_error" if log_error else
                    "unique_in_isolated_window_not_id_verified" if len(matches) == 1 else
                    "missing" if not matches else "ambiguous_multiple"
                )
                row["trace_count_in_window"] = len(matches)
                row["server_trace"] = matches[0] if len(matches) == 1 else None
                row["trace_error_type"] = log_error
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
                out.append(row)
                print(f"{case} {i}/{args.repeats}: HTTP={row['http_status']}, "
                      f"client_ms={row['client_latency_ms']}, "
                      f"trace={row['trace_match_status']}")
                if len(matches) == 1:
                    print(f"  server_total_ms={matches[0]['latency_total']}; "
                          f"router_ms={matches[0]['latency_router']}; "
                          f"verifier_ms={matches[0]['latency_verifier']}")
                if not row["success"] or len(matches) != 1 or log_error:
                    print("STOP: HTTP failure or unmatched/ambiguous Trace; no further calls.")
                    print(f"Saved: {args.output}")
                    return 1
    print(f"Saved {len(out)} isolated HTTP+Trace pairs: {args.output}")
    if len(out) < 20:
        print("Exploratory samples only: do NOT report P50/P95 as robust estimates.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
