#!/usr/bin/env python3
"""P2-B small, serial HTTP baseline for the Ecommerce RAG Assistant.

Runs **no paid requests** without the explicit --execute flag. Uses only the
Python standard library and does not persist answers, user IDs, or raw logs.

Latency here is CLIENT end-to-end time (including HTTP and network), NOT an
internal stage latency. Correlate with customer_support_trace separately.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import tempfile
import time
import urllib.error
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

CASES = {
    "order": {
        "question": "帮我查一下 ORD202610002 现在到哪里了？",
        "expected_intent": "ORDER_QUERY",
    },
    "knowledge": {
        "question": "七天无理由退货必须保留商品原包装吗？",
        "expected_intent": "KNOWLEDGE_QA",
    },
    "knowledge_insufficient": {
        # Existing frozen evaluation label b2_q60: answerable=false.
        "question": "商家超过48小时没发货的话，平台是不是会自动赔我订单金额的10%？",
        "expected_intent": "KNOWLEDGE_QA",
    },
    "mixed": {
        "question": "ORD202610002 的物流到哪里了？如果收到货想申请七天无理由退货，快递外箱必须保留吗？",
        "expected_intent": None,  # Route must be verified before treating this as a mixed-path sample.
    },
}


def request_json(url: str, *, payload: dict | None, timeout: float) -> tuple[int, dict]:
    body = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {"Accept": "application/json"}
    if body is not None:
        headers["Content-Type"] = "application/json; charset=utf-8"
    req = urllib.request.Request(url, data=body, headers=headers, method="GET" if body is None else "POST")
    # Bypass Windows/VPN proxies: localhost benchmark traffic must stay local.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(req, timeout=timeout) as resp:
        raw = resp.read()
        parsed = json.loads(raw.decode("utf-8"))
        if not isinstance(parsed, dict):
            raise ValueError("API response must be a JSON object")
        return resp.status, parsed


def percentile_nearest_rank(values: list[float], p: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * p) - 1)]


def one_request(base_url: str, name: str, user_id: str, timeout: float, index: int) -> dict:
    spec = CASES[name]
    result = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "scenario": name,
        "iteration": index,
        "client_latency_ms": None,
        "http_status": None,
        "success": False,
        "intent": None,
        "handler": None,
        # Allowed anonymous quality signals: never persist question/answer/IDs.
        "verdict": None,
        "abstained": None,
        "abstain_reason": None,
        "expected_intent": spec["expected_intent"],
        "intent_matches": None,
        "error_type": None,
    }
    started = time.perf_counter()
    try:
        status, data = request_json(
            base_url.rstrip("/") + "/api/v2/chat",
            payload={"question": spec["question"], "user_id": user_id},
            timeout=timeout,
        )
        result["http_status"] = status
        result["success"] = status == 200
        result["intent"] = data.get("intent")
        result["handler"] = data.get("handler")
        component = data.get("result")
        if isinstance(component, dict) and data.get("handler") == "RAG":
            verdict = component.get("verdict")
            result["verdict"] = verdict if verdict in {"SUFFICIENT", "INSUFFICIENT"} else None
            abstained = component.get("abstained")
            result["abstained"] = abstained if isinstance(abstained, bool) else None
            reason = component.get("abstain_reason")
            allowed_reasons = {"evidence_insufficient", "retrieval_unavailable",
                               "retrieval_empty", "verifier_unavailable",
                               "composer_empty", "composer_unavailable"}
            result["abstain_reason"] = reason if reason in allowed_reasons else None
        if spec["expected_intent"] is not None:
            result["intent_matches"] = data.get("intent") == spec["expected_intent"]
    except urllib.error.HTTPError as exc:
        result["http_status"] = exc.code
        result["error_type"] = "HTTPError"  # Never record the potentially sensitive response body.
    except (urllib.error.URLError, TimeoutError, ValueError, OSError, json.JSONDecodeError) as exc:
        result["error_type"] = type(exc).__name__  # Do not persist error messages containing URLs/tokens.
    finally:
        result["client_latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
    return result


def summarize(records: list[dict]) -> None:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        grouped[r["scenario"]].append(r)
    for case_name, rows in grouped.items():
        good = [float(r["client_latency_ms"]) for r in rows if r["success"]]
        error_count = len(rows) - len(good)
        print(f"{case_name}: total={len(rows)} success={len(good)} errors={error_count}")
        if good:
            print(f"  client mean={statistics.mean(good):.1f} ms, min={min(good):.1f} ms, max={max(good):.1f} ms")
            if len(good) >= 20:
                print(f"  client p50={percentile_nearest_rank(good, .50):.1f} ms, p95={percentile_nearest_rank(good, .95):.1f} ms (nearest-rank)")
            else:
                print("  P50/P95 omitted: fewer than 20 successful samples (exploratory data only).")
        for r in rows:
            print(f"  run={r['iteration']} http={r['http_status']} client_ms={r['client_latency_ms']} "
                  f"intent={r['intent']} handler={r['handler']} "
                  f"intent_matches={r['intent_matches']} error={r['error_type']}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case", choices=(*CASES.keys(), "all"), default="order")
    parser.add_argument("--repeats", type=int, default=1, help="Number of sequential requests per scenario")
    parser.add_argument("--timeout", type=float, default=180, help="HTTP request timeout in seconds")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--user-id", default="U001", help="Use ONLY the demo user ID; never production personal data")
    parser.add_argument("--execute", action="store_true", help="Required to send real API requests (may incur LLM costs)")
    parser.add_argument("--output", type=Path, default=Path(tempfile.gettempdir()) / "ecommerce_rag_p2" / "baseline.jsonl")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.repeats < 1 or args.repeats > 100:
        print("--repeats must be in [1, 100]", file=sys.stderr)
        return 2
    if args.timeout <= 0:
        print("--timeout must be positive", file=sys.stderr)
        return 2
    if not (args.base_url.startswith("http://127.0.0.1:") or args.base_url.startswith("http://localhost:")):
        print("Safety check: --base-url must target localhost/127.0.0.1", file=sys.stderr)
        return 2
    if args.user_id != "U001":
        print("Safety check: this pilot script uses only demo user U001", file=sys.stderr)
        return 2
    selected = list(CASES) if args.case == "all" else [args.case]
    print(f"Plan: {len(selected) * args.repeats} serial call(s), cases={', '.join(selected)}, timeout={args.timeout}s/call")
    print("Latency source: HTTP client perf_counter; server stage Trace is NOT inferred from this number.")
    if not args.execute:
        print("DRY RUN ONLY. Add --execute to make real API calls (which may incur DashScope charges).")
        return 0
    try:
        status, health = request_json(args.base_url.rstrip("/") + "/health", payload=None, timeout=min(10.0, args.timeout))
        if status != 200 or health.get("status") != "ok":
            print("Health check did not return {status: ok}; aborting.", file=sys.stderr)
            return 1
    except Exception as exc:
        print(f"Health check failed: {type(exc).__name__}; aborting.", file=sys.stderr)
        return 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    with args.output.open("a", encoding="utf-8", newline="\n") as handle:
        for case in selected:
            for i in range(1, args.repeats + 1):
                result = one_request(args.base_url, case, args.user_id, args.timeout, i)
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                handle.flush()
                records.append(result)
                print(f"Completed {case} {i}/{args.repeats}: HTTP {result['http_status']}, {result['client_latency_ms']} ms")
                if not result["success"]:
                    print("A request failed; stopping to avoid repeated paid failures.")
                    summarize(records)
                    print(f"Local metrics saved to {args.output}")
                    return 1
    summarize(records)
    print(f"Local metrics saved to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
