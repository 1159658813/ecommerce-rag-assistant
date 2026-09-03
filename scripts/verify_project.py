from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run_step(name: str, script: Path) -> bool:
    print()
    print("=" * 78)
    print(f"RUNNING: {name}")
    print("=" * 78)
    print(f"Script: {script.relative_to(ROOT)}")
    print()

    start = time.perf_counter()

    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=ROOT,
    )

    elapsed = time.perf_counter() - start

    print()
    print("-" * 78)

    if result.returncode == 0:
        print(f"[PASS] {name}")
        print(f"       elapsed: {elapsed:.2f}s")
        return True

    print(f"[FAIL] {name}")
    print(f"       exit code: {result.returncode}")
    print(f"       elapsed: {elapsed:.2f}s")
    return False


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Unified verification entry for Ecommerce RAG Assistant."
    )

    parser.add_argument(
        "--mode",
        choices=("quick", "api", "full"),
        default="quick",
        help=(
            "quick: project checks only; "
            "api: project checks + API smoke test; "
            "full: project checks + API smoke test + pipeline smoke test"
        ),
    )

    args = parser.parse_args()

    print()
    print("=" * 78)
    print("Ecommerce RAG Assistant - Unified Verification")
    print("=" * 78)
    print(f"Mode   : {args.mode}")
    print(f"Python : {sys.executable}")
    print(f"Root   : {ROOT}")

    steps = [
        (
            "Project Check",
            ROOT / "scripts" / "project_check.py",
        )
    ]

    if args.mode in {"api", "full"}:
        steps.append(
            (
                "API Smoke Test",
                ROOT / "experiments" / "api_smoke_test.py",
            )
        )

    if args.mode == "full":
        steps.append(
            (
                "Pipeline Smoke Test",
                ROOT / "experiments" / "pipeline_smoke_test.py",
            )
        )

    total_start = time.perf_counter()

    passed = 0

    for name, script in steps:
        if not script.exists():
            print()
            print(f"[FAIL] Missing verification script: {script}")
            return 1

        if not run_step(name, script):
            print()
            print("=" * 78)
            print("VERIFICATION FAILED")
            print("=" * 78)
            return 1

        passed += 1

    total_elapsed = time.perf_counter() - total_start

    print()
    print("=" * 78)
    print("Verification Summary")
    print("=" * 78)
    print(f"Mode       : {args.mode}")
    print(f"Passed     : {passed}/{len(steps)}")
    print(f"Total time : {total_elapsed:.2f}s")
    print("=" * 78)
    print("ALL REQUESTED CHECKS PASSED")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())