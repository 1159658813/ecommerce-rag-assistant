"""Minimal host-side preflight for P2-A Docker demo; never prints secrets."""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    errors: list[str] = []
    warnings: list[str] = []

    for name in ("Dockerfile", "compose.yaml", "requirements.txt", "frontend/index.html"):
        if not (ROOT / name).is_file():
            errors.append(f"Missing: {name}")

    env_path = ROOT / ".env"
    if not env_path.is_file():
        errors.append("Missing .env (copy .env.example and configure DASHSCOPE_API_KEY)")
    else:
        env_lines = env_path.read_text(encoding="utf-8-sig").splitlines()
        api_keys = [
            line.partition("=")[2].strip().strip('"\'')
            for line in env_lines
            if line.strip() and not line.lstrip().startswith("#")
            and line.partition("=")[0].strip() == "DASHSCOPE_API_KEY"
        ]
        if not api_keys or not api_keys[-1] or api_keys[-1] == "your_dashscope_api_key_here":
            errors.append("DASHSCOPE_API_KEY not configured in .env")

    data = ROOT / "data"
    if not (data / "raw").is_dir():
        errors.append("Missing data/raw (knowledge source files)")

    for relative in ("index/knowledge.faiss", "index/chunks.json"):
        target = data / relative
        if not target.is_file() or target.stat().st_size == 0:
            errors.append(
                f"Missing/empty data/{relative}; build index before starting "
                "(see docs/docker_deployment.md)"
            )

    if shutil.which("docker") is None:
        warnings.append("Docker CLI not on PATH; install/start Docker Desktop")
    else:
        result = subprocess.run(
            ["docker", "compose", "version"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            warnings.append("docker compose version failed; check Docker Desktop")

    for message in warnings:
        print(f"[WARN] {message}")
    for message in errors:
        print(f"[FAIL] {message}")

    if errors:
        print(f"P2 Docker preflight: {len(errors)} issue(s).")
        return 1
    print("P2 Docker preflight: PASS (runtime behavior not yet verified).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
