from __future__ import annotations

import compileall
import os
import sys
from pathlib import Path

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / ".env")


passed = 0
warnings = 0
failed = 0


def pass_check(message: str) -> None:
    global passed
    passed += 1
    print(f"[PASS] {message}")


def warn_check(message: str) -> None:
    global warnings
    warnings += 1
    print(f"[WARN] {message}")


def fail_check(message: str) -> None:
    global failed
    failed += 1
    print(f"[FAIL] {message}")


print("=" * 72)
print("Ecommerce RAG Assistant - Project Check")
print("=" * 72)


# ---------------------------------------------------------------------
# 1. Python version
# ---------------------------------------------------------------------

print("\n[1/6] Python environment")

if sys.version_info >= (3, 11):
    pass_check(
        f"Python version: "
        f"{sys.version_info.major}."
        f"{sys.version_info.minor}."
        f"{sys.version_info.micro}"
    )
else:
    fail_check(
        f"Python >= 3.11 required, current: "
        f"{sys.version_info.major}.{sys.version_info.minor}"
    )


# ---------------------------------------------------------------------
# 2. Required project files
# ---------------------------------------------------------------------

print("\n[2/6] Project files")

required_files = [
    ROOT / "README.md",
    ROOT / "requirements.txt",
    ROOT / ".env.example",
    ROOT / "app.py",
    ROOT / "src" / "api" / "app.py",
    ROOT / "src" / "config" / "settings.py",
    ROOT / "src" / "service" / "factory.py",
]

for path in required_files:
    if path.exists():
        pass_check(str(path.relative_to(ROOT)))
    else:
        fail_check(f"Missing: {path.relative_to(ROOT)}")


# ---------------------------------------------------------------------
# 3. Environment variables
# ---------------------------------------------------------------------

print("\n[3/6] Environment configuration")

if os.getenv("DASHSCOPE_API_KEY"):
    pass_check("DASHSCOPE_API_KEY is configured")
else:
    fail_check("DASHSCOPE_API_KEY is missing")

if os.getenv("DASHSCOPE_BASE_URL"):
    pass_check("DASHSCOPE_BASE_URL is configured")
else:
    warn_check(
        "DASHSCOPE_BASE_URL is not explicitly configured"
    )

if os.getenv("HF_TOKEN"):
    pass_check("HF_TOKEN is configured")
else:
    warn_check(
        "HF_TOKEN is not configured "
        "(optional, Hugging Face downloads may have lower rate limits)"
    )


# ---------------------------------------------------------------------
# 4. Central configuration and RAG assets
# ---------------------------------------------------------------------

print("\n[4/6] RAG configuration and assets")

try:
    from src.config import settings

    pass_check("src.config.settings import")

    print(f"       candidate_k   = {settings.candidate_k}")
    print(f"       evidence_k    = {settings.evidence_k}")
    print(f"       verifier      = {settings.verifier_model}")
    print(f"       log_level     = {settings.log_level}")

    if settings.index_path.exists():
        pass_check(f"FAISS index: {settings.index_path}")
    else:
        fail_check(f"FAISS index missing: {settings.index_path}")

    if settings.metadata_path.exists():
        pass_check(f"Metadata: {settings.metadata_path}")
    else:
        fail_check(f"Metadata missing: {settings.metadata_path}")

except Exception as exc:
    fail_check(f"Configuration import failed: {exc}")


# ---------------------------------------------------------------------
# 5. Public module imports
# ---------------------------------------------------------------------

print("\n[5/6] Public API imports")

try:
    from src.service import (
        RAGService,
        build_pipeline,
        build_rag_service,
    )

    pass_check(
        "src.service public API "
        "(RAGService, build_pipeline, build_rag_service)"
    )

except Exception as exc:
    fail_check(f"src.service import failed: {exc}")


try:
    from src.api import app

    pass_check("src.api.app FastAPI application")

except Exception as exc:
    fail_check(f"src.api import failed: {exc}")


# ---------------------------------------------------------------------
# 6. Syntax compilation
# ---------------------------------------------------------------------

print("\n[6/6] Python syntax check")

src_ok = compileall.compile_dir(
    ROOT / "src",
    quiet=1,
)

app_ok = compileall.compile_file(
    ROOT / "app.py",
    quiet=1,
)

if src_ok and app_ok:
    pass_check("Python source compilation")
else:
    fail_check("Python source compilation failed")


# ---------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------

print()
print("=" * 72)
print("Project Check Summary")
print("=" * 72)

print(f"PASS : {passed}")
print(f"WARN : {warnings}")
print(f"FAIL : {failed}")

print("=" * 72)

if failed:
    print("PROJECT CHECK FAILED")
    sys.exit(1)

print("PROJECT CHECK PASSED")
sys.exit(0)