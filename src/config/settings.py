import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


# ============================================================
# Project Root
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

load_dotenv(
    PROJECT_ROOT / ".env"
)


# ============================================================
# Settings
# ============================================================

def optional_bool_env(name: str) -> bool | None:
    """Unset keeps the provider's existing default; invalid values fail fast."""
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return None
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be true/false or left unset")


@dataclass(frozen=True)
class RAGSettings:

    # --------------------------------------------------------
    # Retrieval
    # --------------------------------------------------------

    candidate_k: int = 10
    evidence_k: int = 3

    # --------------------------------------------------------
    # Reranker
    # --------------------------------------------------------

    reranker_model: str = (
        "BAAI/bge-reranker-v2-m3"
    )

    reranker_max_length: int = 512

    # --------------------------------------------------------
    # Generation
    # --------------------------------------------------------

    generator_model: str = (
        "qwen3.6-flash"
    )
    # --------------------------------------------------------
    # Intent Router
    # --------------------------------------------------------

    intent_router_model: str = (
        "qwen3.6-flash"
    )
    # --------------------------------------------------------
    # Query Rewriter
    # --------------------------------------------------------

    query_rewriter_model: str = (
        "qwen3.6-flash"
    )

    # --------------------------------------------------------
    # Answerability Verifier
    # --------------------------------------------------------

    verifier_model: str = (
        "qwen3.6-plus"
    )

    verifier_timeout_seconds: float = 60.0
    verifier_max_retries: int = 3
    # None: do not send a thinking override (preserves the baseline).
    verifier_enable_thinking: bool | None = None

    # --------------------------------------------------------
    # Data
    # --------------------------------------------------------

    index_path: Path = (
        PROJECT_ROOT
        / "data"
        / "index"
        / "knowledge.faiss"
    )

    metadata_path: Path = (
        PROJECT_ROOT
        / "data"
        / "index"
        / "chunks.json"
    )

    # --------------------------------------------------------
    # Observability
    # --------------------------------------------------------

    log_level: str = "INFO"


def load_settings():

    return RAGSettings(
        candidate_k=int(
            os.getenv(
                "RAG_CANDIDATE_K",
                "10",
            )
        ),

        evidence_k=int(
            os.getenv(
                "RAG_EVIDENCE_K",
                "3",
            )
        ),

        reranker_model=os.getenv(
            "RERANKER_MODEL_NAME",
            "BAAI/bge-reranker-v2-m3",
        ),

        reranker_max_length=int(
            os.getenv(
                "RERANKER_MAX_LENGTH",
                "512",
            )
        ),
        generator_model=os.getenv(
            "DASHSCOPE_MODEL",
            "qwen3.6-flash",
        ),
        intent_router_model=os.getenv(
            "INTENT_ROUTER_MODEL",
            "qwen3.6-flash",
        ),
        query_rewriter_model=os.getenv(
            "QUERY_REWRITER_MODEL",
            "qwen3.6-flash",
        ),

        verifier_model=os.getenv(
            "ANSWERABILITY_VERIFIER_MODEL",
            "qwen3.6-plus",
        ),

        verifier_timeout_seconds=float(
            os.getenv(
                "ANSWERABILITY_VERIFIER_TIMEOUT",
                "60",
            )
        ),

        verifier_max_retries=int(
            os.getenv(
                "ANSWERABILITY_VERIFIER_MAX_RETRIES",
                "3",
            )
        ),
        verifier_enable_thinking=optional_bool_env(
            "ANSWERABILITY_VERIFIER_ENABLE_THINKING"
        ),
        log_level=os.getenv(
            "LOG_LEVEL",
            "INFO",
        ),
    )


settings = load_settings()