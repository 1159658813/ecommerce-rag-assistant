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
    # Answerability Verifier
    # --------------------------------------------------------

    verifier_model: str = (
        "qwen3.6-plus"
    )

    verifier_timeout_seconds: float = 60.0
    verifier_max_retries: int = 3

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
        log_level=os.getenv(
            "LOG_LEVEL",
            "INFO",
        ),
    )


settings = load_settings()