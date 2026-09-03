from .factory import build_pipeline
from .rag_service import RAGService


def build_rag_service():

    pipeline = build_pipeline()

    return RAGService(
        pipeline=pipeline
    )


__all__ = [
    "RAGService",
    "build_pipeline",
    "build_rag_service",
]