from src.config import settings

from src.retrieval import (
    Retriever,
    BGEReranker,
    TwoStageRetriever,
)
from src.verification import AnswerabilityVerifier
from src.generation import AnswerGenerator
from src.pipeline import RAGPipeline


def build_pipeline():

    dense_retriever = Retriever(
        index_path=settings.index_path,
        metadata_path=settings.metadata_path,
    )

    reranker = BGEReranker(
        model_name=settings.reranker_model,
        max_length=settings.reranker_max_length,
    )

    retriever = TwoStageRetriever(
        dense_retriever=dense_retriever,
        reranker=reranker,
        candidate_k=settings.candidate_k,
        final_k=settings.evidence_k,
    )

    verifier = AnswerabilityVerifier(
        model=settings.verifier_model,
        timeout_seconds=(
            settings.verifier_timeout_seconds
        ),
        max_retries=(
            settings.verifier_max_retries
        ),
    )

    generator = AnswerGenerator()

    return RAGPipeline(
        retriever=retriever,
        verifier=verifier,
        generator=generator,
        candidate_k=settings.candidate_k,
        evidence_k=settings.evidence_k,
    )