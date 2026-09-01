import os
from pathlib import Path

from dotenv import load_dotenv

from src.retrieval import (
    Retriever,
    BGEReranker,
    TwoStageRetriever,
)

from src.verification import (
    AnswerabilityVerifier,
)

from src.generation import (
    AnswerGenerator,
)

from src.pipeline import (
    RAGPipeline,
)


PROJECT_ROOT = Path(__file__).resolve().parent


load_dotenv(
    PROJECT_ROOT / ".env"
)


INDEX_PATH = (
    PROJECT_ROOT
    / "data"
    / "index"
    / "knowledge.faiss"
)


METADATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "index"
    / "chunks.json"
)


VERIFIER_MODEL = os.getenv(
    "ANSWERABILITY_VERIFIER_MODEL",
    "qwen3.6-plus",
)


def build_pipeline():

    dense_retriever = Retriever(
        index_path=INDEX_PATH,
        metadata_path=METADATA_PATH,
    )


    reranker = BGEReranker()


    retriever = TwoStageRetriever(
        dense_retriever=dense_retriever,
        reranker=reranker,
        candidate_k=10,
        final_k=3,
    )


    verifier = AnswerabilityVerifier(
        model=VERIFIER_MODEL,
        timeout_seconds=60,
        max_retries=3,
    )


    generator = AnswerGenerator()


    pipeline = RAGPipeline(
        retriever=retriever,
        verifier=verifier,
        generator=generator,
        candidate_k=10,
        evidence_k=3,
    )


    return pipeline



if __name__ == "__main__":

    pipeline = build_pipeline()


    while True:

        question = input(
            "\nUser: "
        )


        if question.lower() in [
            "exit",
            "quit",
        ]:
            break


        result = pipeline.ask(
            question
        )


        print(
            "\nAssistant:"
        )

        print(
            result["answer"]
        )