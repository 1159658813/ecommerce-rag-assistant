import os


from dotenv import load_dotenv
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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


# ============================================================
# Project / Environment
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

load_dotenv(
    PROJECT_ROOT / ".env"
)


# ============================================================
# Paths
# ============================================================

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


# ============================================================
# Model Config
# ============================================================

VERIFIER_MODEL = os.getenv(
    "ANSWERABILITY_VERIFIER_MODEL",
    "qwen3.6-plus",
)


# ============================================================
# Build Components
# ============================================================

print("=" * 80)
print("Loading Dense Retriever...")
print("=" * 80)

dense_retriever = Retriever(
    index_path=INDEX_PATH,
    metadata_path=METADATA_PATH,
)


print("\n" + "=" * 80)
print("Loading BGE Reranker...")
print("=" * 80)

reranker = BGEReranker()


print("\n" + "=" * 80)
print("Building TwoStageRetriever...")
print("=" * 80)

two_stage_retriever = (
    TwoStageRetriever(
        dense_retriever=dense_retriever,
        reranker=reranker,
        candidate_k=10,
        final_k=3,
    )
)


print("\n" + "=" * 80)
print(
    "Loading Answerability Verifier:",
    VERIFIER_MODEL,
)
print("=" * 80)

verifier = AnswerabilityVerifier(
    model=VERIFIER_MODEL,
    timeout_seconds=60,
    max_retries=3,
)


print("\n" + "=" * 80)
print("Loading Answer Generator...")
print("=" * 80)

generator = AnswerGenerator()


print("\n" + "=" * 80)
print("Building RAG Pipeline...")
print("=" * 80)

pipeline = RAGPipeline(
    retriever=two_stage_retriever,
    verifier=verifier,
    generator=generator,
    candidate_k=10,
    evidence_k=3,
)


# ============================================================
# Smoke Cases
# ============================================================

CASES = [
    {
        "id": "answerable",
        "question": (
            "银行卡退款多久能到账？"
        ),
    },
    {
        "id": "out_of_kb",
        "question": (
            "平台支持货到付款吗？"
        ),
    },
    {
        "id": "b2_q49",
        "question": (
            "商品原价120元，活动以后98元，"
            "而且还有10元运费，"
            "这样能不能算满100用满100减20？"
        ),
    },
]


# ============================================================
# Run
# ============================================================

for index, case in enumerate(
    CASES,
    start=1,
):

    print("\n\n")
    print("#" * 100)
    print(
        f"CASE {index}: "
        f"{case['id']}"
    )
    print("#" * 100)

    question = case["question"]

    print("\nQuestion:")
    print(question)

    try:
        result = pipeline.ask(
            question=question
        )

    except Exception as error:

        print("\nPIPELINE ERROR:")
        print(
            type(error).__name__,
            str(error),
        )

        continue

    print("\nAbstained:")
    print(
        result["abstained"]
    )

    print("\nAbstain Reason:")
    print(
        result["abstain_reason"]
    )

    print("\nVerifier:")
    print(
        result["verifier"]
    )

    print("\nEvidence:")

    for evidence in result[
        "evidences"
    ]:

        print(
            f"\nRank "
            f"{evidence['rank']}"
        )

        print(
            "Section:",
            evidence["section"],
        )

        print(
            "Reranker Score:",
            evidence[
                "reranker_score"
            ],
        )

    print("\nAnswer:")
    print(
        result["answer"]
    )


print("\n\n")
print("=" * 100)
print("Pipeline Smoke Test Finished")
print("=" * 100)