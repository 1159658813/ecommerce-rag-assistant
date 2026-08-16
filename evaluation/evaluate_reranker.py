import json
import sys
from pathlib import Path


# ============================================================
# Project Path
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)


if str(PROJECT_ROOT) not in sys.path:

    sys.path.insert(
        0,
        str(PROJECT_ROOT)
    )


from src.retriever import Retriever
from src.reranker import BGEReranker
from src.two_stage_retriever import (
    TwoStageRetriever
)


# ============================================================
# Paths
# ============================================================

QUESTIONS_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "questions.json"
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


# ============================================================
# Config
# ============================================================

CANDIDATE_K = 5

FINAL_K = 3


# ============================================================
# Load Dataset
# ============================================================

with QUESTIONS_PATH.open(
    "r",
    encoding="utf-8"
) as f:

    questions = json.load(f)


# 这里只评价 Answerable Questions
questions = [
    item
    for item in questions
    if item["answerable"]
]


print(
    "Answerable Questions:",
    len(questions)
)


# ============================================================
# Load Models
# ============================================================

print(
    "正在加载 Dense Retriever..."
)


dense_retriever = Retriever(
    index_path=INDEX_PATH,
    metadata_path=METADATA_PATH
)


print(
    "正在加载 Reranker..."
)


reranker = BGEReranker()


two_stage_retriever = (
    TwoStageRetriever(

        dense_retriever=dense_retriever,

        reranker=reranker,

        candidate_k=CANDIDATE_K,

        final_k=FINAL_K
    )
)


# ============================================================
# Metrics
# ============================================================

dense_hit1 = 0
dense_hit3 = 0

rerank_hit1 = 0
rerank_hit3 = 0

ranking_changed = 0

regression_cases = []
improvement_cases = []


# ============================================================
# Evaluate
# ============================================================

for item in questions:

    question = item["question"]

    expected = set(
        item["expected_sections"]
    )


    # ==========================================
    # Dense Retrieval
    # ==========================================

    dense_results = (
        dense_retriever.retrieve(
            query=question,
            top_k=CANDIDATE_K
        )
    )


    dense_sections = [
        result["document"]["section"]
        for result in dense_results
    ]


    dense_top1_correct = (
        bool(dense_sections)
        and
        dense_sections[0] in expected
    )


    dense_top3_correct = any(
        section in expected
        for section in dense_sections[:3]
    )


    if dense_top1_correct:
        dense_hit1 += 1

    if dense_top3_correct:
        dense_hit3 += 1


    # ==========================================
    # Rerank
    # ==========================================

    reranked_results = (
        reranker.rerank(
            query=question,
            candidates=dense_results,
            top_k=FINAL_K
        )
    )


    reranked_sections = [
        result["document"]["section"]
        for result in reranked_results
    ]


    rerank_top1_correct = (
        bool(reranked_sections)
        and
        reranked_sections[0] in expected
    )


    rerank_top3_correct = any(
        section in expected
        for section in reranked_sections
    )


    if rerank_top1_correct:
        rerank_hit1 += 1

    if rerank_top3_correct:
        rerank_hit3 += 1


    # ==========================================
    # Ranking Change
    # ==========================================

    if (
        dense_sections[:FINAL_K]
        != reranked_sections
    ):

        ranking_changed += 1


    # Dense错 → Reranker对
    if (
        not dense_top1_correct
        and rerank_top1_correct
    ):

        improvement_cases.append({
            "id": item["id"],
            "question": question,
            "expected":
                list(expected),
            "dense":
                dense_sections[:3],
            "reranked":
                reranked_sections
        })


    # Dense对 → Reranker错
    if (
        dense_top1_correct
        and not rerank_top1_correct
    ):

        regression_cases.append({
            "id": item["id"],
            "question": question,
            "expected":
                list(expected),
            "dense":
                dense_sections[:3],
            "reranked":
                reranked_sections
        })


# ============================================================
# Result
# ============================================================

total = len(questions)


print("\n")

print("=" * 100)
print("Dense vs Reranker Evaluation")
print("=" * 100)


print(
    f"Questions: {total}"
)


print("\nDense Retriever:")

print(
    f"Hit@1: "
    f"{dense_hit1 / total:.2%}"
)

print(
    f"Hit@3: "
    f"{dense_hit3 / total:.2%}"
)


print("\nDense + Reranker:")

print(
    f"Hit@1: "
    f"{rerank_hit1 / total:.2%}"
)

print(
    f"Hit@3: "
    f"{rerank_hit3 / total:.2%}"
)


print()

print(
    "Ranking Changed:",
    ranking_changed
)

print(
    "Improvement Cases:",
    len(improvement_cases)
)

print(
    "Regression Cases:",
    len(regression_cases)
)


# ============================================================
# Print Improvement
# ============================================================

print("\n")

print("=" * 100)
print("Improvement Cases")
print("=" * 100)


if not improvement_cases:

    print(
        "没有 Dense→Reranker 的 Hit@1 改善案例。"
    )


for case in improvement_cases:

    print("\n" + "-" * 100)

    print(
        "ID:",
        case["id"]
    )

    print(
        "Question:",
        case["question"]
    )

    print(
        "Expected:",
        case["expected"]
    )

    print(
        "Dense:",
        case["dense"]
    )

    print(
        "Reranked:",
        case["reranked"]
    )


# ============================================================
# Print Regression
# ============================================================

print("\n")

print("=" * 100)
print("Regression Cases")
print("=" * 100)


if not regression_cases:

    print(
        "没有 Reranker 导致的 Hit@1 Regression。"
    )


for case in regression_cases:

    print("\n" + "-" * 100)

    print(
        "ID:",
        case["id"]
    )

    print(
        "Question:",
        case["question"]
    )

    print(
        "Expected:",
        case["expected"]
    )

    print(
        "Dense:",
        case["dense"]
    )

    print(
        "Reranked:",
        case["reranked"]
    )