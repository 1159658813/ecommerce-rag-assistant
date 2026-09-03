import json
import sys
from pathlib import Path


# ============================================================
# Project Path
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT)
    )


from src.retrieval.retriever import Retriever
from experiments.legacy_rag_v1.llm import QwenGenerator

from src.verification.evidence_judge import (
    EvidenceJudge,
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
# Load Dataset
# ============================================================

with QUESTIONS_PATH.open(
    "r",
    encoding="utf-8"
) as f:

    questions = json.load(f)


print(
    "Questions:",
    len(questions)
)


# ============================================================
# Load Models
# ============================================================

print("正在加载 Retriever...")


retriever = Retriever(
    index_path=INDEX_PATH,
    metadata_path=METADATA_PATH
)


print("正在加载 Qwen 1.5B...")


generator = QwenGenerator(
    model_name=(
        "Qwen/Qwen2.5-1.5B-Instruct"
    )
)


judge = EvidenceJudge(
    generator=generator
)


print("加载完成。\n")


# ============================================================
# Metrics
# ============================================================

tp = 0
tn = 0
fp = 0
fn = 0


# ============================================================
# Evaluation
# ============================================================

for item in questions:

    question = item["question"]
    truth = item["answerable"]


    # --------------------------------------------------------
    # Retrieve Top-1
    # --------------------------------------------------------

    results = retriever.retrieve(
        query=question,
        top_k=1
    )


    if not results:

        predicted = False

        raw_output = (
            "NO_RETRIEVAL_RESULT"
        )

        top1_section = None
        score = None


    else:

        result = results[0]

        document = result["document"]

        score = result["score"]

        top1_section = (
            document["section"]
        )


        # ----------------------------------------------------
        # Evidence Sufficiency Judge
        # ----------------------------------------------------

        judge_result = judge.judge(
            query=question,
            document=document
        )


        predicted = (
            judge_result["sufficient"]
        )

        raw_output = (
            judge_result["raw_output"]
        )


    # --------------------------------------------------------
    # Confusion Matrix
    # --------------------------------------------------------

    if truth and predicted:

        tp += 1

    elif not truth and not predicted:

        tn += 1

    elif not truth and predicted:

        fp += 1

    elif truth and not predicted:

        fn += 1


    # --------------------------------------------------------
    # Print Case
    # --------------------------------------------------------

    print(
        "\n" + "=" * 100
    )

    print(
        f"[{item['id']}] "
        f"{question}"
    )

    print(
        "Ground Truth:",
        truth
    )

    print(
        "Prediction:",
        predicted
    )

    print(
        "Judge Output:",
        repr(raw_output)
    )

    print(
        "Top-1:",
        top1_section
    )


    if score is not None:

        print(
            "Dense Score:",
            f"{score:.4f}"
        )


# ============================================================
# Metrics
# ============================================================

total = (
    tp
    + tn
    + fp
    + fn
)


accuracy = (
    (tp + tn) / total
    if total
    else 0
)


precision = (
    tp / (tp + fp)
    if tp + fp
    else 0
)


recall = (
    tp / (tp + fn)
    if tp + fn
    else 0
)


f1 = (
    2
    * precision
    * recall
    / (precision + recall)
    if precision + recall
    else 0
)


print("\n\n")

print("=" * 100)

print(
    "Evidence Sufficiency Evaluation"
)

print("=" * 100)


print(
    f"TP: {tp}"
)

print(
    f"TN: {tn}"
)

print(
    f"FP: {fp}"
)

print(
    f"FN: {fn}"
)

print()

print(
    f"Accuracy: "
    f"{accuracy:.2%}"
)

print(
    f"Precision: "
    f"{precision:.2%}"
)

print(
    f"Recall: "
    f"{recall:.2%}"
)

print(
    f"F1: "
    f"{f1:.2%}"
)