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

from src.cloud_llm import (
    DashScopeQwenGenerator
)

from src.evidence_judge import (
    EvidenceJudge
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
# Retriever
# ============================================================

print(
    "\n正在加载 Retriever..."
)

retriever = Retriever(
    index_path=INDEX_PATH,
    metadata_path=METADATA_PATH
)


# ============================================================
# Cloud Qwen3-8B
# ============================================================

print(
    "\n正在初始化 qwen3-8b API..."
)

generator = DashScopeQwenGenerator(


    # 本轮严格关闭思考模式
    enable_thinking=False,

    # Judge任务降低随机性
    temperature=0.0
)


judge = EvidenceJudge(
    generator=generator
)


print(
    "初始化完成。\n"
)


# ============================================================
# Metrics
# ============================================================

tp = 0
tn = 0
fp = 0
fn = 0

error_records = []


# ============================================================
# Evaluation
# ============================================================

for item in questions:

    question = item["question"]

    truth = item["answerable"]


    # --------------------------------------------------------
    # 1. Retrieve Top-1
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

        section = None
        dense_score = None


    else:

        result = results[0]

        document = (
            result["document"]
        )

        section = (
            document["section"]
        )

        dense_score = (
            result["score"]
        )


        # ----------------------------------------------------
        # 2. Evidence Sufficiency Judge
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
    # 3. Confusion Matrix
    # --------------------------------------------------------

    if truth and predicted:

        tp += 1


    elif (
        not truth
        and not predicted
    ):

        tn += 1


    elif (
        not truth
        and predicted
    ):

        fp += 1


    elif (
        truth
        and not predicted
    ):

        fn += 1


    # --------------------------------------------------------
    # 4. Error Record
    # --------------------------------------------------------

    if truth != predicted:

        error_records.append({

            "id":
                item["id"],

            "question":
                question,

            "truth":
                truth,

            "prediction":
                predicted,

            "raw_output":
                raw_output,

            "section":
                section,

            "dense_score":
                dense_score
        })


    # --------------------------------------------------------
    # 5. Print Case
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
        section
    )

    if dense_score is not None:

        print(
            "Dense Score:",
            f"{dense_score:.4f}"
        )


# ============================================================
# Metrics
# ============================================================

total = tp + tn + fp + fn


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
    (
        2
        * precision
        * recall
        / (
            precision
            + recall
        )
    )

    if (
        precision
        + recall
    )

    else 0
)


# ============================================================
# Final Result
# ============================================================

print("\n\n")

print("=" * 100)

print("=" * 100)

print(
    f"{generator.model_name} "
    f"Evidence Sufficiency Evaluation"
)

print("=" * 100)

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


# ============================================================
# Error Cases
# ============================================================

print("\n\n")

print("=" * 100)

print(
    "Qwen3-8B Gate Error Cases"
)

print("=" * 100)


if not error_records:

    print(
        "\nNo errors."
    )


for record in error_records:

    print(
        "\n" + "-" * 100
    )


    error_type = (
        "FALSE POSITIVE"
        if (
            record["prediction"]
            and not record["truth"]
        )
        else
        "FALSE NEGATIVE"
    )


    print(
        "Type:",
        error_type
    )

    print(
        "ID:",
        record["id"]
    )

    print(
        "Question:",
        record["question"]
    )

    print(
        "Top-1:",
        record["section"]
    )


    if (
        record["dense_score"]
        is not None
    ):

        print(
            "Dense Score:",
            f"{record['dense_score']:.4f}"
        )


    print(
        "Judge Output:",
        repr(
            record["raw_output"]
        )
    )