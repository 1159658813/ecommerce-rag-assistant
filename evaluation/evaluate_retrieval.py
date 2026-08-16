import json
import sys
from pathlib import Path

# ============================================================
# Project Path
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retriever import Retriever


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

TOP_K = 3

# 这是上一轮 15 条数据得到的候选阈值
# 这一轮先固定它，用 45 条数据验证
MIN_RETRIEVAL_SCORE = 0.72


# ============================================================
# Load Dataset
# ============================================================

with QUESTIONS_PATH.open(
    "r",
    encoding="utf-8"
) as f:
    questions = json.load(f)


print(f"测试问题总数：{len(questions)}")


# ============================================================
# Load Retriever
# ============================================================

print("正在加载 Retriever...")

retriever = Retriever(
    index_path=INDEX_PATH,
    metadata_path=METADATA_PATH
)

print("Retriever 加载完成。\n")


# ============================================================
# Helper
# ============================================================

def safe_divide(a, b):
    if b == 0:
        return 0.0
    return a / b


def calculate_gate_metrics(
    records,
    threshold
):

    tp = 0
    tn = 0
    fp = 0
    fn = 0

    for record in records:

        truth = record["answerable"]

        predicted = (
            record["best_score"]
            >= threshold
        )

        if truth and predicted:
            tp += 1

        elif not truth and not predicted:
            tn += 1

        elif not truth and predicted:
            fp += 1

        elif truth and not predicted:
            fn += 1


    accuracy = safe_divide(
        tp + tn,
        tp + tn + fp + fn
    )

    precision = safe_divide(
        tp,
        tp + fp
    )

    recall = safe_divide(
        tp,
        tp + fn
    )

    f1 = safe_divide(
        2 * precision * recall,
        precision + recall
    )

    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


# ============================================================
# Retrieval Evaluation
# ============================================================

answerable_count = 0

hit_at_1_count = 0
hit_at_3_count = 0

evaluation_records = []


print("=" * 110)
print("RAG Retrieval Evaluation")
print("=" * 110)


for item in questions:

    question_id = item["id"]
    question = item["question"]

    answerable = item["answerable"]

    expected_sections = set(
        item["expected_sections"]
    )


    # --------------------------------------------------------
    # Retrieval
    # --------------------------------------------------------

    results = retriever.retrieve(
        query=question,
        top_k=TOP_K
    )


    if results:

        best_score = results[0]["score"]

        retrieved_sections = [
            result["document"]["section"]
            for result in results
        ]

        top1_section = (
            results[0]["document"]["section"]
        )

        top1_source = (
            results[0]["document"]["source"]
        )

    else:

        best_score = float("-inf")
        retrieved_sections = []

        top1_section = None
        top1_source = None


    # --------------------------------------------------------
    # Hit@K
    # --------------------------------------------------------

    hit1 = False
    hit3 = False


    if answerable:

        answerable_count += 1

        if (
            retrieved_sections
            and
            retrieved_sections[0]
            in expected_sections
        ):

            hit_at_1_count += 1
            hit1 = True


        if any(
            section in expected_sections
            for section in retrieved_sections
        ):

            hit_at_3_count += 1
            hit3 = True


    # --------------------------------------------------------
    # Current Gate Prediction
    # --------------------------------------------------------

    predicted_answerable = (
        best_score
        >= MIN_RETRIEVAL_SCORE
    )


    # --------------------------------------------------------
    # Save Record
    # --------------------------------------------------------

    evaluation_records.append({
        "id": question_id,
        "question": question,
        "answerable": answerable,
        "expected_sections": list(
            expected_sections
        ),
        "best_score": best_score,
        "top1_section": top1_section,
        "top1_source": top1_source,
        "retrieved_sections":
            retrieved_sections
    })


    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print("\n" + "-" * 110)

    print(
        f"[{question_id}] "
        f"{question}"
    )

    print(
        f"Ground Truth Answerable: "
        f"{answerable}"
    )

    print(
        f"Best Score: "
        f"{best_score:.4f}"
    )

    print(
        f"Gate Prediction "
        f"(threshold={MIN_RETRIEVAL_SCORE:.2f}): "
        f"{predicted_answerable}"
    )

    print(
        f"Expected Sections: "
        f"{list(expected_sections)}"
    )

    print(
        f"Retrieved Sections: "
        f"{retrieved_sections}"
    )


    if answerable:

        print(
            f"Hit@1: {hit1}"
        )

        print(
            f"Hit@3: {hit3}"
        )


# ============================================================
# Retrieval Metrics
# ============================================================

hit_at_1 = safe_divide(
    hit_at_1_count,
    answerable_count
)

hit_at_3 = safe_divide(
    hit_at_3_count,
    answerable_count
)


# ============================================================
# Gate Metrics
# ============================================================

current_metrics = calculate_gate_metrics(
    evaluation_records,
    MIN_RETRIEVAL_SCORE
)


print("\n\n")

print("=" * 110)
print("Evaluation Result")
print("=" * 110)

print(
    f"Total Questions: "
    f"{len(questions)}"
)

print(
    f"Answerable Questions: "
    f"{answerable_count}"
)

print(
    f"Unanswerable Questions: "
    f"{len(questions) - answerable_count}"
)

print()

print(
    f"Hit@1: "
    f"{hit_at_1:.2%}"
)

print(
    f"Hit@3: "
    f"{hit_at_3:.2%}"
)

print()

print(
    f"Current Threshold: "
    f"{MIN_RETRIEVAL_SCORE:.2f}"
)

print()

print(
    "Confusion Matrix"
)

print(
    f"TP: {current_metrics['tp']}"
)

print(
    f"TN: {current_metrics['tn']}"
)

print(
    f"FP: {current_metrics['fp']}"
)

print(
    f"FN: {current_metrics['fn']}"
)

print()

print(
    f"Gate Accuracy: "
    f"{current_metrics['accuracy']:.2%}"
)

print(
    f"Gate Precision: "
    f"{current_metrics['precision']:.2%}"
)

print(
    f"Gate Recall: "
    f"{current_metrics['recall']:.2%}"
)

print(
    f"Gate F1: "
    f"{current_metrics['f1']:.2%}"
)


# ============================================================
# Error Cases at Current Threshold
# ============================================================

print("\n\n")

print("=" * 110)
print(
    f"Gate Error Cases "
    f"(Threshold={MIN_RETRIEVAL_SCORE:.2f})"
)
print("=" * 110)


error_count = 0


for record in evaluation_records:

    predicted = (
        record["best_score"]
        >= MIN_RETRIEVAL_SCORE
    )

    truth = record["answerable"]


    if predicted == truth:
        continue


    error_count += 1


    if predicted and not truth:
        error_type = "FALSE POSITIVE"

    else:
        error_type = "FALSE NEGATIVE"


    print("\n" + "-" * 110)

    print(
        f"Type: {error_type}"
    )

    print(
        f"ID: {record['id']}"
    )

    print(
        f"Question: "
        f"{record['question']}"
    )

    print(
        f"Best Score: "
        f"{record['best_score']:.4f}"
    )

    print(
        f"Top-1 Source: "
        f"{record['top1_source']}"
    )

    print(
        f"Top-1 Section: "
        f"{record['top1_section']}"
    )

    print(
        f"Expected Sections: "
        f"{record['expected_sections']}"
    )


if error_count == 0:

    print(
        "\n当前阈值下没有 Gate Error。"
    )


# ============================================================
# Retrieval Ranking Failure Cases
# ============================================================

print("\n\n")

print("=" * 110)
print("Retrieval Ranking Failure Cases")
print("=" * 110)


retrieval_failure_count = 0


for record in evaluation_records:

    if not record["answerable"]:
        continue


    expected = set(
        record["expected_sections"]
    )


    retrieved = (
        record["retrieved_sections"]
    )


    hit1 = (
        bool(retrieved)
        and retrieved[0] in expected
    )


    if not hit1:

        retrieval_failure_count += 1

        print("\n" + "-" * 110)

        print(
            f"ID: {record['id']}"
        )

        print(
            f"Question: "
            f"{record['question']}"
        )

        print(
            f"Best Score: "
            f"{record['best_score']:.4f}"
        )

        print(
            f"Expected: "
            f"{record['expected_sections']}"
        )

        print(
            f"Retrieved: "
            f"{record['retrieved_sections']}"
        )


if retrieval_failure_count == 0:

    print(
        "\n当前数据集上没有 Hit@1 Failure。"
    )


# ============================================================
# Threshold Sweep
# ============================================================

print("\n\n")

print("=" * 110)
print("Threshold Sweep")
print("=" * 110)

print(
    f"{'Threshold':<12}"
    f"{'Accuracy':<12}"
    f"{'Precision':<12}"
    f"{'Recall':<12}"
    f"{'F1':<12}"
    f"{'FP':<6}"
    f"{'FN':<6}"
)

print("-" * 80)


threshold_results = []


for i in range(40, 86):

    threshold = i / 100


    metrics = calculate_gate_metrics(
        evaluation_records,
        threshold
    )


    threshold_results.append({
        "threshold": threshold,
        **metrics
    })


    print(
        f"{threshold:<12.2f}"
        f"{metrics['accuracy']:<12.2%}"
        f"{metrics['precision']:<12.2%}"
        f"{metrics['recall']:<12.2%}"
        f"{metrics['f1']:<12.2%}"
        f"{metrics['fp']:<6}"
        f"{metrics['fn']:<6}"
    )


# ============================================================
# Best Threshold by F1
# ============================================================

best_f1_result = max(
    threshold_results,
    key=lambda x: x["f1"]
)


# ============================================================
# Conservative Threshold
#
# 对客服场景：
# 优先 FP 少
# 再考虑 FN
# ============================================================

conservative_result = min(
    threshold_results,
    key=lambda x: (
        x["fp"],
        x["fn"],
        -x["f1"],
        x["threshold"]
    )
)


print("\n\n")

print("=" * 110)
print("Threshold Analysis")
print("=" * 110)


print("\nBest F1 Threshold:")

print(
    f"Threshold: "
    f"{best_f1_result['threshold']:.2f}"
)

print(
    f"Accuracy: "
    f"{best_f1_result['accuracy']:.2%}"
)

print(
    f"Precision: "
    f"{best_f1_result['precision']:.2%}"
)

print(
    f"Recall: "
    f"{best_f1_result['recall']:.2%}"
)

print(
    f"F1: "
    f"{best_f1_result['f1']:.2%}"
)

print(
    f"FP: "
    f"{best_f1_result['fp']}"
)

print(
    f"FN: "
    f"{best_f1_result['fn']}"
)


print("\nConservative Candidate:")

print(
    f"Threshold: "
    f"{conservative_result['threshold']:.2f}"
)

print(
    f"Accuracy: "
    f"{conservative_result['accuracy']:.2%}"
)

print(
    f"Precision: "
    f"{conservative_result['precision']:.2%}"
)

print(
    f"Recall: "
    f"{conservative_result['recall']:.2%}"
)

print(
    f"F1: "
    f"{conservative_result['f1']:.2%}"
)

print(
    f"FP: "
    f"{conservative_result['fp']}"
)

print(
    f"FN: "
    f"{conservative_result['fn']}"
)


# ============================================================
# Score Distribution
# ============================================================

answerable_scores = sorted(
    [
        record["best_score"]
        for record in evaluation_records
        if record["answerable"]
    ]
)


unanswerable_scores = sorted(
    [
        record["best_score"]
        for record in evaluation_records
        if not record["answerable"]
    ],
    reverse=True
)


print("\n\n")

print("=" * 110)
print("Score Distribution")
print("=" * 110)


if answerable_scores:

    print(
        "最低 Answerable Score:",
        f"{answerable_scores[0]:.4f}"
    )

    print(
        "最高 Answerable Score:",
        f"{answerable_scores[-1]:.4f}"
    )


if unanswerable_scores:

    print(
        "最高 Unanswerable Score:",
        f"{unanswerable_scores[0]:.4f}"
    )

    print(
        "最低 Unanswerable Score:",
        f"{unanswerable_scores[-1]:.4f}"
    )


print("\nTop-5 Highest Unanswerable Scores:")


unanswerable_records = sorted(
    [
        record
        for record in evaluation_records
        if not record["answerable"]
    ],
    key=lambda x: x["best_score"],
    reverse=True
)


for record in unanswerable_records[:5]:

    print(
        f"{record['best_score']:.4f} | "
        f"{record['question']} | "
        f"Top1={record['top1_section']}"
    )


print("\nBottom-5 Lowest Answerable Scores:")


answerable_records = sorted(
    [
        record
        for record in evaluation_records
        if record["answerable"]
    ],
    key=lambda x: x["best_score"]
)


for record in answerable_records[:5]:

    print(
        f"{record['best_score']:.4f} | "
        f"{record['question']} | "
        f"Top1={record['top1_section']}"
    )