import json
import statistics
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULT_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "generation_results_v1.json"
)


def safe_divide(a, b):
    return 0.0 if b == 0 else a / b


def describe(values):
    if not values:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
        }

    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
    }


def print_stats(title, values):
    stats = describe(values)

    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

    print("Count :", stats["count"])

    if not values:
        return

    print(f"Min   : {stats['min']:.6f}")
    print(f"Max   : {stats['max']:.6f}")
    print(f"Mean  : {stats['mean']:.6f}")
    print(f"Median: {stats['median']:.6f}")


def get_evidence_scores(record):
    scores = []

    for evidence in record.get("evidences", []):
        score = evidence.get("reranker_score")

        if isinstance(score, (int, float)):
            scores.append(float(score))

    return scores


def extract_features(record):
    scores = get_evidence_scores(record)

    top1_score = scores[0] if len(scores) >= 1 else None
    top2_score = scores[1] if len(scores) >= 2 else None
    top3_score = scores[2] if len(scores) >= 3 else None

    margin_1_2 = None
    margin_1_3 = None

    if top1_score is not None and top2_score is not None:
        margin_1_2 = top1_score - top2_score

    if top1_score is not None and top3_score is not None:
        margin_1_3 = top1_score - top3_score

    return {
        "id": record.get("id"),
        "question": record.get("question"),
        "answerable": record.get("answerable"),
        "status": record.get("status"),
        "retrieval_top1_correct": record.get(
            "retrieval_top1_correct"
        ),
        "top1_score": top1_score,
        "top2_score": top2_score,
        "top3_score": top3_score,
        "margin_1_2": margin_1_2,
        "margin_1_3": margin_1_3,
    }


def evaluate_threshold(rows, threshold):
    """
    简单实验：

    top1_score >= threshold
        → 预测知识库可回答

    top1_score < threshold
        → 预测知识库不可回答

    注意：
    这里只是实验，不代表 reranker score 是概率。
    """

    valid_rows = [
        row
        for row in rows
        if row["top1_score"] is not None
    ]

    tp = 0
    tn = 0
    fp = 0
    fn = 0

    for row in valid_rows:
        actual = row["answerable"] is True
        predicted = row["top1_score"] >= threshold

        if predicted and actual:
            tp += 1
        elif predicted and not actual:
            fp += 1
        elif not predicted and actual:
            fn += 1
        else:
            tn += 1

    accuracy = safe_divide(tp + tn, len(valid_rows))
    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    specificity = safe_divide(tn, tn + fp)

    return {
        "threshold": threshold,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
    }

def evaluate_score_margin_gate(
    rows,
    score_threshold,
    margin_threshold,
):
    valid_rows = [
        row
        for row in rows
        if row["top1_score"] is not None
        and row["margin_1_2"] is not None
    ]

    tp = tn = fp = fn = 0

    for row in valid_rows:
        actual = row["answerable"] is True

        predicted = (
            row["top1_score"] >= score_threshold
            and row["margin_1_2"] >= margin_threshold
        )

        if predicted and actual:
            tp += 1
        elif predicted and not actual:
            fp += 1
        elif not predicted and actual:
            fn += 1
        else:
            tn += 1

    return {
        "score_threshold": score_threshold,
        "margin_threshold": margin_threshold,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": safe_divide(
            tp + tn,
            len(valid_rows),
        ),
        "precision": safe_divide(
            tp,
            tp + fp,
        ),
        "recall": safe_divide(
            tp,
            tp + fn,
        ),
        "specificity": safe_divide(
            tn,
            tn + fp,
        ),
    }

def main():
    if not RESULT_PATH.exists():
        raise FileNotFoundError(
            f"找不到评测结果：{RESULT_PATH}"
        )

    with RESULT_PATH.open(
        "r",
        encoding="utf-8",
    ) as f:
        payload = json.load(f)

    records = payload.get("records", [])

    if not records:
        raise ValueError(
            "generation_results_v1.json 中没有 records。"
        )

    rows = [
        extract_features(record)
        for record in records
    ]

    score_rows = [
        row
        for row in rows
        if row["top1_score"] is not None
    ]

    print("=" * 80)
    print("Reranker Confidence Analysis V1")
    print("=" * 80)

    print("Total Records:", len(records))
    print("Records With Score:", len(score_rows))

    # =========================================================
    # 1. Answerable vs Unanswerable
    # =========================================================

    answerable_scores = [
        row["top1_score"]
        for row in score_rows
        if row["answerable"] is True
    ]

    unanswerable_scores = [
        row["top1_score"]
        for row in score_rows
        if row["answerable"] is False
    ]

    print_stats(
        "Answerable - Top1 Reranker Score",
        answerable_scores,
    )

    print_stats(
        "Unanswerable - Top1 Reranker Score",
        unanswerable_scores,
    )

    # =========================================================
    # 2. Retrieval Correct vs Incorrect
    # =========================================================

    retrieval_correct_scores = [
        row["top1_score"]
        for row in score_rows
        if row["retrieval_top1_correct"] is True
    ]

    retrieval_wrong_scores = [
        row["top1_score"]
        for row in score_rows
        if row["retrieval_top1_correct"] is False
    ]

    print_stats(
        "Retrieval Top1 Correct - Score",
        retrieval_correct_scores,
    )

    print_stats(
        "Retrieval Top1 Incorrect - Score",
        retrieval_wrong_scores,
    )

    # =========================================================
    # 3. Top1 - Top2 Margin
    # =========================================================

    answerable_margins = [
        row["margin_1_2"]
        for row in score_rows
        if row["answerable"] is True
        and row["margin_1_2"] is not None
    ]

    unanswerable_margins = [
        row["margin_1_2"]
        for row in score_rows
        if row["answerable"] is False
        and row["margin_1_2"] is not None
    ]

    print_stats(
        "Answerable - Top1/Top2 Margin",
        answerable_margins,
    )

    print_stats(
        "Unanswerable - Top1/Top2 Margin",
        unanswerable_margins,
    )

    # =========================================================
    # 4. Threshold Sweep
    # =========================================================

    print("\n" + "=" * 80)
    print("Top1 Score Threshold Sweep")
    print("=" * 80)

    print(
        f"{'Threshold':<12}"
        f"{'Accuracy':<12}"
        f"{'Precision':<12}"
        f"{'Recall':<12}"
        f"{'Specificity':<12}"
        f"{'FP':<6}"
        f"{'FN':<6}"
    )

    thresholds = [
        i / 20
        for i in range(1, 20)
    ]

    results = []

    for threshold in thresholds:
        result = evaluate_threshold(
            score_rows,
            threshold,
        )

        results.append(result)

        print(
            f"{threshold:<12.2f}"
            f"{result['accuracy']:<12.2%}"
            f"{result['precision']:<12.2%}"
            f"{result['recall']:<12.2%}"
            f"{result['specificity']:<12.2%}"
            f"{result['fp']:<6}"
            f"{result['fn']:<6}"
        )

    # =========================================================
    # 5. Inspect hardest cases
    # =========================================================

    print("\n" + "=" * 80)
    print("Highest-score Unanswerable Cases")
    print("=" * 80)

    high_unanswerable = sorted(
        [
            row
            for row in score_rows
            if row["answerable"] is False
        ],
        key=lambda x: x["top1_score"],
        reverse=True,
    )

    for row in high_unanswerable:
        print(
            f"\n{row['id']} | "
            f"score={row['top1_score']:.6f}"
        )
        print(row["question"])

    print("\n" + "=" * 80)
    print("Lowest-score Answerable Cases")
    print("=" * 80)

    low_answerable = sorted(
        [
            row
            for row in score_rows
            if row["answerable"] is True
        ],
        key=lambda x: x["top1_score"],
    )

    for row in low_answerable[:10]:
        print(
            f"\n{row['id']} | "
            f"score={row['top1_score']:.6f} | "
            f"retrieval_correct="
            f"{row['retrieval_top1_correct']}"
        )
        print(row["question"])

    print("\n" + "=" * 80)
    print("Score + Margin Grid Search")
    print("=" * 80)

    score_thresholds = [
        0.10,
        0.20,
        0.30,
        0.40,
        0.50,
        0.60,
        0.70,
        0.80,
        0.85,
    ]

    margin_thresholds = [
        0.00,
        0.01,
        0.02,
        0.05,
        0.10,
        0.15,
        0.20,
    ]

    grid_results = []

    for score_threshold in score_thresholds:
        for margin_threshold in margin_thresholds:
            result = evaluate_score_margin_gate(
                rows,
                score_threshold,
                margin_threshold,
            )
            grid_results.append(result)

    # 优先：
    # 1. FP 少
    # 2. FN 少
    # 3. Accuracy 高
    grid_results.sort(
        key=lambda x: (
            x["fp"],
            x["fn"],
            -x["accuracy"],
        )
    )

    print(
        f"{'Score':<8}"
        f"{'Margin':<8}"
        f"{'Acc':<10}"
        f"{'Prec':<10}"
        f"{'Recall':<10}"
        f"{'Spec':<10}"
        f"{'FP':<5}"
        f"{'FN':<5}"
    )

    for result in grid_results[:20]:
        print(
            f"{result['score_threshold']:<8.2f}"
            f"{result['margin_threshold']:<8.2f}"
            f"{result['accuracy']:<10.2%}"
            f"{result['precision']:<10.2%}"
            f"{result['recall']:<10.2%}"
            f"{result['specificity']:<10.2%}"
            f"{result['fp']:<5}"
            f"{result['fn']:<5}"
        )


if __name__ == "__main__":
    main()