import json
import sys
from pathlib import Path

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)


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

# 当前这组 27 个 Answerable Query 来自 KB V1 的评测集。
# KB V2 改变了部分原 Unanswerable 问题的标签，
# 因此这里暂时只评价原本就 Answerable 的 Query。
QUESTIONS_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "retrieval_questions_batch2.json"
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

# Dense Retriever 一次召回 10 个候选，
# 用来分析 Candidate Recall。
CANDIDATE_K = 10

RERANKER_MODEL_NAME = (
    "BAAI/bge-reranker-v2-m3"
)

RERANKER_MAX_LENGTH = 512

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# Helper Functions
# ============================================================

def safe_divide(a, b):
    if b == 0:
        return 0.0

    return a / b


def hit_at_k(
    retrieved_sections,
    expected_sections,
    k
):
    """
    判断正确 Section 是否出现在 Top-K。
    """

    return any(
        section in expected_sections
        for section in retrieved_sections[:k]
    )


def find_correct_rank(
    retrieved_sections,
    expected_sections
):
    """
    找正确 Section 第一次出现的 Rank。

    找不到返回 None。
    """

    for rank, section in enumerate(
        retrieved_sections,
        start=1
    ):

        if section in expected_sections:
            return rank

    return None


def reciprocal_rank(rank):
    """
    Reciprocal Rank:

    Rank 1 -> 1
    Rank 2 -> 1/2
    Rank 3 -> 1/3
    ...
    """

    if rank is None:
        return 0.0

    return 1.0 / rank


# ============================================================
# Load Evaluation Dataset
# ============================================================

if not QUESTIONS_PATH.exists():

    raise FileNotFoundError(
        f"找不到测试集：{QUESTIONS_PATH}"
    )


with QUESTIONS_PATH.open(
    "r",
    encoding="utf-8"
) as f:

    all_questions = json.load(f)


# ------------------------------------------------------------
# 只评价 Answerable Query
# ------------------------------------------------------------

questions = [
    item
    for item in all_questions
    if item["answerable"]
]


print(
    "Answerable Questions:",
    len(questions)
)


# ============================================================
# Load Dense Retriever
# ============================================================

print("\n正在加载 Dense Retriever...")


retriever = Retriever(
    index_path=INDEX_PATH,
    metadata_path=METADATA_PATH
)


print("Dense Retriever 加载完成。")


# ============================================================
# Load Reranker
# ============================================================

print("\n正在加载 Reranker...")

print(
    "Model:",
    RERANKER_MODEL_NAME
)

print(
    "Device:",
    DEVICE
)


reranker_tokenizer = (
    AutoTokenizer.from_pretrained(
        RERANKER_MODEL_NAME
    )
)


# GPU 使用 FP16，降低显存占用
if DEVICE == "cuda":

    reranker_model = (
        AutoModelForSequenceClassification
        .from_pretrained(
            RERANKER_MODEL_NAME,
            dtype=torch.float16
        )
        .to(DEVICE)
    )

else:

    reranker_model = (
        AutoModelForSequenceClassification
        .from_pretrained(
            RERANKER_MODEL_NAME
        )
        .to(DEVICE)
    )


reranker_model.eval()


print("Reranker 加载完成。\n")


# ============================================================
# Reranker Function
# ============================================================

def rerank(
    query,
    dense_results
):
    """
    对 Dense Retriever 的候选结果重新排序。

    输入：
        query
        dense_results

    输出：
        按 reranker raw logit 降序排列的 list
    """

    if not dense_results:
        return []


    passages = [
        result["document"]["text"]
        for result in dense_results
    ]


    queries = [
        query
        for _ in passages
    ]


    # --------------------------------------------------------
    # Query + Passage 一起送入 Cross-Encoder
    # --------------------------------------------------------

    inputs = reranker_tokenizer(
        queries,
        passages,
        padding=True,
        truncation=True,
        max_length=RERANKER_MAX_LENGTH,
        return_tensors="pt"
    )


    inputs = {
        key: value.to(DEVICE)
        for key, value in inputs.items()
    }


    with torch.inference_mode():

        outputs = reranker_model(
            **inputs
        )


        # BGE Reranker 输出每个 Query-Passage Pair
        # 的一个相关性 logit。
        raw_scores = (
            outputs.logits
            .view(-1)
            .float()
        )


    # sigmoid 主要用于方便我们观察。
    #
    # 排序本身直接使用 raw logit 即可，
    # 因为 sigmoid 是单调函数，不改变排序。
    normalized_scores = torch.sigmoid(
        raw_scores
    )


    reranked_results = []


    for dense_result, raw_score, normalized_score in zip(
        dense_results,
        raw_scores,
        normalized_scores
    ):

        reranked_results.append({
            "document":
                dense_result["document"],

            "dense_score":
                float(
                    dense_result["score"]
                ),

            "reranker_logit":
                float(
                    raw_score.item()
                ),

            "reranker_score":
                float(
                    normalized_score.item()
                )
        })


    # --------------------------------------------------------
    # Reranker 按 raw logit 排序
    # --------------------------------------------------------

    reranked_results.sort(
        key=lambda x: x["reranker_logit"],
        reverse=True
    )


    return reranked_results


# ============================================================
# Evaluation Counters
# ============================================================

total = len(questions)


# ------------------------------------------------------------
# Dense
# ------------------------------------------------------------

dense_hit1_count = 0
dense_hit3_count = 0
dense_hit5_count = 0
dense_hit10_count = 0

dense_mrr_sum = 0.0


# ------------------------------------------------------------
# Reranker
# ------------------------------------------------------------

rerank_hit1_count = 0
rerank_hit3_count = 0
rerank_hit5_count = 0
rerank_hit10_count = 0

rerank_mrr_sum = 0.0


# ------------------------------------------------------------
# Case Analysis
# ------------------------------------------------------------

ranking_changed_count = 0

improvement_cases = []

regression_cases = []

dense_top3_failure_cases = []

dense_top10_failure_cases = []

all_records = []


# ============================================================
# Evaluation
# ============================================================

print("=" * 120)
print("Dense vs Reranker Evaluation")
print("=" * 120)


for question_index, item in enumerate(
    questions,
    start=1
):

    question_id = item["id"]

    question = item["question"]

    expected_sections = set(
        item["expected_sections"]
    )


    print(
        f"\n[{question_index}/{total}] "
        f"{question_id}: {question}"
    )


    # ========================================================
    # 1. Dense Retrieval Top-10
    # ========================================================

    dense_results = retriever.retrieve(
        query=question,
        top_k=CANDIDATE_K
    )


    dense_sections = [
        result["document"]["section"]
        for result in dense_results
    ]


    # ========================================================
    # 2. Dense Correct Rank
    # ========================================================

    dense_correct_rank = find_correct_rank(
        dense_sections,
        expected_sections
    )


    # --------------------------------------------------------
    # Dense Hit@K
    # --------------------------------------------------------

    dense_hit1 = hit_at_k(
        dense_sections,
        expected_sections,
        1
    )

    dense_hit3 = hit_at_k(
        dense_sections,
        expected_sections,
        3
    )

    dense_hit5 = hit_at_k(
        dense_sections,
        expected_sections,
        5
    )

    dense_hit10 = hit_at_k(
        dense_sections,
        expected_sections,
        10
    )


    if dense_hit1:
        dense_hit1_count += 1

    if dense_hit3:
        dense_hit3_count += 1

    if dense_hit5:
        dense_hit5_count += 1

    if dense_hit10:
        dense_hit10_count += 1


    dense_mrr_sum += reciprocal_rank(
        dense_correct_rank
    )


    # ========================================================
    # 3. Rerank Top-10 Candidates
    # ========================================================

    reranked_results = rerank(
        query=question,
        dense_results=dense_results
    )


    reranked_sections = [
        result["document"]["section"]
        for result in reranked_results
    ]


    rerank_correct_rank = find_correct_rank(
        reranked_sections,
        expected_sections
    )


    # --------------------------------------------------------
    # Reranker Hit@K
    # --------------------------------------------------------

    rerank_hit1 = hit_at_k(
        reranked_sections,
        expected_sections,
        1
    )

    rerank_hit3 = hit_at_k(
        reranked_sections,
        expected_sections,
        3
    )

    rerank_hit5 = hit_at_k(
        reranked_sections,
        expected_sections,
        5
    )

    rerank_hit10 = hit_at_k(
        reranked_sections,
        expected_sections,
        10
    )


    if rerank_hit1:
        rerank_hit1_count += 1

    if rerank_hit3:
        rerank_hit3_count += 1

    if rerank_hit5:
        rerank_hit5_count += 1

    if rerank_hit10:
        rerank_hit10_count += 1


    rerank_mrr_sum += reciprocal_rank(
        rerank_correct_rank
    )


    # ========================================================
    # 4. Ranking Changed
    # ========================================================

    if dense_sections != reranked_sections:

        ranking_changed_count += 1


    # ========================================================
    # 5. Improvement
    #
    # Dense Top-1 错
    # Reranker Top-1 对
    # ========================================================

    if (
        not dense_hit1
        and rerank_hit1
    ):

        improvement_cases.append({
            "id": question_id,
            "question": question,

            "expected":
                list(expected_sections),

            "dense_sections":
                dense_sections,

            "reranked_sections":
                reranked_sections,

            "dense_correct_rank":
                dense_correct_rank,

            "rerank_correct_rank":
                rerank_correct_rank
        })


    # ========================================================
    # 6. Regression
    #
    # Dense Top-1 对
    # Reranker Top-1 错
    # ========================================================

    if (
        dense_hit1
        and not rerank_hit1
    ):

        regression_cases.append({
            "id": question_id,
            "question": question,

            "expected":
                list(expected_sections),

            "dense_sections":
                dense_sections,

            "reranked_sections":
                reranked_sections,

            "dense_correct_rank":
                dense_correct_rank,

            "rerank_correct_rank":
                rerank_correct_rank
        })


    # ========================================================
    # 7. Dense Top-3 Recall Failure
    # ========================================================

    if not dense_hit3:

        case = {
            "id": question_id,
            "question": question,

            "expected":
                list(expected_sections),

            "dense_sections":
                dense_sections,

            "dense_correct_rank":
                dense_correct_rank,

            "reranked_sections":
                reranked_sections,

            "rerank_correct_rank":
                rerank_correct_rank
        }


        dense_top3_failure_cases.append(
            case
        )


        # ----------------------------------------------------
        # 连 Top-10 都没找到
        # ----------------------------------------------------

        if dense_correct_rank is None:

            dense_top10_failure_cases.append(
                case
            )


    # ========================================================
    # 8. Save Record
    # ========================================================

    all_records.append({
        "id": question_id,
        "question": question,

        "expected":
            list(expected_sections),

        "dense_sections":
            dense_sections,

        "reranked_sections":
            reranked_sections,

        "dense_rank":
            dense_correct_rank,

        "rerank_rank":
            rerank_correct_rank
    })


    # ========================================================
    # 9. Progress Output
    # ========================================================

    print(
        "Dense Correct Rank:",
        dense_correct_rank
    )

    print(
        "Reranker Correct Rank:",
        rerank_correct_rank
    )


# ============================================================
# Calculate Metrics
# ============================================================

dense_hit1 = safe_divide(
    dense_hit1_count,
    total
)

dense_hit3 = safe_divide(
    dense_hit3_count,
    total
)

dense_hit5 = safe_divide(
    dense_hit5_count,
    total
)

dense_hit10 = safe_divide(
    dense_hit10_count,
    total
)

dense_mrr = safe_divide(
    dense_mrr_sum,
    total
)


rerank_hit1 = safe_divide(
    rerank_hit1_count,
    total
)

rerank_hit3 = safe_divide(
    rerank_hit3_count,
    total
)

rerank_hit5 = safe_divide(
    rerank_hit5_count,
    total
)

rerank_hit10 = safe_divide(
    rerank_hit10_count,
    total
)

rerank_mrr = safe_divide(
    rerank_mrr_sum,
    total
)


# ============================================================
# Final Result
# ============================================================

print("\n\n")

print("=" * 120)
print("Dense vs Reranker Evaluation Result")
print("=" * 120)

print(
    f"Questions: {total}"
)


# ------------------------------------------------------------
# Dense
# ------------------------------------------------------------

print("\nDense Retriever:")

print(
    f"Hit@1 : {dense_hit1:.2%} "
    f"({dense_hit1_count}/{total})"
)

print(
    f"Hit@3 : {dense_hit3:.2%} "
    f"({dense_hit3_count}/{total})"
)

print(
    f"Hit@5 : {dense_hit5:.2%} "
    f"({dense_hit5_count}/{total})"
)

print(
    f"Hit@10: {dense_hit10:.2%} "
    f"({dense_hit10_count}/{total})"
)

print(
    f"MRR@10: {dense_mrr:.4f}"
)


# ------------------------------------------------------------
# Reranker
# ------------------------------------------------------------

print("\nDense + Reranker:")

print(
    f"Hit@1 : {rerank_hit1:.2%} "
    f"({rerank_hit1_count}/{total})"
)

print(
    f"Hit@3 : {rerank_hit3:.2%} "
    f"({rerank_hit3_count}/{total})"
)

print(
    f"Hit@5 : {rerank_hit5:.2%} "
    f"({rerank_hit5_count}/{total})"
)

print(
    f"Hit@10: {rerank_hit10:.2%} "
    f"({rerank_hit10_count}/{total})"
)

print(
    f"MRR@10: {rerank_mrr:.4f}"
)


# ------------------------------------------------------------
# Changes
# ------------------------------------------------------------

print()

print(
    "Ranking Changed:",
    ranking_changed_count
)

print(
    "Improvement Cases:",
    len(improvement_cases)
)

print(
    "Regression Cases:",
    len(regression_cases)
)

print(
    "Dense Top-3 Failure:",
    len(dense_top3_failure_cases)
)

print(
    "Dense Top-10 Failure:",
    len(dense_top10_failure_cases)
)


# ============================================================
# Improvement Cases
# ============================================================

print("\n\n")

print("=" * 120)
print("Improvement Cases")
print("=" * 120)


if not improvement_cases:

    print(
        "\n没有 Dense -> Reranker "
        "Hit@1 改善案例。"
    )

else:

    for case in improvement_cases:

        print(
            "\n" + "-" * 120
        )

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
            "Dense Correct Rank:",
            case["dense_correct_rank"]
        )

        print(
            "Reranker Correct Rank:",
            case["rerank_correct_rank"]
        )

        print(
            "Dense Top-10:",
            case["dense_sections"]
        )

        print(
            "Reranked Top-10:",
            case["reranked_sections"]
        )


# ============================================================
# Regression Cases
# ============================================================

print("\n\n")

print("=" * 120)
print("Regression Cases")
print("=" * 120)


if not regression_cases:

    print(
        "\n没有 Reranker 导致的 "
        "Hit@1 Regression。"
    )

else:

    for case in regression_cases:

        print(
            "\n" + "-" * 120
        )

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
            "Dense Correct Rank:",
            case["dense_correct_rank"]
        )

        print(
            "Reranker Correct Rank:",
            case["rerank_correct_rank"]
        )

        print(
            "Dense Top-10:",
            case["dense_sections"]
        )

        print(
            "Reranked Top-10:",
            case["reranked_sections"]
        )


# ============================================================
# Dense Top-3 Failure Analysis
# ============================================================

print("\n\n")

print("=" * 120)
print("Dense Top-3 Failure Analysis")
print("=" * 120)


if not dense_top3_failure_cases:

    print(
        "\n所有 Answerable Query 的正确 Section "
        "都进入 Dense Top-3。"
    )

else:

    for case in dense_top3_failure_cases:

        print(
            "\n" + "-" * 120
        )

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
            "Dense Top-10:",
            case["dense_sections"]
        )


        if (
            case["dense_correct_rank"]
            is None
        ):

            print(
                "Correct Dense Rank: "
                "NOT FOUND IN TOP-10"
            )

        else:

            print(
                "Correct Dense Rank:",
                case["dense_correct_rank"]
            )


        print(
            "Reranked Top-10:",
            case["reranked_sections"]
        )

        print(
            "Correct Reranker Rank:",
            case["rerank_correct_rank"]
        )


# ============================================================
# Top-10 Recall Failure
# ============================================================

print("\n\n")

print("=" * 120)
print("Dense Top-10 Recall Failure")
print("=" * 120)


if not dense_top10_failure_cases:

    print(
        "\n所有正确 Section 都能进入 Dense Top-10。"
    )

else:

    print(
        "\n以下问题的正确 Section "
        "连 Dense Top-10 都没有召回："
    )


    for case in dense_top10_failure_cases:

        print(
            "\n" + "-" * 120
        )

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
            "Dense Top-10:",
            case["dense_sections"]
        )