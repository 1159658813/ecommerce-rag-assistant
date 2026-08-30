import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv


# ============================================================
# Project Root / Imports
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )

from src.verification.answerability_verifier_v1 import (
    AnswerabilityVerifier,
)


# ============================================================
# Environment / Paths
# ============================================================

load_dotenv(PROJECT_ROOT / ".env")

INPUT_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "generation_results_prompt_v2_1_k3.json"
)

LABELS_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "evidence_sufficiency_labels_v1.json"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "answerability_results_v1_1.json"
)

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "answerability_results_v1_1.checkpoint.json"
)


# ============================================================
# Configuration
# ============================================================

VERIFIER_MODEL = os.getenv(
    "ANSWERABILITY_VERIFIER_MODEL",
    "qwen-max",
)

VERIFIER_TIMEOUT_SECONDS = float(
    os.getenv(
        "ANSWERABILITY_VERIFIER_TIMEOUT",
        "60",
    )
)

VERIFIER_MAX_RETRIES = int(
    os.getenv(
        "ANSWERABILITY_VERIFIER_MAX_RETRIES",
        "3",
    )
)

VERIFIER_RETRY_BASE_SECONDS = float(
    os.getenv(
        "ANSWERABILITY_VERIFIER_RETRY_BASE",
        "2",
    )
)

EXPECTED_EVIDENCE_K = 3

VALID_VERDICTS = {
    "SUFFICIENT",
    "INSUFFICIENT",
}


# ============================================================
# Basic Helpers
# ============================================================

def safe_divide(
    numerator,
    denominator,
):
    return (
        0.0
        if denominator == 0
        else numerator / denominator
    )


def write_json(
    path,
    data,
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2,
        )


def print_separator(
    char="=",
    width=120,
):
    print(char * width)


def classify_exception(
    error,
):
    name = type(
        error
    ).__name__.upper()

    message = repr(
        error
    ).upper()

    if (
        "TIMEOUT" in name
        or "TIMED OUT" in message
        or "TIMEOUT" in message
    ):
        return "VERIFIER_TIMEOUT"

    if (
        "JSON" in name
        or "PARSE" in name
        or "可解析 JSON" in str(error)
        or "VERDICT 非法" in str(error).upper()
    ):
        return "VERIFIER_PARSE_ERROR"

    return "VERIFIER_API_ERROR"


# ============================================================
# Load Frozen Generation Results
# ============================================================

def load_generation_payload():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"找不到冻结的 Generation 结果："
            f"{INPUT_PATH}"
        )

    with INPUT_PATH.open(
        "r",
        encoding="utf-8",
    ) as f:
        payload = json.load(f)

    if not isinstance(
        payload,
        dict,
    ):
        raise ValueError(
            "Generation 结果文件必须是 JSON Object。"
        )

    records = payload.get(
        "records",
        [],
    )

    if not isinstance(
        records,
        list,
    ) or not records:
        raise ValueError(
            "Generation 结果文件中没有 records。"
        )

    return payload, records


# ============================================================
# Load Manual Evidence Sufficiency Labels
# ============================================================

def load_evidence_sufficiency_labels():
    if not LABELS_PATH.exists():
        raise FileNotFoundError(
            f"找不到人工 Evidence Sufficiency 标签："
            f"{LABELS_PATH}"
        )

    with LABELS_PATH.open(
        "r",
        encoding="utf-8",
    ) as f:
        items = json.load(f)

    if not isinstance(
        items,
        list,
    ) or not items:
        raise ValueError(
            "evidence_sufficiency_labels_v1.json "
            "必须是非空 JSON 数组。"
        )

    label_map = {}
    errors = []

    for index, item in enumerate(
        items,
        start=1,
    ):
        if not isinstance(
            item,
            dict,
        ):
            errors.append(
                f"第 {index} 条 label 不是 JSON Object。"
            )
            continue

        question_id = str(
            item.get(
                "id",
                "",
            )
        ).strip()

        evidence_sufficient = item.get(
            "evidence_sufficient"
        )

        if not question_id:
            errors.append(
                f"第 {index} 条 label 缺少 id。"
            )
            continue

        if question_id in label_map:
            errors.append(
                f"Label ID 重复：{question_id}"
            )
            continue

        if not isinstance(
            evidence_sufficient,
            bool,
        ):
            errors.append(
                f"{question_id}: "
                "evidence_sufficient 必须是 bool。"
            )
            continue

        label_map[
            question_id
        ] = evidence_sufficient

    if errors:
        raise ValueError(
            "Evidence Sufficiency 标签校验失败：\n- "
            + "\n- ".join(errors)
        )

    return items, label_map


# ============================================================
# Frozen Input / Label Join Validation
# ============================================================

def validate_and_join(
    payload,
    source_records,
    label_items,
    label_map,
):
    """
    V1.1 的关键：
    - answerable = KB 层级标签
    - evidence_sufficient = 当前冻结 Top-3 Evidence 层级标签

    Verifier 评测 Ground Truth 必须使用 evidence_sufficient。
    """

    config = payload.get(
        "config",
        {},
    )

    evidence_k = config.get(
        "reranker_evidence_k"
    )

    if evidence_k != EXPECTED_EVIDENCE_K:
        raise ValueError(
            "当前实验要求冻结 Top-3 Evidence，"
            f"但输入文件 reranker_evidence_k="
            f"{evidence_k!r}"
        )

    source_ids = []
    errors = []

    seen_source_ids = set()

    for index, record in enumerate(
        source_records,
        start=1,
    ):
        if not isinstance(
            record,
            dict,
        ):
            errors.append(
                f"第 {index} 条 source record "
                "不是 JSON Object。"
            )
            continue

        question_id = str(
            record.get(
                "id",
                "",
            )
        ).strip()

        question = str(
            record.get(
                "question",
                "",
            )
        ).strip()

        answerable = record.get(
            "answerable"
        )

        evidences = record.get(
            "evidences"
        )

        if not question_id:
            errors.append(
                f"第 {index} 条 source record 缺少 id。"
            )
            continue

        if question_id in seen_source_ids:
            errors.append(
                f"Source ID 重复：{question_id}"
            )
            continue

        seen_source_ids.add(
            question_id
        )

        source_ids.append(
            question_id
        )

        if not question:
            errors.append(
                f"{question_id}: question 为空。"
            )

        if not isinstance(
            answerable,
            bool,
        ):
            errors.append(
                f"{question_id}: answerable 必须是 bool。"
            )

        if not isinstance(
            evidences,
            list,
        ):
            errors.append(
                f"{question_id}: evidences 必须是数组。"
            )
            continue

        if len(
            evidences
        ) < EXPECTED_EVIDENCE_K:
            errors.append(
                f"{question_id}: "
                f"Evidence 数量不足 {EXPECTED_EVIDENCE_K}，"
                f"实际 {len(evidences)}。"
            )

        if question_id not in label_map:
            errors.append(
                f"{question_id}: "
                "缺少 evidence_sufficient 人工标签。"
            )

    source_id_set = set(
        source_ids
    )

    label_id_set = set(
        label_map.keys()
    )

    extra_label_ids = sorted(
        label_id_set
        - source_id_set
    )

    if extra_label_ids:
        errors.append(
            "Label 文件存在 Source 中没有的未知 ID："
            + ", ".join(
                extra_label_ids
            )
        )

    if len(
        label_items
    ) != len(
        source_records
    ):
        errors.append(
            "Label 数量与冻结 Source 数量不一致："
            f"labels={len(label_items)}, "
            f"source={len(source_records)}"
        )

    if errors:
        raise ValueError(
            "Answerability V1.1 输入/标签校验失败：\n- "
            + "\n- ".join(errors)
        )

    joined_records = []

    for source_record in source_records:
        question_id = source_record["id"]

        joined = dict(
            source_record
        )

        joined[
            "evidence_sufficient"
        ] = label_map[
            question_id
        ]

        joined_records.append(
            joined
        )

    return joined_records


# ============================================================
# Metrics
# ============================================================

def build_summary(
    records,
    total,
):
    evaluated_records = [
        record
        for record in records
        if record.get(
            "status"
        ) in VALID_VERDICTS
    ]

    error_records = [
        record
        for record in records
        if record.get(
            "status"
        ) not in VALID_VERDICTS
    ]

    tp = tn = fp = fn = 0

    for record in evaluated_records:
        actual = (
            record.get(
                "evidence_sufficient"
            )
            is True
        )

        predicted = (
            record.get(
                "status"
            )
            == "SUFFICIENT"
        )

        if predicted and actual:
            tp += 1

        elif predicted and not actual:
            fp += 1

        elif not predicted and actual:
            fn += 1

        else:
            tn += 1

    accuracy = safe_divide(
        tp + tn,
        len(
            evaluated_records
        ),
    )

    precision = safe_divide(
        tp,
        tp + fp,
    )

    recall = safe_divide(
        tp,
        tp + fn,
    )

    specificity = safe_divide(
        tn,
        tn + fp,
    )

    f1 = safe_divide(
        2
        * precision
        * recall,
        precision
        + recall,
    )

    error_types = {}

    for record in error_records:
        status = record.get(
            "status",
            "UNKNOWN_ERROR",
        )

        error_types[
            status
        ] = (
            error_types.get(
                status,
                0,
            )
            + 1
        )

    kb_answerable_count = sum(
        record.get(
            "answerable"
        ) is True
        for record in records
    )

    evidence_sufficient_count = sum(
        record.get(
            "evidence_sufficient"
        ) is True
        for record in records
    )

    kb_answerable_but_evidence_insufficient = [
        record["id"]
        for record in records
        if record.get(
            "answerable"
        ) is True
        and record.get(
            "evidence_sufficient"
        ) is False
    ]

    return {
        "total": total,
        "evaluated": len(
            evaluated_records
        ),
        "errors": len(
            error_records
        ),
        "evaluation_coverage": safe_divide(
            len(
                evaluated_records
            ),
            total,
        ),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "kb_answerable_count":
            kb_answerable_count,
        "evidence_sufficient_count":
            evidence_sufficient_count,
        "kb_answerable_but_evidence_insufficient":
            kb_answerable_but_evidence_insufficient,
        "error_types":
            error_types,
    }


# ============================================================
# Save Helpers
# ============================================================

def save_checkpoint(
    records,
):
    payload = {
        "config": {
            "input_path": str(
                INPUT_PATH
            ),
            "labels_path": str(
                LABELS_PATH
            ),
            "verifier_model":
                VERIFIER_MODEL,
            "expected_evidence_k":
                EXPECTED_EVIDENCE_K,
        },
        "records":
            records,
    }

    write_json(
        CHECKPOINT_PATH,
        payload,
    )


def print_case_detail(
    case,
):
    print(
        "\n" + "-" * 120
    )

    print(
        "ID:",
        case.get(
            "id"
        ),
    )

    print(
        "Question:",
        case.get(
            "question"
        ),
    )

    print(
        "KB Answerable:",
        case.get(
            "answerable"
        ),
    )

    print(
        "Evidence Sufficient Ground Truth:",
        case.get(
            "evidence_sufficient"
        ),
    )

    print(
        "Verifier Prediction:",
        case.get(
            "status"
        ),
    )

    print(
        "Covered Facts:",
        case.get(
            "verifier",
            {},
        ).get(
            "covered_facts"
        ),
    )

    print(
        "Missing Facts:",
        case.get(
            "verifier",
            {},
        ).get(
            "missing_facts"
        ),
    )

    print(
        "Reason:",
        case.get(
            "verifier",
            {},
        ).get(
            "reason"
        ),
    )

    print(
        "\nTop-3 Evidence:"
    )

    for evidence in case.get(
        "evidences",
        [],
    ):
        print(
            "\nSection:",
            evidence.get(
                "section"
            ),
        )

        print(
            "Reranker Score:",
            evidence.get(
                "reranker_score"
            ),
        )

        print(
            "Content:",
            evidence.get(
                "content"
            ),
        )


# ============================================================
# Load + Validate
# ============================================================

payload, source_records = (
    load_generation_payload()
)

label_items, label_map = (
    load_evidence_sufficiency_labels()
)

joined_records = (
    validate_and_join(
        payload=payload,
        source_records=source_records,
        label_items=label_items,
        label_map=label_map,
    )
)

print_separator()
print(
    "Answerability / "
    "Evidence Coverage Evaluation V1.1"
)
print_separator()

print(
    "Frozen Input:",
    INPUT_PATH,
)

print(
    "Manual Labels:",
    LABELS_PATH,
)

print(
    "Verifier Model:",
    VERIFIER_MODEL,
)

print(
    "Questions:",
    len(
        joined_records
    ),
)

print(
    "Evidence K:",
    EXPECTED_EVIDENCE_K,
)

print(
    "Manual evidence_sufficient=true:",
    sum(
        record[
            "evidence_sufficient"
        ]
        is True
        for record in joined_records
    ),
)

print(
    "Manual evidence_sufficient=false:",
    sum(
        record[
            "evidence_sufficient"
        ]
        is False
        for record in joined_records
    ),
)


# ============================================================
# Load Verifier
# ============================================================

print(
    "\n正在加载 "
    "AnswerabilityVerifier..."
)

verifier = (
    AnswerabilityVerifier(
        model=VERIFIER_MODEL,
        timeout_seconds=(
            VERIFIER_TIMEOUT_SECONDS
        ),
        max_retries=(
            VERIFIER_MAX_RETRIES
        ),
        retry_base_seconds=(
            VERIFIER_RETRY_BASE_SECONDS
        ),
    )
)

print(
    "AnswerabilityVerifier 加载完成。"
)


# ============================================================
# Evaluation
# ============================================================

records = []

total = len(
    joined_records
)

for index, source_record in enumerate(
    joined_records,
    start=1,
):
    question_id = source_record[
        "id"
    ]

    question = source_record[
        "question"
    ]

    answerable = source_record[
        "answerable"
    ]

    evidence_sufficient = (
        source_record[
            "evidence_sufficient"
        ]
    )

    evidences = (
        source_record
        .get(
            "evidences",
            [],
        )[
            :EXPECTED_EVIDENCE_K
        ]
    )

    print(
        "\n\n"
    )

    print_separator()

    print(
        f"[{index}/{total}] "
        f"{question_id}"
    )

    print(
        "Question:",
        question,
    )

    print(
        "KB Answerable:",
        answerable,
    )

    print(
        "Evidence Sufficient Ground Truth:",
        evidence_sufficient,
    )

    try:
        verifier_result = (
            verifier.verify(
                question=question,
                evidences=evidences,
            )
        )

        status = verifier_result[
            "verdict"
        ]

        error_info = None

    except Exception as error:
        status = classify_exception(
            error
        )

        error_info = {
            "type": status,
            "message": repr(
                error
            ),
        }

        verifier_result = {
            "verdict":
                "VERIFIER_ERROR",
            "covered_facts": [],
            "missing_facts": [],
            "reason": repr(
                error
            ),
            "raw_output": None,
        }

    print(
        "Verifier Status:",
        status,
    )

    print(
        "Covered Facts:",
        verifier_result.get(
            "covered_facts"
        ),
    )

    print(
        "Missing Facts:",
        verifier_result.get(
            "missing_facts"
        ),
    )

    print(
        "Reason:",
        verifier_result.get(
            "reason"
        ),
    )

    record = {
        "id":
            question_id,
        "question":
            question,
        "category":
            source_record.get(
                "category"
            ),
        "answerable":
            answerable,
        "evidence_sufficient":
            evidence_sufficient,
        "expected_sections":
            source_record.get(
                "expected_sections",
                [],
            ),
        "evidences":
            evidences,
        "retrieval_top1_section":
            source_record.get(
                "retrieval_top1_section"
            ),
        "retrieval_top1_correct":
            source_record.get(
                "retrieval_top1_correct"
            ),
        "status":
            status,
        "verifier":
            verifier_result,
    }

    if error_info is not None:
        record[
            "error"
        ] = error_info

    records.append(
        record
    )

    save_checkpoint(
        records
    )


# ============================================================
# Final Metrics
# ============================================================

summary = build_summary(
    records=records,
    total=total,
)

print(
    "\n\n"
)

print_separator()
print(
    "Answerability V1.1 Result"
)
print_separator()

print(
    "Questions:",
    total,
)

print(
    "Successfully Evaluated:",
    summary[
        "evaluated"
    ],
)

print(
    "ERROR:",
    summary[
        "errors"
    ],
)

print(
    f"Evaluation Coverage: "
    f"{summary['evaluation_coverage']:.2%}"
)

print(
    "\nGround Truth Layers"
)

print(
    "KB Answerable:",
    summary[
        "kb_answerable_count"
    ],
)

print(
    "Evidence Sufficient:",
    summary[
        "evidence_sufficient_count"
    ],
)

print(
    "KB Answerable but Evidence Insufficient:",
    summary[
        "kb_answerable_but_evidence_insufficient"
    ],
)

print(
    "\nConfusion Matrix "
    "(Ground Truth = evidence_sufficient)"
)

print(
    "TP:",
    summary[
        "tp"
    ],
)

print(
    "TN:",
    summary[
        "tn"
    ],
)

print(
    "FP:",
    summary[
        "fp"
    ],
)

print(
    "FN:",
    summary[
        "fn"
    ],
)

print(
    "\nMetrics"
)

print(
    f"Accuracy: "
    f"{summary['accuracy']:.2%}"
)

print(
    f"Precision: "
    f"{summary['precision']:.2%}"
)

print(
    f"Recall: "
    f"{summary['recall']:.2%}"
)

print(
    f"Specificity: "
    f"{summary['specificity']:.2%}"
)

print(
    f"F1: "
    f"{summary['f1']:.2%}"
)

if summary[
    "error_types"
]:
    print(
        "\nInfrastructure Errors:"
    )

    for error_type, count in sorted(
        summary[
            "error_types"
        ].items()
    ):
        print(
            f"{error_type}: "
            f"{count}"
        )


# ============================================================
# Failure Analysis
# ============================================================

false_positive_cases = []
false_negative_cases = []

for record in records:
    if record.get(
        "status"
    ) not in VALID_VERDICTS:
        continue

    actual = (
        record.get(
            "evidence_sufficient"
        )
        is True
    )

    predicted = (
        record.get(
            "status"
        )
        == "SUFFICIENT"
    )

    if predicted and not actual:
        false_positive_cases.append(
            record
        )

    elif not predicted and actual:
        false_negative_cases.append(
            record
        )


print(
    "\n\n"
)
print_separator()
print(
    "FALSE POSITIVE ANALYSIS "
    "(Verifier 放行了实际不充分 Evidence)"
)
print_separator()

if not false_positive_cases:
    print(
        "\n没有 False Positive。"
    )

else:
    for case in false_positive_cases:
        print_case_detail(
            case
        )


print(
    "\n\n"
)
print_separator()
print(
    "FALSE NEGATIVE ANALYSIS "
    "(Verifier 拒绝了实际充分 Evidence)"
)
print_separator()

if not false_negative_cases:
    print(
        "\n没有 False Negative。"
    )

else:
    for case in false_negative_cases:
        print_case_detail(
            case
        )


# ============================================================
# Save Final JSON
# ============================================================

output_data = {
    "config": {
        "input_path": str(
            INPUT_PATH
        ),
        "labels_path": str(
            LABELS_PATH
        ),
        "verifier_model":
            VERIFIER_MODEL,
        "verifier_timeout_seconds":
            VERIFIER_TIMEOUT_SECONDS,
        "verifier_max_retries":
            VERIFIER_MAX_RETRIES,
        "expected_evidence_k":
            EXPECTED_EVIDENCE_K,
        "ground_truth_field":
            "evidence_sufficient",
    },
    "source_config":
        payload.get(
            "config",
            {},
        ),
    "summary":
        summary,
    "records":
        records,
}

write_json(
    OUTPUT_PATH,
    output_data,
)

if CHECKPOINT_PATH.exists():
    try:
        CHECKPOINT_PATH.unlink()
    except OSError:
        pass

print(
    "\n结果已经保存到："
)

print(
    OUTPUT_PATH
)
