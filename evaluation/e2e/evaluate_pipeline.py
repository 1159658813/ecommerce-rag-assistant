import json
import os
import sys
from pathlib import Path


# ============================================================
# Project Root
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.service import build_pipeline


# ============================================================
# Paths
# ============================================================

QUESTIONS_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "datasets"
    / "generation_questions_v1.json"
)

LABELS_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "datasets"
    / "evidence_sufficiency_labels_v1.json"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "results"
    / "e2e"
    / "e2e_pipeline_results_v1.json"
)

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "results"
    / "e2e"
    / "e2e_pipeline_results_v1.checkpoint.json"
)


# ============================================================
# Config
# ============================================================

# 默认全部运行。
#
# 开发阶段可以：
#
#   $env:E2E_MAX_CASES="5"
#
# 只跑前 5 条。
E2E_MAX_CASES = int(
    os.getenv(
        "E2E_MAX_CASES",
        "0",
    )
)

# 也支持指定 case：
#
# $env:E2E_CASE_IDS="b2_q01,b2_q31,b2_q49"
E2E_CASE_IDS = {
    item.strip()
    for item in os.getenv(
        "E2E_CASE_IDS",
        "",
    ).split(",")
    if item.strip()
}


VALID_VERDICTS = {
    "SUFFICIENT",
    "INSUFFICIENT",
}


# ============================================================
# Helpers
# ============================================================

def safe_divide(
    numerator,
    denominator,
):
    if denominator == 0:
        return 0.0

    return numerator / denominator


def load_json(path):
    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


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


# ============================================================
# Load + Validate Dataset
# ============================================================

def load_and_join_dataset():

    if not QUESTIONS_PATH.exists():
        raise FileNotFoundError(
            f"找不到 Questions：{QUESTIONS_PATH}"
        )

    if not LABELS_PATH.exists():
        raise FileNotFoundError(
            f"找不到 Evidence Sufficiency Labels：{LABELS_PATH}"
        )

    questions = load_json(
        QUESTIONS_PATH
    )

    labels = load_json(
        LABELS_PATH
    )

    if not isinstance(
        questions,
        list,
    ) or not questions:
        raise ValueError(
            "generation_questions_v1.json "
            "必须是非空数组。"
        )

    if not isinstance(
        labels,
        list,
    ) or not labels:
        raise ValueError(
            "evidence_sufficiency_labels_v1.json "
            "必须是非空数组。"
        )

    label_map = {}

    for item in labels:

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
            raise ValueError(
                "Evidence Sufficiency Label 缺少 id。"
            )

        if question_id in label_map:
            raise ValueError(
                f"Label ID 重复：{question_id}"
            )

        if not isinstance(
            evidence_sufficient,
            bool,
        ):
            raise ValueError(
                f"{question_id}: "
                "evidence_sufficient 必须是 bool。"
            )

        label_map[
            question_id
        ] = evidence_sufficient

    joined = []

    seen_ids = set()

    for item in questions:

        question_id = str(
            item.get(
                "id",
                "",
            )
        ).strip()

        question = str(
            item.get(
                "question",
                "",
            )
        ).strip()

        if not question_id:
            raise ValueError(
                "Question 缺少 id。"
            )

        if question_id in seen_ids:
            raise ValueError(
                f"Question ID 重复：{question_id}"
            )

        seen_ids.add(
            question_id
        )

        if not question:
            raise ValueError(
                f"{question_id}: question 为空。"
            )

        if question_id not in label_map:
            raise ValueError(
                f"{question_id}: "
                "缺少 evidence_sufficient 标签。"
            )

        record = dict(
            item
        )

        record[
            "evidence_sufficient"
        ] = label_map[
            question_id
        ]

        joined.append(
            record
        )

    extra_label_ids = (
        set(
            label_map.keys()
        )
        - seen_ids
    )

    if extra_label_ids:
        raise ValueError(
            "存在 Question 中没有的 Label ID："
            + ", ".join(
                sorted(
                    extra_label_ids
                )
            )
        )

    return joined


# ============================================================
# Case Filter
# ============================================================

def select_cases(records):

    selected = records

    if E2E_CASE_IDS:
        selected = [
            item
            for item in selected
            if item["id"] in E2E_CASE_IDS
        ]

        missing_ids = (
            E2E_CASE_IDS
            - {
                item["id"]
                for item in selected
            }
        )

        if missing_ids:
            raise ValueError(
                "指定的 E2E_CASE_IDS 不存在："
                + ", ".join(
                    sorted(
                        missing_ids
                    )
                )
            )

    if E2E_MAX_CASES > 0:
        selected = selected[
            :E2E_MAX_CASES
        ]

    return selected


# ============================================================
# Retrieval Diagnostics
# ============================================================

def evaluate_retrieval_hit(
    expected_sections,
    evidences,
):

    if not expected_sections:
        return None

    retrieved_sections = {
        evidence.get(
            "section"
        )
        for evidence in evidences
        if evidence.get(
            "section"
        )
    }

    return any(
        section in retrieved_sections
        for section in expected_sections
    )


# ============================================================
# Literal Diagnostics
#
# 只用于观察，不作为最终语义判分。
# ============================================================

def run_literal_diagnostics(
    answer,
    must_include,
    must_not_include,
):

    answer = str(
        answer or ""
    )

    missing_items = [
        item
        for item in must_include
        if item not in answer
    ]

    forbidden_items = [
        item
        for item in must_not_include
        if item in answer
    ]

    return (
        missing_items,
        forbidden_items,
    )


# ============================================================
# Metrics
# ============================================================

def build_summary(
    records,
    total,
):

    completed = [
        record
        for record in records
        if record.get(
            "status"
        ) == "OK"
    ]

    errors = [
        record
        for record in records
        if record.get(
            "status"
        ) != "OK"
    ]

    tp = tn = fp = fn = 0

    routing_correct = 0
    pipeline_consistent = 0

    retrieval_records = []
    retrieval_hits = 0

    unexpected_generation = []
    missed_generation = []

    for record in completed:

        actual = (
            record[
                "evidence_sufficient"
            ]
            is True
        )

        predicted = (
            record[
                "verifier_verdict"
            ]
            == "SUFFICIENT"
        )

        if actual and predicted:
            tp += 1

        elif not actual and not predicted:
            tn += 1

        elif not actual and predicted:
            fp += 1

            unexpected_generation.append(
                record["id"]
            )

        else:
            fn += 1

            missed_generation.append(
                record["id"]
            )

        if record[
            "route_correct"
        ]:
            routing_correct += 1

        if record[
            "pipeline_routing_consistent"
        ]:
            pipeline_consistent += 1

        if record[
            "retrieval_hit"
        ] is not None:

            retrieval_records.append(
                record
            )

            if record[
                "retrieval_hit"
            ]:
                retrieval_hits += 1

    evaluated = len(
        completed
    )

    accuracy = safe_divide(
        tp + tn,
        evaluated,
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

    return {
        "total":
            total,

        "evaluated":
            evaluated,

        "errors":
            len(
                errors
            ),

        "evaluation_coverage":
            safe_divide(
                evaluated,
                total,
            ),

        "verifier": {
            "tp":
                tp,

            "tn":
                tn,

            "fp":
                fp,

            "fn":
                fn,

            "accuracy":
                accuracy,

            "precision":
                precision,

            "recall":
                recall,

            "specificity":
                specificity,

            "f1":
                f1,
        },

        "routing": {
            "correct":
                routing_correct,

            "accuracy":
                safe_divide(
                    routing_correct,
                    evaluated,
                ),

            "pipeline_consistent":
                pipeline_consistent,

            "pipeline_consistency_rate":
                safe_divide(
                    pipeline_consistent,
                    evaluated,
                ),

            "unexpected_generation_ids":
                unexpected_generation,

            "missed_generation_ids":
                missed_generation,
        },

        "retrieval": {
            "evaluated":
                len(
                    retrieval_records
                ),

            "hit":
                retrieval_hits,

            "hit_rate":
                safe_divide(
                    retrieval_hits,
                    len(
                        retrieval_records
                    ),
                ),
        },

        "error_ids": [
            record["id"]
            for record in errors
        ],
    }


# ============================================================
# Load Dataset
# ============================================================

dataset = (
    load_and_join_dataset()
)

selected_cases = (
    select_cases(
        dataset
    )
)


print_separator()
print(
    "Full RAG Pipeline E2E Evaluation V1"
)
print_separator()

print(
    "Dataset:",
    QUESTIONS_PATH,
)

print(
    "Evidence Labels:",
    LABELS_PATH,
)

print(
    "Dataset Questions:",
    len(
        dataset
    ),
)

print(
    "Selected Questions:",
    len(
        selected_cases
    ),
)

print(
    "Evidence Sufficient = True:",
    sum(
        item[
            "evidence_sufficient"
        ]
        is True
        for item in selected_cases
    ),
)

print(
    "Evidence Sufficient = False:",
    sum(
        item[
            "evidence_sufficient"
        ]
        is False
        for item in selected_cases
    ),
)


# ============================================================
# Build REAL Product Pipeline
# ============================================================

print(
    "\nBuilding RAGPipeline..."
)

pipeline = (
    build_pipeline()
)

print(
    "RAGPipeline ready."
)


# ============================================================
# Evaluation
# ============================================================

records = []

total = len(
    selected_cases
)


for index, item in enumerate(
    selected_cases,
    start=1,
):

    question_id = item[
        "id"
    ]

    question = item[
        "question"
    ]

    evidence_sufficient = item[
        "evidence_sufficient"
    ]

    expected_sections = item.get(
        "expected_sections",
        [],
    )

    expected_abstained = (
        not evidence_sufficient
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
        "Evidence Sufficient GT:",
        evidence_sufficient,
    )

    print(
        "Expected Route:",
        (
            "ANSWER"
            if evidence_sufficient
            else "ABSTAIN"
        ),
    )

    try:

        result = (
            pipeline.ask(
                question=question
            )
        )

        verifier = result.get(
            "verifier"
        )

        verifier_verdict = (
            None
            if verifier is None
            else verifier.get(
                "verdict"
            )
        )

        if (
            verifier_verdict
            not in VALID_VERDICTS
        ):
            raise ValueError(
                "Pipeline 返回非法 Verifier Verdict："
                f"{verifier_verdict!r}"
            )

        predicted_abstained = bool(
            result.get(
                "abstained"
            )
        )

        evidences = result.get(
            "evidences",
            [],
        )

        answer = str(
            result.get(
                "answer",
                "",
            )
            or ""
        ).strip()

        retrieval_hit = (
            evaluate_retrieval_hit(
                expected_sections=
                    expected_sections,
                evidences=
                    evidences,
            )
        )

        # Ground Truth Route
        route_correct = (
            predicted_abstained
            == expected_abstained
        )

        # Pipeline 是否忠实执行 Verifier
        #
        # SUFFICIENT  → 应回答
        # INSUFFICIENT → 应拒答
        pipeline_routing_consistent = (
            (
                verifier_verdict
                == "SUFFICIENT"
                and predicted_abstained
                is False
            )
            or
            (
                verifier_verdict
                == "INSUFFICIENT"
                and predicted_abstained
                is True
            )
        )

        (
            missing_literal_items,
            forbidden_literal_items,
        ) = run_literal_diagnostics(
            answer=answer,
            must_include=item.get(
                "must_include",
                [],
            ),
            must_not_include=item.get(
                "must_not_include",
                [],
            ),
        )

        record = {
            "id":
                question_id,

            "question":
                question,

            "category":
                item.get(
                    "category"
                ),

            "answerable":
                item.get(
                    "answerable"
                ),

            "evidence_sufficient":
                evidence_sufficient,

            "expected_sections":
                expected_sections,

            "expected_abstained":
                expected_abstained,

            "actual_abstained":
                predicted_abstained,

            "abstain_reason":
                result.get(
                    "abstain_reason"
                ),

            "route_correct":
                route_correct,

            "pipeline_routing_consistent":
                pipeline_routing_consistent,

            "verifier_verdict":
                verifier_verdict,

            "verifier":
                verifier,

            "retrieval_hit":
                retrieval_hit,

            "evidences":
                evidences,

            "answer":
                answer,

            "reference_answer":
                item.get(
                    "reference_answer"
                ),

            "must_include":
                item.get(
                    "must_include",
                    [],
                ),

            "must_not_include":
                item.get(
                    "must_not_include",
                    [],
                ),

            "missing_literal_items":
                missing_literal_items,

            "forbidden_literal_items":
                forbidden_literal_items,

            "status":
                "OK",
        }

        print(
            "Verifier:",
            verifier_verdict,
        )

        print(
            "Abstained:",
            predicted_abstained,
        )

        print(
            "Route Correct:",
            route_correct,
        )

        print(
            "Pipeline Routing Consistent:",
            pipeline_routing_consistent,
        )

        print(
            "Retrieval Hit:",
            retrieval_hit,
        )

    except Exception as error:

        record = {
            "id":
                question_id,

            "question":
                question,

            "evidence_sufficient":
                evidence_sufficient,

            "status":
                "PIPELINE_ERROR",

            "error": {
                "type":
                    type(
                        error
                    ).__name__,

                "message":
                    repr(
                        error
                    ),
            },
        }

        print(
            "PIPELINE ERROR:",
            repr(
                error
            ),
        )

    records.append(
        record
    )

    write_json(
        CHECKPOINT_PATH,
        {
            "records":
                records,
        },
    )


# ============================================================
# Summary
# ============================================================

summary = (
    build_summary(
        records=records,
        total=total,
    )
)


print(
    "\n\n"
)

print_separator()
print(
    "Full Pipeline E2E Result"
)
print_separator()

print(
    "Questions:",
    summary[
        "total"
    ],
)

print(
    "Successfully Evaluated:",
    summary[
        "evaluated"
    ],
)

print(
    "Errors:",
    summary[
        "errors"
    ],
)

print(
    f"Evaluation Coverage: "
    f"{summary['evaluation_coverage']:.2%}"
)


print(
    "\nVerifier / Evidence Sufficiency"
)

print(
    json.dumps(
        summary[
            "verifier"
        ],
        ensure_ascii=False,
        indent=2,
    )
)


print(
    "\nPipeline Routing"
)

print(
    json.dumps(
        summary[
            "routing"
        ],
        ensure_ascii=False,
        indent=2,
    )
)


print(
    "\nRetrieval Diagnostic"
)

print(
    json.dumps(
        summary[
            "retrieval"
        ],
        ensure_ascii=False,
        indent=2,
    )
)


# ============================================================
# Failure Attribution
# ============================================================

print(
    "\n\n"
)

print_separator()
print(
    "Layer Attribution"
)
print_separator()


for record in records:

    if record.get(
        "status"
    ) != "OK":
        continue

    if not record.get(
        "pipeline_routing_consistent"
    ):

        print(
            "\n[PIPELINE ROUTING ERROR]",
            record["id"],
        )

        continue

    if not record.get(
        "route_correct"
    ):

        actual = record[
            "evidence_sufficient"
        ]

        predicted = (
            record[
                "verifier_verdict"
            ]
            == "SUFFICIENT"
        )

        if predicted and not actual:

            print(
                "\n[VERIFIER FALSE POSITIVE]",
                record["id"],
            )

        elif actual and not predicted:

            print(
                "\n[VERIFIER FALSE NEGATIVE]",
                record["id"],
            )


# ============================================================
# Save
# ============================================================

output_data = {
    "config": {
        "questions_path":
            str(
                QUESTIONS_PATH
            ),

        "labels_path":
            str(
                LABELS_PATH
            ),

        "ground_truth":
            "evidence_sufficient",

        "max_cases":
            E2E_MAX_CASES,

        "case_ids":
            sorted(
                E2E_CASE_IDS
            ),
    },

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
    "\nResults saved to:"
)

print(
    OUTPUT_PATH
)