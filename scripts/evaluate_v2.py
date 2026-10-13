from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

# ============================================================
# Project Root
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )

from src.agents.order_agent import OrderAgent
from src.customer_support.mixed_query_service import (
    MixedQueryService,
)
from src.memory import (
    HistorySelector,
    QueryRewriter,
)
from src.orders.init_db import init_database
from src.orders.service import OrderService
from src.routing.intent_router import (
    IntentRouter,
)
from src.service.factory import build_pipeline
from src.service.rag_service import RAGService

# ============================================================
# Paths
# ============================================================

DATASET_DIR = (
        PROJECT_ROOT
        / "evaluation"
        / "datasets"
        / "v2"
)

OUTPUT_DIR = (
        PROJECT_ROOT
        / "evaluation"
        / "results"
        / "v2"
)

DATASET_PATHS = {
    "intent": (
            DATASET_DIR
            / "intent_cases_v1.json"
    ),
    "tool": (
            DATASET_DIR
            / "tool_cases_v1.json"
    ),
    "memory": (
            DATASET_DIR
            / "memory_cases_v1.json"
    ),
    "mixed": (
            DATASET_DIR
            / "mixed_cases_v1.json"
    ),
    "security": (
            DATASET_DIR
            / "security_cases_v1.json"
    ),
}

JSON_OUTPUT_PATH = (
        OUTPUT_DIR
        / "evaluation_v2_results.json"
)

CSV_OUTPUT_PATH = (
        OUTPUT_DIR
        / "evaluation_v2_cases.csv"
)

MARKDOWN_OUTPUT_PATH = (
        OUTPUT_DIR
        / "evaluation_v2_report.md"
)

VALID_SUITES = (
    "intent",
    "tool",
    "memory",
    "mixed",
    "security",
)

VALID_INTENTS = (
    "KNOWLEDGE_QA",
    "ORDER_QUERY",
    "MIXED_QUERY",
    "GENERAL_CHAT",
    "UNSUPPORTED",
)


# ============================================================
# Generic Helpers
# ============================================================

def safe_divide(
        numerator: int | float,
        denominator: int | float,
) -> float:
    if denominator == 0:
        return 0.0

    return numerator / denominator


def load_json(
        path: Path,
) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"找不到 Evaluation Dataset：{path}"
        )

    with path.open(
            "r",
            encoding="utf-8",
    ) as file:
        data = json.load(file)

    if (
            not isinstance(data, list)
            or not data
    ):
        raise ValueError(
            f"{path.name} 必须是非空 JSON 数组。"
        )

    seen_ids = set()

    for item in data:

        if not isinstance(
                item,
                dict,
        ):
            raise ValueError(
                f"{path.name} 中存在非对象 case。"
            )

        case_id = str(
            item.get(
                "id",
                "",
            )
        ).strip()

        if not case_id:
            raise ValueError(
                f"{path.name} 中存在缺少 id 的 case。"
            )

        if case_id in seen_ids:
            raise ValueError(
                f"{path.name} 存在重复 ID："
                f"{case_id}"
            )

        seen_ids.add(
            case_id
        )

    return data


def select_cases(
        records: list[dict],
        case_ids: set[str],
        max_cases: int,
) -> list[dict]:
    selected = records

    if case_ids:
        selected = [
            item
            for item in selected
            if item["id"] in case_ids
        ]

    if max_cases > 0:
        selected = selected[
                   :max_cases
                   ]

    return selected


def normalize_optional_string(
        value,
) -> str | None:
    if value is None:
        return None

    value = str(
        value
    ).strip()

    if not value:
        return None

    return value


def normalize_order_id(
        value,
) -> str | None:
    value = normalize_optional_string(
        value
    )

    if value is None:
        return None

    return value.upper()


def make_error_record(
        suite: str,
        case_id: str,
        exc: Exception,
) -> dict:
    return {
        "suite": suite,
        "id": case_id,
        "status": "ERROR",
        "passed": False,
        "error": repr(exc),
    }


def build_basic_summary(
        records: list[dict],
) -> dict:
    total = len(records)

    evaluated = [
        item
        for item in records
        if item.get("status") == "OK"
    ]

    errors = [
        item
        for item in records
        if item.get("status") != "OK"
    ]

    passed = [
        item
        for item in evaluated
        if item.get("passed") is True
    ]

    return {
        "total": total,
        "evaluated": len(
            evaluated
        ),
        "errors": len(
            errors
        ),
        "passed": len(
            passed
        ),
        "evaluation_coverage": (
            safe_divide(
                len(evaluated),
                total,
            )
        ),
        "pass_rate": (
            safe_divide(
                len(passed),
                len(evaluated),
            )
        ),
        "error_ids": [
            item["id"]
            for item in errors
        ],
    }


# ============================================================
# Intent Evaluation
# ============================================================

def evaluate_intent(
        cases: list[dict],
) -> dict:
    print()
    print("=" * 78)
    print("Intent Routing Evaluation")
    print("=" * 78)

    router = IntentRouter()

    records = []

    confusion_matrix = {
        expected: {
            predicted: 0
            for predicted
            in VALID_INTENTS
        }
        for expected
        in VALID_INTENTS
    }

    for index, case in enumerate(
            cases,
            start=1,
    ):
        case_id = case["id"]

        print(
            f"[Intent {index}/{len(cases)}] "
            f"{case_id}"
        )

        try:
            expected = str(
                case[
                    "expected_intent"
                ]
            ).strip().upper()

            result = router.route(
                case["question"]
            )

            predicted = str(
                result.get(
                    "intent",
                    "",
                )
            ).strip().upper()

            correct = (
                    predicted
                    == expected
            )

            if (
                    expected
                    in confusion_matrix
                    and predicted
                    in confusion_matrix[
                expected
            ]
            ):
                confusion_matrix[
                    expected
                ][
                    predicted
                ] += 1

            records.append(
                {
                    "suite": "intent",
                    "id": case_id,
                    "status": "OK",
                    "passed": correct,
                    "question": (
                        case["question"]
                    ),
                    "expected_intent": (
                        expected
                    ),
                    "actual_intent": (
                        predicted
                    ),
                    "reason": result.get(
                        "reason"
                    ),
                }
            )

            print(
                "  Expected:",
                expected,
            )
            print(
                "  Actual  :",
                predicted,
            )
            print(
                "  Result  :",
                "PASS"
                if correct
                else "FAIL",
            )

        except Exception as exc:

            records.append(
                make_error_record(
                    suite="intent",
                    case_id=case_id,
                    exc=exc,
                )
            )

            print(
                "  ERROR:",
                repr(exc),
            )

    summary = build_basic_summary(
        records
    )

    evaluated = [
        item
        for item in records
        if item.get("status")
           == "OK"
    ]

    correct = sum(
        item["passed"] is True
        for item in evaluated
    )

    summary.update(
        {
            "intent_accuracy": (
                safe_divide(
                    correct,
                    len(evaluated),
                )
            ),
            "confusion_matrix": (
                confusion_matrix
            ),
        }
    )

    return {
        "summary": summary,
        "records": records,
    }


# ============================================================
# Tool Calling Evaluation
# ============================================================

def evaluate_tool(
        cases: list[dict],
) -> dict:
    print()
    print("=" * 78)
    print("Tool Calling Evaluation")
    print("=" * 78)

    agent = OrderAgent()

    records = []

    for index, case in enumerate(
            cases,
            start=1,
    ):
        case_id = case["id"]

        print(
            f"[Tool {index}/{len(cases)}] "
            f"{case_id}"
        )

        try:
            expected_tool = str(
                case[
                    "expected_tool"
                ]
            ).strip()

            expected_arguments = (
                    case.get(
                        "expected_arguments",
                        {},
                    )
                    or {}
            )

            result = agent.ask(
                question=case[
                    "question"
                ],
                user_id=case[
                    "user_id"
                ],
            )

            tools_called = (
                    result.get(
                        "tools_called",
                        [],
                    )
                    or []
            )

            tool_results = (
                    result.get(
                        "tool_results",
                        [],
                    )
                    or []
            )

            # Dataset 中每条问题只期望一个工具，
            # 因此多调用额外工具也记为 Tool Selection 错误。
            selection_correct = (
                    tools_called
                    == [
                        expected_tool
                    ]
            )

            actual_arguments = None

            for tool_result in tool_results:

                if (
                        tool_result.get(
                            "tool_name"
                        )
                        == expected_tool
                ):
                    actual_arguments = (
                            tool_result.get(
                                "arguments",
                                {},
                            )
                            or {}
                    )
                    break

            argument_correct = (
                    actual_arguments
                    == expected_arguments
            )

            passed = (
                    selection_correct
                    and argument_correct
            )

            records.append(
                {
                    "suite": "tool",
                    "id": case_id,
                    "status": "OK",
                    "passed": passed,
                    "question": (
                        case["question"]
                    ),
                    "expected_tool": (
                        expected_tool
                    ),
                    "actual_tools": (
                        tools_called
                    ),
                    "tool_selection_correct": (
                        selection_correct
                    ),
                    "expected_arguments": (
                        expected_arguments
                    ),
                    "actual_arguments": (
                        actual_arguments
                    ),
                    "argument_correct": (
                        argument_correct
                    ),
                }
            )

            print(
                "  Expected tool:",
                expected_tool,
            )
            print(
                "  Actual tools :",
                tools_called,
            )
            print(
                "  Arguments OK :",
                argument_correct,
            )
            print(
                "  Result       :",
                "PASS"
                if passed
                else "FAIL",
            )

        except Exception as exc:

            records.append(
                make_error_record(
                    suite="tool",
                    case_id=case_id,
                    exc=exc,
                )
            )

            print(
                "  ERROR:",
                repr(exc),
            )

    summary = build_basic_summary(
        records
    )

    evaluated = [
        item
        for item in records
        if item.get("status")
           == "OK"
    ]

    selection_correct = sum(
        item[
            "tool_selection_correct"
        ]
        is True
        for item in evaluated
    )

    argument_correct = sum(
        item[
            "argument_correct"
        ]
        is True
        for item in evaluated
    )

    summary.update(
        {
            "tool_selection_accuracy": (
                safe_divide(
                    selection_correct,
                    len(evaluated),
                )
            ),
            "argument_accuracy": (
                safe_divide(
                    argument_correct,
                    len(evaluated),
                )
            ),
        }
    )

    return {
        "summary": summary,
        "records": records,
    }


# ============================================================
# Memory Evaluation
# ============================================================

def evaluate_memory(
        cases: list[dict],
) -> dict:
    print()
    print("=" * 78)
    print("Conversation Memory Evaluation")
    print("=" * 78)

    selector = HistorySelector()

    rewriter = QueryRewriter()

    records = []

    for index, case in enumerate(
            cases,
            start=1,
    ):
        case_id = case["id"]

        print(
            f"[Memory {index}/{len(cases)}] "
            f"{case_id}"
        )

        try:
            history_result = (
                selector.select(
                    question=case[
                        "question"
                    ],
                    messages=case.get(
                        "messages",
                        [],
                    ),
                    active_order_id=(
                        case.get(
                            "active_order_id"
                        )
                    ),
                    last_intent=(
                        case.get(
                            "last_intent"
                        )
                    ),
                )
            )

            rewrite_result = (
                rewriter.rewrite(
                    question=case[
                        "question"
                    ],
                    messages=(
                        history_result[
                            "messages"
                        ]
                    ),
                    active_order_id=(
                        case.get(
                            "active_order_id"
                        )
                    ),
                    last_intent=(
                        case.get(
                            "last_intent"
                        )
                    ),
                )
            )

            expected_order_id = (
                normalize_order_id(
                    case.get(
                        "expected_order_id"
                    )
                )
            )

            actual_order_id = (
                normalize_order_id(
                    rewrite_result
                    .get(
                        "resolved_entities",
                        {},
                    )
                    .get(
                        "order_id"
                    )
                )
            )

            expected_used_history = bool(
                case.get(
                    "expected_used_history",
                    False,
                )
            )

            actual_used_history = bool(
                rewrite_result.get(
                    "used_history",
                    False,
                )
            )

            resolution_correct = (
                    actual_order_id
                    == expected_order_id
            )

            history_usage_correct = (
                    actual_used_history
                    == expected_used_history
            )

            # Wrong-memory：
            # 模型从上下文中恢复出了错误实体。
            #
            # 如果 expected=None，
            # 却产生了任意订单号，也算 Wrong-memory。
            wrong_memory = (
                    actual_order_id
                    is not None
                    and actual_order_id
                    != expected_order_id
            )

            passed = (
                    resolution_correct
                    and history_usage_correct
                    and not wrong_memory
            )

            records.append(
                {
                    "suite": "memory",
                    "id": case_id,
                    "status": "OK",
                    "passed": passed,
                    "question": (
                        case["question"]
                    ),
                    "expected_order_id": (
                        expected_order_id
                    ),
                    "actual_order_id": (
                        actual_order_id
                    ),
                    "context_resolution_correct": (
                        resolution_correct
                    ),
                    "expected_used_history": (
                        expected_used_history
                    ),
                    "actual_used_history": (
                        actual_used_history
                    ),
                    "history_usage_correct": (
                        history_usage_correct
                    ),
                    "wrong_memory": (
                        wrong_memory
                    ),
                    "rewritten_question": (
                        rewrite_result.get(
                            "rewritten_question"
                        )
                    ),
                    "history_candidate_count": (
                        history_result.get(
                            "candidate_count"
                        )
                    ),
                    "history_selected_count": (
                        history_result.get(
                            "selected_count"
                        )
                    ),
                    "history_char_count": (
                        history_result.get(
                            "char_count"
                        )
                    ),
                    "used_relevance_selection": (
                        history_result.get(
                            "used_relevance_selection"
                        )
                    ),
                }
            )

            print(
                "  Expected order:",
                expected_order_id,
            )
            print(
                "  Actual order  :",
                actual_order_id,
            )
            print(
                "  Used history  :",
                actual_used_history,
            )
            print(
                "  Wrong memory  :",
                wrong_memory,
            )
            print(
                "  Result        :",
                "PASS"
                if passed
                else "FAIL",
            )

        except Exception as exc:

            records.append(
                make_error_record(
                    suite="memory",
                    case_id=case_id,
                    exc=exc,
                )
            )

            print(
                "  ERROR:",
                repr(exc),
            )

    summary = build_basic_summary(
        records
    )

    evaluated = [
        item
        for item in records
        if item.get("status")
           == "OK"
    ]

    resolution_correct = sum(
        item[
            "context_resolution_correct"
        ]
        is True
        for item in evaluated
    )

    history_usage_correct = sum(
        item[
            "history_usage_correct"
        ]
        is True
        for item in evaluated
    )

    wrong_memory = sum(
        item[
            "wrong_memory"
        ]
        is True
        for item in evaluated
    )

    summary.update(
        {
            "context_resolution_accuracy": (
                safe_divide(
                    resolution_correct,
                    len(evaluated),
                )
            ),
            "history_usage_accuracy": (
                safe_divide(
                    history_usage_correct,
                    len(evaluated),
                )
            ),
            "wrong_memory_rate": (
                safe_divide(
                    wrong_memory,
                    len(evaluated),
                )
            ),
        }
    )

    return {
        "summary": summary,
        "records": records,
    }


# ============================================================
# Security Evaluation
# ============================================================

def evaluate_security(
        cases: list[dict],
) -> dict:
    print()
    print("=" * 78)
    print("Security Evaluation")
    print("=" * 78)

    service = OrderService()

    records = []

    for index, case in enumerate(
            cases,
            start=1,
    ):
        case_id = case["id"]

        print(
            f"[Security {index}/{len(cases)}] "
            f"{case_id}"
        )

        try:
            operation = str(
                case[
                    "operation"
                ]
            ).strip()

            method = getattr(
                service,
                operation,
                None,
            )

            if method is None:
                raise ValueError(
                    f"未知 Security operation："
                    f"{operation}"
                )

            arguments = dict(
                case.get(
                    "arguments",
                    {},
                )
                or {}
            )

            result = method(
                user_id=case[
                    "user_id"
                ],
                **arguments,
            )

            expected_success = bool(
                case[
                    "expected_success"
                ]
            )

            expected_code = (
                normalize_optional_string(
                    case.get(
                        "expected_code"
                    )
                )
            )

            actual_success = bool(
                result.get(
                    "success",
                    False,
                )
            )

            actual_code = (
                normalize_optional_string(
                    result.get(
                        "code"
                    )
                )
            )

            block_correct = (
                    actual_success
                    == expected_success
                    and actual_code
                    == expected_code
            )

            records.append(
                {
                    "suite": "security",
                    "id": case_id,
                    "status": "OK",
                    "passed": block_correct,
                    "operation": operation,
                    "user_id": case[
                        "user_id"
                    ],
                    "arguments": arguments,
                    "expected_success": (
                        expected_success
                    ),
                    "actual_success": (
                        actual_success
                    ),
                    "expected_code": (
                        expected_code
                    ),
                    "actual_code": (
                        actual_code
                    ),
                    "unauthorized_access_blocked": (
                        block_correct
                    ),
                }
            )

            print(
                "  Operation:",
                operation,
            )
            print(
                "  Code     :",
                actual_code,
            )
            print(
                "  Result   :",
                "PASS"
                if block_correct
                else "FAIL",
            )

        except Exception as exc:

            records.append(
                make_error_record(
                    suite="security",
                    case_id=case_id,
                    exc=exc,
                )
            )

            print(
                "  ERROR:",
                repr(exc),
            )

    summary = build_basic_summary(
        records
    )

    evaluated = [
        item
        for item in records
        if item.get("status")
           == "OK"
    ]

    blocked = sum(
        item[
            "unauthorized_access_blocked"
        ]
        is True
        for item in evaluated
    )

    summary.update(
        {
            "unauthorized_access_block_rate": (
                safe_divide(
                    blocked,
                    len(evaluated),
                )
            ),
        }
    )

    return {
        "summary": summary,
        "records": records,
    }


# ============================================================
# Mixed Query Evaluation
# ============================================================

def build_mixed_query_service():
    """
    Build the real product RAG + Mixed Query stack.

    It is intentionally constructed only when the mixed
    suite is requested because model loading is expensive.
    """

    pipeline = build_pipeline()

    rag_service = RAGService(
        pipeline=pipeline
    )

    return MixedQueryService(
        rag_service=rag_service
    )


def evaluate_mixed(
        cases: list[dict],
) -> dict:
    print()
    print("=" * 78)
    print("Mixed Query Evaluation")
    print("=" * 78)

    service = (
        build_mixed_query_service()
    )

    records = []

    for index, case in enumerate(
            cases,
            start=1,
    ):
        case_id = case["id"]

        print(
            f"[Mixed {index}/{len(cases)}] "
            f"{case_id}"
        )

        try:
            result = service.answer(
                question=case[
                    "question"
                ],
                user_id=case[
                    "user_id"
                ],
                order_id=case.get(
                    "order_id"
                ),
            )

            order_result = (
                result.get(
                    "order_result"
                )
            )

            actual_order_success = bool(
                order_result
                and order_result.get(
                    "success"
                )
            )

            expected_order_success = bool(
                case[
                    "expected_order_success"
                ]
            )

            expected_order_status = (
                normalize_optional_string(
                    case.get(
                        "expected_order_status"
                    )
                )
            )

            actual_order_id = None
            actual_order_status = None

            if actual_order_success:
                order_data = (
                    order_result
                    .get(
                        "data",
                        {},
                    )
                    .get(
                        "order",
                        {},
                    )
                )

                actual_order_id = (
                    normalize_order_id(
                        order_data.get(
                            "order_id"
                        )
                    )
                )

                actual_order_status = (
                    normalize_optional_string(
                        order_data.get(
                            "status"
                        )
                    )
                )

            if expected_order_success:

                order_fact_correct = (
                        actual_order_success
                        and actual_order_id
                        == normalize_order_id(
                    case.get(
                        "order_id"
                    )
                )
                )

                if (
                        expected_order_status
                        is not None
                ):
                    order_fact_correct = (
                            order_fact_correct
                            and actual_order_status
                            == expected_order_status
                    )

            else:
                order_fact_correct = (
                    not actual_order_success
                )

            expected_verdict = (
                normalize_optional_string(
                    case.get(
                        "expected_verdict"
                    )
                )
            )

            actual_verdict = (
                normalize_optional_string(
                    result.get(
                        "verdict"
                    )
                )
            )

            evidence_sufficiency_correct = (
                    actual_verdict
                    == expected_verdict
            )

            combined = bool(
                result.get(
                    "combined",
                    False,
                )
            )

            abstained = bool(
                result.get(
                    "abstained",
                    False,
                )
            )

            evidences = (
                    result.get(
                        "evidences",
                        [],
                    )
                    or []
            )

            rag_result = result.get(
                "rag_result"
            )

            # ------------------------------------------------
            # Policy grounding guard
            #
            # 1. Unauthorized / missing order:
            #    policy RAG should not run.
            #
            # 2. SUFFICIENT:
            #    verified evidence exists and composition runs.
            #
            # 3. INSUFFICIENT:
            #    composition must be blocked.
            # ------------------------------------------------

            if not expected_order_success:

                policy_grounding_correct = (
                        combined is False
                        and rag_result is None
                )

            elif expected_verdict == "SUFFICIENT":

                policy_grounding_correct = (
                        actual_verdict
                        == "SUFFICIENT"
                        and combined is True
                        and abstained is False
                        and len(evidences) > 0
                )

            elif expected_verdict == "INSUFFICIENT":

                policy_grounding_correct = (
                        actual_verdict
                        == "INSUFFICIENT"
                        and combined is False
                        and abstained is True
                )

            else:
                policy_grounding_correct = (
                        actual_verdict is None
                )

            answer = str(
                result.get(
                    "answer",
                    "",
                )
                or ""
            ).strip()

            if not expected_order_success:

                combined_answer_success = (
                        bool(answer)
                        and combined is False
                        and rag_result is None
                )

            elif expected_verdict == "SUFFICIENT":

                combined_answer_success = (
                        bool(answer)
                        and combined is True
                        and abstained is False
                )

            elif expected_verdict == "INSUFFICIENT":

                combined_answer_success = (
                        bool(answer)
                        and combined is False
                        and abstained is True
                )

            else:
                combined_answer_success = (
                    bool(answer)
                )

            passed = (
                    order_fact_correct
                    and evidence_sufficiency_correct
                    and policy_grounding_correct
                    and combined_answer_success
            )

            records.append(
                {
                    "suite": "mixed",
                    "id": case_id,
                    "status": "OK",
                    "passed": passed,
                    "question": (
                        case["question"]
                    ),
                    "user_id": (
                        case["user_id"]
                    ),
                    "order_id": (
                        case.get(
                            "order_id"
                        )
                    ),
                    "expected_order_success": (
                        expected_order_success
                    ),
                    "actual_order_success": (
                        actual_order_success
                    ),
                    "expected_order_status": (
                        expected_order_status
                    ),
                    "actual_order_status": (
                        actual_order_status
                    ),
                    "order_fact_correct": (
                        order_fact_correct
                    ),
                    "expected_verdict": (
                        expected_verdict
                    ),
                    "actual_verdict": (
                        actual_verdict
                    ),
                    "evidence_sufficiency_correct": (
                        evidence_sufficiency_correct
                    ),
                    "policy_grounding_correct": (
                        policy_grounding_correct
                    ),
                    "combined": combined,
                    "abstained": abstained,
                    "evidence_count": len(
                        evidences
                    ),
                    "combined_answer_success": (
                        combined_answer_success
                    ),
                }
            )

            print(
                "  Order Fact :",
                order_fact_correct,
            )
            print(
                "  Verdict    :",
                actual_verdict,
            )
            print(
                "  Grounding  :",
                policy_grounding_correct,
            )
            print(
                "  Combined   :",
                combined_answer_success,
            )
            print(
                "  Result     :",
                "PASS"
                if passed
                else "FAIL",
            )

        except Exception as exc:

            records.append(
                make_error_record(
                    suite="mixed",
                    case_id=case_id,
                    exc=exc,
                )
            )

            print(
                "  ERROR:",
                repr(exc),
            )

    summary = build_basic_summary(
        records
    )

    evaluated = [
        item
        for item in records
        if item.get("status")
           == "OK"
    ]

    order_fact_correct = sum(
        item[
            "order_fact_correct"
        ]
        is True
        for item in evaluated
    )

    policy_grounding_correct = sum(
        item[
            "policy_grounding_correct"
        ]
        is True
        for item in evaluated
    )

    combined_success = sum(
        item[
            "combined_answer_success"
        ]
        is True
        for item in evaluated
    )

    evidence_records = [
        item
        for item in evaluated
        if item.get(
            "expected_verdict"
        )
           is not None
    ]

    evidence_correct = sum(
        item[
            "evidence_sufficiency_correct"
        ]
        is True
        for item in evidence_records
    )

    summary.update(
        {
            "order_fact_correctness": (
                safe_divide(
                    order_fact_correct,
                    len(evaluated),
                )
            ),
            "policy_grounding": (
                safe_divide(
                    policy_grounding_correct,
                    len(evaluated),
                )
            ),
            "evidence_sufficiency_accuracy": (
                safe_divide(
                    evidence_correct,
                    len(evidence_records),
                )
            ),
            "combined_answer_success_rate": (
                safe_divide(
                    combined_success,
                    len(evaluated),
                )
            ),
        }
    )

    return {
        "summary": summary,
        "records": records,
    }


# ============================================================
# Output
# ============================================================

def write_json_report(
        payload: dict,
) -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with JSON_OUTPUT_PATH.open(
            "w",
            encoding="utf-8",
    ) as file:
        json.dump(
            payload,
            file,
            ensure_ascii=False,
            indent=2,
        )


def write_csv_report(
        records: list[dict],
) -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with CSV_OUTPUT_PATH.open(
            "w",
            encoding="utf-8-sig",
            newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "suite",
                "id",
                "status",
                "passed",
                "error",
                "details_json",
            ],
        )

        writer.writeheader()

        for record in records:
            details = {
                key: value
                for key, value
                in record.items()
                if key
                   not in {
                       "suite",
                       "id",
                       "status",
                       "passed",
                       "error",
                   }
            }

            writer.writerow(
                {
                    "suite": record.get(
                        "suite"
                    ),
                    "id": record.get(
                        "id"
                    ),
                    "status": record.get(
                        "status"
                    ),
                    "passed": record.get(
                        "passed"
                    ),
                    "error": record.get(
                        "error"
                    ),
                    "details_json": (
                        json.dumps(
                            details,
                            ensure_ascii=False,
                        )
                    ),
                }
            )


def percent(
        value,
) -> str:
    if value is None:
        return "-"

    return (
        f"{float(value) * 100:.2f}%"
    )


def markdown_metric_table(
        metrics: list[
            tuple[str, Any]
        ],
) -> list[str]:
    lines = [
        "| Metric | Value |",
        "|---|---:|",
    ]

    for name, value in metrics:

        if (
                isinstance(
                    value,
                    float,
                )
                and 0.0
                <= value
                <= 1.0
        ):
            rendered = percent(
                value
            )
        else:
            rendered = str(
                value
            )

        lines.append(
            f"| {name} | {rendered} |"
        )

    return lines


def write_markdown_report(
        payload: dict,
) -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    lines = [
        "# Ecommerce RAG Assistant — Evaluation V2",
        "",
        f"- Generated at: `{payload['generated_at']}`",
        f"- Suites: `{', '.join(payload['suites'])}`",
        f"- Max cases per suite: `{payload['max_cases']}`",
        "",
        "## Overall",
        "",
    ]

    overall = payload[
        "overall"
    ]

    lines.extend(
        markdown_metric_table(
            [
                (
                    "Total Cases",
                    overall["total"],
                ),
                (
                    "Evaluated",
                    overall["evaluated"],
                ),
                (
                    "Errors",
                    overall["errors"],
                ),
                (
                    "Passed",
                    overall["passed"],
                ),
                (
                    "Evaluation Coverage",
                    overall[
                        "evaluation_coverage"
                    ],
                ),
                (
                    "Overall Pass Rate",
                    overall[
                        "pass_rate"
                    ],
                ),
            ]
        )
    )

    for suite_name, suite_data in (
            payload["results"].items()
    ):

        summary = suite_data[
            "summary"
        ]

        lines.extend(
            [
                "",
                f"## {suite_name.title()}",
                "",
            ]
        )

        common_metrics = [
            (
                "Total",
                summary.get(
                    "total",
                    0,
                ),
            ),
            (
                "Evaluated",
                summary.get(
                    "evaluated",
                    0,
                ),
            ),
            (
                "Errors",
                summary.get(
                    "errors",
                    0,
                ),
            ),
            (
                "Pass Rate",
                summary.get(
                    "pass_rate",
                    0.0,
                ),
            ),
        ]

        suite_metrics = []

        if suite_name == "intent":
            suite_metrics.extend(
                [
                    (
                        "Intent Accuracy",
                        summary.get(
                            "intent_accuracy"
                        ),
                    ),
                ]
            )

        elif suite_name == "tool":
            suite_metrics.extend(
                [
                    (
                        "Tool Selection Accuracy",
                        summary.get(
                            "tool_selection_accuracy"
                        ),
                    ),
                    (
                        "Argument Accuracy",
                        summary.get(
                            "argument_accuracy"
                        ),
                    ),
                ]
            )

        elif suite_name == "memory":
            suite_metrics.extend(
                [
                    (
                        "Context Resolution Accuracy",
                        summary.get(
                            "context_resolution_accuracy"
                        ),
                    ),
                    (
                        "History Usage Accuracy",
                        summary.get(
                            "history_usage_accuracy"
                        ),
                    ),
                    (
                        "Wrong-memory Rate",
                        summary.get(
                            "wrong_memory_rate"
                        ),
                    ),
                ]
            )

        elif suite_name == "security":
            suite_metrics.extend(
                [
                    (
                        "Unauthorized Access Block Rate",
                        summary.get(
                            "unauthorized_access_block_rate"
                        ),
                    ),
                ]
            )

        elif suite_name == "mixed":
            suite_metrics.extend(
                [
                    (
                        "Order Fact Correctness",
                        summary.get(
                            "order_fact_correctness"
                        ),
                    ),
                    (
                        "Policy Grounding",
                        summary.get(
                            "policy_grounding"
                        ),
                    ),
                    (
                        "Evidence Sufficiency Accuracy",
                        summary.get(
                            "evidence_sufficiency_accuracy"
                        ),
                    ),
                    (
                        "Combined Answer Success Rate",
                        summary.get(
                            "combined_answer_success_rate"
                        ),
                    ),
                ]
            )

        lines.extend(
            markdown_metric_table(
                common_metrics
                + suite_metrics
            )
        )

        # Intent Confusion Matrix
        if suite_name == "intent":

            matrix = summary.get(
                "confusion_matrix",
                {},
            )

            lines.extend(
                [
                    "",
                    "### Confusion Matrix",
                    "",
                    "| Expected \\ Predicted | "
                    + " | ".join(
                        VALID_INTENTS
                    )
                    + " |",
                    "|---|"
                    + "---:|"
                    * len(
                        VALID_INTENTS
                    ),
                ]
            )

            for expected in VALID_INTENTS:
                row = matrix.get(
                    expected,
                    {},
                )

                values = [
                    str(
                        row.get(
                            predicted,
                            0,
                        )
                    )
                    for predicted
                    in VALID_INTENTS
                ]

                lines.append(
                    f"| {expected} | "
                    + " | ".join(
                        values
                    )
                    + " |"
                )

    lines.extend(
        [
            "",
            "## Failed / Error Cases",
            "",
        ]
    )

    failed_records = [
        item
        for item in payload[
            "records"
        ]
        if (
                item.get(
                    "passed"
                )
                is not True
                or item.get(
            "status"
        )
                != "OK"
        )
    ]

    if not failed_records:

        lines.append(
            "None."
        )

    else:
        lines.extend(
            [
                "| Suite | Case | Status |",
                "|---|---|---|",
            ]
        )

        for record in failed_records:
            lines.append(
                "| "
                f"{record.get('suite')} | "
                f"{record.get('id')} | "
                f"{record.get('status')} |"
            )

    with MARKDOWN_OUTPUT_PATH.open(
            "w",
            encoding="utf-8",
    ) as file:

        file.write(
            "\n".join(
                lines
            )
        )


# ============================================================
# Overall Summary
# ============================================================

def build_overall_summary(
        records: list[dict],
) -> dict:
    return build_basic_summary(
        records
    )


# ============================================================
# Main
# ============================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluation V2 for Ecommerce RAG Assistant."
        )
    )

    parser.add_argument(
        "--suite",
        choices=(
            *VALID_SUITES,
            "all",
        ),
        default="all",
        help=(
            "Evaluation suite to run."
        ),
    )

    parser.add_argument(
        "--max-cases",
        type=int,
        default=0,
        help=(
            "Maximum cases per suite. "
            "0 means run all selected cases."
        ),
    )

    parser.add_argument(
        "--case-ids",
        default="",
        help=(
            "Comma-separated case IDs."
        ),
    )

    args = parser.parse_args()

    if args.max_cases < 0:
        raise ValueError(
            "--max-cases 不能小于 0。"
        )

    requested_case_ids = {
        item.strip()
        for item
        in args.case_ids.split(",")
        if item.strip()
    }

    if args.suite == "all":
        suites = list(
            VALID_SUITES
        )
    else:
        suites = [
            args.suite
        ]

    print()
    print("=" * 78)
    print(
        "Ecommerce RAG Assistant - Evaluation V2"
    )
    print("=" * 78)
    print(
        "Suites     :",
        ", ".join(
            suites
        ),
    )
    print(
        "Max Cases  :",
        args.max_cases
        if args.max_cases > 0
        else "ALL",
    )

    datasets = {}

    all_available_ids = set()

    for suite in suites:
        dataset = load_json(
            DATASET_PATHS[
                suite
            ]
        )

        datasets[
            suite
        ] = dataset

        all_available_ids.update(
            item["id"]
            for item in dataset
        )

    if requested_case_ids:

        missing_ids = (
                requested_case_ids
                - all_available_ids
        )

        if missing_ids:
            raise ValueError(
                "指定的 case ID 不存在："
                + ", ".join(
                    sorted(
                        missing_ids
                    )
                )
            )

    selected_datasets = {}

    for suite in suites:

        selected = select_cases(
            records=datasets[
                suite
            ],
            case_ids=(
                requested_case_ids
            ),
            max_cases=(
                args.max_cases
            ),
        )

        if selected:
            selected_datasets[
                suite
            ] = selected

    if not selected_datasets:
        raise ValueError(
            "没有选中任何 Evaluation Case。"
        )

    # Tool / Security / Mixed 都依赖本地结构化数据。
    if any(
            suite
            in {
                "tool",
                "security",
                "mixed",
            }
            for suite
            in selected_datasets
    ):
        init_database()

    suite_runners = {
        "intent": evaluate_intent,
        "tool": evaluate_tool,
        "memory": evaluate_memory,
        "mixed": evaluate_mixed,
        "security": evaluate_security,
    }

    total_start = (
        time.perf_counter()
    )

    results = {}
    all_records = []

    for suite in suites:

        cases = selected_datasets.get(
            suite
        )

        if not cases:
            continue

        suite_start = (
            time.perf_counter()
        )

        suite_result = (
            suite_runners[
                suite
            ](
                cases
            )
        )

        suite_result[
            "elapsed_seconds"
        ] = (
                time.perf_counter()
                - suite_start
        )

        results[
            suite
        ] = suite_result

        all_records.extend(
            suite_result[
                "records"
            ]
        )

    elapsed = (
            time.perf_counter()
            - total_start
    )

    overall = (
        build_overall_summary(
            all_records
        )
    )

    payload = {
        "version": "v2",
        "generated_at": (
            datetime.now()
            .astimezone()
            .isoformat()
        ),
        "suites": list(
            results.keys()
        ),
        "max_cases": (
            args.max_cases
        ),
        "case_ids": sorted(
            requested_case_ids
        ),
        "elapsed_seconds": elapsed,
        "overall": overall,
        "results": results,
        "records": all_records,
    }

    write_json_report(
        payload
    )

    write_csv_report(
        all_records
    )

    write_markdown_report(
        payload
    )

    print()
    print("=" * 78)
    print("Evaluation V2 Summary")
    print("=" * 78)
    print(
        "Total      :",
        overall["total"],
    )
    print(
        "Evaluated  :",
        overall["evaluated"],
    )
    print(
        "Passed     :",
        overall["passed"],
    )
    print(
        "Errors     :",
        overall["errors"],
    )
    print(
        "Pass Rate  :",
        percent(
            overall[
                "pass_rate"
            ]
        ),
    )
    print(
        "Elapsed    :",
        f"{elapsed:.2f}s",
    )

    print()
    print(
        "JSON Report:",
        JSON_OUTPUT_PATH,
    )
    print(
        "CSV Report :",
        CSV_OUTPUT_PATH,
    )
    print(
        "MD Report  :",
        MARKDOWN_OUTPUT_PATH,
    )

    print("=" * 78)

    # Evaluation 本身允许出现模型 Case FAIL，
    # 只有基础设施错误才返回非 0。
    if overall["errors"] > 0:
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )