import sys
from pathlib import Path

from fastapi.testclient import TestClient

# ============================================================
# Project Root
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )

from src.api import app


# ============================================================
# Fake Customer Support Service
# ============================================================

class FakeCustomerSupportService:
    """
    Lightweight fake service for API contract testing.

    No real LLM, RAG, tool calling, or database query is executed.
    """

    CONVERSATION_ID = (
        "smoke-conversation-001"
    )

    def ask(
            self,
            question,
            user_id,
            conversation_id=None,
    ):
        # ----------------------------------------------------
        # First turn
        # ----------------------------------------------------

        if conversation_id is None:
            return {
                "conversation_id":
                    self.CONVERSATION_ID,

                "user_id":
                    user_id,

                "question":
                    question,

                "resolved_question":
                    question,

                "memory_used":
                    False,

                "intent":
                    "ORDER_QUERY",

                "intent_reason":
                    "用户正在查询具体订单。",

                "handler":
                    "ORDER_AGENT",

                "answer":
                    "已查询到您的订单信息。",

                "result": {
                    "answer":
                        "已查询到您的订单信息。",

                    "tools_called": [
                        "get_order_detail"
                    ],

                    "tool_results": [],
                },
            }

        # ----------------------------------------------------
        # Second turn
        # ----------------------------------------------------

        return {
            "conversation_id":
                conversation_id,

            "user_id":
                user_id,

            "question":
                question,

            "resolved_question": (
                "当前正在讨论的订单号是 "
                "ORD202610002。"
                "用户当前问题："
                f"{question}"
            ),

            "memory_used":
                True,

            "intent":
                "ORDER_QUERY",

            "intent_reason":
                "结合会话记忆后，用户正在查询订单物流。",

            "handler":
                "ORDER_AGENT",

            "answer":
                "您的订单正在运输中。",

            "result": {
                "answer":
                    "您的订单正在运输中。",

                "tools_called": [
                    "get_logistics"
                ],

                "tool_results": [],
            },
        }


# ============================================================
# Helpers
# ============================================================

def assert_equal(
        actual,
        expected,
        message,
):
    if actual != expected:
        raise AssertionError(
            f"{message}: "
            f"expected={expected!r}, "
            f"actual={actual!r}"
        )


def require_request_id(response):
    request_id = response.headers.get(
        "X-Request-ID"
    )

    if not request_id:
        raise AssertionError(
            "response missing X-Request-ID"
        )


# ============================================================
# Smoke Test
# ============================================================

print("=" * 80)
print("FastAPI V2 Chat Smoke Test")
print("=" * 80)

with TestClient(app) as client:
    original_service = (
        client
        .app
        .state
        .customer_support_service
    )

    client.app.state.customer_support_service = (
        FakeCustomerSupportService()
    )

    try:

        # ====================================================
        # Turn 1
        # ====================================================

        response = client.post(
            "/api/v2/chat",
            json={
                "question": (
                    "帮我查一下 "
                    "ORD202610002 "
                    "这个订单"
                ),
                "user_id": "U001",
            },
        )

        require_request_id(response)

        assert_equal(
            response.status_code,
            200,
            "turn 1 status",
        )

        first = response.json()

        assert_equal(
            first["conversation_id"],
            "smoke-conversation-001",
            "turn 1 conversation id",
        )

        assert_equal(
            first["intent"],
            "ORDER_QUERY",
            "turn 1 intent",
        )

        assert_equal(
            first["handler"],
            "ORDER_AGENT",
            "turn 1 handler",
        )

        assert_equal(
            first["memory_used"],
            False,
            "turn 1 memory flag",
        )

        assert_equal(
            first["result"][
                "tools_called"
            ],
            [
                "get_order_detail"
            ],
            "turn 1 tool",
        )

        print(
            "[PASS] POST /api/v2/chat "
            "first turn"
        )

        # ====================================================
        # Turn 2
        # ====================================================

        conversation_id = first[
            "conversation_id"
        ]

        response = client.post(
            "/api/v2/chat",
            json={
                "question":
                    "它现在到哪了？",

                "user_id":
                    "U001",

                "conversation_id":
                    conversation_id,
            },
        )

        require_request_id(response)

        assert_equal(
            response.status_code,
            200,
            "turn 2 status",
        )

        second = response.json()

        assert_equal(
            second["conversation_id"],
            conversation_id,
            "turn 2 conversation reuse",
        )

        assert_equal(
            second["memory_used"],
            True,
            "turn 2 memory flag",
        )

        assert_equal(
            second["intent"],
            "ORDER_QUERY",
            "turn 2 intent",
        )

        assert_equal(
            second["handler"],
            "ORDER_AGENT",
            "turn 2 handler",
        )

        assert_equal(
            second["result"][
                "tools_called"
            ],
            [
                "get_logistics"
            ],
            "turn 2 tool",
        )

        if (
                "ORD202610002"
                not in second[
            "resolved_question"
        ]
        ):
            raise AssertionError(
                "resolved question does not "
                "contain remembered order id"
            )

        print(
            "[PASS] POST /api/v2/chat "
            "conversation reuse"
        )

    finally:

        client.app.state.customer_support_service = (
            original_service
        )

print()
print("=" * 80)
print("V2 Chat API Smoke Test PASSED")
print("=" * 80)