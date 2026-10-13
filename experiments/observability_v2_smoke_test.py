from unittest.mock import patch

from src.customer_support.service import (
    CustomerSupportService,
)
from src.pipeline.rag_pipeline import (
    RAGPipeline,
)


# ============================================================
# RAG Pipeline Fakes
# ============================================================


class FakeRetriever:

    def retrieve(
            self,
            query,
            candidate_k,
            final_k,
    ):
        return [
            {
                "document": {
                    "source": "fake.md",
                    "section": "fake",
                    "text": "fake evidence",
                },
                "rerank_score": 0.9,
            }
        ]


class FakeVerifier:

    def verify(
            self,
            question,
            evidences,
    ):
        return {
            "verdict": "SUFFICIENT",
            "reason": "fake",
        }


class FakeGenerator:

    def generate(
            self,
            question,
            evidences,
    ):
        return "fake answer"


# ============================================================
# Customer Support Fakes
# ============================================================


class FakeMemoryService:

    def __init__(self):
        self.conversation = {
            "conversation_id": "C-OBS-001",
            "user_id": "U001",
            "active_order_id": None,
            "last_intent": None,
        }

    def get_conversation(
            self,
            conversation_id,
            user_id,
    ):
        return dict(
            self.conversation
        )

    def start_conversation(
            self,
            user_id,
    ):
        return dict(
            self.conversation
        )

    def get_recent_messages(
            self,
            conversation_id,
            user_id,
            limit,
    ):
        return []

    def save_user_message(
            self,
            **kwargs,
    ):
        return None

    def save_assistant_message(
            self,
            **kwargs,
    ):
        return None

    def remember_intent(
            self,
            **kwargs,
    ):
        return None

    def remember_order(
            self,
            **kwargs,
    ):
        return None


class FakeHistorySelector:
    candidate_limit = 6

    def select(
            self,
            **kwargs,
    ):
        return {
            "messages": [],
            "candidate_count": 0,
            "selected_count": 0,
            "char_count": 0,
            "used_relevance_selection": False,
        }


class FakeQueryRewriter:

    def rewrite(
            self,
            question,
            **kwargs,
    ):
        return {
            "rewritten_question": question,
            "resolved_entities": {
                "order_id": None,
            },
            "used_history": False,
        }


class FakeContextResolver:

    def extract_order_id(
            self,
            question,
    ):
        return None

    def resolve(
            self,
            question,
            active_order_id,
    ):
        return {
            "original_question": question,
            "resolved_question": question,
            "used_memory": False,
            "active_order_id": (
                active_order_id
            ),
        }


class FakeIntentRouter:

    def route(
            self,
            question,
    ):
        return {
            "intent": "ORDER_QUERY",
            "reason": "fake",
        }


class FakeOrderAgent:

    def ask(
            self,
            question,
            user_id,
    ):
        return {
            "answer": "fake order answer",
            "tools_called": [
                "get_order_detail"
            ],
            "tool_results": [],
            "observability": {
                "latency_tool": 1.25,
                "latency_composer": 2.5,
                "tools_called": [
                    "get_order_detail"
                ],
                "verifier_retry_count": 0,
                "abstained": False,
                "latency_unit": "ms",
            },
        }


# ============================================================
# Tests
# ============================================================


def test_rag_observability():
    pipeline = RAGPipeline(
        retriever=FakeRetriever(),
        verifier=FakeVerifier(),
        generator=FakeGenerator(),
    )

    result = pipeline.ask(
        "fake question"
    )

    observability = result[
        "observability"
    ]

    assert (
            observability[
                "latency_retrieval"
            ]
            >= 0
    )

    assert (
            observability[
                "latency_verifier"
            ]
            >= 0
    )

    assert (
            observability[
                "latency_composer"
            ]
            >= 0
    )

    assert (
            observability[
                "verifier_retry_count"
            ]
            == 0
    )

    assert (
            observability[
                "abstained"
            ]
            is False
    )

    print(
        "[PASS] RAG component timing"
    )


def test_customer_support_trace():
    service = CustomerSupportService(
        intent_router=FakeIntentRouter(),
        order_agent=FakeOrderAgent(),
        memory_service=FakeMemoryService(),
        context_resolver=(
            FakeContextResolver()
        ),
        query_rewriter=(
            FakeQueryRewriter()
        ),
        history_selector=(
            FakeHistorySelector()
        ),
    )

    with patch(
            "src.customer_support.service.log_event"
    ) as mocked_log_event:
        result = service.ask(
            question="帮我查订单",
            user_id="U001",
            conversation_id=(
                "C-OBS-001"
            ),
        )

    assert (
            result["handler"]
            == "ORDER_AGENT"
    )

    assert (
            mocked_log_event.call_count
            == 1
    )

    call_args = (
        mocked_log_event.call_args
    )

    assert (
            call_args.args[1]
            == "customer_support_trace"
    )

    trace = call_args.kwargs

    assert (
            trace["conversation_id"]
            == "C-OBS-001"
    )

    assert (
            trace["intent"]
            == "ORDER_QUERY"
    )

    assert (
            trace["tools_called"]
            == [
                "get_order_detail"
            ]
    )

    assert (
            trace["latency_tool"]
            == 1.25
    )

    assert (
            trace["latency_composer"]
            == 2.5
    )

    assert (
            trace["verifier_retry_count"]
            == 0
    )

    assert (
            trace["abstained"]
            is False
    )

    assert (
            trace["latency_total"]
            >= 0
    )

    assert (
            trace["latency_router"]
            >= 0
    )

    print(
        "[PASS] Customer support trace"
    )


def main():
    print(
        "=" * 70
    )
    print(
        "Observability V2 Smoke Test"
    )
    print(
        "=" * 70
    )

    test_rag_observability()
    test_customer_support_trace()

    print()
    print(
        "=" * 70
    )
    print(
        "Observability V2 Smoke Test: PASS"
    )
    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()