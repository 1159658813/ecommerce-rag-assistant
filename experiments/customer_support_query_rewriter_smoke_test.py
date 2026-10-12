from src.customer_support.service import (
    CustomerSupportService,
)
from src.memory import ContextResolver


class FakeMemoryService:

    def __init__(self):
        self.conversation = {
            "conversation_id": "C001",
            "user_id": "U001",
            "active_order_id": "ORD202610002",
            "last_intent": "ORDER_QUERY",
        }

        self.messages = [
            {
                "role": "user",
                "content": (
                    "帮我查一下 ORD202610002"
                ),
            },
            {
                "role": "assistant",
                "content": (
                    "已查询订单 ORD202610002。"
                ),
            },
        ]

    def get_conversation(
            self,
            conversation_id,
            user_id,
    ):
        if (
                conversation_id
                == self.conversation[
            "conversation_id"
        ]
                and user_id
                == self.conversation[
            "user_id"
        ]
        ):
            return dict(
                self.conversation
            )

        return None

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
            limit=10,
    ):
        return self.messages[-limit:]

    def save_user_message(
            self,
            conversation_id,
            user_id,
            content,
    ):
        return {
            "role": "user",
            "content": content,
        }

    def save_assistant_message(
            self,
            conversation_id,
            user_id,
            content,
    ):
        return {
            "role": "assistant",
            "content": content,
        }

    def remember_intent(
            self,
            conversation_id,
            user_id,
            intent,
    ):
        self.conversation[
            "last_intent"
        ] = intent

    def remember_order(
            self,
            conversation_id,
            user_id,
            order_id,
    ):
        self.conversation[
            "active_order_id"
        ] = order_id


class FakeIntentRouter:

    def route(
            self,
            question,
    ):
        return {
            "intent": "ORDER_QUERY",
            "reason": (
                "query rewrite smoke test"
            ),
        }


class FakeOrderAgent:

    def __init__(self):
        self.last_question = None

    def ask(
            self,
            question,
            user_id,
    ):
        self.last_question = question

        return {
            "answer": "Fake order answer",
            "tool_results": [],
        }


class FakeQueryRewriter:

    def rewrite(
            self,
            question,
            messages=None,
            active_order_id=None,
            last_intent=None,
    ):
        return {
            "rewritten_question": (
                "订单 ORD202610002 "
                "现在到哪了？"
            ),
            "resolved_entities": {
                "order_id": (
                    "ORD202610002"
                ),
            },
            "used_history": True,
        }


class FailingQueryRewriter:

    def rewrite(
            self,
            question,
            messages=None,
            active_order_id=None,
            last_intent=None,
    ):
        raise RuntimeError(
            "fake rewriter failure"
        )


def run_rewrite_success_case():
    memory_service = (
        FakeMemoryService()
    )

    order_agent = (
        FakeOrderAgent()
    )

    service = CustomerSupportService(
        intent_router=FakeIntentRouter(),
        order_agent=order_agent,
        memory_service=memory_service,
        context_resolver=ContextResolver(),
        query_rewriter=FakeQueryRewriter(),
    )

    result = service.ask(
        question="它现在到哪了？",
        user_id="U001",
        conversation_id="C001",
    )

    expected_question = (
        "订单 ORD202610002 "
        "现在到哪了？"
    )

    assert (
            result["resolved_question"]
            == expected_question
    )

    assert (
            order_agent.last_question
            == expected_question
    )

    assert (
            result["memory_used"]
            is True
    )

    assert (
            result["intent"]
            == "ORDER_QUERY"
    )

    print()
    print(
        "Rewrite Integration: PASS"
    )


def run_fallback_case():
    memory_service = (
        FakeMemoryService()
    )

    order_agent = (
        FakeOrderAgent()
    )

    service = CustomerSupportService(
        intent_router=FakeIntentRouter(),
        order_agent=order_agent,
        memory_service=memory_service,
        context_resolver=ContextResolver(),
        query_rewriter=(
            FailingQueryRewriter()
        ),
    )

    result = service.ask(
        question="它现在到哪了？",
        user_id="U001",
        conversation_id="C001",
    )

    resolved_question = (
        result["resolved_question"]
    )

    assert (
            "ORD202610002"
            in resolved_question
    )

    assert (
            "它现在到哪了？"
            in resolved_question
    )

    assert (
            result["memory_used"]
            is True
    )

    assert (
            order_agent.last_question
            == resolved_question
    )

    print(
        "Context Resolver Fallback: PASS"
    )


def main():
    print(
        "=" * 70
    )
    print(
        "Customer Support Query Rewrite "
        "Smoke Test"
    )
    print(
        "=" * 70
    )

    run_rewrite_success_case()

    run_fallback_case()

    print()
    print(
        "=" * 70
    )
    print(
        "Customer Support Query Rewrite "
        "Smoke Test: PASS"
    )
    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()