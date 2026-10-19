"""P1-6 fault injection: no network, no real LLM calls, no database writes."""
from types import SimpleNamespace
from unittest.mock import Mock

from src.agents.order_agent import OrderAgent
from src.customer_support.mixed_query_service import MixedQueryService
from src.orders.tools import OrderToolExecutor
from src.pipeline.rag_pipeline import RAGPipeline
from src.reliability import RetryPolicy
from src.verification.answerability_verifier_v1_2 import (
    AnswerabilityVerifierV12,
    VerifierExhausted,
)


def make_pipeline(verifier, generator):
    return RAGPipeline(
        retriever=SimpleNamespace(retrieve=lambda **_: [
            {"document": {"source": "test", "section": "test", "text": "verified evidence"}}
        ]),
        verifier=verifier,
        generator=generator,
    )


def verify_retry():
    verifier = AnswerabilityVerifierV12.__new__(AnswerabilityVerifierV12)
    verifier.max_retries = 3
    verifier.retry_base_seconds = 0
    calls = []

    def once(**kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise TimeoutError("simulated timeout")
        return {"verdict": "SUFFICIENT", "covered_facts": [], "missing_facts": []}

    verifier._call_once = once
    result = verifier.verify("question", [{"content": "something"}])
    assert result["retry_count"] == 1

    verifier._call_once = lambda **_: (_ for _ in ()).throw(TimeoutError())
    try:
        verifier.verify("question", [{"content": "something"}])
    except VerifierExhausted as exc:
        assert exc.retry_count == 2
    else:
        raise AssertionError("Expected verifier exhaustion")
    print("[PASS] Verifier retry count and exhaustion")


def verify_rag_fallback():
    failing = SimpleNamespace(verify=lambda **_: (_ for _ in ()).throw(TimeoutError()))
    generator = SimpleNamespace(generate=lambda **_: "grounded answer")
    result = make_pipeline(failing, generator).ask("question")
    assert result["abstained"] and result["abstain_reason"] == "verifier_unavailable"
    sufficient = SimpleNamespace(verify=lambda **_: {"verdict": "SUFFICIENT", "retry_count": 1})
    bad_generator = SimpleNamespace(generate=lambda **_: (_ for _ in ()).throw(TimeoutError()))
    result = make_pipeline(sufficient, bad_generator).ask("question")
    assert result["abstained"] and result["abstain_reason"] == "composer_unavailable"
    assert result["observability"]["verifier_retry_count"] == 1
    print("[PASS] RAG verifier/composer fail closed")


def verify_tool_guards():
    executor = OrderToolExecutor.__new__(OrderToolExecutor)
    executor.service = Mock()
    for payload in [[], {"order_id": 123}, {"order_id": "ORD1", "user_id": "U002"}]:
        result = executor.execute("get_order_detail", payload, user_id="U001")
        assert result["code"] == "INVALID_ARGUMENT"
    executor.service.get_order_detail.assert_not_called()
    print("[PASS] Tool argument types / identity injection blocked")


class ToolCall:
    def __init__(self, args):
        self.id = "call-1"
        self.function = SimpleNamespace(name="get_order_detail", arguments=args)


class Message:
    def __init__(self, tool_calls=None):
        self.tool_calls = tool_calls
        self.content = None

    def model_dump(self, **kwargs):
        return {"role": "assistant", "tool_calls": []}


def verify_order_fallback():
    agent = OrderAgent.__new__(OrderAgent)
    agent.model = "fake"
    executor = Mock()
    agent.tool_executor = executor
    agent.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        create=Mock(side_effect=TimeoutError("fake")),
    )))
    result = agent.ask("ORD202610002 到哪了？", "U001")
    assert result["abstained"] and result["tools_called"] == []
    executor.execute.assert_not_called()

    first = SimpleNamespace(choices=[SimpleNamespace(message=Message([ToolCall("[oops")]))])
    create = Mock(side_effect=[first])
    agent.client.chat.completions.create = create
    executor.execute = Mock(return_value={"success": False, "code": "INVALID_ARGUMENT", "message": "参数错误"})
    result = agent.ask("查询订单", "U001")
    assert result["abstained"] and result["tool_results"][0]["result"]["code"] == "INVALID_ARGUMENT"
    assert create.call_count == 1
    executor.execute.assert_called_once()
    assert executor.execute.call_args.kwargs["user_id"] == "U001"

    first = SimpleNamespace(choices=[SimpleNamespace(message=Message([ToolCall('{"order_id":"ORD202610002"}')]))])
    create = Mock(side_effect=[first, TimeoutError("final answer timeout")])
    agent.client.chat.completions.create = create
    executor.execute = Mock(return_value={"success": True, "data": {
        "order": {"order_id": "ORD202610002", "status": "SHIPPED"}
    }})
    result = agent.ask("查询订单", "U001")
    assert result["abstained"] and "ORD202610002" in result["answer"]
    assert "SHIPPED" in result["answer"] and result["tool_results"]
    print("[PASS] Order agent initial/tool/final failure fallback")


def verify_mixed_fallback():
    mixed = MixedQueryService.__new__(MixedQueryService)
    mixed.order_service = SimpleNamespace(get_order_detail=lambda **_: {
        "success": True, "data": {"order": {"order_id": "ORD202610002", "status": "SHIPPED"}}
    })
    mixed.rag_service = SimpleNamespace(ask=lambda *_: {
        "verdict": "SUFFICIENT", "abstained": False, "evidences": [{"content": "rule"}],
        "answer": "policy", "observability": {"verifier_retry_count": 1},
    })
    mixed._compose_answer = lambda **_: (_ for _ in ()).throw(TimeoutError())
    result = mixed.answer("能退货吗", "U001", "ORD202610002")
    assert result["abstained"] and not result["combined"]
    assert result["abstain_reason"] == "composer_unavailable"
    assert "SHIPPED" in result["answer"]
    assert result["observability"]["verifier_retry_count"] == 1
    print("[PASS] Mixed composer safe fallback")


def verify_router_fallback():
    from src.routing.intent_router import IntentRouter
    router = IntentRouter.__new__(IntentRouter)
    router.model = "fake"
    router.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        create=Mock(side_effect=TimeoutError()),
    )))
    routed = router.route("我的订单在哪里？")
    assert routed["intent"] == "UNSUPPORTED" and routed.get("fallback") is True
    print("[PASS] Router timeout conservative fallback")


def main():
    assert RetryPolicy(max_attempts=2).run(lambda: "ok", (TimeoutError,)) == "ok"
    verify_retry()
    verify_rag_fallback()
    verify_tool_guards()
    verify_order_fallback()
    verify_mixed_fallback()
    verify_router_fallback()
    print("Reliability V2 Smoke Test: PASS")


if __name__ == "__main__":
    main()
