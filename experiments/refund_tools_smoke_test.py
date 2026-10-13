from src.agents.order_agent import OrderAgent
from src.orders.init_db import init_database
from src.orders.service import OrderService
from src.orders.tools import OrderToolExecutor


def main():
    print(
        "=" * 70
    )
    print(
        "Refund Tools Smoke Test"
    )
    print(
        "=" * 70
    )

    # ========================================================
    # Prepare database
    # ========================================================

    init_database()

    service = OrderService()

    # ========================================================
    # Case 1:
    # Refund status by owned order
    # ========================================================

    result_1 = service.get_refund_status(
        user_id="U001",
        order_id="ORD202610001",
    )

    assert (
            result_1["success"]
            is True
    )

    assert (
            result_1["data"]["refund_id"]
            == "REF202610001"
    )

    assert (
            result_1["data"]["status"]
            == "PROCESSING"
    )

    print(
        "[PASS] Refund status"
    )

    # ========================================================
    # Case 2:
    # Refund detail
    # ========================================================

    result_2 = service.get_refund_detail(
        user_id="U001",
        refund_id="REF202610001",
    )

    assert (
            result_2["success"]
            is True
    )

    assert (
            result_2["data"]["order_id"]
            == "ORD202610001"
    )

    assert (
            "user_id"
            not in result_2["data"]
    )

    print(
        "[PASS] Refund detail"
    )

    # ========================================================
    # Case 3:
    # Owned order without refund
    # ========================================================

    result_3 = service.get_refund_status(
        user_id="U001",
        order_id="ORD202610002",
    )

    assert (
            result_3["success"]
            is False
    )

    assert (
            result_3["code"]
            == "REFUND_NOT_FOUND"
    )

    print(
        "[PASS] No refund record"
    )

    # ========================================================
    # Case 4:
    # Unauthorized refund access
    # ========================================================

    result_4 = service.get_refund_detail(
        user_id="U001",
        refund_id="REF202610002",
    )

    assert (
            result_4["success"]
            is False
    )

    assert (
            result_4["code"]
            == "REFUND_NOT_FOUND"
    )

    print(
        "[PASS] Unauthorized refund blocked"
    )

    # ========================================================
    # Case 5:
    # Tool Executor
    # ========================================================

    executor = OrderToolExecutor(
        service=service
    )

    result_5 = executor.execute(
        tool_name="get_refund_status",
        arguments={
            "order_id": "ORD202610001"
        },
        user_id="U001",
    )

    assert (
            result_5["success"]
            is True
    )

    print(
        "[PASS] Refund tool executor"
    )

    # ========================================================
    # Case 6:
    # Native LLM Tool Calling
    # ========================================================

    agent = OrderAgent(
        tool_executor=executor
    )

    result_6 = agent.ask(
        question=(
            "ORD202610001 "
            "的退款进度怎么样？"
        ),
        user_id="U001",
    )

    assert (
            "get_refund_status"
            in result_6[
                "tools_called"
            ]
    )

    assert (
        result_6["answer"]
    )

    print(
        "[PASS] Refund native tool calling"
    )

    print()
    print(
        "Tools Called:",
        result_6[
            "tools_called"
        ],
    )

    print(
        "Answer:",
        result_6["answer"],
    )

    print()
    print(
        "=" * 70
    )
    print(
        "Refund Tools Smoke Test: PASS"
    )
    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()