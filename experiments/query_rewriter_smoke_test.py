from src.memory import QueryRewriter


def print_result(
        title: str,
        result: dict,
):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)
    print(
        "Rewritten Question :",
        result["rewritten_question"],
    )
    print(
        "Order ID           :",
        result["resolved_entities"].get(
            "order_id"
        ),
    )
    print(
        "Used History       :",
        result["used_history"],
    )


def main():
    rewriter = QueryRewriter()

    # ========================================================
    # Case 1
    # 完整问题，不应强行依赖历史
    # ========================================================

    result_1 = rewriter.rewrite(
        question="退款一般多久到账？",
        messages=[],
        active_order_id=None,
        last_intent=None,
    )

    print_result(
        "Case 1: Standalone Question",
        result_1,
    )

    assert (
        result_1["rewritten_question"]
    )

    assert (
            result_1["used_history"]
            is False
    )

    # ========================================================
    # Case 2
    # 通过结构化 active_order_id 消除“它”的指代
    # ========================================================

    result_2 = rewriter.rewrite(
        question="它现在到哪了？",
        messages=[
            {
                "role": "user",
                "content": (
                    "帮我查一下 "
                    "ORD202610002"
                ),
            },
            {
                "role": "assistant",
                "content": (
                    "已查询订单 "
                    "ORD202610002。"
                ),
            },
        ],
        active_order_id=(
            "ORD202610002"
        ),
        last_intent=(
            "ORDER_QUERY"
        ),
    )

    print_result(
        "Case 2: Active Order Reference",
        result_2,
    )

    assert (
            "ORD202610002"
            in result_2[
                "rewritten_question"
            ]
    )

    assert (
            result_2[
                "resolved_entities"
            ].get(
                "order_id"
            )
            == "ORD202610002"
    )

    # ========================================================
    # Case 3
    # 真正利用历史解析“第二个订单”
    # ========================================================

    result_3 = rewriter.rewrite(
        question="第二个订单现在到哪了？",
        messages=[
            {
                "role": "user",
                "content": "我的订单有哪些？",
            },
            {
                "role": "assistant",
                "content": (
                    "你目前有两个订单："
                    "ORD202610001 和 "
                    "ORD202610002。"
                ),
            },
        ],
        active_order_id=None,
        last_intent=(
            "ORDER_QUERY"
        ),
    )

    print_result(
        "Case 3: History Resolution",
        result_3,
    )

    assert (
            "ORD202610002"
            in result_3[
                "rewritten_question"
            ]
    )

    assert (
            result_3[
                "resolved_entities"
            ].get(
                "order_id"
            )
            == "ORD202610002"
    )

    assert (
            result_3["used_history"]
            is True
    )

    print()
    print("=" * 70)
    print(
        "Query Rewriter Smoke Test: PASS"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()