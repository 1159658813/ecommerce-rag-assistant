from src.memory import HistorySelector


def main():
    selector = HistorySelector(
        candidate_limit=12,
        max_messages=6,
        char_budget=2400,
    )

    # ========================================================
    # Case 1:
    # Active order context should be retained.
    # ========================================================

    messages_1 = [
        {
            "role": "user",
            "content": "你好",
        },
        {
            "role": "assistant",
            "content": "您好，有什么可以帮您？",
        },
        {
            "role": "user",
            "content": "帮我查 ORD202610001",
        },
        {
            "role": "assistant",
            "content": "订单 ORD202610001 已签收。",
        },
        {
            "role": "user",
            "content": "退款一般多久到账？",
        },
        {
            "role": "assistant",
            "content": "退款到账时间需要根据退款规则判断。",
        },
        {
            "role": "user",
            "content": "帮我查 ORD202610002",
        },
        {
            "role": "assistant",
            "content": "订单 ORD202610002 已发货。",
        },
        {
            "role": "user",
            "content": "谢谢",
        },
        {
            "role": "assistant",
            "content": "不客气。",
        },
    ]

    result_1 = selector.select(
        question="它现在到哪了？",
        messages=messages_1,
        active_order_id="ORD202610002",
        last_intent="ORDER_QUERY",
    )

    selected_text_1 = " ".join(
        message["content"]
        for message
        in result_1["messages"]
    )

    assert (
            "ORD202610002"
            in selected_text_1
    )

    assert (
            result_1["selected_count"]
            <= 6
    )

    print(
        "[PASS] Active order history"
    )

    # ========================================================
    # Case 2:
    # An older message listing multiple orders should survive
    # selection for "第二个订单".
    # ========================================================

    messages_2 = [
        {
            "role": "user",
            "content": "我的订单有哪些？",
        },
        {
            "role": "assistant",
            "content": (
                "你有两个订单："
                "ORD202610001 和 "
                "ORD202610002。"
            ),
        },
        {
            "role": "user",
            "content": "好的",
        },
        {
            "role": "assistant",
            "content": "好的。",
        },
        {
            "role": "user",
            "content": "退款多久到账？",
        },
        {
            "role": "assistant",
            "content": "需要根据退款规则判断。",
        },
        {
            "role": "user",
            "content": "发票怎么开？",
        },
        {
            "role": "assistant",
            "content": "可以查看发票规则。",
        },
        {
            "role": "user",
            "content": "谢谢",
        },
        {
            "role": "assistant",
            "content": "不客气。",
        },
    ]

    result_2 = selector.select(
        question="第二个订单现在到哪了？",
        messages=messages_2,
        active_order_id=None,
        last_intent="ORDER_QUERY",
    )

    selected_text_2 = " ".join(
        message["content"]
        for message
        in result_2["messages"]
    )

    assert (
            "ORD202610001"
            in selected_text_2
    )

    assert (
            "ORD202610002"
            in selected_text_2
    )

    print(
        "[PASS] Historical order list retained"
    )

    # ========================================================
    # Case 3:
    # Character budget should be respected.
    # ========================================================

    messages_3 = [
        {
            "role": "user",
            "content": "A" * 300,
        },
        {
            "role": "assistant",
            "content": "B" * 300,
        },
        {
            "role": "user",
            "content": "C" * 300,
        },
        {
            "role": "assistant",
            "content": "D" * 300,
        },
    ]

    small_selector = HistorySelector(
        candidate_limit=12,
        max_messages=6,
        char_budget=650,
        per_message_char_limit=500,
    )

    result_3 = small_selector.select(
        question="刚才那个问题呢？",
        messages=messages_3,
        active_order_id=None,
        last_intent=None,
    )

    assert (
            result_3["char_count"]
            <= 650
    )

    assert (
            result_3["selected_count"]
            <= 6
    )

    print(
        "[PASS] Character budget"
    )

    # ========================================================
    # Case 4:
    # Output history must remain chronological.
    # ========================================================

    messages_4 = [
        {
            "role": "user",
            "content": "message-1",
        },
        {
            "role": "assistant",
            "content": "message-2",
        },
        {
            "role": "user",
            "content": "ORD202610002 message-3",
        },
        {
            "role": "assistant",
            "content": "message-4",
        },
    ]

    result_4 = selector.select(
        question="这个订单呢？",
        messages=messages_4,
        active_order_id="ORD202610002",
        last_intent="ORDER_QUERY",
    )

    original_order = {
        message["content"]: index
        for index, message
        in enumerate(messages_4)
    }

    selected_positions = [
        original_order[
            message["content"]
        ]
        for message
        in result_4["messages"]
    ]

    assert (
            selected_positions
            == sorted(
        selected_positions
    )
    )

    print(
        "[PASS] Chronological order"
    )

    print()
    print(
        "=" * 70
    )
    print(
        "History Selector Smoke Test: PASS"
    )
    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()