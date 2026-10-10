import sys
from pathlib import Path

# ============================================================
# Project Root
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )

from src.customer_support import CustomerSupportService
from src.memory import ConversationMemoryService


def main():
    service = CustomerSupportService()

    print("=" * 70)
    print("Conversation Memory Smoke Test")
    print("=" * 70)

    # ========================================================
    # Turn 1
    # ========================================================

    print()
    print("[Turn 1]")
    print(
        "User: 帮我查一下 ORD202610002 "
        "这个订单。"
    )

    first_result = service.ask(
        question=(
            "帮我查一下 ORD202610002 "
            "这个订单。"
        ),
        user_id="U001",
    )

    conversation_id = (
        first_result[
            "conversation_id"
        ]
    )

    print(
        f"Conversation ID : "
        f"{conversation_id}"
    )
    print(
        f"Intent          : "
        f"{first_result['intent']}"
    )
    print(
        f"Handler         : "
        f"{first_result['handler']}"
    )
    print(
        f"Memory Used     : "
        f"{first_result['memory_used']}"
    )
    print(
        f"Answer          : "
        f"{first_result['answer']}"
    )

    # ========================================================
    # Inspect Memory
    # ========================================================

    memory = ConversationMemoryService()

    context = memory.get_context(
        conversation_id=conversation_id,
        user_id="U001",
    )

    active_order_id = (
        context["conversation"][
            "active_order_id"
        ]
    )

    print()
    print(
        f"Remembered Order: "
        f"{active_order_id}"
    )

    # ========================================================
    # Turn 2
    # ========================================================

    print()
    print("[Turn 2]")
    print(
        "User: 它现在到哪了？"
    )

    second_result = service.ask(
        question="它现在到哪了？",
        user_id="U001",
        conversation_id=conversation_id,
    )

    print(
        f"Intent          : "
        f"{second_result['intent']}"
    )
    print(
        f"Handler         : "
        f"{second_result['handler']}"
    )
    print(
        f"Memory Used     : "
        f"{second_result['memory_used']}"
    )
    print(
        f"Resolved        : "
        f"{second_result['resolved_question']}"
    )
    print(
        f"Answer          : "
        f"{second_result['answer']}"
    )

    # ========================================================
    # Verification
    # ========================================================

    passed = (
            active_order_id
            == "ORD202610002"
            and second_result[
                "memory_used"
            ] is True
            and "ORD202610002"
            in second_result[
                "resolved_question"
            ]
            and second_result[
                "intent"
            ]
            == "ORDER_QUERY"
    )

    print()
    print("=" * 70)

    if passed:
        print(
            "Conversation Memory Result: PASS"
        )
    else:
        print(
            "Conversation Memory Result: FAIL"
        )

    print("=" * 70)


if __name__ == "__main__":
    main()