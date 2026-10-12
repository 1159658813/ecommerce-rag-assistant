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


def main():
    print("=" * 80)
    print("Mixed Query Smoke Test")
    print("=" * 80)

    service = CustomerSupportService()

    question = (
        "我的订单 ORD202610001 还没有发货，"
        "可以按照退款规则取消吗？"
    )

    print()
    print(f"Question : {question}")
    print("User ID  : U001")
    print()

    try:
        result = service.ask(
            question=question,
            user_id="U001",
        )

    except Exception as exc:
        print("Result   : ERROR")
        print(f"Error    : {exc}")
        raise

    print(
        f"Intent   : "
        f"{result.get('intent')}"
    )

    print(
        f"Handler  : "
        f"{result.get('handler')}"
    )

    print(
        f"Memory   : "
        f"{result.get('memory_used')}"
    )

    print()
    print("Answer:")
    print(result.get("answer"))

    mixed_result = (
            result.get("result")
            or {}
    )

    print()
    print(
        f"Order ID : "
        f"{mixed_result.get('order_id')}"
    )

    print(
        f"Combined : "
        f"{mixed_result.get('combined')}"
    )

    print(
        f"Verdict  : "
        f"{mixed_result.get('verdict')}"
    )

    print(
        f"Abstained: "
        f"{mixed_result.get('abstained')}"
    )

    order_result = (
            mixed_result.get(
                "order_result"
            )
            or {}
    )

    rag_result = (
            mixed_result.get(
                "rag_result"
            )
            or {}
    )

    evidences = (
            mixed_result.get(
                "evidences"
            )
            or []
    )

    print(
        f"Order OK : "
        f"{order_result.get('success')}"
    )

    print(
        f"Evidence : "
        f"{len(evidences)}"
    )

    # ========================================================
    # Verification
    # ========================================================

    passed = (
            result.get("intent")
            == "MIXED_QUERY"

            and result.get("handler")
            == "MIXED_QUERY"

            and mixed_result.get("order_id")
            == "ORD202610001"

            and order_result.get("success")
            is True

            and rag_result
    )

    print()
    print("=" * 80)

    if passed:
        print("Mixed Query Result: PASS")
    else:
        print("Mixed Query Result: FAIL")

    print("=" * 80)


if __name__ == "__main__":
    main()