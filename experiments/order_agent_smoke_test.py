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

from src.agents.order_agent import OrderAgent
from src.orders.tools import OrderToolExecutor

# ============================================================
# LLM Tool Calling Cases
# ============================================================

LLM_TEST_CASES = [
    {
        "name": "Logistics Query",
        "user_id": "U001",
        "question": "帮我查一下 ORD202610002 现在到哪里了？",
        "expected_tool": "get_logistics",
    },
    {
        "name": "User Orders Query",
        "user_id": "U001",
        "question": "帮我看看我现在有哪些订单。",
        "expected_tool": "get_user_orders",
    },
]


def run_llm_tests() -> tuple[int, int]:
    """
    Run real LLM tool-calling tests.
    """

    agent = OrderAgent()

    passed = 0
    total = len(LLM_TEST_CASES)

    print("=" * 70)
    print("Order Agent LLM Tool Calling Tests")
    print("=" * 70)

    for index, case in enumerate(
            LLM_TEST_CASES,
            start=1,
    ):
        print()
        print(f"[LLM Case {index}]")
        print(f"Name          : {case['name']}")
        print(f"User ID       : {case['user_id']}")
        print(f"Question      : {case['question']}")
        print(
            f"Expected Tool : "
            f"{case['expected_tool']}"
        )

        try:
            result = agent.ask(
                question=case["question"],
                user_id=case["user_id"],
            )

            tools_called = result.get(
                "tools_called",
                [],
            )

            answer = result.get(
                "answer",
                "",
            )

            print(
                f"Tools Called  : "
                f"{tools_called}"
            )
            print(f"Answer        : {answer}")

            if case["expected_tool"] in tools_called:
                print("Result        : PASS")
                passed += 1
            else:
                print("Result        : FAIL")

        except Exception as exc:
            print("Result        : ERROR")
            print(f"Error         : {exc}")

    return passed, total


# ============================================================
# Security Test
# ============================================================

def run_security_test() -> tuple[int, int]:
    """
    Verify that one user cannot query another user's order.

    This test does not call the LLM.
    """

    executor = OrderToolExecutor()

    print()
    print("=" * 70)
    print("Order Tool Security Test")
    print("=" * 70)

    print()
    print("[Security Case]")
    print("Current User : U001")
    print("Order        : ORD202610003")
    print("Owner        : U002")

    result = executor.execute(
        tool_name="get_order_detail",
        arguments={
            "order_id": "ORD202610003"
        },
        user_id="U001",
    )

    print(f"Tool Result  : {result}")

    if (
            result.get("success") is False
            and result.get("code")
            == "ORDER_NOT_FOUND"
    ):
        print("Result       : PASS")
        return 1, 1

    print("Result       : FAIL")
    return 0, 1


# ============================================================
# Main
# ============================================================

def main():
    llm_passed, llm_total = (
        run_llm_tests()
    )

    security_passed, security_total = (
        run_security_test()
    )

    total_passed = (
            llm_passed
            + security_passed
    )

    total_cases = (
            llm_total
            + security_total
    )

    print()
    print("=" * 70)
    print("Order Agent Smoke Test Summary")
    print("=" * 70)

    print(
        f"Passed: "
        f"{total_passed}/{total_cases}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()