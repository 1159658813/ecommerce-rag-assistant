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

from src.routing.intent_router import IntentRouter

TEST_CASES = [
    {
        "question": "退款一般多久到账？",
        "expected": "KNOWLEDGE_QA",
    },
    {
        "question": "帮我查一下 ORD202610002 到哪里了？",
        "expected": "ORDER_QUERY",
    },
    {
        "question": "我的订单已经签收8天了，现在还能退吗？",
        "expected": "MIXED_QUERY",
    },
    {
        "question": "你好，谢谢你的帮助。",
        "expected": "GENERAL_CHAT",
    },
    {
        "question": "帮我写一个冒泡排序的 Python 程序。",
        "expected": "UNSUPPORTED",
    },
]


def main():
    router = IntentRouter()

    passed = 0

    print("=" * 70)
    print("Intent Router Smoke Test")
    print("=" * 70)

    for index, case in enumerate(TEST_CASES, start=1):
        question = case["question"]
        expected = case["expected"]

        print()
        print(f"[Case {index}]")
        print(f"Question : {question}")
        print(f"Expected : {expected}")

        try:
            result = router.route(question)

            actual = result["intent"]
            reason = result["reason"]

            print(f"Actual   : {actual}")
            print(f"Reason   : {reason}")

            if actual == expected:
                print("Result   : PASS")
                passed += 1
            else:
                print("Result   : FAIL")

        except Exception as exc:
            print(f"Result   : ERROR")
            print(f"Error    : {exc}")

    print()
    print("=" * 70)
    print(
        f"Passed: {passed}/{len(TEST_CASES)}"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()