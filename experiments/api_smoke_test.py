import sys
from pathlib import Path

from fastapi.testclient import TestClient


# ============================================================
# Project Root
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from src.api import app


# ============================================================
# Helpers
# ============================================================

def assert_equal(
    actual,
    expected,
    message,
):
    if actual != expected:
        raise AssertionError(
            f"{message}: "
            f"expected={expected!r}, "
            f"actual={actual!r}"
        )


def print_pass(name):
    print(
        f"[PASS] {name}"
    )


# ============================================================
# Smoke Test
# ============================================================

print(
    "=" * 80
)
print(
    "FastAPI RAG Smoke Test"
)
print(
    "=" * 80
)


with TestClient(app) as client:

    # --------------------------------------------------------
    # Case 1: Health
    # --------------------------------------------------------

    response = client.get(
        "/health"
    )

    request_id = response.headers.get(
        "X-Request-ID"
    )

    if not request_id:
        raise AssertionError(
            "response missing X-Request-ID"
        )

    print_pass(
        "response request id"
    )
    response = client.get(
        "/health",
        headers={
            "X-Request-ID":
                "smoke-test-request-id"
        },
    )

    assert_equal(
        response.status_code,
        200,
        "custom request id status",
    )

    assert_equal(
        response.headers.get(
            "X-Request-ID"
        ),
        "smoke-test-request-id",
        "custom request id",
    )

    print_pass(
        "custom X-Request-ID"
    )

    assert_equal(
        response.status_code,
        200,
        "health status code",
    )

    assert_equal(
        response.json(),
        {
            "status": "ok",
        },
        "health response",
    )

    print_pass(
        "GET /health"
    )


    # --------------------------------------------------------
    # Case 2: Answerable
    # --------------------------------------------------------

    response = client.post(
        "/api/v1/query",
        json={
            "question":
                "银行卡退款多久能到账？"
        },
    )

    assert_equal(
        response.status_code,
        200,
        "answerable status code",
    )

    data = response.json()

    assert_equal(
        data[
            "abstained"
        ],
        False,
        "answerable abstained",
    )

    assert_equal(
        data[
            "verdict"
        ],
        "SUFFICIENT",
        "answerable verdict",
    )

    if not data.get(
        "answer"
    ):
        raise AssertionError(
            "answerable case "
            "returned empty answer"
        )

    if not isinstance(
        data.get(
            "evidences"
        ),
        list,
    ):
        raise AssertionError(
            "evidences must be a list"
        )

    print_pass(
        "POST /api/v1/query "
        "answerable"
    )


    # --------------------------------------------------------
    # Case 3: Abstention
    # --------------------------------------------------------

    response = client.post(
        "/api/v1/query",
        json={
            "question":
                "平台支持货到付款吗？"
        },
    )

    assert_equal(
        response.status_code,
        200,
        "abstention status code",
    )

    data = response.json()

    assert_equal(
        data[
            "abstained"
        ],
        True,
        "abstention flag",
    )

    assert_equal(
        data[
            "verdict"
        ],
        "INSUFFICIENT",
        "abstention verdict",
    )

    assert_equal(
        data[
            "abstain_reason"
        ],
        "evidence_insufficient",
        "abstention reason",
    )

    print_pass(
        "POST /api/v1/query "
        "abstention"
    )


    # --------------------------------------------------------
    # Case 4: Invalid Input
    # --------------------------------------------------------

    response = client.post(
        "/api/v1/query",
        json={
            "question": ""
        },
    )

    assert_equal(
        response.status_code,
        422,
        "invalid input status code",
    )

    print_pass(
        "POST /api/v1/query "
        "invalid input"
    )


print(
    "\n"
    + "=" * 80
)

print(
    "API Smoke Test PASSED"
)

print(
    "=" * 80
)