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
class FailingRAGService:

    def ask(
        self,
        question,
    ):

        raise RuntimeError(
            "forced smoke test failure"
        )

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

def assert_error_response(
    response,
    expected_status,
    expected_code,
):

    assert_equal(
        response.status_code,
        expected_status,
        "error status code",
    )

    data = response.json()

    if "error" not in data:
        raise AssertionError(
            "error response missing "
            "'error' field"
        )

    error = data[
        "error"
    ]

    assert_equal(
        error.get(
            "code"
        ),
        expected_code,
        "error code",
    )

    if not error.get(
        "message"
    ):
        raise AssertionError(
            "error message is empty"
        )

    request_id = error.get(
        "request_id"
    )

    if not request_id:
        raise AssertionError(
            "error response missing "
            "request_id"
        )

    assert_equal(
        response.headers.get(
            "X-Request-ID"
        ),
        request_id,
        "error request id header",
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

    original_service = (
        client.app.state.rag_service
    )

    try:

        client.app.state.rag_service = (
            FailingRAGService()
        )

        response = client.post(
            "/api/v1/query",
            json={
                "question":
                    "测试内部异常"
            },
        )

    finally:

        client.app.state.rag_service = (
            original_service
        )

    assert_error_response(
        response=response,
        expected_status=500,
        expected_code="RAG_SERVICE_ERROR",
    )

    print_pass(
        "POST /api/v1/query "
        "service error"
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

    assert_error_response(
        response=response,
        expected_status=422,
        expected_code="VALIDATION_ERROR",
    )

    print_pass(
        "POST /api/v1/query "
        "validation error"
    )


    response = client.post(
        "/api/v1/query",
        json={
            "question": "   "
        },
    )

    assert_error_response(
        response=response,
        expected_status=400,
        expected_code="INVALID_REQUEST",
    )

    print_pass(
        "POST /api/v1/query "
        "invalid request"
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