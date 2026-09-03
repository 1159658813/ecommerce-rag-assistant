from contextlib import asynccontextmanager

from fastapi import (
    FastAPI,
    HTTPException,
    Request,
)

from src.api.schemas import (
    HealthResponse,
    QueryRequest,
    QueryResponse,
)
from src.service import build_rag_service

import time
import uuid

from src.config import settings

from src.observability import (
    configure_logging,
    get_logger,
)

configure_logging(
    settings.log_level
)

logger = get_logger(
    "api"
)


@asynccontextmanager
async def lifespan(
    app: FastAPI,
):

    print(
        "Loading RAGService..."
    )

    app.state.rag_service = (
        build_rag_service()
    )

    print(
        "RAGService ready."
    )

    yield

    print(
        "RAGService shutting down."
    )



app = FastAPI(
    title="Ecommerce RAG Assistant",
    version="1.0.0",
    description=(
        "RAG service for Chinese "
        "e-commerce customer support."
    ),
    lifespan=lifespan,
)

@app.middleware(
    "http"
)
async def request_logging_middleware(
    request: Request,
    call_next,
):

    request_id = (
        request.headers.get(
            "X-Request-ID"
        )
        or uuid.uuid4().hex
    )

    request.state.request_id = (
        request_id
    )

    start_time = (
        time.perf_counter()
    )

    logger.info(
        "request_started "
        "request_id=%s "
        "method=%s "
        "path=%s",
        request_id,
        request.method,
        request.url.path,
    )

    try:

        response = await call_next(
            request
        )

    except Exception:

        latency_ms = (
            time.perf_counter()
            - start_time
        ) * 1000

        logger.exception(
            "request_failed "
            "request_id=%s "
            "method=%s "
            "path=%s "
            "latency_ms=%.2f",
            request_id,
            request.method,
            request.url.path,
            latency_ms,
        )

        raise

    latency_ms = (
        time.perf_counter()
        - start_time
    ) * 1000

    response.headers[
        "X-Request-ID"
    ] = request_id

    logger.info(
        "request_completed "
        "request_id=%s "
        "method=%s "
        "path=%s "
        "status=%s "
        "latency_ms=%.2f",
        request_id,
        request.method,
        request.url.path,
        response.status_code,
        latency_ms,
    )

    return response


@app.get(
    "/health",
    response_model=HealthResponse,
)
def health():

    return {
        "status": "ok",
    }


@app.post(
    "/api/v1/query",
    response_model=QueryResponse,
)
def query(
    payload: QueryRequest,
    request: Request,
):

    service = (
        request.app.state.rag_service
    )

    request_id = (
        request.state.request_id
    )

    try:

        result = service.ask(
            payload.question
        )

        logger.info(
            "rag_completed "
            "request_id=%s "
            "verdict=%s "
            "abstained=%s "
            "evidence_count=%s",
            request_id,
            result.get(
                "verdict"
            ),
            result.get(
                "abstained"
            ),
            len(
                result.get(
                    "evidences",
                    [],
                )
            ),
        )

    except ValueError as error:
        logger.warning(
            "rag_invalid_request "
            "request_id=%s "
            "error=%s",
            request_id,
            str(error),
        )

        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error



    except Exception as error:
        logger.exception(
            "rag_failed "
            "request_id=%s",
            request_id,
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "RAG service failed "
                "to process the request."
            ),
        ) from error

    return result