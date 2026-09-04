from contextlib import asynccontextmanager

from src.api.schemas import (
    HealthResponse,
    QueryRequest,
    QueryResponse,
    ErrorResponse,
)
from src.service import build_rag_service

import time
import uuid

from src.config import settings

from src.observability import (
    configure_logging,
    get_logger,
)

from pathlib import Path

from fastapi import (
    FastAPI,
    HTTPException,
    Request,
)

from fastapi.exceptions import (
    RequestValidationError,
)

from fastapi.responses import (
    FileResponse,
    JSONResponse,
)

from fastapi.staticfiles import StaticFiles

configure_logging(
    settings.log_level
)

logger = get_logger(
    "api"
)

def get_request_id(
    request: Request,
):

    return (
        getattr(
            request.state,
            "request_id",
            None,
        )
        or uuid.uuid4().hex
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

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

FRONTEND_DIR = (
    PROJECT_ROOT
    / "frontend"
)

app.mount(
    "/frontend",
    StaticFiles(
        directory=str(
            FRONTEND_DIR
        )
    ),
    name="frontend",
)


@app.get(
    "/",
    include_in_schema=False,
)
def frontend_home():

    return FileResponse(
        FRONTEND_DIR
        / "index.html"
    )

@app.exception_handler(
    RequestValidationError
)
async def validation_exception_handler(
    request: Request,
    error: RequestValidationError,
):

    request_id = get_request_id(
        request
    )

    logger.warning(
        "request_validation_failed "
        "request_id=%s "
        "path=%s",
        request_id,
        request.url.path,
    )

    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "code":
                    "VALIDATION_ERROR",

                "message":
                    "Request validation failed.",

                "request_id":
                    request_id,
            }
        },
        headers={
            "X-Request-ID":
                request_id,
        },
    )

@app.exception_handler(
    HTTPException
)
async def http_exception_handler(
    request: Request,
    error: HTTPException,
):

    request_id = get_request_id(
        request
    )

    detail = error.detail

    if isinstance(
        detail,
        dict,
    ):

        code = str(
            detail.get(
                "code",
                "HTTP_ERROR",
            )
        )

        message = str(
            detail.get(
                "message",
                "Request failed.",
            )
        )

    else:

        code = "HTTP_ERROR"

        message = str(
            detail
        )

    headers = dict(
        error.headers
        or {}
    )

    headers[
        "X-Request-ID"
    ] = request_id

    return JSONResponse(
        status_code=error.status_code,
        content={
            "error": {
                "code":
                    code,

                "message":
                    message,

                "request_id":
                    request_id,
            }
        },
        headers=headers,
    )

@app.exception_handler(
    Exception
)
async def unexpected_exception_handler(
    request: Request,
    error: Exception,
):

    request_id = get_request_id(
        request
    )

    logger.error(
        "unexpected_error "
        "request_id=%s "
        "path=%s",
        request_id,
        request.url.path,
        exc_info=(
            type(error),
            error,
            error.__traceback__,
        ),
    )

    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "code":
                    "INTERNAL_SERVER_ERROR",

                "message":
                    "Internal server error.",

                "request_id":
                    request_id,
            }
        },
        headers={
            "X-Request-ID":
                request_id,
        },
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
    responses={
        400: {
            "model":
                ErrorResponse,

            "description":
                "Invalid request",
        },

        422: {
            "model":
                ErrorResponse,

            "description":
                "Validation error",
        },

        500: {
            "model":
                ErrorResponse,

            "description":
                "RAG service error",
        },
    },
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
            detail={
                "code":
                    "INVALID_REQUEST",

                "message":
                    str(error),
            },
        ) from error

    except Exception as error:

        logger.exception(
            "rag_failed "
            "request_id=%s",
            request_id,
        )

        raise HTTPException(
            status_code=500,
            detail={
                "code":
                    "RAG_SERVICE_ERROR",

                "message":
                    (
                        "RAG service failed "
                        "to process the request."
                    ),
            },
        ) from error

    return result

    return result