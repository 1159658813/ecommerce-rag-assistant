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

    try:

        result = service.ask(
            payload.question
        )

    except ValueError as error:

        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=(
                "RAG service failed "
                "to process the request."
            ),
        ) from error

    return result