from typing import Any

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):

    question: str = Field(
        ...,
        min_length=1,
        max_length=2000,
    )


class EvidenceResponse(BaseModel):

    rank: int
    source: str | None = None
    section: str | None = None
    content: str | None = None
    reranker_score: float | None = None


class QueryResponse(BaseModel):

    question: str
    answer: str

    abstained: bool
    abstain_reason: str | None = None

    verdict: str | None = None

    evidences: list[
        EvidenceResponse
    ]


class HealthResponse(BaseModel):

    status: str


class ErrorDetail(BaseModel):

    code: str
    message: str
    request_id: str


class ErrorResponse(BaseModel):

    error: ErrorDetail