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


class ChatRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        max_length=2000,
    )

    user_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
    )

    conversation_id: str | None = Field(
        default=None,
        max_length=100,
    )


class ChatResponse(BaseModel):
    conversation_id: str
    user_id: str

    question: str
    resolved_question: str

    memory_used: bool

    intent: str
    intent_reason: str

    handler: str
    answer: str

    result: Any | None = None

class HealthResponse(BaseModel):

    status: str


class ErrorDetail(BaseModel):

    code: str
    message: str
    request_id: str


class ErrorResponse(BaseModel):

    error: ErrorDetail