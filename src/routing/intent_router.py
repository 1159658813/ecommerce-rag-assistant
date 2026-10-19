import json
import os

from openai import OpenAI
from openai import APIError

from src.config import settings
from src.reliability import FallbackPolicy, RetryPolicy
from src.routing.prompt import SYSTEM_PROMPT

VALID_INTENTS = {
    "KNOWLEDGE_QA",
    "ORDER_QUERY",
    "MIXED_QUERY",
    "GENERAL_CHAT",
    "UNSUPPORTED",
}


class IntentRouter:
    """
    Classify a user question into one of the supported
    ecommerce customer-service intents.

    This class only performs intent classification.
    It does not execute RAG, order queries, or tools.
    """

    def __init__(
            self,
            model: str | None = None,
    ):
        api_key = os.getenv(
            "DASHSCOPE_API_KEY"
        )

        if not api_key:
            raise ValueError(
                "未检测到 DASHSCOPE_API_KEY，"
                "请检查 .env 文件。"
            )

        base_url = os.getenv(
            "DASHSCOPE_BASE_URL",
            (
                "https://dashscope.aliyuncs.com/"
                "compatible-mode/v1"
            ),
        )

        self.model = (
                model
                or settings.intent_router_model
        )

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=12.0,
            max_retries=0,
        )

    # P1-6 ROUTER FAIL-CLOSED
    def route(self, question: str) -> dict:
        question = str(question or "").strip()
        if not question:
            return {"intent": "UNSUPPORTED", "reason": "用户问题为空。"}

        def attempt():
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": question},
                ],
                temperature=0.0,
                response_format={"type": "json_object"},
                extra_body={"enable_thinking": False},
            )
            content = (response.choices[0].message.content or "").strip()
            payload = json.loads(content)
            if not isinstance(payload, dict):
                raise ValueError("Invalid intent payload")
            intent = str(payload.get("intent", "")).strip().upper()
            if intent not in VALID_INTENTS:
                raise ValueError("Unknown intent")
            return {
                "intent": intent,
                "reason": str(payload.get("reason", "")).strip(),
            }

        try:
            return RetryPolicy(max_attempts=2).run(
                attempt,
                retry_on=(APIError, TimeoutError, ConnectionError,
                          ValueError, IndexError, TypeError, AttributeError),
            )
        except Exception:
            return FallbackPolicy.router_result()
