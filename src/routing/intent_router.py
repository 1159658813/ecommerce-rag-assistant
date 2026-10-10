import json
import os

from openai import OpenAI

from src.config import settings
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
        )

    def route(
            self,
            question: str,
    ) -> dict:
        """
        Classify the user question.

        Returns:
            {
                "intent": "...",
                "reason": "..."
            }
        """

        question = question.strip()

        if not question:
            return {
                "intent": "UNSUPPORTED",
                "reason": "用户问题为空。",
            }

        response = (
            self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": question,
                    },
                ],
                temperature=0.0,

                response_format={
                    "type": "json_object"
                },

                extra_body={
                    "enable_thinking": False
                },
            )
        )

        content = (
            response
            .choices[0]
            .message
            .content
            .strip()
        )

        try:
            result = json.loads(content)

        except json.JSONDecodeError as exc:
            raise ValueError(
                "Intent Router 返回了无效 JSON。"
            ) from exc

        intent = str(
            result.get(
                "intent",
                ""
            )
        ).strip().upper()

        reason = str(
            result.get(
                "reason",
                ""
            )
        ).strip()

        if intent not in VALID_INTENTS:
            return {
                "intent": "UNSUPPORTED",
                "reason": (
                    "Intent Router 返回了未知意图："
                    f"{intent}"
                ),
            }

        return {
            "intent": intent,
            "reason": reason,
        }