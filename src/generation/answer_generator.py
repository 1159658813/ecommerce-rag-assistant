import os

from dotenv import load_dotenv
from openai import OpenAI

from .prompt import (
    SYSTEM_PROMPT,
    build_answer_prompt
)


load_dotenv()


class AnswerGenerator:

    def __init__(
        self,
        model="qwen-plus-2025-07-28"
    ):

        api_key = os.getenv(
            "DASHSCOPE_API_KEY"
        )

        if not api_key:

            raise ValueError(
                "未检测到 DASHSCOPE_API_KEY，"
                "请检查 .env 文件。"
            )

        self.model = model

        self.client = OpenAI(
            api_key=api_key,
            base_url=(
                "https://dashscope.aliyuncs.com/"
                "compatible-mode/v1"
            )
        )


    def generate(
        self,
        question,
        evidences
    ):
        if not evidences:
            return (
                "根据当前知识库信息，"
                "暂时无法确认。"
            )

        user_prompt = build_answer_prompt(
            question=question,
            evidences=evidences
        )

        response = (
            self.client.chat.completions.create(
                model=self.model,

                messages=[
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT
                    },
                    {
                        "role": "user",
                        "content": user_prompt
                    }
                ],

                temperature=0.0
            )
        )

        answer = (
            response
            .choices[0]
            .message
            .content
            .strip()
        )

        return answer