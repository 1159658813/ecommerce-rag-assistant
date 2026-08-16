import os

from dotenv import load_dotenv
from openai import OpenAI


class DashScopeQwenGenerator:

    def __init__(
        self,
        model_name=None,
        temperature=0.0,
        enable_thinking=False
    ):
        # 加载项目根目录附近的 .env
        load_dotenv()

        api_key = os.getenv(
            "DASHSCOPE_API_KEY"
        )

        base_url = os.getenv(
            "DASHSCOPE_BASE_URL"
        )

        if model_name is None:
            model_name = os.getenv(
                "DASHSCOPE_MODEL",
                "qwen-plus-2025-07-28"
            )

        if not api_key:
            raise ValueError(
                "没有读取到 DASHSCOPE_API_KEY。"
                "请检查 .env 配置。"
            )

        if not base_url:
            raise ValueError(
                "没有读取到 DASHSCOPE_BASE_URL。"
                "请检查 .env 配置。"
            )

        self.model_name = model_name

        self.temperature = temperature

        self.enable_thinking = (
            enable_thinking
        )

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=120.0
        )


    def generate(
        self,
        messages,
        max_new_tokens=256
    ):

        response = (
            self.client
            .chat
            .completions
            .create(
                model=self.model_name,

                messages=messages,

                # 为分类 / Judge 尽量降低随机性
                temperature=(
                    self.temperature
                ),

                # 对 qwen3-8b 关闭思考
                # 保证和 1.5B baseline
                # 尽可能公平比较
                extra_body={
                    "enable_thinking":
                        self.enable_thinking
                },

                max_tokens=(
                    max_new_tokens
                )
            )
        )

        content = (
            response
            .choices[0]
            .message
            .content
        )

        if content is None:
            return ""

        return content.strip()