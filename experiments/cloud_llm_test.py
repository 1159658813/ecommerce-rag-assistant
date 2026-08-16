from src.cloud_llm import (
    DashScopeQwenGenerator
)


generator = DashScopeQwenGenerator(
    model_name="qwen-plus-2025-07-28",
    enable_thinking=False
)


messages = [
    {
        "role": "system",
        "content": (
            "你是一个测试助手。"
            "严格按照要求输出。"
        )
    },
    {
        "role": "user",
        "content": (
            "只输出 OK，"
            "不要输出其他内容。"
        )
    }
]


answer = generator.generate(
    messages=messages,
    max_new_tokens=16
)


print(
    "Model:",
    generator.model_name
)

print(
    "Answer:",
    repr(answer)
)