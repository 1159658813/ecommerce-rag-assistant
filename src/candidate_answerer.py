class CandidateAnswerer:

    UNKNOWN = "UNKNOWN"


    def __init__(
        self,
        generator
    ):

        self.generator = generator


    def generate(
        self,
        query,
        document
    ):

        evidence = (
            document["text"]
        )


        system_prompt = """
你是一个知识库候选答案生成器。

你的任务是根据给出的知识库资料，
为用户问题生成一个非常简短的候选答案。

规则：

1. 只能使用知识库资料中明确存在的信息。

2. 不允许使用模型自身知识、常识或猜测。

3. 不需要解释推理过程。

4. 尽量只生成一句能够直接回答用户问题的陈述。

5. 如果资料中没有明确答案，
   输出 UNKNOWN。

6. 不要输出引用、资料编号或额外说明。
""".strip()


        user_prompt = f"""
【知识库资料】

{evidence}


【用户问题】

{query}


请生成候选答案。
""".strip()


        messages = [
            {
                "role": "system",
                "content":
                    system_prompt
            },

            {
                "role": "user",
                "content":
                    user_prompt
            }
        ]


        output = (
            self.generator.generate(
                messages=messages,
                max_new_tokens=96
            )
        )


        answer = output.strip()


        return {
            "answer": answer,

            "unknown": (
                answer.upper()
                == self.UNKNOWN
            )
        }