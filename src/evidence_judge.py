class EvidenceJudge:

    def __init__(self, generator):
        self.generator = generator


    def judge(
        self,
        query,
        document
    ):

        evidence = document["text"]


        system_prompt = """
你是一个知识库证据充分性判断器。

你的任务不是回答用户问题，
而是判断给出的【知识库资料】是否明确包含
回答【用户问题】所需的信息。

判断标准：

1. 如果资料明确包含回答问题所需的事实、条件、时间、金额、
   范围或规则，则输出 YES。

2. 如果资料只是和问题主题相关，
   但没有说明用户真正询问的具体信息，则输出 NO。

3. 不允许使用常识、模型自身知识或猜测。

4. 不允许根据相似主题推断答案。

5. 只允许输出：
YES
或
NO

不要输出任何解释。
""".strip()


        user_prompt = f"""
【知识库资料】

{evidence}


【用户问题】

{query}


该资料是否足以明确回答这个问题？
""".strip()


        messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ]


        output = self.generator.generate(
            messages=messages,
            max_new_tokens=8
        )


        normalized = (
            output
            .strip()
            .upper()
        )


        # Fail Closed：
        # 只有明确输出 YES 才允许通过
        sufficient = (
            normalized == "YES"
        )


        return {
            "sufficient": sufficient,
            "raw_output": output
        }