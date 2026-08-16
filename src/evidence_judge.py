class EvidenceJudge:

    SUFFICIENT = "SUFFICIENT"
    INSUFFICIENT = "INSUFFICIENT"

    def __init__(self, generator):
        self.generator = generator


    def judge(
        self,
        query,
        document
    ):

        evidence = document["text"]


        system_prompt = """
你是一个“知识库证据充分性分类器”。

你的任务不是回答用户问题。
你的任务只判断：

【知识库资料】是否包含足够的信息，
使我们能够仅根据该资料明确回答【用户问题】。

你必须特别注意：

1. 用户问题的实际答案是“是”还是“否”，
   与证据是否充分是两回事。

2. 即使用户问题的正确答案是否定的，
   只要知识库资料能够明确支持这个否定结论，
   仍然应该判断为 SUFFICIENT。

3. 如果资料只是和问题主题相关，
   但没有包含用户真正询问的具体事实，
   判断为 INSUFFICIENT。

4. 如果用户询问具体的时间、金额、数量、范围、
   条件、对象或操作方式，而资料没有明确给出该信息，
   判断为 INSUFFICIENT。

5. 不允许使用模型自身知识、常识或猜测补充资料。

6. 不允许因为“主题很相似”就认为资料足够。

7. 只判断资料能否支持明确回答，
   不要真正回答用户的问题。


示例一：

知识库资料：
某服务的工作时间为每天上午9点至下午6点。

用户问题：
晚上8点还能办理该服务吗？

分类：
SUFFICIENT

说明：
虽然用户问题的实际答案是否定的，
但资料已经足以明确得出结论。


示例二：

知识库资料：
某套餐每月包含100GB流量。

用户问题：
这个套餐是否支持国际漫游？

分类：
INSUFFICIENT

说明：
资料与套餐相关，
但完全没有说明国际漫游。


示例三：

知识库资料：
发生退货时运输费用由用户承担。

用户问题：
运输费用最高不能超过多少钱？

分类：
INSUFFICIENT

说明：
资料说明了谁承担费用，
但没有说明费用上限。


最终只允许输出以下两个标签之一：

SUFFICIENT
INSUFFICIENT

禁止输出解释、标点或其他内容。
""".strip()


        user_prompt = f"""
【知识库资料】

{evidence}


【用户问题】

{query}


请判断该知识库资料是否足以明确回答用户问题。

只输出：
SUFFICIENT
或
INSUFFICIENT
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
            max_new_tokens=16
        )


        normalized = (
            output
            .strip()
            .upper()
        )


        # Fail Closed：
        # 只有明确输出 SUFFICIENT 才允许通过。
        sufficient = (
            normalized
            == self.SUFFICIENT
        )


        return {
            "sufficient": sufficient,
            "raw_output": output,
            "normalized_output": normalized
        }