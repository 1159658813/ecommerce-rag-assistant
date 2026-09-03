class RAGSystem:

    def __init__(
        self,
        retriever,
        generator,
        min_retrieval_score=0.72,
        context_k=1
    ):
        self.retriever = retriever
        self.generator = generator

        self.min_retrieval_score = (
            min_retrieval_score
        )

        self.context_k = context_k


    def build_context(
        self,
        retrieval_results
    ):

        context_parts = []

        for index, result in enumerate(
            retrieval_results,
            start=1
        ):

            document = result["document"]

            context_parts.append(
                f"""
【资料{index}】

知识主题：
{document['section']}

知识内容：
{document['text']}
""".strip()
            )

        return "\n\n".join(
            context_parts
        )


    def build_messages(
        self,
        query,
        context
    ):

        system_prompt = """
你是电商平台客服。

你只能依据用户消息中提供的【知识库资料】回答问题。

禁止使用模型自身记忆、常识或猜测补充资料中不存在的信息。

如果资料没有明确说明某项信息，回答：
“根据当前知识库资料无法确定。”

回答应简洁，并优先直接回答用户的问题。
""".strip()


        user_prompt = f"""
【知识库资料】

{context}


【用户问题】

{query}
""".strip()


        return [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ]


    def ask(
        self,
        query,
        candidate_k=3
    ):

        # ====================================
        # 1. Retrieve candidates
        # ====================================

        retrieval_results = (
            self.retriever.retrieve(
                query=query,
                top_k=candidate_k
            )
        )


        # ====================================
        # 2. Hard Abstention Gate
        # ====================================

        if not retrieval_results:

            return {
                "query": query,
                "answer": (
                    "根据当前知识库资料无法确定。"
                ),
                "abstained": True,
                "retrieval_results": []
            }


        best_score = (
            retrieval_results[0]["score"]
        )


        if (
            best_score
            < self.min_retrieval_score
        ):

            return {
                "query": query,
                "answer": (
                    "根据当前知识库资料无法确定。"
                ),
                "abstained": True,
                "retrieval_results": (
                    retrieval_results
                )
            }


        # ====================================
        # 3. Context Selection
        # ====================================

        context_results = (
            retrieval_results[
                :self.context_k
            ]
        )


        # ====================================
        # 4. Build Context
        # ====================================

        context = self.build_context(
            context_results
        )


        # ====================================
        # 5. Build Prompt
        # ====================================

        messages = self.build_messages(
            query=query,
            context=context
        )


        # ====================================
        # 6. Generation
        # ====================================

        answer = self.generator.generate(
            messages
        )


        return {
            "query": query,
            "answer": answer,
            "abstained": False,

            # 搜索出的候选
            "retrieval_results":
                retrieval_results,

            # 真正给模型看的证据
            "context_results":
                context_results
        }