class RAGService:

    def __init__(
        self,
        pipeline,
    ):
        self.pipeline = pipeline

    def ask(
        self,
        question,
    ):
        question = str(
            question or ""
        ).strip()

        if not question:
            raise ValueError(
                "question 不能为空。"
            )

        result = self.pipeline.ask(
            question=question
        )

        return {
            "question": result["question"],
            "answer": result["answer"],
            "abstained": result["abstained"],
            "abstain_reason": result["abstain_reason"],
            "verdict": (
                result["verifier"]["verdict"]
                if result["verifier"]
                else None
            ),
            "evidences": result["evidences"],
        }