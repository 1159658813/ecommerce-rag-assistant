class RAGPipeline:

    def __init__(
        self,
        retriever,
        verifier,
        generator,
        candidate_k=10,
        evidence_k=3,
    ):
        self.retriever = retriever
        self.verifier = verifier
        self.generator = generator

        self.candidate_k = candidate_k
        self.evidence_k = evidence_k

    @staticmethod
    def _build_evidences(
        retrieval_results,
    ):
        evidences = []

        for rank, result in enumerate(
            retrieval_results,
            start=1,
        ):
            document = result["document"]

            evidences.append(
                {
                    "rank": rank,
                    "source": document.get(
                        "source"
                    ),
                    "section": document.get(
                        "section"
                    ),
                    "content": document.get(
                        "text"
                    ),
                    "reranker_score": (
                        result.get(
                            "rerank_score"
                        )
                    ),
                }
            )

        return evidences

    def ask(
        self,
        question,
    ):
        # ====================================================
        # 1. Dense Retrieval + Reranker
        # ====================================================

        retrieval_results = (
            self.retriever.retrieve(
                query=question,
                candidate_k=self.candidate_k,
                final_k=self.evidence_k,
            )
        )

        # ====================================================
        # 2. Retrieval Empty
        # ====================================================

        if not retrieval_results:
            return {
                "question": question,
                "answer": (
                    "根据当前知识库信息，"
                    "暂时无法确认。"
                ),
                "abstained": True,
                "abstain_reason": (
                    "retrieval_empty"
                ),
                "retrieval_results": [],
                "evidences": [],
                "verifier": None,
            }

        # ====================================================
        # 3. Normalize Evidence
        # ====================================================

        evidences = (
            self._build_evidences(
                retrieval_results
            )
        )

        # ====================================================
        # 4. Answerability Verification
        # ====================================================

        verifier_result = (
            self.verifier.verify(
                question=question,
                evidences=evidences,
            )
        )

        verdict = verifier_result[
            "verdict"
        ]

        # ====================================================
        # 5. Abstention Gate
        # ====================================================

        if verdict == "INSUFFICIENT":
            return {
                "question": question,
                "answer": (
                    "根据当前知识库信息，"
                    "暂时无法确认。"
                ),
                "abstained": True,
                "abstain_reason": (
                    "evidence_insufficient"
                ),
                "retrieval_results": (
                    retrieval_results
                ),
                "evidences": evidences,
                "verifier": verifier_result,
            }

        # ====================================================
        # 6. Generation
        # ====================================================

        answer = (
            self.generator.generate(
                question=question,
                evidences=evidences,
            )
        )

        # ====================================================
        # 7. Final Result
        # ====================================================

        return {
            "question": question,
            "answer": answer,
            "abstained": False,
            "abstain_reason": None,
            "retrieval_results": (
                retrieval_results
            ),
            "evidences": evidences,
            "verifier": verifier_result,
        }