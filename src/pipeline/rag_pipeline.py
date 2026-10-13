import time
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
    def _elapsed_ms(
            started_at: float,
    ) -> float:
        return round(
            (
                    time.perf_counter()
                    - started_at
            )
            * 1000,
            3,
        )

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

        retrieval_started = (
            time.perf_counter()
        )

        retrieval_results = (
            self.retriever.retrieve(
                query=question,
                candidate_k=self.candidate_k,
                final_k=self.evidence_k,
            )
        )

        latency_retrieval = (
            self._elapsed_ms(
                retrieval_started
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
                "observability": {
                    "latency_retrieval": (
                        latency_retrieval
                    ),
                    "latency_verifier": 0.0,
                    "latency_composer": 0.0,
                    "verifier_retry_count": 0,
                    "abstained": True,
                    "latency_unit": "ms",
                },
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

        verifier_started = (
            time.perf_counter()
        )

        verifier_result = (
            self.verifier.verify(
                question=question,
                evidences=evidences,
            )
        )

        latency_verifier = (
            self._elapsed_ms(
                verifier_started
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
                "observability": {
                    "latency_retrieval": (
                        latency_retrieval
                    ),
                    "latency_verifier": (
                        latency_verifier
                    ),
                    "latency_composer": 0.0,
                    "verifier_retry_count": 0,
                    "abstained": True,
                    "latency_unit": "ms",
                },
            }

        # ====================================================
        # 6. Generation
        # ====================================================

        composer_started = (
            time.perf_counter()
        )

        answer = (
            self.generator.generate(
                question=question,
                evidences=evidences,
            )
        )

        latency_composer = (
            self._elapsed_ms(
                composer_started
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
            "observability": {
                "latency_retrieval": (
                    latency_retrieval
                ),
                "latency_verifier": (
                    latency_verifier
                ),
                "latency_composer": (
                    latency_composer
                ),
                "verifier_retry_count": 0,
                "abstained": False,
                "latency_unit": "ms",
            },
        }