import time

from src.reliability import FallbackPolicy


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

    def ask(self, question):
        retrieval_ms = verifier_ms = composer_ms = 0.0
        retry_count = 0
        retrieval_results = []
        evidences = []
        verifier_result = None

        def response(answer, abstained, reason):
            return {
                "question": question,
                "answer": answer,
                "abstained": abstained,
                "abstain_reason": reason,
                "retrieval_results": retrieval_results,
                "evidences": evidences,
                "verifier": verifier_result,
                "observability": {
                    "latency_retrieval": retrieval_ms,
                    "latency_verifier": verifier_ms,
                    "latency_composer": composer_ms,
                    "verifier_retry_count": retry_count,
                    "abstained": abstained,
                    "latency_unit": "ms",
                },
            }

        started = time.perf_counter()
        try:
            retrieval_results = self.retriever.retrieve(
                query=question,
                candidate_k=self.candidate_k,
                final_k=self.evidence_k,
            ) or []
            evidences = self._build_evidences(retrieval_results)
        except Exception:
            retrieval_ms = self._elapsed_ms(started)
            return response(FallbackPolicy.rag_message(), True, "retrieval_unavailable")
        finally:
            retrieval_ms = self._elapsed_ms(started)

        if not retrieval_results:
            return response("根据当前知识库信息，暂时无法确认。", True, "retrieval_empty")

        started = time.perf_counter()
        try:
            verifier_result = self.verifier.verify(question=question, evidences=evidences)
            retry_count = int(verifier_result.get("retry_count", 0) or 0)
            verdict = verifier_result.get("verdict")
            if verdict not in {"SUFFICIENT", "INSUFFICIENT"}:
                raise ValueError("Invalid verifier verdict")
        except Exception as exc:
            retry_count = int(getattr(exc, "retry_count", 0) or 0)
            verifier_ms = self._elapsed_ms(started)
            return response(FallbackPolicy.rag_message(), True, "verifier_unavailable")
        finally:
            verifier_ms = self._elapsed_ms(started)

        if verdict == "INSUFFICIENT":
            return response("根据当前知识库信息，暂时无法确认。", True, "evidence_insufficient")

        started = time.perf_counter()
        try:
            answer = self.generator.generate(question=question, evidences=evidences)
            if not isinstance(answer, str) or not answer.strip():
                composer_ms = self._elapsed_ms(started)
                return response(FallbackPolicy.rag_message(), True, "composer_empty")
        except Exception:
            composer_ms = self._elapsed_ms(started)
            return response(FallbackPolicy.rag_message(), True, "composer_unavailable")
        finally:
            composer_ms = self._elapsed_ms(started)

        return response(answer.strip(), False, None)
