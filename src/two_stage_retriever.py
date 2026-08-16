class TwoStageRetriever:

    def __init__(
        self,
        dense_retriever,
        reranker,
        candidate_k=5,
        final_k=3
    ):

        self.dense_retriever = (
            dense_retriever
        )

        self.reranker = reranker

        self.candidate_k = candidate_k
        self.final_k = final_k


    def retrieve(
        self,
        query,
        candidate_k=None,
        final_k=None
    ):

        if candidate_k is None:
            candidate_k = self.candidate_k

        if final_k is None:
            final_k = self.final_k


        # ==========================================
        # Stage 1
        #
        # Dense Retrieval
        # ==========================================

        candidates = (
            self.dense_retriever.retrieve(
                query=query,
                top_k=candidate_k
            )
        )


        # ==========================================
        # Stage 2
        #
        # Cross-Encoder Reranking
        # ==========================================

        reranked_results = (
            self.reranker.rerank(
                query=query,
                candidates=candidates,
                top_k=final_k
            )
        )


        return reranked_results