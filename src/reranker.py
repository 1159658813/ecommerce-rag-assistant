import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)


class BGEReranker:

    def __init__(
        self,
        model_name="BAAI/bge-reranker-v2-m3",
        device=None,
        max_length=512
    ):

        if device is None:
            device = (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

        self.device = device
        self.max_length = max_length


        # ==========================================
        # Tokenizer
        # ==========================================

        self.tokenizer = (
            AutoTokenizer.from_pretrained(
                model_name
            )
        )


        # ==========================================
        # Model
        # ==========================================

        dtype = (
            torch.float16
            if device == "cuda"
            else torch.float32
        )


        self.model = (
            AutoModelForSequenceClassification
            .from_pretrained(
                model_name,
                dtype=dtype
            )
            .to(device)
        )


        self.model.eval()


    def compute_scores(
        self,
        query,
        documents
    ):
        """
        对一个 query 和多个 document 进行 Cross-Encoder 打分。

        Parameters
        ----------
        query:
            str

        documents:
            list[str]

        Returns
        -------
        list[dict]
        """

        if not documents:
            return []


        # 每一个 document 都和同一个 query 组成一对
        queries = [
            query
            for _ in documents
        ]


        # ==========================================
        # Tokenization
        #
        # 每一个样本实际上都是：
        #
        # query + document
        #
        # Cross-Encoder 会让它们一起经过 Transformer。
        # ==========================================

        inputs = self.tokenizer(
            queries,
            documents,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )


        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }


        # ==========================================
        # Forward
        # ==========================================

        with torch.inference_mode():

            outputs = self.model(
                **inputs
            )


        # 每个 query-document pair 对应一个 logit
        logits = (
            outputs
            .logits
            .view(-1)
            .float()
        )


        # 仅用于观察，不能理解成 Answerability Probability
        normalized_scores = torch.sigmoid(
            logits
        )


        results = []


        for logit, normalized in zip(
            logits,
            normalized_scores
        ):

            results.append({
                "score": logit.item(),
                "normalized_score":
                    normalized.item()
            })


        return results


    def rerank(
        self,
        query,
        candidates,
        top_k=None
    ):
        """
        对 Retriever 返回的 candidates 重新排序。

        candidate 的结构预计为：

        {
            "score": dense_score,
            "document": {...}
        }
        """

        if not candidates:
            return []


        documents = [
            candidate["document"]["text"]
            for candidate in candidates
        ]


        rerank_scores = self.compute_scores(
            query=query,
            documents=documents
        )


        results = []


        for candidate, rerank_result in zip(
            candidates,
            rerank_scores
        ):

            results.append({

                # 第一阶段 Dense Retriever 分数
                "dense_score":
                    candidate["score"],

                # 第二阶段 Cross-Encoder 原始 logit
                "rerank_score":
                    rerank_result["score"],

                # 仅便于调试查看
                "rerank_normalized_score":
                    rerank_result[
                        "normalized_score"
                    ],

                "document":
                    candidate["document"]
            })


        # ==========================================
        # 关键：
        # 最终按照 rerank_score，而不是 dense_score 排序
        # ==========================================

        results.sort(
            key=lambda x: x["rerank_score"],
            reverse=True
        )


        if top_k is not None:
            results = results[:top_k]


        return results