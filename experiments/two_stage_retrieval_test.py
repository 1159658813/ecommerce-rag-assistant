from pathlib import Path

from src.retrieval.retriever import (
    Retriever
)

from src.retrieval.reranker import (
    BGEReranker
)

from src.retrieval.two_stage_retriever import (
    TwoStageRetriever
)


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)


INDEX_PATH = (
    PROJECT_ROOT
    / "data"
    / "index"
    / "knowledge.faiss"
)


METADATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "index"
    / "chunks.json"
)


# ============================================================
# Dense Retriever
# ============================================================

print(
    "正在加载 Dense Retriever..."
)


dense_retriever = Retriever(
    index_path=INDEX_PATH,
    metadata_path=METADATA_PATH
)


# ============================================================
# Reranker
# ============================================================

print(
    "正在加载 Reranker..."
)


reranker = BGEReranker()


# ============================================================
# Two-Stage Retriever
# ============================================================

retriever = TwoStageRetriever(
    dense_retriever=dense_retriever,
    reranker=reranker,
    candidate_k=5,
    final_k=3
)


# ============================================================
# Query
# ============================================================

query = "银行卡退款多久到账？"


# ============================================================
# Stage 1
#
# 为了学习，先单独观察 Dense 排名
# ============================================================

dense_results = (
    dense_retriever.retrieve(
        query=query,
        top_k=5
    )
)


print("\n")

print("=" * 100)
print("Stage 1 - Dense Retrieval")
print("=" * 100)


for rank, result in enumerate(
    dense_results,
    start=1
):

    doc = result["document"]

    print(
        f"\nRank {rank}"
    )

    print(
        f"Dense Score: "
        f"{result['score']:.4f}"
    )

    print(
        f"Section: "
        f"{doc['section']}"
    )


# ============================================================
# Stage 2
# ============================================================

reranked_results = (
    retriever.retrieve(
        query=query
    )
)


print("\n")

print("=" * 100)
print("Stage 2 - Cross-Encoder Reranking")
print("=" * 100)


for rank, result in enumerate(
    reranked_results,
    start=1
):

    doc = result["document"]

    print(
        f"\nRank {rank}"
    )

    print(
        f"Dense Score: "
        f"{result['dense_score']:.4f}"
    )

    print(
        f"Rerank Logit: "
        f"{result['rerank_score']:.4f}"
    )

    print(
        f"Rerank Normalized: "
        f"{result['rerank_normalized_score']:.4f}"
    )

    print(
        f"Section: "
        f"{doc['section']}"
    )