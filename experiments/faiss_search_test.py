from pathlib import Path

from src.retrieval.embedding_model import (
    EmbeddingModel
)

from src.retrieval.vector_store import (
    FaissVectorStore
)


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


# ======================
# Load Index
# ======================

vector_store = (
    FaissVectorStore.load(
        index_path=INDEX_PATH,
        metadata_path=METADATA_PATH
    )
)


print(
    "FAISS向量数量：",
    vector_store.index.ntotal
)


# ======================
# Query Embedding
# ======================

embedding_model = EmbeddingModel()


query = "银行卡退款多久到账？"


query_embedding = (
    embedding_model.encode_query(
        query
    )
)


query_embedding = (
    query_embedding
    .detach()
    .cpu()
    .numpy()
    .astype("float32")
)


# ======================
# Search
# ======================

results = vector_store.search(
    query_embedding,
    top_k=3
)


print("\n用户问题：")
print(query)


for rank, result in enumerate(
    results,
    start=1
):

    document = result["document"]

    print("\n" + "=" * 70)

    print("Rank:", rank)

    print(
        "Score:",
        round(
            result["score"],
            4
        )
    )

    print(
        "Source:",
        document["source"]
    )

    print(
        "Section:",
        document["section"]
    )

    print()

    print(
        document["text"]
    )