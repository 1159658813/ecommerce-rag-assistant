from pathlib import Path

import torch

from transformers import AutoTokenizer

from src.ingestion.document_loader import (
    load_markdown_documents
)

from src.ingestion.markdown_splitter import (
    MarkdownChunker
)

from src.retrieval.embedding_model import (
    EmbeddingModel
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw"


# -------------------------
# Chunking
# -------------------------

tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen2.5-0.5B-Instruct"
)

documents = load_markdown_documents(
    DATA_DIR
)

chunker = MarkdownChunker(
    tokenizer=tokenizer,
    chunk_size=120,
    chunk_overlap=20
)


chunks = []


for document in documents:

    document_chunks = chunker.split_document(
        document["content"]
    )

    for chunk_id, chunk in enumerate(document_chunks):

        chunk["source"] = document["source"]
        chunk["chunk_id"] = chunk_id

        chunks.append(chunk)


# -------------------------
# Embedding
# -------------------------

embedding_model = EmbeddingModel()


chunk_texts = [
    chunk["text"]
    for chunk in chunks
]


document_embeddings = (
    embedding_model.encode_documents(
        chunk_texts
    )
)


print(
    "Document Embeddings:",
    document_embeddings.shape
)


# -------------------------
# Query
# -------------------------

query = "七天无理由退货的运费谁承担？"


query_embedding = (
    embedding_model.encode_query(query)
)


print(
    "Query Embedding:",
    query_embedding.shape
)


# -------------------------
# Similarity
# -------------------------

scores = (
    query_embedding
    @ document_embeddings.T
)


scores = scores[0]


# -------------------------
# Top-K
# -------------------------

TOP_K = 3


top_scores, top_indices = torch.topk(
    scores,
    k=min(TOP_K, len(chunks))
)


print("\n用户问题：")
print(query)


print("\n===== Top-K Retrieval =====")


for rank, (
    score,
    index
) in enumerate(
    zip(top_scores, top_indices),
    start=1
):

    index = index.item()
    score = score.item()

    chunk = chunks[index]

    print("\n" + "=" * 80)

    print("Rank:", rank)
    print("Score:", round(score, 4))

    print(
        "Source:",
        chunk["source"]
    )

    print(
        "Section:",
        chunk["section"]
    )

    print()

    print(chunk["text"])