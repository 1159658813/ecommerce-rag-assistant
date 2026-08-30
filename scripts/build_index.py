from pathlib import Path

from transformers import AutoTokenizer

from src.ingestion.document_loader import (
    load_markdown_documents,
)

from src.ingestion.markdown_splitter import (
    MarkdownChunker,
)

from src.retrieval.embedding_model import (
    EmbeddingModel,
)

from src.retrieval.vector_store import (
    FaissVectorStore,
)


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)


DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
)


INDEX_DIR = (
    PROJECT_ROOT
    / "data"
    / "index"
)


INDEX_PATH = (
    INDEX_DIR
    / "knowledge.faiss"
)


METADATA_PATH = (
    INDEX_DIR
    / "chunks.json"
)


# ======================
# 1. Load Documents
# ======================

documents = load_markdown_documents(
    DATA_DIR
)


# ======================
# 2. Chunking
# ======================

tokenizer = (
    AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-0.5B-Instruct"
    )
)


chunker = MarkdownChunker(
    tokenizer=tokenizer,
    chunk_size=120,
    chunk_overlap=20
)


chunks = []


for document in documents:

    document_chunks = (
        chunker.split_document(
            document["content"]
        )
    )

    for chunk_id, chunk in enumerate(
        document_chunks
    ):

        chunk["source"] = (
            document["source"]
        )

        chunk["chunk_id"] = chunk_id

        chunks.append(chunk)


print(
    f"共生成 {len(chunks)} 个 Chunk"
)


# ======================
# 3. Embedding
# ======================

embedding_model = EmbeddingModel()


texts = [
    chunk["text"]
    for chunk in chunks
]


embeddings = (
    embedding_model.encode_documents(
        texts
    )
)


embeddings = (
    embeddings
    .detach()
    .cpu()
    .numpy()
    .astype("float32")
)


print(
    "Embedding shape:",
    embeddings.shape
)


# ======================
# 4. Build FAISS
# ======================

dimension = embeddings.shape[1]


vector_store = FaissVectorStore(
    dimension=dimension
)


vector_store.add(
    embeddings=embeddings,
    documents=chunks
)


print(
    "FAISS中向量数量：",
    vector_store.index.ntotal
)


# ======================
# 5. Save
# ======================

vector_store.save(
    index_path=INDEX_PATH,
    metadata_path=METADATA_PATH
)


print("\n索引构建完成")

print(
    "FAISS:",
    INDEX_PATH
)

print(
    "Metadata:",
    METADATA_PATH
)