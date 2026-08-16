from pathlib import Path

from transformers import AutoTokenizer

from src.document_loader import (
    load_markdown_documents
)

from src.markdown_splitter import (
    MarkdownChunker
)


MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


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


tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)


documents = load_markdown_documents(
    DATA_DIR
)


chunker = MarkdownChunker(
    tokenizer=tokenizer,
    chunk_size=120,
    chunk_overlap=20
)


all_chunks = []


for document in documents:

    chunks = chunker.split_document(
        document["content"]
    )

    for chunk_id, chunk in enumerate(chunks):

        chunk["source"] = document["source"]
        chunk["chunk_id"] = chunk_id

        all_chunks.append(chunk)


for chunk in all_chunks:

    print("=" * 80)

    print(
        "source:",
        chunk["source"]
    )

    print(
        "title:",
        chunk["title"]
    )

    print(
        "section:",
        chunk["section"]
    )

    print(
        "chunk_id:",
        chunk["chunk_id"]
    )

    print(
        "tokens:",
        chunk["token_count"]
    )

    print()

    print(chunk["text"])


print("\n总文档数：", len(documents))
print("总Chunk数：", len(all_chunks))