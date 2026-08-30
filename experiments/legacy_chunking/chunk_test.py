from pathlib import Path
from transformers import AutoTokenizer

from src.ingestion.document_loader import load_markdown_documents
from experiments.legacy_chunking.text_splitter import TokenTextSplitter


MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw"


print("项目根目录：", PROJECT_ROOT)
print("数据目录：", DATA_DIR)


tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)


documents = load_markdown_documents(
    DATA_DIR
)


splitter = TokenTextSplitter(
    tokenizer=tokenizer,
    chunk_size=120,
    chunk_overlap=20
)


all_chunks = []


for document in documents:

    chunks = splitter.split_text(
        document["content"]
    )

    for index, chunk in enumerate(chunks):

        chunk["source"] = document["source"]
        chunk["chunk_id"] = index

        all_chunks.append(chunk)


for chunk in all_chunks:

    print("=" * 80)

    print(
        f"source: {chunk['source']}"
    )

    print(
        f"chunk_id: {chunk['chunk_id']}"
    )

    print(
        f"tokens: {chunk['token_count']}"
    )

    print()

    print(chunk["text"])


print("\n总文档数：", len(documents))
print("总Chunk数：", len(all_chunks))