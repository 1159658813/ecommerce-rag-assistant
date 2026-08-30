from pathlib import Path
from src.retrieval.retriever import Retriever
from experiments.legacy_rag_v1.llm import QwenGenerator
from experiments.legacy_rag_v1.rag import RAGSystem

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
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


print("正在加载 Embedding Model...")

retriever = Retriever(
    index_path=INDEX_PATH,
    metadata_path=METADATA_PATH
)


print("正在加载 Qwen...")

generator = QwenGenerator()


rag = RAGSystem(
    retriever=retriever,
    generator=generator
)


print("\n电商 RAG 智能客服启动成功")
print("输入 exit 退出\n")


while True:

    query = input("用户：").strip()

    if query.lower() == "exit":
        break

    if not query:
        continue

    result = rag.ask(
        query=query,
        candidate_k=3
    )


    print("\n客服：")
    print(result["answer"])


    print("\n--- 检索来源 ---")

    for rank, retrieval in enumerate(
        result["retrieval_results"],
        start=1
    ):

        doc = retrieval["document"]

        print(
            f"[资料{rank}] "
            f"{doc['source']} > "
            f"{doc['section']} "
            f"(score={retrieval['score']:.4f})"
        )

    print("\n" + "=" * 80 + "\n")