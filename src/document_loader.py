from pathlib import Path


def load_markdown_documents(data_dir):

    documents = []

    data_path = Path(data_dir)

    if not data_path.exists():
        raise FileNotFoundError(
            f"数据目录不存在：{data_path.resolve()}"
        )

    for file_path in data_path.glob("*.md"):

        text = file_path.read_text(
            encoding="utf-8"
        )

        documents.append({
            "content": text,
            "source": file_path.name
        })

    if not documents:
        raise ValueError(
            f"目录中没有找到 Markdown 文件：{data_path.resolve()}"
        )

    return documents