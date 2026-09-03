import re


def parse_markdown_sections(text: str):
    """
    将 Markdown 文档解析成：
    文档标题 + 二级标题 Section + Section 内容
    """

    title = ""
    current_section = ""
    current_lines = []

    sections = []

    lines = text.splitlines()

    for line in lines:

        stripped = line.strip()

        # 一级标题：文档标题
        if stripped.startswith("# ") and not stripped.startswith("## "):

            title = stripped[2:].strip()
            continue

        # 二级标题：Section
        if stripped.startswith("## "):

            # 保存上一个 Section
            if current_section and current_lines:

                content = "\n".join(
                    current_lines
                ).strip()

                if content:
                    sections.append({
                        "title": title,
                        "section": current_section,
                        "content": content
                    })

            current_section = stripped[3:].strip()
            current_lines = []

            continue

        if stripped:
            current_lines.append(stripped)

        else:
            # 保留段落边界
            if current_lines and current_lines[-1] != "":
                current_lines.append("")

    # 保存最后一个 Section
    if current_section and current_lines:

        content = "\n".join(
            current_lines
        ).strip()

        if content:
            sections.append({
                "title": title,
                "section": current_section,
                "content": content
            })

    return sections

class MarkdownChunker:

    def __init__(
        self,
        tokenizer,
        chunk_size=120,
        chunk_overlap=20
    ):
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap


    def count_tokens(self, text):

        return len(
            self.tokenizer.encode(
                text,
                add_special_tokens=False
            )
        )


    def build_chunk_text(
        self,
        title,
        section,
        body
    ):

        return (
            f"{title}\n"
            f"{section}\n\n"
            f"{body}"
        )


    def split_long_text(
        self,
        text
    ):
        """
        当一个单独段落本身就超过 chunk_size 时，
        才退化为 Token Sliding Window。
        """

        token_ids = self.tokenizer.encode(
            text,
            add_special_tokens=False
        )

        chunks = []

        step = (
            self.chunk_size
            - self.chunk_overlap
        )

        for start in range(
            0,
            len(token_ids),
            step
        ):

            end = start + self.chunk_size

            current_ids = token_ids[
                start:end
            ]

            chunks.append(
                self.tokenizer.decode(
                    current_ids,
                    skip_special_tokens=True
                )
            )

            if end >= len(token_ids):
                break

        return chunks
    def split_section(
        self,
        title,
        section,
        content
    ):

        paragraphs = [
            paragraph.strip()
            for paragraph in content.split("\n\n")
            if paragraph.strip()
        ]

        chunks = []

        current_paragraphs = []

        for paragraph in paragraphs:

            # 一个段落本身已经超过上限
            paragraph_tokens = self.count_tokens(
                paragraph
            )

            if paragraph_tokens > self.chunk_size:

                # 先保存已经累计的内容
                if current_paragraphs:

                    body = "\n\n".join(
                        current_paragraphs
                    )

                    chunks.append(
                        self.build_chunk_text(
                            title,
                            section,
                            body
                        )
                    )

                    current_paragraphs = []

                # 超长段落再使用 Token 切分
                sub_chunks = self.split_long_text(
                    paragraph
                )

                for sub_chunk in sub_chunks:

                    chunks.append(
                        self.build_chunk_text(
                            title,
                            section,
                            sub_chunk
                        )
                    )

                continue


            candidate_paragraphs = (
                current_paragraphs
                + [paragraph]
            )

            candidate_body = "\n\n".join(
                candidate_paragraphs
            )

            candidate_text = self.build_chunk_text(
                title,
                section,
                candidate_body
            )


            # 加入当前段落后仍然没有超过上限
            if (
                self.count_tokens(candidate_text)
                <= self.chunk_size
            ):

                current_paragraphs.append(
                    paragraph
                )


            else:

                # 先保存前面的 Chunk
                if current_paragraphs:

                    body = "\n\n".join(
                        current_paragraphs
                    )

                    chunks.append(
                        self.build_chunk_text(
                            title,
                            section,
                            body
                        )
                    )

                # 当前段落成为新 Chunk 的开始
                current_paragraphs = [
                    paragraph
                ]


        # 保存最后剩余内容
        if current_paragraphs:

            body = "\n\n".join(
                current_paragraphs
            )

            chunks.append(
                self.build_chunk_text(
                    title,
                    section,
                    body
                )
            )


        return chunks
    def split_document(self, text):

        sections = parse_markdown_sections(
            text
        )

        chunks = []

        for section_data in sections:

            section_chunks = self.split_section(
                title=section_data["title"],
                section=section_data["section"],
                content=section_data["content"]
            )

            for chunk_text in section_chunks:

                chunks.append({
                    "text": chunk_text,
                    "title": section_data["title"],
                    "section": section_data["section"],
                    "token_count": self.count_tokens(
                        chunk_text
                    )
                })

        return chunks