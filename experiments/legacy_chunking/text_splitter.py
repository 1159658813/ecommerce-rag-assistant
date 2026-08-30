class TokenTextSplitter:

    def __init__(
        self,
        tokenizer,
        chunk_size=120,
        chunk_overlap=20
    ):
        if chunk_overlap >= chunk_size:
            raise ValueError(
                "chunk_overlap 必须小于 chunk_size"
            )

        self.tokenizer = tokenizer
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap


    def split_text(self, text):

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

            chunk_token_ids = token_ids[
                start:end
            ]

            chunk_text = self.tokenizer.decode(
                chunk_token_ids,
                skip_special_tokens=True
            )

            chunks.append({
                "text": chunk_text,
                "token_count": len(
                    chunk_token_ids
                ),
                "start_token": start,
                "end_token": min(
                    end,
                    len(token_ids)
                )
            })

            if end >= len(token_ids):
                break

        return chunks