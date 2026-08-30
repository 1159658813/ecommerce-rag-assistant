import json
from pathlib import Path

import faiss
import numpy as np


class FaissVectorStore:

    def __init__(self, dimension):

        self.dimension = dimension

        # Inner Product Index
        self.index = faiss.IndexFlatIP(
            dimension
        )

        self.documents = []


    def add(
        self,
        embeddings,
        documents
    ):

        embeddings = np.asarray(
            embeddings,
            dtype="float32"
        )

        if embeddings.ndim != 2:
            raise ValueError(
                "embeddings 必须是二维矩阵"
            )

        if embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Embedding维度错误："
                f"期望 {self.dimension}，"
                f"实际 {embeddings.shape[1]}"
            )

        if len(embeddings) != len(documents):
            raise ValueError(
                "Embedding数量必须与Document数量一致"
            )

        self.index.add(
            embeddings
        )

        self.documents.extend(
            documents
        )


    def search(
        self,
        query_embedding,
        top_k=3
    ):

        query_embedding = np.asarray(
            query_embedding,
            dtype="float32"
        )

        if query_embedding.ndim == 1:
            query_embedding = (
                query_embedding.reshape(1, -1)
            )

        scores, indices = self.index.search(
            query_embedding,
            top_k
        )

        results = []

        for score, index in zip(
            scores[0],
            indices[0]
        ):

            # FAISS 在结果不足时可能返回 -1
            if index == -1:
                continue

            document = self.documents[index]

            results.append({
                "score": float(score),
                "document": document
            })

        return results


    def save(
        self,
        index_path,
        metadata_path
    ):

        index_path = Path(index_path)
        metadata_path = Path(metadata_path)

        index_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        metadata_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        faiss.write_index(
            self.index,
            str(index_path)
        )

        with metadata_path.open(
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                self.documents,
                f,
                ensure_ascii=False,
                indent=2
            )


    @classmethod
    def load(
        cls,
        index_path,
        metadata_path
    ):

        index = faiss.read_index(
            str(index_path)
        )

        with open(
            metadata_path,
            "r",
            encoding="utf-8"
        ) as f:

            documents = json.load(f)

        store = cls(
            index.d
        )

        store.index = index
        store.documents = documents

        return store