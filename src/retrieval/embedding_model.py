from sentence_transformers import SentenceTransformer
import torch


class EmbeddingModel:

    def __init__(
        self,
        model_name="BAAI/bge-small-zh-v1.5",
        device=None
    ):

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = SentenceTransformer(
            model_name,
            device=device
        )


    def encode_documents(
        self,
        texts
    ):

        return self.model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_tensor=True,
            show_progress_bar=True
        )


    def encode_query(
        self,
        query
    ):

        embedding = self.model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_tensor=True
        )

        return embedding