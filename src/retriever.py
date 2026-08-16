from src.embedding_model import EmbeddingModel
from src.vector_store import FaissVectorStore


class Retriever:

    def __init__(
        self,
        index_path,
        metadata_path,
        embedding_model_name="BAAI/bge-small-zh-v1.5",
        device="cuda"
    ):

        self.embedding_model = EmbeddingModel(
            model_name=embedding_model_name,
            device=device
        )

        self.vector_store = FaissVectorStore.load(
            index_path=index_path,
            metadata_path=metadata_path
        )


    def retrieve(
        self,
        query,
        top_k=3
    ):

        query_embedding = (
            self.embedding_model
            .encode_query(query)
        )

        query_embedding = (
            query_embedding
            .detach()
            .cpu()
            .numpy()
            .astype("float32")
        )

        results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k
        )

        return results