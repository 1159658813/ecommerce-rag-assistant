from sentence_transformers import SentenceTransformer
import numpy as np


MODEL_NAME = "BAAI/bge-small-zh-v1.5"


model = SentenceTransformer(
    MODEL_NAME,
    device="cuda"
)


texts = [
    "银行卡退款多久到账？",
    "银行卡退款可能需要3至7个工作日。",
    "优惠券超过有效期后会自动失效。"
]


embeddings = model.encode(
    texts,
    normalize_embeddings=True
)


print("Embedding shape:")
print(embeddings.shape)

print("\n第一个向量前10维：")
print(embeddings[0][:10])




query_embedding = embeddings[0]
refund_embedding = embeddings[1]
coupon_embedding = embeddings[2]


refund_similarity = np.dot(
    query_embedding,
    refund_embedding
)

coupon_similarity = np.dot(
    query_embedding,
    coupon_embedding
)


print("\n退款相关相似度：")
print(refund_similarity)

print("\n优惠券相关相似度：")
print(coupon_similarity)

print(type(embeddings))