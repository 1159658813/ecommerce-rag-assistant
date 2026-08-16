import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)


# ============================================================
# Config
# ============================================================

MODEL_NAME = "BAAI/bge-reranker-v2-m3"

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# Load Tokenizer
# ============================================================

print("正在加载 Reranker Tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)


# ============================================================
# Load Model
# ============================================================

print("正在加载 Reranker Model...")

dtype = (
    torch.float16
    if DEVICE == "cuda"
    else torch.float32
)


model = (
    AutoModelForSequenceClassification
    .from_pretrained(
        MODEL_NAME,
        dtype=dtype
    )
    .to(DEVICE)
)

model.eval()


print(
    "Reranker 加载完成"
)

print(
    "Device:",
    DEVICE
)


# ============================================================
# Test Query
# ============================================================

query = (
    "满100减20的100元门槛，"
    "是按商品原价还是优惠后的金额计算？"
)


passages = [

    (
        "优惠券使用条件。"
        "例如满100元减20元优惠券，"
        "只有参与优惠计算的商品金额达到100元时"
        "才可以使用。"
    ),

    (
        "优惠券有效期。"
        "优惠券只能在有效期内使用。"
    ),

    (
        "普通现货商品一般会在付款后的"
        "48小时内完成发货。"
    )
]


# ============================================================
# Query-Document Pairs
# ============================================================

queries = [
    query
    for _ in passages
]


inputs = tokenizer(
    queries,
    passages,

    padding=True,
    truncation=True,

    # 我们当前 Chunk 很短，
    # 512 已经完全足够做第一版实验
    max_length=512,

    return_tensors="pt"
)


inputs = {
    key: value.to(DEVICE)
    for key, value in inputs.items()
}


print(
    "\ninput_ids shape:",
    inputs["input_ids"].shape
)


# ============================================================
# Cross-Encoder Forward
# ============================================================

with torch.inference_mode():

    outputs = model(
        **inputs
    )


# 每个 Query-Document Pair
# 得到一个 relevance logit
logits = (
    outputs
    .logits
    .view(-1)
    .float()
)


# sigmoid只是为了方便观察成0~1
scores = torch.sigmoid(
    logits
)


# ============================================================
# Ranking
# ============================================================

ranking = torch.argsort(
    logits,
    descending=True
)


print("\n")
print("=" * 90)
print("Reranker Result")
print("=" * 90)


for rank, index in enumerate(
    ranking,
    start=1
):

    index = index.item()

    print(
        f"\nRank {rank}"
    )

    print(
        "Raw Logit:",
        f"{logits[index].item():.4f}"
    )

    print(
        "Normalized Score:",
        f"{scores[index].item():.4f}"
    )

    print(
        "Passage:"
    )

    print(
        passages[index]
    )