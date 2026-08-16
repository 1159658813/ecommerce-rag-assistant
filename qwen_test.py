import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16
).to("cuda")

model.eval()


messages = [
    {
        "role": "system",
        "content": (
            "你是一名大语言模型与自然语言处理领域的AI助手。"
            "用户的问题默认讨论LLM、NLP和机器学习领域。"
        )
    },
    {
        "role": "user",
        "content": "什么是RAG？"
    }
]


prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)


inputs = tokenizer(
    prompt,
    return_tensors="pt"
).to("cuda")


with torch.inference_mode():
    outputs = model(**inputs)


print("logits shape:")
print(outputs.logits.shape)


# 取最后一个位置的 logits
next_token_logits = outputs.logits[0, -1, :]


print("\n下一个 Token logits shape:")
print(next_token_logits.shape)


# logits -> probability
probabilities = torch.softmax(
    next_token_logits.float(),
    dim=-1
)


# 找概率最高的 10 个 Token
top_probs, top_ids = torch.topk(
    probabilities,
    k=10
)


print("\n===== 下一个 Token Top-10 =====")

for probability, token_id in zip(top_probs, top_ids):

    token_id = token_id.item()
    probability = probability.item()

    token_text = tokenizer.decode([token_id])

    print(
        f"Token ID: {token_id:<8} "
        f"Probability: {probability:.4%} "
        f"Token: {repr(token_text)}"
    )