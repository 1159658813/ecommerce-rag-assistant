import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)


class QwenGenerator:

    def __init__(
        self,
        model_name="Qwen/Qwen2.5-1.5B-Instruct",
        device="cuda"
    ):

        self.device = device

        self.tokenizer = (
            AutoTokenizer.from_pretrained(
                model_name
            )
        )

        self.model = (
            AutoModelForCausalLM
            .from_pretrained(
                model_name,
                dtype=torch.float16
            )
            .to(device)
        )

        self.model.eval()


    def generate(
        self,
        messages,
        max_new_tokens=256
    ):

        prompt = (
            self.tokenizer
            .apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        )

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt"
        ).to(self.device)


        with torch.inference_mode():

            outputs = self.model.generate(
                **inputs,

                max_new_tokens=max_new_tokens,

                # V1客服优先稳定性
                do_sample=False,

                # 避免极端重复
                repetition_penalty=1.05,

                # 使用KV Cache
                use_cache=True,

                pad_token_id=(
                    self.tokenizer.eos_token_id
                )
            )


        input_length = (
            inputs["input_ids"].shape[1]
        )

        generated_ids = outputs[
            :,
            input_length:
        ]


        answer = (
            self.tokenizer
            .batch_decode(
                generated_ids,
                skip_special_tokens=True
            )[0]
        )

        return answer.strip()