import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)


class NLIVerifier:

    LABELS = [
        "entailment",
        "neutral",
        "contradiction"
    ]


    def __init__(
        self,
        model_name=(
            "MoritzLaurer/"
            "mDeBERTa-v3-base-xnli-"
            "multilingual-nli-2mil7"
        ),
        device=None
    ):

        if device is None:

            device = (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )


        self.device = device


        print(
            f"正在加载 NLI 模型到 "
            f"{self.device}..."
        )


        self.tokenizer = (
            AutoTokenizer.from_pretrained(
                model_name
            )
        )


        # 注意：
        # 这个模型暂时不要使用 float16。
        self.model = (
            AutoModelForSequenceClassification
            .from_pretrained(
                model_name
            )
            .to(self.device)
        )


        self.model.eval()


        print(
            "NLI 模型加载完成。"
        )


    def verify(
        self,
        evidence,
        candidate_answer
    ):

        """
        evidence:
            Premise

        candidate_answer:
            Hypothesis
        """


        inputs = self.tokenizer(
            evidence,
            candidate_answer,

            truncation=True,

            return_tensors="pt"
        )


        inputs = {
            key: value.to(self.device)
            for key, value
            in inputs.items()
        }


        with torch.inference_mode():

            outputs = self.model(
                **inputs
            )


        logits = (
            outputs.logits[0]
        )


        probabilities = torch.softmax(
            logits.float(),
            dim=-1
        )


        probabilities = (
            probabilities
            .detach()
            .cpu()
            .tolist()
        )


        scores = {
            label: float(probability)
            for label, probability
            in zip(
                self.LABELS,
                probabilities
            )
        }


        predicted_index = int(
            torch.argmax(
                logits
            ).item()
        )


        predicted_label = (
            self.LABELS[
                predicted_index
            ]
        )


        return {
            "label": predicted_label,

            "entailment": (
                scores["entailment"]
            ),

            "neutral": (
                scores["neutral"]
            ),

            "contradiction": (
                scores["contradiction"]
            ),

            "supported": (
                predicted_label
                == "entailment"
            )
        }