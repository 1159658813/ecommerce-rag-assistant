import json
import sys

from pathlib import Path


# ============================================================
# Project Path
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)


if str(PROJECT_ROOT) not in sys.path:

    sys.path.insert(
        0,
        str(PROJECT_ROOT)
    )


from src.retriever import Retriever

from src.llm import QwenGenerator

from src.candidate_answerer import (
    CandidateAnswerer
)

from src.nli_verifier import (
    NLIVerifier
)


# ============================================================
# Paths
# ============================================================

QUESTIONS_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "questions.json"
)


INDEX_PATH = (
    PROJECT_ROOT
    / "data"
    / "index"
    / "knowledge.faiss"
)


METADATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "index"
    / "chunks.json"
)


# ============================================================
# Load Dataset
# ============================================================

with QUESTIONS_PATH.open(
    "r",
    encoding="utf-8"
) as f:

    questions = json.load(f)


print(
    "Questions:",
    len(questions)
)


# ============================================================
# Load Models
# ============================================================

print(
    "\n正在加载 Retriever..."
)


retriever = Retriever(
    index_path=INDEX_PATH,
    metadata_path=METADATA_PATH
)


print(
    "\n正在加载 Qwen 1.5B..."
)


generator = QwenGenerator(
    model_name=(
        "Qwen/"
        "Qwen2.5-1.5B-Instruct"
    )
)


candidate_answerer = (
    CandidateAnswerer(
        generator=generator
    )
)


print(
    "\n正在加载 NLI Verifier..."
)


nli_verifier = NLIVerifier()


print(
    "\n全部模型加载完成。\n"
)


# ============================================================
# Metrics
# ============================================================

tp = 0
tn = 0
fp = 0
fn = 0


error_records = []


# ============================================================
# Evaluation
# ============================================================

for item in questions:

    question = (
        item["question"]
    )

    truth = (
        item["answerable"]
    )


    # --------------------------------------------------------
    # 1. Retrieve Top-1
    # --------------------------------------------------------

    results = retriever.retrieve(
        query=question,
        top_k=1
    )


    if not results:

        predicted = False

        candidate_answer = (
            "NO_RETRIEVAL_RESULT"
        )

        nli_result = None

        section = None

        dense_score = None


    else:

        result = results[0]

        document = (
            result["document"]
        )

        section = (
            document["section"]
        )

        dense_score = (
            result["score"]
        )


        # ----------------------------------------------------
        # 2. Generate Candidate Answer
        # ----------------------------------------------------

        candidate_result = (
            candidate_answerer
            .generate(
                query=question,
                document=document
            )
        )


        candidate_answer = (
            candidate_result["answer"]
        )


        # ----------------------------------------------------
        # 3. UNKNOWN -> Abstain
        # ----------------------------------------------------

        if candidate_result[
            "unknown"
        ]:

            predicted = False

            nli_result = None


        else:

            # ------------------------------------------------
            # 4. NLI Verification
            # ------------------------------------------------

            nli_result = (
                nli_verifier.verify(
                    evidence=(
                        document["text"]
                    ),

                    candidate_answer=(
                        candidate_answer
                    )
                )
            )


            predicted = (
                nli_result["supported"]
            )


    # --------------------------------------------------------
    # Confusion Matrix
    # --------------------------------------------------------

    if truth and predicted:

        tp += 1


    elif (
        not truth
        and not predicted
    ):

        tn += 1


    elif (
        not truth
        and predicted
    ):

        fp += 1


    elif (
        truth
        and not predicted
    ):

        fn += 1


    # --------------------------------------------------------
    # Error Record
    # --------------------------------------------------------

    if truth != predicted:

        error_records.append({

            "id": item["id"],

            "question":
                question,

            "truth":
                truth,

            "prediction":
                predicted,

            "candidate_answer":
                candidate_answer,

            "section":
                section,

            "dense_score":
                dense_score,

            "nli_result":
                nli_result
        })


    # --------------------------------------------------------
    # Print Case
    # --------------------------------------------------------

    print(
        "\n" + "=" * 110
    )


    print(
        f"[{item['id']}] "
        f"{question}"
    )


    print(
        "Ground Truth:",
        truth
    )


    print(
        "Prediction:",
        predicted
    )


    print(
        "Top-1:",
        section
    )


    if dense_score is not None:

        print(
            "Dense Score:",
            f"{dense_score:.4f}"
        )


    print(
        "Candidate Answer:",
        repr(candidate_answer)
    )


    if nli_result is not None:

        print(
            "NLI Label:",
            nli_result["label"]
        )

        print(
            "Entailment:",
            f"{nli_result['entailment']:.4f}"
        )

        print(
            "Neutral:",
            f"{nli_result['neutral']:.4f}"
        )

        print(
            "Contradiction:",
            f"{nli_result['contradiction']:.4f}"
        )


# ============================================================
# Metrics
# ============================================================

total = (
    tp
    + tn
    + fp
    + fn
)


accuracy = (
    (tp + tn) / total
    if total
    else 0
)


precision = (
    tp / (tp + fp)
    if tp + fp
    else 0
)


recall = (
    tp / (tp + fn)
    if tp + fn
    else 0
)


f1 = (
    (
        2
        * precision
        * recall
        / (
            precision
            + recall
        )
    )

    if (
        precision
        + recall
    )

    else 0
)


# ============================================================
# Final Result
# ============================================================

print("\n\n")

print("=" * 110)

print(
    "NLI Verification Pipeline "
    "Evaluation"
)

print("=" * 110)


print(
    f"TP: {tp}"
)

print(
    f"TN: {tn}"
)

print(
    f"FP: {fp}"
)

print(
    f"FN: {fn}"
)

print()


print(
    f"Accuracy: "
    f"{accuracy:.2%}"
)

print(
    f"Precision: "
    f"{precision:.2%}"
)

print(
    f"Recall: "
    f"{recall:.2%}"
)

print(
    f"F1: "
    f"{f1:.2%}"
)


# ============================================================
# Error Cases
# ============================================================

print("\n\n")

print("=" * 110)

print(
    "NLI Pipeline Error Cases"
)

print("=" * 110)


if not error_records:

    print(
        "\nNo errors."
    )


for record in error_records:

    print(
        "\n" + "-" * 110
    )


    error_type = (
        "FALSE POSITIVE"
        if (
            record["prediction"]
            and not record["truth"]
        )
        else
        "FALSE NEGATIVE"
    )


    print(
        "Type:",
        error_type
    )

    print(
        "ID:",
        record["id"]
    )

    print(
        "Question:",
        record["question"]
    )

    print(
        "Top-1:",
        record["section"]
    )


    if (
        record["dense_score"]
        is not None
    ):

        print(
            "Dense Score:",
            f"{record['dense_score']:.4f}"
        )


    print(
        "Candidate Answer:",
        repr(
            record[
                "candidate_answer"
            ]
        )
    )


    if (
        record["nli_result"]
        is not None
    ):

        result = (
            record["nli_result"]
        )


        print(
            "NLI Label:",
            result["label"]
        )

        print(
            "Entailment:",
            f"{result['entailment']:.4f}"
        )

        print(
            "Neutral:",
            f"{result['neutral']:.4f}"
        )

        print(
            "Contradiction:",
            f"{result['contradiction']:.4f}"
        )