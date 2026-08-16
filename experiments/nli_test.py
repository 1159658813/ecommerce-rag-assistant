from src.nli_verifier import (
    NLIVerifier
)


verifier = NLIVerifier()


cases = [

    {
        "evidence": (
            "银行卡退款可能需要"
            "3至7个工作日到账。"
        ),

        "answer": (
            "银行卡退款可能需要"
            "3至7个工作日到账。"
        )
    },


    {
        "evidence": (
            "七天无理由退货的运费"
            "原则上由用户承担。"
        ),

        "answer": (
            "七天无理由退货的运费"
            "最高不能超过20元。"
        )
    },


    {
        "evidence": (
            "银行卡退款可能需要"
            "3至7个工作日到账。"
        ),

        "answer": (
            "银行卡退款一定当天到账。"
        )
    }

]


for index, case in enumerate(
    cases,
    start=1
):

    result = verifier.verify(
        evidence=case["evidence"],
        candidate_answer=(
            case["answer"]
        )
    )


    print(
        "\n" + "=" * 80
    )

    print(
        "Case:",
        index
    )

    print(
        "Evidence:",
        case["evidence"]
    )

    print(
        "Candidate:",
        case["answer"]
    )

    print()

    print(
        "Label:",
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