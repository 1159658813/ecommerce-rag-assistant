from src.generation import AnswerGenerator


generator = AnswerGenerator()


test_cases = [

    {
        "question":
            "银行卡退款是不是当天一定能收到？",

        "evidences": [
            {
                "section":
                    "退款到账时间",

                "content":
                    (
                        "退款发起后，支付宝、微信等"
                        "支付渠道通常会在1至3个工作日内到账；"
                        "银行卡退款可能需要3至7个工作日。"
                    )
            }
        ]
    },

    {
        "question":
            "商品已经影响二次销售了，还能申请七天无理由退货吗？",

        "evidences": [
            {
                "section":
                    "七天无理由退货",

                "content":
                    (
                        "消费者申请七天无理由退货时，"
                        "商品应保持完好。"
                        "如果商品已经影响二次销售，"
                        "通常不符合七天无理由退货条件。"
                    )
            }
        ]
    },

    {
        "question":
            "满100减20，是按商品原价还是优惠后的价格算门槛？",

        "evidences": [
            {
                "section":
                    "使用条件",

                "content":
                    (
                        "满100元减20元优惠券，"
                        "只有参与优惠计算的商品金额"
                        "达到100元时才可以使用。"
                    )
            }
        ]
    },

    {
        "question":
            "平台支持货到付款吗？",

        "evidences": [
            {
                "section":
                    "退款到账时间",

                "content":
                    (
                        "银行卡退款可能需要"
                        "3至7个工作日到账。"
                    )
            }
        ]
    }

]


for i, case in enumerate(
    test_cases,
    start=1
):

    print(
        "\n" + "=" * 100
    )

    print(
        f"Case {i}"
    )

    print(
        "Question:",
        case["question"]
    )

    print(
        "\nEvidence:"
    )

    for evidence in case["evidences"]:

        print(
            evidence["section"],
            ":",
            evidence["content"]
        )

    answer = generator.generate(
        question=case["question"],
        evidences=case["evidences"]
    )

    print(
        "\nAnswer:"
    )

    print(
        answer
    )