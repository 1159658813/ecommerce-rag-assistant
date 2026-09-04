new Vue({
    el: "#app",

    data: function () {
        return {
            question: "",
            sending: false,
            activeEvidence: [],

            quickQuestions: [
                "活动后98元还能使用满100减20吗？",
                "退款后优惠券会退回来吗？",
                "运费是否计入优惠门槛？"
            ],

            messages: [
                {
                    role: "assistant",
                    content:
                        "你好，我是 Ecommerce RAG Assistant。\n" +
                        "你可以询问优惠券、活动规则、退款、物流以及售后相关问题。"
                }
            ]
        };
    },

    methods: {
        sendMessage: function () {
            const text = this.question.trim();

            if (!text || this.sending) {
                return;
            }

            this.messages.push({
                role: "user",
                content: text
            });

            this.question = "";
            this.sending = true;

            this.scrollToBottom();

            // M2 暂时模拟模型请求
            setTimeout(() => {
                this.messages.push(
                    this.createMockAnswer(text)
                );

                this.sending = false;

                this.scrollToBottom();
            }, 900);
        },

        handleKeydown: function (event) {
            if (
                event.key === "Enter" &&
                !event.shiftKey
            ) {
                event.preventDefault();
                this.sendMessage();
            }
        },

        useQuickQuestion: function (question) {
            this.question = question;
            this.sendMessage();
        },

        createMockAnswer: function (question) {
            return {
                role: "assistant",

                content:
                    "这是当前 M2 阶段的前端模拟回答。\n" +
                    "下一阶段我们会把这里替换为真实的 RAG Pipeline 返回结果。",

                verdict: "SUFFICIENT",

                abstained: false,

                evidences: [
                    {
                        rank: 1,
                        source: "优惠券政策.md",
                        section: "优惠门槛计算规则",
                        content:
                            "优惠券门槛按照活动后的实际商品金额进行计算，" +
                            "运费不参与优惠门槛计算。",
                        reranker_score: 8.42
                    },
                    {
                        rank: 2,
                        source: "活动规则.md",
                        section: "实际支付金额",
                        content:
                            "活动优惠后的商品金额用于判断优惠券使用门槛。",
                        reranker_score: 7.93
                    }
                ],

                debugQuestion: question
            };
        },

        scrollToBottom: function () {
            this.$nextTick(() => {
                const container =
                    this.$refs.messagesContainer;

                if (!container) {
                    return;
                }

                container.scrollTop =
                    container.scrollHeight;
            });
        }
    }
});