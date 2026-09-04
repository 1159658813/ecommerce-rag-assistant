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
        sendMessage: async function () {
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

            try {
                const response = await axios.post(
                    "/api/v1/query",
                    {
                        question: text
                    }
                );

                const data = response.data;

                this.messages.push({
                    role: "assistant",

                    content:
                        data.answer ||
                        "系统没有返回回答内容。",

                    verdict:
                    data.verdict,

                    abstained:
                        Boolean(data.abstained),

                    abstainReason:
                    data.abstain_reason,

                    evidences:
                        Array.isArray(data.evidences)
                            ? data.evidences
                            : [],

                    requestId:
                        response.headers[
                            "x-request-id"
                            ]
                });

            } catch (error) {
                this.handleRequestError(error);
            } finally {
                this.sending = false;

                this.scrollToBottom();
            }
        },
        handleRequestError: function (error) {
            let message =
                "请求失败，请稍后重试。";

            let requestId = null;

            if (
                error.response &&
                error.response.data &&
                error.response.data.error
            ) {
                const errorData =
                    error.response.data.error;

                message =
                    errorData.message ||
                    message;

                requestId =
                    errorData.request_id;
            } else if (error.request) {
                message =
                    "无法连接到 RAG 服务，请确认后端已经启动。";
            }

            this.messages.push({
                role: "assistant",

                content: message,

                isError: true,

                requestId: requestId
            });

            this.$message.error(
                "RAG 请求失败"
            );
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