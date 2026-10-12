new Vue({
    el: "#app",

    data: function () {
        return {
            question: "",
            sending: false,
            activeEvidence: [],
            currentUserId: "U001",
            conversationId: null,

            quickQuestions: [
                "活动后98元还能使用满100减20吗？",
                "帮我查一下 ORD202610002 现在到哪里了？",
                "退款后优惠券会退回来吗？"
            ],

            messages: [
                {
                    role: "assistant",
                    content:
                        "你好，我是 Ecommerce RAG Assistant。\n" +
                        "你可以询问优惠券、活动规则、退款、物流以及售后相关问题。"
                }
            ],
            serviceHealthy: null,
            lastLatency: null
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
            const startedAt = performance.now();

            try {
                const response = await axios.post(
                    "/api/v2/chat",
                    {
                        question: text,
                        user_id: this.currentUserId,
                        conversation_id: this.conversationId
                    }
                );
                const latencyMs =
                    performance.now() - startedAt;

                const data = response.data;
                if (data.conversation_id) {
                    this.conversationId =
                        data.conversation_id;
                }


                const handlerResult =
                    data.result || {};

                this.messages.push({
                    role: "assistant",

                    content:
                        data.answer ||
                        "系统没有返回回答内容。",

                    intent:
                        data.intent || null,

                    handler:
                        data.handler || null,

                    memoryUsed:
                        Boolean(data.memory_used),

                    conversationId:
                        data.conversation_id || null,

                    verdict:
                        handlerResult.verdict || null,

                    abstained:
                        Boolean(
                            handlerResult.abstained
                        ),

                    abstainReason:
                        handlerResult.abstain_reason || null,

                    evidences:
                        Array.isArray(
                            handlerResult.evidences
                        )
                            ? handlerResult.evidences
                            : [],

                    toolsCalled:
                        Array.isArray(
                            handlerResult.tools_called
                        )
                            ? handlerResult.tools_called
                            : [],

                    requestId:
                        response.headers[
                            "x-request-id"
                            ],

                    latencyMs:
                    latencyMs
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
        },
        checkHealth: async function () {
            try {
                await axios.get("/health");

                this.serviceHealthy = true;
            } catch (error) {
                this.serviceHealthy = false;
            }
        },
        formatAbstainReason: function (reason) {
            const reasonMap = {
                evidence_insufficient:
                    "当前知识库中的证据不足以可靠回答该问题。"
            };

            return (
                reasonMap[reason] ||
                reason ||
                "当前证据不足，系统已主动停止生成答案。"
            );
        },
        clearConversation: function () {
            this.messages = [
                {
                    role: "assistant",
                    content:
                        "你好，我是 Ecommerce RAG Assistant。\n" +
                        "你可以询问优惠券、活动规则、退款、物流以及售后相关问题。"
                }
            ];

            this.question = "";
            this.activeEvidence = [];
            this.conversationId = null;

            this.$message({
                message: "当前会话已清空",
                type: "success",
                duration: 1600
            });

            this.scrollToBottom();
        }
    },

    mounted: function () {
        this.checkHealth();
    }
});