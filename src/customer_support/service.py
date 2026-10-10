from typing import Optional

from src.agents.order_agent import OrderAgent
from src.memory import (
    ContextResolver,
    ConversationMemoryService,
)
from src.routing.intent_router import IntentRouter
from src.service import build_rag_service


class CustomerSupportService:
    """
    Unified customer-service entry point.

    Responsibilities:
    - manage conversation sessions
    - resolve conversational context
    - classify user intent
    - route requests to RAG or order agent
    - store conversation memory

    No LangChain or LangGraph is used.
    """

    def __init__(
            self,
            intent_router: Optional[
                IntentRouter
            ] = None,
            order_agent: Optional[
                OrderAgent
            ] = None,
            memory_service: Optional[
                ConversationMemoryService
            ] = None,
            context_resolver: Optional[
                ContextResolver
            ] = None,
            rag_service=None,
    ):
        self.intent_router = (
                intent_router
                or IntentRouter()
        )

        self.order_agent = (
                order_agent
                or OrderAgent()
        )

        self.memory_service = (
                memory_service
                or ConversationMemoryService()
        )

        self.context_resolver = (
                context_resolver
                or ContextResolver()
        )

        # Lazy initialization:
        # order-only queries should not initialize
        # the full RAG pipeline.
        self._rag_service = rag_service

    # ========================================================
    # RAG Service
    # ========================================================

    def _get_rag_service(self):
        """
        Lazily initialize the existing v1 RAG service.
        """

        if self._rag_service is None:
            self._rag_service = (
                build_rag_service()
            )

        return self._rag_service

    # ========================================================
    # Conversation
    # ========================================================

    def _get_or_create_conversation(
            self,
            user_id: str,
            conversation_id: Optional[str],
    ) -> dict:
        """
        Load an existing conversation or create a new one.
        """

        if conversation_id:
            conversation = (
                self.memory_service.get_conversation(
                    conversation_id=conversation_id,
                    user_id=user_id,
                )
            )

            if conversation is None:
                raise ValueError(
                    "Conversation 不存在或无权访问。"
                )

            return conversation

        return (
            self.memory_service.start_conversation(
                user_id=user_id
            )
        )

    # ========================================================
    # Main Entry
    # ========================================================

    def ask(
            self,
            question: str,
            user_id: str,
            conversation_id: Optional[str] = None,
    ) -> dict:
        """
        Handle one customer-service request.

        If conversation_id is not provided,
        a new conversation is created automatically.
        """

        question = str(
            question or ""
        ).strip()

        user_id = str(
            user_id or ""
        ).strip()

        conversation_id = str(
            conversation_id or ""
        ).strip() or None

        if not question:
            raise ValueError(
                "question 不能为空。"
            )

        if not user_id:
            raise ValueError(
                "user_id 不能为空。"
            )

        # ----------------------------------------------------
        # Step 1: Load or create conversation
        # ----------------------------------------------------

        conversation = (
            self._get_or_create_conversation(
                user_id=user_id,
                conversation_id=conversation_id,
            )
        )

        conversation_id = conversation[
            "conversation_id"
        ]

        active_order_id = conversation.get(
            "active_order_id"
        )

        # ----------------------------------------------------
        # Step 2: Save original user message
        # ----------------------------------------------------

        self.memory_service.save_user_message(
            conversation_id=conversation_id,
            user_id=user_id,
            content=question,
        )

        # ----------------------------------------------------
        # Step 3: Resolve context
        # ----------------------------------------------------

        context_result = (
            self.context_resolver.resolve(
                question=question,
                active_order_id=active_order_id,
            )
        )

        resolved_question = (
            context_result[
                "resolved_question"
            ]
        )

        # ----------------------------------------------------
        # Step 4: Intent Routing
        # ----------------------------------------------------

        route_result = (
            self.intent_router.route(
                resolved_question
            )
        )

        intent = route_result["intent"]
        intent_reason = route_result[
            "reason"
        ]

        self.memory_service.remember_intent(
            conversation_id=conversation_id,
            user_id=user_id,
            intent=intent,
        )

        # ----------------------------------------------------
        # Step 5: Execute handler
        # ----------------------------------------------------

        handler = None
        answer = ""
        result = None

        if intent == "KNOWLEDGE_QA":

            rag_service = (
                self._get_rag_service()
            )

            result = rag_service.ask(
                resolved_question
            )

            handler = "RAG"

            answer = result[
                "answer"
            ]

        elif intent == "ORDER_QUERY":

            result = (
                self.order_agent.ask(
                    question=resolved_question,
                    user_id=user_id,
                )
            )

            handler = "ORDER_AGENT"

            answer = result[
                "answer"
            ]

            self._remember_order_from_result(
                conversation_id=conversation_id,
                user_id=user_id,
                question=resolved_question,
                order_result=result,
            )

        elif intent == "GENERAL_CHAT":

            handler = "GENERAL_CHAT"

            answer = (
                "您好，我可以帮助您查询订单、"
                "物流状态，也可以解答退款、"
                "优惠、售后、发票等电商相关问题。"
            )

        elif intent == "MIXED_QUERY":

            handler = (
                "MIXED_QUERY_PENDING"
            )

            answer = (
                "这个问题同时涉及您的订单信息"
                "和相关政策规则。"
                "当前混合查询能力将在后续阶段接入。"
            )

            # Even though Mixed Query is not implemented yet,
            # remember an explicit order ID if the user
            # mentioned one.
            explicit_order_id = (
                self.context_resolver.extract_order_id(
                    resolved_question
                )
            )

            if explicit_order_id:
                self.memory_service.remember_order(
                    conversation_id=conversation_id,
                    user_id=user_id,
                    order_id=explicit_order_id,
                )

        else:

            handler = "UNSUPPORTED"

            answer = (
                "抱歉，我目前主要处理电商客服相关问题，"
                "包括订单、物流、退款、优惠、"
                "售后和发票等。"
            )

        # ----------------------------------------------------
        # Step 6: Save assistant response
        # ----------------------------------------------------

        self.memory_service.save_assistant_message(
            conversation_id=conversation_id,
            user_id=user_id,
            content=answer,
        )

        # ----------------------------------------------------
        # Step 7: Return unified response
        # ----------------------------------------------------

        return {
            "conversation_id": conversation_id,
            "user_id": user_id,
            "question": question,
            "resolved_question": (
                resolved_question
            ),
            "memory_used": (
                context_result[
                    "used_memory"
                ]
            ),
            "intent": intent,
            "intent_reason": intent_reason,
            "handler": handler,
            "answer": answer,
            "result": result,
        }

    # ========================================================
    # Memory Helpers
    # ========================================================

    def _remember_order_from_result(
            self,
            conversation_id: str,
            user_id: str,
            question: str,
            order_result: dict,
    ) -> None:
        """
        Remember the order currently being discussed.

        Priority:
        1. order_id actually used by a successful tool call
        2. explicit order ID from the question
        """

        tool_results = order_result.get(
            "tool_results",
            [],
        )

        for tool_result in tool_results:

            result = tool_result.get(
                "result",
                {}
            )

            arguments = tool_result.get(
                "arguments",
                {}
            )

            if result.get("success") is not True:
                continue

            order_id = arguments.get(
                "order_id"
            )

            if order_id:
                self.memory_service.remember_order(
                    conversation_id=conversation_id,
                    user_id=user_id,
                    order_id=str(
                        order_id
                    ).upper(),
                )

                return

        explicit_order_id = (
            self.context_resolver.extract_order_id(
                question
            )
        )

        if explicit_order_id:
            self.memory_service.remember_order(
                conversation_id=conversation_id,
                user_id=user_id,
                order_id=explicit_order_id,
            )