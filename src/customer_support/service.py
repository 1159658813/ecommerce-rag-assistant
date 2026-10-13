import time
from typing import Optional

from src.agents.order_agent import OrderAgent
from src.memory import (
    ContextResolver,
    ConversationMemoryService,
    HistorySelector,
    QueryRewriter,
)
from src.routing.intent_router import IntentRouter
from src.service import build_rag_service

from src.customer_support.mixed_query_service import (
    MixedQueryService,
)
from src.observability import (
    get_logger,
    log_event,
)

logger = get_logger(
    "customer_support"
)

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
            query_rewriter: Optional[
                QueryRewriter
            ] = None,
            history_selector: Optional[
                HistorySelector
            ] = None,
            rag_service=None,
            mixed_query_service: Optional[
                MixedQueryService
            ] = None,
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
        self.query_rewriter = (
                query_rewriter
                or QueryRewriter()
        )
        self.history_selector = (
                history_selector
                or HistorySelector()
        )

        # Lazy initialization:
        # order-only queries should not initialize
        # the full RAG pipeline.
        self._rag_service = rag_service
        self._mixed_query_service = (
            mixed_query_service
        )

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

    def _get_mixed_query_service(
            self,
    ) -> MixedQueryService:
        """
        Lazily initialize mixed-query handling.

        The existing RAGService instance is reused.
        """

        if self._mixed_query_service is None:
            self._mixed_query_service = (
                MixedQueryService(
                    rag_service=(
                        self._get_rag_service()
                    )
                )
            )

        return self._mixed_query_service

    # ========================================================
    # Query Rewrite
    # ========================================================

    def _rewrite_with_fallback(
            self,
            question: str,
            messages: list[dict],
            active_order_id: Optional[str],
            last_intent: Optional[str],
    ) -> dict:
        """
        Rewrite the current conversational question with
        recent history and structured conversation state.

        If the LLM rewriter fails, fall back to the existing
        rule-based ContextResolver.
        """

        try:
            rewrite_result = (
                self.query_rewriter.rewrite(
                    question=question,
                    messages=messages,
                    active_order_id=active_order_id,
                    last_intent=last_intent,
                )
            )

            rewritten_question = str(
                rewrite_result.get(
                    "rewritten_question",
                    question,
                )
                or question
            ).strip()

            resolved_entities = (
                    rewrite_result.get(
                        "resolved_entities",
                        {},
                    )
                    or {}
            )

            rewritten_order_id = (
                resolved_entities.get(
                    "order_id"
                )
            )

            # Explicit order IDs in the current user message
            # have the highest priority.
            explicit_order_id = (
                self.context_resolver
                .extract_order_id(
                    question
                )
            )

            resolved_order_id = (
                    explicit_order_id
                    or rewritten_order_id
                    or active_order_id
            )

            used_history = bool(
                rewrite_result.get(
                    "used_history",
                    False,
                )
            )

            # Structured active_order_id is also conversation
            # memory even when no textual history was needed.
            used_structured_memory = bool(
                active_order_id
                and not explicit_order_id
                and active_order_id
                in rewritten_question.upper()
            )

            return {
                "original_question": question,
                "resolved_question": rewritten_question,
                "used_memory": (
                        used_history
                        or used_structured_memory
                ),
                "active_order_id": resolved_order_id,
            }

        except Exception:
            # Preserve the existing deterministic resolver as
            # a safe fallback. Query rewriting must never make
            # the whole customer-support request unavailable.
            return (
                self.context_resolver.resolve(
                    question=question,
                    active_order_id=active_order_id,
                )
            )

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

        total_started = (
            time.perf_counter()
        )

        latency_router = 0.0

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
        last_intent = conversation.get(
            "last_intent"
        )

        # ----------------------------------------------------
        # Step 2: Select relevant conversation history
        # ----------------------------------------------------

        candidate_messages = (
            self.memory_service
            .get_recent_messages(
                conversation_id=conversation_id,
                user_id=user_id,
                limit=(
                    self.history_selector
                    .candidate_limit
                ),
            )
        )

        history_result = (
            self.history_selector.select(
                question=question,
                messages=candidate_messages,
                active_order_id=active_order_id,
                last_intent=last_intent,
            )
        )

        recent_messages = (
            history_result[
                "messages"
            ]
        )

        # ----------------------------------------------------
        # Step 3: Save original user message
        # ----------------------------------------------------

        self.memory_service.save_user_message(
            conversation_id=conversation_id,
            user_id=user_id,
            content=question,
        )

        # ----------------------------------------------------
        # Step 4: Rewrite conversational query
        # ----------------------------------------------------

        context_result = (
            self._rewrite_with_fallback(
                question=question,
                messages=recent_messages,
                active_order_id=active_order_id,
                last_intent=last_intent,
            )
        )

        resolved_question = (
            context_result[
                "resolved_question"
            ]
        )

        # ----------------------------------------------------
        # Step 5: Intent Routing
        # ----------------------------------------------------

        router_started = (
            time.perf_counter()
        )

        route_result = (
            self.intent_router.route(
                resolved_question
            )
        )

        latency_router = round(
            (
                    time.perf_counter()
                    - router_started
            )
            * 1000,
            3,
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
        # Step 6: Execute handler
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

            order_id = (

                context_result.get(

                    "active_order_id"

                )

            )

            if not order_id:
                order_id = (

                    self.context_resolver

                    .extract_order_id(

                        resolved_question

                    )

                )

            mixed_service = (

                self._get_mixed_query_service()

            )

            result = (

                mixed_service.answer(

                    question=resolved_question,

                    user_id=user_id,

                    order_id=order_id,

                )

            )

            handler = "MIXED_QUERY"

            answer = result[

                "answer"

            ]

            order_result = (

                result.get(

                    "order_result"

                )

            )

            if (

                    order_id

                    and order_result

                    and order_result.get(

                "success"

            )

            ):
                self.memory_service.remember_order(

                    conversation_id=conversation_id,

                    user_id=user_id,

                    order_id=order_id,

                )

        else:

            handler = "UNSUPPORTED"

            answer = (
                "抱歉，我目前主要处理电商客服相关问题，"
                "包括订单、物流、退款、优惠、"
                "售后和发票等。"
            )

        # ----------------------------------------------------
        # Step 7: Save assistant response
        # ----------------------------------------------------

        self.memory_service.save_assistant_message(
            conversation_id=conversation_id,
            user_id=user_id,
            content=answer,
        )

        # ----------------------------------------------------
        # Step 8: Observability V2
        # ----------------------------------------------------

        component_observability = {}

        if isinstance(
                result,
                dict,
        ):
            component_observability = (
                    result.get(
                        "observability",
                        {},
                    )
                    or {}
            )

        tools_called = []

        if isinstance(
                result,
                dict,
        ):
            tools_called = (
                    result.get(
                        "tools_called",
                        [],
                    )
                    or component_observability.get(
                "tools_called",
                [],
            )
                    or []
            )

        abstained = False

        if isinstance(
                result,
                dict,
        ):
            abstained = bool(
                result.get(
                    "abstained",
                    component_observability.get(
                        "abstained",
                        False,
                    ),
                )
            )

        latency_total = round(
            (
                    time.perf_counter()
                    - total_started
            )
            * 1000,
            3,
        )

        trace = {
            "latency_total": (
                latency_total
            ),
            "latency_router": (
                latency_router
            ),
            "latency_retrieval": float(
                component_observability.get(
                    "latency_retrieval",
                    0.0,
                )
                or 0.0
            ),
            "latency_verifier": float(
                component_observability.get(
                    "latency_verifier",
                    0.0,
                )
                or 0.0
            ),
            "latency_tool": float(
                component_observability.get(
                    "latency_tool",
                    0.0,
                )
                or 0.0
            ),
            "latency_composer": float(
                component_observability.get(
                    "latency_composer",
                    0.0,
                )
                or 0.0
            ),
            "tools_called": tools_called,
            "verifier_retry_count": int(
                component_observability.get(
                    "verifier_retry_count",
                    0,
                )
                or 0
            ),
            "abstained": abstained,
            "conversation_id": (
                conversation_id
            ),
            "intent": intent,
            "handler": handler,
            "memory_used": bool(
                context_result[
                    "used_memory"
                ]
            ),
        }

        log_event(
            logger,
            "customer_support_trace",
            **trace,
        )

        # ----------------------------------------------------
        # Step 9: Return unified response
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