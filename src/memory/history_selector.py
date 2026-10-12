import re
from typing import Optional

ORDER_ID_PATTERN = re.compile(
    r"\bORD\d+\b",
    re.IGNORECASE,
)


class HistorySelector:
    """
    Select a compact and relevant subset of recent
    conversation history for contextual query rewriting.

    Design goals:
    - keep recent context
    - prioritize relevant order context
    - avoid sending unnecessary history to the LLM
    - enforce a simple character budget
    - remain deterministic and lightweight
    """

    REFERENCE_HINTS = (
        "它",
        "那个",
        "这个",
        "刚才",
        "刚刚",
        "之前",
        "前面",
        "上一个",
        "下一个",
        "第一个",
        "第二个",
        "第三个",
        "已发货的",
        "已签收的",
    )

    ORDER_HINTS = (
        "订单",
        "物流",
        "快递",
        "运单",
        "发货",
        "签收",
        "到哪",
        "送到",
        "配送",
    )

    KNOWLEDGE_HINTS = (
        "退款",
        "退货",
        "售后",
        "优惠",
        "发票",
        "运费",
        "规则",
        "政策",
        "到账",
        "多久",
    )

    def __init__(
            self,
            candidate_limit: int = 12,
            max_messages: int = 6,
            char_budget: int = 2400,
            per_message_char_limit: int = 800,
    ):
        self.candidate_limit = max(
            1,
            int(candidate_limit),
        )

        self.max_messages = max(
            1,
            int(max_messages),
        )

        self.char_budget = max(
            1,
            int(char_budget),
        )

        self.per_message_char_limit = max(
            1,
            int(per_message_char_limit),
        )

    # ========================================================
    # Public API
    # ========================================================

    def select(
            self,
            question: str,
            messages: Optional[list[dict]] = None,
            active_order_id: Optional[str] = None,
            last_intent: Optional[str] = None,
    ) -> dict:
        """
        Select history for QueryRewriter.

        Returns:
            {
                "messages": [...],
                "candidate_count": int,
                "selected_count": int,
                "char_count": int,
                "used_relevance_selection": bool
            }
        """

        question = str(
            question or ""
        ).strip()

        prepared_messages = (
            self._prepare_messages(
                messages or []
            )
        )

        if not prepared_messages:
            return {
                "messages": [],
                "candidate_count": 0,
                "selected_count": 0,
                "char_count": 0,
                "used_relevance_selection": False,
            }

        preferred_indices = (
            self._find_preferred_indices(
                question=question,
                messages=prepared_messages,
                active_order_id=active_order_id,
                last_intent=last_intent,
            )
        )

        # ----------------------------------------------------
        # Selection priority
        #
        # 1. relevant messages, newest first
        # 2. remaining messages, newest first
        #
        # Final output is restored to chronological order.
        # ----------------------------------------------------

        priority_indices = []

        for index in sorted(
                preferred_indices,
                reverse=True,
        ):
            priority_indices.append(
                index
            )

        for index in range(
                len(prepared_messages) - 1,
                -1,
                -1,
        ):
            if index not in preferred_indices:
                priority_indices.append(
                    index
                )

        selected_by_index = {}
        char_count = 0

        for index in priority_indices:

            if (
                    len(selected_by_index)
                    >= self.max_messages
            ):
                break

            message = prepared_messages[
                index
            ]

            content = (
                self._trim_content(
                    message["content"]
                )
            )

            if not content:
                continue

            content_size = len(
                content
            )

            if (
                    char_count
                    + content_size
                    > self.char_budget
            ):
                continue

            selected_by_index[
                index
            ] = {
                "role": message["role"],
                "content": content,
            }

            char_count += (
                content_size
            )

        # ----------------------------------------------------
        # Safety fallback:
        # if every message exceeded the budget,
        # keep a shortened version of the newest message.
        # ----------------------------------------------------

        if not selected_by_index:

            newest_index = (
                    len(prepared_messages) - 1
            )

            newest_message = (
                prepared_messages[
                    newest_index
                ]
            )

            content = newest_message[
                          "content"
                      ][:self.char_budget]

            if content:
                selected_by_index[
                    newest_index
                ] = {
                    "role": newest_message[
                        "role"
                    ],
                    "content": content,
                }

                char_count = len(
                    content
                )

        selected_messages = [
            selected_by_index[index]
            for index in sorted(
                selected_by_index
            )
        ]

        return {
            "messages": selected_messages,
            "candidate_count": len(
                prepared_messages
            ),
            "selected_count": len(
                selected_messages
            ),
            "char_count": char_count,
            "used_relevance_selection": bool(
                preferred_indices
            ),
        }

    # ========================================================
    # Relevance
    # ========================================================

    def _find_preferred_indices(
            self,
            question: str,
            messages: list[dict],
            active_order_id: Optional[str],
            last_intent: Optional[str],
    ) -> set[int]:

        preferred_indices = set()

        question_lower = (
            question.lower()
        )

        last_intent = str(
            last_intent or ""
        ).strip().upper()

        active_order_id = (
                str(
                    active_order_id or ""
                )
                .strip()
                .upper()
                or None
        )

        has_reference = any(
            hint in question_lower
            for hint in self.REFERENCE_HINTS
        )

        has_order_question = any(
            hint in question_lower
            for hint in self.ORDER_HINTS
        )

        has_knowledge_question = any(
            hint in question_lower
            for hint in self.KNOWLEDGE_HINTS
        )

        order_context_needed = (
                has_reference
                or has_order_question
                or last_intent
                in {
                    "ORDER_QUERY",
                    "MIXED_QUERY",
                }
        )

        knowledge_context_needed = (
                has_knowledge_question
                or last_intent
                in {
                    "KNOWLEDGE_QA",
                    "MIXED_QUERY",
                }
        )

        for index, message in enumerate(
                messages
        ):

            content = message[
                "content"
            ]

            content_upper = (
                content.upper()
            )

            # ----------------------------------------------
            # Exact active order context
            # ----------------------------------------------

            if (
                    active_order_id
                    and order_context_needed
                    and active_order_id
                    in content_upper
            ):
                preferred_indices.add(
                    index
                )
                continue

            order_ids = (
                ORDER_ID_PATTERN.findall(
                    content
                )
            )

            # ----------------------------------------------
            # Order-related history
            # ----------------------------------------------

            if (
                    order_context_needed
                    and order_ids
            ):
                preferred_indices.add(
                    index
                )
                continue

            # "第二个订单" usually requires a message
            # containing multiple previously listed orders.
            if (
                    has_reference
                    and len(order_ids) >= 2
            ):
                preferred_indices.add(
                    index
                )
                continue

            # ----------------------------------------------
            # Knowledge / policy context
            # ----------------------------------------------

            if (
                    knowledge_context_needed
                    and any(
                hint in content.lower()
                for hint
                in self.KNOWLEDGE_HINTS
            )
            ):
                preferred_indices.add(
                    index
                )

        return preferred_indices

    # ========================================================
    # Helpers
    # ========================================================

    def _prepare_messages(
            self,
            messages: list[dict],
    ) -> list[dict]:

        prepared = []

        for message in messages:

            if not isinstance(
                    message,
                    dict,
            ):
                continue

            role = str(
                message.get(
                    "role",
                    ""
                )
            ).strip()

            content = str(
                message.get(
                    "content",
                    ""
                )
            ).strip()

            if (
                    role not in {
                "user",
                "assistant",
            }
                    or not content
            ):
                continue

            prepared.append(
                {
                    "role": role,
                    "content": content,
                }
            )

        return prepared

    def _trim_content(
            self,
            content: str,
    ) -> str:

        content = str(
            content or ""
        ).strip()

        if (
                len(content)
                <= self.per_message_char_limit
        ):
            return content

        return content[
               :self.per_message_char_limit
               ]