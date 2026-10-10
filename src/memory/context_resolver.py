import re
from typing import Optional

ORDER_ID_PATTERN = re.compile(
    r"\bORD\d+\b",
    re.IGNORECASE,
)


class ContextResolver:
    """
    Resolve simple conversational references
    using structured conversation memory.

    No LLM is used in this first version.
    """

    def resolve(
            self,
            question: str,
            active_order_id: Optional[str] = None,
    ) -> dict:
        """
        Return the original and context-enriched question.
        """

        original_question = str(
            question or ""
        ).strip()

        if not original_question:
            return {
                "original_question": "",
                "resolved_question": "",
                "used_memory": False,
                "active_order_id": active_order_id,
            }

        # ----------------------------------------------------
        # If the user already provides an explicit order ID,
        # no memory resolution is necessary.
        # ----------------------------------------------------

        explicit_order_id = self.extract_order_id(
            original_question
        )

        if explicit_order_id:
            return {
                "original_question": original_question,
                "resolved_question": original_question,
                "used_memory": False,
                "active_order_id": explicit_order_id,
            }

        # ----------------------------------------------------
        # No active order in memory.
        # ----------------------------------------------------

        if not active_order_id:
            return {
                "original_question": original_question,
                "resolved_question": original_question,
                "used_memory": False,
                "active_order_id": None,
            }

        # ----------------------------------------------------
        # Enrich the current question with remembered order.
        #
        # We do not replace the user's words.
        # We only provide missing context.
        # ----------------------------------------------------

        resolved_question = (
            f"当前正在讨论的订单号是 "
            f"{active_order_id}。"
            f"用户当前问题："
            f"{original_question}"
        )

        return {
            "original_question": original_question,
            "resolved_question": resolved_question,
            "used_memory": True,
            "active_order_id": active_order_id,
        }

    @staticmethod
    def extract_order_id(
            text: str,
    ) -> Optional[str]:
        """
        Extract an explicit order ID from text.
        """

        match = ORDER_ID_PATTERN.search(
            str(text or "")
        )

        if match is None:
            return None

        return match.group(0).upper()
