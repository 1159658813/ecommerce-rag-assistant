from typing import Optional
from uuid import uuid4

from src.memory.repository import (
    ConversationRepository,
)


class ConversationMemoryService:
    """
    Business layer for conversation memory.

    Responsibilities:
    - create conversation sessions
    - verify conversation ownership
    - store conversation messages
    - maintain structured conversation state
    """

    def __init__(
            self,
            repository: Optional[
                ConversationRepository
            ] = None,
    ):
        self.repository = (
                repository
                or ConversationRepository()
        )

    # ========================================================
    # Conversation Lifecycle
    # ========================================================

    def start_conversation(
            self,
            user_id: str,
    ) -> dict:
        """
        Create a new conversation for one user.
        """

        user_id = str(
            user_id or ""
        ).strip()

        if not user_id:
            raise ValueError(
                "user_id 不能为空。"
            )

        conversation_id = (
            uuid4().hex
        )

        conversation = (
            self.repository.create_conversation(
                conversation_id=conversation_id,
                user_id=user_id,
            )
        )

        return conversation

    def get_conversation(
            self,
            conversation_id: str,
            user_id: str,
    ) -> Optional[dict]:
        """
        Get a conversation only when it belongs
        to the current user.
        """

        return (
            self.repository.get_conversation(
                conversation_id=conversation_id,
                user_id=user_id,
            )
        )

    # ========================================================
    # Messages
    # ========================================================

    def save_user_message(
            self,
            conversation_id: str,
            user_id: str,
            content: str,
    ) -> dict:
        """
        Save a user message after ownership check.
        """

        self._require_owned_conversation(
            conversation_id=conversation_id,
            user_id=user_id,
        )

        return self.repository.add_message(
            conversation_id=conversation_id,
            role="user",
            content=content,
        )

    def save_assistant_message(
            self,
            conversation_id: str,
            user_id: str,
            content: str,
    ) -> dict:
        """
        Save an assistant message after ownership check.
        """

        self._require_owned_conversation(
            conversation_id=conversation_id,
            user_id=user_id,
        )

        return self.repository.add_message(
            conversation_id=conversation_id,
            role="assistant",
            content=content,
        )

    def get_recent_messages(
            self,
            conversation_id: str,
            user_id: str,
            limit: int = 10,
    ) -> list[dict]:
        """
        Get recent conversation messages.
        """

        self._require_owned_conversation(
            conversation_id=conversation_id,
            user_id=user_id,
        )

        return (
            self.repository.get_recent_messages(
                conversation_id=conversation_id,
                limit=limit,
            )
        )

    # ========================================================
    # Structured State
    # ========================================================

    def remember_order(
            self,
            conversation_id: str,
            user_id: str,
            order_id: str,
    ) -> None:
        """
        Remember the active order in this conversation.
        """

        self._require_owned_conversation(
            conversation_id=conversation_id,
            user_id=user_id,
        )

        self.repository.set_active_order_id(
            conversation_id=conversation_id,
            user_id=user_id,
            order_id=order_id,
        )

    def remember_intent(
            self,
            conversation_id: str,
            user_id: str,
            intent: str,
    ) -> None:
        """
        Remember the latest intent.
        """

        self._require_owned_conversation(
            conversation_id=conversation_id,
            user_id=user_id,
        )

        self.repository.set_last_intent(
            conversation_id=conversation_id,
            user_id=user_id,
            intent=intent,
        )

    def get_context(
            self,
            conversation_id: str,
            user_id: str,
            message_limit: int = 10,
    ) -> dict:
        """
        Return structured state and recent messages.
        """

        conversation = (
            self._require_owned_conversation(
                conversation_id=conversation_id,
                user_id=user_id,
            )
        )

        messages = (
            self.repository.get_recent_messages(
                conversation_id=conversation_id,
                limit=message_limit,
            )
        )

        return {
            "conversation": conversation,
            "messages": messages,
        }

    # ========================================================
    # Internal Helpers
    # ========================================================

    def _require_owned_conversation(
            self,
            conversation_id: str,
            user_id: str,
    ) -> dict:
        """
        Verify that the conversation belongs
        to the current user.
        """

        conversation = (
            self.repository.get_conversation(
                conversation_id=conversation_id,
                user_id=user_id,
            )
        )

        if conversation is None:
            raise ValueError(
                "Conversation 不存在或无权访问。"
            )

        return conversation