import sqlite3
from typing import Optional

from src.orders.init_db import DATABASE_PATH


class ConversationRepository:
    """
    SQLite data access layer for conversation memory.

    Responsibilities:
    - create conversations
    - store messages
    - read conversation state
    - update conversation state

    This layer contains no LLM logic.
    """

    def __init__(
            self,
            database_path=DATABASE_PATH,
    ):
        self.database_path = database_path

    # ========================================================
    # Connection
    # ========================================================

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(
            self.database_path
        )

        conn.row_factory = sqlite3.Row

        conn.execute(
            "PRAGMA foreign_keys = ON"
        )

        return conn

    # ========================================================
    # Conversation
    # ========================================================

    def create_conversation(
            self,
            conversation_id: str,
            user_id: str,
    ) -> dict:
        """
        Create a new conversation.
        """

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO conversations (
                    conversation_id,
                    user_id
                )
                VALUES (?, ?)
                """,
                (
                    conversation_id,
                    user_id,
                ),
            )

            cursor = conn.execute(
                """
                SELECT
                    conversation_id,
                    user_id,
                    active_order_id,
                    last_intent,
                    created_at,
                    updated_at
                FROM conversations
                WHERE conversation_id = ?
                AND user_id = ?
                """,
                (
                    conversation_id,
                    user_id,
                ),
            )

            row = cursor.fetchone()

        return dict(row)

    def get_conversation(
            self,
            conversation_id: str,
            user_id: str,
    ) -> Optional[dict]:
        """
        Get one conversation owned by a user.
        """

        with self._connect() as conn:
            cursor = conn.execute(
                """
                SELECT
                    conversation_id,
                    user_id,
                    active_order_id,
                    last_intent,
                    created_at,
                    updated_at
                FROM conversations
                WHERE conversation_id = ?
                AND user_id = ?
                """,
                (
                    conversation_id,
                    user_id,
                ),
            )

            row = cursor.fetchone()

        if row is None:
            return None

        return dict(row)

    # ========================================================
    # Messages
    # ========================================================

    def add_message(
            self,
            conversation_id: str,
            role: str,
            content: str,
    ) -> dict:
        """
        Save one conversation message.
        """

        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO conversation_messages (
                    conversation_id,
                    role,
                    content
                )
                VALUES (?, ?, ?)
                """,
                (
                    conversation_id,
                    role,
                    content,
                ),
            )

            message_id = cursor.lastrowid

            cursor = conn.execute(
                """
                SELECT
                    message_id,
                    conversation_id,
                    role,
                    content,
                    created_at
                FROM conversation_messages
                WHERE message_id = ?
                """,
                (
                    message_id,
                ),
            )

            row = cursor.fetchone()

        return dict(row)

    def get_recent_messages(
            self,
            conversation_id: str,
            limit: int = 10,
    ) -> list[dict]:
        """
        Return recent messages in chronological order.
        """

        with self._connect() as conn:
            cursor = conn.execute(
                """
                SELECT *
                FROM (
                    SELECT
                        message_id,
                        conversation_id,
                        role,
                        content,
                        created_at
                    FROM conversation_messages
                    WHERE conversation_id = ?
                    ORDER BY message_id DESC
                    LIMIT ?
                )
                ORDER BY message_id ASC
                """,
                (
                    conversation_id,
                    limit,
                ),
            )

            rows = cursor.fetchall()

        return [
            dict(row)
            for row in rows
        ]

    # ========================================================
    # Conversation State
    # ========================================================

    def set_active_order_id(
            self,
            conversation_id: str,
            user_id: str,
            order_id: Optional[str],
    ) -> None:
        """
        Remember which order is currently being discussed.
        """

        with self._connect() as conn:
            conn.execute(
                """
                UPDATE conversations
                SET
                    active_order_id = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE conversation_id = ?
                AND user_id = ?
                """,
                (
                    order_id,
                    conversation_id,
                    user_id,
                ),
            )

    def set_last_intent(
            self,
            conversation_id: str,
            user_id: str,
            intent: Optional[str],
    ) -> None:
        """
        Remember the latest classified intent.
        """

        with self._connect() as conn:
            conn.execute(
                """
                UPDATE conversations
                SET
                    last_intent = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE conversation_id = ?
                AND user_id = ?
                """,
                (
                    intent,
                    conversation_id,
                    user_id,
                ),
            )