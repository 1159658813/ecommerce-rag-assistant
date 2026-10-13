import sqlite3
from typing import Optional

from src.orders.init_db import DATABASE_PATH


class OrderRepository:
    """
    Data access layer for orders, shipments, and refunds.

    This class is responsible only for database operations.
    It should not contain business logic or LLM logic.
    """

    def __init__(self, database_path=DATABASE_PATH):
        self.database_path = database_path

    # ========================================================
    # Database Connection
    # ========================================================

    def _connect(self) -> sqlite3.Connection:
        """
        Create a SQLite connection.

        sqlite3.Row allows query results to be accessed
        using column names.
        """

        conn = sqlite3.connect(self.database_path)

        conn.row_factory = sqlite3.Row

        conn.execute("PRAGMA foreign_keys = ON")

        return conn

    # ========================================================
    # Order Queries
    # ========================================================

    def get_order_by_id(
            self,
            order_id: str,
    ) -> Optional[dict]:
        """
        Find one order by order_id.

        Returns:
            dict if the order exists,
            None if the order does not exist.
        """

        with self._connect() as conn:
            cursor = conn.execute(
                """
                SELECT
                    order_id,
                    user_id,
                    status,
                    total_amount,
                    created_at,
                    paid_at
                FROM orders
                WHERE order_id = ?
                """,
                (order_id,),
            )

            row = cursor.fetchone()

        if row is None:
            return None

        return dict(row)

    def get_orders_by_user(
            self,
            user_id: str,
    ) -> list[dict]:
        """
        Find all orders belonging to one user.
        """

        with self._connect() as conn:
            cursor = conn.execute(
                """
                SELECT
                    order_id,
                    user_id,
                    status,
                    total_amount,
                    created_at,
                    paid_at
                FROM orders
                WHERE user_id = ?
                ORDER BY created_at DESC
                """,
                (user_id,),
            )

            rows = cursor.fetchall()

        return [
            dict(row)
            for row in rows
        ]

    # ========================================================
    # Shipment Queries
    # ========================================================

    def get_shipment_by_order_id(
            self,
            order_id: str,
    ) -> Optional[dict]:
        """
        Find shipment information for one order.
        """

        with self._connect() as conn:
            cursor = conn.execute(
                """
                SELECT
                    shipment_id,
                    order_id,
                    carrier,
                    tracking_number,
                    status,
                    latest_location,
                    estimated_delivery
                FROM shipments
                WHERE order_id = ?
                """,
                (order_id,),
            )

            row = cursor.fetchone()

        if row is None:
            return None

        return dict(row)

    # ========================================================
    # Refund Queries
    # ========================================================

    def get_refund_by_id(
            self,
            refund_id: str,
    ) -> Optional[dict]:
        """
        Find one refund by refund_id.
        """

        with self._connect() as conn:
            cursor = conn.execute(
                """
                SELECT
                    refund_id,
                    order_id,
                    user_id,
                    status,
                    amount,
                    reason,
                    created_at,
                    completed_at
                FROM refunds
                WHERE refund_id = ?
                """,
                (refund_id,),
            )

            row = cursor.fetchone()

        if row is None:
            return None

        return dict(row)

    def get_latest_refund_by_order_id(
            self,
            order_id: str,
    ) -> Optional[dict]:
        """
        Find the latest refund record for one order.
        """

        with self._connect() as conn:
            cursor = conn.execute(
                """
                SELECT
                    refund_id,
                    order_id,
                    user_id,
                    status,
                    amount,
                    reason,
                    created_at,
                    completed_at
                FROM refunds
                WHERE order_id = ?
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (order_id,),
            )

            row = cursor.fetchone()

        if row is None:
            return None

        return dict(row)