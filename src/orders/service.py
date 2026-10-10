from typing import Optional

from src.orders.repository import OrderRepository


class OrderService:
    """
    Business service for order-related operations.

    Responsibilities:
    - validate order ownership
    - combine order and shipment data
    - provide stable structured results to upper layers

    This layer does not contain LLM logic.
    """

    def __init__(
            self,
            repository: Optional[OrderRepository] = None,
    ):
        self.repository = repository or OrderRepository()

    # ========================================================
    # Internal Helpers
    # ========================================================

    def _get_owned_order(
            self,
            user_id: str,
            order_id: str,
    ) -> Optional[dict]:
        """
        Return the order only if it belongs to the given user.

        For security, an order that does not exist and an order
        that belongs to another user are both treated as unavailable.
        """

        order = self.repository.get_order_by_id(order_id)

        if order is None:
            return None

        if order["user_id"] != user_id:
            return None

        return order

    # ========================================================
    # Order Detail
    # ========================================================

    def get_order_detail(
            self,
            user_id: str,
            order_id: str,
    ) -> dict:
        """
        Get order detail together with shipment information.
        """

        order = self._get_owned_order(
            user_id=user_id,
            order_id=order_id,
        )

        if order is None:
            return {
                "success": False,
                "code": "ORDER_NOT_FOUND",
                "message": "未找到该订单。",
                "data": None,
            }

        shipment = self.repository.get_shipment_by_order_id(
            order_id
        )

        return {
            "success": True,
            "code": "OK",
            "message": "订单查询成功。",
            "data": {
                "order": order,
                "shipment": shipment,
            },
        }

    # ========================================================
    # User Orders
    # ========================================================

    def get_user_orders(
            self,
            user_id: str,
    ) -> dict:
        """
        Get all orders belonging to one user.
        """

        orders = self.repository.get_orders_by_user(
            user_id
        )

        return {
            "success": True,
            "code": "OK",
            "message": "订单列表查询成功。",
            "data": {
                "orders": orders,
                "count": len(orders),
            },
        }

    # ========================================================
    # Logistics
    # ========================================================

    def get_logistics(
            self,
            user_id: str,
            order_id: str,
    ) -> dict:
        """
        Get shipment information for an order owned by the user.
        """

        order = self._get_owned_order(
            user_id=user_id,
            order_id=order_id,
        )

        if order is None:
            return {
                "success": False,
                "code": "ORDER_NOT_FOUND",
                "message": "未找到该订单。",
                "data": None,
            }

        shipment = self.repository.get_shipment_by_order_id(
            order_id
        )

        if shipment is None:
            return {
                "success": False,
                "code": "SHIPMENT_NOT_FOUND",
                "message": "该订单暂无物流信息。",
                "data": {
                    "order_id": order_id,
                    "order_status": order["status"],
                },
            }

        return {
            "success": True,
            "code": "OK",
            "message": "物流查询成功。",
            "data": shipment,
        }