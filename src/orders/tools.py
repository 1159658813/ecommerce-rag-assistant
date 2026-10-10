from typing import Optional

from src.orders.service import OrderService

# ============================================================
# OpenAI-Compatible Tool Definitions
# ============================================================

ORDER_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_order_detail",
            "description": (
                "查询当前用户某个具体订单的详细信息，"
                "包括订单状态、金额、创建时间以及物流信息。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": (
                            "需要查询的订单号，"
                            "例如 ORD202610002。"
                        ),
                    }
                },
                "required": [
                    "order_id"
                ],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_user_orders",
            "description": (
                "查询当前用户自己的全部订单列表。"
                "当用户说查看我的订单、有哪些订单等问题时使用。"
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_logistics",
            "description": (
                "查询当前用户某个订单的物流状态，"
                "包括物流公司、运单号、当前位置和预计送达时间。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": (
                            "需要查询物流的订单号，"
                            "例如 ORD202610002。"
                        ),
                    }
                },
                "required": [
                    "order_id"
                ],
                "additionalProperties": False,
            },
        },
    },
]


# ============================================================
# Tool Executor
# ============================================================

class OrderToolExecutor:
    """
    Execute order tools selected by the LLM.

    Important:
    user_id is supplied by the application,
    not by the language model.
    """

    def __init__(
            self,
            service: Optional[OrderService] = None,
    ):
        self.service = (
                service
                or OrderService()
        )

    def execute(
            self,
            tool_name: str,
            arguments: dict,
            user_id: str,
    ) -> dict:
        """
        Execute one order tool.
        """

        if tool_name == "get_order_detail":
            order_id = arguments.get(
                "order_id"
            )

            if not order_id:
                return {
                    "success": False,
                    "code": "INVALID_ARGUMENT",
                    "message": "缺少订单号。",
                    "data": None,
                }

            return self.service.get_order_detail(
                user_id=user_id,
                order_id=order_id,
            )

        if tool_name == "get_user_orders":
            return self.service.get_user_orders(
                user_id=user_id,
            )

        if tool_name == "get_logistics":
            order_id = arguments.get(
                "order_id"
            )

            if not order_id:
                return {
                    "success": False,
                    "code": "INVALID_ARGUMENT",
                    "message": "缺少订单号。",
                    "data": None,
                }

            return self.service.get_logistics(
                user_id=user_id,
                order_id=order_id,
            )

        return {
            "success": False,
            "code": "UNKNOWN_TOOL",
            "message": (
                f"未知订单工具：{tool_name}"
            ),
            "data": None,
        }