"""Conservative, synchronous reliability primitives for P1-6."""

import time
from dataclasses import dataclass
from typing import Callable, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class RetryPolicy:
    """Explicit retries; SDK auto retries must be disabled by callers."""

    max_attempts: int = 2
    delay_seconds: float = 0.0

    def __post_init__(self):
        if self.max_attempts < 1 or self.delay_seconds < 0:
            raise ValueError("Invalid retry policy configuration")

    def run(self, operation: Callable[[], T], retry_on: tuple[type[Exception], ...]) -> T:
        for attempt in range(1, self.max_attempts + 1):
            try:
                return operation()
            except retry_on:
                if attempt == self.max_attempts:
                    raise
                if self.delay_seconds:
                    time.sleep(self.delay_seconds)
        raise AssertionError("unreachable")


class FallbackPolicy:
    """No model-generated facts or policies in fallback answers."""

    @staticmethod
    def router_result() -> dict:
        return {
            "intent": "UNSUPPORTED",
            "reason": "router_unavailable",
            "fallback": True,
        }

    @staticmethod
    def unavailable_message() -> str:
        return "当前服务暂时繁忙，无法可靠处理此问题，请稍后重试。"

    @staticmethod
    def rag_message() -> str:
        return "根据当前可用信息，暂时无法可靠确认，请稍后重试。"

    @staticmethod
    def order_answer(tool_results: list[dict]) -> str:
        """Only echo values that were returned by authenticated services."""
        if not tool_results:
            return "订单查询服务暂时不可用，请稍后重试。"
        if len(tool_results) == 1:
            tool_result = tool_results[0].get("result") or {}
            if not tool_result.get("success"):
                return str(tool_result.get("message") or "订单查询暂时无法完成，请稍后重试。")
            data = tool_result.get("data") or {}
            if isinstance(data, dict):
                order = data.get("order")
                if isinstance(order, dict) and order.get("order_id") and order.get("status"):
                    return (f"已查询到订单 {order['order_id']}，当前状态为 {order['status']}。"
                            "详细回复暂时无法生成，请稍后重试。")
                if data.get("refund_id") and data.get("status"):
                    return (f"已查询到退款单 {data['refund_id']}，当前状态为 {data['status']}。"
                            "详细回复暂时无法生成，请稍后重试。")
        return "已获取订单系统的查询结果，但暂时无法生成完整回复，请稍后重试。"
