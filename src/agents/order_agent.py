import json
import os
import time
from typing import Optional

from openai import OpenAI

from src.reliability import FallbackPolicy
from src.config import settings
from src.orders.tools import (
    ORDER_TOOLS,
    OrderToolExecutor,
)

SYSTEM_PROMPT = """
你是电商客服系统中的订单助手。

你的职责是处理当前用户自己的订单、物流和退款业务数据查询。

规则：

1. 涉及订单状态、订单详情、物流位置、
   退款状态、退款金额、退款详情等事实时，
   必须调用提供的工具获取真实数据。

2. 禁止根据常识、猜测或模型记忆编造订单信息。

3. 当前用户身份由系统负责验证，
   不要自行生成或修改用户身份。

4. 如果工具返回订单不存在，
   直接礼貌告知用户未找到该订单。

5. 如果工具返回暂无物流信息，
   应根据工具提供的信息向用户解释。

6. 如果工具返回暂无退款记录或未找到退款，
   应直接根据工具结果说明，不允许自行编造退款进度。

7. 通用退款政策、退款条件、退款到账规则等问题
   不属于实时退款事实，不要凭常识编造政策。

8. 回答应简洁、自然，符合电商客服语气。

9. 不要向用户展示内部工具名称、
   内部错误代码或数据库结构。
"""


class OrderAgent:
    """
    Native LLM tool-calling agent for order queries.

    No LangChain or LangGraph is used.
    """

    def __init__(
            self,
            model: Optional[str] = None,
            tool_executor: Optional[
                OrderToolExecutor
            ] = None,
    ):
        api_key = os.getenv(
            "DASHSCOPE_API_KEY"
        )

        if not api_key:
            raise ValueError(
                "未检测到 DASHSCOPE_API_KEY，"
                "请检查 .env 文件。"
            )

        base_url = os.getenv(
            "DASHSCOPE_BASE_URL",
            (
                "https://dashscope.aliyuncs.com/"
                "compatible-mode/v1"
            ),
        )

        self.model = (
                model
                or settings.generator_model
        )

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=20.0,
            max_retries=0,
        )

        self.tool_executor = (
                tool_executor
                or OrderToolExecutor()
        )

    def ask(self, question: str, user_id: str) -> dict:
        question = str(question or "").strip()
        latency_tool = 0.0
        latency_composer = 0.0
        tools_called = []
        tool_results = []

        def response(answer, abstained=False, reason=None):
            return {
                "answer": answer,
                "tools_called": tools_called,
                "tool_results": tool_results,
                "abstained": bool(abstained),
                "abstain_reason": reason,
                "observability": {
                    "latency_tool": round(latency_tool, 3),
                    "latency_composer": round(latency_composer, 3),
                    "tools_called": list(tools_called),
                    "verifier_retry_count": 0,
                    "abstained": bool(abstained),
                    "latency_unit": "ms",
                },
            }

        if not question:
            return response("请输入需要查询的订单问题。")

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]

        # Initial routing is untrusted; no tool was run if this call fails.
        try:
            first = self.client.chat.completions.create(
                model=self.model, messages=messages, tools=ORDER_TOOLS,
                tool_choice="auto", temperature=0.0,
                extra_body={"enable_thinking": False},
            )
            message = first.choices[0].message
            tool_calls = message.tool_calls or []
        except Exception:
            return response(FallbackPolicy.unavailable_message(), True, "tool_selection_unavailable")

        if not tool_calls:
            # No verified tool facts: do not pass through an LLM-created order status.
            return response("请提供需要查询的订单号或退款单号，以便核实真实信息。")

        messages.append(message.model_dump(exclude_none=True))

        for tool_call in tool_calls:
            tool_name = getattr(tool_call.function, "name", "")
            raw = getattr(tool_call.function, "arguments", "") or "{}"
            tools_called.append(tool_name)
            try:
                arguments = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                arguments = None

            started = time.perf_counter()
            try:
                result = self.tool_executor.execute(
                    tool_name=tool_name, arguments=arguments, user_id=user_id,
                )
            except Exception:
                result = {
                    "success": False,
                    "code": "TOOL_UNAVAILABLE",
                    "message": "订单数据查询暂时不可用，请稍后重试。",
                    "data": None,
                }
            finally:
                latency_tool += (time.perf_counter() - started) * 1000

            tool_results.append({
                "tool_name": tool_name,
                "arguments": arguments if isinstance(arguments, dict) else {},
                "result": result,
            })
            messages.append({
                "role": "tool", "tool_call_id": tool_call.id,
                "content": json.dumps(result, ensure_ascii=False),
            })

        # An invalid or unavailable tool result cannot be repaired by free-text LLM output.
        if any(not x["result"].get("success") for x in tool_results):
            return response(FallbackPolicy.order_answer(tool_results), True, "tool_failed")

        started = time.perf_counter()
        try:
            final = self.client.chat.completions.create(
                model=self.model, messages=messages, temperature=0.0,
                extra_body={"enable_thinking": False},
            )
            answer = (final.choices[0].message.content or "").strip()
            if not answer:
                latency_composer = (time.perf_counter() - started) * 1000
                return response(FallbackPolicy.order_answer(tool_results), True, "composer_empty")
            latency_composer = (time.perf_counter() - started) * 1000
            return response(answer)
        except Exception:
            latency_composer = (time.perf_counter() - started) * 1000
            return response(FallbackPolicy.order_answer(tool_results), True, "composer_unavailable")
        finally:
            latency_composer = (time.perf_counter() - started) * 1000
