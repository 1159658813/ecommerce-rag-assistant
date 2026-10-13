import json
import os
import re
from typing import Optional

from openai import OpenAI

from src.config import settings

ORDER_ID_PATTERN = re.compile(
    r"\bORD\d+\b",
    re.IGNORECASE,
)

CONTEXT_REFERENCE_HINTS = (
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
    "这单",
    "那单",
    "已发货的",
    "已签收的",
    "现在到哪",
    "到哪了",
    "物流呢",
    "还能",
    "退款进度",
    "处理到哪",
    "处理完了吗",
)

GENERIC_POLICY_HINTS = (
    "一般",
    "通常",
    "规则",
    "政策",
    "规定",
    "条件",
    "多久",
    "到账时间",
)

SYSTEM_PROMPT = """
你是中文电商客服系统中的 Conversation-aware Query Rewriter。

你的任务不是回答用户问题，而是根据：
1. 当前用户问题
2. 最近对话历史
3. 当前结构化会话状态

把用户当前问题改写成一个语义完整、可以独立理解的问题。

你只能使用提供给你的信息，不允许编造订单号、订单状态、退款状态或其他业务事实。

规则：

1. 如果当前问题本身已经完整，不需要依赖历史信息：
   - rewritten_question 保持原问题
   - used_history 为 false

2. 如果当前问题包含指代，例如：
   - 它
   - 那个订单
   - 刚才那个
   - 第二个订单
   - 已发货的那个
   - 我刚才说的那个退款

   可以结合对话历史和结构化状态进行补全。

3. 如果历史中无法确定指代对象：
   - 不要猜测
   - 不要生成不存在的订单号
   - 尽量保留用户原始问题

4. 如果能够明确确定订单号，将它写入：
   resolved_entities.order_id

5. rewritten_question 应该是可以直接交给后续 Intent Router、
   RAG 或 Order Agent 使用的自然语言问题。

6. 不要回答用户的问题。

7. 必须只返回 JSON。

返回格式：

{
  "rewritten_question": "改写后的问题",
  "resolved_entities": {
    "order_id": null
  },
  "used_history": false
}

8. 如果当前问题已经明确包含订单号：
   - 当前问题中的订单号优先级最高
   - resolved_entities.order_id 必须使用当前问题中的订单号
   - 不要被历史中的其他订单号覆盖
   - used_history 为 false

9. 如果当前问题本身是完整的通用政策问题，例如：
   - 退款一般多久到账？
   - 退货政策是什么？
   - 售后规则怎么规定？
   则不要使用历史订单或 active_order_id，
   resolved_entities.order_id 应为 null，
   used_history 为 false。
"""


class QueryRewriter:
    """
    Rewrite a conversational user question into a
    self-contained question using recent history and
    structured conversation state.

    This component only rewrites the question.
    It does not perform intent routing, RAG, or tool calling.
    """

    def __init__(
            self,
            model: Optional[str] = None,
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

        # Query Rewriter 属于轻量、高频 LLM 任务。
        # 当前复用项目已有的 qwen3.6-flash 配置，
        # 暂时不新增独立环境变量。
        self.model = (
                model
                or settings.query_rewriter_model
        )

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )

    def rewrite(
            self,
            question: str,
            messages: Optional[list[dict]] = None,
            active_order_id: Optional[str] = None,
            last_intent: Optional[str] = None,
    ) -> dict:
        """
        Rewrite the current question using conversation context.

        Args:
            question:
                Current user question.

            messages:
                Recent conversation messages.
                Expected message shape:
                {
                    "role": "user" | "assistant",
                    "content": "..."
                }

            active_order_id:
                Structured conversation state.

            last_intent:
                Previous conversation intent.

        Returns:
            {
                "rewritten_question": "...",
                "resolved_entities": {
                    "order_id": "..." | None
                },
                "used_history": bool
            }
        """

        question = str(
            question or ""
        ).strip()

        if not question:
            return {
                "rewritten_question": "",
                "resolved_entities": {
                    "order_id": None,
                },
                "used_history": False,
            }
        # ----------------------------------------------------
        # Deterministic guard 1:
        # Explicit entity in the current question always wins.
        #
        # A self-contained question with an explicit order ID
        # does not require conversation history.
        # ----------------------------------------------------

        explicit_order_id = (
            self._extract_explicit_order_id(
                question
            )
        )

        if explicit_order_id:
            return {
                "rewritten_question": question,
                "resolved_entities": {
                    "order_id": explicit_order_id,
                },
                "used_history": False,
            }

        # ----------------------------------------------------
        # Deterministic guard 2:
        # Generic standalone policy questions must not inherit
        # an unrelated active order from the previous turn.
        # ----------------------------------------------------

        if self._is_standalone_policy_question(
                question
        ):
            return {
                "rewritten_question": question,
                "resolved_entities": {
                    "order_id": None,
                },
                "used_history": False,
            }

        messages = messages or []

        context_payload = {
            "current_question": question,
            "structured_state": {
                "active_order_id": (
                    active_order_id
                ),
                "last_intent": (
                    last_intent
                ),
            },
            "recent_messages": (
                self._prepare_messages(
                    messages
                )
            ),
        }

        response = (
            self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": json.dumps(
                            context_payload,
                            ensure_ascii=False,
                        ),
                    },
                ],
                temperature=0.0,
                response_format={
                    "type": "json_object"
                },
                extra_body={
                    "enable_thinking": False
                },
            )
        )

        content = (
            response
            .choices[0]
            .message
            .content
            .strip()
        )

        try:
            result = json.loads(
                content
            )
        except json.JSONDecodeError as exc:
            raise ValueError(
                "Query Rewriter 返回了无效 JSON。"
            ) from exc

        rewritten_question = str(
            result.get(
                "rewritten_question",
                question,
            )
            or question
        ).strip()

        resolved_entities = result.get(
            "resolved_entities",
            {},
        )

        if not isinstance(
                resolved_entities,
                dict,
        ):
            resolved_entities = {}

        order_id = resolved_entities.get(
            "order_id"
        )

        if order_id is not None:
            order_id = str(
                order_id
            ).strip().upper()

            if not order_id:
                order_id = None

        used_history = bool(
            result.get(
                "used_history",
                False,
            )
        )

        return {
            "rewritten_question": (
                rewritten_question
            ),
            "resolved_entities": {
                "order_id": order_id,
            },
            "used_history": (
                used_history
            ),
        }

    @staticmethod
    def _extract_explicit_order_id(
            question: str,
    ) -> Optional[str]:

        match = ORDER_ID_PATTERN.search(
            str(
                question or ""
            )
        )

        if match is None:
            return None

        return (
            match.group(0)
            .strip()
            .upper()
        )

    @staticmethod
    def _is_standalone_policy_question(
            question: str,
    ) -> bool:

        question = str(
            question or ""
        ).strip()

        if not question:
            return False

        if ORDER_ID_PATTERN.search(
                question
        ):
            return False

        has_context_reference = any(
            hint in question
            for hint
            in CONTEXT_REFERENCE_HINTS
        )

        if has_context_reference:
            return False

        has_generic_policy_signal = any(
            hint in question
            for hint
            in GENERIC_POLICY_HINTS
        )

        return has_generic_policy_signal

    @staticmethod
    def _prepare_messages(
            messages: list[dict],
    ) -> list[dict]:
        """
        Keep only the fields required by the LLM.

        Database-specific fields such as message_id,
        conversation_id and created_at are intentionally
        excluded.
        """

        prepared_messages = []

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

            prepared_messages.append(
                {
                    "role": role,
                    "content": content,
                }
            )

        return prepared_messages