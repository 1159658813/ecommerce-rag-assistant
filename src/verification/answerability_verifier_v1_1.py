import json
import os
import re
import time

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()


VALID_VERDICTS = {
    "SUFFICIENT",
    "INSUFFICIENT",
}


class AnswerabilityVerifierV11:
    """
    Question + Top-K Evidence -> Evidence Coverage verdict

    目标：
    判断当前 Evidence 是否覆盖了回答用户问题所需的核心事实。

    注意：
    - 这不是 NLI。
    - contradiction 并不等于 insufficient。
    - 主题相关并不等于 evidence sufficient。
    - KB 未提到某规则，不能据此推断该规则不存在。
    """

    def __init__(
        self,
        model="qwen-max",
        timeout_seconds=60,
        max_retries=3,
        retry_base_seconds=2,
    ):
        api_key = os.getenv("DASHSCOPE_API_KEY")

        if not api_key:
            raise ValueError(
                "未检测到 DASHSCOPE_API_KEY，"
                "请检查项目根目录 .env 文件。"
            )

        base_url = os.getenv(
            "DASHSCOPE_BASE_URL",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        )

        self.model = model
        self.max_retries = max_retries
        self.retry_base_seconds = retry_base_seconds

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout_seconds,
            # 禁用 SDK 内部隐式重试，由本类统一控制。
            max_retries=0,
        )

    @staticmethod
    def _parse_json_response(text):
        text = str(text or "").strip()

        if not text:
            raise ValueError("Verifier 返回为空。")

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        code_block_match = re.search(
            r"```(?:json)?\s*(\{.*?\})\s*```",
            text,
            flags=re.S,
        )

        if code_block_match:
            try:
                return json.loads(
                    code_block_match.group(1)
                )
            except json.JSONDecodeError:
                pass

        object_match = re.search(
            r"\{.*\}",
            text,
            flags=re.S,
        )

        if object_match:
            try:
                return json.loads(
                    object_match.group(0)
                )
            except json.JSONDecodeError:
                pass

        raise ValueError(
            "Verifier 没有返回可解析 JSON：\n"
            f"{text}"
        )

    @staticmethod
    def _normalize_string_list(value):
        if value is None:
            return []

        if not isinstance(value, list):
            raise ValueError(
                "covered_facts / missing_facts 必须是数组。"
            )

        normalized = []

        for item in value:
            text = str(item or "").strip()
            if text:
                normalized.append(text)

        return normalized

    @classmethod
    def _validate_result(cls, result):
        if not isinstance(result, dict):
            raise ValueError(
                "Verifier 解析结果不是 JSON Object。"
            )

        verdict = str(
            result.get("verdict", "")
        ).upper().strip()

        if verdict not in VALID_VERDICTS:
            raise ValueError(
                f"Verifier verdict 非法：{verdict!r}"
            )

        covered_facts = cls._normalize_string_list(
            result.get("covered_facts")
        )

        missing_facts = cls._normalize_string_list(
            result.get("missing_facts")
        )

        reason = str(
            result.get("reason", "")
        ).strip()

        # 结构一致性检查。
        if verdict == "SUFFICIENT" and missing_facts:
            raise ValueError(
                "verdict=SUFFICIENT 时 "
                "missing_facts 必须为空。"
            )

        if verdict == "INSUFFICIENT" and not missing_facts:
            raise ValueError(
                "verdict=INSUFFICIENT 时 "
                "missing_facts 不能为空。"
            )

        return {
            "verdict": verdict,
            "covered_facts": covered_facts,
            "missing_facts": missing_facts,
            "reason": reason,
        }

    @staticmethod
    def _build_prompt(
        question,
        evidences,
    ):
        evidence_parts = []

        for index, evidence in enumerate(
            evidences,
            start=1,
        ):
            section = str(
                evidence.get(
                    "section",
                    "未知知识点",
                )
            ).strip()

            content = str(
                evidence.get(
                    "content",
                    "",
                )
            ).strip()

            evidence_parts.append(
                (
                    f"【证据 {index}】\n"
                    f"主题：{section}\n"
                    f"内容：{content}"
                )
            )

        evidence_text = "\n\n".join(
            evidence_parts
        )

        return f"""
你是一个严格的中文电商 RAG Evidence Coverage Verifier。

你的任务不是回答用户问题，
也不是判断某个用户命题是否被 Evidence 蕴含。

你的唯一任务是：

判断给出的多条【知识库证据】，
是否已经覆盖了回答【用户问题】所需要的核心事实。

==================================================
核心定义
==================================================

SUFFICIENT：
当前 Evidence 已经提供足够事实，
能够对用户真正询问的核心问题作出确定回答。

INSUFFICIENT：
Evidence 虽然可能与问题主题高度相关，
但缺少回答用户核心询问所必需的一个或多个事实。

==================================================
必须严格遵守
==================================================

1. 这不是 NLI / entailment 分类。

Evidence 与用户问题中的某个命题相矛盾，
仍然可能是 SUFFICIENT。

例如：

用户问：
“银行卡退款是不是当天一定到账？”

Evidence：
“银行卡退款通常需要3至7个工作日到账。”

这里应判断为 SUFFICIENT。
因为 Evidence 足以明确回答“不是当天一定到账”。

2. “主题相关”不等于“证据充分”。

例如：

用户问：
“商家超过48小时没发货，
平台是不是会自动赔订单金额的10%？”

Evidence：
“普通现货商品通常应在付款后48小时内发货；
超过48小时未发货时可以申请取消或联系客服。”

这里应判断为 INSUFFICIENT。

Evidence 覆盖了“48小时未发货”，
但没有覆盖：
- 平台是否自动赔偿
- 是否赔订单金额10%

3. 知识库没有提到某个规则，
不能推断该规则不存在、不会发生或不支持。

缺失证据 != 否定证据。

4. 不允许使用外部常识、模型记忆或猜测补全 Evidence。

5. 多条 Evidence 可以联合覆盖一个问题。
必须综合判断全部 Evidence，
不能只判断第一条。

6. 如果问题包含多个核心事实或多个需要判断的子问题，
只有当这些核心事实都已被 Evidence 覆盖时，
才可以判断为 SUFFICIENT。

如果只覆盖一部分，
应判断为 INSUFFICIENT，
并在 missing_facts 中指出缺失的核心事实。

7. covered_facts / missing_facts
只写简短的“事实槽位/规则点”，
不要写长篇分析过程。

8. reason 只需用一两句话说明最终判断依据，
不要输出隐藏推理过程或逐步思维链。

9. Evidence 不要求必须逐字出现用户问题中的最终结论。

如果 Evidence 已经明确给出不同规则各自的适用对象、条件或处理阶段，
并且这些信息足以区分用户混淆的两个规则，
则可以据此判断 Evidence 是 SUFFICIENT。

例如：

用户问：
“质量问题换货是不是也直接按七天无理由退货规则走？”

Evidence 分别说明：
- 七天无理由退货的适用条件；
- 商品质量问题可以申请换货，并有独立的换货申请条件。

即使 Evidence 没有逐字写出
“换货不按七天无理由退货规则处理”，
这些信息仍足以区分两个规则，因此应判断为 SUFFICIENT。

10. 如果 Evidence 已明确给出回答用户强断言所需的限定条件，
则不需要 Evidence 再逐字写出对该强断言的否定句。

例如：

用户问：
“取消未付款订单以后，优惠券是不是马上一定能用于下一单？”

Evidence：
“未付款订单取消后，优惠券通常会在10分钟内释放。”

这里应判断为 SUFFICIENT。

因为 Evidence 已足以说明：
不能保证“马上”，而是需要等待释放，通常在10分钟内。

11. 对“一定、完全、任何、必然、都不用”等绝对化表述，
只要 Evidence 提供了一个明确的例外、可能性或条件，
就足以否定该绝对命题。

例如：

用户问：
“质量问题退货是不是一定什么证据都不用提供？”

Evidence：
“质量问题售后时，平台或商家可能要求上传照片或视频。”

这里应判断为 SUFFICIENT。

因为“可能要求证据”已经足以否定
“一定不需要任何证据”。

==================================================
用户问题
==================================================

{question}

==================================================
知识库证据
==================================================

{evidence_text}

==================================================
输出
==================================================

只输出一个 JSON Object，不要输出 Markdown。

SUFFICIENT 示例：

{{
  "verdict": "SUFFICIENT",
  "covered_facts": [
    "银行卡退款到账时效"
  ],
  "missing_facts": [],
  "reason": "Evidence 已明确给出银行卡退款所需时间，足以回答用户关于是否当天到账的判断。"
}}

INSUFFICIENT 示例：

{{
  "verdict": "INSUFFICIENT",
  "covered_facts": [
    "48小时未发货的处理规则"
  ],
  "missing_facts": [
    "平台是否自动赔偿",
    "赔偿是否为订单金额10%"
  ],
  "reason": "Evidence 只覆盖发货超时处理，没有提供用户询问的赔偿规则。"
}}
""".strip()

    def _call_once(
        self,
        question,
        evidences,
    ):
        prompt = self._build_prompt(
            question=question,
            evidences=evidences,
        )

        completion = (
            self.client
            .chat
            .completions
            .create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a strict RAG "
                            "evidence coverage verifier."
                        ),
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                temperature=0,
            )
        )

        raw_output = (
            completion
            .choices[0]
            .message
            .content
        )

        parsed = self._parse_json_response(
            raw_output
        )

        result = self._validate_result(
            parsed
        )

        result["raw_output"] = raw_output

        return result

    def verify(
        self,
        question,
        evidences,
    ):
        """
        Fail closed：
        - 没有 Evidence -> INSUFFICIENT
        - API/解析异常由调用方记录为 ERROR，
          不应被当作 SUFFICIENT 放行。
        """
        question = str(
            question or ""
        ).strip()

        if not question:
            raise ValueError(
                "question 不能为空。"
            )

        if not evidences:
            return {
                "verdict": "INSUFFICIENT",
                "covered_facts": [],
                "missing_facts": [
                    "缺少可用于回答问题的知识库证据"
                ],
                "reason": (
                    "当前没有提供任何 Evidence。"
                ),
                "raw_output": None,
            }

        last_error = None

        for attempt in range(
            1,
            self.max_retries + 1,
        ):
            try:
                return self._call_once(
                    question=question,
                    evidences=evidences,
                )

            except Exception as error:
                last_error = error

                print(
                    "[Verifier Retry] "
                    f"attempt={attempt}/"
                    f"{self.max_retries} "
                    f"error={repr(error)}"
                )

                if attempt < self.max_retries:
                    time.sleep(
                        self.retry_base_seconds
                        * attempt
                    )

        raise last_error
