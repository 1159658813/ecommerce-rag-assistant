"""P2-B non-billable verifier thinking-toggle and Evidence Gate regressions.

Run: python -m unittest tests.test_p2b_verifier_toggle -v
No models, API keys, Docker, or network needed.
"""
import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from src.config.settings import optional_bool_env, load_settings
from src.pipeline.rag_pipeline import RAGPipeline
from src.verification.answerability_verifier_v1_2 import (
    AnswerabilityVerifierV12,
    VerifierExhausted,
)


class ThinkingFlagTests(unittest.TestCase):
    def test_unset_does_not_override_provider_default(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(optional_bool_env("ANSWERABILITY_VERIFIER_ENABLE_THINKING"))
            self.assertIsNone(load_settings().verifier_enable_thinking)

    def test_env_values_and_invalid_value(self):
        for value, expected in [("false", False), ("0", False), ("off", False),
                                ("true", True), ("YES", True), ("1", True)]:
            with self.subTest(value=value), patch.dict(os.environ, {"ANSWERABILITY_VERIFIER_ENABLE_THINKING": value}):
                self.assertIs(optional_bool_env("ANSWERABILITY_VERIFIER_ENABLE_THINKING"), expected)
        with patch.dict(os.environ, {"ANSWERABILITY_VERIFIER_ENABLE_THINKING": "maybe"}):
            with self.assertRaises(ValueError):
                optional_bool_env("ANSWERABILITY_VERIFIER_ENABLE_THINKING")

    @staticmethod
    def mock_verifier(flag):
        verifier = AnswerabilityVerifierV12.__new__(AnswerabilityVerifierV12)
        verifier.model = "qwen3.6-plus"
        verifier.enable_thinking = flag
        message = SimpleNamespace(content='{"verdict":"SUFFICIENT","covered_facts":["有效规则"],"missing_facts":[],"reason":"证据覆盖"}')
        completion = SimpleNamespace(choices=[SimpleNamespace(message=message)])
        create = Mock(return_value=completion)
        verifier.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        return verifier, create

    def test_request_flag_default_off_and_on(self):
        for flag in (None, False, True):
            with self.subTest(flag=flag):
                verifier, create = self.mock_verifier(flag)
                result = verifier._call_once("确认规则", [{"section": "售后", "content": "有效规则"}])
                self.assertEqual(result["verdict"], "SUFFICIENT")
                kwargs = create.call_args.kwargs
                self.assertEqual(kwargs["model"], "qwen3.6-plus")
                self.assertEqual(kwargs["temperature"], 0)
                self.assertNotIn("max_tokens", kwargs)
                if flag is None:
                    self.assertNotIn("extra_body", kwargs)
                else:
                    self.assertEqual(kwargs["extra_body"], {"enable_thinking": flag})
                self.assertIn("严格遵守", kwargs["messages"][1]["content"])

    def test_invalid_sufficient_is_rejected_even_with_thinking_off(self):
        verifier, create = self.mock_verifier(False)
        create.return_value.choices[0].message.content = (
            '{"verdict":"SUFFICIENT","covered_facts":[],"missing_facts":["物流赔偿规则"]}'
        )
        with self.assertRaises(ValueError):
            verifier._call_once("是否赔偿", [{"content": "48小时未发货可以联系客服"}])

    def test_exhaustion_abstains_and_does_not_call_generator(self):
        verifier, create = self.mock_verifier(False)
        verifier.max_retries = 2
        verifier.retry_base_seconds = 0
        create.side_effect = TimeoutError("mock network failure")
        generator = Mock()
        pipeline = RAGPipeline(
            retriever=SimpleNamespace(retrieve=lambda **_: [{"document": {"section": "rules", "text": "any evidence"}}]),
            verifier=verifier,
            generator=generator,
        )
        result = pipeline.ask("退款规则是什么")
        self.assertTrue(result["abstained"])
        self.assertEqual(result["abstain_reason"], "verifier_unavailable")
        self.assertEqual(result["observability"]["verifier_retry_count"], 1)
        self.assertEqual(create.call_count, 2)
        generator.generate.assert_not_called()

    def test_benchmark_collects_only_approved_quality_fields(self):
        from scripts.p2_perf_baseline import one_request
        response = {"intent": "KNOWLEDGE_QA", "handler": "RAG", "answer": "secret response",
                    "user_id": "secret-user", "result": {"verdict": "INSUFFICIENT",
                    "abstained": True, "abstain_reason": "evidence_insufficient",
                    "answer": "secret reply"}}
        with patch("scripts.p2_perf_baseline.request_json", return_value=(200, response)):
            sample = one_request("http://127.0.0.1:8000", "knowledge_insufficient", "U001", 30, 1)
        self.assertEqual(sample["verdict"], "INSUFFICIENT")
        self.assertTrue(sample["abstained"])
        self.assertTrue(sample["intent_matches"])
        self.assertNotIn("answer", sample)
        self.assertNotIn("user_id", sample)
        self.assertNotIn("question", sample)

    def test_insufficient_never_calls_composer(self):
        verifier, create = self.mock_verifier(False)
        create.return_value.choices[0].message.content = (
            '{"verdict":"INSUFFICIENT","covered_facts":[],"missing_facts":["赔偿规则"],"reason":"缺证据"}'
        )
        verifier.max_retries = 1
        generator = Mock()
        pipeline = RAGPipeline(
            retriever=SimpleNamespace(retrieve=lambda **_: [{"document": {"section": "rules", "text": "only shipping deadline"}}]),
            verifier=verifier,
            generator=generator,
        )
        result = pipeline.ask("超过48小时未发货是否赔偿10%")
        self.assertTrue(result["abstained"])
        self.assertEqual(result["abstain_reason"], "evidence_insufficient")
        generator.generate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
