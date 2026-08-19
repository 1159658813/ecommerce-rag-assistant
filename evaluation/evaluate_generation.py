import json
import os
import re
import sys
import time
from pathlib import Path

import torch
from dotenv import load_dotenv
from openai import OpenAI
from transformers import AutoModelForSequenceClassification, AutoTokenizer


# ============================================================
# Project Root / Imports
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retriever import Retriever
from generation.answer_generator import AnswerGenerator


# ============================================================
# Environment / Paths
# ============================================================

load_dotenv(PROJECT_ROOT / ".env")

QUESTIONS_PATH = PROJECT_ROOT / "evaluation" / "generation_questions_v1.json"
INDEX_PATH = PROJECT_ROOT / "data" / "index" / "knowledge.faiss"
METADATA_PATH = PROJECT_ROOT / "data" / "index" / "chunks.json"
OUTPUT_PATH = PROJECT_ROOT / "evaluation" / "generation_results_v1.json"
CHECKPOINT_PATH = PROJECT_ROOT / "evaluation" / "generation_results_v1.checkpoint.json"


# ============================================================
# Configuration
# ============================================================

# oracle:
#   expected_sections -> KB -> AnswerGenerator
#   单独测试 Generation 能力。
#
# reranker_top1:
#   Dense Top-K -> BGE Reranker -> Top-1 -> AnswerGenerator
#   用于端到端 RAG 测试。
EVIDENCE_MODE = os.getenv("GENERATION_EVIDENCE_MODE", "oracle").strip().lower()

CANDIDATE_K = int(os.getenv("GENERATION_CANDIDATE_K", "10"))

RERANKER_MODEL_NAME = os.getenv(
    "RERANKER_MODEL_NAME",
    "BAAI/bge-reranker-v2-m3",
)
RERANKER_MAX_LENGTH = int(os.getenv("RERANKER_MAX_LENGTH", "512"))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

JUDGE_MODEL = os.getenv("GENERATION_JUDGE_MODEL", "qwen-max")
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
DASHSCOPE_BASE_URL = os.getenv(
    "DASHSCOPE_BASE_URL",
    "https://dashscope.aliyuncs.com/compatible-mode/v1",
)

JUDGE_TIMEOUT_SECONDS = float(os.getenv("GENERATION_JUDGE_TIMEOUT", "60"))
JUDGE_MAX_RETRIES = int(os.getenv("GENERATION_JUDGE_MAX_RETRIES", "3"))
JUDGE_RETRY_BASE_SECONDS = float(os.getenv("GENERATION_JUDGE_RETRY_BASE", "2"))

GENERATOR_MAX_RETRIES = int(os.getenv("GENERATION_MAX_RETRIES", "2"))
GENERATOR_RETRY_BASE_SECONDS = float(os.getenv("GENERATION_RETRY_BASE", "2"))

VALID_VERDICTS = {"PASS", "PARTIAL", "FAIL"}


# ============================================================
# Basic Helpers
# ============================================================

def safe_divide(numerator, denominator):
    return 0.0 if denominator == 0 else numerator / denominator


def normalize_section_name(value):
    return str(value or "").strip()


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def print_separator(char="=", width=120):
    print(char * width)


def classify_exception(error):
    """把常见 API / 解析异常归到稳定的 ERROR 类型。"""
    name = type(error).__name__.upper()
    message = repr(error).upper()

    if "TIMEOUT" in name or "TIMED OUT" in message or "TIMEOUT" in message:
        return "JUDGE_TIMEOUT"

    if "JSON" in name or "PARSE" in name or "可解析 JSON" in str(error):
        return "JUDGE_PARSE_ERROR"

    return "JUDGE_API_ERROR"


# ============================================================
# Validation
# ============================================================

def validate_environment():
    if EVIDENCE_MODE not in {"oracle", "reranker_top1"}:
        raise ValueError(
            "GENERATION_EVIDENCE_MODE 只能是 'oracle' 或 'reranker_top1'。"
        )

    if not DASHSCOPE_API_KEY:
        raise ValueError(
            "没有检测到 DASHSCOPE_API_KEY。\n"
            "请在项目根目录 .env 中配置：\n"
            "DASHSCOPE_API_KEY=你的API_KEY"
        )

    if not QUESTIONS_PATH.exists():
        raise FileNotFoundError(f"找不到 Generation 测试集：{QUESTIONS_PATH}")

    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"找不到 Chunk Metadata：{METADATA_PATH}")

    if EVIDENCE_MODE == "reranker_top1" and not INDEX_PATH.exists():
        raise FileNotFoundError(f"找不到 FAISS Index：{INDEX_PATH}")


def validate_dataset(questions):
    if not isinstance(questions, list) or not questions:
        raise ValueError("generation_questions_v1.json 必须是非空 JSON 数组。")

    seen_ids = set()
    errors = []

    for index, item in enumerate(questions, start=1):
        if not isinstance(item, dict):
            errors.append(f"第 {index} 条不是 JSON Object。")
            continue

        question_id = str(item.get("id", "")).strip()
        question = str(item.get("question", "")).strip()
        reference_answer = str(item.get("reference_answer", "")).strip()
        expected_sections = item.get("expected_sections")
        answerable = item.get("answerable", True)

        if not question_id:
            errors.append(f"第 {index} 条缺少 id。")
        elif question_id in seen_ids:
            errors.append(f"重复 id：{question_id}")
        else:
            seen_ids.add(question_id)

        if not question:
            errors.append(f"{question_id or index}: question 为空。")

        if not reference_answer:
            errors.append(f"{question_id or index}: reference_answer 为空。")

        if not isinstance(expected_sections, list):
            errors.append(f"{question_id or index}: expected_sections 必须是数组。")
            continue

        if not isinstance(answerable, bool):
            errors.append(f"{question_id or index}: answerable 必须是 true/false。")
            continue

        normalized_expected = [
            normalize_section_name(section)
            for section in expected_sections
            if normalize_section_name(section)
        ]

        if answerable and not normalized_expected:
            errors.append(
                f"{question_id or index}: answerable=true 但 expected_sections 为空。"
            )

        if not answerable and normalized_expected:
            errors.append(
                f"{question_id or index}: answerable=false 时 expected_sections 应为空。"
            )

    if errors:
        raise ValueError(
            "Generation 数据集校验失败：\n- " + "\n- ".join(errors)
        )


# ============================================================
# Load Data
# ============================================================

def load_json(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


validate_environment()

questions = load_json(QUESTIONS_PATH)
chunks = load_json(METADATA_PATH)

validate_dataset(questions)

if not isinstance(chunks, list) or not chunks:
    raise ValueError("chunks.json 为空或格式不是数组。")

print("Generation Questions:", len(questions))
print("Knowledge Chunks:", len(chunks))


# ============================================================
# Build immutable Section -> Chunks Map
# ============================================================

section_map = {}

for chunk in chunks:
    section = normalize_section_name(chunk.get("section"))
    text = str(chunk.get("text") or "").strip()

    if not section or not text:
        continue

    section_map.setdefault(section, []).append(chunk)

print("Knowledge Sections:", len(section_map))


# ============================================================
# Oracle Preflight
# ============================================================

def preflight_oracle_coverage():
    """
    Oracle 评测开始前一次性检查：
    所有 answerable=true 的 expected_sections 必须真实存在于 KB。

    这样可以避免跑到第 30 条才发现数据/KB 配置错误，
    也避免把基础设施问题误记成 Generation FAIL。
    """
    if EVIDENCE_MODE != "oracle":
        return

    missing = []

    for item in questions:
        if not item.get("answerable", True):
            continue

        for raw_section in item.get("expected_sections", []):
            section = normalize_section_name(raw_section)
            if section not in section_map:
                missing.append(
                    {
                        "id": item["id"],
                        "question": item["question"],
                        "section": section,
                    }
                )

    if missing:
        lines = [
            f"{x['id']} | section={x['section']} | {x['question']}"
            for x in missing
        ]
        raise ValueError(
            "Oracle preflight 失败：以下 expected_sections 在 chunks.json 中不存在：\n- "
            + "\n- ".join(lines)
        )


preflight_oracle_coverage()


# ============================================================
# Load Answer Generator
# ============================================================

print("\n正在加载 AnswerGenerator...")
answer_generator = AnswerGenerator()
print("AnswerGenerator 加载完成。")


# ============================================================
# Optional Retriever + Reranker
# ============================================================

retriever = None
reranker_tokenizer = None
reranker_model = None

if EVIDENCE_MODE == "reranker_top1":
    print("\n正在加载 Dense Retriever...")
    retriever = Retriever(
        index_path=INDEX_PATH,
        metadata_path=METADATA_PATH,
    )
    print("Dense Retriever 加载完成。")

    print("\n正在加载 Reranker...")
    print("Model:", RERANKER_MODEL_NAME)
    print("Device:", DEVICE)

    reranker_tokenizer = AutoTokenizer.from_pretrained(RERANKER_MODEL_NAME)

    if DEVICE == "cuda":
        reranker_model = (
            AutoModelForSequenceClassification.from_pretrained(
                RERANKER_MODEL_NAME,
                dtype=torch.float16,
            )
            .to(DEVICE)
        )
    else:
        reranker_model = (
            AutoModelForSequenceClassification.from_pretrained(
                RERANKER_MODEL_NAME
            )
            .to(DEVICE)
        )

    reranker_model.eval()
    print("Reranker 加载完成。")


# ============================================================
# Qwen Judge Client
# ============================================================

judge_client = OpenAI(
    api_key=DASHSCOPE_API_KEY,
    base_url=DASHSCOPE_BASE_URL,
    timeout=JUDGE_TIMEOUT_SECONDS,
    # 禁用 SDK 内部隐式重试，由本脚本统一控制，便于记录错误类型。
    max_retries=0,
)


# ============================================================
# Retrieval / Evidence Functions
# ============================================================

def rerank(query, dense_results):
    if not dense_results:
        return []

    passages = [result["document"]["text"] for result in dense_results]
    queries = [query] * len(passages)

    inputs = reranker_tokenizer(
        queries,
        passages,
        padding=True,
        truncation=True,
        max_length=RERANKER_MAX_LENGTH,
        return_tensors="pt",
    )
    inputs = {key: value.to(DEVICE) for key, value in inputs.items()}

    with torch.inference_mode():
        outputs = reranker_model(**inputs)
        raw_scores = outputs.logits.view(-1).float()
        normalized_scores = torch.sigmoid(raw_scores)

    results = []

    for dense_result, raw_score, normalized_score in zip(
        dense_results,
        raw_scores,
        normalized_scores,
    ):
        results.append(
            {
                "document": dense_result["document"],
                "dense_score": float(dense_result["score"]),
                "reranker_logit": float(raw_score.item()),
                "reranker_score": float(normalized_score.item()),
            }
        )

    results.sort(key=lambda item: item["reranker_logit"], reverse=True)
    return results


def get_oracle_evidences(expected_sections):
    """只读 section_map；绝不 pop / del / 修改 KB 映射。"""
    evidences = []
    missing_sections = []

    for raw_section in expected_sections:
        section = normalize_section_name(raw_section)
        matching_chunks = section_map.get(section, [])

        if not matching_chunks:
            missing_sections.append(section)
            continue

        for chunk in matching_chunks:
            evidences.append(
                {
                    "section": section,
                    "content": str(chunk["text"]).strip(),
                }
            )

    return evidences, missing_sections


def get_reranker_top1_evidence(question):
    dense_results = retriever.retrieve(query=question, top_k=CANDIDATE_K)
    reranked_results = rerank(query=question, dense_results=dense_results)

    debug_info = {
        "dense_sections": [
            item["document"].get("section")
            for item in dense_results
        ],
        "reranked_sections": [
            item["document"].get("section")
            for item in reranked_results
        ],
        "top1_section": None,
        "reranker_score": None,
    }

    if not reranked_results:
        return [], debug_info

    top1 = reranked_results[0]
    document = top1["document"]

    evidence = {
        "section": document["section"],
        "content": document["text"],
    }

    debug_info["top1_section"] = evidence["section"]
    debug_info["reranker_score"] = top1["reranker_score"]

    return [evidence], debug_info


# ============================================================
# Deterministic Diagnostics
# ============================================================

def run_keyword_diagnostics(answer, must_include, must_not_include):
    """
    只做诊断，不做最终判分。
    must_include / must_not_include 的最终语义判断交给 LLM Judge。
    """
    missing_items = [item for item in must_include if item not in answer]
    forbidden_items = [item for item in must_not_include if item in answer]
    return missing_items, forbidden_items


# ============================================================
# Judge Helpers
# ============================================================

def parse_json_response(text):
    text = str(text or "").strip()

    if not text:
        raise ValueError("Judge 返回为空。")

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
            return json.loads(code_block_match.group(1))
        except json.JSONDecodeError:
            pass

    object_match = re.search(r"\{.*\}", text, flags=re.S)
    if object_match:
        try:
            return json.loads(object_match.group(0))
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Judge 没有返回可解析 JSON：\n{text}")


def validate_judge_result(result):
    if not isinstance(result, dict):
        raise ValueError("Judge 解析结果不是 JSON Object。")

    verdict = str(result.get("verdict", "")).upper().strip()
    if verdict not in VALID_VERDICTS:
        raise ValueError(f"Judge verdict 非法：{verdict!r}")

    result["verdict"] = verdict

    for key in ("correctness", "groundedness", "completeness"):
        value = result.get(key)
        if value not in (0, 1, 2):
            raise ValueError(f"Judge 字段 {key} 非法：{value!r}")

    for key in ("contradiction", "hallucination"):
        value = result.get(key)
        if not isinstance(value, bool):
            raise ValueError(f"Judge 字段 {key} 必须是 bool：{value!r}")

    if "reason" not in result:
        result["reason"] = ""

    return result


def build_judge_prompt(
    question,
    answerable,
    evidences,
    candidate_answer,
    reference_answer,
    must_include,
    must_not_include,
):
    evidence_text = "\n\n".join(
        f"[Section] {item['section']}\n[Content] {item['content']}"
        for item in evidences
    )

    if not evidence_text:
        evidence_text = "[NO EVIDENCE]"

    return f"""
你是一个严格的中文电商客服 RAG 回答评测器。

你的任务不是重新回答用户问题，而是判断 Candidate Answer 是否正确。
只能根据：
1. User Question
2. Answerable Annotation
3. Retrieved Evidence
4. Reference Answer
5. Important Semantic Requirements
进行评判。

禁止使用你自己的外部常识替 Candidate Answer 补充内容。

==================================================
User Question
==================================================
{question}

==================================================
Answerable Annotation
==================================================
answerable = {str(answerable).lower()}

特别规则：
- answerable=false 表示当前知识库故意没有足够信息回答该问题。
- 当 answerable=false 且 Candidate Answer 明确表示“根据当前知识库无法确认/暂无足够信息”时，这是正确的拒答行为。
- 不得因为 Retrieved Evidence 为空而惩罚这种正确拒答。
- answerable=true 时，回答必须由 Evidence 支撑。

==================================================
Retrieved Evidence
==================================================
{evidence_text}

==================================================
Reference Answer
==================================================
{reference_answer}

==================================================
Important Semantic Requirements
==================================================
must_include:
{json.dumps(must_include, ensure_ascii=False)}

must_not_include:
{json.dumps(must_not_include, ensure_ascii=False)}

==================================================
Candidate Answer
==================================================
{candidate_answer}

==================================================
Evaluation Rules
==================================================
从以下四个维度判断：

1. correctness
   核心结论是否正确。

2. groundedness
   answerable=true 时，回答是否由 Evidence 支持，是否出现 Evidence 中不存在的事实性断言。
   answerable=false 时，正确拒答不要求 Evidence 提供事实答案。

3. completeness
   是否覆盖用户真正询问的核心问题和 Reference Answer 中的关键语义。

4. contradiction
   是否与 Evidence、Reference Answer 或 answerable 标注明显矛盾。

最终 verdict 只能是：

PASS
    核心结论正确，必要信息完整，没有重要幻觉或矛盾。

PARTIAL
    基本方向正确，但存在明显遗漏、表达不完整或轻微 unsupported claim。

FAIL
    核心结论错误、与 Evidence/Reference Answer 矛盾、严重遗漏，
    对 answerable=false 的问题擅自给出无依据确定性答案，
    或出现会误导用户的重要幻觉。

must_include 是语义要求，不是字符串完全匹配要求。
must_not_include 也是语义判断，不是简单字符串匹配。

==================================================
Output
==================================================
只输出一个 JSON Object，不要输出 Markdown：

{{
  "verdict": "PASS",
  "correctness": 2,
  "groundedness": 2,
  "completeness": 2,
  "contradiction": false,
  "hallucination": false,
  "reason": "简短说明原因"
}}

correctness / groundedness / completeness：
2 = 好
1 = 部分满足
0 = 不满足
""".strip()


def call_judge_once(prompt):
    completion = judge_client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are a strict RAG evaluation judge.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=0,
    )

    raw_output = completion.choices[0].message.content
    parsed = parse_json_response(raw_output)
    parsed = validate_judge_result(parsed)
    parsed["raw_judge_output"] = raw_output
    return parsed


def judge_answer(
    question,
    answerable,
    evidences,
    candidate_answer,
    reference_answer,
    must_include,
    must_not_include,
):
    prompt = build_judge_prompt(
        question=question,
        answerable=answerable,
        evidences=evidences,
        candidate_answer=candidate_answer,
        reference_answer=reference_answer,
        must_include=must_include,
        must_not_include=must_not_include,
    )

    last_error = None

    for attempt in range(1, JUDGE_MAX_RETRIES + 1):
        try:
            return call_judge_once(prompt)
        except Exception as error:
            last_error = error
            print(
                f"[Judge Retry] attempt={attempt}/{JUDGE_MAX_RETRIES} "
                f"error={repr(error)}"
            )

            if attempt < JUDGE_MAX_RETRIES:
                time.sleep(JUDGE_RETRY_BASE_SECONDS * attempt)

    raise last_error


def generate_answer_with_retry(question, evidences):
    last_error = None

    for attempt in range(1, GENERATOR_MAX_RETRIES + 1):
        try:
            answer = answer_generator.generate(
                question=question,
                evidences=evidences,
            )
            return str(answer).strip()
        except Exception as error:
            last_error = error
            print(
                f"[Generator Retry] attempt={attempt}/{GENERATOR_MAX_RETRIES} "
                f"error={repr(error)}"
            )

            if attempt < GENERATOR_MAX_RETRIES:
                time.sleep(GENERATOR_RETRY_BASE_SECONDS * attempt)

    raise last_error


# ============================================================
# Result Helpers
# ============================================================

def make_base_record(
    item,
    evidences,
    retrieval_debug,
    top1_section,
    retrieval_top1_correct,
):
    return {
        "id": item["id"],
        "question": item["question"],
        "category": item.get("category"),
        "answerable": item.get("answerable", True),
        "expected_sections": item.get("expected_sections", []),
        "reference_answer": item.get("reference_answer"),
        "evidence_mode": EVIDENCE_MODE,
        "evidences": evidences,
        "retrieval_debug": retrieval_debug,
        "retrieval_top1_section": top1_section,
        "retrieval_top1_correct": retrieval_top1_correct,
        "must_include": item.get("must_include", []),
        "must_not_include": item.get("must_not_include", []),
    }


def build_summary(records, total):
    pass_records = [r for r in records if r.get("status") == "PASS"]
    partial_records = [r for r in records if r.get("status") == "PARTIAL"]
    fail_records = [r for r in records if r.get("status") == "FAIL"]
    error_records = [r for r in records if str(r.get("status", "")).endswith("ERROR") or r.get("status") in {
        "ORACLE_EVIDENCE_MISSING",
        "RETRIEVAL_EMPTY_ERROR",
        "GENERATION_ERROR",
        "JUDGE_TIMEOUT",
        "JUDGE_API_ERROR",
        "JUDGE_PARSE_ERROR",
    }]

    evaluated_count = len(pass_records) + len(partial_records) + len(fail_records)
    acceptable_count = len(pass_records) + len(partial_records)

    hallucination_count = sum(
        1
        for r in records
        if r.get("judge", {}).get("hallucination") is True
    )

    forbidden_violation_count = sum(
        1
        for r in records
        if r.get("forbidden_literal_items")
    )

    answerable_records = [r for r in records if r.get("answerable") is True]
    unanswerable_records = [r for r in records if r.get("answerable") is False]

    def verdict_counts(group):
        return {
            "total": len(group),
            "pass": sum(r.get("status") == "PASS" for r in group),
            "partial": sum(r.get("status") == "PARTIAL" for r in group),
            "fail": sum(r.get("status") == "FAIL" for r in group),
            "error": sum(r in error_records for r in group),
        }

    error_type_counts = {}
    for r in error_records:
        status = r.get("status", "UNKNOWN_ERROR")
        error_type_counts[status] = error_type_counts.get(status, 0) + 1

    retrieval_eval_records = [
        r
        for r in records
        if r.get("answerable") is True
        and r.get("retrieval_top1_correct") is not None
    ]
    retrieval_correct_count = sum(
        r.get("retrieval_top1_correct") is True
        for r in retrieval_eval_records
    )

    return {
        "total": total,
        "evaluated": evaluated_count,
        "errors": len(error_records),
        "pass": len(pass_records),
        "partial": len(partial_records),
        "fail": len(fail_records),
        "pass_rate_among_evaluated": safe_divide(len(pass_records), evaluated_count),
        "acceptable_rate_among_evaluated": safe_divide(acceptable_count, evaluated_count),
        "evaluation_coverage": safe_divide(evaluated_count, total),
        "hallucination_count": hallucination_count,
        "hallucination_rate_among_evaluated": safe_divide(
            hallucination_count,
            evaluated_count,
        ),
        "literal_forbidden_violations": forbidden_violation_count,
        "answerable": verdict_counts(answerable_records),
        "unanswerable": verdict_counts(unanswerable_records),
        "error_types": error_type_counts,
        "retrieval": {
            "evaluated_answerable_questions": len(retrieval_eval_records),
            "top1_correct": retrieval_correct_count,
            "top1_accuracy": safe_divide(
                retrieval_correct_count,
                len(retrieval_eval_records),
            ),
        },
    }


def save_checkpoint(records):
    payload = {
        "config": {
            "evidence_mode": EVIDENCE_MODE,
            "candidate_k": CANDIDATE_K,
            "reranker_model": RERANKER_MODEL_NAME,
            "judge_model": JUDGE_MODEL,
        },
        "records": records,
    }
    write_json(CHECKPOINT_PATH, payload)


# ============================================================
# Main Evaluation
# ============================================================

records = []
total = len(questions)

print("\n")
print_separator()
print("Generation Evaluation V2")
print_separator()
print("Evidence Mode:", EVIDENCE_MODE)
print("Judge Model:", JUDGE_MODEL)
print("Total:", total)

for index, item in enumerate(questions, start=1):
    question_id = item["id"]
    question = item["question"]
    answerable = item.get("answerable", True)
    expected_sections = item.get("expected_sections", [])
    reference_answer = item["reference_answer"]
    must_include = item.get("must_include", [])
    must_not_include = item.get("must_not_include", [])

    print("\n")
    print_separator()
    print(f"[{index}/{total}] {question_id}")
    print("Question:", question)
    print("Answerable:", answerable)
    print("Expected Sections:", expected_sections)

    evidences = []
    retrieval_debug = {}
    top1_section = None
    retrieval_top1_correct = None

    # --------------------------------------------------------
    # 1. Evidence
    # --------------------------------------------------------
    if EVIDENCE_MODE == "oracle":
        if answerable:
            evidences, missing_sections = get_oracle_evidences(expected_sections)

            # 理论上 preflight 已经保证不会发生；这里保留运行时防线。
            if missing_sections:
                record = make_base_record(
                    item=item,
                    evidences=evidences,
                    retrieval_debug={},
                    top1_section=None,
                    retrieval_top1_correct=None,
                )
                record.update(
                    {
                        "status": "ORACLE_EVIDENCE_MISSING",
                        "error": {
                            "type": "ORACLE_EVIDENCE_MISSING",
                            "missing_sections": missing_sections,
                        },
                    }
                )
                records.append(record)
                print("[ERROR] Missing Oracle Sections:", missing_sections)
                save_checkpoint(records)
                continue
        else:
            # Unanswerable / Out-of-KB：Oracle 模式下 Evidence 为空是正确状态。
            evidences = []

    else:  # reranker_top1
        evidences, retrieval_debug = get_reranker_top1_evidence(question)
        top1_section = retrieval_debug.get("top1_section")

        # 对 answerable=true 才计算“Top-1 是否命中 expected section”。
        # unanswerable 本来就没有正确 section，不能强行记成 retrieval failure。
        if answerable:
            retrieval_top1_correct = top1_section in expected_sections
        else:
            retrieval_top1_correct = None

        if not evidences:
            record = make_base_record(
                item=item,
                evidences=[],
                retrieval_debug=retrieval_debug,
                top1_section=None,
                retrieval_top1_correct=retrieval_top1_correct,
            )
            record.update(
                {
                    "status": "RETRIEVAL_EMPTY_ERROR",
                    "error": {
                        "type": "RETRIEVAL_EMPTY_ERROR",
                        "message": "Dense + Reranker 没有返回任何 Evidence。",
                    },
                }
            )
            records.append(record)
            print("[ERROR] Dense + Reranker returned no evidence.")
            save_checkpoint(records)
            continue

    print("\nEvidence:")
    if not evidences:
        print("[NO EVIDENCE]")
    else:
        for evidence in evidences:
            print("Section:", evidence["section"])
            print("Content:", evidence["content"])

    if EVIDENCE_MODE == "reranker_top1":
        print("\nReranker Top-1:", top1_section)
        print("Retrieval Correct:", retrieval_top1_correct)

    # --------------------------------------------------------
    # 2. Generate Answer
    # --------------------------------------------------------
    try:
        candidate_answer = generate_answer_with_retry(
            question=question,
            evidences=evidences,
        )
    except Exception as error:
        record = make_base_record(
            item=item,
            evidences=evidences,
            retrieval_debug=retrieval_debug,
            top1_section=top1_section,
            retrieval_top1_correct=retrieval_top1_correct,
        )
        record.update(
            {
                "status": "GENERATION_ERROR",
                "error": {
                    "type": "GENERATION_ERROR",
                    "message": repr(error),
                },
            }
        )
        records.append(record)
        print("\nGenerator ERROR:", repr(error))
        save_checkpoint(records)
        continue

    print("\nCandidate Answer:")
    print(candidate_answer)
    print("\nReference Answer:")
    print(reference_answer)

    # --------------------------------------------------------
    # 3. Deterministic Diagnostics
    # --------------------------------------------------------
    missing_items, forbidden_items = run_keyword_diagnostics(
        answer=candidate_answer,
        must_include=must_include,
        must_not_include=must_not_include,
    )

    # --------------------------------------------------------
    # 4. LLM Judge with Retry
    # --------------------------------------------------------
    try:
        judge_result = judge_answer(
            question=question,
            answerable=answerable,
            evidences=evidences,
            candidate_answer=candidate_answer,
            reference_answer=reference_answer,
            must_include=must_include,
            must_not_include=must_not_include,
        )
        status = judge_result["verdict"]
        error_info = None

    except Exception as error:
        status = classify_exception(error)
        error_info = {
            "type": status,
            "message": repr(error),
        }
        judge_result = {
            "verdict": "JUDGE_ERROR",
            "correctness": None,
            "groundedness": None,
            "completeness": None,
            "contradiction": None,
            "hallucination": None,
            "reason": repr(error),
            "raw_judge_output": None,
        }

    print("\nJudge Status:", status)
    print("Judge Verdict:", judge_result.get("verdict"))
    print("Correctness:", judge_result.get("correctness"))
    print("Groundedness:", judge_result.get("groundedness"))
    print("Completeness:", judge_result.get("completeness"))
    print("Hallucination:", judge_result.get("hallucination"))
    print("Reason:", judge_result.get("reason"))

    # --------------------------------------------------------
    # 5. Save Record
    # --------------------------------------------------------
    record = make_base_record(
        item=item,
        evidences=evidences,
        retrieval_debug=retrieval_debug,
        top1_section=top1_section,
        retrieval_top1_correct=retrieval_top1_correct,
    )
    record.update(
        {
            "status": status,
            "candidate_answer": candidate_answer,
            "missing_literal_items": missing_items,
            "forbidden_literal_items": forbidden_items,
            "judge": judge_result,
        }
    )

    if error_info is not None:
        record["error"] = error_info

    records.append(record)
    save_checkpoint(records)


# ============================================================
# Final Metrics
# ============================================================

summary = build_summary(records=records, total=total)

print("\n\n")
print_separator()
print("Generation Evaluation Result")
print_separator()
print("Evidence Mode:", EVIDENCE_MODE)
print("Judge Model:", JUDGE_MODEL)
print("Questions:", total)
print("Successfully Evaluated:", summary["evaluated"])
print("ERROR:", summary["errors"])

print("\nPASS:", summary["pass"])
print("PARTIAL:", summary["partial"])
print("FAIL:", summary["fail"])

print(
    f"\nPASS Rate (evaluated only): "
    f"{summary['pass_rate_among_evaluated']:.2%}"
)
print(
    f"PASS + PARTIAL Rate (evaluated only): "
    f"{summary['acceptable_rate_among_evaluated']:.2%}"
)
print(
    f"Evaluation Coverage: "
    f"{summary['evaluation_coverage']:.2%} "
    f"({summary['evaluated']}/{total})"
)

print("\nHallucination Cases:", summary["hallucination_count"])
print(
    f"Hallucination Rate (evaluated only): "
    f"{summary['hallucination_rate_among_evaluated']:.2%}"
)
print(
    "Literal Forbidden Violations:",
    summary["literal_forbidden_violations"],
)

print("\nAnswerable:")
print(json.dumps(summary["answerable"], ensure_ascii=False, indent=2))

print("\nUnanswerable / Abstention:")
print(json.dumps(summary["unanswerable"], ensure_ascii=False, indent=2))

if summary["error_types"]:
    print("\nInfrastructure Errors:")
    for error_type, count in sorted(summary["error_types"].items()):
        print(f"{error_type}: {count}")

if EVIDENCE_MODE == "reranker_top1":
    retrieval_summary = summary["retrieval"]
    print("\nRetrieval Top-1 (answerable only):")
    print(
        "Correct:",
        f"{retrieval_summary['top1_correct']}/"
        f"{retrieval_summary['evaluated_answerable_questions']}",
    )
    print(
        f"Accuracy: {retrieval_summary['top1_accuracy']:.2%}"
    )


# ============================================================
# Failure / Error Analysis
# ============================================================

fail_records = [r for r in records if r.get("status") == "FAIL"]
partial_records = [r for r in records if r.get("status") == "PARTIAL"]
error_records = [
    r
    for r in records
    if r.get("status") not in VALID_VERDICTS
]

print("\n\n")
print_separator()
print("Generation FAIL Analysis")
print_separator()

if not fail_records:
    print("\n没有真实 Generation FAIL。")
else:
    for case in fail_records:
        print("\n" + "-" * 120)
        print("ID:", case.get("id"))
        print("Question:", case.get("question"))
        print("Answerable:", case.get("answerable"))
        print("Expected Sections:", case.get("expected_sections"))
        print("Candidate Answer:", case.get("candidate_answer"))
        print("Reference Answer:", case.get("reference_answer"))
        print("Reason:", case.get("judge", {}).get("reason"))
        if EVIDENCE_MODE == "reranker_top1":
            print("Reranker Top-1:", case.get("retrieval_top1_section"))
            print("Retrieval Correct:", case.get("retrieval_top1_correct"))

print("\n\n")
print_separator()
print("Generation PARTIAL Analysis")
print_separator()

if not partial_records:
    print("\n没有 PARTIAL。")
else:
    for case in partial_records:
        print("\n" + "-" * 120)
        print("ID:", case.get("id"))
        print("Question:", case.get("question"))
        print("Candidate Answer:", case.get("candidate_answer"))
        print("Reference Answer:", case.get("reference_answer"))
        print("Reason:", case.get("judge", {}).get("reason"))

print("\n\n")
print_separator()
print("Infrastructure ERROR Analysis")
print_separator()

if not error_records:
    print("\n没有 Infrastructure ERROR。")
else:
    for case in error_records:
        print("\n" + "-" * 120)
        print("ID:", case.get("id"))
        print("Question:", case.get("question"))
        print("Status:", case.get("status"))
        print("Error:", case.get("error"))
        print("Candidate Answer:", case.get("candidate_answer"))


# ============================================================
# Save Final JSON
# ============================================================

output_data = {
    "config": {
        "evidence_mode": EVIDENCE_MODE,
        "candidate_k": CANDIDATE_K,
        "reranker_model": RERANKER_MODEL_NAME,
        "judge_model": JUDGE_MODEL,
        "judge_timeout_seconds": JUDGE_TIMEOUT_SECONDS,
        "judge_max_retries": JUDGE_MAX_RETRIES,
        "generator_max_retries": GENERATOR_MAX_RETRIES,
    },
    "summary": summary,
    "records": records,
}

write_json(OUTPUT_PATH, output_data)

# 正常完成后删除临时 checkpoint，避免下一轮误以为它是正式结果。
if CHECKPOINT_PATH.exists():
    try:
        CHECKPOINT_PATH.unlink()
    except OSError:
        pass

print("\n结果已经保存到：")
print(OUTPUT_PATH)
