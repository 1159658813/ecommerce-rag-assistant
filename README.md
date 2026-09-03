# Ecommerce RAG Assistant

面向中文电商客服场景的工程化 RAG（Retrieval-Augmented Generation）系统。

项目目标不是实现一个简单的“知识库问答 Demo”，而是构建一套具备：

- 文档结构化切分
- Dense Retrieval
- Cross-Encoder Reranking
- Evidence Sufficiency Verification
- Evidence 不足时主动拒答
- Grounded Answer Generation
- FastAPI 服务接口
- 请求追踪与结构化日志
- 分层 Evaluation
- Full Pipeline Regression

能力的完整 RAG 系统。

---

# 1. System Architecture

当前正式链路：

```text
User Question
      |
      v
Dense Retrieval
      |
      | Top-10 Candidates
      v
BGE Cross-Encoder Reranker
      |
      | Top-3 Evidence
      v
Answerability Verifier
      |
      +---------------------------+
      |                           |
  SUFFICIENT                 INSUFFICIENT
      |                           |
      v                           v
Answer Generator               Abstain
      |
      v
Final Answer
```

服务化后的整体结构：

```text
Client
  |
  v
FastAPI
  |
  v
RAGService
  |
  v
RAGPipeline
  |
  +-- Retriever
  |     |
  |     +-- Dense Retrieval
  |     |
  |     +-- BGE Reranker
  |
  +-- Answerability Verifier
  |
  +-- Answer Generator
```

---

# 2. Core Design

## 2.1 Dense Retrieval

Embedding Model:

```text
BAAI/bge-small-zh-v1.5
```

Vector Store:

```text
FAISS
```

默认召回：

```text
candidate_k = 10
```

索引文件：

```text
data/index/knowledge.faiss
data/index/chunks.json
```

---

## 2.2 Cross-Encoder Reranker

Reranker Model:

```text
BAAI/bge-reranker-v2-m3
```

Dense Retriever 首先召回候选文档，然后使用 Cross-Encoder 对：

```text
Query + Candidate Document
```

进行联合相关性打分并重新排序。

正式 Pipeline 默认保留：

```text
Top-3 Evidence
```

---

## 2.3 Evidence Sufficiency Verification

Verifier 的任务不是判断：

```text
“模型能不能猜出答案？”
```

而是判断：

```text
“当前实际检索得到的 Evidence
是否已经充分支持回答？”
```

这是本项目区别于普通 RAG Demo 的核心设计之一。

### KB Answerable

```text
answerable
```

表示：

> 整个知识库中是否存在该问题所需的信息。

### Evidence Sufficient

```text
evidence_sufficient
```

表示：

> 当前 Retrieval + Reranker 返回的 Top-K Evidence 是否足够回答。

两者不能混淆。

例如：

```text
KB 中存在答案
        |
        v
answerable = true

但本次 Top-3 没有检索完整
        |
        v
evidence_sufficient = false
        |
        v
Pipeline Abstain
```

---

## 2.4 Grounded Generation

Generator 只在：

```text
Verifier = SUFFICIENT
```

时执行。

生成原则：

```text
Evidence
   |
   v
Answer Generator
```

模型只能依据当前 Evidence 回答，不应自由补充知识库外事实。

Evidence 不足时由 Verifier Gate 提前阻断生成。

---

# 3. Models

当前主要模型配置：

| Component | Model |
|---|---|
| Embedding | `BAAI/bge-small-zh-v1.5` |
| Reranker | `BAAI/bge-reranker-v2-m3` |
| Development Verifier | `qwen3.6-plus` |
| Reference Verifier Baseline | `qwen-max` |
| Answer Generator | `qwen-plus-2025-07-28` |

LLM API 当前通过阿里云百炼 DashScope 的 OpenAI-compatible API 调用。

模型与 Pipeline 参数统一由：

```text
src/config/
```

管理。

---

# 4. Project Structure

```text
ecommerce_rag_assistant/
|
├── app.py
|
├── data/
│   ├── raw/
│   └── index/
│       ├── knowledge.faiss
│       └── chunks.json
|
├── src/
│   ├── ingestion/
│   │   ├── document_loader.py
│   │   └── markdown_splitter.py
│   │
│   ├── retrieval/
│   │   ├── embedding_model.py
│   │   ├── vector_store.py
│   │   ├── retriever.py
│   │   ├── reranker.py
│   │   └── two_stage_retriever.py
│   │
│   ├── verification/
│   │   └── ...
│   │
│   ├── generation/
│   │   ├── answer_generator.py
│   │   └── prompt.py
│   │
│   ├── pipeline/
│   │   └── rag_pipeline.py
│   │
│   ├── service/
│   │   ├── factory.py
│   │   └── rag_service.py
│   │
│   ├── api/
│   │   ├── app.py
│   │   └── schemas.py
│   │
│   ├── config/
│   │   └── settings.py
│   │
│   └── observability/
│       └── logging_config.py
|
├── evaluation/
│   ├── datasets/
│   ├── retrieval/
│   ├── generation/
│   ├── answerability/
│   ├── e2e/
│   ├── reports/
│   └── results/
|
├── experiments/
│   ├── pipeline_smoke_test.py
│   └── api_smoke_test.py
|
├── scripts/
│   └── build_index.py
|
├── .env.example
├── .gitignore
└── README.md
```

---

# 5. Layer Responsibilities

项目采用明确的模块边界。

## ingestion

负责：

```text
Raw Documents
      |
      v
Structured Chunks
```

不负责 Retrieval 或 Generation。

---

## retrieval

负责：

```text
Question
   |
   v
Dense Retrieval
   |
   v
Reranker
   |
   v
Top-K Evidence
```

---

## verification

负责：

```text
Question + Evidence
        |
        v
SUFFICIENT / INSUFFICIENT
```

不负责生成最终答案。

---

## generation

负责：

```text
Question + Sufficient Evidence
              |
              v
          Final Answer
```

---

## pipeline

负责模块编排：

```text
Retriever
    |
Verifier
    |
Generator / Abstain
```

Pipeline 不重新实现 Retrieval、Reranking 或业务规则。

---

## service

负责向上层提供稳定的 RAG 服务接口，并隔离 API 与底层 Pipeline 实现。

---

## api

负责：

- HTTP Request / Response
- Input Validation
- Error Mapping
- Request ID
- API Lifecycle

不承载核心 RAG 算法。

---

## observability

负责：

- Structured Logging
- Request Tracing
- Request ID
- Latency Logging
- Service Error Logging

---

# 6. Environment

当前主要开发环境：

```text
Python 3.11
Conda
Windows
PyCharm
```

推荐创建独立环境：

```powershell
conda create -n rag-shop python=3.11
conda activate rag-shop
```

> 当前项目依赖清单将在 Project Delivery 阶段进一步标准化。

---

# 7. Environment Variables

复制：

```text
.env.example
```

并创建：

```text
.env
```

`.env` 不应提交到 Git。

核心配置包括：

```env
DASHSCOPE_API_KEY=
DASHSCOPE_BASE_URL=

ANSWERABILITY_VERIFIER_MODEL=qwen3.6-plus

RAG_CANDIDATE_K=10
RAG_EVIDENCE_K=3

LOG_LEVEL=INFO
```

具体支持的配置项以：

```text
src/config/settings.py
```

和：

```text
.env.example
```

为准。

---

# 8. Build Knowledge Index

知识库原始文档位于：

```text
data/raw/
```

构建索引：

```powershell
python scripts/build_index.py
```

生成：

```text
data/index/knowledge.faiss
data/index/chunks.json
```

---

# 9. Run the RAG Application

CLI / application entry point：

```powershell
python app.py
```

---

# 10. Run FastAPI

启动 API：

```powershell
uvicorn src.api:app --host 127.0.0.1 --port 8000
```

启动后访问：

```text
http://127.0.0.1:8000/docs
```

即可打开 Swagger API 文档。

Health Check：

```text
GET /health
```

返回：

```json
{
  "status": "ok"
}
```

---

# 11. Query API

Endpoint:

```text
POST /api/v1/query
```

Request:

```json
{
  "question": "银行卡退款多久能到账？"
}
```

典型 Response：

```json
{
  "question": "银行卡退款多久能到账？",
  "answer": "银行卡退款通常需要3至7个工作日到账。",
  "abstained": false,
  "abstain_reason": null,
  "verdict": "SUFFICIENT",
  "evidences": [
    {
      "rank": 1,
      "source": "refund_policy.md",
      "section": "退款到账时间",
      "content": "...",
      "reranker_score": 2.95703125
    }
  ]
}
```

当 Evidence 不足时：

```json
{
  "question": "...",
  "answer": "根据当前知识库信息，暂时无法确认。",
  "abstained": true,
  "abstain_reason": "evidence_insufficient",
  "verdict": "INSUFFICIENT",
  "evidences": []
}
```

---

# 12. API Error Handling

API 已对主要错误场景进行统一处理。

包括：

```text
400  Invalid Request
422  Request Validation Error
500  Internal / Service Error
```

同时为请求生成：

```text
X-Request-ID
```

用于日志链路追踪。

---

# 13. Observability

服务层支持结构化日志。

典型日志：

```text
request_started
request_id=...
method=POST
path=/api/v1/query
```

```text
rag_completed
request_id=...
verdict=SUFFICIENT
abstained=False
evidence_count=3
```

```text
request_completed
request_id=...
status=200
latency_ms=...
```

发生异常时：

```text
rag_failed
request_id=...
```

可以通过同一个 `request_id` 追踪一次完整 API 请求。

---

# 14. Evaluation Framework

项目 Evaluation 采用分层测试。

```text
Retrieval Evaluation
        |
Reranker Evaluation
        |
Answerability Evaluation
        |
Generation Evaluation
        |
Full Pipeline E2E Regression
```

这样可以避免出现：

```text
最终答案错了
但不知道是 Retrieval、
Verifier 还是 Generator 的问题
```

---

# 15. Current E2E Baseline

Full Pipeline E2E Regression V1：

| Metric | Result |
|---|---:|
| Total Cases | 60 |
| Successfully Evaluated | 60 |
| Infrastructure Errors | 0 |
| Evaluation Coverage | 100% |
| Retrieval Hit Rate | 100% |
| Verifier Accuracy | 98.33% |
| Verifier Precision | 100% |
| Verifier Recall | 97.78% |
| Verifier F1 | 98.88% |
| Pipeline Routing Consistency | 100% |

Verifier Confusion Matrix：

```text
TP = 44
TN = 15
FP = 0
FN = 1
```

当前 Full Pipeline Regression 中唯一错误归因为：

```text
Verifier False Negative
```

而不是 Pipeline Routing Error。

---

# 16. Layer Attribution

Full Pipeline Evaluation 不只输出：

```text
PASS / FAIL
```

而是尝试定位失败层。

例如：

```text
Retrieval Failure
        |
Verifier Failure
        |
Pipeline Routing Failure
        |
Generation Failure
```

其中：

```text
Pipeline Routing Consistency = 100%
```

表示 Pipeline 能够严格执行 Verifier 的：

```text
SUFFICIENT
```

或：

```text
INSUFFICIENT
```

判断。

---

# 17. Run Smoke Tests

Pipeline Smoke Test：

```powershell
python experiments/pipeline_smoke_test.py
```

API Smoke Test：

```powershell
python experiments/api_smoke_test.py
```

API Smoke Test 当前覆盖：

```text
GET /health

POST /api/v1/query
    ├── answerable
    ├── abstention
    ├── service error
    ├── validation error
    └── invalid request
```

---

# 18. Run Full Pipeline Regression

```powershell
python evaluation/e2e/evaluate_pipeline.py
```

测试链路：

```text
Question
   |
Dense Retrieval
   |
BGE Reranker
   |
Top-3 Evidence
   |
Verifier
   |
Answer / Abstain
```

评测结果用于 Regression，而不是针对单个失败 Case 反复修改 Prompt。

---

# 19. Engineering Principles

本项目开发遵循以下原则。

### 1. Git 管历史，目录管职责

正式代码不通过：

```text
xxx_v1.py
xxx_final.py
xxx_final2.py
```

保存历史版本。

历史通过：

```text
Commit
Branch
Tag
Evaluation Results
```

管理。

### 2. Pipeline 只负责编排

不在 Pipeline 中重新实现 Retrieval、Reranking 或业务规则。

### 3. Evidence Sufficiency 优先于模型推测

```text
LLM 能推断答案
```

不代表：

```text
Evidence 已充分覆盖答案
```

### 4. Regression Dataset 不作为调参集

发现 Regression Failure 后首先进行：

```text
Layer Attribution
```

而不是针对该 Case 修改 Prompt。

### 5. Product Path = Evaluation Path

Full E2E Evaluation 调用正式产品 Pipeline，而不是复制另一套 RAG 实现。

---

# 20. Development Milestones

项目目前已经完成：

- [x] Structured Markdown Ingestion
- [x] BGE Embedding
- [x] FAISS Vector Retrieval
- [x] Cross-Encoder Reranking
- [x] Top-K Evidence Construction
- [x] Evidence Sufficiency Verification
- [x] Abstention Gate
- [x] Grounded Answer Generation
- [x] Formal RAGPipeline
- [x] Retrieval Public API
- [x] Application Entry Point
- [x] Full Pipeline E2E Regression
- [x] RAG Service Layer
- [x] FastAPI Service
- [x] Centralized Configuration
- [x] Standardized API Errors
- [x] Request ID / Request Tracing
- [x] Structured Logging
- [x] API Smoke Tests
- [ ] Reproducible Dependency Manifest
- [ ] Final Project Delivery Documentation

---

# 21. Current Stage

Current branch:

```text
feature/project-delivery-v1
```

Current stage:

```text
Project Delivery / Reproducibility
```

这一阶段主要完成：

- dependency specification
- environment reproduction
- documentation cleanup
- final acceptance checks
- release baseline

---

# 22. Security

请勿提交：

```text
.env
DASHSCOPE_API_KEY
其他私有 API Key
```

真实密钥仅保存在本地环境中。

仓库只提供：

```text
.env.example
```

作为配置模板。