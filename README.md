# Ecommerce RAG Assistant

面向中文电商客服场景的工程化 RAG（Retrieval-Augmented Generation）系统。

项目核心不是“尽量回答”，而是：

> **只有当当前检索证据足够支持答案时才生成回答；证据不足时主动拒答。**

系统包含 Dense Retrieval、FAISS、Cross-Encoder Reranking、Evidence Sufficiency Verification、Grounded Generation、Abstention、FastAPI、结构化日志、Evaluation，以及 Vue 2 + Element UI 前端。

---

## 前端预览

![Ecommerce RAG Assistant Frontend](docs/images/frontend-demo.png)

---

# 快速开始

下面的步骤适合第一次拿到项目的用户。

## 1. 克隆项目

```powershell
git clone https://github.com/1159658813/ecommerce-rag-assistant.git
cd ecommerce-rag-assistant
```

## 2. 创建 Python 环境

推荐使用 Python 3.11。

```powershell
conda create -n rag-shop python=3.11
conda activate rag-shop
```

## 3. 安装依赖

```powershell
python -m pip install -r requirements.txt
```

## 4. 配置 API Key

复制环境变量模板：

```powershell
Copy-Item .env.example .env
```

打开 `.env`，至少配置：

```env
DASHSCOPE_API_KEY=你的_DashScope_API_Key
```

> `.env` 包含私密信息，请勿提交到 Git。

`HF_TOKEN` 当前为可选项；未配置时通常只会影响 Hugging Face 模型下载限流。

## 5. 检查项目环境

```powershell
python scripts/verify_project.py --mode quick
```

看到：

```text
ALL REQUESTED CHECKS PASSED
```

说明基础环境、配置、索引和代码检查通过。

## 6. 启动 Web 应用

在项目根目录执行：

```powershell
python -m uvicorn src.api.app:app --host 127.0.0.1 --port 8000
```

首次启动需要加载 Embedding / Reranker 等模型，可能需要等待一段时间。

看到：

```text
RAGService ready.
```

说明服务已经就绪。

## 7. 打开前端页面

浏览器访问：

```text
http://127.0.0.1:8000/
```

即可使用 Ecommerce RAG Assistant。

其他入口：

```text
Swagger API 文档
http://127.0.0.1:8000/docs

Health Check
http://127.0.0.1:8000/health
```

> 前端使用 **Vue 2 + Element UI**，由 FastAPI 直接托管。  
> 本地运行时 **不需要单独安装 Node.js，也不需要单独启动前端服务**。

---

# 系统架构

```text
User Question
      |
      v
Dense Retrieval
      |
      v
Top-K Candidates
      |
      v
Cross-Encoder Reranker
      |
      v
Top Evidence
      |
      v
Evidence Verifier
   /       \\
  /         \\
SUFFICIENT  INSUFFICIENT
   |             |
   v             v
Generator      Abstain
   |
   v
Final Answer
```

服务层：

```text
Frontend
   |
FastAPI
   |
RAGService
   |
RAGPipeline
```

核心设计是 Retrieval 与 Generation 之间存在独立的 **Evidence Verification Gate**。

---

# 主要功能

- Markdown 文档结构化切分
- BGE Dense Embedding
- FAISS 向量检索
- Cross-Encoder Reranking
- Two-stage Retrieval
- Evidence Sufficiency Verification
- Evidence 不足时主动 Abstain
- Evidence-grounded Answer Generation
- FastAPI API
- Vue 2 + Element UI Web UI
- Evidence / Verdict 可视化
- Request ID 与结构化日志
- API / Pipeline Smoke Test
- 分层 Evaluation 与 E2E Regression

---

# 前端功能

Web 页面目前支持：

- 用户与 AI 对话
- `SUFFICIENT / INSUFFICIENT` 状态展示
- Evidence 展开查看
- Reranker Score 展示
- Evidence 不足时展示拒答提示
- 请求耗时与 Request Tracing 信息
- RAG Service 健康状态
- 清空当前前端会话

当前聊天接口是 **Single-turn RAG**：

```text
每次请求只发送当前 question
```

前端显示的历史消息仅用于界面展示，不会作为多轮上下文发送给模型。

---

# 主要模型

| 组件 | 模型 |
|---|---|
| Embedding | `BAAI/bge-small-zh-v1.5` |
| Reranker | `BAAI/bge-reranker-v2-m3` |
| Verifier | `qwen3.6-plus` |
| Answer Generator | `qwen-plus-2025-07-28` |

LLM 通过阿里云 DashScope 的 OpenAI-compatible API 调用。

默认 Retrieval 参数：

```text
candidate_k = 10
evidence_k = 3
```

---

# API

## Health Check

```http
GET /health
```

## RAG Query

```http
POST /api/v1/query
```

Request：

```json
{
  "question": "银行卡退款多久能到账？"
}
```

Response 主要字段：

```json
{
  "question": "...",
  "answer": "...",
  "abstained": false,
  "abstain_reason": null,
  "verdict": "SUFFICIENT",
  "evidences": []
}
```

Evidence 中包含：

```text
rank
source
section
content
reranker_score
```

当当前 Evidence 不足时，系统返回：

```text
verdict = INSUFFICIENT
abstained = true
```

并停止无依据生成。

---

# Evaluation Baseline

当前 Full Pipeline E2E Regression V1：

| Metric | Result |
|---|---:|
| Total Cases | 60 |
| Retrieval Hit Rate | 100% |
| Verifier Accuracy | 98.33% |
| Verifier Precision | 100% |
| Verifier Recall | 97.78% |
| Verifier F1 | 98.88% |
| Pipeline Routing Consistency | 100% |

当前 baseline 的主要价值是验证：

```text
Retrieval
→ Verification
→ Generation / Abstention
```

整条正式产品链路能够稳定工作。

---

# 项目结构

```text
ecommerce-rag-assistant/
├── frontend/
│   ├── index.html
│   ├── app.js
│   └── styles.css
│
├── data/
│   ├── raw/
│   └── index/
│       ├── knowledge.faiss
│       └── chunks.json
│
├── src/
│   ├── ingestion/
│   ├── retrieval/
│   ├── verification/
│   ├── generation/
│   ├── pipeline/
│   ├── service/
│   ├── api/
│   ├── config/
│   └── observability/
│
├── evaluation/
├── experiments/
├── scripts/
├── requirements.txt
├── .env.example
└── README.md
```

---

# 项目验证

基础检查：

```powershell
python scripts/verify_project.py --mode quick
```

API 检查：

```powershell
python scripts/verify_project.py --mode api
```

完整检查：

```powershell
python scripts/verify_project.py --mode full
```

`full` 会运行：

```text
Project Check
API Smoke Test
Pipeline Smoke Test
```

完整测试需要真实加载模型和调用 LLM API，因此耗时会明显高于 `quick`。

---

# 重新构建知识库索引

仓库当前正式 Pipeline 使用：

```text
data/index/knowledge.faiss
data/index/chunks.json
```

如果你修改了 `data/raw/` 中的知识库内容，需要重新构建索引：

```powershell
python scripts/build_index.py
```

---

# 常见问题

### 1. 启动后需要等待很久？

首次运行可能需要下载并加载 Hugging Face 模型，同时初始化 RAGService，这是正常现象。

### 2. 浏览器打不开前端？

先确认终端已经出现：

```text
RAGService ready.
```

然后访问：

```text
http://127.0.0.1:8000/
```

### 3. 提示没有 `DASHSCOPE_API_KEY`？

检查项目根目录是否存在 `.env`，并确认：

```env
DASHSCOPE_API_KEY=...
```

已经正确填写。

### 4. 为什么系统有时拒绝回答？

这是项目的预期行为。

系统不会为了提高回答率强制生成内容。只有当前检索 Evidence 足够支持答案时，Verifier 才允许进入 Generation。

---

# 当前版本

正式工程基线：

```text
v1.0.0
```

当前前端里程碑：

```text
frontend-mvp-v1
```

项目当前已经完成：

```text
Retrieval
+ Reranking
+ Evidence Verification
+ Abstention
+ Generation
+ Service Layer
+ FastAPI
+ Observability
+ Evaluation
+ Vue 2 Frontend
```

---

# Security

请勿提交：

```text
.env
DASHSCOPE_API_KEY
其他私有 API Key
```

仓库只提供 `.env.example` 作为配置模板。
