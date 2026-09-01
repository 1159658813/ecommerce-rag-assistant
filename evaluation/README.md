# Evaluation Framework

本目录用于评估 Ecommerce RAG Assistant 的检索、证据判断、答案生成以及完整 Pipeline 性能。

Evaluation 采用分层设计，将 RAG 系统拆分为多个可独立分析的模块，同时提供 Full Pipeline E2E Regression 验证真实用户请求流程。


## Evaluation Architecture

Question
|
v
Retriever
|
v
Reranker
|
v
Evidence Set
|
v
Answerability Verifier
|
+----------------+
| |
SUFFICIENT INSUFFICIENT
| |
v v
Generator Abstain
|
v
Final Answer



## Evaluation Modules


## 1. Retrieval Evaluation

目录：
evaluation/retrieval/


目标：

验证检索系统是否能够召回包含答案的知识片段。


主要指标：

- Recall
- Hit Rate
- Top-K Retrieval Accuracy


相关文件：
evaluate_retrieval.py
evaluate_reranker.py



---


## 2. Answerability / Evidence Sufficiency Evaluation


目录：
evaluation/answerability/


目标：

判断当前检索 Evidence 是否足够支持回答。


核心概念：
answerable
表示知识库中是否存在答案。

evidence_sufficient

表示当前 Retrieval + Reranker 输出的 Evidence 是否足够回答。


两者不能混淆。


主要指标：

- Accuracy
- Precision
- Recall
- F1


---


## 3. Generation Evaluation


目录：
evaluation/generation/


目标：

评估最终生成答案质量。


评估维度：

- Correctness
- Groundedness
- Completeness
- Hallucination


Generation Evaluation 使用：

- Reference Answer
- Evidence Context
- Must Include
- Must Not Include


进行约束评估。


---


## 4. Full Pipeline E2E Evaluation


目录：
evaluation/e2e/


目标：

验证真实产品链路：
Question

↓

Retriever

↓

Reranker

↓

Verifier

↓

Generator / Abstain


与模块 Evaluation 不同：

E2E Evaluation 不单独调用 Retriever 或 Generator，而直接调用正式 RAGPipeline。


这样可以保证：
Evaluation Pipeline

Production Pipeline


避免测试逻辑与线上逻辑不一致。


## Current Baseline


Version:
Full Pipeline E2E Evaluation V1


Dataset:
generation_questions_v1.json


Cases:60


Ground Truth:evidence_sufficient


Results:


| Metric | Result |
|---|---:|
| Evaluation Coverage | 100% |
| Pipeline Errors | 0 |
| Verifier Accuracy | 98.33% |
| Verifier F1 | 98.88% |
| Routing Consistency | 100% |
| Retrieval Hit Rate | 100% |


## Error Attribution


E2E Evaluation 支持 Layer Attribution。


错误不会简单标记为 FAIL，而会定位到具体模块：
Retrieval Failure

Verifier Failure

Pipeline Routing Failure

Generation Failure


当前 Baseline:

Verifier False Negative:

b2_q50


说明：

- Evidence 实际充分
- Verifier 判断不足
- Pipeline 正确执行 Verifier 决策


## Evaluation Workflow


运行完整 E2E Regression:


```bash
python evaluation/e2e/evaluate_pipeline.py
结果保存：
evaluation/results/e2e/

Development Principle

Evaluation 遵循以下原则：

测试必须调用真实 Pipeline。
Evaluation 不复制业务逻辑。
每一次失败必须能够定位责任模块。
Regression Dataset 保持稳定，用于模型和系统迭代比较。
