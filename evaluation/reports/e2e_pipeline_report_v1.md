# Full Pipeline E2E Evaluation Report V1


## Overview


本报告记录 Ecommerce RAG Assistant Full Pipeline E2E Regression V1 测试结果。


测试目标： 验证完整 RAG Pipeline 在真实用户问题下的：

- Retrieval
- Evidence Sufficiency Verification
- Routing
- Answer Generation / Abstention


## Evaluation Flow
Question
|
v
Retriever
|
v
Reranker
|
v
Evidence
|
v
Verifier
|
+----------------+
| |
SUFFICIENT INSUFFICIENT
| |
v v
Generator Abstain




## Dataset


Dataset:generation_questions_v1.json


Number of Cases:60

Ground Truth:evidence_sufficient



## Overall Result


| Metric | Value |
|-|-:|
| Total Cases | 60 |
| Successfully Evaluated | 60 |
| Errors | 0 |
| Coverage | 100% |



## Verifier Evaluation


Verifier evaluates whether retrieved evidence is sufficient.


Results:


| Metric | Value |
|-|-:|
| TP | 44 |
| TN | 15 |
| FP | 0 |
| FN | 1 |
| Accuracy | 98.33% |
| Precision | 100% |
| Recall | 97.78% |
| F1 | 98.88% |



## Pipeline Routing Evaluation


Pipeline routing verifies whether the system correctly follows Verifier decisions.


Results:


| Metric | Value |
|-|-:|
| Routing Correct | 59 / 60 |
| Routing Accuracy | 98.33% |
| Pipeline Consistency | 100% |


Pipeline Consistency:60 / 60



说明：

Pipeline 能够严格执行：
Verifier Decision
|
v
Routing Action


不存在 Pipeline 编排错误。



## Retrieval Diagnostic


Retrieval diagnostics evaluated cases:46


Result:


| Metric | Value |
|-|-:|
| Hit | 46 |
| Hit Rate | 100% |



## Failure Attribution


Only one case requires further analysis:


### b2_q50


Type:Verifier False Negative


Behavior:

Ground Truth:

evidence_sufficient = True

Verifier:INSUFFICIENT


Impact:系统选择 Abstain。


Classification:Verifier Error


Not:Pipeline Error


because:pipeline_consistency = True




## Conclusion


Full Pipeline E2E Regression V1 successfully passed.


Key conclusions:


1. Pipeline integration is stable.

2. Retrieval layer achieves complete hit rate on evaluated cases.

3. Verifier achieves 98.33% accuracy.

4. Pipeline routing consistency reaches 100%.

5. Remaining issue is isolated to Verifier boundary judgment.


The current version can serve as the baseline for future RAG model and system optimization.

