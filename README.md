实验 1
Dense Retrieval
↓
小规模知识库 Hit@1 = 100%


实验 2
Cosine Threshold Answerability Gate
↓
Hard Negative 导致 Answerable / Unanswerable
score distribution 严重重叠


实验 3
1.5B / Qwen Plus / Qwen Max Evidence Judge
↓
大模型改善有限，仍存在 FN / FP


实验 4
NLI Verifier
↓
QA answer 与 NLI entailment 存在任务错配


实验 5
Cross-Encoder Reranker
↓
小规模 KB 上 Hit@1 保持 100%
Ranking changed 9
Regression 0
但存在 ceiling effect


