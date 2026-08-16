# Evidence Judge Model Scaling

Development set: 45 queries
Retriever: BAAI/bge-small-zh-v1.5 + FAISS
Context: Top-1
Judge prompt: Evidence Judge V1

| Model | TP | TN | FP | FN | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen2.5-1.5B-Instruct | 22 | 16 | 2 | 5 | 84.44% | 91.67% | 81.48% | 86.27% |
| qwen-plus-2025-07-28 | 24 | 14 | 4 | 3 | 84.44% | 85.71% | 88.89% | 87.27% |
| qwen-max | 21 | 18 | 0 | 6 | 86.67% | 100.00% | 77.78% | 87.50% |


## Conclusion

Increasing judge model capability did not fundamentally solve
the evidence sufficiency classification problem.

- Qwen Plus achieved the highest recall, but increased false positives.
- Qwen Max eliminated false positives on the development set,
  but became overly conservative and increased false negatives.
- F1 improved only marginally from 86.27% to 87.50%.

This suggests that model scale is not the primary bottleneck.
The main limitation is the formulation of answerability as a
single binary evidence-sufficiency classification task.