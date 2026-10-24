# P2-B 性能探索性采样记录（Pilot）

> 项目：`ecommerce-rag-assistant`  
> 日期：2026-10-21  
> 开发分支：`feature/p2-performance-baseline`  
> 基线：P2-A 已合并的 `main`（合并提交 `974407f`）  
> 状态：**探索性采样；不是正式性能基线或优化结论**

## 1. 目的与范围

本轮 P2-B 先验证性能采集链路能否工作，并初步定位值得调查的耗时阶段：

- **客户端耗时**：通过 Python `time.perf_counter()` 测量一次 HTTP 请求从发送到收到响应的耗时。
- **服务端耗时**：读取应用日志中的 `customer_support_trace`，获取总耗时和 Router、Retrieval、Verifier、Tool、Composer 等阶段指标（单位：毫秒）。
- **安全与成本约束**：串行、低次数调用真实 API；不运行完整 60 条付费 E2E；不因性能问题绕过 Evidence Gate。

测试通过 Windows 上的 Python 脚本请求本机 Docker Compose 运行的服务（`http://127.0.0.1:8000`）。当前容器已在 P2-A 验证过健康检查、知识库问答、订单查询及缓存持久化。**本报告只记录 P2-B 已实际采集的两条请求。**

## 2. 工具与采集方法

| 工具 / 数据来源 | 用途 |
| --- | --- |
| `scripts/p2_perf_baseline.py` | 串行发送 HTTP 请求并写入客户端基线 JSONL |
| `scripts/p2_trace_summary.py` | 解析 Docker 日志中的 `customer_support_trace` |
| `scripts/p2_perf_trace_pilot.py` | 单请求隔离窗口采集 HTTP 耗时与 Trace，输出配对 JSONL |
| `src/customer_support/service.py` | 产生 `customer_support_trace`；`latency_total` 使用 `time.perf_counter()` |
| Docker Compose `rag-app` 日志 | 服务端阶段耗时来源 |

日志字段中的 `latency_router` 包括 `intent_router.route(resolved_question)` 调用及异常回退路径的测量区间。`latency_retrieval`、`latency_verifier`、`latency_composer` 等来自下游组件观测字段。不能假设所有阶段时间互不重叠，也不能用它们的简单相加替代总耗时。

**关联可靠性限制：**

- `order` 样本的服务端 Trace 是通过单请求后人工核对日志得到的，没有 Request ID 级精确关联。
- `knowledge` 样本的配对标记为 `unique_in_isolated_window_not_id_verified`：隔离时间窗口内仅命中一条 Trace，但**仍未通过 Request ID 验证**。
- 采样时不应同时在前端发起其他业务请求；未来如开展并发或批量测试，应先补齐可靠的请求级关联。

## 3. 原始样本摘要

两种场景各 **1 个成功请求**，没有预热与正式样本分组，也没有同条件重复采样。

| 指标 | ORDER_QUERY（订单查询） | KNOWLEDGE_QA（知识库问答） |
| --- | ---: | ---: |
| 请求结果 | HTTP 200 | HTTP 200 |
| 意图 / Handler | `ORDER_QUERY` / `ORDER_AGENT` | `KNOWLEDGE_QA` / `RAG` |
| 客户端端到端延迟 | **7,890.230 ms** | **64,383.480 ms** |
| 服务端 `latency_total` | 7,875.686 ms | 64,367.605 ms |
| `latency_router` | 3,788.912 ms | 3,061.606 ms |
| `latency_retrieval` | 0 ms | 2,388.066 ms |
| `latency_verifier` | 0 ms | **38,265.745 ms** |
| `latency_tool` | 7.734 ms | 0 ms |
| `latency_composer` | 1,450.176 ms | **18,020.214 ms** |
| `verifier_retry_count` | 0 | 0 |
| 客户端与服务端差值 | 14.544 ms | 15.875 ms |
| 阶段求和后的算术差值（非独立计时项） | 2,628.864 ms | 2,631.974 ms |

### 3.1 ORDER_QUERY：一次订单查询

- 客户端耗时约 **7.89 秒**；服务端耗时约 **7.88 秒**。
- Router 约 **3.79 秒**，Composer 约 **1.45 秒**；订单工具仅约 **7.7 毫秒**。
- 该请求未走 RAG Retrieval / Verifier 阶段（对应值为零）。
- **初步线索**：优先排查 Router 与其余处理时间；本次数据不支持将 SQLite 查询认定为主要瓶颈。

### 3.2 KNOWLEDGE_QA：一次知识库问答

- 客户端耗时约 **64.38 秒**；服务端耗时约 **64.37 秒**。
- Verifier **38.27 秒**，约占服务端总耗时 **59.45%**，是本次最大的已记录阶段。
- Composer **18.02 秒**，约占 **28.00%**；Router **3.06 秒**，约占 **4.76%**；Retrieval **2.39 秒**，约占 **3.71%**。
- `verifier_retry_count=0`：**未记录到应用层 Verifier 重试**；但不能据此排除 SDK、网络或服务提供方内部的重试与等待。
- **初步线索**：优先调查 Verifier 实际使用的实现版本、模型、Prompt 长度、生成上限与外部 API 延迟，其次调查 Composer。

## 4. 指标解释与限制

1. **样本量不足**：各场景仅 1 次成功请求，不报告 P50、P95、吞吐量、稳定错误率或统计显著性，也不将单次值称为场景平均性能。
2. **Trace 不是请求 ID 精确关联**：目前仅支持人工核对或单请求隔离窗口配对，不能用于并发负载下的一一对应分析。
3. **阶段值不构成完整独立分解**：服务总耗时与已列阶段之和的差值仅称为“未归因算术差值”，可能涉及未单独计时步骤或时间重叠，不等于某个真实组件的测量结果。
4. **HTTP 200 不代表回答质量已达标**：性能样本需另行关联意图、回答正确性、证据充分性等质量指标；本次没有对两条性能请求做完整的答案正确性复核。
5. **外部依赖影响较大**：DashScope / 网络状态 / 模型缓存可能影响耗时。缺少 Token 使用量、调用次数、服务提供方耗时等数据，不能认定长延迟完全由本地 Python 代码造成。
6. **尚无优化前后对照**：本报告是诊断性 Pilot，不宣称任何性能提升。

## 5. Verifier 源码定位：当前进展

只读代码搜索已发现多个实现文件：

- `src/verification/answerability_verifier_v1.py`：`AnswerabilityVerifier`
- `src/verification/answerability_verifier_v1_1.py`：`AnswerabilityVerifierV11`
- `src/verification/answerability_verifier_v1_2.py`：`AnswerabilityVerifierV12`，另有 `VerifierExhausted`
- `src/pipeline/rag_pipeline.py`：组装 `latency_verifier` 和 `verifier_retry_count`

**尚未确认当前 Docker 实际实例化的 Verifier 版本、模型参数、单次外部调用次数与输入 Token 数。** 不应仅凭文件名推断线上所用版本。

## 6. 后续计划（尚未执行）

1. **确认实际调用链**：沿 `rag_pipeline.py` 查看 Verifier 的创建与调用，明确当前生效的实现、模型、超时、重试、Prompt 构造和 Token 上限。
2. **补足诊断观测**：优先复用已有日志；如确有缺口，再以最小修改增加请求级关联与必要的调用计时，避免记录问题原文、API Key 和敏感数据。
3. **固定实验条件**：同一代码版本、Docker 配置、模型、问题、用户条件与串行负载；明确预热、超时、API 费用和采样次数。
4. **采集正式基线**：以控制成本为前提完成足够的重复测量，并如实说明 P95 对样本量的要求和统计不确定性。
5. **单点优化与回归**：优先调查 Verifier，不跳过证据验证；确保优化前后对比同时覆盖延迟、正确性、拒答行为和故障率。
6. **形成最终报告**：另建正式的 P2-B benchmark 报告，不使用当前探索性结果冒充正式基线。

## 7. 临时数据保存与清理约定

本轮使用过的 Windows 临时文件（实际存在与否以本机检查为准）：

```text
%TEMP%\p2a_staged_diff.txt
%TEMP%\ecommerce_rag_p2\baseline.jsonl
%TEMP%\ecommerce_rag_p2\paired_pilot.jsonl
```

**当前状态：暂不删除 P2-B 的两份 JSONL。** 先确保本 Markdown 文件已保存到 `docs/p2b_pilot_results.md`，并完成需要的后续分析。如果原始记录含会话信息或业务内容，不要将 JSONL 提交至 Git。

在确认不再需要原始记录之后，仅清理以上明确命名的文件，不清空整个 `%TEMP%`，也不执行 Docker 缓存或卷清理：

```powershell
$files = @(
    (Join-Path $env:TEMP 'p2a_staged_diff.txt'),
    (Join-Path $env:TEMP 'ecommerce_rag_p2\baseline.jsonl'),
    (Join-Path $env:TEMP 'ecommerce_rag_p2\paired_pilot.jsonl')
)
foreach ($file in $files) {
    if (Test-Path -LiteralPath $file -PathType Leaf) {
        Remove-Item -LiteralPath $file -Force
        Write-Host "Removed: $file"
    }
}
```

## 8. 当前结论

性能采集链路已跑通，且本轮 **1 次订单查询 + 1 次知识库问答** 均返回 HTTP 200。知识库样本中 Verifier 耗时 **38.27 秒（约 59.45%）**，是后续应优先排查的阶段；订单样本的 Router 耗时 **3.79 秒**，值得进一步观察。**这些都是探索性单样本线索，不构成稳定性能基线、问题根因判定或优化效果证明。**
