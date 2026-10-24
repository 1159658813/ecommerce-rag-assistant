# P2-B：Verifier 思考模式 A/B 小样本实验

> 2026-10-24。实验提案 / 非优化结论。只改 `ANSWERABILITY_VERIFIER_ENABLE_THINKING`；不切换模型、不改 Prompt、不降低 Evidence Gate 安全约束。与 `docs/p2b_pilot_results.md` 的 2026-10-21 探索样本相区别。

## 为什么选这个变量

`src/config/settings.py` 的 Verifier 默认模型是 `qwen3.6-plus`；阿里云官方文档说明该模型默认开启思考模式，且可在 OpenAI-compatible SDK 请求中用 `extra_body={"enable_thinking": false}` 单独关闭。当前源码之前未覆盖该参数。**这只构成有根据的候选优化，尚未证明延迟变短或判定质量不变。**

来源：<https://help.aliyun.com/zh/model-studio/deep-thinking>

## 行为与安全契约

- `.env` **不设置** `ANSWERABILITY_VERIFIER_ENABLE_THINKING`：完全不传额外参数，保留旧行为（baseline）。
- `.env` 设置 `ANSWERABILITY_VERIFIER_ENABLE_THINKING=false`：仅 Verifier 请求增加 `extra_body={"enable_thinking": False}`。实际模型、重试、Prompt、证据及解析/校验均不变。
- `.env` 设置 `ANSWERABILITY_VERIFIER_ENABLE_THINKING=true`：明确指定开启；通常不必设置。
- 若配置值非法，启动时 fail fast。变更前后只接受 `SUFFICIENT`/`INSUFFICIENT`；解析错误和服务异常仍然安全拒答。`SUFFICIENT` 不得含非空 `missing_facts`；`INSUFFICIENT` 必须含 `missing_facts`。
- 禁止为了缩短延迟而更改 timeout、重试次数、Evidence Gate 或替换模型。

## 0. 离线回归（不会调用 DashScope）

从项目根目录、已安装 `requirements.txt` 的本地 `rag-shop` 环境运行：

```powershell
python -m unittest tests.test_p2b_verifier_toggle -v
python .\scripts\p2_trace_summary.py --self-test
python .\scripts\p2_perf_trace_pilot.py --self-test
python -m compileall -q src tests scripts
```

## 1. 首次上线候选前：部署补丁、保持旧默认，检查健康

```powershell
git status --short --branch
python -m unittest tests.test_p2b_verifier_toggle -v
# .env 中先不要配置 ANSWERABILITY_VERIFIER_ENABLE_THINKING
# 只要 src 变化，Docker 镜像需重建；requirements 未改，通常能复用依赖缓存。
docker compose build rag-app
docker compose up -d --no-deps rag-app
docker compose ps
docker compose exec -T rag-app python -c "from src.config import settings; print('model=', settings.verifier_model, 'thinking_override=', settings.verifier_enable_thinking)"
```

如果 Docker 需要此前已验证的构建镜像参数，仍使用 P2-A 的已验证方案，不要重新排查网络。Docker 日志与配置不得粘贴任何 API key。以上命令不会主动调用 LLM。

## 2. A/B 小样本前明确费用上限

真实调用**不自动执行**。预算从 **2 个固定问题（1 个冻结标签可答 + 1 个冻结标签不可答）× 2 种模式 × 1 次 = 最少 4 次 HTTP 知识问答**开始；其中每个请求可能触发 Router、Verifier、Composer 及内部重试，具体 Token/费用未知，请以账户实际计费为准。质量和路由检查不通过则停止。只有在用户同意付费测试后才给命令加 `--execute`。

样例：
- `knowledge`：「七天无理由退货必须保留商品原包装吗？」（旧 Pilot 曾判为 `SUFFICIENT`；仍需检查答案事实）
- `knowledge_insufficient`：「商家超过48小时没发货的话，平台是不是会自动赔我订单金额的10%？」（冻结集 `generation_questions_v1.json` 的 `b2_q60` 标注 `answerable=false`，预期拒答；注意标签不等于新请求的自动真值）

**Baseline：** `.env` 不设置该变量并重启服务，确认容器内 `thinking_override=None`。

```powershell
# 默认只 dry run；用户确认预算后在对应行末尾附加 --execute
python .\scripts\p2_perf_trace_pilot.py --case knowledge --repeats 1 --output "$env:TEMP\ecommerce_rag_p2\thinking_baseline.jsonl"
python .\scripts\p2_perf_trace_pilot.py --case knowledge_insufficient --repeats 1 --output "$env:TEMP\ecommerce_rag_p2\thinking_baseline.jsonl"
```

**候选：** 只在本机 `.env` 加入 `ANSWERABILITY_VERIFIER_ENABLE_THINKING=false`，重启后确认容器里显示 `False`：

```powershell
docker compose up -d --no-deps --force-recreate rag-app
docker compose exec -T rag-app python -c "from src.config import settings; print('model=', settings.verifier_model, 'thinking_override=', settings.verifier_enable_thinking)"
python .\scripts\p2_perf_trace_pilot.py --case knowledge --repeats 1 --output "$env:TEMP\ecommerce_rag_p2\thinking_off.jsonl"
python .\scripts\p2_perf_trace_pilot.py --case knowledge_insufficient --repeats 1 --output "$env:TEMP\ecommerce_rag_p2\thinking_off.jsonl"
```

仅当已核对 `--execute` 的费用和本机无其他访问时再执行真实请求。脚本时间窗口配对 **不是 Request ID 级关联**，遇到 0 / 多条 Trace 应停止。客户端 JSONL 只记录匿名路由、`verdict`、`abstained`、白名单安全拒答原因和延迟，不保存答复正文；**回答事实正确性仍需独立核对**，不能仅凭返回 `SUFFICIENT` 判为正确。注意 `--case all` 现包含新增的 insufficient 用例；避免无意扩大调用次数。

## 3. 接受门槛与回滚

- 首先检查各样例是否正确走 `KNOWLEDGE_QA` / `RAG`；否则对应样本无效。
- 不可答问题不允许由 `INSUFFICIENT` 变为无根据的 `SUFFICIENT`，不允许生成臆测赔偿比例；无法确认预期结果或质量下降，立即回滚。可答问题需兼顾证据与回答质量，拒答增加也不能算成功。
- 记录每次 `client_latency_ms`、服务端 `latency_total`、`latency_verifier`、`latency_composer`、`verdict`、`abstained`、`verifier_retry_count`、错误数以及相同模型版本/网络条件。超时/失败不是性能成功。
- 只有每组 n=1 的探索性样本时不公布稳健的 P50/P95 和百分比提升。后续预算许可才考虑更多重复测量。
- 回滚：从 `.env` **删除** `ANSWERABILITY_VERIFIER_ENABLE_THINKING` 一行，然后 `docker compose up -d --no-deps --force-recreate rag-app`。无需回滚安全模块或改动模型。

## 4. Temp 清理

在将匿名统计与实验结论写入正式报告、确认不再需要原始 JSONL 后，精准删除 `%TEMP%\ecommerce_rag_p2\` 中本次新生成的 `thinking_baseline.jsonl`、`thinking_off.jsonl`；之前的 `baseline.jsonl`、`paired_pilot.jsonl` 按交接文档单独核验清理。绝不清空整个 Temp、删除数据库或执行 `docker compose down -v`。
