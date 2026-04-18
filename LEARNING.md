# Mini-SGLang 学习计划

记录已完成的任务和备选学习点。

---

## 已完成

### 1. Tokenizer 批量编码（batch tokenization）
- **文件**: `python/minisgl/tokenizer/tokenize.py`
- **内容**: 把 `TokenizeManager.tokenize()` 从逐条 `encode` 改成批量调用 HuggingFace tokenizer，用 `attention_mask` 裁掉 padding，支持 fallback
- **分支**: `feat/batch-tokenization`
- **测试**: `tests/misc/test_tokenize.py`
- **可视化**: `tests/misc/visualize_tokenize.py`（通过 `PrintHook` 回调展示每步数据变换）

---

### 2. abort_req O(1) 字典查找
- **文件**: `python/minisgl/scheduler/decode.py`
- **内容**: 把 `DecodeManager.running_reqs` 从 `Set[Req]` 改成 `Dict[int, Req]`（uid → Req），`abort_req` 从 O(n) 线性扫描变为 O(1) `dict.pop`；同步更新 `filter_reqs`、`remove_req`、`inflight_tokens`、`schedule_next_batch`
- **分支**: `main`
- **测试**: `tests/scheduler/test_abort_req.py`（11 个用例，覆盖 hit/miss/empty/idempotent/filter/inflight/batch）

---

## 备选学习点

### 简单（和 batch tokenization 类似难度）

#### B. 补全 API sampling 参数传递
- **文件**: `python/minisgl/server/api_server.py:265, 298`
- **现状**: `/v1/chat/completions` 只传了部分参数，缺少 `repetition_penalty`、`min_p`、`frequency_penalty` 等
- **目标**: 把这些参数从 HTTP 请求完整传到 `SamplingParams`
- **学习收获**: 理解请求从 API 入口到模型推理的完整数据流路径

#### C. NaiveCache evict 策略
- **文件**: `python/minisgl/kvcache/naive_cache.py:35`
- **现状**: `evict()` 直接抛 `NotImplementedError`
- **目标**: 实现一个简单的 LRU 淘汰策略
- **学习收获**: 理解 KV cache 的分配、使用和回收机制

#### F. 正确传递 finish_reason
- **文件**: `python/minisgl/message/tokenizer.py`, `python/minisgl/scheduler/scheduler.py:153`, `python/minisgl/server/api_server.py:185`
- **现状**: `api_server.py` 硬编码 `finish_reason: "stop"`，即使是被 `max_tokens` 截断的请求也汇报 `"stop"`，客户端无法感知截断
- **目标**: 给 `DetokenizeMsg` 加 `finish_reason` 字段（`"stop"` / `"length"`），在 scheduler 层区分两种结束原因，在 API 层正确返回
- **学习收获**: 理解跨多层（调度器 → 消息队列 → API 层）的字段贯通，以及 OpenAI API 规范中 `finish_reason` 的语义

### 中等（需要理解更多系统上下文）

#### D. 调度策略可配置（PREFILL-first / DECODE-first）
- **文件**: `python/minisgl/scheduler/scheduler.py:220`
- **现状**: 只支持 PREFILL-first 调度
- **目标**: 抽象出调度策略枚举，支持 DECODE-first 或其他策略
- **学习收获**: 深入理解 prefill 和 decode 两阶段如何竞争计算资源

#### E. RadixCache reset() 和 check_integrity()
- **文件**: `python/minisgl/kvcache/radix_cache.py:178, 188`
- **现状**: `reset()` 抛 `NotImplementedError`，`check_integrity()` 是空 `pass`
- **目标**: 实现 radix tree 的清空和完整性校验（引用计数一致性、树结构不变量）
- **学习收获**: 深入理解前缀共享 KV cache 的数据结构

---

## 参考资源

- 项目架构说明: `docs/structures.md`
- 功能列表: `docs/features.md`
