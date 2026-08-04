# Kernel Lab 生产应用与 E2E A/B（v1）

日期：2026-08-04

硬件：AMD Radeon RX 7900 XTX（ROCm 7.2.1 / PyTorch 2.9.1）

结论状态：**已应用、已完成真实 replay 驱动的 learner E2E A/B；不启用 `torch.compile` 训练路径。**

## 1. 执行结论

本轮没有把 `kernel-lab/workload/src/small_gemm.hip` 直接伪装成可用于训练的 PyTorch 内核。该文件是固定尺寸的前向 GEMM benchmark，没有 PyTorch dispatcher、动态 shape、bias、autograd/backward 和优化器 ABI；并且实验室自己的对照数据已经表明，手写 workload GEMM 比当前 `torch.mm`/hipBLAS 路径更慢：

| 尺寸 | workload 最快手写 GEMM | hipBLAS / `torch.mm` | 直接替换判断 |
|---|---:|---:|---|
| 128³ | 0.414–0.428 TFLOPS | 0.643–0.672 TFLOPS | 手写版约慢 35%–38% |
| 256³ | 1.944–1.954 TFLOPS | 3.046–3.060 TFLOPS | 手写版约慢 36% |

`fp32/` 是峰值 FMA 形态实验，不是矩阵乘法；`int32/` 与训练主体的 FP32/BF16 计算面不匹配。因此直接移植这三族 benchmark 内核不仅缺少正确训练 ABI，还会让真实 workload 变慢。

真正适用于当前 4.46M 参数 learner 的结论来自 `PLAYBOOK.md` 第 5 节：它不是 FLOP-bound，而是大量小调用、launch、CPU 调度和 device-to-host 同步构成的延迟负载。生产路径中发现了一个确定的同步点：

```python
missing = ~valid.any(dim=-1)
if missing.any():              # CUDA/ROCm Tensor -> Python bool，强制 host sync
    valid = valid.clone()
    valid[missing, 0] = True
```

该逻辑位于 `GroundedCandidateModel` 的 `_safe_valid_mask`，每次模型调用会分别为 local/world/action mask 执行三次；failure-credit replay 每次 learner update 又会调用模型数百次。

生产实现已改为完全等价的 branchless tensor kernel：

```python
valid = mask.bool()
missing = ~valid.any(dim=-1, keepdim=True)
first_column = torch.arange(valid.shape[-1], device=valid.device).eq(0)
return valid | (missing & first_column)
```

它只为全空行打开第 0 列，非空行逐位不变；没有 host 分支、没有 `.item()`/`bool(tensor)`，同时消除了 `torch.compile(fullgraph=True)` 的 graph break。

代码位置：

- `packages/rl-agent/sts2_rl/models/grounded_candidate.py`
- 等价性回归：`packages/rl-agent/tests/test_grounded_candidate_model.py`
- 真实 E2E harness：`packages/rl-agent/scripts/benchmark_kernel_lab_e2e.py`

## 2. E2E 实验契约

### 2.1 冻结来源

检查点（只读，未改动）：

```text
E:\game\project\sts2_mcp_artifacts\runtime\checkpoints\
full-run-revival-v32-budget64-mature-model-init\
run-481995a5-0221-4f59-9293-9105cd336068\
periodic-step-000090152
```

从该检查点的 replay sidecar 制作了一个小型、不可变的 benchmark fixture：

```text
E:\game\project\sts2_mcp_artifacts\runtime\analysis\kernel-lab-e2e-fixture-v1
```

fixture 本身不是 checkpoint，也不能用于 resume。每个 A/B 进程都重新加载完全相同的：

- 4 个真实在线 unroll，共 64 个环境步；
- 4 条 failure-credit plan，上下文长度为 256、33、256、33；
- failure strata：2 条 `RISK_SEQUENCE/UNRESOLVED_STALL` 与 2 条 `COMPLETION_CONTROL`；
- 2 条 episodic sequence，burn-in 为 41/32，learn steps 为 32/32；
- 网络、AdamW 优化器、policy/learner schedule 状态；
- 稳定 math-SDPA backend；
- 相同的逐 update dropout seed。

该冻结 v32 检查点携带 V12 training config，而当前代码契约为 V13。harness 使用仓库中已有、受审查的 `model_initialization_config_from_mapping` 做 V12→V13 模型初始化解析；这不是、也没有被表述为 exact resume。两臂使用同一迁移结果，因此不影响内核 A/B 的内部效度。

每个进程预热 1 次，随后测量完整 `VTraceLearner.update()`。旧实现仅由 harness 在进程内 monkeypatch 回同步版本；生产代码始终是 branchless 版本。没有启动环境、collector 或训练 lineage。

权威主指标是 `wall E2E`：harness 在 update 前后都调用 `torch.cuda.synchronize()`。组件表沿用 learner 自身的 CPU-envelope/progress 计时，适合定位收益来源，但由于 GPU 异步执行不应被当成可严格相加的独立 kernel 时间。

### 2.2 重复与顺序

| 重复 | 执行顺序 | seed | 每臂测量 update |
|---|---|---:|---:|
| r1 | 同步版 → branchless | 331704 | 5 |
| r2 | branchless → 同步版 | 331705 | 7 |
| r3 | 同步版 → branchless | 331706 | 5 |

总计：每臂 17 个完整 learner update。r2 反转顺序，用于检查热状态、执行顺序与降频造成的假提升。

## 3. E2E 结果

### 3.1 每轮 wall-clock

| 重复 | 旧同步版均值 | branchless 均值 | wall E2E 降低 |
|---|---:|---:|---:|
| r1 | 16.932 s | 16.615 s | 1.87% |
| r2（反向顺序） | 17.106 s | 16.194 s | 5.33% |
| r3 | 16.642 s | 15.867 s | 4.66% |
| **17 update/臂加权汇总** | **16.918 s** | **16.222 s** | **4.12%** |

加权汇总相当于：

- 每个 learner update 节省约 **696.7 ms**；
- wall-clock speedup 为 **1.043×**；
- 理论 update 吞吐由约 **212.8 update/h** 提升至 **221.9 update/h**，即 **+4.30%**。

原始样本完整保留的 r2+r3（每臂 12 个 update）中，均值下降 **5.06%**；按相同 update 序号计算的下降均值为 5.03%，bootstrap 指示区间约为 4.03%–6.04%。该区间只用于说明复测稳定性，不应视为跨机器或跨训练配方的统计保证。

### 3.2 组件加权均值

| 组件 | 旧同步版 | branchless | 变化 |
|---|---:|---:|---:|
| wall E2E | 16,918.31 ms | 16,221.58 ms | **−4.12%** |
| learner 内部 total | 16,570.73 ms | 15,867.05 ms | **−4.25%** |
| online recurrent forward | 369.02 ms | 346.90 ms | −5.99% |
| backward envelope | 16,047.05 ms | 15,374.79 ms | **−4.19%** |
| liveness replay | 13,918.77 ms | 13,345.09 ms | **−4.12%** |
| episodic replay | 1,830.44 ms | 1,752.22 ms | −4.27% |
| optimizer | 81.89 ms | 75.97 ms | −7.23%（绝对量很小） |

`liveness_replay` 仍然是绝对主瓶颈。此次修改移除了高频同步点，但没有改变 replay 的逐记录/逐片段执行结构；因此 4%左右是可信、有限的第一阶段收益，不是数量级突破。

## 4. 数值、参数与显存验证

- mask 单元测试覆盖“全空行只打开第 0 列”和“非空行逐位不变”。
- `tests/test_grounded_candidate_model.py`：**29 passed**。
- 全量 RL package 回归：**1,121 passed**。
- r1 与 r3 两臂最终 model SHA256 完全相同。
- r3 保存两臂完整 CPU `state_dict` 后逐参数比较：
  - 参数标量数：4,455,626；
  - 不同标量数：0；
  - max absolute delta：0；
  - RMS delta：0。
- r2 出现 ROCm 浮点归约级非确定性：最终 loss 差 `3.26e-9`、gradient norm 差 `2.98e-8`；不影响策略语义。其余两轮逐位一致。
- 六个进程的峰值 allocated memory 均为 **3,580,888,064 bytes**；新内核没有增加显存峰值。
- `ruff` 检查通过；修改模型的 strict mypy 检查通过。
- ROCm one-shot 进程在正常退出后均打印 `SharedSignalPool, 2 Signals leaked` 警告；两臂对称出现、结果文件已完整落盘、退出码为 0，且复查没有残留 benchmark/Python 进程。该运行时退出告警未被计入性能收益，也不能据此宣称资源生命周期已经修复。

## 5. `torch.compile` 判定

移除 host 分支后，生产 shape 的 fullgraph 编译首次能够通过。单独的模型 microbenchmark 结果为：

| 模式 | eager | fullgraph compile | 结论 |
|---|---:|---:|---|
| forward-only 均值 | 10.659 ms | 5.146 ms | **−51.72%** |
| forward-only 中位数 | 9.215 ms | 4.773 ms | −48.20% |
| training-step 均值 | 24.447 ms | 45.657 ms | **+86.76%（退化）** |
| training-step 中位数 | 20.810 ms | 44.803 ms | +115.30%（退化） |

此外，forward-only 编译与预热约 74.4 s，training-step 编译与预热约 142.4 s。

因此本轮决策是：

1. **保留 branchless 生产实现**；
2. **不把 `torch.compile` 接入 learner 训练路径**；
3. forward-only 的 compile 收益只能作为将来 actor/evaluator 专用实验，不能用它推断训练 E2E；
4. 不改变 checkpoint ABI、模型参数 shape、优化器、训练 config 或 lineage。

## 6. 产物与复现

汇总：

```text
E:\game\project\sts2_mcp_artifacts\runtime\analysis\kernel-lab-e2e-comparison.json
```

单轮原始结果：

```text
E:\game\project\sts2_mcp_artifacts\runtime\analysis\kernel-lab-e2e-synchronizing.json
E:\game\project\sts2_mcp_artifacts\runtime\analysis\kernel-lab-e2e-branchless.json
E:\game\project\sts2_mcp_artifacts\runtime\analysis\kernel-lab-e2e-synchronizing-r2.json
E:\game\project\sts2_mcp_artifacts\runtime\analysis\kernel-lab-e2e-branchless-r2.json
E:\game\project\sts2_mcp_artifacts\runtime\analysis\kernel-lab-e2e-synchronizing-r3.json
E:\game\project\sts2_mcp_artifacts\runtime\analysis\kernel-lab-e2e-branchless-r3.json
```

完整状态等价性样本：

```text
E:\game\project\sts2_mcp_artifacts\runtime\analysis\kernel-lab-e2e-synchronizing-r3-state.pt
E:\game\project\sts2_mcp_artifacts\runtime\analysis\kernel-lab-e2e-branchless-r3-state.pt
```

模型 microbenchmark：

```text
E:\game\project\sts2_mcp_artifacts\runtime\analysis\kernel-lab-eager-fp32.json
E:\game\project\sts2_mcp_artifacts\runtime\analysis\kernel-lab-fullgraph-fp32.json
E:\game\project\sts2_mcp_artifacts\runtime\analysis\kernel-lab-eager-train-fp32.json
E:\game\project\sts2_mcp_artifacts\runtime\analysis\kernel-lab-fullgraph-train-fp32.json
```

## 7. 回滚与后续边界

回滚只需恢复 `_safe_valid_mask` 的旧同步实现；没有 checkpoint/data/schema 迁移。由于新旧逻辑已经通过完整参数级 A/B 等价性验证，正常情况下不需要回滚。

下一阶段若继续优化，不应手写固定 shape GEMM；应优先处理仍占约 13.3 s/update 的 liveness replay：

1. 将同 shape 的 context segment 做 active-shape batching；
2. 减少逐记录 forward/backward 和 Python 调度；
3. 在保持 evidence/credit 语义完全不变的条件下合并 autograd microbatch；
4. 每次改造继续使用本 harness 做真实 replay E2E 与最终参数等价性 A/B。
