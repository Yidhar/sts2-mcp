# MuZero / Search-free World Model

本目录现在统一收纳 **STS2 MuZero 系列实现**：旧 MCTS 兼容层仍保留，但当前
目标态已经转向 **attention-first / JEPA-style / search-free planner**。

英文版见：[`README.md`](./README.md)

## 当前定位

- **combat sandbox 必须保留**：它是高吞吐快速战斗训练器，不能被 full-run 吞吐拖慢。
- **MCTS 保留为兼容/对照路径**：旧 checkpoint、旧 replay、build/route 对照实验仍可用。
- **战斗主路线正在转向 search-free**：`action_rollout_planner` 用共享 dynamics 在模型内部做
  Q-like / lookahead value，不再依赖在线树搜索来“想明白”每一步。
- **token_memory_v1 是目标结构**：手牌、抽牌堆、弃牌堆、消耗牌堆、药水、遗物、
  能量/X 费、血条、敌人 buff/intent/reaction、全局牌库、路线都 token 化后进入
  banked attention / memory slots。
- **混合精度已经纳入训练路径**：`--mixed-precision auto` 在 CUDA 上优先用 bf16，
  不支持 bf16 时用 fp16 + GradScaler；CPU auto 保持 fp32/off。token-memory 默认训练
  窗口改为 `--batch-size 32`、`--unroll-steps 3`；旧的 128×5 fp32 对当前结构激活显存太重。
- **search-free rollout 已改成 bucketed tensor 路径**：`--action-rollout-buckets`
  默认 `8,16,32,64,80`，root legal actions 和 latent beam branches 会先 pad 到稳定桶，
  再进入重的 dynamics/value heads；continuation 剪枝也改成 root-bucket grouped
  top-k，不再走 per-root `nonzero` / Python list / `cat` 循环。这是面向
  ROCm/HIP allocator 的方案 B，用来降低 reserved 远大于 allocated 的碎片化风险，
  同时不退回 MCTS。

## 目录结构

```text
packages/rl-agent/muzero/
├── README.md
├── README.zh-CN.md
├── __init__.py
├── train.py                    # 训练入口：combat sandbox / full run / direct combat policy
├── evaluate.py                 # checkpoint 评估
├── analyze_replay.py           # replay 分析
├── eval_latent_probes.py       # 线性 probe：检查 latent 是否编码 HP/能量/牌堆/敌人/路线等
└── sts2_env/
    ├── __init__.py
    ├── mcts.py                 # 兼容 MCTS / 对照实验
    ├── muzero_buffer.py        # replay buffer / trajectory
    ├── muzero_model.py         # MuZeroNetwork + search-free rollout planner
    ├── token_memory.py         # token world memory encoder / banked attention / token dynamics
    ├── latent_regularizers.py  # JEPA/SIGReg-style latent Gaussian regularizers
    └── semantic_rollout.py
```

## 推荐入口

```bash
cd packages/rl-agent

# 训练
python -m muzero.train --obs-mode token_v3 --model-arch token_memory_v1 \
  --mixed-precision auto --batch-size 32 --unroll-steps 3 ...

# 显式启用战斗 search-free direct policy
python -m muzero.train --combat-sandbox --combat-direct-policy \
  --combat-rollout-steps 3 --combat-rollout-beam-width 2 \
  --action-rollout-buckets 8,16,32,64,80 ...

# 当前 auto 模式下：token_memory_v1 + combat sandbox 会默认走 direct；
# 如果要强制旧 MCTS 对照，用 --combat-policy-mode mcts。

# 评估 / replay 分析
python -m muzero.evaluate ...
python -m muzero.analyze_replay ...

# latent probe：确认模型内部是否线性可读出 HP/能量/牌堆/敌人/路线信息
python -m muzero.eval_latent_probes --checkpoint <checkpoint_dir> \
  --markdown-output runs/latent_probe.md
```

## 当前模型方法

### 1. Token-world memory encoder

`token_memory_v1` 不再把观测简单压成一个 dense 向量，而是把世界拆成 token bank：

- `runtime`：玩家状态、资源、手牌、抽牌堆、弃牌堆、消耗牌堆、循环计划、能量预算
- `support`：遗物、药水、支持图
- `enemy`：敌人核心状态、意图、power/buff、特性、反应
- `build`：全局牌库、拿牌/商店/强化/变化选项
- `route`：地图路线、节点风险、路线价值
- `powers`：power slot、卡牌关键词
- `history`：近期动作/回合摘要

编码路径：

1. world tokens 自注意力；
2. candidate action query + local action context 自注意力/交叉注意力；
3. candidate set 自注意力；
4. **按 bank 显式分组的 cross-attention**：动作 query 不再一次性盲连全世界，而是先路由到
   runtime/support/enemy/build/route/powers/history 等 bank，再对相应 bank 做交叉注意力；
5. persistent memory slots：每个 slot 有 bank identity，专门吸收对应 world bank；
6. 输出 flat hidden state 给 MuZero dynamics/prediction，同时保留 action embeddings。

### 2. 共享 text / numeric trunk

`world_embedder`、`query_embedder`、`local_embedder` 不再各自重复编码文本和数值模态：

- `shared_text_trunk`：统一处理 token text vector；
- `shared_numeric_trunk`：统一处理 numeric vector；
- `_project_shared_modal_trunk_with_reuse(...)`：把 world/query/local 的重复 rows 合并去重后投影，
  再 scatter 回原位置，减少重复编码。

### 3. Dynamics + JEPA world modeling

MuZero recurrent step 输出：

- next hidden state
- reward / reward components
- policy/value/objective heads
- next legal surface / domain / phase
- future world-bank reconstruction
- token slot state/mask/type/zone/source 预测
- trainable `surprise` head

训练损失增加：

- `loss/jepa_next_hidden`：dynamics next hidden 对齐 representation(next_obs)
- `loss/latent_gaussian_reg`：latent Gaussian / isotropy 正则，降低 collapse
- `loss/surprise`：用 hidden prediction error + future-world/surface auxiliary error 监督 surprise
- future world-bank / token-slot auxiliary losses：显式要求 latent 记住牌堆、遗物药水、敌人、
  构筑和路线 token 的下一步状态。

### 4. Search-free action rollout planner

`MuZeroNetwork.action_rollout_planner(...)` 是战斗 direct policy 的核心：

1. 对所有真实 legal root actions 并行过 shared dynamics；
2. 用 shared value/objective value 得到 one-step Q；
3. 用 latent policy + predicted legal surface 做多步 beam continuation；
4. 聚合回 root action：
   - `planner_q`
   - `planner_objective_q`
   - `planner_risk_q`
   - `planner_uncertainty`
5. direct planner 的 shape 稳定策略：
   - root legal actions pad 到 `--action-rollout-buckets`；
   - latent branch dynamics/value 调用也 pad 到同一组 bucket；
   - continuation 剪枝使用 root-bucket grouped top-k，不再为每个 root 动态
     `nonzero`/`cat`。
6. `planner_uncertainty` 由以下组成：
   - trainable surprise
   - next-surface entropy
   - latent norm drift
   - branch Q disagreement

direct combat policy 会把这些 Q/uncertainty bias 融入 policy logits，从而用模型内部规划替代在线 MCTS。

## 参数量

当前默认实例化参数量：

- `MuZeroNetwork(obs_mode="token_v3", model_arch="token_memory_v1")`：**23,084,242**
- `MuZeroNetwork(obs_mode="dense_v2", model_arch="dense_v1")`：**14,261,639**

## 兼容层策略

旧路径仍保留为 wrapper/兼容导入，实际实现以 `muzero/` 为准：

- 顶层 wrapper：
  - `train_muzero.py`
  - `evaluate_muzero.py`
  - `analyze_muzero_replay.py`
- `legacy/` wrapper
- `sts2_env/` 下的 `mcts.py`、`muzero_model.py`、`muzero_buffer.py`、`semantic_rollout.py`

新代码和新文档优先使用：

```bash
python -m muzero.train
python -m muzero.evaluate
python -m muzero.analyze_replay
python -m muzero.eval_latent_probes
```

## 训练建议

当前 search-free 改造建议从 combat sandbox 开始：

- 先跑 `--combat-sandbox`（auto 会启用 direct policy），确认 direct rollout 指标稳定；
- 观察 TensorBoard / log 中：
  - `loss/jepa_next_hidden`
  - `loss/latent_gaussian_reg`
  - `loss/surprise`
  - `metric/surprise_target_offset`
  - `direct_rollout_uncertainty_mean`
  - `direct_rollout_branch_disagreement_mean`
- 定期跑 `eval_latent_probes.py`，确认 hidden/memory slots 能线性读出 HP、能量、牌堆、
  敌人 intent/power、support、build、route 等信号。
