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
  默认 `8,16,32,64,96`，root legal actions 和 latent beam branches 会先 pad 到稳定桶，
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
├── train.py                    # 薄兼容入口；禁止继续堆策略
├── evaluate.py                 # checkpoint 评估
├── analyze_replay.py           # replay 分析
├── eval_latent_probes.py       # 线性 probe：检查 latent 是否编码 HP/能量/牌堆/敌人/路线等
├── training/                   # 训练编排/路径/文件预算；不放战斗策略
│   ├── paths.py                # RunPaths / StrategyModulePaths / HeuristicSearchModulePaths
│   ├── file_budget.py          # 单文件 2000 行预算守卫
│   ├── monitoring.py           # recent-tail monitor / Null writer / episode capture buffer
│   ├── cli_parsing.py          # encounter/session/tier/weight/int-list CLI helper
│   ├── env_factory.py          # train env/session/pool 构造
│   ├── checkpointing.py        # checkpoint load/save/prune + obs tensor helper
│   └── losses.py               # policy/value/reward/objective/future-bank/surface loss mixin
├── strategy/                   # 战斗/选牌/药水/HP/X 费等策略特征与通用决策 helper
│   └── encounters/             # boss/怪物特化机制，按 encounter 拆小文件
│       ├── kaiser.py           # BACK_ATTACK 左右解析 / facing-change 候选
│       └── insatiable.py       # Sandpit countdown / Frantic Escape 候选
├── combat_quality/             # no-pressure pure-block、空过、hard guard、offender 指标
│   ├── block_waste.py          # no-pressure pure-block waste 判断
│   ├── action_bias.py          # pure-block root-prior bias
│   ├── metrics.py              # combat-quality 聚合/metric helper
│   ├── guard_metrics.py        # hard guard 默认 TB metric key 合约；禁止继续塞进 train.py
│   ├── hard_guard_orchestrator.py      # combat hard guard 调度顺序与默认指标初始化
│   ├── basic_hard_guards.py            # discard / Kaiser / Insatiable / X-cost / HP-cost / lethal EndTurn
│   ├── boss_survival_hard_guards.py    # elite/boss 药水、生存格挡、race/setup 窄窗口
│   ├── late_normal_hard_guards.py      # late-Act1 normal lethal/survival/race guard
│   ├── survival_non_endturn_guard.py   # 非 EndTurn 动作在危险窗口的生存替换
│   ├── potion_bad_use_guard.py         # 低时机药水使用拦截，含 fail-open 例外
│   ├── selection_loop_guard.py         # 净化/选卡类重复选择死循环保护
│   └── potion_guard.py                 # 药水 profile/trait 窄窗口 helper
├── route_heuristics/           # route graph/path candidate/scoring/safety/bias
├── search/                     # root prior/search glue；不写 STS2 卡牌策略
├── diagnostics/                # 诊断 dump schema/聚合 helper；运行期文件仍写入 log_dir/diagnostics
└── sts2_env/
    ├── __init__.py
    ├── mcts.py                 # 兼容 MCTS / 对照实验
    ├── muzero_buffer.py        # replay buffer / trajectory
    ├── muzero_model.py         # MuZeroNetwork + search-free rollout planner
    ├── token_memory.py         # token world memory encoder / banked attention / token dynamics
    ├── latent_regularizers.py  # JEPA/SIGReg-style latent Gaussian regularizers
    └── semantic_rollout.py
```

### 文件治理硬规则

- 新代码默认不能再写进 `train.py`；`train.py` 现在只保留 CLI delegation + legacy import compatibility。
- 新 Python 文件必须 `< 2000` 行；超过 1500 行时先拆子模块，再继续实现。
- 运行期路径（log、checkpoint、diagnostics、replay buffer）统一走 `training.paths.RunPaths`。
- 策略源码路径统一走 `training.paths.StrategyModulePaths`：
  - 通用策略：`strategy_file(...)`
  - 怪物/Boss 特化：`encounter_file(...)`
  - 战斗质量/guard：`combat_quality_file(...)`
- route/search 源码路径统一走 `training.paths.HeuristicSearchModulePaths`：
  - route 候选/评分/安全守卫：`route_heuristic_file(...)`
  - root prior/search adapter：`search_file(...)`
- 过渡期需要一把拿全路径时才用 `PolicyModulePaths`。
- 检查命令：

```bash
cd packages/rl-agent
./.venv-wsl-rocm/bin/python scripts/check_muzero_file_budget.py --quiet-ok
```

当前只有历史债务文件在 allowlist 中；新超限文件会让检查失败。

### 已拆出的策略模块

- `combat_quality/block_waste.py` / `action_bias.py`：无伤害压力下纯格挡 waste 与 root-prior bias。
- `combat_quality/guard_metrics.py`：hard guard 默认 TensorBoard metric key 合约。
- `combat_quality/potion_guard.py`：药水不是“有就用”；只在 boss race、0 能量无非药水替代、真实压力/关键 HP、Lagavulin setup 等窄窗口放行。
- `combat_quality/hard_guard_orchestrator.py` + `basic_hard_guards.py` / `boss_survival_hard_guards.py` / `late_normal_hard_guards.py` / `survival_non_endturn_guard.py` / `potion_bad_use_guard.py` / `selection_loop_guard.py`：把原 `train.py::_apply_combat_action_hard_guards` 的 2700+ 行拆成按职责分桶的 mixin；每个新文件均 `< 2000` 行。
- `strategy/encounters/kaiser.py`：Kaiser/back-attack/facing-change 解析；左右只来自 `BACK_ATTACK_LEFT_POWER` / `BACK_ATTACK_RIGHT_POWER`，不能用 `enemy.side == "Enemy"`。
- `strategy/encounters/insatiable.py`：Insatiable Sandpit countdown 与 `Frantic Escape` 候选识别；先用内部 id，再用中英文标题兜底。
- `training/monitoring.py` / `cli_parsing.py` / `env_factory.py`：把监控、CLI 解析、环境构造移出 `train.py`。
- `training/checkpointing.py` / `losses.py`：把 checkpoint save/load/prune 和 tensor loss helper 移出 `train.py`；`MuZeroTrainer` 只通过 mixin 继承。

`train.py` 里对应方法只允许作为兼容 wrapper / adapter 存在，不能继续把策略阈值和机制判断写回去。

## 推荐入口

```bash
cd packages/rl-agent

# 训练
python -m muzero.train --obs-mode token_v3 --model-arch token_memory_v1 \
  --mixed-precision auto --batch-size 32 --unroll-steps 3 ...

# 显式启用战斗 search-free direct policy
python -m muzero.train --combat-sandbox --combat-direct-policy \
  --combat-rollout-steps 3 --combat-rollout-beam-width 2 \
  --action-rollout-buckets 8,16,32,64,96 ...

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
