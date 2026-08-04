# v33 恢复与修复方案 v1

状态：**实现完成，100k 实证验收待执行**。  
实现日期：2026-08-04。  
依据：2026-08-02～04 的坍缩取证、跨血统行为审计（锻造零使用、药水正常、
贪心锚点）、reward v5 数学审计与两轮文献对抗审查。

本文现在同时承担三项职责：

1. 记录 v33 稳定性包的设计决策；
2. 给出已经落入代码、配置、checkpoint ABI 和测试的实现映射；
3. 明确区分“代码验收完成”和“100k 在线训练效果验收尚未完成”。

分层原则保持不变：**v33 = 稳定性包**，reward v5 与课程
`revival_budget = 64` 不变；**v34 = 奖励/课程包**，只允许在 v33 终审通过后启动。
两包分离是为了保住归因能力，避免把学习动力学修复与奖励曲率修改混在同一血统。

---

## 0. 实施结论

| 范围 | 状态 | 结论 |
|---|---|---|
| B1～B8 学习语义与运行守卫 | 已完成 | 已进入 config v13、learner、collector、checkpoint 与 runtime 正式路径 |
| v33 model-init 启动器 | 已完成 | 父 checkpoint 的路径、ID、manifest/metadata 哈希、policy/update 均被钉死 |
| checkpoint / rollback ABI | 已完成 | learner 动态状态、健康角色、守卫回滚次数和训练死锁连击可 exact round-trip |
| E2 商店删牌动作覆盖 | 已完成 | 已确认参数化购买→remove→confirm→牌组减少的完整链，修复策略 subtype 丢失并补结果遥测 |
| 静态与单元/集成验证 | 已完成 | Ruff、strict mypy、全量 1120 项 pytest 全通过 |
| 源 checkpoint 身份与训练 dry-run | 已完成 | 精确身份校验通过；model-init 命令含 `--initialize-from` 且不含 `--resume` |
| 100k ROCm 在线训练与 n=64 终审 | **未执行** | 这是效果验收，不得用单元测试结果替代 |

### 0.1 关键实现文件

- `packages/rl-agent/sts2_rl/training/learner.py`
- `packages/rl-agent/sts2_rl/training/collector.py`
- `packages/rl-agent/sts2_rl/training/runtime.py`
- `packages/rl-agent/sts2_rl/training/checkpointing.py`
- `packages/rl-agent/sts2_rl/training/pipeline.py`
- `packages/rl-agent/sts2_rl/training/config.py`
- `packages/rl-agent/sts2_rl/encoding/grounded.py`
- `packages/rl-agent/config/experiments/full_run_revival_v33_stability_recovery_model_init.toml`
- `packages/rl-agent/scripts/launch_v33_stability_recovery_model_init.py`

---

## A. v33 启动配置：明确的 model-init 新血统

### A1. 父 checkpoint 已固定，不再运行时择优

v33 唯一允许的参数初始化来源是：

```text
E:\game\project\sts2_mcp_artifacts\runtime\checkpoints\
full-run-revival-v32-budget64-mature-model-init\
run-481995a5-0221-4f59-9293-9105cd336068\
periodic-step-000090152
```

固定身份：

| 字段 | 值 |
|---|---|
| source run ID | `481995a5-0221-4f59-9293-9105cd336068` |
| environment steps | `90,152` |
| policy version / learner updates | `1,467 / 1,467` |
| checkpoint ID | `1c0d5284-6be0-4367-a669-166f37b9b3ec` |
| manifest SHA-256 | `6d9c6047d44df0d6e22eb8aa7b04cc5bc317e2b5e999c074d41357444d82b1c2` |
| metadata SHA-256 | `ea372f065ad99eed368e5de71fbac1d2ac9b60f57041fb76c2220d677abf54e6` |

启动器会 fail-closed 校验上述每一个字段。90152 是 v32 坍缩前最后一个被选定的周期点；
`guard-stop-100105` 只保留为失败证据，不允许作为初始化父本。

### A2. 继承边界

这是 **model parameter initialization**，不是 exact resume：

- 继承：网络参数、成熟的主熵/ε 调度时钟；
- 重置：optimizer、replay、RNG、rollout queue、血统本地计数；
- 独立重置：liveness head 校准与 risk-actor 相位时钟；
- 不允许把 config v12 父 checkpoint 伪装成 config v13 exact resume。

启动命令由测试保证使用 `--initialize-from`，且不出现 `--resume`。config v12 只被允许作为
受控 model-init 迁移来源，v13 exact resume 继续要求严格 ABI 一致。

### A3. v33 配方

| 项 | v33 决定 |
|---|---|
| reward | v5，不改奖励身份 |
| revival budget | 64 |
| 总训练步 | 100,000 |
| 普通评估 | 0 / 25k / 50k / 75k，各 16 局 |
| 早期评估 | 5k / 10k，各 16 局 |
| 最终审计 | 100k，64 局 |
| entropy | 0.006 → 0.004，带 one-hot breaker |
| 普通 ε | 0.15 → 0.05；继承成熟调度时钟 |
| 选择面 ε floor | 0.25，仅训练、仅目标选择面 |
| guard 行为 | 回滚最近健康锚点并继续；最多 2 次 |
| 训练死锁告警 | 连续 4 个 deadlock episode 立即触发 |

---

## B. 学习语义修复：实现状态与精确契约

### B1. cost-actor 的 π 侧梯度泄漏——已修复

`centered_risk = C_selected - Σ_a π(a|s) C(a|s)` 中，成本张量原本 detach，策略概率没有
detach，使 baseline 项反向流入策略分支。现在 `legal_probabilities` 在 baseline 中显式 detach，
策略梯度只通过 selected log-probability 路径进入。

回归测试：
`test_cost_actor_policy_baseline_is_detached_from_policy_gradient`。

### B2. direct AVOID 软化与独立 liveness 梯度预算——已修复

原 `-log(1-p_selected)` 在接近 one-hot 时无界。现在使用 alternative mass 的 detached
权重构造软化目标，保留“降低失败动作概率”的方向，同时抑制单条 witness 劫持整个更新。

liveness 损失不再与主 V-trace 共抢同一个全局 clip 配额：

1. 对主目标 backward；
2. 保存主梯度；
3. 对 liveness 记录逐条/逐微批 backward；
4. 只对新增的 liveness 梯度 delta 施加 `0.50` 范数预算；
5. episodic 目标进入后，主目标与 episodic 目标执行原全局 clip；
6. 最后恢复已独立裁剪的 liveness delta。

这样不会因为 liveness 爆发而缩小主训练梯度，也不会让主梯度占满配额后吞掉 liveness 信号。

回归测试：

- `test_direct_avoid_softening_is_finite_and_bounded_near_one_hot_policy`
- `test_liveness_gradient_delta_clip_preserves_the_primary_gradient`

### B3. episodic 成功模仿信赖域——已修复

正优势成功标签只在行为比率 `ρ <= 1 + 0.20` 时提供策略梯度；超过信赖域后停止继续锐化。
价值监督仍然保留，因此这不是丢弃成功轨迹，而是阻止 already-overfit 动作继续获得无界正压。

回归测试：
`test_success_imitation_stops_outside_the_positive_trust_region`。

### B4. 熵地板与批级 one-hot 断路器——已修复

- `entropy_weight_end` 从 0.002 提高到 0.004；
- 连续 8 个**非强制决策批次**满足饱和条件时触发断路器；
- 断路器持续 8 个 learner update，熵权重临时提升到 0.012；
- 强制单候选步不计入重要性比率、clip fraction 与 breaker 判据；
- breaker streak、剩余持续期和触发次数进入 stochastic checkpoint state。

因此 exact resume 不会悄悄重置 breaker 动态。

回归测试：
`test_one_hot_entropy_breaker_triggers_after_eight_batches_and_resumes_exactly`。

### B5. 完成信用重新开启——已修复

`liveness_completion_policy_weight` 由 0 恢复为 0.15。成功完成 selection transaction 的
completion control 可以给策略头正信用，修复“循环有惩罚、成功完成却永远没有平反标签”的不对称。

### B6. 选择面定向探索——已修复

训练 collector 只在以下通用语义面将有效 ε 提高到 `max(schedule_epsilon, 0.25)`：

- card-selection；
- rest-site 选项面。

评估仍为 deterministic/ε=0。journal 记录实际使用的 effective epsilon、targeted exploration
标志、行为概率与 behavior log-probability，learner 不会把定向探索误当成原策略采样。

回归测试：
`test_selection_epsilon_floor_is_targeted_and_behavior_is_journaled`。

### B7. 选择事务入口的失败标签豁免——已修复

observed-selection 状态机现在独立于已禁用的 legacy transaction replay，因此
`transaction_learning.enabled=false` 时仍能识别入口、事务存续、完成和循环失败。

失败语义：

- 对 selection cycle / semantic deadlock，打开选择面的入口决策保留 value 标签；
- 入口不接收该合成终局的负策略标签，避免把“尝试进入锻造/多选”学成禁忌；
- clean exit 产生完成控制信用；
- forge 只由 `operation_type in {upgrade, forge}` 且来源为 rest-site 的精确语义计数，
  不用模糊字符串匹配。

流式 FIFO 的入口可能已经发给 learner，无法在未来终局发生后安全回写。因此 FIFO 入口从一开始
就是 value-only；episodic 副本可以在已知终局后精确回标。这个保守权衡是显式契约，不是假装支持
跨已消费 unroll 的回溯修改。正向完成信用由 B5 补回。

回归测试：

- `test_selection_cycle_exempts_entrance_policy_without_legacy_transaction_replay`
- `test_forge_surface_and_evaluation_metric_are_exact`

### B8. 守卫、回滚、告警与可观测性——已修复

#### 训练侧 deadlock streak

连续 4 个训练 episode 为 deadlock 时：

- 立即发布 `deadlock_alert_evidence` checkpoint；
- 写入结构化 `training_deadlock_streak_alert` 指标事件；
- 同一连续 streak 只告警一次，成功/普通失败会重置 streak。

#### Evaluation guard

guard 失败不再只把 run 标记为停止：

1. 先保存 `guard_failure_evidence`；
2. 找到最近一次带有**哈希保护角色** `healthy_evaluation_anchor` 的 checkpoint；
3. 要求 actor 已暂停、边界确认清空、actor side-channel 为空、rollout queue 为空；
4. 恢复模型、optimizer、RNG、replay、learner dynamics、训练状态与 actor 私有计数；
5. 记录 `in_process_rollback` provenance；
6. 重新执行被 guard 拒绝的评估并继续训练；
7. 单个血统最多回滚 2 次，避免无限回滚循环。

目录名、prefix 或“曾经完成过评估”都不能把普通 checkpoint 冒充成健康锚点。
新 checkpoint 将角色写入原子发布且受哈希保护的 metadata。旧 checkpoint 若缺少 role，仍可按原 ABI
读取并视为 ordinary，以保持历史 exact-resume 兼容；它绝不会被当作健康回滚锚点。显式非法 role
继续 fail-closed。

checkpoint 还新增持久化：

- learner breaker dynamics；
- `training_deadlock_streak` / alerts；
- `evaluation_guard_rollbacks`；
- `maximum_observed_candidates` 等现有诊断状态的兼容迁移。

回归/集成测试：

- `test_checkpoint_roundtrip_restores_v33_learner_dynamics_and_health_role`
- `test_rollback_anchor_requires_hashed_health_role_not_directory_name`
- `test_guard_failure_rolls_back_to_hashed_health_anchor_and_continues`
- `test_four_training_deadlocks_publish_one_immediate_alert_checkpoint`
- `test_paused_actor_can_rewind_private_checkpoint_counters`

#### 常驻评估指标

评估 summary 现已包括 transaction/forge 完成、固定探针 top1-top2 margin、shaping 与楼层比、
按 boss 序位的首杀结果，以及定向探索与选择事务诊断字段。它们用于发现“总胜率尚未变化，但
宏观能力已经塌缩”的早期退化。

---

## C. v33 在线验收标准

以下条件必须通过真实 ROCm 训练和 held-out 评估验证；当前尚未声称通过：

1. 100k 步内无策略坍缩，训练侧 deadlock 连击告警为 0；
2. 100k 终审 `n=64` 胜率至少 0.70，恢复约 75.9k 时的水平；
3. **锻造完成率 > 0**，结束历史恒为零的能力洞；
4. 非强制决策熵稳定在至少 0.15，无批级 one-hot 连击；
5. 在固定、逐 seed 配对的 held-out 集上，继承能力不低于 90152 父 checkpoint；
6. guard 如被触发，必须能证明回滚锚点角色、恢复 provenance 和重试结果，而不是只看目录名；
7. deadlock、Act 1/Act 3、复活次数、玩家失血、锻造/选择事务完成率同时报告，禁止只看最远楼层。

任何一项失败都保留 checkpoint 与 journal 证据；不得在同一 v33 血统内临时改 reward、budget 或
模型结构来“救曲线”。需要改变这些变量时新建下一版本和明确 model-init lineage。

---

## D. v34 奖励/课程包（仅在 v33 通过后）

1. **战斗作用域凸性复活成本**（reward v6，新 identity）：按单场战斗计数、场内凸性；首
   1～2 次便宜，第 5 次起陡增。若保留 run 级线性项，斜率须位于 `(0.0083, 0.0156)`：
   高于每层楼价值，同时总和不覆盖终局成功边际。
2. **budget 阶梯**：64 保持到稳定性与胜率恢复，然后降到 48，制造低复活成功对照样本。
   毕业条件使用 budget=0 的独立评估通道，而不是按训练步数自动推进。
3. reward v5 下复活是近似平坦定价的“可购买资源”；v6 必须让靠额外复活购买一层楼在边际上
   变成净亏，但不得使模型通过提早结束来规避复活。

---

## E. 中期机制（v34+，按实证缺口选择）

- 成就对比辅助头：预测“下一里程碑”，事件集包括首次完成锻造；
- SMDP 宏观 bootstrapping：在非战斗决策子序列上建立独立时间尺度目标；
- matched-pair 覆盖扩展：让游走型 stall 的失败臂能够进入 outcome matching；
- PLR 种子加权与 multi-gamma 辅助头；
- unroll 16→64 必须作为独立配方实验：更新频率会同时下降约 4 倍，不能当作免费吞吐优化。

这些项目没有偷偷并入 v33，以免再制造无法归因的技术债。

## E2. 动作面覆盖审计——商店删牌是参数化事务，策略语义缺口已修复

### E2.1 审计结论

2026-08-04 的端到端核查推翻了“商店删牌动作完全不存在、模型从未删牌”的初步结论。
STS2/HeadlessSim 没有设计独立的 `remove_card` 顶层动作；其可执行协议是：

```text
shop_purchase(item.category = card_removal)
  -> card_select(operation_type = remove)
  -> select_card
  -> confirm_selection
  -> 下一决策观察到 deck_count - 1
```

证据链不是只看 schema 字段，而是逐层核过：

1. 游戏/模拟器库存包含 `MerchantCardRemovalEntry`；
2. `FullRunSimulationStateBuilder.BuildShopLegalActions` 把该 entry 与卡牌、遗物、药水一样枚举为
   `shop_purchase(index)`；
3. `FullRunSimulatorRuntimeFacade` 将该 index 交给 `TryPurchaseShopItemAsync`，并明确处理会打开
   card selection 的异步购买；
4. Python transport 保留稀疏 index，并原样派发
   `{"action": "shop_purchase", "index": ...}`；
5. 后继选择页公开 `operation_type = remove`，确认后牌组数实际减少；
6. 历史 compact journal 的 selected action 也保留了 `item.category = card_removal`。

对保留下来的三个**非缓存主评估 trajectory**重新按“确认后的下一决策必须观察到牌组减少”统计：

| 指标 | 数量 |
|---|---:|
| card-removal purchase 尝试 | 57 |
| 实际完成 | 8 |
| 取消 | 44 |
| 未决/无法由后继状态确认 | 5 |

完成率为 `8 / 57 = 14.0%`，取消率为 `44 / 57 = 77.2%`。因此真实问题不是动作不可执行，
而是**可执行动作长期缺少稳定的策略侧身份，且模型频繁购买后取消**。例如保存的
`v27-paired-top2-64-base1200000-cpu2-r2` 评估中，seed `2400017` 在 step 1055 选择
card-removal purchase，step 1056 选择卡牌，step 1057 确认，step 1058 的 `deck_count`
从 37 降到 36；这是一条已经发生过的完整成功链。

### E2.2 根因与正式修复

原生 HeadlessSim JSON 的权威字段是 `item.category`。恢复后的 simulator source/binary 不保证应用
维护补丁中 `category -> type` 的镜像，而 grounded canonicalizer 过去只接受
`type/item_type/item_kind/kind`。结果是动作仍能执行、也能写入 journal，但策略候选会把
`card_removal` 丢成匿名 shop item，只能靠价格、槽位等偶然特征学习。

本次正式修复为：

1. grounded canonical shop item 直接接受权威 `category`，生成稳定 item type 与
   `purchase_card_removal` transaction；
2. transport 回归覆盖 purchase 稀疏 index 的原样 RPC 派发，以及后继 `remove` 选择/确认页；
3. compact journal 新增 `legal_action_semantics`，同时记录
   `shop_purchase:card_removal` 等参数化动作，而不是只记录顶层 model kind；
4. greedy evaluation 新增 `legal / attempt / completed / cancelled / unresolved` 五类计数；
   `completed` 必须由确认后的实际 `deck_count` 下降证明，不能把打开页面或点击确认冒充成功；
5. 保留原有 `card_removal_cancel_cycle_episode_count`，继续识别购买→选择→取消循环。

旧 compact journal 继续可读，并可从 selected action 与后继 `deck_count` 回算
attempt/completed/cancelled/unresolved；但旧记录没有完整 legal candidate subtype，不能伪造历史
`legal` 暴露次数，该项只从本次修改后的新 journal 开始具备权威值。

核心回归测试：

- `test_native_shop_card_removal_category_reaches_purchase_and_selection_semantics`；
- `test_shop_card_removal_round_trips_purchase_and_remove_selection`；
- `test_trajectory_summary_keeps_parameterized_shop_action_surface`；
- `test_card_removal_telemetry_requires_observed_deck_mutation`；
- `test_three_step_shop_removal_cancel_cycle_is_a_liveness_failure`。

### E2.3 后续动作面审计准则

任何动作只有同时通过以下六层才可宣称“模型可学”：

1. **枚举资格**：当前状态下游戏/模拟器确实将其列为 legal；
2. **传输保真**：参数、稀疏 index、目标与 subtype 不被重编号或擦除；
3. **候选可辨识**：encoder 能区分具有不同后果的参数化动作；
4. **执行可达**：dispatch 调到游戏真正的 mutation handler；
5. **事务闭合**：多步选择、确认、取消都有明确后继语义；
6. **结果可观测**：journal 能用后继状态证明操作完成，而不是只记录意图。

本次静态对照还确认当前 `FullRunSimulationStateBuilder` 中 19 个字面 legal action enum 全部存在
Python fail-closed model mapping，缺失数为 0；但这不等价于未来版本的永久完备性。以后升级
STS2/HeadlessSim 时必须重跑上述六层对照，并同时比较嵌套 item/option category，禁止再次只枚举
顶层 action kind 后宣称动作不存在。

---

## F. 吞吐边界（与 T1～T3 计划对齐）

v33 只实施稳定性修复，不把吞吐实验混入主血统。后续优先级仍是：

1. liveness replay 批处理化，消除逐记录重放与 Python 锁内扫描；
2. `torch.compile`，但先固定 active-shape bucket，防止重编译；
3. learner 提速后再增加 actor；当前 learner 是瓶颈时，多 actor 只会增加排队；
4. BF16 已在 T1 预实验中出现前向变慢，不作为默认方案；
5. unroll 16→64 属于学习配方实验，不与纯工程吞吐优化混淆。

详细方案见：

- `docs/architecture/liveness-replay-optimization-v1.md`
- `docs/architecture/t1-t3-execution-plan-v1.md`

---

## G. 验证记录

### G1. 最终本地验证（2026-08-04）

在 WSL ROCm 环境、`packages/rl-agent` 工作目录执行：

```bash
PY=/mnt/e/game/project/sts2_mcp_artifacts/runtime/environments/wsl-rocm/bin/python
$PY -m ruff check .
$PY -m mypy sts2_rl sts2_baseline
$PY -m pytest -q
```

结果：

```text
All checks passed!
Success: no issues found in 82 source files
1120 passed in 212.91s
```

全量测试覆盖除了 v33 聚焦测试外，还包括旧 checkpoint 兼容、训练 pipeline、failure-credit、
dashboard/monitoring、headless client、宏观评估、seed 地图复现及其他当前工作区功能。

### G2. 启动预检

已单独完成并通过：

- source checkpoint 路径、run/checkpoint ID、step、policy/update 与两份 SHA-256 的精确校验；
- `sts2_rl.train --initialize-from <pinned checkpoint> --dry-run`；
- 命令语义检查：存在 `--initialize-from`，不存在 `--resume`；
- supervisor 重新进入 v33 adapter 后仍保留 v33 pins。

完整 `initialize-preflight` 还包含“Git checkout 必须干净且已提交”的发布纪律检查。当前工作区包含本次
实现及此前监控改造的未提交文件，因此它按设计拒绝启动；这不是 checkpoint 或训练配置预检失败。
正式启动前应先审查并提交目标改动，再在 clean checkout 执行：

```bash
python scripts/launch_v33_stability_recovery_model_init.py initialize-preflight
python scripts/launch_v33_stability_recovery_model_init.py initialize
```

### G3. 已知尚未完成的证据

- 没有启动 v33 的 100k 在线训练；
- 没有生成 0/5k/10k/25k/50k/75k/100k held-out 结果；
- 因此 C 节中的胜率、熵、锻造和 deadlock 标准仍是门禁，不是结论。

---

## H. 实施纪律

1. 每项修改遵循：聚焦测试 → 全量 pytest / Ruff / strict mypy → pinned-source dry-run；
2. config 升级到 v13，未知字段、显式非法 checkpoint role 与身份不一致继续 fail-closed；
3. 系数全部进入版本化配置或版本化 learner dynamics，不用临时 CLI 旗标；
4. ROCm 在线结果与合成/单元测试分开报告；
5. 不引入卡牌、事件或 boss 名称特判；
6. checkpoint 目录名不携带权威健康语义，受哈希保护的 metadata role 才是权威；
7. v33 未通过 C 节前，不启动 reward v6、budget 48、unroll 64 或扩大模型等第二变量。
