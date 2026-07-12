# MuZero Human/Offline Alignment Plan — 减少 hard guard，补 combat 战损信号

日期：2026-05-16
目标：把“少掉血、正确构筑、会逛商店、能过 Act1”从硬规则改成可学习信号。

## 1. 结论

当前继续堆 hard guard 会让策略碎片化：每个机制都被外部规则接管，模型本体仍然学不会“为什么这样做”。
下一阶段应改为三类训练信号：

1. **Combat HP-preservation loss**：让战斗模型显式学习“同样赢战斗，少掉血更好”。
2. **Offline build/route/shop/reward alignment**：用历史/人类 run 记录对齐选牌、删牌、商店、休息/强化、路线。
3. **Combat action-level human demo**：只有在有逐动作记录时，才做战斗行为克隆；不能把战斗前快照误当逐动作标签。

## 2. 数据源分工

### 2.1 curated combat snapshot 数据

路径：

```text
datasets/curated_combat_ironclad_mixed_provenance/combined/bootstrap_human_plus_local_all_minus_combat_reset_failures.jsonl
```

已审计：这是**战斗前状态快照**，字段包括 deck/relic/potion/hp/encounter/floor/outcome 等；不包含：

```text
selected_action_id
selected_action_index
legal_actions
turn_actions
action_history
```

因此它不能直接做 combat action BC。正确用途：

- 初始化 combat sandbox 分布；
- 统计战斗胜率、HP loss、turns；
- 训练/校准 HP-preservation、survival、combat value；
- 分 encounter 做 curriculum，例如召唤类、Boss、低血量残局。

### 2.2 raw history / offline build v2

路径：

```text
tmp/offline_build_v2_full/parquet
```

当前可用规模：

```text
runs_summary:        86
floor_records:       3215
decision_records:    5273
route_samples:       3010
card_choice_samples: 1434
build_samples:       3755
build_v2_total:      5739
```

适合做以下对齐：

- card reward 选牌 / skip；
- shop buy / remove；
- smith/rest；
- route choice；
- relic/potion/event choice；
- 死亡时 deck quality 回归分析。

这是解决“不拿牌、不会购物/删牌、卡组质量低、路线不稳”的主要数据源。

### 2.3 human demo recorder

现有 schema 已能承接逐动作 demo，但当前 demo 文件为空。
只有当录制器实际产出：

```text
obs + legal_actions + selected_action_id/selected_action_index
```

时，才可以对 combat 打牌做 BC。否则会把错误标签塞进模型，风险比不用更大。

## 2.4 当前实现状态（2026-05-16）

已完成第一批低风险 plumbing：

- `muzero/demo_dataset.py`
  - demo batch 增加 `bc_target_policy`
  - 增加 `target_hp_loss / combat_win / turns` 及 mask
- `muzero/training/train_step.py`
  - 增加 `--combat-hp-preservation-aux-weight`
  - 默认 `0.0`，不影响旧 run
  - 建议首次训练用 `0.15`，稳定后再看是否升到 `0.25`
- `muzero/training/offline_alignment.py`
  - 新增 offline build/route/shop/reward 对齐数据模块
  - 只负责加载、审计、collate、summary
  - 暂时不直接改 MuZero policy，避免 action index 未验证就污染训练
- `offline_training_data.py`
  - 修复 build-v2 candidate task 的 `choice` vocab 构建，否则 `regular_card_reward` 等任务无法 collate

测试：

```text
WSL ROCm: tests/test_offline_alignment.py + tests/test_human_demo_dataset.py = 15 passed
Windows no-torch smoke: 12 passed, 1 skipped
```

真实 `tmp/offline_build_v2_full/parquet` 首轮审计：

```text
regular_card_reward          rows=1388
smith_target                 rows=317
remove_card_step             rows=390
shop_remove_binary           rows=265
shop_remove_target_step      rows=185
shop_relic_pick_step         rows=278
shop_potion_pick_step        rows=268
rest_action                  rows=568
route_room_type              rows=3010
route_point_type             rows=3010
total                        rows=9679
```

重要发现：

- `regular_card_reward / shop_*_pick_step` 中，`<skip>` 候选目前主要只在“实际 skip/不买”样本中出现。
- 这说明不能立刻把这些数据 CE 到 MuZero policy；否则模型会学到错误候选集合。
- 下一步必须先做 **candidate/action alignment audit**：
  - 线上 legal action 是否包含 skip；
  - offline candidate 是否也包含 skip；
  - label index 是否和 policy logits index 一致；
  - route candidate index 是否和 map action index 一致。

结论：数据可以用，但要先通过 adapter/audit，再小权重接入；不能直接粗暴 BC。

## 3. 已落地的低风险改动

### 3.1 demo batch 扩展

`muzero/demo_dataset.py` 已支持：

```text
bc_target_policy
target_hp_loss / target_hp_loss_mask
combat_win / combat_win_mask
turns / turns_mask
```

缺失 outcome 会被 mask 掉，不会被当作 0 学。

### 3.2 Combat HP-preservation auxiliary loss

新增训练开关：

```text
--combat-hp-preservation-aux-weight
```

默认 `0.0`，不影响旧训练。开启后复用现有 objective reward head：

```text
HEAD_HP_PRESERVATION
```

不新增网络 head，不改 checkpoint/model schema，风险低。

建议初始值：

```text
--combat-hp-preservation-aux-weight 0.15
```

如果 loss 稳定且 HP loss 仍高，再尝试：

```text
--combat-hp-preservation-aux-weight 0.25
```

## 4. 不再优先增加 hard guard 的原则

只保留以下 hard guard：

1. 非法动作；
2. 明确自杀；
3. 协议/桥接错误；
4. 已知游戏机制必死窗口，例如倒计时为 0 的不可逆死亡。

其他行为问题应优先变成训练信号：

| 问题 | 推荐处理 |
|---|---|
| 普通战斗掉血太多 | HP-preservation aux loss + HP loss metric |
| 非攻击意图时乱防 | avoidable damage / overblock metric + supervised alignment |
| 满能量空过 | bad end-turn metric + replay priority，不优先写 override |
| 召唤类怪物打错目标 | encounter-specific outcome/value + target-priority demo |
| 不拿卡 | offline card reward BC + deck quality target |
| 不进商店/不删牌 | offline shop/remove BC |
| 路线差 | offline route BC + successor value，不直接硬 bias |

## 5. 下一步实施顺序

### Phase A：开启 HP-preservation loss

目的：先让 combat 学“少掉血”。
训练命令增加：

```text
--combat-hp-preservation-aux-weight 0.15
```

必须观察：

```text
loss/combat_hp_preservation_aux
combat_hp_preservation/mae
combat_hp_preservation/pred_mean
combat_hp_preservation/target_mean
combat_hp_preservation/active_rate
normal_hp_loss_mean
elite_hp_loss_mean
boss_hp_loss_mean
```

### Phase B：接 offline build/route/shop/reward alignment

新增独立模块，避免继续膨胀 `train.py`：

```text
packages/rl-agent/muzero/training/offline_alignment.py
```

默认关闭，建议 CLI：

```text
--offline-alignment-path tmp/offline_build_v2_full/parquet
--offline-alignment-weight 0.0
--offline-alignment-batch-size 0
--offline-alignment-tasks regular_card_reward,shop_remove_target_step,smith_target,rest_action,route
```

初期只做 shadow/readout 和小权重 CE，不直接改策略选择。

### Phase C：修 combat human demo recorder

目标：让人工游戏内操作能记录为逐动作 demo。

每行至少包含：

```text
obs
legal_actions
selected_action_id 或 selected_action_index
encounter_id
floor
turn
step
```

战斗结束后回填：

```text
hp_loss
combat_win
turns
```

之后才能对 combat policy 做真正的 human BC。

## 6. 验收线

短期不是只看 win rate，而是看行为质量是否恢复：

```text
combat_hp_preservation/mae 下降
normal_hp_loss_mean 下降
elite_hp_loss_mean 不上升
boss_hp_loss_mean 下降
reward_skip_rate 下降
shop_open_rate / shop_buy_rate / remove_rate 上升
death_deck_size 不再长期偏小
deck_quality atk_per_energy/block_per_energy/draw_per_turn 改善
act1_boss_seen_rate 回升
act1_pass_rate > 0
```

## 7. 风险

1. curated combat snapshot 不能假装是 combat BC；否则会制造错误标签。
2. HP-preservation 权重过大可能导致模型过度防御；从 0.15 开始，不要一上来拉满。
3. offline build/route 样本要做 action index 对齐测试；否则会重演 route/action alignment 污染。
4. 新功能默认关闭，确认指标正常后再进入主训练。
