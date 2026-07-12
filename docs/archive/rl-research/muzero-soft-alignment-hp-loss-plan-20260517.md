# MuZero Soft Alignment + HP Loss Plan — 2026-05-17

目标：把当前 Act1 卡住的问题从“继续叠 hard guard”切回可学习的 soft signal：战斗少掉血、构筑/商店/奖励对齐、人类逐动作 demo 对齐。本文是 2026-05-17 的执行口径，优先级高于 2026-05-16 旧文中的高权重建议。

## 0. 结论

你的判断是对的：当前 hard guard 已经太多，继续靠 override 会产生三个副作用：

1. **污染 policy target**：训练看到的是 guard 改写后的动作，模型学不到原始局面为什么该这么做。
2. **碎片化策略**：每个怪/卡/场景都有一条规则，组合场景里互相打架。
3. **掩盖主因**：combat 没有稳定惩罚 HP loss 时，模型只学“赢/输”和短期 value，普通战斗会多掉血，导致后面没血、没篝火升级、进 boss 前质量差。

下一轮主线要改成：

- 开启 **combat HP-preservation auxiliary loss**；
- human/demo 只在有 **obs + legal_actions + selected_action_id** 时做 action CE；
- snapshot/human_zip 只做 deck/survival/HP calibration，不能当动作标签；
- offline build/shop/reward/rest 先做 shadow/audit，再小权重接入专用对齐，不直接污染 combat policy；
- hard guard 分层，只保留协议/生存必需，行为类逐步降成 telemetry + loss + replay priority。

## 1. 当前代码真实状态

### 1.1 HP-loss / HP-preservation 不是没代码，而是默认没开

已存在 flag：

```text
packages/rl-agent/muzero/training/cli_args.py
--combat-hp-preservation-aux-weight
```

默认：

```text
0.0
```

训练里已经接入：

```text
packages/rl-agent/muzero/training/train_step.py
loss/combat_hp_preservation_aux
combat_hp_preservation/mae
combat_hp_preservation/pred_mean
combat_hp_preservation/target_mean
combat_hp_preservation/active_rate
combat_hp_preservation/aux_weight
combat_hp_preservation/loss_applied
```

因此之前 full-run 实际等价于：**HP-preservation soft loss 关闭**。

下一轮启动脚本已准备：

```text
packages/rl-agent/launch_soft_hp_alignment_fullrun_20260517.sh
```

关键新增：

```bash
--combat-hard-guard-policy emergency
--build-hard-guard-policy emergency
--combat-hp-preservation-aux-weight 0.05
# 不启用 --route-safety-guard
```

为什么不是直接 0.15/0.25：当前模型已经有“非攻击意图乱防 / 无效防御 / 空过”问题，HP loss 权重过大可能把模型推得更保守。先用 0.05 验证方向，观察 normal/elite hp loss 和 no-pressure block，再考虑 0.08/0.10。

为什么先关 `--route-safety-guard`：这轮要验证“soft HP loss + emergency guard”本身，不让路线 hard override 成为隐藏变量。若后续需要路线保护，应做成单独的 route emergency policy，并用 `route_safety_guard_override_rate` 证明它只在极少数必死/低血强制精英场景介入。

### 1.2 Human demo policy alignment 代码有，但有效逐动作 demo 当前为空

代码路径已存在：

```text
packages/rl-agent/muzero/demo_dataset.py
packages/rl-agent/muzero/training/human_demo_alignment.py
```

但当前真实文件：

```text
packages/rl-agent/human_demos/*/decisions.jsonl
```

都是 0 字节。

所以结论是：

```text
human-demo alignment 代码 ready；当前 action-level demo 数据不 ready。
```

不能现在打开：

```bash
--human-demo-alignment-enable-loss
```

否则不是“利用人类样本”，而是用空/错数据制造噪声。

### 1.3 Snapshot / human_zip 不能做 combat action CE

已有 snapshot/human_zip 字段主要是：

```text
floor / act / encounter / hp / deck_before / relics_before / potions_before / outcome
```

缺少：

```text
obs
ordered legal_actions
selected_action_id
selected_action_index
turn action sequence
```

因此它不能监督“这一手该打哪张牌/打谁/是否结束回合”。正确用途是：

- combat sandbox 初始状态分布；
- HP loss / win / turns 的 outcome calibration；
- 死亡卡组质量分析；
- encounter curriculum；
- long-horizon survival/deck quality target。

## 2. 为什么之前没直接用人类样本对齐

不是不该用，而是数据类型不同，不能混用。

| 数据 | 能不能做 combat policy CE | 正确用途 |
|---|---:|---|
| action-level human demo：obs + legal_actions + selected_action_id | 可以 | 战斗逐动作行为克隆、target priority、药水时机、end turn 纠偏 |
| combat snapshot：战前状态 + outcome | 不可以 | sandbox 分布、HP loss/value/outcome、死亡卡组诊断 |
| human_zip/full-run summary | 不可以 | deck/survival/remaining-floor calibration |
| offline build/shop/reward/rest normalized tasks | 部分可以 | 选牌、删牌、商店、休息/强化的候选集排序对齐 |
| offline route task | 暂不直接用 | route action index 对齐未完全可靠，先 shadow |

核心风险：如果把 snapshot 当 action label，模型会学到假的动作监督，比不用更坏。

## 3. Offline 对齐现在能用哪些

已审计的 offline action alignment：

```text
packages/rl-agent/tmp/offline_action_alignment_report.json
```

当前可优先使用的任务：

| task | 状态 | 用途 |
|---|---|---|
| regular_card_reward | ready | 选牌 / skip 对齐，解决连续不拿牌 |
| rest_action | ready | rest/smith 对齐 |
| smith_target | ready | 强化目标 |
| remove_card_step | ready | 删牌目标 |
| shop_remove_binary | ready | 是否删牌 |
| shop_remove_target_step | ready | 删哪张牌 |
| shop_relic_pick_step | ready | 买遗物/跳过 |
| shop_potion_pick_step | ready | 买药/跳过 |
| route_room_type / route_point_type | not ready | 暂不 CE，先检查 route action alignment |

短期接入方式：

1. 先 shadow：只跑 top1_match、label_rank、candidate_count、skip/no-buy 分布；
2. 再小权重：只接 build/shop/reward/rest 专用对齐，不接 combat root；
3. route 等 action index 完整对齐后再考虑；
4. 所有 offline loss 必须带 `enable-loss` 开关，默认 off。

## 4. Hard guard 分层方案

### 4.1 继续保留 hard 的范围

这些仍然该 hard：

- illegal action mask；
- bridge/protocol recovery；
- invalid target；
- selection loop 防死循环；
- 明确 self-lethal emergency；
- 倒计时 0/1 这种机制必死窗口；
- 极窄的 urgent rescue，例如没有争议的可斩杀/必死逃离。

### 4.2 应降级为 soft / telemetry 的范围

这些不应该长期 hard override：

- no-pressure block；
- meaningful damage end-turn；
- strategic skip；
- refund no-followup；
- hp-cost low margin；
- late-normal Act1 hardcode；
- potion bad-use 复合规则；
- card reward skip hard threshold；
- shop buy/remove 强规则；
- route unsafe elite hard bias。

降级方式：

```text
hard override -> telemetry flag -> replay priority -> auxiliary loss / ranking loss -> only emergency hard fallback
```

## 5. 今晚可执行主线

### Phase A：立刻恢复 full-run，但加 HP soft loss

使用：

```bash
cd /mnt/e/game/project/sts2_mcp/packages/rl-agent
bash launch_soft_hp_alignment_fullrun_20260517.sh
```

关键保证：

```bash
--combat-hp-preservation-aux-weight 0.05
# 不启用 --human-demo-alignment-enable-loss
# 不把 offline route CE 接进主策略
```

### Phase B：监控是否真的改善战斗质量

必须看：

```text
combat_hp_preservation/loss_applied = 1
combat_hp_preservation/aux_weight = 0.05
combat_hp_preservation/mae 是否下降
combat_hp_preservation/active_rate 是否非 0
normal_hp_loss_mean 是否下降
elite_hp_loss_mean 是否不恶化
boss_hp_loss_mean 是否下降
combat_quality_hard_guard_override_any 是否下降
no_pressure_block_guard_applied 不应上升
full_energy_endturn_selected 不应上升
summoner_targeting_error / source-vs-summon 错误不应上升
```

### Phase C：构筑/商店/奖励用 offline 对齐，但不碰 combat root

优先目标：

```text
card_reward/skip_rate 下降
card_reward/consecutive_skip_count 下降
death_deck/deck_size 不再过小
shop/open_rate 上升
shop/buy_rate 上升
shop/remove_rate 上升
rest/smith 合理分布
```

## 6. 录制人类 combat demo 的最低 schema

如果要真正“根据游戏内操作录制”，每个动作行必须至少有：

```json
{
  "obs": {"...": "完整 MuZero 输入前的 raw obs 或可重建 obs"},
  "legal_actions": [
    {"action_id": "...", "kind": "play_card", "card_id": "...", "target": 0},
    {"action_id": "...", "kind": "end_turn"}
  ],
  "selected_action_id": "...",
  "selected_action_index": 0,
  "surface": "combat",
  "encounter_id": "...",
  "floor": 11,
  "turn": 2,
  "step_in_turn": 3
}
```

战斗结束后回填：

```json
{
  "combat_win": true,
  "hp_loss": 6,
  "turns": 4,
  "death": false
}
```

没有这些字段，只能做 outcome/value calibration，不能做 action CE。

## 7. 验收线

短期 2-3 小时内不要求直接 pass Act1，但必须看到方向：

```text
loss/combat_hp_preservation_aux 有效非 0 且无 spike
combat_hp_preservation/mae 下降或稳定
normal combat hp loss 均值下降
normal/elite win 不继续掉
hard_guard_override_any 不再继续升
card reward skip 不继续升
death_deck_size 不再长期 < 14-16
act1_boss_seen_rate 回升
```

如果 HP loss 下降但 no-pressure block / end_turn 变差，说明 0.05 仍偏保守或 guard 互相污染，需要先降 behavioral hard guard，而不是继续加 HP 权重。

## 8. 当前已改文件

```text
packages/rl-agent/launch_soft_hp_alignment_fullrun_20260517.sh
packages/rl-agent/launch_soft_hp_alignment_combat_sandbox_20260517.sh
```

新增目的：启动新 full-run 时默认打开低权重 HP-preservation aux，combat/build hard guard 降到 emergency，关闭 route safety hard override，并明确不启用空的人类 demo CE。

`launch_soft_hp_alignment_combat_sandbox_20260517.sh` 是 full-run 前的战斗修复入口：

- 使用 `datasets/curated_combat_ironclad_mixed_provenance` 作为 **战斗前 snapshot 分布**；
- 默认 subset 为 `bootstrap_human_plus_local_all_minus_combat_reset_failures`，不再只使用 `roomwin_only`；
- 使用 `tier_weighted_encounter_balanced`，默认 `weak=0.45,normal=1.00,elite=2.20,boss=2.80`，补足 full-run 里 elite/boss 稀疏问题；
- 继承 pass-large 规格：`d_model=192`、`memory_slots=24`、`pass_large_v1` slot layout、`bank_top_k=7`；
- 打开 `--combat-hp-preservation-aux-weight 0.05`；
- 使用 `--combat-hard-guard-policy emergency` / `--build-hard-guard-policy emergency`；
- 明确 `DATASET_USE=curated_snapshot_distribution_only__no_action_ce`，避免把 snapshot 数据误当逐动作 human BC。

启动前必须确认 bridge/game 不是 stale session。若 bridge 未启动，此脚本不应作为后台训练直接启动。
