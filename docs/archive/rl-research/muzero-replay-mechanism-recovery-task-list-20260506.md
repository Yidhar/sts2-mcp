# MuZero Replay 分布与机制恢复任务清单（2026-05-06）

> 目的：把 2026-05-06 最新训练日志暴露的问题拆成可执行、可验收、可回滚的任务，避免继续盲训导致 normal / elite 遗忘、boss 机制反复失效。
> 适用范围：`packages/rl-agent` MuZero 训练、replay scheduler、combat action quality bias、STS2 bridge action/runtime payload、diagnostics、固定评估脚本。
> 当前结论：**这不是 loss 爆炸问题，而是 replay 采样分布偏移 + Kaiser/Insatiable 关键机制未 hard guard + 药水/X 费/扣血回费等低质量动作仍污染策略。**
> 建议训练策略：不要默认使用 latest checkpoint。先固定 eval 选 best checkpoint，再完成 P0 修改后重新正式训练。

---

## 0. 当前最新状态摘要

### 0.1 当前 run

```text
packages\rl-agent\logs_muzero\muzero_boss_recovery_20260505_162209_post_dll_fix
```

最新已知状态：

```text
optimizer step       ≈ 673,784
episode              ≈ 21,812
buffer               ≈ 99,964 / 100,000
latest checkpoint    packages\rl-agent\checkpoints_muzero\muzero_boss_recovery_20260505_162209_post_dll_fix\muzero_step_00673808
```

### 0.2 关键指标

RecentTail 64：

| 指标 | latest | avg20 | avg100 | 判断 |
|---|---:|---:|---:|---|
| `recent_tail/64/win_rate` | 0.4375 | 0.4148 | 0.4884 | 短窗偏弱 |
| `recent_tail/64/boss_win_rate` | 0.3171 | 0.2963 | 0.3757 | boss 仍弱 |
| `recent_tail/64/normal_win_rate` | 0.5714 | 0.5851 | 0.7848 | normal 明显回落 |
| `recent_tail/64/elite_win_rate` | 0.6875 | 0.6926 | 0.7494 | elite 回落 |
| `recent_tail/64/reward_mean` | -5.3356 | -5.8873 | -4.4198 | reward 不佳 |

RecentTail 256：

| 指标 | latest | avg20 | avg100 | 判断 |
|---|---:|---:|---:|---|
| `recent_tail/256/win_rate` | 0.4805 | 0.4711 | 0.4790 | 低于 run 内高点 |
| `recent_tail/256/boss_win_rate` | 0.3626 | 0.3491 | 0.3577 | 低于 run 内高点 |
| `recent_tail/256/normal_win_rate` | 0.7576 | 0.7584 | 0.7829 | 从高点明显下降 |
| `recent_tail/256/elite_win_rate` | 0.6923 | 0.7024 | 0.7115 | 小幅下降 |
| `recent_tail/256/reward_mean` | -4.5892 | -4.8464 | -4.6720 | 低于 run 内高点 |

当前 run 内高点对比：

| 指标 | run 内最佳 | latest | 退化 |
|---|---:|---:|---:|
| `recent_tail/256/win_rate` | 0.5234 | 0.4805 | -4.29 pp |
| `recent_tail/256/boss_win_rate` | 0.4068 | 0.3626 | -4.42 pp |
| `recent_tail/256/normal_win_rate` | 0.9048 | 0.7576 | -14.72 pp |
| `recent_tail/256/reward_mean` | -3.6676 | -4.5892 | 恶化 |

### 0.3 per-encounter 当前瓶颈

| encounter | latest 256 win | avg20 | sample | 优先级 |
|---|---:|---:|---:|---|
| `kaiser_crab_boss` | 0.1026 | 0.0965 | 39 | P0 |
| `knowledge_demon_boss` | 0.1364 | 0.1632 | 22 | P2 先诊断 |
| `the_insatiable_boss` | 0.3000 | 0.2895 | 30 | P0 |
| `ceremonial_beast_boss` | 0.4694 | 0.4447 | 49 | P1 |
| `the_kin_boss` | 0.7419 | 0.7509 | 31 | 暂不加强 |
| `soul_nexus_elite` | 0.5000 | 0.4935 | 24 | P2 先诊断 |
| `slumbering_beetle_normal` | 0.4286 | 0.3662 | 7 | P2 先诊断 |

### 0.4 loss 结论

当前 loss 稳定，无近期数值爆炸迹象：

```text
loss/total                    latest 11.074 avg20 11.066 avg100 11.385
loss/future_world_aux         latest 0.355  avg20 0.330  avg100 0.332
loss/future_bank_state        latest 0.007  avg20 0.013  avg100 0.013
loss/future_bank_delta        latest 0.016  avg20 0.0145 avg100 0.0149
loss/future_bank_token_source latest/avg = 0
```

结论：

- [ ] 不把当前问题归因于 optimizer / AMP / loss spike。
- [ ] 当前主要问题按行为与数据分布处理。

### 0.5 offender 排名

last 5000 `action_offenders.jsonl`：

| offender | count | 判断 |
|---|---:|---|
| `kaiser_facing_missed` | 1404 | Kaiser 最大确定性机制失败 |
| `ceremonial_missed_stun_window` | 945 | 需要先修 detector 口径 |
| `low_quality_potion_selected` | 890 | 药水时机错误 |
| `kaiser_risky_end_turn` | 812 | Kaiser 风险下仍结束回合 |
| `ceremonial_low_impact_under_lock` | 361 | Ceremonial one-card lock 策略不足 |
| `high_save_value_potion_unused` | 224 | 药水保存/使用校准仍矛盾 |
| `insatiable_frantic_escape_missed_lt3` | 123 | Insatiable countdown 机制未 hard force |
| `zero_energy_x_cost_selected` | 69 | X 费动态能量仍有漏洞 |
| `insatiable_frantic_escape_missed_at_1` | 43 | countdown=1 仍会错过逃离 |

---

## 1. 全局执行规则

### 1.1 代码修改前必须检查

- [ ] 查看 dirty diff，避免覆盖已有修改：

```powershell
git status --short
git diff --stat
```

- [ ] 搜索已有实现，避免重复写平行逻辑：

```powershell
rg -n "ReplayScheduler|sample_boss_rate|sample_elite_rate|sample_normal_rate|replay-encounter-tier-weights|_combat_action_quality_bias|kaiser|insatiable|frantic|x_cost|hp_cost|potion|offender|action_offenders" packages mods -S --glob "!**/logs_muzero/**" --glob "!**/bin/**" --glob "!**/obj/**"
```

### 1.2 不允许继续走的错误路线

- [ ] 不允许只调 learning rate / loss weight 来解决当前问题。
- [ ] 不允许只看 global win，不看 per-encounter 与 offender。
- [ ] 不允许继续默认使用 latest checkpoint。
- [ ] 不允许 replay 继续让 boss 样本超过 90%。
- [ ] 不允许把 Kaiser / Insatiable 的即时死亡机制继续完全交给 policy 学。
- [ ] 不允许用文本正则作为 hard safety 的主依据；文本只能做 fallback diagnostic。

### 1.3 每个任务完成时必须交付

- [ ] 修改文件列表。
- [ ] 新增/修改字段列表。
- [ ] 新增/修改 TensorBoard 指标列表。
- [ ] 新增/修改 diagnostics 输出列表。
- [ ] 单元测试 / fake obs 测试 / 最小 smoke 说明。
- [ ] 是否影响 replay buffer schema；如果影响，更新 `schema_version`。
- [ ] 如果有 hard guard，必须有 guard applied / override / exemption 指标。

---

# P0 — 正式继续训练前必须完成

---

## P0-1. 固定 eval 选择 best checkpoint，不再默认 latest

### 问题

当前 latest checkpoint 不一定最佳：

```text
latest 256 win_rate       0.4805
best   256 win_rate       0.5234
latest boss_win_rate      0.3626
best   boss_win_rate      0.4068
latest normal_win_rate    0.7576
best   normal_win_rate    0.9048
```

继续从 latest resume 可能把已经退化的策略固化。

### 目标

建立固定评估流程，比较多个 checkpoint，选出综合分最高的 checkpoint 作为下一次正式训练起点。

### 候选 checkpoint

- [ ] 至少评估以下 checkpoint：

```text
muzero_step_00667650
muzero_step_00669705
muzero_step_00671783
muzero_step_00673808
```

- [ ] 如果 checkpoints 目录里存在更早的 0065xxxx 或 0066xxxx 稳定点，补充 1-2 个。

### 固定 eval 集

每个 checkpoint 使用相同 seed 集。

Boss：

| encounter | episodes |
|---|---:|
| `kaiser_crab_boss` | 64 |
| `knowledge_demon_boss` | 64 |
| `the_insatiable_boss` | 64 |
| `ceremonial_beast_boss` | 64 |
| `the_kin_boss` | 64 |

Elite：

| encounter | episodes |
|---|---:|
| `soul_nexus_elite` | 32 |
| `knights_elite` | 32 |
| `phrog_parasite_elite` | 32 |

Normal：

| encounter | episodes |
|---|---:|
| `slumbering_beetle_normal` | 32 |
| `construct_menagerie_normal` | 32 |
| `ovicopter_normal` | 32 |

### 评分公式

固定 eval 后输出综合分：

```text
score =
  0.35 * boss_win
+ 0.20 * elite_win
+ 0.15 * normal_win
+ 0.15 * normalized_reward
- 0.15 * mechanic_offender_score
```

`mechanic_offender_score` 至少包含：

```text
kaiser_facing_missed
kaiser_risky_end_turn
insatiable_frantic_escape_missed_at_1
insatiable_frantic_escape_missed_lt3
ceremonial_missed_stun_window_true
low_quality_potion_selected
zero_energy_x_cost_selected
hp_cost_low_margin_selected
```

### 主要文件/脚本

- [ ] `packages/rl-agent/muzero/train.py`
- [ ] `packages/rl-agent/muzero/eval*.py` 或现有 eval launcher。
- [ ] `packages/rl-agent/scripts/*eval*`。
- [ ] 新增或扩展固定 eval 输出：

```text
packages/rl-agent/eval_reports/<run_name>/checkpoint_eval_summary.json
packages/rl-agent/eval_reports/<run_name>/checkpoint_eval_summary.md
packages/rl-agent/eval_reports/<run_name>/best_checkpoint.txt
```

### 验收标准

- [ ] 能对候选 checkpoint 产出同 seed、同 encounter count 的固定评估报告。
- [ ] 报告包含 global / tier / per-encounter win、reward、offender rates。
- [ ] 报告明确给出 `best_checkpoint`。
- [ ] 后续训练命令引用 `best_checkpoint.txt`，而不是 hardcode latest。

---

## P0-2. Replay sampling 改成 batch-level hard quota

### 问题

当前 optimizer 采样极端 boss-heavy：

```text
buffer/sample_boss_rate        avg100 ≈ 0.9550
buffer/sample_elite_rate       avg100 ≈ 0.0381
buffer/sample_normal_rate      avg100 ≈ 0.0069
buffer/sample_hard_normal_rate avg100 ≈ 0
```

这会造成：

1. normal / elite 遗忘；
2. boss 机制过拟合；
3. 通用 combat quality 下降；
4. 总 win 与 reward 从高点回落。

### 目标

把 replay 从软权重改成 batch-level quota。不能只依赖 `--replay-encounter-tier-weights`。

### 当前不可信设置

现有命令中类似：

```text
--replay-encounter-tier-weights weak=0.00,normal=0.70,elite=1.50,boss=3.25
--replay-encounter-weights ENCOUNTER.CEREMONIAL_BEAST_BOSS=6.00,ENCOUNTER.KAISER_CRAB_BOSS=5.60,...
```

这些权重会把 boss 推到 90%+。之后即使继续保留 encounter weights，也必须受到 tier quota 限制。

### 实现要求

假设 batch size = 16，推荐 hard quota：

```text
boss   max 10
elite  min 3
normal min 2
flex   1
```

目标分布：

```text
boss   56% - 62.5%
elite  18.75% - 25%
normal 12.5% - 18.75%
```

如果 batch size 非 16，按比例计算：

```text
boss_target   = 0.60
elite_target  = 0.25
normal_target = 0.15
boss_max      = ceil(batch_size * 0.65)
elite_min     = floor(batch_size * 0.18)
normal_min    = floor(batch_size * 0.10)
```

### 抽样逻辑

- [ ] 先按 tier pool 分桶：

```text
normal_pool
elite_pool
boss_pool
weak_pool
unknown_pool
```

- [ ] 构建每个 batch：

```text
1. 从 normal_pool 抽 normal_min。
2. 从 elite_pool 抽 elite_min。
3. 从 boss_pool 抽 boss_target，但不得超过 boss_max。
4. 剩余位置从 flex pool 抽，按 priority / recency / encounter weights。
5. 如果某 tier pool 不足，允许 fallback，但必须记录 fallback。
```

- [ ] boss 内部仍可使用 encounter weights，但只影响 boss quota 内部分配。
- [ ] normal/elite 不得因为 boss priority 高而被挤掉。
- [ ] 若 pool 不足，记录 `normal_min_unfilled_rate` / `elite_min_unfilled_rate`。

### 主要文件

- [ ] `packages/rl-agent/muzero/replay_scheduler.py`
- [ ] `packages/rl-agent/muzero/train.py`
- [ ] replay buffer / sampler 相关文件。

### 新增指标

```text
buffer/quota_boss_rate
buffer/quota_elite_rate
buffer/quota_normal_rate
buffer/quota_weak_rate
buffer/quota_fallback_rate

buffer/sample_tier_target_boss
buffer/sample_tier_target_elite
buffer/sample_tier_target_normal
buffer/sample_tier_kl_to_target
buffer/sample_tier_l1_to_target

buffer/boss_cap_hit_rate
buffer/normal_min_unfilled_rate
buffer/elite_min_unfilled_rate
buffer/flex_fill_rate
```

### 测试

- [ ] unit test：batch size 16，所有 pool 充足时，boss 不超过 10，elite 至少 3，normal 至少 2。
- [ ] unit test：normal pool 不足时，fallback 被记录。
- [ ] unit test：encounter weights 不得突破 boss tier cap。
- [ ] smoke：训练 500-1000 optimizer step 后 TB 中分布满足目标。

### 验收标准

训练 1 小时内必须看到：

```text
buffer/sample_boss_rate   <= 0.65
buffer/sample_elite_rate  >= 0.18
buffer/sample_normal_rate >= 0.10
```

如果不满足，视为 P0-2 未完成，不进入正式长训。

---

## P0-3. Kaiser facing change hard guard

### 问题

当前 Kaiser 是最大短板：

```text
kaiser_crab_boss win_rate latest ≈ 0.1026
kaiser_facing_change_selected_rate avg20 ≈ 0.026
kaiser_risky_end_turn_selected_rate avg20 ≈ 0.062
```

典型失败：

```text
encounter = kaiser_crab_boss
turn = 6
hp = 7/80
incoming = 26
block = 0
energy = 3
kaiser_back_attack_risk = 1
kaiser_facing_change_candidate_count = 1
selected = 杀灭
之后 selected = End Turn
```

说明：

- [x] back attack risk 已经能进来。
- [x] facing change candidate 已经存在。
- [ ] policy/value 仍不优先处理。

这类机制死亡不应该继续只靠学习。

### facing change 正确定义

Kaiser 转身不是 `enemy.side` / `target.side`。

正确口径：

```text
转身 = 使用 targeted card / potion / operation 指向当前 facing 另一侧的敌人。
```

左右位置从 enemy powers 解析：

```text
BACK_ATTACK_LEFT_POWER
BACK_ATTACK_RIGHT_POWER
```

不要使用：

```text
enemy.side == "left" / "right"
action.target.side == "left" / "right"
```

因为 bridge 实际 side 是敌我方：

```text
Player / Enemy
```

### 实现要求

增加 hard guard：

```text
if encounter == Kaiser
and kaiser_back_attack_risk > 0
and facing_change_candidate exists
and selected action is not facing_change
and selected action is not confirmed_immediate_lethal:
    override selected action to best facing_change candidate
```

如果实现层面暂时不能 override，至少使用强 bias：

```text
facing_change        +6.0 ~ +8.0
risk_handling        +1.5
non_facing_nonlethal -3.0
risky_end_turn       -6.0 ~ -8.0
pressure             只有 confirmed lethal 时才奖励，否则不奖励或很小
```

但推荐优先做 override guard。

### confirmed lethal 例外

允许例外：

```text
如果当前非 facing action 是 confirmed immediate lethal，则不强制转身。
```

要求：

- [ ] `_is_action_confirmed_lethal` 必须保守。
- [ ] 不能把“可能 lethal”当作 confirmed lethal。
- [ ] lethal exemption 必须写 metrics。

### 主要文件

- [ ] `packages/rl-agent/muzero/train.py`
- [ ] `packages/rl-agent/muzero/sts2_env/mcts.py`
- [ ] bridge action payload / raw obs 解析相关文件。
- [ ] diagnostics writer。

### 新增指标

```text
boss_combat/kaiser_risk_with_facing_candidate_rate
boss_combat/kaiser_facing_guard_available_rate
boss_combat/kaiser_facing_guard_applied_rate
boss_combat/kaiser_facing_guard_override_rate
boss_combat/kaiser_facing_guard_lethal_exemption_rate
boss_combat/kaiser_nonfacing_nonlethal_selected_rate
boss_combat/kaiser_end_turn_under_risk_with_candidate_rate
```

per-encounter 命名空间也要 emit：

```text
boss_combat/kaiser_crab_boss/kaiser_facing_guard_override_rate
boss_combat/kaiser_crab_boss/kaiser_risky_end_turn_selected_rate
```

### diagnostics

新增或扩展：

```text
diagnostics/kaiser_facing_guard.jsonl
```

每条记录至少包含：

```json
{
  "episode": 0,
  "turn": 0,
  "hp": 0,
  "incoming": 0,
  "block": 0,
  "energy": 0,
  "current_facing": "left|right|null",
  "selected_action": "...",
  "selected_was_facing_change": false,
  "facing_candidates": [],
  "override_action": "...",
  "lethal_exemption": false,
  "enemy_positions_from_powers": {}
}
```

### 测试

- [ ] fake raw_obs：两个 Kaiser 部位分别带 `BACK_ATTACK_LEFT_POWER` / `BACK_ATTACK_RIGHT_POWER`。
- [ ] 当前 facing=left，target right 部位的攻击牌算 facing_change。
- [ ] 当前 facing=left，target left 部位的攻击牌不算 facing_change。
- [ ] potion 指向另一侧也算 facing_change。
- [ ] self-target / no-target action 不算 facing_change。
- [ ] 有 facing candidate 且 selected 非 lethal 非 facing 时触发 override。
- [ ] confirmed lethal 时不 override，但 emit lethal_exemption。

### 验收标准

```text
kaiser_facing_change_selected_rate >= 0.25
kaiser_risky_end_turn_selected_rate < 0.01
kaiser_facing_missed offender 下降 80%+
kaiser_crab_boss win_rate 从 0.10 至少提升到 0.20+
```

---

## P0-4. Insatiable Frantic Escape hard force

### 问题

用户明确规则：

```text
sandpit countdown = 0 时就是死，不会活着。
countdown = 1 时，只要狂乱逃离 legal，就应该强制。
countdown < 3 时，因为卡牌不一定在手，需要大幅提高预制/打出权重。
```

当前仍存在：

```text
insatiable_frantic_escape_selected_rate avg20 ≈ 0.0176
insatiable_frantic_escape_missed_lt3_rate avg20 ≈ 0.0186
insatiable_frantic_escape_missed_at_1_rate avg20 ≈ 0.0082
```

典型失败：

```text
encounter = the_insatiable_boss
turn = 7
hp = 22/80
incoming = 30
energy = 2
sandpit_countdown = 1
Frantic Escape available
selected = 愤怒
alternative includes 狂乱逃离
```

### 实现要求

#### countdown <= 1

必须 hard force：

```text
if encounter == Insatiable
and sandpit_countdown <= 1
and Frantic Escape legal
and no confirmed_immediate_lethal:
    override selected action to Frantic Escape
```

#### countdown == 2

强 bias：

```text
Frantic Escape        +5.0 ~ +7.0
end_turn              -5.0
non_escape_nonlethal  -3.0
low_impact_action     -2.0
```

如果 Frantic Escape 不在手但存在 draw / cycle / retain / search 能提高摸到概率，则提高这些动作权重。

#### countdown == 3

根据 deck cycle risk 做轻到中等 bias：

```text
if escape not in hand
and draw/discard state indicates low chance to see it before countdown <=1:
    boost draw/cycle/retain/search actions
```

### `_is_frantic_escape_action` 要求

- [ ] 优先使用 card id / internal id。
- [ ] 不得只依赖展示文本。
- [ ] 文本匹配只能作为 fallback diagnostics。

### 主要文件

- [ ] `packages/rl-agent/muzero/train.py`
- [ ] `packages/rl-agent/muzero/sts2_env/mcts.py`
- [ ] action semantic / card identity 相关文件。

### 新增指标

```text
boss_combat/insatiable_frantic_escape_force_available_rate
boss_combat/insatiable_frantic_escape_force_applied_rate
boss_combat/insatiable_frantic_escape_force_override_rate
boss_combat/insatiable_frantic_escape_force_lethal_exemption_rate
boss_combat/insatiable_non_escape_at1_blocked_rate
boss_combat/insatiable_countdown2_escape_selected_rate
boss_combat/insatiable_countdown2_non_escape_selected_rate
boss_combat/insatiable_escape_cycle_risk_mean
```

per-encounter 也要 emit：

```text
boss_combat/the_insatiable_boss/insatiable_frantic_escape_missed_at_1_rate
boss_combat/the_insatiable_boss/insatiable_frantic_escape_force_applied_rate
```

### diagnostics

新增或扩展：

```text
diagnostics/insatiable_escape_guard.jsonl
```

字段至少包含：

```json
{
  "episode": 0,
  "turn": 0,
  "hp": 0,
  "incoming": 0,
  "block": 0,
  "energy": 0,
  "sandpit_countdown": 1,
  "frantic_escape_legal": true,
  "selected_action": "...",
  "override_action": "狂乱逃离",
  "lethal_exemption": false,
  "hand": [],
  "draw_count": 0,
  "discard_count": 0
}
```

### 测试

- [ ] countdown=1，escape legal，selected 非 lethal 非 escape，必须 override。
- [ ] countdown=1，escape legal，但 selected 是 confirmed lethal，不 override，emit lethal exemption。
- [ ] countdown=2，escape legal，bias 后排序应显著提升 escape。
- [ ] countdown=2，escape 不在手，有 draw/cycle action，draw/cycle 获得 boost。
- [ ] countdown=3 时不硬强制，但 emit cycle risk。

### 验收标准

```text
insatiable_frantic_escape_missed_at_1_rate == 0
insatiable_frantic_escape_missed_lt3_rate < 0.005
the_insatiable_boss win_rate > 0.30 且持续上升
```

---

# P1 — 行为质量收紧

---

## P1-1. 药水 type-specific gate

### 问题

当前药水不是“完全不用”，而是：

```text
有就容易用；
低 urgency 也用；
高 save value 偶尔不用；
不同药水类型没有足够区别。
```

典型 bad use：

```text
encounter = the_kin_boss
turn = 3
hp = 71/80
incoming = 17
block = 5
energy = 0
selected = 瓶中精灵
alternative = End Turn
offender = low_quality_potion_selected
```

### 目标

把药水从 generic action quality 改成 type-specific gate。

### 药水分类

至少实现以下类别：

```text
save_life       保命/复活/治疗类，如瓶中精灵
block           格挡/护甲类
damage          直接伤害/毒/易伤窗口类
energy          当前或下回合能量类
draw            抽牌/发现/生成牌类
scaling         力量/敏捷/长期成长类
mechanic        可处理 boss 机制的药水，例如 Kaiser 转身、打断、解机制
low_value       小即时收益/随机低价值药水
unknown         无法分类，谨慎处理
```

### save_life 药水 gate

规则：

```text
if potion.type == save_life
and hp_ratio > 0.35
and incoming_after_block < hp
and no near_term_lethal_forecast:
    hard penalty or ban
```

建议 penalty：

```text
-8.0
```

允许使用：

```text
hp_ratio <= 0.35
or incoming_after_block >= hp
or boss/elite 关键回合且未来 1-2 回合死亡风险高
```

### damage 药水 gate

允许：

```text
confirmed lethal
or 机制窗口，例如 Kaiser facing / Ceremonial stun / Insatiable countdown
or 能显著减少本回合/下回合受伤
or boss/elite 高价值爆发窗口
```

否则：

```text
low urgency damage potion penalty -3.0 ~ -5.0
```

### energy/draw 药水 gate

能量药：

```text
if energy_potion:
    require playable_positive_after_energy > 0
    or X-cost with effective_energy > 0
    or confirmed lethal line
```

抽牌药：

```text
if draw_potion:
    require enough energy after draw
    or zero-cost/card-generation follow-up
    or urgent mechanism out
```

否则 emit：

```text
potion_no_followup_selected
```

### 主要文件

- [ ] `packages/rl-agent/muzero/train.py`
- [ ] action semantic / potion semantic 构建位置。
- [ ] bridge potion payload，如果当前 potion id/type 不足。

### 新增指标

```text
potion/by_id/<potion_id>/selected_rate
potion/by_id/<potion_id>/bad_selected_rate
potion/by_id/<potion_id>/good_selected_rate

potion_low_urgency_selected_rate_by_tier/boss
potion_low_urgency_selected_rate_by_tier/elite
potion_low_urgency_selected_rate_by_tier/normal

potion_save_life_bad_selected_rate
potion_damage_lethal_selected_rate
potion_energy_no_followup_selected_rate
potion_draw_no_followup_selected_rate
potion_mechanic_use_selected_rate
```

### diagnostics

扩展：

```text
diagnostics/potion_transitions.jsonl
diagnostics/action_offenders.jsonl
```

新增字段：

```text
potion_id
potion_type
urgency_score
save_value_score
followup_count
incoming_after_block
near_term_lethal_forecast
mechanic_window
bad_use_reason
```

### 测试

- [ ] hp 71/80、incoming_after_block 12、瓶中精灵，不应选择。
- [ ] hp 8/80、incoming_after_block 12、瓶中精灵，允许或强烈推荐。
- [ ] energy potion 但无 follow-up，应惩罚。
- [ ] energy potion + X-cost 且 effective_energy > 0，应允许。
- [ ] damage potion lethal，应允许。
- [ ] Kaiser 药水 targeted 另一侧可作为 mechanic/facing action。

### 验收标准

```text
low_quality_potion_selected offender 下降 70%+
potion_unused_on_death_rate == 0
potion_save_recommended_selected_rate 不恶化
normal/elite bad potion 显著下降
```

---

## P1-2. X 费牌 0 能量 hard invalid

### 问题

当前仍有：

```text
x_cost_zero_bad_selected_rate_p0 latest ≈ 0.04
zero_energy_x_cost_selected_rate same
```

模型可能仍被 turn 初始 3 能量或 stale energy 误导。

### 实现要求

```text
if card.cost_type == X
and effective_energy <= 0
and card has no valid zero-energy effect:
    remove from candidate preference / huge penalty / override
```

注意：

- [ ] 必须使用当前动作选择时的 `effective_energy`。
- [ ] 不得使用回合初始 energy。
- [ ] 每次打牌后 hand/action state 必须刷新后再评估 X-cost。

### 主要文件

- [ ] `packages/rl-agent/muzero/train.py`
- [ ] `packages/rl-agent/muzero/sts2_env/mcts.py`
- [ ] observation/action feature encoder。
- [ ] bridge action payload，如果缺 `cost_type` / `effective_cost`。

### 新增指标

```text
x_cost_effective_energy_at_select_mean
x_cost_raw_energy_at_select_mean
x_cost_initial_energy_at_turn_mean
x_cost_zero_guard_available_rate
x_cost_zero_guard_applied_rate
x_cost_zero_guard_override_rate
x_cost_zero_bad_selected_rate_p0
```

### 测试

- [ ] effective_energy=0，X damage card，不允许选。
- [ ] effective_energy=0，X card 有明确 zero-energy effect，允许但不得计入 bad。
- [ ] effective_energy=2，X card 允许。
- [ ] 连续打牌后 energy 从 3 变 0，后续 X card 使用 energy=0 评估。

### 验收标准

```text
x_cost_zero_bad_selected_rate_p0 == 0
zero_energy_x_cost_selected_rate == 0
```

连续至少 1000 个 boss decision 无 offender。

---

## P1-3. HP-cost low-margin 与回费无 follow-up guard

### 问题

当前：

```text
hp_cost_self_lethal_selected_rate = 0
hp_cost_low_margin_selected_rate latest ≈ 0.20
hp_cost_low_margin_selected_rate avg20 ≈ 0.0209
```

说明直接自杀被防住了，但低安全边际自损仍存在。

用户还指出：

```text
打出回费牌但没有任何其余牌可以打，那这张牌其实是白打。
```

### 实现要求：HP-cost margin

计算：

```text
after_hp = current_hp - unblockable_hp_cost
projected_damage = max(0, incoming - block)
survival_margin = after_hp - projected_damage
low_margin_threshold = max(4, max_hp * 0.08)
```

规则：

```text
if after_hp <= 0:
    hard forbid unless impossible state

if survival_margin <= 0
and not confirmed_immediate_lethal:
    hard forbid / override

if survival_margin <= low_margin_threshold
and not confirmed_immediate_lethal:
    strong penalty
```

### 实现要求：回费/降费无 follow-up

对以下动作类型：

```text
energy_refund
cost_reduce
gain_energy
free_to_play_setup
hp_cost_energy_refund
```

判断：

```text
after playing this action,
if no meaningful follow-up playable action exists:
    mark no_followup
```

特别是扣血回费牌：

```text
if hp_cost > 0
and refund_energy > 0
and no meaningful_followup:
    huge penalty
```

`meaningful_followup` 不等于任意 legal action，至少满足：

```text
damage > 0
or block reduces actual incoming
or draw/cycle improves urgent mechanism
or boss-specific mechanism action
or confirmed setup value
```

### 主要文件

- [ ] `packages/rl-agent/muzero/train.py`
- [ ] action semantic / card runtime encoder。
- [ ] bridge card payload，如果缺 hp cost / energy refund / cost reduce 字段。

### 新增指标

```text
hp_cost_self_lethal_selected_rate
hp_cost_low_margin_candidate_rate
hp_cost_low_margin_selected_rate
hp_cost_guard_applied_rate
hp_cost_guard_override_rate
hp_cost_lethal_exemption_rate

energy_refund_no_followup_candidate_rate
energy_refund_no_followup_selected_rate
hp_cost_energy_refund_no_followup_selected_rate
```

### 测试

- [ ] hp_cost 会使 hp <= 0，必须禁止。
- [ ] hp_cost 后 survival_margin <= 0，非 lethal 时禁止。
- [ ] hp_cost 后 margin 很低，非 lethal 时强惩罚。
- [ ] 放血/预借时间类回费后没有可打正收益牌，应惩罚。
- [ ] 回费后能接 lethal / high block / mechanism action，应允许。

### 验收标准

```text
hp_cost_self_lethal_selected_rate == 0
hp_cost_low_margin_selected_rate < 0.005
energy_refund_no_followup_selected_rate 接近 0
hp_cost_energy_refund_no_followup_selected_rate 接近 0
```

---

## P1-4. Ceremonial detector 口径修正，然后再调 bias

### 问题

offender 中：

```text
ceremonial_missed_stun_window 945 / last 5000
ceremonial_low_impact_under_lock 361 / last 5000
```

但存在 false positive：

```text
selected = End Turn
energy = 0
legal_action_count = 1
positive_action_count = 0
ceremonial_one_card_lock = 1
ceremonial_stun_window = 1
alternative_actions = []
offender = ceremonial_missed_stun_window
```

这不应算 missed。

### detector 修正

`ceremonial_missed_stun_window` 只在以下条件同时满足时 emit：

```text
ceremonial_stun_window == true
and selected_action is not high_impact
and exists legal high_impact alternative
and not forced_end_turn
and positive_action_count > 0
```

禁止 emit 的情况：

```text
legal_action_count <= 1
positive_action_count == 0
selected=end_turn 且没有可用 high-impact alternative
energy=0 且没有可用 potion / 0-cost high-impact action
```

### bias 调整

detector 修正后再做：

```text
during stun_window:
    high_impact card/potion +2.5 ~ +4.0
    low_impact action       -2.0
    low_urgency potion      -3.0
    end_turn if high-impact exists -5.0
```

### 主要文件

- [ ] `packages/rl-agent/muzero/train.py`
- [ ] diagnostics/offender writer。

### 新增指标

```text
ceremonial_stun_window_high_impact_available_rate
ceremonial_stun_window_high_impact_selected_rate
ceremonial_stun_window_forced_end_turn_rate
ceremonial_missed_stun_window_true_rate
ceremonial_missed_stun_window_false_positive_suppressed_rate
ceremonial_low_impact_under_lock_selected_rate
```

### 测试

- [ ] only End Turn legal，不 emit missed stun。
- [ ] positive_action_count=0，不 emit missed stun。
- [ ] high-impact alternative 存在且 selected low-impact，emit true missed。
- [ ] high-impact selected，不 emit missed。

### 验收标准

```text
forced End Turn 不再计入 missed stun
ceremonial_missed_stun_window_true_rate 下降
ceremonial_beast_boss win_rate 不低于当前约 0.46
```

---

# P2 — 机制覆盖与诊断增强

---

## P2-1. Runtime card state 完整覆盖

### 问题

很多卡牌效果改变的是“这张卡牌实例在本场战斗中的状态”，例如：

```text
重放
虚无
消耗
保留
复制
变化
耗能降低
临时 0 费
附魔
选择 X 张手牌替换
选择保留一张卡
保留整手牌
```

如果模型只看到 card name/card id，而看不到 runtime instance state，就会误判行动价值。

### 原则

- [ ] 优先使用内部 id / instance uuid / card modifiers / flags。
- [ ] 不用文本正则驱动 hard decision。
- [ ] 文本只能 fallback diagnostic。

### 需要覆盖的字段

identity：

```text
card_id
instance_uuid
upgraded
misc
magic_number
base_damage
base_block
cost
cost_for_turn
```

zone：

```text
hand
draw_pile
discard_pile
exhaust_pile
limbo
retained
generated
temporary
```

runtime modifiers：

```text
exhaust_this_combat
ethereal_this_combat
retain_this_turn
purge_on_use
autoplay
replay_count
copied_from
duplicated
transformed
enchanted
cost_modified_this_turn
cost_modified_combat
free_to_play_once
temporary_zero_cost
x_cost_effective_energy
```

selection effect：

```text
requires_card_selection
min_select
max_select
valid_selection_zone
selection_target_type
selection_can_cancel
selection_changes_cost
selection_changes_exhaust
selection_changes_retain
selection_transforms_cards
selection_duplicates_cards
selection_discards_cards
selection_exhausts_cards
```

### 主要文件

- [ ] `mods/sts2-bridge/Scripts/BridgeGameApi*.cs`
- [ ] `packages/rl-agent/sts2_env/observation_v3.py`
- [ ] `packages/rl-agent/sts2_env/observation_common.py`
- [ ] `packages/rl-agent/muzero/train.py`
- [ ] action feature encoder。

### 新增指标

```text
card_runtime/instance_uuid_present_rate
card_runtime/modified_cost_present_rate
card_runtime/exhaust_flag_present_rate
card_runtime/ethereal_flag_present_rate
card_runtime/retain_flag_present_rate
card_runtime/enchantment_present_rate
card_runtime/replay_flag_present_rate
card_runtime/selection_effect_present_rate
```

### 验收标准

```text
card_identity_runtime_internal_selected_rate 不异常为 0
selection_runtime_internal_selected_rate 在选择类卡牌出现时 > 0
```

---

## P2-2. Card selection flow 防重复选择循环

### 问题

选择类卡牌可能进入重复选择循环，例如“净化”类动作不断选卡/返回同一状态。

### 目标

把 card selection 当成独立状态机，而不是普通 combat action。

### selection state 字段

```text
selection_session_id
selection_source_action_id
selection_min
selection_max
selection_selected_count
selection_remaining_count
selection_can_confirm
selection_can_cancel
selection_timeout_turn
selection_legal_option_ids
selection_selected_ids
```

### dead-loop guard

如果出现：

```text
same selection_session_id
same selected set
same legal options
repeated N times
```

则：

```text
if selected_count >= min_select:
    force confirm
elif can_cancel and expected_value_low:
    cancel
else:
    choose highest heuristic legal option
```

### 新增指标

```text
selection_loop_detected_rate
selection_loop_auto_confirm_rate
selection_loop_auto_cancel_rate
selection_repeated_same_option_rate
selection_invalid_repeat_selected_rate
```

### 测试

- [ ] same selection state 重复 N 次后自动 confirm。
- [ ] 未达 min_select 时不错误 confirm。
- [ ] 可以 cancel 且收益低时 auto cancel。
- [ ] 选择项 instance_uuid 不丢失。

---

## P2-3. Knowledge Demon / Soul Nexus / Slumbering Beetle 先做死亡切片

### 问题

当前这几个 encounter 弱，但不能先靠猜机制写 bias：

```text
knowledge_demon_boss    ≈ 0.136
soul_nexus_elite        ≈ 0.50
slumbering_beetle       ≈ 0.37 - 0.43
```

### 目标

先从失败样本中提取机制证据，再决定是否补 hard guard / reward / obs。

### 每个 encounter 抓取

每个至少 50 个失败样本：

```text
encounter_id
episode
turn
hp/max_hp
block
incoming
energy
hand
draw_count
discard_count
exhaust_count
selected_action
top_k_policy_actions
top_k_value_actions
legal_actions
positive_action_count
mechanic_context
offender_flags
death_turn_previous_3_actions
potion_state
card_runtime_state
```

### 输出

```text
diagnostics/death_slices/<encounter_id>.jsonl
diagnostics/death_slices/<encounter_id>_summary.md
```

summary 至少包含：

```text
top selected actions before death
top offender flags
top missing candidate types
average incoming/block/hp margin
potion held on death
end_turn under risk
x_cost involvement
hp_cost involvement
selection-loop involvement
targeting errors
```

### 验收标准

- [ ] 每个 encounter 至少 50 个 death slice。
- [ ] 每个 summary 给出前三个可行动根因。
- [ ] 不在无证据情况下添加新 boss-specific bias。

---

# P3 — 人类示范 / 模仿学习，用于突破瓶颈

---

## P3-1. 机制稳定后再接入 human demo

### 前置条件

只有当以下条件满足后，才接入 imitation learning：

```text
replay quota 生效
Kaiser hard guard 生效
Insatiable hard guard 生效
药水 bad use 明显下降
X-cost zero bad 为 0
HP-cost low-margin 接近 0
selection loop 修复
```

否则 human demo 会和旧 bug / 旧 schema 混在一起，污染 buffer。

### demo 范围

优先录制：

```text
Kaiser facing handling
Insatiable escape timing
Ceremonial stun/one-card-lock window
Knowledge Demon 关键机制
Soul Nexus elite
Slumbering Beetle normal
药水保存与使用时机
扣血回费牌 follow-up
X-cost 正确使用
```

### 训练方式

建议：

```text
demo minibatch ratio = 10% - 20%
持续 50k optimizer steps
之后线性衰减到 0% - 5%
```

增加 behavior cloning loss：

```text
loss_bc = CE(policy_logits, human_action)
total_loss += bc_weight * loss_bc
```

初始：

```text
bc_weight = 0.05 ~ 0.10
```

### 验收标准

- [ ] demo 数据 schema 与当前 observation/action schema 一致。
- [ ] demo 不包含旧 bridge bug 阶段样本。
- [ ] BC loss 不压制 MuZero policy/value 正常学习。
- [ ] Kaiser / Insatiable demo 机制 offender 继续下降。

---

# 4. 正式训练恢复策略

## 4.1 不建议直接做的事

- [ ] 不从 latest checkpoint 直接长训。
- [ ] 不沿用 boss-heavy soft weights。
- [ ] 不混入 schema 变化前的旧 buffer 作为主训练数据。
- [ ] 不在 Kaiser / Insatiable hard guard 前继续跑 10h+ 长训。

## 4.2 推荐策略

```text
1. 固定 eval 选择 best checkpoint。
2. 完成 P0-2 replay hard quota。
3. 完成 P0-3 Kaiser hard guard。
4. 完成 P0-4 Insatiable hard force。
5. 完成 P1-1/P1-2/P1-3 低质量动作 guard。
6. 从 best checkpoint resume。
7. 使用新 replay quota 与新 run 目录。
8. 旧 buffer 只保留最近且 schema 兼容的一部分，或者重新 warm buffer。
```

### buffer 策略

推荐 partial warm buffer：

```text
保留最近 30% - 50% buffer
丢弃明显旧 bug 阶段数据
对 offender-heavy transition 降权
对修复后新数据提高 priority
```

必须降权的 offender：

```text
kaiser_facing_missed
kaiser_risky_end_turn
insatiable_frantic_escape_missed_at_1
insatiable_frantic_escape_missed_lt3
zero_energy_x_cost_selected
low_quality_potion_selected
hp_cost_low_margin_selected
selection_loop_detected
```

---

# 5. 每小时监控看板

## 5.1 总体

```text
recent_tail/64/win_rate
recent_tail/256/win_rate
recent_tail/64/boss_win_rate
recent_tail/256/boss_win_rate
recent_tail/256/normal_win_rate
recent_tail/256/elite_win_rate
recent_tail/256/reward_mean
```

必须同时看：

```text
latest
avg20
avg100
run best
```

如果 latest 比 run best 低超过 5pp，必须评估是否 checkpoint 回滚。

## 5.2 replay 分布

```text
buffer/sample_boss_rate
buffer/sample_elite_rate
buffer/sample_normal_rate
buffer/quota_boss_rate
buffer/quota_elite_rate
buffer/quota_normal_rate
buffer/quota_fallback_rate
```

验收门槛：

```text
boss <= 0.65
elite >= 0.18
normal >= 0.10
```

## 5.3 Kaiser

```text
boss_combat/kaiser_back_attack_risk_mean
boss_combat/kaiser_facing_change_candidate_count_mean
boss_combat/kaiser_facing_change_selected_rate
boss_combat/kaiser_facing_guard_override_rate
boss_combat/kaiser_risky_end_turn_selected_rate
boss_combat/kaiser_nonfacing_nonlethal_selected_rate
boss_combat/kaiser_crab_boss/win_rate
```

门槛：

```text
kaiser_facing_change_selected_rate >= 0.25
kaiser_risky_end_turn_selected_rate < 0.01
```

## 5.4 Insatiable

```text
boss_combat/insatiable_sandpit_countdown_lt3_rate
boss_combat/insatiable_frantic_escape_available_rate
boss_combat/insatiable_frantic_escape_selected_rate
boss_combat/insatiable_frantic_escape_missed_lt3_rate
boss_combat/insatiable_frantic_escape_missed_at_1_rate
boss_combat/insatiable_frantic_escape_force_applied_rate
```

门槛：

```text
missed_at_1_rate == 0
missed_lt3_rate < 0.005
```

## 5.5 药水

```text
potion_low_urgency_selected_rate
potion_no_followup_selected_rate
potion_save_recommended_selected_rate
potion_unused_on_death_rate
potion/by_id/*/bad_selected_rate
```

门槛：

```text
potion_unused_on_death_rate == 0
low_quality_potion_selected offender 下降 70%+
```

## 5.6 X-cost

```text
x_cost_zero_bad_selected_rate_p0
zero_energy_x_cost_selected_rate
x_cost_effective_energy_at_select_mean
x_cost_zero_guard_override_rate
```

门槛：

```text
x_cost_zero_bad_selected_rate_p0 == 0
```

## 5.7 HP-cost / 回费

```text
hp_cost_self_lethal_selected_rate
hp_cost_low_margin_selected_rate
energy_refund_no_followup_selected_rate
hp_cost_energy_refund_no_followup_selected_rate
```

门槛：

```text
self_lethal == 0
low_margin < 0.005
energy_refund_no_followup 接近 0
```

## 5.8 Ceremonial

```text
ceremonial_stun_window_high_impact_available_rate
ceremonial_stun_window_high_impact_selected_rate
ceremonial_stun_window_forced_end_turn_rate
ceremonial_missed_stun_window_true_rate
ceremonial_missed_stun_window_false_positive_suppressed_rate
```

---

# 6. 推荐执行顺序

## 第一批：必须先做

- [ ] P0-1 固定 eval checkpoint 选择。
- [ ] P0-2 replay hard quota。
- [ ] P0-3 Kaiser facing hard guard。
- [ ] P0-4 Insatiable escape hard force。

完成后允许短 smoke train。

## 第二批：降低低质量动作

- [ ] P1-1 药水 type-specific gate。
- [ ] P1-2 X 费 0 能量 hard invalid。
- [ ] P1-3 HP-cost low-margin 与回费无 follow-up guard。
- [ ] P1-4 Ceremonial detector 口径修正。

完成后允许正式恢复训练。

## 第三批：机制覆盖和诊断

- [ ] P2-1 runtime card state 完整覆盖。
- [ ] P2-2 selection flow 防循环。
- [ ] P2-3 Knowledge Demon / Soul Nexus / Beetle death slice。

## 第四批：突破瓶颈

- [ ] P3-1 human demo / imitation learning。

---

# 7. 最终验收门槛

正式长训前 smoke 验收：

```text
buffer/sample_boss_rate   <= 0.65
buffer/sample_elite_rate  >= 0.18
buffer/sample_normal_rate >= 0.10

kaiser_risky_end_turn_selected_rate < 0.01
insatiable_frantic_escape_missed_at_1_rate == 0
x_cost_zero_bad_selected_rate_p0 == 0
hp_cost_self_lethal_selected_rate == 0
potion_unused_on_death_rate == 0
```

正式训练 3-6 小时趋势验收：

```text
recent_tail/256/win_rate 不低于当前 0.4805，并向 run best 0.5234 回升
recent_tail/256/boss_win_rate 不低于当前 0.3626，并向 run best 0.4068 回升
recent_tail/256/normal_win_rate 不再低于 0.75，目标恢复到 0.85+
recent_tail/256/elite_win_rate 不低于 0.70
kaiser_crab_boss win_rate 从 0.10 提升到 0.20+
the_insatiable_boss win_rate 高于 0.30 且 missed_at_1=0
```

长期目标：

```text
recent_tail/256/boss_win_rate > 0.50
recent_tail/256/win_rate 稳定 > 0.60
normal/elite 不因 boss recovery 再次遗忘
```

---

# 8. 给执行代理的简短结论

当前不要把问题当作“继续训就会好”。
最新日志显示 loss 稳，问题在：

```text
1. replay 采样极端 boss-heavy，normal/elite 被饿死；
2. Kaiser facing 与 Insatiable escape 是确定性机制失败，必须 hard guard；
3. 药水、X 费、HP-cost/回费无 follow-up 仍有低质量动作；
4. latest checkpoint 已低于 run 内最佳，不应默认 resume latest。
```

执行优先级：

```text
先 fixed eval 选 checkpoint
再 replay hard quota
再 Kaiser / Insatiable hard guard
再药水 / X-cost / HP-cost guard
最后恢复正式训练
```
