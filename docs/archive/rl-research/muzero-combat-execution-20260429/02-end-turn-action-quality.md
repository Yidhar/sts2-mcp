# 02 — End Turn and Action Quality

本文件定义 end_turn 和特殊动作质量的目标实现。
重点：不要让“能打但不打”的合理策略被当成空过，也不要让真正空过逃过 detector。

---

## TASK-B1 — EndTurnTaxonomy

### 目标

把 selected `end_turn` 明确分成三类：

```text
forced_end_turn
strategic_defer_end_turn
bad_end_turn
```

旧的 `wasteful_end_turn` 保留为 alias，但内部实现必须迁移到 taxonomy。

### Owned paths

```text
packages/rl-agent/muzero/train.py
packages/rl-agent/tests/test_end_turn_taxonomy.py
```

### 分类优先级

```text
bad_end_turn override > forced_end_turn > strategic_defer_end_turn > unknown
```

说明：

- 如果存在 urgent/mandatory 动作，不应被 strategic defer 掩盖。
- forced 只用于确实没有可执行正收益动作/只有合法 end_turn。
- strategic defer 需要明确未来理由。

### forced_end_turn 条件

满足下列之一：

1. 没有 legal non-end_turn action。
2. 有 non-end_turn action，但全部不可执行或无效果。
3. energy=0 且没有 0-cost / potion / free trigger / retain / discard 等动作。
4. bridge actionability 标记为 stable no actions。

注意：如果 bridge 标记 `transient_only_end_turn=true`，不能算 forced，应延迟暴露或标记为 transient。

### bad_end_turn 条件

满足任意强条件：

1. `energy > 0` 且存在 positive action。
2. 存在 0-cost urgent action。
3. incoming damage 高，且存在 mitigation action：
   - block
   - weak
   - vulnerable kill setup
   - strength down / artifact strip
   - potion 防御
4. 存在 lethal / kill line。
5. 存在 hand mutation 打开后续：
   - draw
   - create card
   - cost reduction
   - upgrade hand
   - replay/duplicate
   - retain/keep important card
6. Kaiser back attack risk 存在，且有 facing change / pressure / defense action。
7. Ceremonial one-card/stun window 存在，且有 high impact action。
8. 有高质量 potion 使用窗口。
9. 有 refund action 且 refund 后有明确 followup。

### strategic_defer_end_turn 条件

必须满足：

1. 不是 bad。
2. 至少有一个明确 future value reason：
   - exhaust 关键牌现在打会永久离开循环，当前收益低。
   - refund 牌打完没有 followup，等待下一轮收益更高。
   - retain/保留机制使跳过能保留关键牌。
   - curse/void/status 处理需要等待特定窗口。
   - boss 机制鼓励等待，如下一回合 stun/lock 窗口。
   - 低价值 potion 不该现在用。
3. 当前风险可接受：
   - incoming 不致命或有足够 block/hp。
   - 不会错过 lethal 或机制窗口。

### 指标

```text
combat/forced_end_turn_selected_rate
combat/strategic_defer_end_turn_selected_rate
combat/bad_end_turn_selected_rate
combat/bad_end_turn_available_rate
combat/end_turn_unknown_rate

boss_combat/<same>
boss_combat/<encounter_id>/<same>
```

旧指标映射：

```text
wasteful_end_turn_rate := bad_end_turn_selected_rate
wasteful_end_turn_bias_applied_rate := bad_end_turn_bias_applied_rate
```

### 测试

1. energy=2，hand 有 Defend，incoming=12，选 end_turn => bad。
2. energy=0，只有 Strike/Defend 都不可打，选 end_turn => forced。
3. energy=1，只有低价值 exhaust card，incoming=0，打出会消耗关键循环牌，选 end_turn => strategic defer。
4. Kaiser back attack risk + 可攻击另一侧，选 end_turn => bad。
5. transient only-end-turn => transient，不计 forced/bad，等待 bridge 修复。

---

## TASK-B2 — Strategic Skip Narrowing

### 问题

“有些消耗牌当前能打但策略上不该打”是真的。
但如果 detector 太宽，会把大量本该打的牌标成 strategic skip，模型会变怂。

### 目标

战略性不打必须是窄条件、强证据。

### Owned paths

```text
packages/rl-agent/muzero/train.py
packages/rl-agent/sts2_env/card_effect_profile.py
packages/rl-agent/tests/test_strategic_skip_narrowing.py
```

### 不允许的宽泛规则

下面规则禁止单独作为 strategic skip：

```text
is_exhaust == true
is_refund == true
is_special == true
cost > 0
damage == 0
```

这些只能作为候选原因，必须结合 future reason。

### 必需 future reason

至少满足一项：

1. 当前收益低于阈值，且该牌消耗/改变后无法回到循环。
2. 当前打出 refund 后没有 followup，但保留到下一回合有 followup 概率。
3. 当前打出会破坏 hand/pile setup。
4. 当前打出会错过 boss 机制窗口。
5. 当前打出 potion/card 会浪费高稀缺资源。

### 反例：不能 strategic skip

1. 消耗牌能直接击杀敌人。
2. 消耗牌能防止本回合死亡。
3. refund 牌打出后能立刻接 draw/create/cost-reduction。
4. 低费 exhaust 牌触发 relic/power 关键收益。

### 指标

```text
combat/strategic_skip_candidate_rate
combat/strategic_skip_selected_rate
combat/strategic_skip_false_positive_guard_rate
combat/strategic_skip_overridden_by_urgent_rate
```

### 测试

1. 单纯 exhaust 不产生 strategic skip。
2. exhaust + lethal 不产生 strategic skip。
3. exhaust + 低收益 + 当前无风险 + future setup 产生 strategic skip。
4. refund + no followup 产生 candidate，但如果有 draw followup 则取消。

---

## TASK-B3 — Refund-No-Followup Recompute

### 问题

回费牌不是“打了必赚”。
如果打出回费牌后没有任何其它可打动作，它可能是白打，甚至消耗/损失节奏。

但 followup 不能只看当前手牌静态 cost，因为打出动作可能：

- 抽牌。
- 生成牌。
- 降低费用。
- 触发 relic/power。
- 改变手牌状态。
- 改变 energy。

### 目标

重算 `refund_no_followup`：基于 after-action estimate，而不是 before-action 静态合法动作。

### Owned paths

```text
packages/rl-agent/muzero/train.py
packages/rl-agent/sts2_env/semantic_action.py
packages/rl-agent/sts2_env/card_effect_profile.py
packages/rl-agent/tests/test_refund_no_followup.py
```

### after-action estimate 必需考虑

```text
energy_after = energy_before - effective_cost + expected_energy_gain
hand_after_count = hand_count - played_cards + expected_draw + expected_create
cost_reduction_after = expected_cost_reduction
free_card_after = expected_free_card_count
replay/duplicate_after = expected_extra_action_count
```

### 分级

```text
refund_good_followup
refund_no_followup_but_intrinsic_value
refund_no_followup_low_value
refund_unknown
```

### intrinsic value

即使无 followup，也可能不是坏动作：

- 防止死亡。
- 打出高伤害。
- 触发 boss 机制。
- 产生永久/本战强 buff。
- 移除 debuff/状态。
- 改变 pile 形成下一轮循环。

### 指标

```text
combat/refund_candidate_rate
combat/refund_good_followup_selected_rate
combat/refund_no_followup_low_value_selected_rate
combat/refund_no_followup_but_intrinsic_selected_rate
```

### 测试

1. refund 后 energy>0 但手牌无可打，无 intrinsic => low value。
2. refund 后抽 2 张，预估有 0/1 费牌可打 => good followup。
3. refund 无 followup 但能 block lethal => intrinsic。
4. refund 无 followup 但触发 Kaiser facing change => intrinsic/mechanism。
