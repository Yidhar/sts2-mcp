# Combat 低战损与战斗质量奖励策略设计

> 目的：随着模型战斗能力提升，逐步从“能赢”推进到“以更低战损、更少资源浪费、更好机制处理赢”。
>
> 重要修正：**boss 战本身的 HP 战损通常不应作为核心优化目标**，因为除 A10 等特殊规则外，绝大多数 boss 战后会回满生命值。因此 boss 战奖励应更关注：胜利、机制处理、药水/关键资源是否合理使用、是否避免死亡、是否为后续 run 保留关键构筑资源。低战损压力主要施加在 normal / elite / hallway 战斗上。

---

## 1. 设计目标

当前模型已经开始学会基础出牌、药水使用、部分 boss 机制处理。下一阶段 reward 不应该只问：

```text
是否赢了
```

而应该逐步优化为：

```text
以低损耗赢普通战斗和精英战斗
在 boss 战中稳定处理机制并获胜
避免空过、能量浪费、0 能量 X 牌等明显错误
合理使用药水、消耗牌、自伤换资源牌
保留 full run 长期资源
```

核心原则：

1. **先保证胜率，再压低战损**。
2. **普通 / 精英战斗重点压 HP 损失**。
3. **boss 战不强压 HP 战损，除非 A10 或特定模式不会回满**。
4. **不要把所有“能打但没打”的牌都当成空过**，因为消耗牌、自伤牌、关键循环牌可能应该保留。
5. **奖励必须随模型进步动态增强**，不能一开始强迫低战损导致模型只防不杀。

---

## 2. 总体奖励结构

建议 combat reward 拆为：

```text
R_total =
    R_outcome
  + R_hp_efficiency
  + R_turn_efficiency
  + R_action_quality
  + R_resource_quality
  + R_mechanic_quality
```

| 奖励项 | 主要作用 |
|---|---|
| `R_outcome` | 确保模型优先学会胜利 |
| `R_hp_efficiency` | 压低普通 / 精英战斗战损 |
| `R_turn_efficiency` | 防止无限苟、防止拖回合 |
| `R_action_quality` | 防止空过、能量浪费、0 费 X 牌 |
| `R_resource_quality` | 约束药水、消耗牌、自伤换资源牌 |
| `R_mechanic_quality` | boss 机制处理，例如 Kaiser 面向 |

---

## 3. 按战斗类型区分目标

### 3.1 Normal / Hallway

目标：

```text
稳定胜利 + 尽量低损 + 不浪费资源
```

HP 战损非常重要，因为会影响后续路线、精英、事件和 boss 前状态。

建议：

```text
R_outcome          高
R_hp_efficiency    中到高，随胜率提升而增强
R_turn_efficiency  中
R_action_quality   中
R_resource_quality 中
R_mechanic_quality 低
```

---

### 3.2 Elite

目标：

```text
稳定胜利 + 明显压低战损 + 不能过度消耗药水/关键资源
```

Elite 的 HP 损失比 normal 更关键，因为它们通常是 run 中最大血量压力之一。

建议：

```text
R_outcome          高
R_hp_efficiency    高
R_turn_efficiency  中
R_action_quality   中
R_resource_quality 中到高
R_mechanic_quality 中
```

---

### 3.3 Boss

目标：

```text
赢 + 正确处理机制 + 避免死亡 + 合理使用资源
```

关键修正：

```text
非 A10 情况下，boss 战后通常回满 HP，所以 boss 战 HP 损失不应作为核心 reward 压力。
```

Boss 战中应该更关注：

- 是否胜利；
- 是否处理 boss 机制；
- 是否在高风险状态下正确防守 / 转身 / 击杀关键部位；
- 是否避免带药水死亡；
- 是否误用关键消耗牌；
- 是否出现明显错误，例如 0 能量 X 牌、无意义空过、错误目标。

建议：

```text
R_outcome          最高
R_hp_efficiency    低，A10 或不回血规则下才提高
R_turn_efficiency  低到中
R_action_quality   中
R_resource_quality 中
R_mechanic_quality 高
```

---

## 4. 训练阶段 curriculum

奖励权重应随模型能力提升动态变化。

### Phase 0：基础可行动阶段

适用：

```text
win_rate < 30%
```

目标：

- 学会出牌；
- 学会攻击；
- 学会 block；
- 学会不用 0 能量打 X 费牌；
- 学会不明显空过。

建议权重：

```text
R_outcome          1.00
R_hp_efficiency    0.05 ~ 0.10
R_turn_efficiency  0.10
R_action_quality   0.30
R_resource_quality 0.05
R_mechanic_quality 0.10
```

### Phase 1：稳定胜利阶段

适用：

```text
30% <= win_rate < 60%
```

目标：

- 提升胜率；
- 普通 / 精英开始压低 HP 损失；
- boss 开始强化机制处理。

建议权重：

```text
R_outcome          1.00
R_hp_efficiency    0.20 ~ 0.35
R_turn_efficiency  0.15
R_action_quality   0.25
R_resource_quality 0.10
R_mechanic_quality 0.20
```

### Phase 2：低战损阶段

适用：

```text
60% <= win_rate < 80%
```

目标：

- normal / elite 明显压低战损；
- 避免为了低损无限防守；
- boss 重点提升机制处理和稳定击杀。

建议权重：

```text
R_outcome          0.90
R_hp_efficiency    0.50 ~ 0.70   # normal/elite
R_hp_efficiency    0.05 ~ 0.15   # boss 非 A10
R_turn_efficiency  0.20
R_action_quality   0.20
R_resource_quality 0.20
R_mechanic_quality 0.35
```

### Phase 3：高质量战斗阶段

适用：

```text
win_rate >= 80%
或 elite_win_rate / boss_win_rate 已稳定
```

目标：

- normal / elite 追求低战损；
- boss 追求机制正确、胜率稳定、资源不乱用；
- full run 长期价值最大化。

建议权重：

```text
R_outcome          0.80
R_hp_efficiency    0.80 ~ 1.00   # normal/elite
R_hp_efficiency    0.05 ~ 0.20   # boss 非 A10
R_turn_efficiency  0.25
R_action_quality   0.15
R_resource_quality 0.35
R_mechanic_quality 0.50
```

---

## 5. 胜负奖励 `R_outcome`

基础：

```text
胜利: +1.0
失败: -1.0
```

按 tier 加权：

```text
normal: 1.0
elite:  1.5
boss:   2.0
```

示例：

```text
normal win: +1.0
elite win:  +1.5
boss win:   +2.0

normal loss: -1.0
elite loss:  -1.5
boss loss:   -2.0
```

Boss 的核心还是赢。不要用 boss HP 损失强行塑形，否则会让模型在一个会回血的战斗里优化无长期价值目标。

---

## 6. HP 战损奖励 `R_hp_efficiency`

### 6.1 基础 HP delta

定义：

```text
hp_start = 战斗开始玩家 HP
hp_end   = 战斗结束玩家 HP
hp_loss  = max(0, hp_start - hp_end)
hp_max   = 玩家最大 HP
hp_loss_ratio = hp_loss / max(1, hp_max)
```

基础惩罚：

```text
R_hp_loss = - hp_loss_ratio
```

按 tier：

```text
normal: 1.0
elite:  1.5
boss:   0.1 ~ 0.3    # 非 A10，因为 boss 后回血
boss_A10_or_no_heal: 1.5 ~ 2.0
```

---

### 6.2 胜利后的低战损 bonus

只建议对 normal / elite 强使用：

```text
hp_preserve_ratio = hp_end / max_hp
R_hp_preserve = win ? sqrt(hp_preserve_ratio) : 0
```

Boss 非 A10 情况：

```text
R_hp_preserve_boss = 0 或极低权重
```

因为 boss 后回血时，最终剩余 HP 对后续 run 价值很低。

---

### 6.3 相对基准战损

为每个 encounter 维护滑动平均：

```text
baseline_hp_loss_ratio[encounter_id]
```

当前战斗：

```text
relative_improvement = baseline_hp_loss_ratio - current_hp_loss_ratio
R_relative_hp = clamp(relative_improvement, -0.5, +0.5)
```

使用范围：

```text
normal / elite: 强使用
boss 非 A10: 弱使用或不用
boss A10/no-heal: 正常使用
```

---

## 7. 每 step HP 损失 shaping

只在战斗结束给 HP reward 延迟太长，可以加 step-level 血损惩罚。

```text
delta_hp = hp_after - hp_before
damage_taken = max(0, hp_before - hp_after)
R_step_hp_loss = - damage_taken / max_hp
```

但需要区分：

```text
enemy_damage_taken
self_inflicted_hp_loss
```

建议：

```text
R_enemy_damage = -1.0 * enemy_damage / max_hp
R_self_damage  = -0.3 * self_damage / max_hp
```

Boss 非 A10 情况：

```text
R_enemy_damage_boss = 很低或仅在接近死亡时惩罚
R_self_damage_boss  = 根据收益判断，不能硬禁
```

---

## 8. 回合效率 `R_turn_efficiency`

低战损 reward 会导致模型过度防守，所以必须配合回合效率。

### 8.1 基于 encounter baseline

```text
turn_count = 当前战斗回合数
expected_turns = encounter 滑动平均回合数
```

奖励：

```text
R_turn_efficiency = clamp((expected_turns - turn_count) / expected_turns, -0.5, +0.5)
```

### 8.2 每回合轻微惩罚

```text
normal: -0.02 / turn
elite:  -0.015 / turn
boss:   -0.005 ~ -0.01 / turn
```

Boss 回合惩罚不要过重，否则会让模型为了快杀吃不必要风险。

---

## 9. 行为质量 `R_action_quality`

用于解决：

- 空过；
- 有能量不用；
- 0 能量打 X 费牌；
- 可击杀却不击杀；
- 错误目标；
- 打牌顺序弱。

### 9.1 空过惩罚

定义：

```text
wasteful_end_turn =
    selected_action == end_turn
    and energy > 0
    and safe_positive_action_count > 0
```

奖励：

```text
normal: -0.15
elite:  -0.25
boss:   -0.25 ~ -0.35
```

注意：`safe_positive_action_count` 不能等于所有 playable card。

---

### 9.2 正收益动作分类

应将可打动作分为：

```text
must_consider_positive
optional_positive
strategic_skip_allowed
negative_or_risky
```

#### `must_consider_positive`

明显应该考虑：

- 攻击可击杀；
- block 可挡住当前威胁；
- debuff 可显著降低本回合伤害；
- 药水可避免死亡；
- 0 费无副作用收益牌。

#### `optional_positive`

可能打，也可能不打：

- 低收益攻击；
- 过量 block；
- 多余 debuff；
- 无关紧要的能力牌。

#### `strategic_skip_allowed`

合法但可以不打：

- 消耗牌；
- 自伤牌；
- 当前无后续收益的费用恢复牌；
- 关键循环牌；
- 需要 combo 的牌；
- 可能应进入弃牌堆而非消耗堆的牌。

#### `negative_or_risky`

可能不该打：

- 0 能量 X 牌；
- 自伤但无收益；
- 消耗关键牌但无 immediate payoff；
- 错误目标导致机制风险上升。

---

### 9.3 能量浪费

只在有安全正收益动作时惩罚：

```text
R_energy_waste = -0.03 * unused_energy * has_safe_positive_action
```

Boss 可以稍高一点，因为 boss 中空过更致命：

```text
boss: -0.05 * unused_energy * has_safe_positive_action
```

---

### 9.4 0 能量 X 牌

如果：

```text
selected_card.is_x_cost == true
and current_energy <= 0
and card_has_no_zero_energy_effect
```

惩罚：

```text
R_zero_energy_x = -0.25
```

监控：

```text
combat/x_cost_available_rate
combat/zero_energy_x_cost_available_rate
combat/zero_energy_x_cost_selected_rate
combat/zero_energy_x_cost_penalty_rate
```

---

## 10. 消耗牌 / 牌堆循环策略

有些牌合法可打，但策略上应该不打，让它进入弃牌堆参与后续循环，或者避免被消耗。

例子：

- 生产制造；
- 预借时间；
- 放血；
- 关键消耗牌；
- 关键费用恢复牌；
- 关键抽牌牌；
- 需要 combo 的牌。

### 10.1 消耗牌动作属性

每张牌应有：

```text
will_exhaust
exhaust_self
ethereal
retain
self_damage
energy_gain
card_draw
deck_cycle_value
combo_value
```

行动质量应估计：

```text
action_value = immediate_value - future_exhaust_cost
```

### 10.2 immediate value

```text
damage
block
draw
energy_gain
debuff
kill
survival
mechanic_solution
```

### 10.3 future exhaust cost

```text
是否关键牌
当前牌组是否依赖它循环
是否长战斗
当前 draw/discard/exhaust pile 状态
是否 boss/elite 需要留资源
```

### 10.4 消耗牌误用惩罚

如果：

```text
will_exhaust == true
and immediate_value_low
and future_value_high
```

惩罚：

```text
normal: -0.15
elite:  -0.25
boss:   -0.20 ~ -0.35
```

### 10.5 合理保留消耗牌

如果：

```text
selected_action == end_turn
and 剩余可打牌主要是 strategic_skip_allowed
and safe_positive_action_count == 0
```

则：

```text
不惩罚空过
可给很小奖励 R_strategic_hold = +0.03 ~ +0.05
```

这个奖励必须很小，防止模型学会乱留牌。

---

## 11. 资源质量 `R_resource_quality`

### 11.1 药水

当前模型有药水时已经倾向使用，所以不要继续强推药水。

#### 死亡且有药水未用

```text
R_potion_unused_on_death = -0.5
```

#### 用药水避免死亡

```text
R_potion_survival = +0.5
```

#### 无必要乱用药水

如果：

```text
potion_used
and combat_threat_low
and hp_safe
and no_kill_or_survival_gain
```

惩罚：

```text
R_potion_overuse = -0.1
```

Boss 战中药水换胜率是合理的，因此 boss 对 `potion_overuse` 惩罚应更轻。

---

### 11.2 自伤换资源

例如放血、预借时间等。

不能简单惩罚自伤，应看收益：

```text
self_damage_efficiency = gained_energy_or_cards_or_damage_prevented / self_hp_loss
```

如果自伤后：

- 多打出关键牌；
- 击杀敌人；
- 获得 block 避免更大伤害；
- 触发 combo；

奖励：

```text
R_good_self_damage = +0.1 ~ +0.3
```

如果：

```text
self_damage > 0
and no follow-up value
```

惩罚：

```text
R_bad_self_damage = -0.2
```

---

## 12. Boss 机制 `R_mechanic_quality`

Boss 非 A10 情况下不强压 HP 损失，所以机制 reward 更重要。

### 12.1 Kaiser 面向机制

已有信号：

```text
kaiser_back_attack_risk
kaiser_defense_candidate_count
kaiser_pressure_candidate_count
kaiser_facing_change_candidate_count
```

#### 背刺风险下降

```text
risk_before = back_attack_risk_before
risk_after  = back_attack_risk_after
R_kaiser_risk_reduce = +0.2 * max(0, risk_before - risk_after)
```

#### 高风险 end_turn 惩罚

```text
if selected_end_turn and back_attack_risk > 0:
    R_kaiser_risky_end_turn = -0.3
```

如果没有任何处理候选：

```text
if defense_candidate_count == 0 and pressure_candidate_count == 0:
    penalty *= 0.3
```

#### 正确转身奖励

攻击或药水指向另一侧目标导致 facing 改变并降低风险：

```text
R_kaiser_facing_change = +0.3
```

#### 压制击杀奖励

如果不转身，但通过攻击压死关键部位，也应奖励：

```text
R_kaiser_pressure = +0.2
```

这样不会强迫模型只学转身，也允许直接压死。

---

## 13. 动态低战损推进

### 13.1 基于滑动胜率调权重

```text
progress = clamp((win_rate_256 - 0.3) / 0.5, 0, 1)
```

即：

```text
win_rate <= 30%: progress = 0
win_rate >= 80%: progress = 1
```

HP 权重：

```text
hp_weight = hp_weight_min + progress * (hp_weight_max - hp_weight_min)
```

Normal / elite：

```text
hp_weight_min = 0.1
hp_weight_max = 1.0
```

Boss 非 A10：

```text
hp_weight_min = 0.0
hp_weight_max = 0.2
```

Boss A10 或 no-heal：

```text
hp_weight_min = 0.1
hp_weight_max = 1.0
```

---

### 13.2 按 encounter 单独推进

不要用全局 win rate 控制所有战斗。

```text
encounter_progress[encounter_id] =
    clamp((encounter_win_rate_128 - 0.3) / 0.5, 0, 1)
```

例如：

```text
kaiser_crab_boss win_rate = 40%
progress = 0.2
```

说明 Kaiser 还没稳定，不应该过度压机制以外的目标。

---

### 13.3 基于 baseline hp loss 的目标线

每个 encounter 维护：

```text
hp_loss_p50
hp_loss_p25
hp_loss_p10
```

Normal / elite：

```text
target_hp_loss = lerp(hp_loss_p50, hp_loss_p10, encounter_progress)
```

Boss 非 A10：

```text
不使用该目标，或仅用于诊断
```

Boss A10 / no-heal：

```text
正常使用 target_hp_loss
```

奖励：

```text
if current_hp_loss_ratio <= target_hp_loss:
    R_hp_target = +0.3
else:
    R_hp_target = -0.3 * (current_hp_loss_ratio - target_hp_loss)
```

---

## 14. 防 reward hacking

### 14.1 防止为了低战损不攻击

加入：

```text
turn_efficiency
enemy_hp_progress
kill_reward
threat_reduction_reward
```

### 14.2 防止为了保药水死掉

```text
R_potion_unused_on_death = -0.5
```

### 14.3 防止为了不用消耗牌而空过

只有当：

```text
剩余可打牌属于 strategic_skip_allowed
and safe_positive_action_count == 0
```

才允许 strategic hold。

### 14.4 防止拖回合刷 block

```text
R_long_combat = -0.1 * over_baseline_turn_ratio
```

但 boss 该项应更轻。

---

## 15. 推荐监控指标

### 15.1 HP / 战损

```text
combat/hp_loss_mean
combat/hp_loss_ratio_mean
combat/hp_preserve_ratio_mean
combat/enemy_damage_taken_mean
combat/self_damage_taken_mean
combat/hp_loss_vs_baseline_mean
combat/hp_target_hit_rate
```

按 encounter：

```text
combat/encounter/{name}/hp_loss_ratio_mean
combat/encounter/{name}/hp_target_hit_rate
combat/encounter/{name}/win_rate
```

Boss 非 A10 的 `hp_loss` 指标主要用于诊断，不作为强 reward 目标。

---

### 15.2 空过 / 能量浪费

```text
combat/wasteful_end_turn_rate
combat/strategic_hold_rate
combat/unused_energy_on_end_turn_mean
combat/safe_positive_action_left_mean
combat/playable_cards_left_mean
```

---

### 15.3 X 费牌

```text
combat/x_cost_available_rate
combat/zero_energy_x_cost_available_rate
combat/zero_energy_x_cost_selected_rate
combat/zero_energy_x_cost_penalty_rate
combat/positive_x_cost_selected_rate
```

---

### 15.4 消耗牌策略

```text
combat/exhaust_card_available_rate
combat/exhaust_card_selected_rate
combat/low_value_exhaust_play_rate
combat/high_value_exhaust_play_rate
combat/strategic_exhaust_hold_rate
combat/exhausted_key_card_count
combat/discarded_playable_exhaust_count
```

---

### 15.5 药水

```text
combat/potion_use_rate
combat/potion_survival_use_rate
combat/potion_overuse_rate
combat/potion_unused_on_death_rate
```

---

### 15.6 Boss 机制

```text
combat/kaiser/back_attack_risk_mean
combat/kaiser/risk_reduction_mean
combat/kaiser/risky_end_turn_rate
combat/kaiser/facing_change_candidate_rate
combat/kaiser/facing_change_selected_rate
combat/kaiser/pressure_selected_rate
combat/kaiser/defense_selected_rate
```

---

## 16. 实施顺序建议

### Step 1：先加统计，不改 reward

先记录：

```text
hp_loss_ratio
hp_preserve_ratio
turn_count
unused_energy
strategic_hold
exhaust_play_quality
zero_energy_x
mechanic_risk
```

目的：先确认 detector 正确。

---

### Step 2：给 normal / elite 加弱 HP reward

```text
hp_weight = 0.1 ~ 0.2
```

Boss 非 A10 不加或极低。

观察：

```text
win_rate 是否下降
normal/elite hp_loss 是否下降
turn_count 是否暴涨
```

---

### Step 3：加入 relative baseline

对 normal / elite：

```text
R_relative_hp
```

让模型和过去的自己比，而不是被固定阈值压死。

---

### Step 4：加入 strategic skip 分类

在没有 `strategic_skip_allowed` 之前，不要加重空过惩罚。

---

### Step 5：按 encounter 做 curriculum

```text
encounter_progress
target_hp_loss
```

Normal / elite 使用；boss 非 A10 只用于诊断或弱权重。

---

## 17. 最终策略总结

最终建议：

```text
胜率主导 early stage
normal/elite 逐步压低战损
boss 非 A10 不强压战损，重点压机制错误和死亡风险
encounter-specific baseline 推进
战略性保留牌不算空过
消耗牌、自伤牌、费用牌按 immediate value 与 future cost 判断
```

一句话：

> 普通和精英战斗要逐步打得更省血；boss 战除 A10 外血损不是核心问题，应该重点训练机制处理、稳定击杀和资源合理使用。
