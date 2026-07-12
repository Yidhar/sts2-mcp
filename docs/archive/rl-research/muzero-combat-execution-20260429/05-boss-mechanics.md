# 05 — Boss Mechanics

本文件定义 Kaiser / Ceremonial / Insatiable 等机制进入 observation/action/planner/metrics 的目标。

---

## TASK-E1 — Kaiser Facing Semantics

### 目标

正确建模 Kaiser Crab 的面向/背刺机制。
任何指向操作（卡牌或药水）攻击/作用到某个方向的单位，都可能改变玩家 facing。不能基于 `enemy.side`。

### 已知事实

Live bridge 侧已有真实机制链路：

```text
ResolvePlayerFacing
  从 playerCreature.Powers 中找 SurroundedPower
  使用 surrounded.Facing.ToString() 得到 left/right

ComputeIncomingDamageMultiplier
  扫描敌人的 BackAttackLeftPower / BackAttackRightPower
  按当前面向组合返回伤害倍率
```

Kaiser 左右位置：

```text
enemy.powers[].id == BACK_ATTACK_LEFT_POWER  => 该敌人/部位在 left
enemy.powers[].id == BACK_ATTACK_RIGHT_POWER => 该敌人/部位在 right
```

错误做法：

```text
enemy.side
action.target.side
```

这些是敌我阵营，不是左右。

### Owned paths

```text
packages/rl-agent/sts2_env/semantic_action.py
packages/rl-agent/sts2_env/observation_v3.py
packages/rl-agent/sts2_env/_sim_translate.py
mods/sts2-bridge/Scripts/BridgeGameApi.EnvPayloads.cs
packages/rl-agent/muzero/train.py        # metrics only if needed
packages/rl-agent/tests/test_kaiser_facing_semantics.py
```

### 必需 observation 字段

```json
{
  "boss_mechanics": {
    "kaiser": {
      "active": true,
      "player_facing": "left",
      "back_attack_risk": 1.0,
      "incoming_multiplier": 1.5,
      "left_enemy_ids": ["..."],
      "right_enemy_ids": ["..."],
      "has_left_back_attack_power": true,
      "has_right_back_attack_power": true
    }
  }
}
```

### 必需 action semantic 字段

对每个 targeted action：

```json
{
  "target": {
    "combat_id": "...",
    "back_attack_position": "right"
  },
  "mechanism": {
    "kaiser_can_change_facing": true,
    "kaiser_changes_facing": true,
    "kaiser_facing_before": "left",
    "kaiser_facing_after_if_action": "right",
    "kaiser_incoming_multiplier_before": 1.5,
    "kaiser_incoming_multiplier_after_estimate": 1.0,
    "kaiser_risk_delta": -0.5
  }
}
```

### Candidate 定义

```text
kaiser_facing_change_candidate:
  action family in play_card/use_potion
  AND action has target_combat_id
  AND target enemy has BACK_ATTACK_LEFT/RIGHT_POWER
  AND target position != current player_facing
```

注意：self-target action 不算 facing change，但可能是 defense candidate。

### 额外 candidate

```text
kaiser_defense_candidate:
  block / weak / debuff / reduce incoming / potion defense

kaiser_pressure_candidate:
  damage / lethal setup / vulnerable / high value attack
```

### 指标

必须有 global 和 per Kaiser：

```text
boss_combat/kaiser_back_attack_risk_mean
boss_combat/kaiser_facing_change_candidate_count_mean
boss_combat/kaiser_facing_change_selected_rate
boss_combat/kaiser_risky_end_turn_selected_rate
boss_combat/kaiser_defense_candidate_count_mean
boss_combat/kaiser_defense_selected_rate
boss_combat/kaiser_pressure_candidate_count_mean
boss_combat/kaiser_pressure_selected_rate

boss_combat/kaiser_crab_boss/<same>
```

### 测试

1. enemy.side=`Enemy` 但 powers 含 `BACK_ATTACK_LEFT_POWER` => position=left。
2. player_facing=left，target left => changes_facing=false。
3. player_facing=left，target right => changes_facing=true。
4. potion targeted to right enemy 也 changes_facing=true。
5. self-target block 不 changes_facing，但是 defense candidate。

---

## TASK-E2 — Ceremonial Beast Mechanics

### 目标

Ceremonial Beast 当前可能从 30% 掉到 0%，需要明确建模它的 lock/stun/窗口机制。
任务不是硬编码“遇到 boss 打某张牌”，而是把机制状态和动作价值暴露给模型。

### Owned paths

```text
packages/rl-agent/sts2_env/observation_v3.py
packages/rl-agent/sts2_env/semantic_action.py
packages/rl-agent/muzero/train.py
packages/rl-agent/tests/test_ceremonial_mechanics.py
mods/sts2-bridge/Scripts/BridgeGameApi.EnvPayloads.cs
```

### 必需字段

```json
{
  "boss_mechanics": {
    "ceremonial": {
      "active": true,
      "one_card_lock_active": true,
      "stun_window_active": false,
      "actions_remaining_this_turn": 1,
      "lock_counter": 2,
      "stun_counter": 0,
      "incoming_after_lock": 18
    }
  }
}
```

每个 action：

```json
{
  "mechanism": {
    "ceremonial_single_action_impact_score": 0.82,
    "ceremonial_wastes_one_card_lock": false,
    "ceremonial_uses_stun_window": true,
    "ceremonial_sets_up_stun": false
  }
}
```

### single_action_impact_score

one-card lock 下，动作质量不能只按普通 damage/block。应考虑：

- 伤害是否高。
- 是否 lethal。
- 是否 block lethal。
- 是否弱化 boss 大攻击。
- 是否触发 stun 或利用 stun。
- 是否 draw/create 但因 lock 不能继续使用，导致低价值。
- 是否浪费 key card。

### offender

```text
ceremonial_low_impact_under_lock
ceremonial_missed_stun_window
ceremonial_bad_end_turn_under_lock
```

### 指标

```text
boss_combat/ceremonial_beast_boss/one_card_lock_active_rate
boss_combat/ceremonial_beast_boss/stun_window_active_rate
boss_combat/ceremonial_beast_boss/low_impact_under_lock_selected_rate
boss_combat/ceremonial_beast_boss/high_impact_under_lock_selected_rate
boss_combat/ceremonial_beast_boss/missed_stun_window_rate
```

### 测试

1. one-card lock active 时，低伤害 Strike 被标低 impact。
2. one-card lock active 时，lethal action 高 impact。
3. stun window active 时，不利用窗口的 end_turn/off-action 标 offender。
4. draw-only action 在 one-card lock 下如果无法继续打，impact 降低。

---

## TASK-E3 — Insatiable Offenders

### 目标

The Insatiable 回归明显。需要专项 offender dump 定位它是否因为：

- 过度 strategic skip。
- refund/no-followup 误判。
- 不处理吞噬/成长/压力机制。
- bad end_turn。

### Owned paths

```text
packages/rl-agent/sts2_env/observation_v3.py
packages/rl-agent/sts2_env/semantic_action.py
packages/rl-agent/muzero/train.py
packages/rl-agent/tests/test_insatiable_offenders.py
```

### 必需字段

由于具体机制可能随版本调整，先从 bridge payload / powers 中结构化导出：

```json
{
  "boss_mechanics": {
    "insatiable": {
      "active": true,
      "pressure": 0.7,
      "growth_or_scaling": 2,
      "special_counter": 1,
      "urgent_window": false,
      "powers": []
    }
  }
}
```

### offender

```text
insatiable_strategic_skip
insatiable_refund_no_followup
insatiable_bad_end_turn
insatiable_missed_pressure_window
```

### 指标

```text
boss_combat/the_insatiable_boss/strategic_skip_selected_rate
boss_combat/the_insatiable_boss/refund_no_followup_low_value_selected_rate
boss_combat/the_insatiable_boss/bad_end_turn_selected_rate
boss_combat/the_insatiable_boss/missed_pressure_window_rate
boss_combat/the_insatiable_boss/pressure_action_selected_rate
```

### 验收

短跑后能回答：

- Insatiable 输是因为不打、乱打、机制窗口错过，还是 value/planner 低估？
- strategic skip 是否在该 encounter 过高？
- refund-no-followup 是否在该 encounter 过高？
