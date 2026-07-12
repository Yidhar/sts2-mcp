# 01 — Diagnostics First

诊断优先阶段只允许新增观测、dump、指标、测试，不允许改变策略分数或 reward。
原因：当前有多个 detector 口径冲突，先改策略会让问题更难定位。

---

## TASK-A1 — Selected End Turn Context Dump

### 目标

当模型选择 `end_turn` 时，把当时 root context 完整 dump 到 JSONL。
这个 dump 必须能回答：

- 是真的没有动作，还是 transient only-end-turn？
- 有能量吗？
- 有正收益动作吗？
- 有 urgent/mandatory 行动吗？
- 有战略性不打的合理理由吗？
- 有 X-cost / refund / exhaust / boss mechanism 相关候选吗？
- top action 分数为什么没选其它动作？

### Owned paths

```text
packages/rl-agent/muzero/train.py
packages/rl-agent/tests/test_end_turn_context_dump.py
```

如当前测试目录不在该路径，请按 repo 现有结构放置，但测试名保持可搜索。

### 输出文件

```text
packages/rl-agent/logs_muzero/<run>/diagnostics/end_turn_contexts.jsonl
```

每行一个 JSON object。不要写巨大 nested raw obs，字段要压缩但足够诊断。

### 必需字段

```json
{
  "global_step": 12345,
  "episode_id": "combat-...",
  "encounter_id": "kaiser_crab_boss",
  "tier": "boss",
  "turn": 3,
  "selected_action_id": "...",
  "selected_family": "end_turn",
  "end_turn_class": "forced_end_turn|strategic_defer_end_turn|bad_end_turn|unknown",
  "reason_flags": {
    "no_legal_positive_action": false,
    "transient_only_end_turn": false,
    "has_energy_and_positive_action": true,
    "has_urgent_or_mandatory_action": true,
    "has_strategic_defer_reason": false,
    "boss_mechanic_risk": true,
    "x_cost_zero_energy_window": false
  },
  "player": {
    "hp": 42,
    "max_hp": 80,
    "block": 3,
    "energy": 2
  },
  "combat": {
    "incoming_damage": 18,
    "hand_count": 5,
    "draw_count": 12,
    "discard_count": 7,
    "exhaust_count": 2
  },
  "counts": {
    "legal_action_count": 8,
    "playable_cards_left": 4,
    "positive_action_count": 3,
    "urgent_action_count": 1,
    "mandatory_action_count": 0,
    "deferable_action_count": 1,
    "strategic_skip_candidate_count": 1,
    "refund_candidate_count": 1,
    "refund_no_followup_candidate_count": 0,
    "x_cost_candidate_count": 1,
    "zero_energy_x_cost_candidate_count": 0
  },
  "boss_context": {
    "kaiser_back_attack_risk": 1.0,
    "kaiser_facing_change_candidate_count": 1,
    "ceremonial_one_card_lock_active": false,
    "insatiable_pressure": 0.0
  },
  "top_legal_actions": [
    {
      "rank": 1,
      "action_id": "...",
      "family": "play_card",
      "card_id": "...",
      "title": "...",
      "target": "...",
      "score": 0.12,
      "policy_logit": 1.7,
      "quality_bias": 0.4,
      "q": -0.1,
      "risk_q": 0.2,
      "uncertainty": 0.0,
      "tags": ["block", "kaiser_facing_change"]
    }
  ]
}
```

### 实现要求

1. 复用或抽取当前 `_root_end_turn_context()` 逻辑，不要复制粘贴两套条件。
2. selected action tracker 必须调用与 `_combat_action_quality_bias()` 相同的 context/classifier。
3. JSONL 写入要有采样控制：
   - 默认只 dump selected end_turn。
   - 可通过参数打开 dump all offender。
   - 单文件过大时滚动或限制 max lines。
4. dump 写入失败不能中断训练，但要有 warning。

### 非目标

- 不改 reward。
- 不改 action score。
- 不改 planner 权重。

### 测试

`test_end_turn_context_dump.py` 至少覆盖：

1. 有能量、有 positive action、选择 end_turn => dump `bad_end_turn`。
2. 无能量、无 positive action、选择 end_turn => dump `forced_end_turn`。
3. 有 strategic defer reason、选择 end_turn => dump `strategic_defer_end_turn`。
4. transient only-end-turn 标记存在时 => reason flag 正确。

---

## TASK-A2 — Action Offender Dump and Metric Namespace

### 目标

不仅 dump end_turn，还要 dump 所有“看起来可疑但可能是策略”的动作，让后续训练回归可以用证据定位。

### Owned paths

```text
packages/rl-agent/muzero/train.py
packages/rl-agent/tests/test_action_offender_metrics.py
```

### 输出文件

```text
packages/rl-agent/logs_muzero/<run>/diagnostics/action_offenders.jsonl
```

### offender types

至少支持：

```text
bad_end_turn
strategic_defer_end_turn
strategic_skip_selected
refund_no_followup_selected
zero_energy_x_cost_selected
x_cost_low_value_selected
low_quality_potion_selected
high_save_value_potion_unused
kaiser_risky_end_turn
kaiser_facing_missed
ceremonial_low_impact_under_lock
ceremonial_missed_stun_window
insatiable_strategic_skip
```

### 命名空间要求

所有 boss combat 指标至少 emit 两层：

```text
boss_combat/<metric>
boss_combat/<encounter_id>/<metric>
```

例如：

```text
boss_combat/bad_end_turn_selected_rate
boss_combat/kaiser_crab_boss/bad_end_turn_selected_rate
boss_combat/kaiser_crab_boss/kaiser_facing_change_selected_rate
```

不要只在 per-encounter emit 新指标，也不要只在 global emit 新指标。

### JSONL 必需字段

```json
{
  "global_step": 12345,
  "episode_id": "...",
  "encounter_id": "ceremonial_beast_boss",
  "tier": "boss",
  "turn": 4,
  "offender_type": "ceremonial_low_impact_under_lock",
  "selected_action_id": "...",
  "selected_family": "play_card",
  "selected_card_id": "...",
  "selected_title": "...",
  "reason_flags": {},
  "state_summary": {},
  "alternative_actions": []
}
```

### 测试

1. encounter id 存在时同时生成 global 和 encounter metric。
2. encounter id 缺失时只生成 global，不 crash。
3. bad end turn offender 能写入 JSONL。
4. Kaiser/Ceremonial offender 不在非对应 encounter 误触发。

---

## TASK-A3 — X-Cost Dynamic Diagnostics

### 目标

确认 X-cost 动作不是被初始 3 能量或静态 cost 误导。
模型和 planner 必须看到“当前有效 X 值”，并且指标能显示 0 能量打 X 费牌是否合理。

### Owned paths

```text
packages/rl-agent/muzero/train.py
packages/rl-agent/sts2_env/semantic_action.py
packages/rl-agent/sts2_env/observation_v3.py
packages/rl-agent/tests/test_x_cost_dynamic_energy.py
```

### 必需 action semantic 字段

```json
{
  "cost": {
    "base_cost": -1,
    "current_cost": -1,
    "effective_cost": 0,
    "is_x_cost": true,
    "x_cost_effective_energy": 0,
    "x_cost_has_non_energy_effect": false,
    "energy_before": 0,
    "energy_after_estimate": 0
  },
  "expected_effect": {
    "damage": 0,
    "block": 0,
    "draw": 0,
    "energy_gain": 0,
    "card_create": 0,
    "mechanism_value": 0
  }
}
```

### 指标

```text
combat/x_cost_available_rate
combat/x_cost_selected_rate
combat/zero_energy_x_cost_available_rate
combat/zero_energy_x_cost_selected_rate
combat/x_cost_selected_effective_energy_mean
combat/x_cost_bad_selected_rate

boss_combat/<same>
boss_combat/<encounter_id>/<same>
```

### 判定建议

`zero_energy_x_cost_selected` 不一定永远错，但需要满足至少一个条件才不是 offender：

- 该 X 费牌 0 能量有非能量效果。
- 有 relic/power/card modifier 让 X 费不按当前 energy 计算。
- 打出该牌触发重要机制，如 facing change、stun、artifact strip。
- 打出该牌是为了 exhaust/hand-size/pile manipulation，并且有明确后续收益。

否则标记：

```text
zero_energy_x_cost_selected
x_cost_low_value_selected
```

### 测试

1. energy=0，X-cost 无非能量效果 => bad。
2. energy=2，X-cost damage scales => effective energy=2。
3. energy=0，但 X-cost 有机制标签 => 不直接 bad，但要记录。

---

## TASK-A4 — Potion Transition Diagnostics

### 目标

解决“模型使用过药水，但死亡时 potion_unused_on_death 仍为 1”这类观测冲突。
必须区分：

- action 真的执行失败。
- bridge slot 没清。
- 死亡 final obs 刷新时机错误。
- snapshot 注入的 potion 状态被错误复用。

### Owned paths

```text
packages/rl-agent/muzero/train.py
packages/rl-agent/sts2_env/combat_env.py
packages/rl-agent/sts2_env/headless_sim_bridge_client.py
mods/sts2-bridge/Scripts/BridgeGameApi.cs
mods/sts2-bridge/Scripts/BridgeGameApi.PotionProfiles.cs
packages/rl-agent/tests/test_potion_transition_diagnostics.py
```

### 必需诊断字段

在 use_potion 前后记录：

```json
{
  "event": "use_potion_transition",
  "global_step": 123,
  "episode_id": "...",
  "action_id": "...",
  "potion_slot": 1,
  "potion_id_before": "...",
  "potion_title_before": "...",
  "execute_ok": true,
  "bridge_error": null,
  "state_version_before": 100,
  "state_version_after": 101,
  "potion_slot_after": {
    "empty": true,
    "id": null,
    "title": null,
    "is_usable": false,
    "is_queued": false
  }
}
```

死亡 final info 记录：

```json
{
  "event": "death_final_potions",
  "loss": true,
  "final_potion_count": 1,
  "used_potion_count": 1,
  "potion_dump": []
}
```

### bridge 侧建议

如果游戏内部使用药水后 slot 没被置空，bridge payload 仍要提供：

```json
{
  "empty": true,
  "is_usable": false,
  "was_used_this_combat": true
}
```

但更推荐从真实 `PotionSlots` 生命周期修复，避免 Python 侧猜。

### 测试

1. use_potion execute ok 后 slot 为空。
2. execute fail 后不计入 `family_use_potion_rate` 成功使用。
3. death final obs 不把已用药水算作 unused。
