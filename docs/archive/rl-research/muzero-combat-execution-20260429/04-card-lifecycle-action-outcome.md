# 04 — Card Lifecycle and Action Outcome

本文件处理卡牌生命周期、动作后果和 future-world aux。
核心观点：手牌/抽牌堆/弃牌堆/消耗堆/费用/保留/附魔/变化/复制等信息必须结构化进入模型，而不是靠文本正则。

---

## TASK-D1 — CardEffectProfile Field Coverage

### 目标

为铁甲战士 88 张牌和无色牌导出结构化 card effect profile，覆盖：

- 手牌状态变化。
- 升级/强化。
- 消耗/虚无/保留/重放。
- 变化/复制/创建。
- 费用改变/回费/X-cost。
- pile movement。
- 附魔/card modifiers。
- boss mechanism relevant effect。

### Owned paths

```text
packages/rl-agent/sts2_env/card_effect_profile.py
packages/rl-agent/sts2_env/generate_card_effect_profiles.py
packages/rl-agent/sts2_env/audit_card_mechanism_coverage.py
mods/sts2-bridge/Scripts/BridgeGameApi.CardEffectProfiles.cs
docs/generated/card-effect-profile-coverage.md
docs/generated/card-effect-profile-schema.md
```

如果这些文件不存在，应按 repo 风格创建，但不要引入额外大框架。

### Profile schema

建议结构：

```json
{
  "card_id": "IRONCLAD_ARMAMENTS",
  "title": "Armaments",
  "color": "red",
  "type": "skill",
  "rarity": "common",
  "cost": {
    "base": 1,
    "upgraded": 1,
    "is_x_cost": false,
    "can_change_cost": false,
    "cost_reduction_tags": []
  },
  "lifecycle": {
    "exhausts_on_play": false,
    "ethereal": false,
    "retain": false,
    "self_purge": false,
    "returns_to_hand": false,
    "replay_or_duplicate": false
  },
  "hand_mutation": {
    "upgrades_hand": true,
    "upgrade_targets": "one_or_all_by_upgrade_state",
    "transforms_cards": false,
    "copies_cards": false,
    "creates_cards": false,
    "discard_hand": false,
    "draw": 0,
    "select_cards": {
      "enabled": true,
      "min": 1,
      "max": 1,
      "target_zone": "hand"
    }
  },
  "pile_mutation": {
    "moves_to_exhaust": false,
    "moves_to_discard": true,
    "shuffles_into_draw": false,
    "puts_card_on_top": false,
    "removes_card_from_combat": false
  },
  "combat_effect": {
    "damage": 0,
    "block": "dynamic",
    "weak": 0,
    "vulnerable": 0,
    "strength": 0,
    "energy_gain": 0
  },
  "mechanism_effect": {
    "can_change_facing": true,
    "can_strip_artifact": false,
    "can_trigger_stun": false,
    "one_card_lock_impact": "medium"
  },
  "source": {
    "primary": "game_internal_id",
    "fallback_text_regex_used": false
  }
}
```

### 字段来源优先级

1. 游戏内部 card data / effect components / power ids / card modifiers。
2. bridge C# 反射/显式 profile。
3. 维护型手写 override table。
4. 文本正则 fallback。

文本正则只能 fallback，并且必须标记：

```json
"fallback_text_regex_used": true
```

### 覆盖审计

生成：

```text
docs/generated/card-effect-profile-coverage.md
```

必须包含：

- Ironclad 88/88 coverage。
- Colorless coverage。
- 每个字段组 coverage percentage。
- fallback regex 使用列表。
- unknown/needs_manual_override 列表。

### 测试

1. Armaments：升级前 one hand target，升级后 all hand target。
2. Exhaust card：`exhausts_on_play=true` 且 pile move 到 exhaust。
3. Ethereal/Void 类：生命周期字段正确。
4. Refund card：energy_gain 字段正确。
5. X-cost：`is_x_cost=true`。
6. Enchantment/card modifier：profile 能暴露 modifier id 或 tags。

---

## TASK-D2 — Observation and Action Lifecycle Tokens

### 目标

让模型在一次出牌决策时看到：

- 当前手牌每张牌的生命周期风险和动作后果。
- 抽牌堆/弃牌堆/消耗堆中关键牌分布。
- 药水/遗物/能量/血量/敌人状态对动作价值的影响。

### Owned paths

```text
packages/rl-agent/sts2_env/observation_v3.py
packages/rl-agent/sts2_env/semantic_action.py
packages/rl-agent/sts2_env/card_effect_profile.py
packages/rl-agent/tests/test_card_lifecycle_tokens.py
```

### Observation token 增量

为 card token / pile token / action token 添加：

```text
is_exhaust
is_ethereal
is_retain
is_void_or_status
is_x_cost
base_cost
current_cost
effective_cost
energy_gain
draw_count
create_count
upgrade_hand
transform_hand
copy_card
replay_or_duplicate
cost_reduction
requires_card_selection
selected_cards_min
selected_cards_max
expected_pile_destination
hand_mutation_score
future_cycle_value
```

### Pile summary

至少提供：

```text
draw_pile_count
discard_pile_count
exhaust_pile_count
important_draw_pile_count
important_discard_pile_count
important_exhaust_pile_count
cycle_density_attack
cycle_density_block
cycle_density_draw
cycle_density_energy
exhausted_key_card_count
```

### Action token

action token 需要表达“打出这张牌后会发生什么”，不仅是“当前牌是什么”：

```text
action_exhausts_card
action_changes_hand
action_draws_cards
action_creates_cards
action_refunds_energy
action_reduces_cost
action_requires_selection
action_changes_facing
action_has_boss_mechanism_value
action_expected_followup_count
```

### 测试

1. 武装升级前/后 action token 不同。
2. 0 能量 X-cost action token 显示 effective X=0。
3. 消耗牌 action token 显示 destination=exhaust。
4. 回费牌 action token 显示 energy_gain 和 expected_followup_count。
5. retain/ethereal/status/void 能进入 token。

---

## TASK-D3 — Future-World Card Lifecycle Aux

### 目标

让世界模型学习“打出动作后游戏状态如何变化”，特别是卡牌生命周期相关的变化。
这样 search-free planner 才能在没有 MCTS 时做近似 lookahead。

### Owned paths

```text
packages/rl-agent/muzero/train.py
packages/rl-agent/sts2_env/observation_v3.py
packages/rl-agent/tests/test_future_world_aux_card_lifecycle.py
```

### Aux targets

对每个 transition 预测：

```text
next_hand_count
next_draw_count
next_discard_count
next_exhaust_count
next_energy
next_block
next_incoming_damage
card_moved_to_exhaust_prob
card_moved_to_discard_prob
card_retained_prob
hand_upgraded_count
hand_transformed_count
hand_copied_count
cost_reduced_count
created_card_count
drawn_card_count
next_kaiser_facing
next_back_attack_risk
next_ceremonial_lock_state
```

### Loss/metrics

```text
loss/future_lifecycle_aux
aux/next_hand_count_mae
aux/next_draw_count_mae
aux/next_discard_count_mae
aux/next_exhaust_count_mae
aux/card_moved_to_exhaust_bce
aux/next_energy_mae
aux/next_boss_mechanic_state_acc
```

### 注意

这些 aux 不是为了让 loss 好看，而是为了 planner：

- 当 `latent_drift` 高时，应降低 rollout Q。
- 当 lifecycle aux 错误高时，应降低涉及 pile/hand mutation 的 lookahead。

### 测试

1. 打出普通 Strike：hand-1，discard+1。
2. 打出 exhaust card：hand-1，exhaust+1。
3. 打出 draw card：hand 变化考虑 draw。
4. 打出 Armaments：hand_upgraded_count 正确。
5. 打出 facing target action：next_kaiser_facing 正确。
