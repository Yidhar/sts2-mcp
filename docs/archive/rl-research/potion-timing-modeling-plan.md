# STS2 药水使用时机建模目标方案

> 目标：让模型不再把“药水”理解成一个统一的正收益按钮，而是把 64 种药水作为不同的战斗工具建模；通过结构化药水画像、状态 token、动作 token、timing auxiliary targets 和 combat sandbox 课程，让 search-free / attention-first planner 学会“什么时候用、什么时候留、为什么现在用是浪费”。

更新时间：2026-04-26
适用范围：`packages/rl-agent`、`mods/sts2-bridge`、`docs`、combat sandbox 训练链路。

---

## 1. 背景问题

当前训练中观察到一个策略异常：

- 模型基本“有药水就会用”；
- 不擅长等待正确使用契机；
- 资源类药水可能在没有后续动作时被浪费；
- 伤害药水可能在低威胁/明显 overkill 场景中提前使用；
- 防御药水可能在没有真实 incoming threat 时被错误使用；
- Kaiser/Crusher/Rocket 等面向机制中，targeted potion 本应可以通过攻击另一侧敌人改变 player facing，但如果不显式建模 target 与背刺方向，模型很难学到。

这不是单纯“策略头弱”或“奖励没调好”的问题，而是数据契约和训练目标把药水表达得过粗。

当前 Bridge 侧 `BuildUsePotionSemantic()` 基本只暴露：

```csharp
family = "use_potion";
roles = new[] { "potion" };
potion_id = potion.Id.ToString();
rarity = potion.Rarity.ToString();
```

当前 `BuildPotionPayload()` 主要暴露：

```csharp
id
title
description
rarity
target_type
selection_screen_prompt
can_throw_at_ally
is_usable
is_queued
has_been_removed_from_state
```

Python 侧虽然已经有 `_potion_timing_profile()` 尝试推断：

- damage / block / heal / draw / energy_gain；
- lethal / overkill；
- prevent_lethal / prevent_major_loss；
- resource followup；
- Kaiser facing / mechanism answer；
- save_recommended / no_followup；

但这些值目前很多时候依赖 action numeric、roles、text 或 description 推断。也就是说，模型和 trainer 很多时候拿不到稳定结构化药水效果，只能猜。

结果就是：

> 对模型来说，“use_potion”经常只是一个可用、稀有、似乎有正收益的动作；只要没有明确的浪费/留存信号，它就容易有就用。

---

## 2. 总体设计原则

### 2.1 不把策略写死

不要写成：

- 火焰药水某些场景必须用；
- 能量药水某些场景禁止用；
- boss 战药水全加分；
- normal 战药水全降分。

这会形成脆弱的规则堆叠。

### 2.2 结构化“药水能力”，让模型学习“药水时机”

正确目标是表达：

1. 这瓶药水是什么工具；
2. 它影响哪些资源；
3. 它依赖哪些上下文；
4. 当前使用是否有 immediate value；
5. 当前使用是否有 waste risk；
6. 留到之后是否有 save value；
7. 是否能回答当前 boss / elite 机制。

最终让模型通过 attention 看到：

```text
药水 token
  ↔ 当前手牌 token
  ↔ 抽牌堆 / 弃牌堆 / 消耗牌堆 token
  ↔ 当前能量 / 费用 token
  ↔ 敌人意图 / buff / debuff token
  ↔ boss 机制 token
  ↔ 遗物 token
  ↔ legal action token
```

并学习 action-level Q-like / lookahead value。

---

## 3. 目标态架构

目标链路：

```text
64 种药水静态 registry
        ↓
Bridge 输出 effect_profile / semantic_tags / timing_tags
        ↓
Observation potion token + use_potion action token 消费同一套 profile
        ↓
_potion_timing_profile(effect + combat context) 生成 use_quality / waste_risk / save_value
        ↓
Direct planner 短期使用 timing bias 稳定训练
        ↓
Aux heads 学习药水时机
        ↓
Combat sandbox 生成药水正反 curriculum 场景
        ↓
逐步退火硬 bias，让模型自己接管
```

---

## 4. 静态药水画像 Registry

### 4.1 文件建议

新增：

```text
packages/rl-agent/content/potions.timing.generated.json
packages/rl-agent/content/potions.timing.overrides.json
packages/rl-agent/sts2_env/potion_profiles.py
tools/generate_potion_profiles.py
```

可选 Bridge 生成文件：

```text
mods/sts2-bridge/Scripts/PotionEffectProfiles.generated.cs
```

### 4.2 覆盖范围

当前游戏导出中药水总数为 64。

Registry 必须覆盖全部 potion id：

- Common / Uncommon / Rare / Event / Token / None；
- 常规掉落池药水；
- Event 药水；
- Token 药水；
- Deprecated 药水。

其中：

```text
POTION.DEPRECATED_POTION
```

必须标记为：

```json
{
  "enabled_for_training": false,
  "deprecated": true
}
```

### 4.3 Registry schema

建议每个药水条目形态如下：

```json
{
  "id": "POTION.FIRE_POTION",
  "title": "...",
  "rarity": "Common",
  "target_scope": "AnyEnemy",
  "enabled_for_training": true,

  "effect_family": [
    "damage",
    "single_target"
  ],

  "effect_profile": {
    "damage": 20.0,
    "block": 0.0,
    "draw": 0.0,
    "energy_gain": 0.0,
    "heal": 0.0,
    "weak": 0.0,
    "vulnerable": 0.0,
    "poison": 0.0,
    "strength": 0.0,
    "dexterity": 0.0,
    "intangible": 0.0,
    "prevent_damage": 0.0,
    "generate_card_count": 0.0,
    "discover_count": 0.0,
    "upgrade_hand": 0.0,
    "duplicate_next": 0.0,
    "retrieve_from_discard": 0.0,
    "replace_or_transform_hand": 0.0,
    "aoe": false,
    "single_target": true,
    "random_target": false,
    "target_required": true,
    "can_change_facing_if_targeted_enemy": true
  },

  "semantic_tags": [
    "attack",
    "damage",
    "single_target"
  ],

  "timing_tags": [
    "lethal_tool",
    "overkill_risk",
    "mechanism_answer_candidate"
  ],

  "training_tags": [
    "combat_immediate",
    "save_if_low_threat"
  ]
}
```

### 4.4 生成 + override 策略

不建议完全靠中文描述 regex，也不建议全部手写。

建议：

1. 从游戏导出 `items.json` 生成基础条目：
   - id；
   - title；
   - rarity；
   - target；
   - description / summary。
2. 用 regex 抽取容易稳定识别的数值：
   - `造成 X 点伤害`；
   - `获得 X 点格挡`；
   - `抽 X 张牌`；
   - `回复 X 点生命`；
   - `获得 X 点能量`；
   - `给予 X 层虚弱 / 易伤 / 中毒`。
3. 用 override 精确修正复杂效果：
   - 发现牌；
   - 生成随机牌；
   - 复制下一张；
   - 手牌升级；
   - 手牌替换；
   - 药水栏位变化；
   - 事件药水；
   - Token 药水；
   - 被动 / 触发型药水；
   - Deprecated 药水。
4. 生成 Bridge C# 和 Python JSON 使用同源数据，避免两边语义漂移。

### 4.5 validation

新增校验：

```text
export/items.json 中 potions 数量 == registry 数量 == 64
所有 dataset 中出现的 potion id 均能查到 profile
POTION.DEPRECATED_POTION 不参与训练正样本
Bridge 输出的 potion effect_profile 与 Python registry 一致
```

---

## 5. 药水能力族设计

### 5.1 伤害类

能力：

```text
damage
aoe_damage
single_target_damage
multi_hit
random_target
poison_damage
```

关键 timing：

```text
lethal_tool
overkill_risk
high_threat_target_removal
mechanism_answer_candidate
can_change_facing_if_targeted_enemy
```

使用价值高的情况：

- 能斩杀；
- 能打掉高威胁单位；
- boss / elite 中能显著压低回合数；
- Kaiser 面向机制下能通过 targeted enemy action 改变 player facing；
- AOE 能清多个敌人。

浪费风险高的情况：

- 普通低威胁战；
- 明显 overkill；
- 目标没有威胁；
- 机制上没有收益。

---

### 5.2 防御 / 保命类

能力：

```text
block
intangible
prevent_damage
heal
regen
weak
artifact / cleanse-like
```

关键 timing：

```text
prevent_lethal_tool
prevent_major_loss_tool
incoming_damage_dependency
block_waste_risk
save_if_no_incoming
```

使用价值高的情况：

- `incoming - current_block` 很大；
- 能防止死亡；
- 能显著降低精英/普通战战损；
- boss 高压攻击回合中防止死亡或机制崩盘。

浪费风险高的情况：

- 敌人不攻击；
- 当前 block 已足够；
- 只是 boss 战中很小的非致命战损，且 boss 后会回血；
- 没有后续收益的低压回合。

---

### 5.3 资源类

能力：

```text
energy_gain
draw
discard_then_draw
generate_cards
discover_cards
cost_reduction
free_play
retrieve_from_discard
```

关键 timing：

```text
requires_followup
requires_hand_context
requires_draw_pile_context
requires_discard_context
waste_if_no_playable_followup
combo_resource_tool
```

使用价值高的情况：

- 当前能量不足，但使用后能打出高价值牌；
- 当前手牌少，抽牌堆有高价值牌；
- 当前弃牌堆有关键牌可回收；
- 当前 boss/elite 高压，需要爆发；
- 当前手牌与生成/发现/回收效果有明确配合。

浪费风险高的情况：

- 使用后仍没有可打牌；
- 当前手牌无高价值 followup；
- 抽牌堆低质量或手牌已满；
- 0 能量使用 X 费牌/资源药水但无实际收益；
- 低威胁回合提前消耗稀缺资源。

---

### 5.4 手牌状态改变类

能力：

```text
upgrade_hand
add_replay
add_ethereal
add_retain
replace_cards
transform_cards
copy_next_card
set_cost_zero
randomize_cost
exhaust_hand
```

关键 timing：

```text
requires_good_hand
bad_on_empty_hand
requires_card_selection
hand_context_dependency
setup_tool
random_outcome
```

使用价值高的情况：

- 当前手牌中有高价值未升级牌；
- 当前手牌中有值得复制/重放/保留的关键牌；
- 当前回合能立即利用升级/复制/降费后的收益；
- 当前手牌很差，需要替换/发现；
- boss/elite 需要强力 setup。

浪费风险高的情况：

- 当前手牌没有目标；
- 手牌全是低价值牌 / 状态牌 / 已升级牌；
- 使用后没有能量打 followup；
- 效果随机且当前低威胁。

这类能力要与卡牌状态改变逻辑共用，不应该只服务药水。后续卡牌如“强化手牌”“赋予虚无/重放/保留”“替换手牌”等也应进入同一套 profile。

---

### 5.5 长期价值 / 非即时战斗类

能力：

```text
max_hp_gain
potion_slot_delta
long_term_buff
passive_or_triggered
event_value
token_value
```

关键 timing：

```text
long_term_value
post_combat_value
not_combat_immediate
save_or_use_outside_combat
```

处理原则：

- 不应把这类药水默认当成 combat immediate positive；
- combat sandbox 中需要小心，因为 sandbox 缺少 full-run 后续价值；
- 如果 Bridge 暴露为可用动作，需要明确 `combat_immediate_value = low` 或 `passive_or_triggered = true`。

---

### 5.6 机制应答类

重点机制：Kaiser / Rocket / Crusher / Crab boss 的背刺面向。

已确认：

- `enemy.side` / `target_side` 是 Player/Enemy 阵营，不是 left/right；
- 左右部位来自 enemy powers：
  - `BACK_ATTACK_LEFT_POWER`；
  - `BACK_ATTACK_RIGHT_POWER`；
- 转身不是一个单独 action；
- 任意 targeted enemy card / potion，只要目标在当前 facing 的另一侧，就可能改变 facing。

因此药水 profile 中只需要表达：

```text
targeted_enemy_action = true
can_change_facing_if_targeted_enemy = true
```

真正判断 facing change 必须在 Python 侧根据：

```text
current player facing
action.target_combat_id
target enemy powers
```

计算：

```python
target_position != current_facing
```

不能再读 `target_side == left/right`。

---

## 6. Bridge 数据契约改造

### 6.1 修改点

文件：

```text
mods/sts2-bridge/Scripts/BridgeGameApi.cs
mods/sts2-bridge/Scripts/BridgeGameApi.EnvPayloads.cs
```

重点函数：

```text
BuildPotionPayload(PotionModel? potion)
BuildUsePotionSemantic(PotionModel? potion)
AddCombatPotionActions(...)
```

### 6.2 `BuildPotionPayload()` 增加字段

目标 payload：

```json
{
  "id": "POTION.FIRE_POTION",
  "title": "...",
  "description": "...",
  "rarity": "Common",
  "target_type": "AnyEnemy",
  "selection_screen_prompt": "...",
  "can_throw_at_ally": false,
  "is_usable": true,
  "is_queued": false,
  "has_been_removed_from_state": false,

  "effect_profile": {
    "damage": 20.0,
    "block": 0.0,
    "draw": 0.0,
    "energy_gain": 0.0,
    "heal": 0.0,
    "weak": 0.0,
    "vulnerable": 0.0,
    "poison": 0.0,
    "aoe": false,
    "single_target": true,
    "target_required": true,
    "random_target": false,
    "can_change_facing_if_targeted_enemy": true,
    "requires_followup": false,
    "long_term_value": false,
    "passive_or_triggered": false
  },

  "semantic_tags": [
    "damage",
    "attack",
    "single_target"
  ],

  "timing_tags": [
    "lethal_tool",
    "overkill_risk",
    "mechanism_answer_candidate"
  ]
}
```

### 6.3 `BuildUsePotionSemantic()` 增加 roles

当前只有：

```csharp
roles = new[] { "potion" }
```

目标是从 profile 映射到现有 semantic role，不改 action feature dim：

| potion effect | role |
|---|---|
| damage | attack |
| block / intangible / prevent_damage | block |
| draw | draw |
| energy / generate / discover / retrieve | resource |
| weak / vulnerable / poison | debuff |
| strength / dexterity / scaling | buff / scaling |
| heal | heal |
| AOE | aoe |
| hand transform / upgrade / duplicate / setup | setup |

示例：

```json
{
  "family": "use_potion",
  "roles": ["potion", "attack"],
  "potion_id": "POTION.FIRE_POTION",
  "rarity": "Common",
  "effect_family": ["damage", "single_target"],
  "timing_tags": ["lethal_tool", "overkill_risk"]
}
```

注意：如果当前 `SEMANTIC_ROLE_NAMES` 没有 `potion`，则 `potion` 可以只保留在 raw semantic tags 中，role bit 只投影到现有 role enum。

---

## 7. Observation / Action Token 改造

### 7.1 修改点

文件：

```text
packages/rl-agent/sts2_env/observation_v3.py
packages/rl-agent/sts2_env/semantic_action.py
packages/rl-agent/content_registry.py
packages/rl-agent/sts2_env/potion_profiles.py
```

### 7.2 `_source_profile()` 支持 potion profile

当前 `_source_profile()` 偏 card preview。目标：

```python
def _source_profile(self, source):
    if is_potion_payload(source):
        return merge_default_profile(get_potion_profile(source["id"]), source.get("effect_profile"))
    ...
```

合并优先级：

```text
Bridge live effect_profile > Python timing registry > text regex fallback > zero profile
```

### 7.3 `_potion_numeric()` 补充 numeric slots

`TOKEN_NUMERIC_DIM = 96`，不需要改 input shape。建议复用现有前 0-10 槽，并在后续追加：

| slot | meaning |
|---:|---|
| 0 | relevance |
| 1 | damage normalized |
| 2 | block normalized |
| 3 | draw normalized |
| 4 | energy_gain normalized |
| 5 | heal normalized |
| 6 | debuff normalized |
| 7 | single_target |
| 8 | aoe_target |
| 9 | slot_index_norm |
| 10 | non_discard_hint |
| 11 | poison |
| 12 | weak |
| 13 | vulnerable |
| 14 | strength_gain |
| 15 | dexterity_gain |
| 16 | intangible / prevent_damage |
| 17 | generate_card_count |
| 18 | discover_count |
| 19 | upgrade_hand |
| 20 | duplicate_next |
| 21 | retrieve_from_discard |
| 22 | replace_or_transform_hand |
| 23 | random_outcome |
| 24 | requires_followup |
| 25 | save_if_low_threat |
| 26 | long_term_value |
| 27 | passive_or_triggered |
| 28 | can_change_facing_if_targeted_enemy |
| 29 | target_required |
| 30 | potion_slot_value |
| 31 | deprecated_or_disabled |
| 32 | setup_tool |
| 33 | scaling_tool |
| 34 | mechanism_answer_candidate |
| 35 | hand_context_dependency |
| 36 | discard_context_dependency |
| 37 | draw_pile_context_dependency |
| 38 | exhaust_pile_context_dependency |

这让模型在 potion world token 和 use_potion action token 中看到一致的能力描述。

### 7.4 `semantic_action._infer_roles()` 消费 potion metadata

当前 `_infer_roles()` 可以从 metadata 的 `semantic_tags` 推出一些 role，但 potion metadata 缺 profile 时效果有限。

目标：

```python
metadata = get_potion_metadata(potion_id)
profile = get_potion_profile(potion_id)
roles = infer_roles_from_effect_profile(profile)
```

映射到现有 role enum，不新增 action dim。

### 7.5 `build_live_potion_semantic_text()` 输出 compact profile

当前 runtime summary 会尝试读取：

```text
damage, block, draw, weak, vulnerable, heal, hp_loss, strength, dexterity, summon
```

目标是加入 effect_profile / timing_tags：

示例输出：

```text
Fire Potion | Common AnyEnemy | sig dmg=20 target=single | timing lethal overkill mechanism_candidate | 造成20点伤害
```

这可以服务 text trunk，但结构化 numeric 仍然是主通道。

---

## 8. Potion Timing Evaluator

### 8.1 修改点

文件：

```text
packages/rl-agent/muzero/train.py
```

当前函数：

```python
def _potion_timing_profile(...)
```

目标重构为：

```python
def _extract_potion_effect_profile(action) -> dict: ...
def _build_potion_timing_context(raw_obs, legal_actions, energy, mask_np) -> dict: ...
def _evaluate_potion_timing(effect, context) -> dict: ...
def _potion_timing_profile(...) -> dict:
    effect = self._extract_potion_effect_profile(action)
    context = self._build_potion_timing_context(raw_obs, legal_actions, energy, mask_np)
    return self._evaluate_potion_timing(effect, context)
```

### 8.2 timing profile 输出

统一输出：

```python
{
    "is_potion": True,
    "available": True,

    "use_quality": 0.0,      # 当前使用即时价值
    "waste_risk": 0.0,       # 当前使用浪费风险
    "save_value": 0.0,       # 留到未来的价值
    "urgency": 0.0,

    "positive": False,
    "urgent": False,
    "deferable": False,
    "low_urgency": False,
    "save_recommended": False,

    "lethal": False,
    "overkill": False,
    "prevent_lethal": False,
    "prevent_major_loss": False,
    "block_waste": False,
    "no_followup": False,

    "mechanism_answer": False,
    "facing_change": False,

    "requires_followup": False,
    "followup_available": False,
    "hand_context_good": False,
    "hand_context_bad": False,
    "discard_context_good": False,
    "draw_pile_context_good": False,

    "damage": 0.0,
    "block": 0.0,
    "draw": 0.0,
    "energy_gain": 0.0,
    "heal": 0.0,
    "debuff": False
}
```

### 8.3 伤害类 timing

```python
if damage > 0:
    if lethal:
        use_quality += high
    elif removes_high_threat:
        use_quality += medium_high
    elif high_damage and encounter_tier in {"elite", "boss"}:
        use_quality += medium

    if overkill and not mechanism_answer:
        waste_risk += medium

    if kaiser_facing_change:
        mechanism_answer = True
        use_quality += high
```

### 8.4 防御类 timing

```python
threat_gap = max(0, incoming - current_block)

if prevent_lethal:
    use_quality += very_high
elif prevent_major_loss:
    use_quality += high
elif block > 0 and threat_gap > 0:
    use_quality += proportional(block / threat_gap)

if block > 0 and threat_gap <= 0:
    block_waste = True
    waste_risk += medium
```

Boss 战特殊处理：

- 防死仍然极高；
- 防大额机制崩盘仍然高；
- 普通小额战损不应过分提高用药价值，因为多数 boss 战后会回血。

### 8.5 资源类 timing

```python
energy_after = current_energy + energy_gain
followup_available = has_playable_followup(energy_after, legal_actions, hand)

if requires_followup:
    if followup_available:
        use_quality += medium_high
    else:
        no_followup = True
        waste_risk += high
        save_recommended = True
```

抽牌类需要额外看：

```text
hand size
hand fullness
draw pile expected value
energy after use
```

如果手牌已满或能量为 0 且抽到也无法打，不能简单给 draw 加正收益。

### 8.6 手牌改造类 timing

```python
if upgrade_hand or duplicate_next or add_replay or add_retain:
    hand_context_good = has_high_value_targets(hand)
    hand_context_bad = not hand_context_good

    if hand_context_good and can_use_after_effect:
        use_quality += medium_high
    elif hand_context_bad:
        waste_risk += medium
        save_recommended = True
```

### 8.7 长期价值类 timing

```python
if long_term_value and not immediate_survival_need:
    combat_immediate_value = low
    save_value = high
```

combat sandbox 中不要把 long-term potion 当作强 immediate positive。

---

## 9. Direct Planner 与 Aux Heads

### 9.1 短期：继续使用 planner bias 稳定训练

当前 `_combat_action_quality_bias()` 已经会处理：

- potion urgent；
- potion save recommended；
- potion no followup；
- lethal / prevent lethal / mechanism answer。

目标是让它基于新的 profile 输出，不再靠脆弱 action fields。

示例打分：

```python
score += +urgent_bonus * potion_urgent
score += +lethal_bonus * potion_lethal
score += +mechanism_bonus * potion_mechanism_answer
score += -waste_penalty * potion_waste_risk
score += -save_penalty * potion_save_recommended
score += -no_followup_penalty * potion_no_followup
```

### 9.2 中期：新增 action-level auxiliary heads

建议新增：

```text
potion_use_now_head
potion_waste_risk_head
potion_save_value_head
```

目标：

```text
potion_use_now_target = use_quality
potion_waste_risk_target = waste_risk
potion_save_value_target = save_value
```

这让模型不是死吃 bias，而是学到：

- 当前 use_potion 的即时 Q-like value；
- 当前使用是否浪费；
- 留着是否有未来价值。

### 9.3 后期：退火硬规则权重

训练稳定后：

```text
planner bias 权重逐步下降
aux head / Q-like planner 权重逐步上升
```

最终让策略从硬 heuristic 迁移到模型内建模。

---

## 10. Combat Sandbox 药水课程

仅靠自然采样很难学会“什么时候不用药水”。必须在 combat sandbox 中主动制造正反样本。

### 10.1 curriculum bucket

#### Bucket A：伤害药水斩杀

场景：

```text
敌人 hp <= potion damage
```

期望：

```text
potion_lethal_selected_rate ↑
```

#### Bucket B：伤害药水 overkill

场景：

```text
低威胁普通战
敌人 hp 很低
手牌能处理
药水伤害明显过量
```

期望：

```text
potion_overkill_selected_rate ↓
potion_save_recommended_selected_rate ↓
```

#### Bucket C：防御药水救命

场景：

```text
incoming > current_block + hp
防御/减伤/虚弱药水可防死
```

期望：

```text
potion_prevent_lethal_selected_rate ↑
```

#### Bucket D：防御药水空放

场景：

```text
敌人不攻击
或 current_block 已足够
```

期望：

```text
potion_block_waste_selected_rate ↓
```

#### Bucket E：能量药水有 followup

场景：

```text
当前能量不足
喝药后可打出高价值牌
```

期望：

```text
potion_resource_followup_selected_rate ↑
```

#### Bucket F：能量药水无 followup

场景：

```text
喝药后仍没有可打牌
或者只剩低价值动作
```

期望：

```text
potion_no_followup_selected_rate ↓
```

#### Bucket G：抽牌药水有效

场景：

```text
手牌少
能量足
抽牌堆有高价值牌
```

期望：

```text
potion_draw_good_context_selected_rate ↑
```

#### Bucket H：抽牌药水无效

场景：

```text
手牌满
能量低
抽牌堆低价值
```

期望：

```text
potion_draw_bad_context_selected_rate ↓
```

#### Bucket I：Kaiser 面向药水

场景：

```text
当前 back_attack_risk 高
targeted enemy potion 可以打另一侧目标改变 facing
```

期望：

```text
kaiser/potion_facing_change_candidate_count_mean > 0
kaiser/potion_facing_change_selected_rate ↑
kaiser_risky_end_turn_selected_rate ↓
```

#### Bucket J：手牌改造药水好目标

场景：

```text
手牌中有高价值未升级/可复制/可重放目标
```

期望：

```text
potion_hand_transform_good_context_selected_rate ↑
```

#### Bucket K：手牌改造药水坏目标

场景：

```text
手牌为空 / 全低价值 / 全状态 / 无法 followup
```

期望：

```text
potion_hand_transform_bad_context_selected_rate ↓
```

---

## 11. 指标设计

### 11.1 全局 boss_combat 指标

```text
boss_combat/potion_available_rate
boss_combat/potion_selected_rate
boss_combat/potion_use_quality_selected_mean
boss_combat/potion_waste_risk_selected_mean
boss_combat/potion_save_value_selected_mean

boss_combat/potion_urgent_available_mean
boss_combat/potion_urgent_selected_rate
boss_combat/potion_low_urgency_selected_rate

boss_combat/potion_lethal_available_mean
boss_combat/potion_lethal_selected_rate
boss_combat/potion_prevent_lethal_available_mean
boss_combat/potion_prevent_lethal_selected_rate

boss_combat/potion_save_recommended_available_mean
boss_combat/potion_save_recommended_selected_rate
boss_combat/potion_no_followup_available_mean
boss_combat/potion_no_followup_selected_rate
boss_combat/potion_block_waste_selected_rate
boss_combat/potion_overkill_selected_rate
boss_combat/potion_mechanism_answer_selected_rate
```

解释：

- `potion_selected_rate` 高但 `potion_use_quality_selected_mean` 低：乱用；
- `potion_no_followup_selected_rate` 高：资源药水乱用；
- `potion_lethal_selected_rate` 低：该用不用；
- `potion_save_recommended_selected_rate` 高：低威胁浪费；
- `potion_waste_risk_selected_mean` 高：整体药水时机差。

### 11.2 encounter-specific 指标

必须按 boss / elite 分桶，避免全局指标被稀释：

```text
boss_combat/kaiser_crab_boss/potion_selected_rate
boss_combat/kaiser_crab_boss/potion_use_quality_selected_mean
boss_combat/kaiser_crab_boss/potion_waste_risk_selected_mean
boss_combat/kaiser_crab_boss/potion_mechanism_answer_selected_rate
boss_combat/kaiser_crab_boss/potion_facing_change_candidate_count_mean
boss_combat/kaiser_crab_boss/potion_facing_change_selected_rate

boss_combat/ceremonial_beast_boss/potion_selected_rate
boss_combat/the_kin_boss/potion_selected_rate
boss_combat/the_insatiable_boss/potion_selected_rate
boss_combat/knowledge_demon_boss/potion_selected_rate
```

### 11.3 potion family 指标

```text
potion_family/damage/selected_rate
potion_family/damage/lethal_selected_rate
potion_family/damage/overkill_selected_rate

potion_family/defense/selected_rate
potion_family/defense/prevent_lethal_selected_rate
potion_family/defense/block_waste_selected_rate

potion_family/resource/selected_rate
potion_family/resource/followup_selected_rate
potion_family/resource/no_followup_selected_rate

potion_family/hand_transform/selected_rate
potion_family/hand_transform/good_context_selected_rate
potion_family/hand_transform/bad_context_selected_rate

potion_family/long_term/selected_rate
```

### 11.4 potion id 指标

对关键药水输出单独指标：

```text
potion_id/<ID>/selected_rate
potion_id/<ID>/use_quality_selected_mean
potion_id/<ID>/waste_risk_selected_mean
potion_id/<ID>/save_value_selected_mean
potion_id/<ID>/no_followup_selected_rate
potion_id/<ID>/overkill_selected_rate
potion_id/<ID>/block_waste_selected_rate
```

例如：

```text
potion_id/POTION.ENERGY_POTION/no_followup_selected_rate
potion_id/POTION.FIRE_POTION/overkill_selected_rate
potion_id/POTION.BLOCK_POTION/block_waste_selected_rate
```

---

## 12. 与现有 checkpoint 的兼容性

### 12.1 不改变输入 shape 的改造

以下可以不破坏 checkpoint：

- Bridge 新增字段；
- Python 读取新字段；
- `_potion_numeric()` 使用 `TOKEN_NUMERIC_DIM = 96` 的未使用槽位；
- `SEMANTIC_ROLE_NAMES` 不新增枚举，只映射到现有 role；
- `_potion_timing_profile()` 内部重构；
- 新增 metrics。

旧模型可以继续训练。

### 12.2 新增 aux heads 的影响

如果新增：

```text
potion_use_now_head
potion_waste_risk_head
potion_save_value_head
```

则 checkpoint 会新增参数。加载旧模型时应：

```python
load_state_dict(..., strict=False)
```

策略：

- trunk / policy / value / world model 继续加载；
- 新 aux heads 随机初始化；
- 前若干 step 给 aux loss 较低权重 warmup；
- 稳定后逐步提高 aux 权重。

不需要从零重训。

---

## 13. 落地顺序

### Phase 1：Registry 与校验

交付：

```text
packages/rl-agent/content/potions.timing.generated.json
packages/rl-agent/content/potions.timing.overrides.json
packages/rl-agent/sts2_env/potion_profiles.py
tools/generate_potion_profiles.py
```

验收：

```text
64/64 potion id 覆盖
Deprecated 标记 disabled
dataset 中所有 potion id 可解析
profile schema 稳定
```

### Phase 2：Bridge 输出结构化药水效果

交付：

```text
BuildPotionPayload() 输出 effect_profile / semantic_tags / timing_tags
BuildUsePotionSemantic() roles 从 profile 映射
AddCombatPotionActions() 保留 target_combat_id / target_scope
```

验收：

```text
use_potion action 中能看到 damage/block/draw/energy/heal 等结构字段
资源/防御/伤害药水 roles 不再全部只是 potion
Kaiser targeted potion 保留 target_combat_id
```

### Phase 3：Observation / Action Token 接入

交付：

```text
_source_profile() 支持 potion profile
_potion_numeric() 填充扩展槽位
semantic_action._infer_roles() 消费 potion profile
build_live_potion_semantic_text() 输出 compact profile
```

验收：

```text
potion world token 与 use_potion action token 语义一致
不同 potion family 的 numeric / roles 可区分
不改变 TOKEN_NUMERIC_DIM / SEMANTIC_ACTION_DIM
```

### Phase 4：Timing Evaluator 重构

交付：

```text
_potion_timing_profile() 基于 effect + context
输出 use_quality / waste_risk / save_value
支持 resource followup / hand context / Kaiser facing / boss tier
```

验收：

```text
potion_no_followup_available_mean 非 0
potion_no_followup_selected_rate 可观测
potion_block_waste_selected_rate 可观测
potion_overkill_selected_rate 可观测
potion_mechanism_answer_selected_rate 可观测
```

### Phase 5：Metrics 分桶

交付：

```text
global boss_combat potion metrics
encounter-specific potion metrics
potion_family metrics
potion_id metrics
```

验收：

```text
TensorBoard 中可直接判断：
- 药水是否乱用
- 是否该用不用
- 是否资源药水无 followup
- Kaiser 药水是否能改变 facing
```

### Phase 6：Combat Sandbox Curriculum

交付：

```text
药水正反样本 bucket
伤害斩杀 / overkill
防御救命 / 空放
能量有 followup / 无 followup
抽牌有效 / 无效
Kaiser facing potion
手牌改造好目标 / 坏目标
```

验收：

```text
potion_use_quality_selected_mean 上升
potion_waste_risk_selected_mean 下降
potion_no_followup_selected_rate 下降
potion_lethal_selected_rate 上升
kaiser potion facing metrics 出现并改善
```

### Phase 7：Aux Heads 与 bias 退火

交付：

```text
potion_use_now_head
potion_waste_risk_head
potion_save_value_head
aux losses / metrics
bias annealing schedule
```

验收：

```text
aux target MAE 下降
planner 对 potion 的 Q-like ranking 与 timing target 对齐
硬 bias 降低后策略不退化
```

---

## 14. 验收标准

### 14.1 结构验收

```text
[ ] 64 种药水全部有 profile
[ ] Bridge payload 有 effect_profile
[ ] use_potion semantic roles 不再全部相同
[ ] observation potion token 能区分 damage/resource/defense/hand_transform/long_term
[ ] action token 能看到 potion profile
[ ] 不再依赖 enemy.side 判断 left/right
```

### 14.2 行为验收

```text
[ ] 伤害药水斩杀率上升
[ ] 伤害药水 overkill 率下降
[ ] 防御药水 prevent lethal 使用率上升
[ ] 防御药水 block waste 率下降
[ ] 能量/抽牌药水 no followup 使用率下降
[ ] 低威胁下 save_recommended 药水使用率下降
[ ] Kaiser targeted potion facing candidate 不再长期为 0
[ ] Kaiser potion facing selected rate 有非零学习信号
```

### 14.3 训练稳定性验收

```text
[ ] checkpoint 可继续加载
[ ] AMP / activation checkpointing 不受影响
[ ] combat sandbox 吞吐不因长等待或复杂搜索明显下降
[ ] TensorBoard 指标能解释药水策略改善或退化
```

---

## 15. 非目标

本方案不做：

```text
硬编码每瓶药水固定什么时候用
把所有 potion 统一加分或统一降分
用 MCTS 搜索弥补药水时机建模
依赖 target_side = left/right 判断 Kaiser 面向
为了短期胜率牺牲结构化语义
```

本方案要做：

```text
结构化药水能力
结构化当前战斗上下文
让模型通过 attention 学会药水时机
用 aux target 和 sandbox curriculum 加速学习
用 metrics 精确定位“乱用 / 不会用 / 没信号 / 没候选”
```

---

## 16. 最终目标状态

完成后，模型应该能够学习到：

- 火焰类药水在能斩杀、改变 Kaiser 面向、处理高威胁目标时使用；
- 伤害药水在低威胁 overkill 场景中保留；
- 防御药水在致死/大额 incoming 时使用；
- 防御药水在无 incoming 或 block 已足够时保留；
- 能量药水必须结合当前手牌、后续可打牌和能量预算；
- 抽牌药水必须结合手牌空间、抽牌堆质量和剩余能量；
- 手牌改造药水必须结合当前手牌质量；
- 长期收益药水不被误当成普通 combat immediate positive；
- targeted enemy potion 可以作为 Kaiser 面向机制的应答候选；
- 药水使用不再是“有就用”，而是 action-level Q-like / timing-aware 决策。
