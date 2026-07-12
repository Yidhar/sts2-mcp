# MuZero EndTurn / 剩余能量判定规范（2026-05-13）

## 结论

**`energy > 0 && selected == EndTurn` 不是 bug。**

STS2 中“牌真的打完了但能量没花完”是正常局面；模型不应该被训练成“必须把能量花光”。
EndTurn 只有在稳定 action frontier 上存在明确的、当前必须处理的非 EndTurn 动作时，才算 hard bad。

---

## 1. 分类目标

每个 EndTurn 选择应归入以下一类：

| 分类 | 是否算模型硬错误 | 含义 |
| --- | --- | --- |
| `forced_end_turn` | 否 | 没有可打/有意义的非 EndTurn 动作；剩余能量可忽略 |
| `benign_leftover_energy` | 否 | 有能量，但剩余动作无收益或明显不该用 |
| `strategic_defer_end_turn` | 否 | 有动作但保留更合理，例如药水、扣血 setup、低价值防御 |
| `unknown` | 否/待查 | 有非 EndTurn 动作，但收益不明确，不能硬判 |
| `bad_end_turn` | 是 | 漏斩杀、漏防死、漏明显防伤、漏 boss 必打机制 |
| `frontier_transient_suspect` | 不是模型错 | bridge 过早暴露 EndTurn，短轮询后才出现真实手牌/动作 |

---

## 2. 明确正常：牌打完但能量没花完

### 2.1 手牌为空，只有 EndTurn

```text
HP              = 80/80
Block           = 0
Energy          = 1
Incoming damage = 16
Hand count      = 0

Legal actions:
1. End Turn
```

判定：

```text
forced_end_turn
```

这不是模型 bug。能量剩余没有意义，因为没有可执行动作。

### 2.2 手牌为空，只剩低紧急药水

```text
HP              = 62/75
Block           = 26
Energy          = 1
Incoming damage = 16
Hand count      = 0

Legal actions:
1. 能量药水
2. 瓶中精灵
3. End Turn
```

判定：

```text
forced_end_turn / benign_leftover_energy
```

这里虽然 legal action 不止 EndTurn，但药水不是当前必须使用的动作；不能因为还有能量或药水就判 bad。

### 2.3 敌人不攻击，只剩防御

```text
HP              = 68/75
Block           = 21
Energy          = 4
Incoming damage = 0

Legal actions:
1. 防御
2. 防御
3. End Turn
```

判定：

```text
benign_leftover_energy / unknown
```

防御不产生进展，也不减少损血，EndTurn 可接受。

---

## 3. 明确错误：有当前必须动作却 EndTurn

### 3.1 漏斩杀

```text
Enemy HP = 6
Energy   = 1
Hand:
- 打击，伤害 6，cost 1

Selected:
End Turn
```

判定：

```text
bad_end_turn
```

### 3.2 漏防死

```text
HP              = 8
Block           = 0
Incoming damage = 10
Energy          = 1
Hand:
- 防御，block 5

Selected:
End Turn
```

判定：

```text
bad_end_turn
```

打防御后仍可能受伤，但可以从死亡变成存活。

### 3.3 漏明显防伤

```text
HP              = 66/80
Block           = 0
Incoming damage = 14
Energy          = 1
Hand:
- 防御，block 5
- 防御，block 5
- 放血

Selected:
End Turn
```

判定：

```text
bad_end_turn
```

这里错误不是“还有 1 能量”，而是“有低风险防御可直接减少真实掉血”。

### 3.4 漏 boss 必打机制

例如沙虫 / Insatiable：

```text
Sandpit countdown = 1
Hand:
- 狂乱逃离
Energy enough

Selected:
End Turn
```

判定：

```text
bad_end_turn
```

倒计时为 0 即死；`countdown <= 1` 时这是强制机制动作。

---

## 4. 不应该硬判 bad 的常见局面

| 局面 | 原因 |
| --- | --- |
| 只剩防御且 block 已足够 | 继续防御无收益 |
| 敌人无攻击意图，只剩防御 | 防御不产生进展 |
| 只剩扣血牌，当前低血或无明确收益 | 不打可能是合理保守 |
| 只剩药水，当前无 lethal/prevent-lethal/机制压力 | 药水保留有长期价值 |
| 只剩 0 能量 X 费牌且无 0 费效果 | EndTurn 合理 |
| 有 setup 牌但收益延迟且当前风险高 | 不能硬判，最多 `unknown` |

---

## 5. 稳定 frontier 要求：防止“手牌还没发完就 EndTurn”

有一种情况不是模型策略错，而是 bridge/action frontier 提早发布：

```text
t = 0ms:
legal_actions = [EndTurn]

t = 150ms:
legal_actions = [EndTurn]

t = 300ms:
legal_actions = [打击, 防御, 技能, EndTurn]
```

如果模型在 `t=0ms` 就收到 EndTurn-only frontier 并执行，应该标记为：

```text
frontier_transient_suspect
```

处理规则：

1. 当 legal frontier 只有 EndTurn 时，不立即交给模型。
2. 对同一 combat state 短轮询 2-3 次。
3. 只有在 hand/action/frontier hash 稳定后才允许 EndTurn。
4. 如果短轮询后出现非 EndTurn 动作，记录 bridge/frontier transient，不算模型 bad。

---

## 6. 推荐实现逻辑

伪代码：

```python
if selected_action != END_TURN:
    return "not_end_turn"

if not stable_frontier:
    return "frontier_transient_suspect"

if legal_non_endturn_count == 0:
    return "forced_end_turn"

if has_lethal_action:
    return "bad_end_turn"

if has_prevent_lethal_action:
    return "bad_end_turn"

if has_boss_mechanism_required_action:
    return "bad_end_turn"

if has_urgent_block_action and incoming_damage > current_block:
    return "bad_end_turn"

if only_low_urgency_potions_or_negative_actions:
    return "benign_leftover_energy"

return "unknown"
```

---

## 7. 当前监控口径

训练 gate 应以严格指标为准：

```text
combat_quality/bad_end_turn_selected_rate
```

不要把宽泛/历史兼容指标当 hard bad：

```text
combat_quality/wasteful_end_turn_rate
action_offenders: bad_end_turn  # 旧 broad offender 可能混有 soft/unknown
```

2026-05-13 已调整：

```text
combat_quality_bad_end_turn_selected -> bad_end_turn
combat_quality_wasteful_end_turn_selected only -> soft_or_ambiguous_end_turn
```

这样监控不会再把“牌打完但能量没花完”的正常局面误报成严格 bad EndTurn。

---

## 8. 验收标准

### 必须为 0

```text
combat_quality/bad_end_turn_selected_rate == 0
strict_bad_end_turn count == 0
frontier_transient_only_end_turn == 0
```

### 可以存在，但要人工抽样

```text
end_turn_unknown_selected_rate
soft_or_ambiguous_end_turn
forced_end_turn_selected_rate
benign_leftover_energy examples
```

### 抽样时重点看

```text
incoming_damage
current_block
energy
hand_count
legal_action_count
playable_cards_left
positive_action_count
urgent_positive_action_count
has_lethal
has_prevent_lethal
has_boss_mechanism
frontier_stable
```
