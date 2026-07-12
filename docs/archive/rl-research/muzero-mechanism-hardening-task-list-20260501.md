# MuZero 机制覆盖硬化任务清单（2026-05-01）

> 目的：把当前审计出的机制合同 / 安全层 / 观测诊断缺口拆成可执行任务，优先减少“长训到一半又必须停训改代码”的概率。
> 范围：STS2 bridge、RL env、MuZero train、observation/aux targets、metrics、测试。
> 当前结论：**不要继续无保护长训**。先完成 P0，跑 smoke train，通过指标门槛后再重新开正式 run。
> 建议训练策略：P0 完成后可以 warm-start 旧 checkpoint，但建议丢弃旧 replay buffer，新 schema / 新 run / 新 diagnostics。

---

## 0. 全局执行规则

### 0.1 不能违反的设计原则

- [ ] **合法可打 ≠ 应该打**：不要把“有可打牌”直接等同于“结束回合就是空过”。必须区分：
  - urgent positive action；
  - deferable positive action；
  - strategic skip；
  - forced/transient end_turn。
- [ ] **HP-cost 自杀永远禁止**：boss 战战损可以低权重，但任何会使 `current_hp - hp_cost <= 0` 的即时 HP cost 动作都必须 hard mask / huge negative，不得只靠 terminal loss 学。
- [ ] **不要用 block 抵消 HP loss**：`cardHpLoss` / `nonCardHpLoss` 在 STS2 源码中是 unblockable。除非 bridge 明确标注为 `blockable_self_damage`，否则默认不能被格挡。
- [ ] **Kaiser 转身不是字段**：转身 = 使用 targeted card / potion / operation 指向当前 facing 另一侧的敌人。左右来自 enemy powers：
  - `BACK_ATTACK_LEFT_POWER`
  - `BACK_ATTACK_RIGHT_POWER`
  不得用 `enemy.side` / `target.side` 当 left/right。
- [ ] **核心机制不要靠文本正则**：文本 fallback 只能做低置信 diagnostics，不得驱动 hard safety / 高权重 reward / critical aux target。
- [ ] **transient only-end_turn 不能靠长 sleep**：必须直接状态判定 + 极短 short poll budget，不允许把总等待时间拉到接近或超过 combat reset。
- [ ] **旧 buffer 不混用**：任何 action schema / observation schema / reward target / mask 变化后，正式训练应新 run + 新 buffer。

### 0.2 每个任务完成时必须包含

- [ ] 修改了哪些文件。
- [ ] 新增 / 修改了哪些字段。
- [ ] 新增 / 修改了哪些指标。
- [ ] 单元测试 / fake bridge 测试 / snapshot 测试。
- [ ] 最小 smoke 验证命令或说明。
- [ ] 是否改变 replay buffer schema；如果改变，标记 `schema_version`。

### 0.3 代码改动前置检查

- [ ] 查看当前 dirty diff，不覆盖已有改动：

```powershell
git status --short
git diff --stat
```

- [ ] 搜索相关现有实现，避免重复写一套逻辑：

```powershell
rg -n "action_masks|transient_only_end_turn|frontier_stable|hp_loss|hp_cost|self_lethal|BACK_ATTACK_LEFT|BACK_ATTACK_RIGHT|ApplyXCostPreviewMapping|ResolveCardSelectionSemantics|GetCardReference|_dump_loss_spike" packages mods -S --glob "!**/logs_muzero/**" --glob "!**/bin/**" --glob "!**/obj/**"
```

---

# P0 — 正式长训前必须完成

---

## P0-1. HP-cost / self-lethal hard safety

### 问题

当前 `combat_env.action_masks()` / `env_v2.action_masks()` 只看 bridge legal actions，不看当前 HP。`train.py` 里 `hp_loss` 只是轻惩罚，不能阻止低血自杀。

### 主要文件

- [ ] `mods/sts2-bridge/Scripts/BridgeGameApi.cs`
- [ ] `mods/sts2-bridge/Scripts/BridgeGameApi.EnvPayloads.cs`
- [ ] `packages/rl-agent/sts2_env/combat_env.py`
- [ ] `packages/rl-agent/sts2_env/env_v2.py`
- [ ] `packages/rl-agent/muzero/train.py`
- [ ] `packages/rl-agent/sts2_env/observation_v3.py`
- [ ] `packages/rl-agent/sts2_env/observation_common.py`
- [ ] 相关 tests 目录。

### Bridge 任务

- [ ] 给每个 combat action 增加 typed safety payload：

```json
{
  "safety": {
    "hp_cost_kind": "none | unblockable_hp_loss | blockable_self_damage | max_hp_loss | delayed_hp_loss",
    "hp_cost": 0,
    "hp_loss_unblockable": 0,
    "self_damage_blockable": 0,
    "max_hp_loss": 0,
    "hp_before": 0,
    "hp_after_self_cost": 0,
    "hp_margin_after_self_cost": 0,
    "self_lethal_now": false,
    "low_hp_margin_after_cost": false,
    "source_confidence": "runtime_internal | effect_preview | fallback"
  }
}
```

- [ ] 明确 `hp_loss_unblockable` 不被 block 抵消。
- [ ] 如果存在“先杀敌再扣血”之类特殊执行顺序，必须 bridge 明确标注；否则默认仍按 unsafe 处理。
- [ ] `max_hp_loss` 单独导出，不与当前 HP loss 混合。

### Python env 任务

- [ ] 增加 shared helper，例如：

```python
def is_self_lethal_action(action: dict, raw_obs: dict | None) -> bool:
    ...
```

优先读 `action["safety"]["self_lethal_now"]`，fallback 再读 effect preview / semantic fields。

- [ ] 在 `combat_env.action_masks()` 中 hard mask：

```python
if is_self_lethal_action(action, self._last_obs_raw):
    mask[i] = False
```

- [ ] 在 `env_v2.action_masks()` 中同样 hard mask。
- [ ] 如果 action 被 mask，diagnostics 里记录 `masked_reason="self_lethal_hp_cost"`。

### Train / reward 任务

- [ ] `_action_immediate_impact()` 不再只做 `-0.5 * hp_loss`。
- [ ] `_classify_positive_combat_action()` 使用当前 HP 判断：
  - `self_lethal_now`：hard negative / impossible；
  - `low_margin_after_cost`：soft negative；
  - `hp_cost_no_followup`：deferable / negative；
  - `hp_cost_with_followup`：可正向，但必须 non-lethal。
- [ ] `_combat_action_quality_bias()` 中加入 HP-cost safety，保证 planner bias 不会奖励自杀动作。
- [ ] death reward 中单独记录 self HP-cost death，不要混在普通战损里。

### 指标

新增 TB scalar / aggregated metrics：

- [ ] `combat/hp_cost_action_available_rate`
- [ ] `combat/hp_cost_action_selected_rate`
- [ ] `combat/hp_cost_self_lethal_available_rate`
- [ ] `combat/hp_cost_self_lethal_selected_rate`
- [ ] `combat/hp_cost_low_margin_available_rate`
- [ ] `combat/hp_cost_low_margin_selected_rate`
- [ ] `combat/hp_cost_no_followup_selected_rate`
- [ ] `death/self_hp_cost_lethal_count`
- [ ] `death/self_hp_cost_lethal_rate`

### JSONL diagnostics

- [ ] 新增或扩展：`diagnostics/hp_cost_actions.jsonl`
- [ ] 每条记录至少包含：

```json
{
  "episode_id": "...",
  "step": 0,
  "encounter": "...",
  "turn": 0,
  "hp_before": 0,
  "max_hp": 0,
  "block": 0,
  "incoming": 0,
  "action_idx": 0,
  "action_id": "...",
  "family": "play_card",
  "card_id": "...",
  "card_title": "...",
  "hp_loss_unblockable": 0,
  "self_damage_blockable": 0,
  "max_hp_loss": 0,
  "hp_after": 0,
  "hp_margin": 0,
  "self_lethal_now": false,
  "low_margin": false,
  "lethal_on_enemy": false,
  "prevent_lethal": false,
  "followup_available": false,
  "mechanism_answer": false,
  "masked": false,
  "selected": false,
  "resulted_death": false
}
```

### Tests

- [ ] `1 HP + hp_cost 3 => masked`
- [ ] `3 HP + hp_cost 3 => masked`
- [ ] `4 HP + hp_cost 3 => allowed but low_margin=true`
- [ ] `HP 2 + block 99 + hp_cost 3 => masked`
- [ ] `hp_cost prevents incoming lethal but non-self-lethal => allowed`
- [ ] `hp_cost with no follow-up => deferable / negative`
- [ ] `boss non-lethal hp-cost => low attrition penalty`
- [ ] `boss self-lethal hp-cost => forbidden`

### Definition of Done

- [ ] `combat/hp_cost_self_lethal_selected_rate == 0` in smoke run。
- [ ] 低血自杀案例在 action mask 层被挡住，而不是只靠 reward 学。
- [ ] diagnostics 能解释每个 hp-cost action 为什么被允许 / mask / 惩罚。

---

## P0-2. transient only-end_turn 不暴露给 policy

### 问题

bridge / env live path 仍可能在抽卡、洗牌、动画、hand 更新窗口只暴露 `end_turn`。这会制造假空过样本。

### 主要文件

- [ ] `mods/sts2-bridge/Scripts/BridgeGameApi.EnvPayloads.cs`
- [ ] `mods/sts2-bridge/Scripts/BridgeGameApi.EnvHelpers.cs`
- [ ] `packages/rl-agent/sts2_env/combat_env.py`
- [ ] `packages/rl-agent/sts2_env/env_v2.py`
- [ ] `packages/rl-agent/sts2_env/bridge_client.py`
- [ ] 相关 tests。

### Bridge 任务

- [ ] `ShouldSuppressTransientCombatEndTurnOnly()` 不得继续 no-op。
- [ ] `BuildEnvActionabilityPayload()` 的 `anyPending` 不得只依赖 `phase == "settling"`。
- [ ] actionability payload 增加 direct-state pending reasons：

```json
{
  "frontier_stable": false,
  "transient_only_end_turn": true,
  "legal_action_count": 1,
  "legal_non_end_turn_count": 0,
  "pending_reasons": [
    "phase_settling",
    "draw_pending",
    "shuffle_pending",
    "hand_not_ready",
    "animation_pending",
    "queue_pending"
  ],
  "wait_budget_ms": 50
}
```

- [ ] 如果 combat 中只有 end_turn，且任一 pending reason 为 true，不应标记为 stable frontier。
- [ ] 如果真实稳定地没有可打动作，允许 end_turn，并标记 `frontier_stable=true`。

### Python env 任务

- [ ] `env_v2._recover_filtered_action_window()` 不能再只判断 `if self._legal_actions: return True`。
- [ ] `combat_env.step()` live path 接入 short poll，而不是只写 diagnostics。
- [ ] short poll 必须是极短 budget：
  - 默认 `max_wait_ms` 建议 30-100ms；
  - `poll_interval_ms` 建议 5-10ms；
  - 参数可配置。
- [ ] timeout 后：
  - 记录 `transient_leaked=1`；
  - 不把这个 end_turn 计为 wasteful；
  - 尽量不作为正常训练样本，或降低权重。

### 指标

- [ ] `bridge/transient_only_end_turn_rate`
- [ ] `bridge/transient_resolved_rate`
- [ ] `bridge/transient_leaked_rate`
- [ ] `bridge/transient_wait_ms_mean`
- [ ] `bridge/transient_wait_ms_p95`
- [ ] `bridge/end_turn_only_unstable_count`
- [ ] `bridge/end_turn_slow_step_rate`

### Tests

- [ ] `phase=settling + only end_turn => transient_only_end_turn=true`
- [ ] `phase=combat + only end_turn + draw_pending => transient_only_end_turn=true`
- [ ] `phase=combat + only end_turn + hand_not_ready => transient_only_end_turn=true`
- [ ] transient within budget resolves => policy never sees fake end_turn。
- [ ] timeout leak => metric emitted，且不计 wasteful。
- [ ] stable no non-end-turn actions => end_turn allowed。
- [ ] p95 wait 不超过配置 budget，不能接近 combat reset 等待量。

### Definition of Done

- [ ] smoke run 中 `bridge/transient_leaked_rate` 接近 0。
- [ ] slow bridge step 仍可能存在，但不再污染 end_turn 空过 detector。
- [ ] end_turn-only 稳定状态和 transient 状态可在 diagnostics 里区分。

---

## P0-3. Kaiser facing resolver 全路径统一

### 问题

正确 helper 已有，但 `combat_env.py` / `train.py` / `semantic_action.py` / `potion_timing.py` 仍有旧 `side` fallback 或近似逻辑。

### 主要文件

- [ ] `packages/rl-agent/sts2_env/boss_kaiser.py`
- [ ] `packages/rl-agent/sts2_env/boss_mechanics.py`
- [ ] `packages/rl-agent/sts2_env/combat_env.py`
- [ ] `packages/rl-agent/sts2_env/potion_timing.py`
- [ ] `packages/rl-agent/sts2_env/semantic_action.py`
- [ ] `packages/rl-agent/muzero/train.py`
- [ ] 相关 tests。

### 任务

- [ ] 确认 `boss_kaiser.py` 提供唯一公共 API：
  - `enemy_back_attack_position(enemy)`
  - `build_kaiser_state(combat, player_obs=None)`
  - `classify_kaiser_action_mechanism(combat, action, player_obs=None)`
  - `action_changes_facing_toward_target(combat, action, player_obs=None)`
- [ ] 所有 Kaiser facing / back attack / target opposite side 判断都调用该 helper。
- [ ] 删除或降级以下逻辑：
  - `target.side in {"left", "right"}`
  - `enemy.side in {"left", "right"}`
  - `target_side` 当左右位置的 fallback。
- [ ] `potion_timing.py` 中 `facing_change = False` 必须改为 shared resolver。
- [ ] `combat_env.py` 中 refund/no-followup intrinsic value 的 Kaiser 特例必须用 shared resolver。
- [ ] `train.py` 中 `_action_target_side()` 的 side fallback 不得把 faction side 当左右。

### 正确判定逻辑

```python
enemy_positions = {
    enemy.combat_id: "left" if enemy.powers contains BACK_ATTACK_LEFT_POWER
    else "right" if enemy.powers contains BACK_ATTACK_RIGHT_POWER
    else None
}

target_pos = enemy_positions.get(action.target_combat_id)
changes_facing = target_pos is not None and target_pos != current_facing
```

### 指标

- [ ] `boss_combat/kaiser_crab_boss/kaiser_back_attack_risk_mean`
- [ ] `boss_combat/kaiser_crab_boss/facing_change_candidate_count_mean`
- [ ] `boss_combat/kaiser_crab_boss/facing_change_selected_rate`
- [ ] `boss_combat/kaiser_crab_boss/risky_end_turn_selected_rate`
- [ ] `boss_combat/kaiser_crab_boss/pressure_selected_rate`
- [ ] `boss_combat/kaiser_crab_boss/defense_selected_rate`

### Tests

- [ ] `enemy.side = "Enemy"` 不得被识别为 left/right。
- [ ] enemy powers 包含 `BACK_ATTACK_LEFT_POWER` => position left。
- [ ] enemy powers 包含 `BACK_ATTACK_RIGHT_POWER` => position right。
- [ ] targeted card 指向 opposite side => facing_change_candidate。
- [ ] targeted potion 指向 opposite side => facing_change_candidate。
- [ ] self-target card / potion => not facing change。
- [ ] AOE card 可作为 pressure，不应误判为 facing change。
- [ ] `combat_env` / `train` / `potion_timing` 对同一 fake state 分类一致。

### Definition of Done

- [ ] `kaiser_facing_change_candidate_count_mean` 不再长期 0，除非 fake/legal actions 中确实没有 targeted opposite-side action。
- [ ] 所有旧 `side` 逻辑被删除或明确标注为 faction-only。

---

## P0-4. X-cost / Star-X dynamic effect preview

### 问题

energy-X 已有 `x_cost_value`，但 Star-X 的 `current_star_cost` 没进入 `ApplyXCostPreviewMapping()`，可能造成 effect preview 错误。

### 主要文件

- [ ] `mods/sts2-bridge/Scripts/BridgeGameApi.cs`
- [ ] `packages/rl-agent/sts2_env/observation_v3.py`
- [ ] `packages/rl-agent/sts2_env/card_effect_profile.py`
- [ ] `packages/rl-agent/muzero/train.py`
- [ ] 相关 tests。

### Bridge 任务

- [ ] action/card payload 增加统一 X-cost contract：

```json
{
  "x_cost": {
    "has_x_cost": true,
    "resource": "energy | stars | other",
    "current_value": 0,
    "is_zero": true,
    "effect_scaled": true,
    "preview_scale_source": "energy_x | star_x | none",
    "semantics": "repeat | damage | block | unknown"
  }
}
```

- [ ] `ApplyXCostPreviewMapping()` 支持 energy-X 和 Star-X：
  - `card.EnergyCost.CostsX` 使用 current energy；
  - `card.HasStarCostX` 使用 current stars。
- [ ] effect preview 的 `damage_per_hit` / `total_damage` / `repeats` 必须按实际当前 X 资源映射。
- [ ] 如果 X 语义未知，标记 `semantics="unknown"`，不要假装 covered。

### Python 任务

- [ ] observation 中分开编码：
  - energy-X 当前值；
  - Star-X 当前值；
  - zero-X flag；
  - zero-X valid reason。
- [ ] train 中 zero-X 选择不应一律 bad：
  - 如果 0-X 仍有非 X 效果，允许；
  - 如果无有效效果，deferable / negative。
- [ ] 加入 followup 判断，防止 0 费 X 因旧 3 energy stale preview 被误判。

### 指标

- [ ] `combat/x_cost_available_rate`
- [ ] `combat/x_cost_selected_rate`
- [ ] `combat/x_cost_zero_available_rate`
- [ ] `combat/x_cost_zero_selected_rate`
- [ ] `combat/x_cost_zero_bad_selected_rate`
- [ ] `combat/x_cost_energy_value_mean`
- [ ] `combat/x_cost_star_value_mean`
- [ ] `combat/star_x_available_rate`
- [ ] `combat/star_x_selected_rate`

### Tests

- [ ] spend energy 后下一 obs 的 energy-X value 降低。
- [ ] gain energy 后下一 obs 的 energy-X value 升高。
- [ ] 0 energy X 无有效效果 => bad/deferable。
- [ ] 0 energy X 有非 X 效果 => 不自动 bad。
- [ ] Star-X current value 跟 current stars。
- [ ] Star-X `total_damage` / `repeats` 按 stars 变化。
- [ ] energy-X 与 Star-X 不共享 stale cached value。

### Definition of Done

- [ ] 模型不再因初始 3 energy stale preview 误打 0-X。
- [ ] Star-X fake state 的 bridge payload 和 Python observation 一致。

---

## P0-5. Card selection / mutation typed contract

### 问题

`ResolveCardSelectionSemantics()` 仍大量依赖 prompt / text contains，例如 discard / retain / transform / upgrade。核心机制不应由文本正则驱动。

### 主要文件

- [ ] `mods/sts2-bridge/Scripts/BridgeGameApi.EnvPayloads.cs`
- [ ] `mods/sts2-bridge/Scripts/BridgeGameApi.cs`
- [ ] `packages/rl-agent/sts2_env/hand_mutation.py`
- [ ] `packages/rl-agent/sts2_env/aux_targets.py`
- [ ] `packages/rl-agent/sts2_env/observation_v3.py`
- [ ] 相关 tests。

### Bridge 任务

- [ ] 用 runtime internal source 优先推断 selection semantics：
  - screen type；
  - command type；
  - selector prefs；
  - operation id；
  - source card/action/effect；
  - source/destination zone；
  - modifier id。
- [ ] 输出统一 selection payload：

```json
{
  "selection": {
    "screen_type": "...",
    "operation_type": "discard | retain | exhaust | remove | transform | upgrade | copy | add | replace | enchant | afflict | unknown",
    "source_zone": "hand | draw | discard | exhaust | deck | unknown",
    "destination_zone": "hand | draw | discard | exhaust | deck | removed | unknown",
    "min_select": 0,
    "max_select": 0,
    "target_filter": {},
    "source_action_id": "...",
    "source_card_instance_id": "...",
    "modifier_id": "...",
    "confidence": "runtime_internal | static_export | text_fallback | unknown",
    "inferred_from_text": false
  }
}
```

- [ ] 文本 fallback 可以保留，但必须：
  - `confidence="text_fallback"`；
  - `inferred_from_text=true`；
  - 不进入 hard safety / high-weight reward。

### Python 任务

- [ ] `hand_mutation.py` 优先读 typed `selection.operation_type`。
- [ ] `aux_targets.py` 的 lifecycle / future bank target 优先读 typed operation，不再靠 title / prompt / description。
- [ ] unknown / text fallback selection 要写 diagnostics。

### 必须覆盖的 operation

- [ ] discard selected cards。
- [ ] retain selected cards。
- [ ] exhaust selected cards。
- [ ] remove card。
- [ ] transform card。
- [ ] upgrade card。
- [ ] copy card。
- [ ] add / create card。
- [ ] replace hand / selected hand cards。
- [ ] enchant card。
- [ ] afflict card。
- [ ] modify cost。
- [ ] add replay / recast。
- [ ] add void / ethereal / retain / exhaust modifier。

### Tests

- [ ] discard selection typed，不靠 prompt regex。
- [ ] retain selection typed，不靠 prompt regex。
- [ ] transform selection typed，不靠 prompt regex。
- [ ] copy selection typed，不靠 prompt regex。
- [ ] enchant selection typed，不靠 prompt regex。
- [ ] afflict selection typed，不靠 prompt regex。
- [ ] unknown selection emits diagnostics。
- [ ] text fallback 不驱动 hard reward / safety。

### Definition of Done

- [ ] 核心 selection / mutation 不再依赖文本 contains。
- [ ] text fallback 占比有指标，并能被监控。

---

## P0-6. Stable card instance identity / provenance

### 问题

bridge 当前使用 `RuntimeHelpers.GetHashCode(card)` 生成 card reference，不适合作为长期训练主 identity。Python fallback 可能退化到 definition id / title，导致同名多实例 collision。

### 主要文件

- [ ] `mods/sts2-bridge/Scripts/BridgeGameApi.cs`
- [ ] `packages/rl-agent/sts2_env/aux_targets.py`
- [ ] `packages/rl-agent/sts2_env/observation_v3.py`
- [ ] `packages/rl-agent/sts2_env/card_lifecycle_tokens.py`
- [ ] 相关 tests。

### Bridge 任务

- [ ] 每张 runtime card 导出：

```json
{
  "definition_id": "...",
  "stable_instance_id": "...",
  "combat_instance_id": "...",
  "card_ref": "...",
  "zone": "hand | draw | discard | exhaust | deck | unknown",
  "created_by_action_id": "...",
  "created_by_card_instance_id": "...",
  "copy_source_instance_id": "...",
  "transform_source_instance_id": "...",
  "generation": 0,
  "identity_confidence": "stable | object_hash | definition_only | title_only"
}
```

- [ ] 如果引擎没有持久 uid，构造 combat-local deterministic id：
  - combat id；
  - initial deck index；
  - card creation counter；
  - transform/copy generation；
  - zone movement generation。
- [ ] `RuntimeHelpers.GetHashCode` 可以保留为 debug ref，但不要作为唯一主 id。

### Python 任务

- [ ] `_card_identity_key()` 优先级改为：
  1. `stable_instance_id`
  2. `combat_instance_id`
  3. `card_ref`
  4. `uid` / `instance_id` / `combat_uuid`
  5. definition id，仅低置信 fallback
  6. title，仅 debug fallback
- [ ] 如果 fallback 到 definition/title，必须记录 `identity_confidence`。
- [ ] future lifecycle target 不应在低置信 identity 上产生高权重 target。

### Tests

- [ ] same card `Hand -> Discard -> Draw -> Hand` instance id 不变。
- [ ] copy card 有 `copy_source_instance_id`。
- [ ] transform result 有 `transform_source_instance_id`。
- [ ] generated card 有 `created_by_card_instance_id`。
- [ ] same-title cards 不 collision。
- [ ] title fallback emits low confidence。

### Definition of Done

- [ ] lifecycle / future bank target 不再依赖 title collision。
- [ ] loss spike dump 可以定位具体 card instance。

---

## P0-7. Loss spike dump + aux target guard

### 问题

future_world_aux / future_bank_state 曾出现巨大 spike，但当前 dump 只有 loss 数字，无法定位 root cause。

### 主要文件

- [ ] `packages/rl-agent/muzero/train.py`
- [ ] `packages/rl-agent/sts2_env/aux_targets.py`
- [ ] `packages/rl-agent/sts2_env/card_lifecycle_tokens.py`
- [ ] 相关 tests。

### 任务

- [ ] 扩展 `_dump_loss_spike()`，至少写入：

```json
{
  "kind": "loss_spike",
  "total_steps": 0,
  "batch_index": 0,
  "sample_id": "...",
  "episode_id": "...",
  "env_id": 0,
  "floor": 0,
  "encounter": "...",
  "turn": 0,
  "losses": {},
  "action": {
    "action_id": "...",
    "family": "...",
    "card_title": "...",
    "card_instance_id": "...",
    "target_combat_id": "..."
  },
  "prev_compact": {
    "hp": 0,
    "energy": 0,
    "stars": 0,
    "hand": [],
    "draw": [],
    "discard": [],
    "exhaust": []
  },
  "next_compact": {
    "hp": 0,
    "energy": 0,
    "stars": 0,
    "hand": [],
    "draw": [],
    "discard": [],
    "exhaust": []
  },
  "target_top_errors": [],
  "bank_token_offenders": [],
  "identity_confidence": "...",
  "raw_obs_path": "..."
}
```

- [ ] 对 aux targets 加 finite guard：
  - no NaN；
  - no Inf；
  - target value bounded；
  - suspicious sample dump + quarantine。
- [ ] 对 future bank / lifecycle target 做 top-k offender 分析。
- [ ] raw JSON 必须 capped，避免日志爆炸。

### Tests

- [ ] 人工构造异常 target，触发 spike dump。
- [ ] dump 包含 action / prev / next / target offender。
- [ ] NaN / Inf target 被 guard。
- [ ] 超界 target 被 clip 或 quarantine，并记录原因。

### Definition of Done

- [ ] 下一次 spike 可以直接定位到具体 sample / action / card instance / target dim。
- [ ] spike 不再只能通过 loss 标量猜测。

---

# P1 — 建议和 P0 同轮或紧随其后完成

---

## P1-1. HP-loss / blockable self-damage / max HP loss 分离进 observation

- [ ] observation dense vector 分开编码：
  - `hp_loss_unblockable_norm`
  - `self_damage_blockable_norm`
  - `max_hp_loss_norm`
  - `hp_after_cost_norm`
  - `hp_margin_after_cost_norm`
  - `self_lethal_flag`
  - `low_margin_flag`
- [ ] 不再把 modifier self_damage 简单加进 hp_loss。
- [ ] train classifier 使用分离字段。
- [ ] 测试 blockable self-damage 与 unblockable hp-loss 差异。

---

## P1-2. Potion timing per-potion profile

### 任务

- [ ] 每种 potion 生成 timing profile：
  - lethal；
  - prevent lethal；
  - mechanism answer；
  - overkill；
  - block waste；
  - no follow-up；
  - save value；
  - low urgency。
- [ ] `potion_timing.py` Kaiser mechanism 使用 shared `boss_kaiser.py` resolver。
- [ ] use_potion transition 记录：

```json
{
  "execute_ok": true,
  "slot_before": 0,
  "slot_after": 0,
  "empty_before": false,
  "empty_after": true,
  "potion_id_before": "...",
  "potion_id_after": null,
  "target": "...",
  "consumed": true
}
```

### 指标

- [ ] `potion/<id>/available_rate`
- [ ] `potion/<id>/selected_rate`
- [ ] `potion/<id>/lethal_use_rate`
- [ ] `potion/<id>/prevent_lethal_use_rate`
- [ ] `potion/<id>/mechanism_answer_use_rate`
- [ ] `potion/<id>/overkill_use_rate`
- [ ] `potion/<id>/low_urgency_use_rate`
- [ ] `potion/<id>/unused_on_death_rate`

### Tests

- [ ] damage potion 在 lethal window 被正向。
- [ ] block potion 在无 incoming 时 low urgency / waste。
- [ ] energy potion 无 follow-up 时 low urgency。
- [ ] use_potion 成功后 slot empty 或 consumed marker true。
- [ ] failed use_potion 不计 consumed。

---

## P1-3. Replay / recast / copy / triggered action provenance

- [ ] action payload 增加：

```json
{
  "action_context": {
    "trigger_source": "manual | replay | recast | copy | forced | queued",
    "trigger_card_instance_id": "...",
    "root_player_action_id": "...",
    "is_policy_decision": true
  }
}
```

- [ ] policy loss / behavior cloning / selected rate 只对 `is_policy_decision=true` 的动作做主监督。
- [ ] replay / recast 不应误算成玩家主动多次选择。
- [ ] lifecycle target 使用 trigger provenance。

---

## P1-4. Enchant / affliction / modifier typed semantics

- [ ] modifier payload 增加：

```json
{
  "modifier_id": "...",
  "modifier_type": "enchantment | affliction | temporary | permanent",
  "semantic_tags": [
    "add_replay",
    "void",
    "retain",
    "ethereal",
    "cost_down",
    "cost_up",
    "exhaust",
    "copy_on_play"
  ],
  "duration": "combat | turn | permanent",
  "stack_count": 1,
  "confidence": "runtime_internal | static_export | text_fallback"
}
```

- [ ] Python 不再用 bind/bound/chain/shackle 等文本正则驱动核心逻辑。
- [ ] unknown modifier 只进 diagnostics，不进 hard safety。

---

## P1-5. Per-encounter metrics namespace

- [ ] 确保所有 boss_combat 关键指标都有 encounter-specific namespace：

```text
boss_combat/kaiser_crab_boss/*
boss_combat/ceremonial_beast_boss/*
boss_combat/the_kin_boss/*
boss_combat/the_insatiable_boss/*
boss_combat/knowledge_demon_boss/*
```

- [ ] 至少覆盖：
  - win / boss_win；
  - wasteful end_turn；
  - urgent / deferable / strategic skip；
  - energy；
  - playable cards；
  - potion timing；
  - HP-cost；
  - X-cost；
  - boss-specific mechanism。

---

## P1-6. Strategic skip / deferable action 指标加强

- [ ] 继续保留 urgent/deferable 分类。
- [ ] 新增：

```text
combat/urgent_positive_action_count_mean
combat/deferable_positive_action_count_mean
combat/strategic_defer_available_rate
combat/strategic_defer_selected_rate
combat/exhaust_deferable_available_rate
combat/retain_deferable_available_rate
combat/refund_no_followup_deferable_rate
combat/x_zero_deferable_rate
```

- [ ] 确认 `wasteful_end_turn_rate` 只在 urgent available 时计入。

---

## P1-7. Observation / action schema versioning

- [ ] 增加 schema version：
  - bridge payload version；
  - observation version；
  - action semantics version；
  - aux target version。
- [ ] replay buffer 写入 schema version。
- [ ] loader 检查 schema mismatch：
  - mismatch 时拒绝混用旧 buffer；
  - 或仅允许 explicit migration。

---

# P2 — P0/P1 稳定后执行

---

## P2-1. 人工手打 imitation learning 管线

前置条件：P0 全部完成，且 smoke run 通过。

- [ ] 记录人工 action trajectories：
  - obs；
  - legal actions；
  - selected action；
  - human rationale 可选；
  - action schema version。
- [ ] 过滤 transient / unstable action window。
- [ ] 过滤低置信 text fallback selection。
- [ ] behavior cloning warmup：policy prior，不直接替代 MuZero value。
- [ ] demo buffer 与 self-play buffer 分开管理。
- [ ] 指标：
  - `bc/policy_loss`
  - `bc/top1_match`
  - `bc/top3_match`
  - `bc/demo_weight`
  - `bc/schema_mismatch_drop_count`

---

## P2-2. Reward curriculum 调整

- [ ] 非 boss 普通/精英战增加低战损通关 shaping。
- [ ] boss 战战损低权重，但 self-lethal 永远 hard forbid。
- [ ] 随 win rate 提升逐步提高 attrition penalty。
- [ ] 单独跟踪：
  - normal combat hp lost；
  - elite combat hp lost；
  - boss combat hp lost；
  - post-boss heal exception；
  - A10 boss 无满血例外。

---

## P2-3. Dashboard / 文档更新

- [ ] 更新 `docs/card-mechanism-coverage-audit.md`。
- [ ] 新增 schema contract 文档：
  - action payload；
  - card payload；
  - selection payload；
  - safety payload；
  - potion payload。
- [ ] 新增 training go/no-go checklist。
- [ ] TensorBoard dashboard 分组：
  - safety；
  - transient；
  - Kaiser；
  - potion；
  - X-cost；
  - aux spike；
  - per-encounter。

---

# 10. 并行拆分建议

如果用多个 worker / agent 执行，建议按写入范围拆分，避免冲突。

## Worker A — HP-cost safety

写入范围：

- `mods/sts2-bridge/Scripts/BridgeGameApi.cs`
- `packages/rl-agent/sts2_env/combat_env.py`
- `packages/rl-agent/sts2_env/env_v2.py`
- `packages/rl-agent/muzero/train.py` 中 HP-cost 相关小范围
- HP-cost tests

交付：P0-1 全部。

---

## Worker B — transient only-end_turn

写入范围：

- `mods/sts2-bridge/Scripts/BridgeGameApi.EnvPayloads.cs`
- `mods/sts2-bridge/Scripts/BridgeGameApi.EnvHelpers.cs`
- `packages/rl-agent/sts2_env/bridge_client.py`
- `packages/rl-agent/sts2_env/combat_env.py`
- `packages/rl-agent/sts2_env/env_v2.py`
- transient tests

交付：P0-2 全部。

---

## Worker C — Kaiser resolver / potion mechanism

写入范围：

- `packages/rl-agent/sts2_env/boss_kaiser.py`
- `packages/rl-agent/sts2_env/boss_mechanics.py`
- `packages/rl-agent/sts2_env/potion_timing.py`
- `packages/rl-agent/sts2_env/combat_env.py` 中 Kaiser 特例
- `packages/rl-agent/muzero/train.py` 中 Kaiser classifier 小范围
- Kaiser tests

交付：P0-3 + P1-2 Kaiser 部分。

---

## Worker D — X-cost / Star-X

写入范围：

- `mods/sts2-bridge/Scripts/BridgeGameApi.cs`
- `packages/rl-agent/sts2_env/observation_v3.py`
- `packages/rl-agent/sts2_env/card_effect_profile.py`
- `packages/rl-agent/muzero/train.py` 中 X-cost 分类小范围
- X-cost tests

交付：P0-4 全部。

---

## Worker E — selection / mutation typed contract

写入范围：

- `mods/sts2-bridge/Scripts/BridgeGameApi.EnvPayloads.cs`
- `packages/rl-agent/sts2_env/hand_mutation.py`
- `packages/rl-agent/sts2_env/aux_targets.py`
- selection / mutation tests

交付：P0-5 全部。

---

## Worker F — identity / provenance / aux targets

写入范围：

- `mods/sts2-bridge/Scripts/BridgeGameApi.cs`
- `packages/rl-agent/sts2_env/aux_targets.py`
- `packages/rl-agent/sts2_env/card_lifecycle_tokens.py`
- `packages/rl-agent/sts2_env/observation_v3.py`
- identity / lifecycle tests

交付：P0-6 + P1-3 / P1-4 部分。

---

## Worker G — spike diagnostics / metrics / docs

写入范围：

- `packages/rl-agent/muzero/train.py` 中 dump / metrics 区域
- metrics aggregation files
- docs
- spike tests

交付：P0-7 + P1-5 + docs 更新。

---

# 11. Smoke run Go/No-Go 门槛

P0 完成后，正式长训前至少跑一次短 smoke train。

## 必须满足

- [ ] `combat/hp_cost_self_lethal_selected_rate == 0`
- [ ] `death/self_hp_cost_lethal_count == 0`
- [ ] `bridge/transient_leaked_rate` 接近 0。
- [ ] `wasteful_end_turn_rate` 与 `wasteful_end_turn_bias_applied_rate` 口径一致。
- [ ] `kaiser_facing_change_candidate_count_mean` 不再因为字段 bug 长期 0。
- [ ] `x_cost_zero_bad_selected_rate` 可见且不异常升高。
- [ ] Star-X fake test 通过。
- [ ] selection text fallback rate 可见；核心 selection 不依赖 text fallback。
- [ ] loss spike dump 字段完整。
- [ ] no NaN / Inf aux target。
- [ ] action/obs schema version 写入 replay buffer。

## 如果不满足

- [ ] 不开正式长训。
- [ ] 不引入人工 demo buffer。
- [ ] 不复用旧 replay buffer。
- [ ] 先修对应 P0。

---

# 12. 正式训练建议

完成 P0 + smoke 通过后：

- [ ] 新建 run dir。
- [ ] 新建 replay buffer。
- [ ] 可以 warm-start 旧 checkpoint。
- [ ] optimizer 建议重建或至少谨慎 reset，避免旧 spike momentum。
- [ ] 记录 schema version。
- [ ] 前 1 小时重点看：
  - HP-cost selected；
  - transient leaked；
  - X-cost zero bad selected；
  - Kaiser facing candidate / selected；
  - potion low urgency use；
  - future aux spike；
  - reward / death floor / boss seen。

---

# 13. 明确不要做的事

- [ ] 不要用长 sleep 解决 transient end_turn。
- [ ] 不要把 `target.side` / `enemy.side` 当 Kaiser 左右。
- [ ] 不要用 prompt / description 文本作为核心机制来源。
- [ ] 不要把所有 positive action 都视为 urgent。
- [ ] 不要把 HP-cost 自杀交给 reward 学。
- [ ] 不要混用旧 replay buffer。
- [ ] 不要在没有 schema version 的情况下改 action/obs payload。
- [ ] 不要让 replay / recast / copied action 误算成 policy 主动选择。
- [ ] 不要在 loss spike dump 里只写 loss 数字。
