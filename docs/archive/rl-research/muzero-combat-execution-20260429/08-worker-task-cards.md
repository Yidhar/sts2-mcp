# 08 — Worker Task Cards

本文件是可以直接复制给 Claude worker 的任务卡片。
每张卡片都包含：读取文件、owned paths、目标、非目标、验收。

---

## 使用方式

给 worker 时建议格式：

```text
请在 E:\game\project\sts2_mcp 中执行 TASK-XX。
先阅读 docs/muzero-combat-execution-20260429/README.md、
docs/muzero-combat-execution-20260429/00-repo-map-and-evidence.md
以及该任务对应专题文档。
只修改 Owned paths 范围内文件。
完成后按 README 的 Worker 输出格式汇报。
```

---

## P0 / Diagnostics

### TASK-A1 — Selected End Turn Context Dump

Read first:

```text
01-diagnostics-first.md#TASK-A1
```

Owned paths:

```text
packages/rl-agent/muzero/train.py
packages/rl-agent/tests/test_end_turn_context_dump.py
```

Goal:

- 当 selected action 是 end_turn 时写 `diagnostics/end_turn_contexts.jsonl`。
- dump energy/block/hp/incoming/hand-draw-discard-exhaust counts。
- dump positive/urgent/mandatory/deferable/refund/x-cost/boss mechanism counts。
- dump top legal actions 和 score components。

Non-goals:

- 不改 action score。
- 不改 reward。
- 不改 planner 权重。

Acceptance:

- 单测覆盖 forced/bad/strategic/transient。
- 短跑出现 JSONL。
- selected tracker 与 bias 端使用同一 context。

---

### TASK-A2 — Action Offender Dump and Metrics

Read first:

```text
01-diagnostics-first.md#TASK-A2
```

Owned paths:

```text
packages/rl-agent/muzero/train.py
packages/rl-agent/tests/test_action_offender_metrics.py
```

Goal:

- 写 `diagnostics/action_offenders.jsonl`。
- 支持 bad_end_turn、strategic_skip、refund_no_followup、zero_energy_x_cost、potion、Kaiser、Ceremonial、Insatiable offenders。
- 所有 boss 指标同时 emit global 和 per-encounter namespace。

Non-goals:

- 不改变 policy。
- 不新增 boss-specific hardcoded action selection。

Acceptance:

- `boss_combat/<metric>` 和 `boss_combat/<encounter_id>/<metric>` 都出现。
- 非对应 encounter 不误触发 boss offender。

---

### TASK-A3 — X-Cost Dynamic Diagnostics

Read first:

```text
01-diagnostics-first.md#TASK-A3
```

Owned paths:

```text
packages/rl-agent/muzero/train.py
packages/rl-agent/sts2_env/semantic_action.py
packages/rl-agent/sts2_env/observation_v3.py
packages/rl-agent/tests/test_x_cost_dynamic_energy.py
```

Goal:

- action semantic 暴露 base/current/effective cost、is_x_cost、x_cost_effective_energy。
- 0 energy X-cost 有指标和 offender。

Non-goals:

- 不用静态初始 3 energy 推断 X。

Acceptance:

- energy=0 X-cost 无额外效果被标 bad。
- energy=2 X-cost effective energy=2。
- 机制性 0-energy X-cost 不被粗暴打死，但必须记录 reason。

---

### TASK-A4 — Potion Transition Diagnostics

Read first:

```text
01-diagnostics-first.md#TASK-A4
```

Owned paths:

```text
packages/rl-agent/muzero/train.py
packages/rl-agent/sts2_env/combat_env.py
packages/rl-agent/sts2_env/headless_sim_bridge_client.py
mods/sts2-bridge/Scripts/BridgeGameApi.cs
mods/sts2-bridge/Scripts/BridgeGameApi.PotionProfiles.cs
packages/rl-agent/tests/test_potion_transition_diagnostics.py
```

Goal:

- 区分 use_potion 失败、slot 未清、final obs 刷新时机、snapshot 状态复用。
- 记录 use_potion before/after slot 状态。

Non-goals:

- 不先调整 potion reward。

Acceptance:

- execute ok 后 slot 为空或明确 unusable/used。
- death final potion count 不把已用药水算 unused。

---

## P0 / End Turn and Bridge

### TASK-B1 — EndTurnTaxonomy

Read first:

```text
02-end-turn-action-quality.md#TASK-B1
```

Owned paths:

```text
packages/rl-agent/muzero/train.py
packages/rl-agent/tests/test_end_turn_taxonomy.py
```

Goal:

- 实现 forced / strategic_defer / bad end_turn 三分类。
- 保留旧 wasteful 指标 alias。

Non-goals:

- 不把所有 end_turn 都惩罚。
- 不把 strategic defer 当空过。

Acceptance:

- bad override > forced > strategic_defer。
- Kaiser risk + facing candidate + end_turn => bad。
- forced/no actions 正常分类。

---

### TASK-B2 — Strategic Skip Narrowing

Read first:

```text
02-end-turn-action-quality.md#TASK-B2
```

Owned paths:

```text
packages/rl-agent/muzero/train.py
packages/rl-agent/sts2_env/card_effect_profile.py
packages/rl-agent/tests/test_strategic_skip_narrowing.py
```

Goal:

- strategic skip 必须有明确 future reason。
- 单纯 exhaust/refund/special 不足以标 skip。

Non-goals:

- 不让模型因为“消耗牌”三个字变怂。

Acceptance:

- exhaust lethal 不 skip。
- exhaust low-value + future setup 才 skip。

---

### TASK-B3 — Refund-No-Followup Recompute

Read first:

```text
02-end-turn-action-quality.md#TASK-B3
```

Owned paths:

```text
packages/rl-agent/muzero/train.py
packages/rl-agent/sts2_env/semantic_action.py
packages/rl-agent/sts2_env/card_effect_profile.py
packages/rl-agent/tests/test_refund_no_followup.py
```

Goal:

- refund followup 基于 after-action estimate，而不是 before static hand。

Non-goals:

- 不把所有 refund 无后续都当坏；intrinsic value 要保留。

Acceptance:

- refund + draw/create/cost reduction 能识别 followup。
- refund + block lethal 识别 intrinsic。

---

### TASK-C1 — Bridge Transient Only-EndTurn Guard

Read first:

```text
03-bridge-fast-step.md#TASK-C1
```

Owned paths:

```text
mods/sts2-bridge/Scripts/BridgeGameApi.cs
mods/sts2-bridge/Scripts/BridgeGameApi.EnvPayloads.cs
mods/sts2-bridge/Scripts/BridgeGameApi.Actions.cs
```

Goal:

- bridge payload 增加 actionability。
- 判断 frontier_stable / transient_only_end_turn / only_end_turn_reason。

Non-goals:

- 不用长固定 sleep。

Acceptance:

- queue/animation/draw-shuffle pending + only end_turn => transient。
- stable no actions => stable_no_actions。

---

### TASK-C2 — Python Fast Step Short Polling

Read first:

```text
03-bridge-fast-step.md#TASK-C2
```

Owned paths:

```text
packages/rl-agent/sts2_env/combat_env.py
packages/rl-agent/sts2_env/headless_sim_bridge_client.py
packages/rl-agent/muzero/train.py
packages/rl-agent/tests/test_bridge_transient_end_turn_guard.py
```

Goal:

- Python 端对 transient only-end_turn 短轮询。
- 输出 wait_ms/poll_count/leaked 指标。

Non-goals:

- 不把等待时间拉到接近 reset。

Acceptance:

- transient 第 2 次出现 play_card => 返回新 obs。
- timeout 有标记但不无限等。
- p95 wait 可控。

---

## P1 / Card and Boss Modeling

### TASK-D1 — CardEffectProfile Field Coverage

Read first:

```text
04-card-lifecycle-action-outcome.md#TASK-D1
```

Owned paths:

```text
packages/rl-agent/sts2_env/card_effect_profile.py
packages/rl-agent/sts2_env/generate_card_effect_profiles.py
packages/rl-agent/sts2_env/audit_card_mechanism_coverage.py
mods/sts2-bridge/Scripts/BridgeGameApi.CardEffectProfiles.cs
docs/generated/card-effect-profile-coverage.md
docs/generated/card-effect-profile-schema.md
```

Goal:

- Ironclad 88 张 + 无色牌关键机制结构化覆盖。
- lifecycle/cost/hand_mutation/pile_mutation/combat/mechanism 全字段。

Non-goals:

- 文本正则不能作为主路径。

Acceptance:

- coverage md 生成。
- fallback regex 列表可见。
- Armaments/exhaust/refund/X-cost/enchantment 测试通过。

---

### TASK-D2 — Observation and Action Lifecycle Tokens

Read first:

```text
04-card-lifecycle-action-outcome.md#TASK-D2
```

Owned paths:

```text
packages/rl-agent/sts2_env/observation_v3.py
packages/rl-agent/sts2_env/semantic_action.py
packages/rl-agent/sts2_env/card_effect_profile.py
packages/rl-agent/tests/test_card_lifecycle_tokens.py
```

Goal:

- observation/action token 包含 lifecycle/cost/hand/pile/action outcome。

Non-goals:

- 不只加文本 embedding。

Acceptance:

- 武装升级前后 token 不同。
- 消耗/回费/X-cost/retain/void 都有 token。

---

### TASK-D3 — Future-World Card Lifecycle Aux

Read first:

```text
04-card-lifecycle-action-outcome.md#TASK-D3
```

Owned paths:

```text
packages/rl-agent/muzero/train.py
packages/rl-agent/sts2_env/observation_v3.py
packages/rl-agent/tests/test_future_world_aux_card_lifecycle.py
```

Goal:

- 预测 next hand/draw/discard/exhaust/energy/block/incoming/boss mechanism state。

Non-goals:

- 不只追求 aux loss 下降；要供 planner gate 使用。

Acceptance:

- 普通牌进 discard。
- exhaust 牌进 exhaust。
- draw/upgrade/facing transition target 正确。

---

### TASK-E1 — Kaiser Facing Semantics

Read first:

```text
05-boss-mechanics.md#TASK-E1
```

Owned paths:

```text
packages/rl-agent/sts2_env/semantic_action.py
packages/rl-agent/sts2_env/observation_v3.py
packages/rl-agent/sts2_env/_sim_translate.py
mods/sts2-bridge/Scripts/BridgeGameApi.EnvPayloads.cs
packages/rl-agent/muzero/train.py
packages/rl-agent/tests/test_kaiser_facing_semantics.py
```

Goal:

- 从 `BACK_ATTACK_LEFT_POWER` / `BACK_ATTACK_RIGHT_POWER` 推断左右。
- targeted card/potion 到另一侧算 facing change candidate。

Non-goals:

- 不用 `enemy.side` 判断左右。

Acceptance:

- `kaiser_facing_change_candidate_count_mean` 不再因为字段错长期为 0。
- self block 是 defense，不是 facing change。

---

### TASK-E2 — Ceremonial Beast Mechanics

Read first:

```text
05-boss-mechanics.md#TASK-E2
```

Owned paths:

```text
packages/rl-agent/sts2_env/observation_v3.py
packages/rl-agent/sts2_env/semantic_action.py
packages/rl-agent/muzero/train.py
packages/rl-agent/tests/test_ceremonial_mechanics.py
mods/sts2-bridge/Scripts/BridgeGameApi.EnvPayloads.cs
```

Goal:

- one-card lock / stun window / single action impact 进入语义和指标。

Non-goals:

- 不硬编码固定出牌序列。

Acceptance:

- low impact under lock offender 出现。
- stun window missed 可诊断。

---

### TASK-E3 — Insatiable Offenders

Read first:

```text
05-boss-mechanics.md#TASK-E3
```

Owned paths:

```text
packages/rl-agent/sts2_env/observation_v3.py
packages/rl-agent/sts2_env/semantic_action.py
packages/rl-agent/muzero/train.py
packages/rl-agent/tests/test_insatiable_offenders.py
```

Goal:

- 为 Insatiable 输出 strategic skip / refund / bad end_turn / pressure offenders。

Non-goals:

- 不在不了解机制时乱写硬编码策略。

Acceptance:

- `boss_combat/the_insatiable_boss/*` namespace 出现。
- 能解释该 boss 回归主要 offender。

---

## P2 / Planner, Replay and Demo

### TASK-F1 — Drift-Gated Direct Planner

Read first:

```text
06-direct-planner-replay-imitation.md#TASK-F1
```

Owned paths:

```text
packages/rl-agent/muzero/train.py
packages/rl-agent/muzero/model.py
packages/rl-agent/tests/test_direct_planner_drift_gate.py
```

Goal:

- direct planner 增加 drift_gate。
- latent drift/Q error/legal F1/disagreement 高时降低 rollout Q 权重。

Non-goals:

- 不恢复高延迟 MCTS。

Acceptance:

- gate=0 rollout 不影响排序。
- bad_end_turn immediate penalty 能压过错误高 Q。

---

### TASK-F2 — Encounter-Balanced Replay Scheduler

Read first:

```text
06-direct-planner-replay-imitation.md#TASK-F2
```

Owned paths:

```text
packages/rl-agent/muzero/train.py
packages/rl-agent/muzero/replay_buffer.py
packages/rl-agent/tests/test_encounter_balanced_replay.py
```

Goal:

- replay tier 分布 boss 50-60%、elite 20-30%、normal 10-20%。
- boss 内按 encounter 平衡。

Non-goals:

- 不让 boss oversample 挤掉基础攻防。

Acceptance:

- sample rate metrics 出现。
- underrepresented encounter 被提升但不过度。

---

### TASK-F3 — Human Demo Imitation Dataset

Read first:

```text
06-direct-planner-replay-imitation.md#TASK-F3
```

Owned paths:

```text
packages/rl-agent/sts2_env/combat_env.py
packages/rl-agent/muzero/train.py
packages/rl-agent/muzero/demo_dataset.py
packages/rl-agent/tests/test_human_demo_dataset.py
docs/human-demo-format.md
```

Goal:

- 支持人类手打 demo JSONL。
- obs snapshot + legal actions + selected action + reason tags 进入 imitation loss。

Non-goals:

- 不让 demo 完全覆盖 RL。

Acceptance:

- demo load。
- selected action 匹配 legal action。
- train step 产生 demo policy CE。
