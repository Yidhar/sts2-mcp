# 06 — Direct Planner, Replay and Imitation

本文件处理 search-free planner、replay 调度和人类手打模仿学习。
目标不是恢复高延迟 MCTS，而是让模型内建 lookahead，同时在不可靠时自动降权。

---

## TASK-F1 — Drift-Gated Direct Planner

### 问题

当前 direct planner 已经有 Q-like / objective / risk / uncertainty 等分支，但当：

- latent drift 高
- Q 误差高
- legal F1 不稳
- branch disagreement 高

时，rollout/lookahead value 可能带偏动作选择。

### 目标

给 planner 增加 drift gate：模型越不可靠，越依赖 immediate tactical quality 和 policy prior；模型越可靠，越使用 lookahead。

### Owned paths

```text
packages/rl-agent/muzero/train.py
packages/rl-agent/muzero/model.py
packages/rl-agent/tests/test_direct_planner_drift_gate.py
```

实际 model 文件名以 repo 为准。

### 评分结构

目标形式：

```text
score =
    policy_prior_weight * policy_prior
  + immediate_tactical_weight * immediate_tactical_quality
  + mechanism_weight * mechanism_objective
  + drift_gate * rollout_q_weight * rollout_q
  + drift_gate * objective_q_weight * objective_q
  - drift_gate * risk_q_weight * risk_q
  - uncertainty_weight * uncertainty_penalty
  + legality_guard
```

### drift_gate

范围：

```text
0.0 <= drift_gate <= 1.0
```

输入：

```text
latent_drift
q_mae_or_value_error
legal_f1
branch_disagreement
future_lifecycle_aux_error
surprise/uncertainty
```

建议：

```python
gate = 1.0
gate *= sigmoid((legal_f1 - legal_f1_floor) / scale)
gate *= sigmoid((drift_ceiling - latent_drift) / scale)
gate *= sigmoid((q_error_ceiling - q_mae) / scale)
gate *= sigmoid((disagreement_ceiling - branch_disagreement) / scale)
gate = clamp(gate, min_gate, 1.0)
```

`min_gate` 可为 0 或小正数。训练早期建议低一些。

### immediate tactical quality

必须覆盖：

- damage/lethal
- block/incoming
- energy use
- zero-energy X-cost penalty
- refund followup
- strategic skip guard
- potion timing
- boss mechanism action
- bad end_turn penalty

### 指标

```text
planner/drift_gate_mean
planner/drift_gate_p10
planner/drift_gate_p90
planner/rollout_q_effective_weight_mean
planner/objective_q_effective_weight_mean
planner/risk_q_effective_weight_mean
planner/immediate_tactical_weight_mean
planner/mechanism_weight_mean
planner/uncertainty_penalty_mean
planner/action_score_component/<component>
```

### 测试

1. latent_drift 高 => drift_gate 降低。
2. legal_f1 低 => drift_gate 降低。
3. drift_gate=0 时 rollout_q 不影响排序。
4. drift_gate=1 时 rollout_q 正常影响排序。
5. bad_end_turn immediate penalty 能压过错误高 Q。

---

## TASK-F2 — Encounter-Balanced Replay Scheduler

### 问题

当前 boss sample rate 可到 0.8+，但 boss win 下降。
过采样 boss 不一定有效，可能造成：

- 基础攻防能力退化。
- normal/elite 分布不足。
- 个别 boss 样本不足但被全局 boss 指标掩盖。

### 目标

replay 采样按 tier 和 encounter 平衡，而不是单纯 boss oversample。

### Owned paths

```text
packages/rl-agent/muzero/train.py
packages/rl-agent/muzero/replay_buffer.py
packages/rl-agent/tests/test_encounter_balanced_replay.py
```

实际 replay 文件以 repo 当前结构为准。

### 推荐目标分布

```text
boss   50-60%
elite  20-30%
normal 10-20%
weak   optional small
```

boss 内部按 encounter 近似均衡：

```text
kaiser_crab_boss
ceremonial_beast_boss
the_kin_boss
the_insatiable_boss
knowledge_demon_boss
...
```

### 采样权重

每条样本 weight 可由：

```text
tier_weight
encounter_underrepresented_weight
recent_regression_weight
offender_weight
demo_weight
freshness_weight
```

组合，但要避免极端权重：

```text
weight = clamp(weight, min_w, max_w)
```

### 指标

```text
replay/sample_rate/tier/boss
replay/sample_rate/tier/elite
replay/sample_rate/tier/normal
replay/sample_rate/encounter/<encounter_id>
replay/offender_sample_rate/<offender_type>
replay/demo_sample_rate
```

### 测试

1. buffer 中 boss 样本多时，采样率仍不超过上限。
2. 某 boss underrepresented 时被提升。
3. normal/elite 不被饿死。
4. offender sample 能提升但不完全主导。

---

## TASK-F3 — Human Demo Imitation Dataset

### 目标

允许用户手打 AI 进行模仿学习，突破当前训练瓶颈。
数据格式必须能保存：

- obs snapshot
- legal actions
- 人类选择动作
- reason tag
- encounter / turn / outcome
- 可选：动作质量标签

### Owned paths

```text
packages/rl-agent/sts2_env/combat_env.py
packages/rl-agent/muzero/train.py
packages/rl-agent/muzero/demo_dataset.py
packages/rl-agent/tests/test_human_demo_dataset.py
docs/human-demo-format.md
```

### Demo JSONL schema

```json
{
  "version": 1,
  "source": "human",
  "timestamp": "2026-04-29T21:00:00+08:00",
  "episode_id": "...",
  "encounter_id": "kaiser_crab_boss",
  "tier": "boss",
  "turn": 3,
  "step_in_turn": 2,
  "obs": {},
  "legal_actions": [
    {
      "action_id": "...",
      "family": "play_card",
      "title": "Defend",
      "semantic": {}
    }
  ],
  "selected_action_id": "...",
  "reason_tags": [
    "avoid_back_attack",
    "block_incoming",
    "save_exhaust_card"
  ],
  "comment": "转身并挡伤害",
  "outcome": {
    "combat_win": true,
    "hp_loss": 6,
    "turns": 5
  }
}
```

### 训练接入

新增 imitation loss：

```text
loss/demo_policy_ce
loss/demo_reason_aux_optional
```

训练参数示例：

```text
--demo-dataset path/to/demo.jsonl
--demo-sample-rate 0.05
--demo-loss-weight 0.2
```

不要让 demo 直接覆盖 RL；它应该是 replay 中的高质量监督信号。

### reason tags

建议固定枚举：

```text
lethal
block_lethal
avoid_back_attack
change_facing
use_stun_window
save_exhaust_card
play_exhaust_now
refund_followup
avoid_refund_no_followup
use_potion_now
save_potion
setup_next_turn
cycle_control
```

### 指标

```text
demo/sample_rate
demo/policy_ce
demo/top1_agreement
demo/per_reason_agreement/<reason>
demo/per_encounter_agreement/<encounter_id>
```

### 测试

1. demo JSONL load。
2. selected_action_id 能匹配 legal action。
3. 缺失 obs 或 action 不匹配时给出清晰错误。
4. demo batch 能进入 train step 并产生 CE loss。
