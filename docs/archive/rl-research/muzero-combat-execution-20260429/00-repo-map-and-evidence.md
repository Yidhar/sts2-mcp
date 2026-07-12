# 00 — Repo Map and Evidence

本文给执行 agent 一个不依赖聊天上下文的代码地图和问题证据。

---

## 1. 关键目录

```text
E:\game\project\sts2_mcp
├─ packages/
│  └─ rl-agent/
│     ├─ muzero/
│     │  ├─ train.py
│     │  ├─ mcts.py
│     │  ├─ model.py / models...
│     │  └─ README / docs...
│     ├─ sts2_env/
│     │  ├─ observation_v3.py
│     │  ├─ semantic_action.py
│     │  ├─ combat_env.py
│     │  ├─ headless_sim_bridge_client.py
│     │  └─ _sim_translate.py
│     ├─ logs_muzero/
│     └─ tests/
├─ mods/
│  └─ sts2-bridge/
│     └─ Scripts/
│        ├─ BridgeGameApi.cs
│        ├─ BridgeGameApi.EnvPayloads.cs
│        ├─ BridgeGameApi.PotionProfiles.cs
│        └─ ...
└─ docs/
```

---

## 2. 关键代码锚点

用下面命令定位当前实现：

```powershell
rg -n "def _combat_action_quality_bias|wasteful_end_turn|strategic_defer|refund_no_followup|x_cost|kaiser|ceremonial|latent_drift|direct_rollout|potion_timing|boss_combat" `
  packages/rl-agent/muzero/train.py `
  packages/rl-agent/sts2_env `
  mods/sts2-bridge/Scripts `
  -g "*.py" -g "*.cs"
```

已知锚点大致如下，行号可能因后续修改漂移：

```text
packages/rl-agent/muzero/train.py
  _semantic_family                            ~1214
  _potion_timing_profile                      ~1773
  _classify_positive_combat_action            ~2077
  _is_kaiser_facing_change_action             ~2432
  _dump_kaiser_facing_diagnostic              ~2435
  _root_end_turn_context / root context        ~2700-2810
  _is_x_cost_action                           ~2888
  _combat_action_quality_bias                 ~2907
  _selected_action_combat_quality_stats        ~3240
  boss_combat aggregation                     ~3820-3900
  direct rollout planner stats                ~4470-4620

mods/sts2-bridge/Scripts/BridgeGameApi.EnvPayloads.cs
  ResolvePlayerFacing
  ComputeIncomingDamageMultiplier

mods/sts2-bridge/Scripts/BridgeGameApi.cs
  BuildPlayCardSemantic
  BuildPotionPayload
```

---

## 3. 当前回归证据

最近多份 log 对比显示，整体并不是 encoder 崩了，而是 combat action quality / 机制 / planner 进入错误区域。

### 3.1 整体指标回退

| metric | previous | current | delta |
|---|---:|---:|---:|
| `256/win` | 0.704 | 0.619 | -0.085 |
| `256/boss_win` | 0.331 | 0.257 | -0.074 |
| `64/boss_win` | 0.411 | 0.210 | -0.201 |
| `256/elite_win` | 0.768 | 0.668 | -0.100 |
| `256/normal_win` | 0.993 | 0.887 | -0.106 |
| `family_end_turn_rate` | 0.200 | 0.251 | +0.051 |
| `wasteful_available` | 0.578 | 0.652 | +0.074 |
| `wasteful_selected` | 0.000 | 0.000 | 与人工观察冲突 |
| `strategic_skip_selected` | 0.084 | 0.098 | +0.014 |
| `refund_no_followup_selected` | 0.038 | 0.051 | +0.013 |
| `latent_drift` | 0.513 | 0.574 | +0.061 |
| `legal_f1` | 0.798 | 0.813 | 没有崩 |
| `boss_sample_rate` | 0.775 | 0.825 | boss 过采样 |
| `temperature` | 0.201 | 0.119 | 探索更低 |

解释：

- `legal_f1` 没有明显崩，说明不是“完全不会合法动作”。
- `latent_drift` 升高，说明 lookahead value/rollout 可能不可靠。
- `family_end_turn_rate` 升高，且人类观察到空过，但 `wasteful_selected=0`，说明 detector 口径有问题。
- boss sample rate 过高但 boss win 下降，说明“采得多”没有转化为“学得好”。

### 3.2 Encounter 回归

| encounter | previous | current | signal |
|---|---:|---:|---|
| `ceremonial_beast_boss` | 30.0% | 0.0% | 严重 |
| `the_insatiable_boss` | 45.5% | 16.7% | 严重 |
| `knights_elite` | 73.3% | 53.3% | 明显 |
| `the_kin_boss` | 55.6% | 35.7% | 明显 |
| `phrog_parasite_elite` | 80.0% | 63.6% | 明显 |
| `kaiser_crab_boss` | 9.1% | 7.1% | 仍弱 |

这说明不能只优化全局 boss 聚合，必须按 encounter 做分桶。

---

## 4. 已知数据契约陷阱

### 4.1 `wasteful_selected=0` 不等于没有空过

之前观测：

```text
boss_combat/wasteful_end_turn_rate              0.0
boss_combat/wasteful_end_turn_bias_applied_rate mean 0.71
boss_combat/energy_mean                         mean 1.92
boss_combat/positive_action_count_mean          3.20
boss_combat/playable_cards_left_mean            4.62
```

这代表：

- bias 端能识别“结束回合很差”。
- selected rate 端却统计不到。
- 因此问题是口径不一致或 selected action tracker 没读同一套 context，不是简单“没有空过”。

### 4.2 Kaiser 左右不在 `enemy.side`

错误假设：

```text
enemy.side = left/right
action.target.side = left/right
```

实际：

```text
enemy.side = Enemy  # 敌我阵营，不是左右
```

Kaiser 左右部位应由 enemy powers 判断：

```text
BACK_ATTACK_LEFT_POWER  => left
BACK_ATTACK_RIGHT_POWER => right
```

任何指向目标的卡牌或药水，只要目标在当前 facing 的另一侧，都可能改变面向。

### 4.3 X-cost 是动态费用，不是初始 3 能量静态字段

X 费动作质量取决于：

- 当前剩余 energy。
- 当前卡牌是否有 X-cost 非能量效果。
- 是否有 relic/power/附魔改变 X 费。
- 打出后能否形成伤害/block/draw/energy/机制收益。

0 能量打 X 费牌通常应被强烈怀疑，除非该 X 费牌有 0 能量也生效的特殊效果。

### 4.4 Exhaust card 合法可打不等于应该打

部分消耗牌、回费牌、生产制造、预借时间、放血一类卡，可能存在：

- 当前合法能打。
- 但打出会消耗关键资源。
- 或回费后没有后续动作，等于白打。
- 或应该留到下一轮循环/爆发窗口。

因此 action quality 必须区分：

- bad skip / 空过
- strategic defer / 明确等待未来收益
- bad play / 白打或浪费

### 4.5 Bridge transient only-end-turn 不能靠长 sleep

某些动画/抽卡/洗牌短窗口中，legal actions 可能暂时只暴露 end_turn。
如果此时 bridge 把 end_turn 暴露给模型，模型就会被训练成“结束回合”。

但不能简单长等：

- combat sandbox 吞吐非常重要。
- 等待时间超过 reset 等待时间是错误设计。

正确方向：

- bridge payload 明确标记 actionability/frontier stable。
- Python fast step 短轮询状态版本/hash。
- 只在 transient only-end-turn 被确认时延迟暴露 end_turn。

---

## 5. 验收时必须看哪些指标

### 5.1 空过/结束回合

全局和 per encounter 都要有：

```text
boss_combat/forced_end_turn_selected_rate
boss_combat/strategic_defer_end_turn_selected_rate
boss_combat/bad_end_turn_selected_rate
boss_combat/bad_end_turn_available_rate
boss_combat/wasteful_end_turn_rate                 # old alias
boss_combat/wasteful_end_turn_bias_applied_rate
boss_combat/energy_mean
boss_combat/positive_action_count_mean
boss_combat/playable_cards_left_mean

boss_combat/<encounter_id>/...
```

### 5.2 X-cost

```text
boss_combat/x_cost_available_rate
boss_combat/zero_energy_x_cost_available_rate
boss_combat/zero_energy_x_cost_selected_rate
boss_combat/x_cost_selected_effective_energy_mean
boss_combat/x_cost_bad_selected_rate
```

### 5.3 Kaiser

```text
boss_combat/kaiser_crab_boss/kaiser_back_attack_risk_mean
boss_combat/kaiser_crab_boss/kaiser_facing_change_candidate_count_mean
boss_combat/kaiser_crab_boss/kaiser_facing_change_selected_rate
boss_combat/kaiser_crab_boss/kaiser_risky_end_turn_selected_rate
boss_combat/kaiser_crab_boss/kaiser_pressure_selected_rate
```

### 5.4 Planner 可信度

```text
planner/latent_drift_mean
planner/drift_gate_mean
planner/rollout_q_effective_weight_mean
planner/objective_q_effective_weight_mean
planner/risk_q_effective_weight_mean
planner/branch_disagreement_mean
planner/legal_f1_mean
```

### 5.5 Replay 分布

```text
replay/tier_sample_rate/boss
replay/tier_sample_rate/elite
replay/tier_sample_rate/normal
replay/encounter_sample_rate/<encounter_id>
```

---

## 6. 推荐本地验证命令

实际命令以 repo 当前脚本为准，可先用 `rg` 查训练入口和测试目录。

```powershell
cd E:\game\project\sts2_mcp

# 定位相关实现
rg -n "wasteful|strategic_defer|x_cost|kaiser|ceremonial|latent_drift|boss_combat" packages/rl-agent mods/sts2-bridge/Scripts

# Python 单测
cd packages/rl-agent
python -m pytest tests -q

# 针对新增测试
python -m pytest tests/test_end_turn_taxonomy.py -q
python -m pytest tests/test_x_cost_dynamic_energy.py -q
python -m pytest tests/test_kaiser_facing_semantics.py -q
```
