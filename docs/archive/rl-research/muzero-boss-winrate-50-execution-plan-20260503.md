# MuZero Boss WinRate >50% 恢复执行方案（2026-05-03）

> 目标：把当前 MuZero combat-sandbox 训练从“普通/精英能打、boss 明显掉队”的瓶颈中拉出来，最终以真实训练日志证明 boss 胜率超过 50%。
> 主验收：`recent_tail/256/boss_win_rate > 0.50`。
> 次验收：`recent_tail/64/boss_win_rate > 0.50` 且 Kaiser / Ceremonial / Knowledge Demon / The Kin / Insatiable 不出现单 boss 崩盘。
> 本文是**执行文档**，用于指导后续代码修复、launcher 调整、训练启动与监控；不要把本文本身当作“已修好”。

---

## 0. 当前判断

### 0.1 最新已知弱点

最新被核查的 run：

```text
LOG_DIR = E:\game\project\sts2_mcp\packages\rl-agent\logs_muzero\muzero_token_memory_combat_sandbox_2envs_20260503_082408
CKPT    = E:\game\project\sts2_mcp\packages\rl-agent\checkpoints_muzero\muzero_token_memory_combat_sandbox_2envs_20260503_082408\muzero_final
```

关键指标：

```text
recent_tail/64/boss_win_rate  last ≈ 0.315789
recent_tail/256/boss_win_rate last ≈ 0.166667
boss/win mean20               ≈ 0.30
boss/loss mean20              ≈ 0.70
```

结论：**boss 胜率远未达到目标**。不能因为 overall win / normal / elite 指标尚可就继续长训碰运气。

### 0.2 当前不是单一 bug

现在的问题是多个因素叠加：

1. **训练分布问题**
   旧 launcher 的 `ENCOUNTER_POOL` 明面上没有 boss，虽然 snapshot 采样和 tier weight 可能间接引入 boss，但这不足以服务“boss >50%”目标。必须显式 boss-heavy。

2. **planner/root bias 可观测性问题**
   旧日志里 `search/root_bias_scale = 0`，且缺失 `root_bias_nonzero_rate` / `root_bias_changed_top1_rate`，无法证明机制 bias 真的进入最终 action ranking。后续新 run 必须先证明这一点。

3. **boss 机制行动优先级问题**
   Kaiser 能看到背刺风险，也能看到 facing-change candidate，但转身选择率低，risky end_turn 高。Ceremonial 在 one-card-lock / stun-window 下仍会选低 impact 行动。

4. **行动质量分类问题**
   合法可打不等于应该打。消耗、保留、重放、虚无、替换、复制、回费、X 费、自损等机制都会让“有 playable card 但选择 end_turn”不一定是空过。

5. **桥接瞬时状态问题**
   抽卡/洗牌/结算动画窗口可能只暴露 `end_turn`。这不能靠长 sleep 解决，应该靠稳定状态判断和短轮询拦截。

6. **future-world / future-bank spike 问题**
   boss-heavy batch 曾触发 `future_world_aux_loss` / `future_bank_state_loss` 数千量级 spike。即使 median 稳定，也说明存在 OOD 或异常 target，必须 dump/quarantine。

---

## 1. 解决原则

### 1.1 先让机制影响最终选择，再谈长训

如果 root bias / action quality bias 没有改变最终 top1 action，那么：

- Kaiser facing bias 无效；
- Ceremonial stun-window bias 无效；
- potion timing bias 无效；
- HP-cost / X-cost 惩罚也可能只停留在日志里。

因此新 run 的第一验收不是 boss win，而是：

```text
search/root_bias_nonzero_rate                    exists and > 0
search/root_bias_changed_top1_rate               exists and > 0 on targeted suite
boss_combat/root_bias_nonzero_rate               exists and > 0
boss_combat/<encounter>/root_bias_changed_top1_rate exists on boss samples
```

如果这些不存在，不要训练 8 小时，直接停。

### 1.2 不再用“有 playable card”定义空过

必须把 end_turn 分成至少 5 类：

| 分类 | 含义 | 训练处理 |
|---|---|---|
| `forced_end_turn` | 当前确实没有合理动作 | 正常，无惩罚 |
| `transient_only_end_turn` | bridge 状态不稳定，暂时只暴露 end_turn | 不给 policy 选择；短轮询等待稳定 |
| `strategic_defer_end_turn` | 有牌可打，但出于消耗/保留/窗口/回费无 follow-up 等原因合理不打 | 不当作空过；可弱监督 |
| `true_wasteful_end_turn` | 有明确 urgent positive action 却结束回合 | 强负 bias / replay 降权 / 诊断 |
| `mechanism_bad_end_turn` | boss 机制窗口或风险下结束回合 | boss-specific 强惩罚 |

不要再使用旧 `boss_combat/wasteful_end_turn_rate = 0` 作为“没有空过”的证据。优先看：

```text
boss_combat/true_wasteful_end_turn_available_rate
boss_combat/true_wasteful_end_turn_selected_rate
boss_combat/bad_end_turn_available_rate
boss_combat/bad_end_turn_selected_rate
boss_combat/forced_end_turn_selected_rate
boss_combat/strategic_defer_end_turn_selected_rate
```

### 1.3 Boss 战不要强追低战损

用户已经明确：多数难度下 boss 后会回满血，boss 战的战损不是主目标。
因此 boss reward 不应强行鼓励“低损过 boss”，而应鼓励：

- 赢；
- 不自杀；
- 不错过机制窗口；
- 不在会被敌方行动击杀时打自损牌；
- 不浪费关键药水或关键一次性卡牌；
- 在 A10 / 不回血规则存在时单独切换战损权重。

### 1.4 Schema 变化后不要加载旧 buffer

只要改了 action annotation / observation / aux target / reward 语义，就必须：

```text
--resume-from <checkpoint>
--resume-without-optimizer
--resume-without-buffer
```

可以 warm-start 权重，但不要加载旧 replay buffer。旧 buffer 的 schema 和旧策略分布会污染新机制学习。

---

## 2. P0：启动新 boss 恢复 run 前必须完成

### P0-1. 确认 root-bias instrumentation 已进入所有路径

#### 目标

无论 combat 使用 direct/search-free planner 还是 MCTS，都必须 emit：

```text
search/root_bias_nonzero_rate
search/root_bias_abs_mean
search/root_bias_max_abs
search/root_bias_changed_top1_rate
search/root_bias_selected_action_delta_mean
search/root_bias_suppressed_by_gate_rate
search/root_bias_scale_effective_mean

boss_combat/root_bias_nonzero_rate
boss_combat/root_bias_changed_top1_rate
boss_combat/<encounter>/root_bias_changed_top1_rate
```

#### 执行者检查路径

```text
packages/rl-agent/muzero/train.py
packages/rl-agent/muzero/sts2_env/mcts.py
```

#### 验收

1. 单元测试通过。
2. 新 run 产生上述 TensorBoard tag。
3. targeted boss suite 中 `root_bias_changed_top1_rate > 0`。
4. 如果 `root_bias_nonzero_rate > 0` 但 `changed_top1_rate = 0`，说明 bias 太弱或 policy logits 过度自信，需要提高 boss 机制 bias 或在 targeted warmup 阶段禁用/降低 decay。

---

### P0-2. 创建 boss-focused launcher，不要直接复用旧 launcher

#### 问题

旧 launcher 名称是 `2envs`，但实际 `--n-envs 1`；`ENCOUNTER_POOL` 主要是 weak/normal/elite，没有显式 boss。对当前目标不合适。

#### 新 launcher 建议

新建：

```text
packages/rl-agent/launch_muzero_boss_recovery_20260503.sh
```

关键原则：

```bash
RESUME_CKPT='/mnt/e/game/project/sts2_mcp/packages/rl-agent/checkpoints_muzero/muzero_token_memory_combat_sandbox_2envs_20260503_082408/muzero_final'

BOSS_POOL='ENCOUNTER.KAISER_CRAB_BOSS,ENCOUNTER.CEREMONIAL_BEAST_BOSS,ENCOUNTER.THE_KIN_BOSS,ENCOUNTER.THE_INSATIABLE_BOSS,ENCOUNTER.KNOWLEDGE_DEMON_BOSS'
HARD_ELITE_POOL='ENCOUNTER.PHROG_PARASITE_ELITE,ENCOUNTER.KNIGHTS_ELITE,ENCOUNTER.SOUL_NEXUS_ELITE'
NORMAL_SUPPORT_POOL='ENCOUNTER.CONSTRUCT_MENAGERIE_NORMAL,ENCOUNTER.SLUMBERING_BEETLE_NORMAL,ENCOUNTER.OVICOPTER_NORMAL'

ENCOUNTER_POOL="${BOSS_POOL},${HARD_ELITE_POOL},${NORMAL_SUPPORT_POOL}"
```

训练参数必须包含：

```bash
--resume-from "${RESUME_CKPT}"
--resume-without-optimizer
--resume-without-buffer
--combat-encounter-tiers boss,elite,normal
--combat-snapshot-sample-mode tier_weighted_encounter_balanced
--combat-tier-weights 'weak=0.00,normal=0.45,elite=0.85,boss=2.75'
--combat-encounter-weights 'ENCOUNTER.CEREMONIAL_BEAST_BOSS=5.20,ENCOUNTER.KAISER_CRAB_BOSS=4.80,ENCOUNTER.THE_KIN_BOSS=3.20,ENCOUNTER.THE_INSATIABLE_BOSS=2.80,ENCOUNTER.KNOWLEDGE_DEMON_BOSS=2.20,ENCOUNTER.PHROG_PARASITE_ELITE=2.20,ENCOUNTER.KNIGHTS_ELITE=2.40,ENCOUNTER.SOUL_NEXUS_ELITE=2.80'
```

保留当前 direct 模式核心参数：

```bash
--combat-policy-mode direct
--build-num-simulations 1
--route-num-simulations 1
--combat-rollout-steps 3
--combat-rollout-beam-width 2
--combat-rollout-q-blend 0.75
--combat-rollout-objective-q-blend 0.50
--combat-rollout-risk-blend 0.35
--combat-rollout-uncertainty-blend 0.35
--future-world-rollout-steps 2
--future-world-rollout-weight 0.35
--future-bank-token-slot-source-weight 0
```

#### 验收

启动 30 分钟内必须看到：

```text
recent_tail/*/boss_win_rate 有样本
recent_tail/*/ENCOUNTER.KAISER_CRAB_BOSS/win_rate 或等价 per-encounter tag 有样本
recent_tail/*/ENCOUNTER.CEREMONIAL_BEAST_BOSS/win_rate 或等价 per-encounter tag 有样本
boss/win, boss/loss 持续更新
```

如果 30 分钟内 boss 样本很少或没有，说明 encounter pool / snapshot filter / tier weights 没生效，立刻停。

---

### P0-3. 训练前测试清单

在 `packages/rl-agent` 下执行：

```powershell
.\venv\Scripts\python.exe -m pytest -q `
  tests/test_hp_cost_safety_p0_1.py `
  tests/test_transient_leak_p0_2.py `
  tests/test_kaiser_resolver_unification_p0_3.py `
  tests/test_x_cost_dynamic_p0_4.py `
  tests/test_selection_typed_p0_5.py `
  tests/test_card_identity_p0_6.py `
  tests/test_loss_spike_dump_p0_7.py `
  tests/test_p0_diag_flat_keys_for_metrics.py `
  tests/test_p0_helpers_action_diagnostics_integration.py
```

验收：全部通过。当前已知基线为：

```text
93 passed, 17 subtests passed
```

---

### P0-4. 启动前环境检查

不要盲目启动训练。先检查：

```powershell
Get-Process python -ErrorAction SilentlyContinue |
  Select-Object Id,CPU,PM,StartTime,Path,CommandLine | Format-List
```

确认没有重复 MuZero 长训进程。

检查 bridge session：

```powershell
Test-Path C:\Users\yidhar\AppData\Roaming\SlayTheSpire2\bridge\session_0.json
```

检查 WSL/ROCm venv：

```powershell
wsl bash -lc "test -x /mnt/e/game/project/sts2_mcp/packages/rl-agent/.venv-wsl-rocm/bin/python && echo OK"
```

---

## 3. P1：第一个小时的强制监控 gate

新 run 启动后，不要等整晚。第一个小时按以下 gate 决策。

### 3.1 10 分钟 gate：链路是否活着

必须满足：

| 指标 | 要求 |
|---|---:|
| events mtime | 持续更新 |
| optimizer step | 增长 |
| buffer/size | 增长 |
| boss/win or boss/loss | 开始出现 |
| loss/total | 没有持续 NaN / inf |

失败处理：停训，查 bridge/session/launcher。

### 3.2 30 分钟 gate：boss 分布是否正确

必须满足：

| 指标 | 要求 |
|---|---:|
| boss attempt count | 持续增长 |
| Kaiser samples | 非零 |
| Ceremonial samples | 非零 |
| `recent_tail/64/boss_win_rate` | 可计算 |
| per-encounter recent-tail | 至少主要 boss 有样本 |

失败处理：停训，修 `ENCOUNTER_POOL` / `--combat-encounter-tiers` / snapshot dataset filter。

### 3.3 30 分钟 gate：root bias 是否真的工作

必须满足：

| 指标 | 要求 |
|---|---:|
| `search/root_bias_nonzero_rate` | > 0 |
| `boss_combat/root_bias_nonzero_rate` | > 0 |
| `boss_combat/root_bias_abs_mean` | > 0 |
| `boss_combat/<boss>/root_bias_changed_top1_rate` | targeted 中 > 0 |

失败处理：

1. 如果所有 root bias tag 缺失：writer/aggregation 没接上。
2. 如果 tag 存在但全 0：`_combat_action_quality_bias` 没进 selected path，或 gated 掉。
3. 如果 nonzero > 0 但 changed_top1 = 0：bias 弱或 policy 过度自信，短期提高 Kaiser/Ceremonial hard bias，或 warmup 禁止 decay。

### 3.4 60 分钟 gate：机制指标是否改善

#### Kaiser

目标：

```text
boss_combat/kaiser_crab_boss/kaiser_back_attack_risk_mean              > 0.5
boss_combat/kaiser_crab_boss/kaiser_facing_change_candidate_count_mean > 0
boss_combat/kaiser_crab_boss/kaiser_facing_change_selected_rate        >= 0.15 first gate, >= 0.20 preferred
boss_combat/kaiser_crab_boss/kaiser_risky_end_turn_selected_rate       <= 0.10 first gate, <= 0.05 preferred
boss_combat/kaiser_crab_boss/root_bias_changed_top1_rate               > 0
```

如果 risk/candidate 都有，但 selected 仍低，说明不是 bridge 观测问题，而是 action ranking 问题。继续加：

- facing change action bonus；
- risky end_turn hard penalty；
- pressure-vs-facing priority arbitration；
- rollout value 中显式加入 `back_attack_risk_delta`。

#### Ceremonial

目标：

```text
boss_combat/ceremonial_beast_boss/ceremonial_high_impact_selected_rate ↑
boss_combat/ceremonial_beast_boss/ceremonial_low_impact_selected_rate  ↓
boss_combat/ceremonial_beast_boss/ceremonial_missed_stun_window_rate   ↓
boss_combat/ceremonial_beast_boss/root_bias_changed_top1_rate          > 0
```

如果 one-card-lock / stun-window 下仍选低 impact：

- 提高 single-action impact score 的 root bias；
- end_turn under lock 单独惩罚；
- 对 stun-window 内 lethal/high-damage/debuff 统一打机制标签；
- dump offender legal actions。

#### Potion

目标不是“更多用药”，而是“更会用药”：

```text
boss_combat/potion_low_urgency_selected_rate      下降
boss_combat/potion_high_urgency_selected_rate     上升或至少非零
boss_combat/potion_lethal_selected_rate           非零 when lethal available
boss_combat/potion_prevent_lethal_selected_rate   非零 when prevent-lethal available
boss_combat/potion_save_recommended_selected_rate 低
```

如果模型“有药就用”：提高 `save_value` / `low_urgency` 惩罚。
如果模型“关键时不用”：提高 `lethal` / `prevent_lethal` / `mechanism_answer` 奖励。

#### HP-cost

目标：

```text
boss_combat/hp_cost_self_lethal_selected_rate = 0
boss_combat/hp_cost_low_margin_selected_rate  下降
```

判断不能只看 `hp_after_cost > 0`，还要看：

```text
survival_margin = hp_after_cost + current_block - incoming_damage_after_action
```

低 margin 时打自损牌必须强惩罚或 hard mask，尤其敌方将攻击时。

#### X-cost

目标：

```text
boss_combat/x_cost_zero_bad_selected_rate_p0 < 0.1%
```

X 费必须绑定当前实际 energy，不得读取初始 3 费或 stale obs。0 能量 X 费如果没有非 X side effect，视为 bad action。

#### End-turn transient

目标：

```text
combat/transient_leaked_selected_rate = 0
slow_bridge_step end_turn 不持续放大
```

禁止用总计数秒级 sleep 解决。应使用：

- 当前 action list 里只有 end_turn；
- player/hand/draw/discard/energy/turn state 是否处于结算变更；
- 连续短轮询 state hash 是否稳定；
- 若 stable 且确实无可用动作，才暴露 end_turn。

---

## 4. P2：训练阶段规划

### 4.1 Stage A：boss mechanism warmup（0~1 小时）

目的：验证机制链路，而不是追最终胜率。

采样建议：

```text
boss 70%~80%
elite 15%~20%
normal 5%~10%
weak 0%
```

重点 boss：

```text
Kaiser Crab
Ceremonial Beast
The Kin
The Insatiable
Knowledge Demon
```

停止条件：

- root bias tag 缺失；
- boss sample 太少；
- Kaiser/Ceremonial 指标无任何改善；
- future-world/bank spike 连续超过阈值；
- X-cost / HP-cost / transient leak 反弹。

### 4.2 Stage B：boss-heavy mixed recovery（1~3 小时）

目的：让 boss 策略从 targeted 机制修复过渡到混合战斗。

采样建议：

```text
boss 45%~55%
elite 25%~35%
normal 15%~25%
weak 0%~5%
```

观察：

```text
recent_tail/64/boss_win_rate >= 0.40
Kaiser 64 win               > 0.20
Ceremonial 64 win            > 0.20
The Kin / Insatiable         不低于旧基线
loss median                  稳定
```

如果 overall win 上升但 boss 不升，说明分布仍被 normal/elite 稀释。

### 4.3 Stage C：formal boss target run（3~8 小时）

目的：冲击 `recent_tail/256/boss_win_rate > 0.50`。

采样建议：

```text
boss 30%~40%
elite 25%~30%
normal 25%~35%
weak 5%~10%
```

通过条件：

```text
recent_tail/64/boss_win_rate  > 0.50
recent_tail/256/boss_win_rate > 0.50
boss/win mean20               > 0.50
```

并且 per-boss 不崩：

```text
Kaiser latest window          不为 0
Ceremonial latest window      不为 0
Knowledge Demon latest window 不为 0
The Kin / Insatiable          不显著低于旧均值
```

### 4.4 Stage D：full distribution 验证

目标达成后再恢复更自然的 mixed distribution，确认不是只会打 boss snapshots。

要求：

```text
overall recent_tail/256/win_rate      不显著下降
recent_tail/256/boss_win_rate         仍 > 0.50 或缓慢下降但稳定 > 0.45
normal / elite per-encounter win      不崩
potion / HP-cost / X-cost bad metrics 不反弹
```

---

## 5. Future-world / Future-bank spike 处理

### 5.1 必须 dump，而不是只看 median

出现以下情况立刻 dump：

```text
loss/total > 100
loss/future_world_aux > 100
loss/future_bank_state > 100
loss/future_bank_delta > 5
```

dump 内容至少包含：

```text
step
loss components
sample tier distribution
encounter ids
prev_obs compact hash
next_obs compact hash
action family / action id / compact signature
transition_state key counts
aux target min/max/mean
future-bank target min/max/mean
raw obs action/legal count
potion slots summary
hand/draw/discard/exhaust counts
enemy powers summary
boss mechanism context
```

### 5.2 Quarantine 规则

如果某类样本反复触发 spike：

- 不要把整个 boss 分布关掉；
- 先按 encounter/action-family/target-field quarantine；
- 记录 offender；
- 只有确认是 target 构造 bug 时才重训。

### 5.3 训练继续/停止判断

| 情况 | 决策 |
|---|---|
| 单次 spike，但 median 恢复，dump 完整 | 可继续观察 |
| 同一 offender 连续 spike | 停训修 target |
| spike 后 policy/value/reward 也漂移 | 停训，不能继续污染 optimizer |
| spike 只在 boss-heavy batch 出现 | 优先查 boss mechanism token / powers / card lifecycle delta |

---

## 6. 模仿学习是否要引入

可以，但不应该作为第一步。当前 first-order 问题仍是机制链路和采样分布。

### 6.1 适合引入 imitation 的条件

当以下条件满足但 boss win 仍卡住时，引入用户手打示范：

```text
root_bias metrics 正常
Kaiser/Ceremonial candidate 与 selected 指标正常
X-cost / HP-cost / transient / potion bad metrics 受控
future-world/bank spike 不再污染训练
recent_tail/64/boss_win_rate 卡在 0.35~0.45 超过 2~3 小时
```

### 6.2 示范数据采集目标

不要只采胜局整局。优先采机制窗口：

| 场景 | 需要示范 |
|---|---|
| Kaiser 背刺风险 + 可转身 | 攻击哪侧、何时转身、何时直接压死 |
| Ceremonial one-card-lock | 哪张牌值得吃锁窗口 |
| Ceremonial stun-window | stun 打开时如何最大化 impact |
| 有关键药水 | 何时保存、何时斩杀、何时保命 |
| 自损回费牌 | 何时打、何时因为血线不打 |
| 消耗/保留/虚无/重放牌 | 何时延迟，何时必须马上用 |

### 6.3 Imitation 训练方式

推荐作为 auxiliary supervised loss / replay priority，而不是完全替代 RL：

```text
human_action_prior_loss
mechanism_window_imitation_loss
boss_encounter_replay_priority_bonus
```

并记录：

```text
imitation/sample_count
imitation/boss_sample_count
imitation/mechanism_window_count
imitation/policy_kl_to_human
imitation/top1_match_rate
```

---

## 7. 监控脚本建议

每次监控至少输出这些 tags 的 last / mean20 / n：

```text
recent_tail/64/boss_win_rate
recent_tail/256/boss_win_rate
boss/win
boss/loss

search/root_bias_nonzero_rate
search/root_bias_changed_top1_rate
search/root_bias_abs_mean
search/root_bias_scale_effective_mean

boss_combat/root_bias_nonzero_rate
boss_combat/root_bias_changed_top1_rate
boss_combat/true_wasteful_end_turn_selected_rate
boss_combat/bad_end_turn_selected_rate
boss_combat/forced_end_turn_selected_rate
boss_combat/strategic_defer_end_turn_selected_rate

boss_combat/kaiser_crab_boss/kaiser_facing_change_candidate_count_mean
boss_combat/kaiser_crab_boss/kaiser_facing_change_selected_rate
boss_combat/kaiser_crab_boss/kaiser_risky_end_turn_selected_rate
boss_combat/kaiser_crab_boss/root_bias_changed_top1_rate

boss_combat/ceremonial_beast_boss/ceremonial_high_impact_selected_rate
boss_combat/ceremonial_beast_boss/ceremonial_low_impact_selected_rate
boss_combat/ceremonial_beast_boss/ceremonial_missed_stun_window_rate
boss_combat/ceremonial_beast_boss/root_bias_changed_top1_rate

boss_combat/potion_low_urgency_selected_rate
boss_combat/potion_high_urgency_selected_rate
boss_combat/potion_lethal_selected_rate
boss_combat/potion_prevent_lethal_selected_rate

boss_combat/x_cost_zero_bad_selected_rate_p0
boss_combat/hp_cost_self_lethal_selected_rate
boss_combat/hp_cost_low_margin_selected_rate

loss/total
loss/future_world_aux
loss/future_bank_state
loss/future_bank_delta
```

如果某些 tag 名实际不同，监控脚本要做 fuzzy search，但报告里必须列出真实 tag 名，不能猜。

---

## 8. 推荐的执行顺序

### 8.1 代码/配置层

1. 确认 root bias instrumentation 编译通过。
2. 新建 boss recovery launcher。
3. 确认 launcher 明确包含 boss pool。
4. 加 `--resume-without-buffer` 和 `--resume-without-optimizer`。
5. 跑 P0 单元测试。
6. 启动短 run。
7. 10/30/60 分钟 gate。
8. 如果 gate 失败，停训修对应链路。
9. gate 通过后才进入 3~8 小时正式 run。

### 8.2 不要做的事

- 不要直接复用旧 launcher 长训。
- 不要用旧 replay buffer。
- 不要看到 loss median 稳就忽略 spike。
- 不要只看 global boss_combat 均值。
- 不要用 sleep 解决 only-end_turn。
- 不要把 potion use rate 当优化目标。
- 不要把 boss 战战损作为主奖励。
- 不要把 `wasteful_end_turn_rate = 0` 当没有空过的证据。
- 不要把 tests passed 当 boss 目标完成。

---

## 9. Definition of Done

本任务只有满足以下条件才能算完成：

### 9.1 训练目标

```text
recent_tail/256/boss_win_rate > 0.50
```

如果暂时只有：

```text
recent_tail/64/boss_win_rate > 0.50
```

只能算“短窗达标”，不能算最终完成。

### 9.2 机制目标

必须同时满足：

```text
root_bias metrics present and nonzero
Kaiser facing selected rate 显著高于旧值
Kaiser risky end_turn 明显下降
Ceremonial high-impact selected 上升
Ceremonial missed stun / low-impact under lock 下降
potion low-urgency use 下降，高 urgency use 不为 0
x_cost_zero_bad 接近 0
hp_cost_self_lethal = 0
transient leaked end_turn = 0
future-world/bank spike 有 dump 或已消失
```

### 9.3 分布目标

不能只靠某一个 boss 或某一类 snapshot 过拟合：

```text
Kaiser / Ceremonial / Knowledge Demon / The Kin / Insatiable 均有样本
没有任意核心 boss latest window 长期 0%
normal / elite 指标不崩
```

---

## 10. 给执行模型的最小任务拆分

如果把本文交给 Claude 或其他执行模型，按下面拆分，避免互相覆盖：

### Task A：launcher/config owner

负责：

```text
packages/rl-agent/launch_muzero_boss_recovery_20260503.sh
```

产出：boss-heavy launcher，fresh buffer resume，明确 boss pool。

### Task B：metric/monitor owner

负责：

```text
packages/rl-agent/scripts 或临时 monitor 脚本
```

产出：自动读取最新 run，输出本文第 7 节 tags，缺失 tag 做 fuzzy search。

### Task C：root-bias verification owner

负责：

```text
packages/rl-agent/muzero/train.py
packages/rl-agent/muzero/sts2_env/mcts.py
```

产出：证明 direct 与 MCTS path 都 emit root-bias metrics；不改动无关训练逻辑。

### Task D：boss-mechanism owner

负责：

```text
packages/rl-agent/sts2_env/boss_mechanics.py
packages/rl-agent/sts2_env/boss_ceremonial.py
packages/rl-agent/muzero/train.py 中 boss diagnostics / quality bias 区域
```

产出：Kaiser/Ceremonial 指标和 bias 能改变 action ranking。

### Task E：safety/action-quality owner

负责：

```text
packages/rl-agent/sts2_env/hp_cost_safety.py
packages/rl-agent/sts2_env/x_cost_dynamic.py
packages/rl-agent/sts2_env/card_identity.py
packages/rl-agent/sts2_env/selection_typed.py
packages/rl-agent/muzero/train.py 中 _combat_action_quality_bias 区域
```

产出：HP-cost/X-cost/end_turn strategic defer 分类不互相误伤。

### Task F：spike-dump owner

负责：

```text
packages/rl-agent/muzero/train.py
packages/rl-agent/tests/test_loss_spike_dump_p0_7.py
```

产出：loss spike dump 可定位 offender，并且不会拖慢正常训练。

---

## 11. 最短执行命令草案

> 注意：下面是草案。启动前必须确认 launcher 文件已经创建并且 bridge/session 可用。

```powershell
cd E:\game\project\sts2_mcp\packages\rl-agent
.\venv\Scripts\python.exe -m pytest -q `
  tests/test_hp_cost_safety_p0_1.py `
  tests/test_transient_leak_p0_2.py `
  tests/test_kaiser_resolver_unification_p0_3.py `
  tests/test_x_cost_dynamic_p0_4.py `
  tests/test_selection_typed_p0_5.py `
  tests/test_card_identity_p0_6.py `
  tests/test_loss_spike_dump_p0_7.py `
  tests/test_p0_diag_flat_keys_for_metrics.py `
  tests/test_p0_helpers_action_diagnostics_integration.py
```

```powershell
cd E:\game\project\sts2_mcp
wsl bash -lc "cd /mnt/e/game/project/sts2_mcp/packages/rl-agent && bash ./launch_muzero_boss_recovery_20260503.sh"
```

监控时每 10~30 分钟读取最新 event，不要只看 stdout。

---

## 12. 当前优先级结论

下一步不是继续分析旧日志，也不是直接跑旧配置。正确顺序是：

1. **新建 boss recovery launcher**。
2. **用最新 checkpoint warm-start，但不加载旧 optimizer/buffer**。
3. **启动 30~60 分钟 boss-heavy 短 run**。
4. **先看 root-bias / per-boss 机制指标是否生效**。
5. **只有 gate 通过后才继续冲 boss 256-window >50%**。
6. **如果 gate 失败，按缺失指标定位：分布、root bias、Kaiser/Ceremonial、potion、HP-cost、X-cost、transient、spike。**
