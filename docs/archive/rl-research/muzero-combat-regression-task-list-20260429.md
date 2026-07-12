# MuZero Combat Sandbox 回归修复与目标态落地 Task List

日期: 2026-04-29
范围: `packages/rl-agent` 的 MuZero / token-memory / combat sandbox / bridge action semantics / reward shaping / telemetry。
目标: 不是做“最小修补”，而是把当前已经暴露出的 **空过、机制 boss、动态费用、卡牌状态变更、药水时机、direct planner 漂移、采样退化** 统一收束成一套可训练、可诊断、可继续扩展的目标实现。

---

## 0. 当前问题结论与证据锚点

### 0.1 最近训练回归不是错觉

最近两轮日志对比显示，最新 run 相对上一轮出现明显回退：

| 指标 | 上一轮约值 | 当前约值 | 变化 | 解释 |
|---|---:|---:|---:|---|
| `256/win` | 70.4% | 61.9% | -8.5 pct | 全局通过率下降 |
| `256/boss_win` | 33.1% | 25.7% | -7.4 pct | boss 整体变弱 |
| `64/boss_win` | 41.1% | 21.0% | -20.1 pct | 短窗 boss 退化更明显 |
| `256/elite_win` | 76.8% | 66.8% | -10.0 pct | elite 也退化 |
| `256/normal_win` | 99.3% | 88.7% | -10.6 pct | normal 也出现遗忘 |
| `boss_combat/family_end_turn_rate` | 20.0% | 25.1% | +5.1 pct | 结束回合比例上升 |
| `boss_combat/wasteful_end_turn_available_rate` | 57.8% | 65.2% | +7.4 pct | 大量状态存在“可浪费”的结束回合风险 |
| `boss_combat/wasteful_end_turn_selected_rate` | 0.0% | 0.0% | 表面正常 | 但与人工观察冲突，说明 detector 口径仍有漏洞 |
| `strategic_skip_selected_rate` | 8.4% | 9.8% | +1.4 pct | strategic skip 可能过宽 |
| `refund_no_followup_selected_rate` | 3.8% | 5.1% | +1.3 pct | 回费牌无后续收益判断可能过宽 |
| `direct_rollout_latent_drift_mean` | 0.513 | 0.574 | +0.061 | search-free direct planner 想象漂移变大 |
| `metric/predicted_legal_f1` | 0.798 | 0.813 | +0.015 | legal head 没崩，退化不是合法动作预测崩溃 |
| `boss_sample_rate` | 77.5% | 82.5% | +5.0 pct | boss-heavy 采样可能导致 normal/elite 遗忘 |
| `temperature` | 0.201 | 0.119 | -0.082 | 探索降低，容易锁死坏策略 |

### 0.2 per-encounter 退化重点

| Encounter | 上一轮 last | 当前 last | 变化 | 初判 |
|---|---:|---:|---:|---|
| `ceremonial_beast_boss` | 30.0% | 0.0% | -30.0 pct | 机制 boss，高影响单卡选择仍没稳定学会 |
| `the_insatiable_boss` | 45.5% | 16.7% | -28.8 pct | strategic skip / refund-no-followup 过宽嫌疑最大 |
| `knights_elite` | 73.3% | 53.3% | -20.0 pct | 基础攻防/节奏退化 |
| `the_kin_boss` | 55.6% | 35.7% | -19.8 pct | boss 机制/伤害节奏退化 |
| `phrog_parasite_elite` | 80.0% | 63.6% | -16.4 pct | elite 行为退化 |
| `kaiser_crab_boss` | 9.1% | 7.1% | -1.9 pct | 仍没解决，但不是这轮最大回退来源 |

### 0.3 最高优先级根因排序

1. **end_turn / strategic_skip / refund-no-followup 的动作质量分类口径仍不完整**
   人眼看到空过，但 `wasteful_end_turn_selected_rate=0`，说明当前 detector 能看到“存在风险”，却不能稳定把“被选中的坏 end_turn”归因出来。

2. **direct planner latent drift 连续升高**
   direct 模式退出 MCTS 后，Q-like lookahead 更依赖世界模型想象。如果 latent drift 增大，rollout rank 会变脆，尤其在 boss 机制和多步卡牌状态变更上。

3. **boss-heavy 采样 + temperature 过低导致行为锁死与遗忘**
   boss 采样过高没有换来 boss 稳定提升，反而 normal/elite 下降，说明训练分布/课程调度需要重排。

4. **机制 boss 的状态信号存在，但 action outcome / value 绑定不足**
   Kaiser 风险信号已进来，但 facing-change/action outcome 不够完整；Ceremonial one-card-lock 信号在，但模型没有稳定选择高影响动作。

5. **卡牌状态变更与战斗内卡牌生命周期没有成为强约束 token**
   消耗、虚无、重放、保留、附魔、降费、变化、复制、临时卡、回费牌后续可打性，都需要进入状态 token、action token、transition aux 和 reward/quality 诊断。

---

## 1. 总体执行原则

### 1.1 不再做的事

- [ ] 不再只靠单点 bias 去“压一下”某个坏动作。
- [ ] 不再把人工观察到的空过简单归因成“模型没学会”，必须先让 detector 能解释为什么它没算作 bad end_turn。
- [ ] 不再用长固定 sleep 解决 bridge 暂态合法动作问题；只能用确定性状态条件和短轮询。
- [ ] 不再让 boss-heavy 采样无限加码；采样要服务学习，不是让 replay 分布被 boss 吃掉。
- [ ] 不再只看全局 boss 聚合指标；必须 per-encounter / per-card / per-action-family 拆开。

### 1.2 目标态定义

最终目标不是“某个 boss 胜率短暂上去”，而是：

- [ ] 模型在 combat sandbox 中能稳定学会基础攻防节奏。
- [ ] direct planner 不依赖 MCTS，也能进行短 horizon Q-like / lookahead value 排序。
- [ ] hand / draw / discard / exhaust / potion / relic / enemy / intent / power / energy / hp / card-state mutation 全部 token 化并互相关联。
- [ ] 动作 token 不只表示“这张牌当前合法”，还表示“打出后的世界变化”：费用、抽弃耗、手牌强化/变化/保留、消耗、重放、目标机制影响、药水消耗、遗物触发机会。
- [ ] end_turn 被拆成 forced / strategic_defer / bad 三类，且每一次 selected end_turn 都能给出 reason。
- [ ] 机制 boss 的特殊机制不靠硬规则通关，而是成为模型可见、可预测、可被 aux 监督的 world feature。
- [ ] 所有关键行为都有 TB 指标 + JSONL offender dump，能定位具体卡牌/动作/状态。

---

## 2. P0 — 先把可观测性补齐，避免继续盲改

> 目标: 先让日志能够解释“为什么模型这样打”，特别是 end_turn、strategic skip、refund、X 费、Kaiser、Ceremonial、Insatiable。

### 2.1 建立 selected end_turn 上下文 dump

**文件:**

- `packages/rl-agent/muzero/train.py`
- 输出目录: `logs_muzero/<run>/diagnostics/end_turn_contexts.jsonl`

**任务:**

- [ ] 在 direct planner / action selection 之后，只要最终选中 `end_turn`，记录一条 JSONL。
- [ ] dump 必须覆盖 global + per-encounter。
- [ ] 不只记录 selected action，还要记录“为什么没有被算成 wasteful”。

**字段要求:**

```json
{
  "step": 123456,
  "episode_id": "...",
  "encounter_id": "the_insatiable_boss",
  "tier": "boss",
  "turn": 3,
  "selected_action_id": "end_turn:...",
  "energy": 1.5,
  "block": 4,
  "hp": 38,
  "max_hp": 80,
  "incoming_damage": 18,
  "hand_count": 5,
  "draw_count": 12,
  "discard_count": 7,
  "exhaust_count": 2,
  "legal_action_count": 8,
  "playable_cards_left": 3,
  "positive_action_count": 4,
  "urgent_positive_count": 1,
  "mandatory_positive_count": 1,
  "deferable_positive_count": 2,
  "strategic_skip_candidate_count": 1,
  "refund_no_followup_candidate_count": 1,
  "zero_cost_urgent": false,
  "x_cost_available": true,
  "x_cost_zero_energy_available": false,
  "boss_mechanics": {
    "kaiser_back_attack_risk": 0.0,
    "ceremonial_one_card_lock": 0.0,
    "insatiable_specific_flags": {}
  },
  "reason_not_bad_end_turn": "urgent_positive_count==0|all_positive_marked_deferable",
  "top_legal_actions": [
    {
      "rank": 0,
      "action_id": "play_card:...",
      "family": "play_card",
      "card_id": "...",
      "title": "...",
      "cost": 1,
      "effective_cost": 1,
      "damage": 6,
      "block": 0,
      "draw": 0,
      "energy_delta": 0,
      "roles": ["attack"],
      "typed_effects": ["damage"],
      "quality_flags": ["positive", "mandatory"],
      "q": -0.2,
      "policy_logit": 1.3,
      "quality_bias": 0.4
    }
  ]
}
```

**验收:**

- [ ] 任意 100 个 selected end_turn 都能解释成 `forced_end_turn` / `strategic_defer_end_turn` / `bad_end_turn` 之一。
- [ ] `reason_not_bad_end_turn` 不能为空。
- [ ] 对人工观察到的空过，dump 中能定位是：
  - [ ] transient legal-action 暂态问题；或
  - [ ] urgent 分类漏掉；或
  - [ ] strategic_defer 过宽；或
  - [ ] planner Q 估值压过了 immediate quality。

### 2.2 统一 global 与 per-encounter 指标命名

**文件:**

- `packages/rl-agent/muzero/train.py`

**任务:**

- [ ] 所有 boss 聚合指标必须同时 emit：
  - `boss_combat/<metric>`
  - `boss_combat/<encounter_id>/<metric>`
- [ ] 目前已知有 per-encounter 出现但 global 根命名空间缺失的情况，需要统一。

**必须覆盖的 metric:**

```text
family_end_turn_rate
wasteful_end_turn_available_rate
wasteful_end_turn_selected_rate
wasteful_end_turn_bias_applied_rate
bad_end_turn_selected_rate
forced_end_turn_selected_rate
strategic_defer_end_turn_selected_rate
energy_mean
positive_action_count_mean
urgent_positive_action_count_mean
mandatory_positive_action_count_mean
playable_cards_left_mean
strategic_skip_selected_rate
refund_no_followup_selected_rate
x_cost_available_rate
x_cost_zero_energy_available_rate
x_cost_zero_energy_selected_rate
x_cost_effective_energy_mean
potion_available_rate
potion_selected_when_available_rate
potion_low_quality_selected_rate
potion_high_save_value_unused_rate
```

**验收:**

- [ ] TensorBoard 中能同时看到 `boss_combat/wasteful_end_turn_available_rate` 和 `boss_combat/the_insatiable_boss/wasteful_end_turn_available_rate`。
- [ ] 每个 metric 在 10 个 episode 内至少 emit 一次，不能等到窗口太大才出现。

### 2.3 添加 action offender dump

**文件:**

- `packages/rl-agent/muzero/train.py`
- 输出目录: `logs_muzero/<run>/diagnostics/action_offenders.jsonl`

**任务:**

- [ ] 当模型选择以下动作时采样 dump：
  - [ ] bad end_turn
  - [ ] strategic skip
  - [ ] refund-no-followup
  - [ ] zero-energy X-cost
  - [ ] low-quality potion
  - [ ] high-save-value potion 被浪费
  - [ ] Kaiser risky end_turn
  - [ ] Ceremonial low-impact action under one-card-lock
- [ ] 每条 dump 必须带 card/action title 和 typed effect，不允许只写 action_id。

**验收:**

- [ ] 可以直接按 `encounter_id + card_id + offender_type` 聚合出 top offenders。
- [ ] 至少能回答：
  - [ ] Insatiable 上哪些牌/动作导致 strategic skip 上升？
  - [ ] 哪些 X 费牌在 0 能量被打？
  - [ ] Ceremonial one-card-lock 下模型选择了哪些低影响动作？

### 2.4 添加 X 费动态费用诊断

**文件:**

- `packages/rl-agent/muzero/train.py`
- `packages/rl-agent/sts2_env/semantic_action.py`
- `packages/rl-agent/sts2_env/observation_v3.py`

**任务:**

- [ ] 在 action feature 中显式区分：
  - `base_cost`
  - `current_cost`
  - `effective_cost`
  - `is_x_cost`
  - `x_cost_effective_energy`
  - `x_cost_expected_damage_or_block`
  - `x_cost_expected_draw_or_gain`
- [ ] `effective_cost` 必须从当前 step 的 `player.energy` 或 bridge action payload 的实时值计算，不能被初始 3 能量状态污染。
- [ ] 对 0 能量 X 费动作加 offender dump。

**验收:**

- [ ] `boss_combat/x_cost_zero_energy_available_rate` 与 `selected_rate` 均能 emit。
- [ ] 如果 `selected_rate > 0`，JSONL 能列出具体牌名、当前能量、预期效果是否为 0。
- [ ] 0 能量打 X 费牌只有在明确存在非能量收益时才允许归类为合理，例如触发遗物/附魔/状态迁移；否则记为 offender。

### 2.5 potion final-state 与使用时机双重诊断

**文件:**

- `packages/rl-agent/muzero/train.py`
- `packages/rl-agent/sts2_env/combat_env.py`
- `mods/sts2-bridge/Scripts/BridgeGameApi.cs`
- `mods/sts2-bridge/Scripts/BridgeGameApi.PotionProfiles.cs`

**任务:**

- [ ] 区分两个问题：
  - [ ] potion timing policy：何时该用。
  - [ ] potion slot final-state：用了以后 bridge 是否仍显示未用。
- [ ] 在 episode end dump 中记录 potion slots 原始 payload。
- [ ] 每次 use_potion 后记录：
  - before slot
  - after slot
  - action success/failure
  - potion empty/null/is_usable/is_queued
- [ ] 如果 bridge 侧用了药水但 slot 未清，修 bridge 数据契约；如果 action 被拒绝，修 target / legal action。

**验收:**

- [ ] `potion_unused_on_death_rate` 不再和 `family_use_potion_rate` 出现无法解释的矛盾。
- [ ] `potion_selected_when_available_rate` 不能简单等价于“有就用”；需要结合 `use_quality` / `save_value`。

---

## 3. P1 — 重做 end_turn / strategic-defer / refund-no-followup 分类

> 目标: 让“空过”不再被 strategic-defer 或 no-followup 误洗白，同时保留真正合理的战术性不打牌。

### 3.1 end_turn 三分类目标态

**文件:**

- `packages/rl-agent/muzero/train.py`

**当前核心问题:**

现有逻辑大致依赖：

```python
wasteful = bool(end_turn_indices) and urgent_positive_count > 0 and (energy > 0.05 or zero_cost_urgent)
strategic_defer_available = bool(end_turn_indices) and positive_progress_count > 0 and urgent_positive_count == 0 and deferable_positive_count > 0
```

这会导致一个危险情况：只要一个正收益动作被归为 deferable，而 urgent 没被打上，就可能让 selected end_turn 逃过 bad/wasteful 统计。

**目标分类:**

```text
forced_end_turn:
  没有实际可执行收益动作。
  例如无能量、无 0 费收益牌、无药水收益、动画/抽牌暂态未稳定，或 legal actions 真的只有 end_turn。

strategic_defer_end_turn:
  有正收益动作，但明确有策略理由不打。
  必须具备显式 reason，例如：
  - exhaust_high_value_card: 当前可打但打出会消耗，且当前收益低、未来收益高。
  - retain_high_value_card: 保留牌当前打收益低，未来机制窗口收益高。
  - refund_no_followup: 回费牌打了也没有任何后续可用牌/可用药水/机制收益。
  - loop_control: 当前不打可让弃牌/抽牌循环进入更优状态。
  - one_card_lock_save_high_impact: Ceremonial 锁一张牌机制下，低影响牌不应浪费单次行动。

bad_end_turn:
  有必须处理的即时问题或高价值动作，却结束回合。
  例如：
  - incoming damage 高且有 block/weak/debuff。
  - 当前能斩杀或显著减伤。
  - 有 0 费强收益动作。
  - 有手牌强化/复制/降费/抽牌可以打开后续行动。
  - boss 机制窗口必须行动，例如 Kaiser 背刺风险、Ceremonial stun window。
  - 有高质量药水可救命/斩杀但不用。
```

**任务:**

- [ ] 新增 `EndTurnClass` 枚举或等价结构：`forced` / `strategic_defer` / `bad`。
- [ ] 新增 `end_turn_reason_flags`，每个 selected end_turn 必须写出 flags。
- [ ] 把原来的 `wasteful` 改为 `bad_end_turn` 的一个子集或兼容 alias。
- [ ] `strategic_defer` 必须依赖强 reason，而不是 `deferable_positive_count > 0` 这种宽条件。
- [ ] bad override 优先级必须高于 strategic defer。

**验收:**

- [ ] `bad_end_turn_selected_rate + strategic_defer_end_turn_selected_rate + forced_end_turn_selected_rate ≈ family_end_turn_rate`。
- [ ] 人工看到的空过应该进入 `bad_end_turn_selected_rate` 或 dump 中给出明确 forced/strategic reason。
- [ ] Insatiable 的 `strategic_skip_selected_rate` 不能再无解释上涨。

### 3.2 strategic skip 收窄

**文件:**

- `packages/rl-agent/muzero/train.py`
- `packages/rl-agent/sts2_env/card_effect_profile.py`

**任务:**

- [ ] strategic skip 不能只因为“牌是消耗牌/回费牌/特殊牌”就成立。
- [ ] 必须计算当前收益与未来机会成本：
  - 当前收益: damage/block/draw/energy/card mutation/mechanism effect。
  - 当前风险: exhaust 后失去未来窗口、虚无/消耗导致循环断裂、回费没有 followup。
  - 未来价值: draw/discard/exhaust 状态、保留状态、boss 机制窗口、遗物/药水/附魔触发。
- [ ] 对每一个 strategic skip 记录具体 reason：
  - `exhaust_low_immediate_high_future`
  - `refund_no_followup`
  - `x_cost_zero_energy_no_effect`
  - `retain_for_window`
  - `avoid_destroying_loop`
  - `avoid_bad_transform_or_exhaust_selection`

**验收:**

- [ ] `strategic_skip_selected_rate` 高时，top offenders 能列出具体卡牌和 reason。
- [ ] 如果 `strategic_skip_selected_rate` 上升但 win rate 下降，能定位是哪类 reason 误触发。

### 3.3 回费牌 no-followup 重算

**涉及卡牌类型:**

- 放血类：扣血/牺牲资源换能量。
- 预借时间类：获得短期能量/行动，但未来有代价。
- 生产制造类：生成卡牌或能量资源但可能消耗/改变牌堆。
- 任何 `energy_delta > 0` 或 `cost_reduction` 的卡。

**任务:**

- [ ] `refund_no_followup` 不能只看当前 hand 中剩余可打牌，还要看：
  - [ ] 打出该牌后是否会抽牌。
  - [ ] 是否生成新牌。
  - [ ] 是否降低其他牌费用。
  - [ ] 是否触发遗物产生牌/能量。
  - [ ] 是否触发附魔/重放。
  - [ ] 是否改变手牌状态，例如强化后才让后续牌有收益。
- [ ] 没有 followup 时，回费牌本身如果有 damage/block/debuff/机制处理，也不能简单视为白打。
- [ ] 对扣血回费牌计算 health cost，如果没有后续收益且当前不斩杀/不救命，应记为 bad action candidate。

**验收:**

- [ ] `refund_no_followup_selected_rate` 下降，且不牺牲真正需要回费爆发的 boss win。
- [ ] dump 能区分：
  - `refund_with_followup_good`
  - `refund_no_followup_bad`
  - `refund_with_intrinsic_value_ok`

---

## 4. P2 — Bridge 暂态合法动作与 fast step 设计

> 目标: 解决“抽卡/洗牌/打牌动画暂态只有 end_turn 暴露”的问题，但不能用长等待拖慢 combat sandbox。

### 4.1 禁止长固定等待

**原则:**

- [ ] 不允许在每一步 action 后追加固定长 sleep。
- [ ] 等待总耗时不能接近或超过 combat reset 等待时间。
- [ ] fast step 必须以状态判定为主，而不是时间判定为主。

**指标:**

```text
bridge_fast_step/wait_ms_mean
bridge_fast_step/wait_ms_p95
bridge_fast_step/stability_poll_count_mean
bridge_fast_step/transient_only_end_turn_count
bridge_fast_step/transient_only_end_turn_delayed_count
bridge_fast_step/transient_only_end_turn_leaked_count
bridge_fast_step/post_action_state_version_delta
```

**验收:**

- [ ] p95 wait 不超过明确阈值，例如 50-120ms，除非 reset/room transition。
- [ ] transient only-end-turn 泄漏率接近 0。
- [ ] 总训练 SPS 不能因为等待显著下降。

### 4.2 Bridge 层动作暴露稳定性条件

**文件:**

- `mods/sts2-bridge/Scripts/BridgeGameApi.EnvPayloads.cs`
- `mods/sts2-bridge/Scripts/BridgeGameApi.Env.cs`
- `mods/sts2-bridge/Scripts/BridgeGameApi.cs`
- `packages/rl-agent/sts2_env/combat_env.py`

**任务:**

- [ ] 找到 live game 中可以代表“动作处理未完成”的状态字段，例如：
  - action queue 是否为空。
  - card queue 是否为空。
  - animation / command 是否 pending。
  - hand/draw/discard/exhaust 是否正在变化。
  - input 是否 locked。
  - end turn button 是否只是 UI 暂态可见。
- [ ] 如果状态未稳定，不要把 `end_turn` 作为唯一合法动作暴露给 RL。
- [ ] 如果确实只有 end_turn 且状态稳定，暴露并标记 `forced_end_turn`。
- [ ] 每次 delayed end_turn 必须记录原因。

**验收:**

- [ ] 在“打出抽牌牌后”的短窗口，不会给模型一个只有 end_turn 的虚假决策。
- [ ] 在真实没牌可打时，不会卡死等待。
- [ ] combat sandbox fast step 仍保持高吞吐。

### 4.3 Python combat_env fast step 对齐

**文件:**

- `packages/rl-agent/sts2_env/combat_env.py`
- `packages/rl-agent/sts2_env/headless_sim_bridge_client.py`

**任务:**

- [ ] `step()` 后不要盲目 sleep。
- [ ] 使用 bridge 返回的 `state_version` / `state_hash` / stable-actionable flags 判断是否继续轮询。
- [ ] 如果连续 N 次发现 only-end-turn 但 state version 刚变化，做短轮询，而不是暴露给模型。
- [ ] N 和 timeout 必须很小，默认 fast path。

**验收:**

- [ ] 日志中能看到等待原因分布，而不是只有总 wait。
- [ ] `transient_only_end_turn_leaked_count` 长期为 0 或极低。

---

## 5. P3 — 卡牌状态变更 / 生命周期建模

> 目标: 让模型知道“当前合法可打”不等于“策略上应该打”。消耗、保留、附魔、虚无、重放、降费、变化、复制、手牌强化都会改变未来可用性。

### 5.1 建立 CardEffectProfile 目标字段

**文件:**

- `packages/rl-agent/sts2_env/card_effect_profile.py`
- `packages/rl-agent/tools/generate_card_effect_profiles.py`
- `mods/sts2-bridge/Scripts/BridgeGameApi.CardEffectProfiles.cs`
- `docs/card-mechanism-coverage-audit.md`

**任务:**

每张卡/药水/特殊动作至少导出以下结构化字段，不靠文本正则作为主路径：

```text
lifecycle:
  exhausts_on_play
  ethereal_or_void
  purge_or_remove_until_combat_end
  temporary_card
  retain
  self_retain
  discard_at_turn_end_override

cost:
  base_cost
  current_cost
  cost_for_turn_delta
  cost_for_combat_delta
  x_cost
  energy_gain
  energy_loss
  hp_cost
  future_energy_debt

hand_mutation:
  upgrade_one_in_hand
  upgrade_all_in_hand
  transform_one_in_hand
  transform_many_in_hand
  duplicate_card
  copy_to_hand
  create_card_in_hand
  discard_selected
  exhaust_selected
  retain_selected
  add_replay
  add_void
  add_enchant
  remove_enchant

pile_mutation:
  draw_cards
  discard_cards
  shuffle_discard_into_draw
  add_to_draw
  add_to_discard
  add_to_exhaust
  fetch_from_draw
  fetch_from_discard
  fetch_from_exhaust

combat_effect:
  damage
  block
  debuff
  buff
  weak
  vulnerable
  strength
  dexterity
  thorns
  multi_hit
  aoe
  target_required

mechanism_effect:
  changes_facing
  target_back_attack_side
  ceremonial_high_impact
  stun_window_exploit
  boss_specific_tags
```

**验收:**

- [ ] Ironclad 全部 88 张牌 + 无色牌覆盖率报告中，每个字段有来源：internal id / bridge profile / known source / fallback text。
- [ ] 文本正则只能作为 fallback，不能作为主来源。
- [ ] 未覆盖字段必须在 coverage CSV 中显示，不允许静默为 0。

### 5.2 Observation token 补齐

**文件:**

- `packages/rl-agent/sts2_env/observation_v3.py`
- `packages/rl-agent/muzero/train.py`
- `packages/rl-agent/muzero/README.zh-CN.md`

**任务:**

- [ ] hand / draw / discard / exhaust 每张卡都带 lifecycle + cost + mutation flags。
- [ ] 当前手牌 token 必须能看到：
  - [ ] 这张牌打出后去弃牌还是消耗堆。
  - [ ] 这张牌是否保留。
  - [ ] 这张牌是否有附魔/虚无/重放。
  - [ ] 当前费用是否被临时修改。
  - [ ] 是否是本回合生成的临时牌。
- [ ] pile token 必须能表达：
  - [ ] 抽牌堆剩余结构。
  - [ ] 弃牌堆可循环结构。
  - [ ] 消耗堆关键卡。
  - [ ] 即将洗牌的概率/状态。

**验收:**

- [ ] 出牌动作可以 cross-attend 到 hand/draw/discard/exhaust/potion/relic/enemy/power。
- [ ] transition aux 能预测打出一张消耗牌后，卡进入 exhaust 而不是 discard。
- [ ] 类似“武装先打强化手牌再打其他牌”的策略不需要写死，只能通过 hand mutation token + lookahead 学出来。

### 5.3 动作后果 token / transition aux

**文件:**

- `packages/rl-agent/muzero/train.py`
- `packages/rl-agent/sts2_env/observation_v3.py`

**任务:**

- [ ] 对每个 legal action 生成 `predicted_action_outcome_token`：
  - damage / block / draw / energy / hp cost
  - hand mutation
  - pile mutation
  - card lifecycle change
  - potion/relic trigger opportunity
  - boss mechanism impact
- [ ] world model aux heads 增加：
  - `next_hand_count`
  - `next_draw_count`
  - `next_discard_count`
  - `next_exhaust_count`
  - `next_energy`
  - `next_block`
  - `next_incoming_damage`
  - `next_boss_mechanic_state`
  - `card_moved_to_exhaust_prob`
  - `card_retained_prob`
  - `hand_mutation_applied_prob`

**验收:**

- [ ] 打出消耗牌 vs 非消耗牌，aux 能分清后续 pile state。
- [ ] 打出强化手牌牌后，next hand upgraded flags 可预测。
- [ ] 选择/消耗/变化/复制界面的 outcome 可预测。

---

## 6. P4 — 机制 boss 目标实现

### 6.1 Kaiser Crab — facing 与 back attack

**文件:**

- `packages/rl-agent/muzero/train.py`
- `packages/rl-agent/sts2_env/semantic_action.py`
- `packages/rl-agent/sts2_env/observation_v3.py`
- `packages/rl-agent/sts2_env/_sim_translate.py`
- `mods/sts2-bridge/Scripts/BridgeGameApi.EnvPayloads.cs`

**已知事实:**

- `enemy.side` 不是左右，而是敌我阵营，不能用它判断 left/right。
- 左右部位应从 enemy powers 中解析：
  - `BACK_ATTACK_LEFT_POWER` => left
  - `BACK_ATTACK_RIGHT_POWER` => right
- 转身不是“专门转身动作”，而是任何指向另一侧单位的卡牌或药水都可能改变 facing。

**任务:**

- [ ] action semantic 增加：
  - `target_back_attack_position`
  - `changes_facing`
  - `facing_before`
  - `facing_after_if_action`
  - `incoming_multiplier_before`
  - `incoming_multiplier_after_if_action`
- [ ] AOE / random target / potion / untargeted action 要明确：
  - [ ] 是否能改变 facing。
  - [ ] 不能判断时标记 unknown，而不是默认为 false。
- [ ] Kaiser 指标按 encounter 输出：
  - `kaiser_back_attack_risk_mean`
  - `kaiser_facing_change_candidate_count_mean`
  - `kaiser_facing_change_selected_rate`
  - `kaiser_risky_end_turn_selected_rate`
  - `kaiser_incoming_multiplier_before_mean`
  - `kaiser_incoming_multiplier_after_selected_mean`

**验收:**

- [ ] `kaiser_facing_change_candidate_count_mean` 不再因为字段语义错而长期 0。
- [ ] `kaiser_facing_change_selected_rate` 至少能解释：不选是因为 pressure kill 更优，还是完全没识别。
- [ ] `kaiser_risky_end_turn_selected_rate` 在 back_attack_risk=1 时下降。

### 6.2 Ceremonial Beast — one-card lock / stun window

**文件:**

- `packages/rl-agent/muzero/train.py`
- `packages/rl-agent/sts2_env/semantic_action.py`
- `packages/rl-agent/sts2_env/combat_env.py`

**任务:**

- [ ] 显式 token：
  - `ceremonial_one_card_lock_active`
  - `ceremonial_stun_window_active`
  - `ceremonial_actions_remaining_this_turn`
  - `ceremonial_best_single_action_score`
- [ ] 每个 legal action 增加：
  - `single_action_impact_score`
  - `wastes_one_card_lock`
  - `uses_stun_window`
- [ ] reward/value aux：
  - high-impact action selected under lock => small positive shaping。
  - low-impact action under lock when high-impact exists => offender + small penalty。

**验收:**

- [ ] `ceremonial_beast_boss` 不再长时间 0%。
- [ ] `ceremonial_high_impact_candidate_count_mean > 0` 时，`selected_rate` 应逐步升高。
- [ ] selected low-impact offender 能列出具体牌。

### 6.3 Insatiable — strategic skip / refund-no-followup 专项

**文件:**

- `packages/rl-agent/muzero/train.py`
- `packages/rl-agent/sts2_env/card_effect_profile.py`

**任务:**

- [ ] 为 `the_insatiable_boss` 单独输出：
  - `strategic_skip_selected_rate`
  - `refund_no_followup_selected_rate`
  - `bad_end_turn_selected_rate`
  - `positive_action_count_mean`
  - `mandatory_positive_action_count_mean`
  - `objective_q_mean`
  - `top_strategic_skip_cards`
- [ ] dump 每次 strategic skip 的 reason。
- [ ] 如果 skip 发生在 incoming 高、能量足、有 block/debuff 时，强制 bad override。

**验收:**

- [ ] Insatiable 胜率不再伴随 strategic skip 上涨而下滑。
- [ ] top offender 可以直接指导是哪个卡牌/标签过宽。

---

## 7. P5 — direct planner / Q-like lookahead 目标态

> 目标: 保留 search-free，但不能盲信漂移变大的 latent rollout。

### 7.1 drift-gated planner blend

**文件:**

- `packages/rl-agent/muzero/train.py`

**任务:**

- [ ] direct planner 排序从固定 blend 改成 drift-gated blend：

```text
score(action) =
  policy_prior
  + immediate_tactical_quality
  + mechanism_objective
  + gate(drift) * rollout_q
  + gate(drift) * objective_q
  - gate(drift) * risk_q
  - uncertainty_penalty
```

- [ ] 当 `latent_drift` 高时，降低 rollout Q 权重，提升 immediate quality / mandatory action / mechanism objective。
- [ ] 当 `legal_f1` 低时，也降低 rollout Q 权重。
- [ ] 当 boss mechanism active 时，机制 objective 不应被 rollout Q 完全压掉。

**验收:**

- [ ] `direct_rollout_latent_drift_mean` 升高时，`bad_end_turn_selected_rate` 不同步升高。
- [ ] `planner_objective_q_mae` 升高时，模型不因错误 Q 估值空过。
- [ ] TB 输出 `planner/drift_gate_mean`、`planner/effective_rollout_q_weight_mean`。

### 7.2 Q-like lookahead value 监督

**文件:**

- `packages/rl-agent/muzero/train.py`

**任务:**

- [ ] 给每个 candidate action 监督短期 outcome：
  - after-action hp delta
  - block delta
  - enemy hp delta
  - energy delta
  - hand/pile delta
  - mechanism risk delta
  - potion/relic resource delta
- [ ] Q-like value 不只学最终胜负，还学“这步动作后世界是否更好”。
- [ ] 把 future-world aux 和 planner score 的关系写清楚，避免 aux loss 收敛但 planner 仍漂。

**验收:**

- [ ] `future_world_aux_loss` 下降时，`planner_objective_q_mae` 也应下降或至少不恶化。
- [ ] `latent_drift` 不再单调上升。

---

## 8. P6 — 采样、课程与模仿学习

### 8.1 encounter-balanced replay scheduler

**文件:**

- `packages/rl-agent/muzero/train.py`
- `packages/rl-agent/combat_snapshot_dataset.py`

**当前问题:**

boss sample 已到约 82.5%，但 boss 没稳定提升，normal/elite 反而退化。

**任务:**

- [ ] tier target 初始建议：
  - boss: 50-60%
  - elite: 20-30%
  - normal/weak: 10-20%
- [ ] boss 内部按 encounter balanced，不让某些 boss 样本被压没。
- [ ] 如果某 encounter 连续退化，则提高该 encounter 采样，但不能挤掉全部 normal/elite。
- [ ] 加入 anti-forgetting replay：每 N 个 boss batch 插入基础攻防 batch。

**验收:**

- [ ] boss 提升不再伴随 normal/elite 大幅下降。
- [ ] `sample_rate/boss`、`sample_rate/elite`、`sample_rate/normal` 可见。
- [ ] 每个 encounter 的样本数与胜率一起显示。

### 8.2 temperature 与探索调度

**文件:**

- `packages/rl-agent/muzero/train.py`

**任务:**

- [ ] temperature 不应在策略未稳定时降到过低。
- [ ] 对低胜率 encounter 保持更高探索。
- [ ] 对已稳定 normal/elite 可以降低探索。
- [ ] selected offender 高时，提高同类场景探索而不是全局降温。

**验收:**

- [ ] `temperature` 与 per-encounter win/uncertainty 联动。
- [ ] 不再出现低温锁死 bad strategy。

### 8.3 人类手打模仿学习突破窗口

**文件:**

- `packages/rl-agent/skada_bc_dataset.py`
- `packages/rl-agent/skada_bc_train.py`
- `packages/rl-agent/muzero/train.py`
- 新增或复用人工轨迹导出工具。

**任务:**

- [ ] 建立 human demonstration 数据格式：
  - observation token snapshot
  - legal actions
  - selected action
  - optional reason tag
  - outcome reward
  - encounter / tier / turn / hp / energy
- [ ] 允许对关键 boss 录制手打轨迹：
  - Kaiser: 背刺风险转向、何时 pressure kill。
  - Ceremonial: one-card-lock 下选高影响动作。
  - Insatiable: 何时不能 strategic skip。
  - 基础战斗: 不空过、正确 block/damage 节奏。
- [ ] 加入 imitation loss：
  - policy CE on selected human action。
  - advantage-weighted imitation：赢的轨迹权重大，输的轨迹低或只学早期正确动作。
  - optional reason-head supervision。
- [ ] 与 RL 训练混合：
  - 初期高 BC ratio。
  - 模型恢复后降低 BC ratio。
  - per-encounter BC replay 防止机制 boss 失忆。

**验收:**

- [ ] 能从手打轨迹生成离线 batch。
- [ ] BC 训练后 top1 agreement / win rate 在目标 encounter 上提升。
- [ ] 不破坏已有基础攻防能力。

---

## 9. P7 — 测试、验证、部署

### 9.1 单元测试

**新增/更新测试:**

```text
packages/rl-agent/tests/test_end_turn_taxonomy.py
packages/rl-agent/tests/test_x_cost_dynamic_energy.py
packages/rl-agent/tests/test_card_lifecycle_profiles.py
packages/rl-agent/tests/test_potion_timing_contract.py
packages/rl-agent/tests/test_kaiser_facing_semantics.py
packages/rl-agent/tests/test_ceremonial_one_card_lock_semantics.py
packages/rl-agent/tests/test_bridge_transient_end_turn_guard.py
```

**验收:**

- [ ] fake obs 下，bad/forced/strategic end_turn 分类正确。
- [ ] 0 能量 X 费动作不会被误判为高收益。
- [ ] 消耗牌进入 exhaust 的 profile 正确。
- [ ] Kaiser 左右侧从 powers id 解析，不使用 enemy.side。
- [ ] transient only-end-turn 被 guard，不暴露给模型。

### 9.2 smoke test

**命令模板:**

```bash
cd /mnt/e/game/project/sts2_mcp/packages/rl-agent
venv/Scripts/python.exe -m pytest tests/test_end_turn_taxonomy.py -q
venv/Scripts/python.exe -m pytest tests/test_x_cost_dynamic_energy.py -q
venv/Scripts/python.exe -m pytest tests/test_kaiser_facing_semantics.py -q
venv/Scripts/python.exe -m muzero.train \
  --obs-mode token_v3 \
  --model-arch token_memory_v1 \
  --combat-sandbox \
  --combat-policy-mode direct \
  --total-timesteps 2000 \
  --batch-size 4 \
  --unroll-steps 2 \
  --n-envs 1 \
  --device cuda
```

**验收:**

- [ ] 2k smoke 无崩溃。
- [ ] diagnostics JSONL 正常生成。
- [ ] TB 出现新增 namespace。
- [ ] fast step wait 指标不过高。

### 9.3 中短跑验证

**阶段:**

1. 20k steps：验证指标出现、无显存爆、无吞吐崩。
2. 50k steps：看基础攻防和空过指标。
3. 100k+ steps：看 boss per-encounter 趋势。

**通过线:**

| 指标 | 目标 |
|---|---|
| `bad_end_turn_selected_rate` | 明显低于当前人工观察空过频率，并能解释所有 selected end_turn |
| `forced_end_turn_selected_rate` | 与实际无动作状态相符 |
| `strategic_defer_end_turn_selected_rate` | 有明确 reason，不无解释上涨 |
| `x_cost_zero_energy_selected_rate` | 接近 0，或 offender 全部合理 |
| `kaiser_facing_change_candidate_count_mean` | 非 0，且 dump 可解释 |
| `kaiser_risky_end_turn_selected_rate` | 下降 |
| `ceremonial_high_impact_selected_rate` | 上升 |
| `the_insatiable_boss/strategic_skip_selected_rate` | 不再高企或有收益解释 |
| `direct_rollout_latent_drift_mean` | 不再持续单调上升 |
| `normal_win/elite_win` | 不因 boss 训练继续下降 |

### 9.4 Bridge DLL 部署检查

**涉及文件:**

- `mods/sts2-bridge/Scripts/*.cs`
- 游戏目录: `E:\Program Files (x86)\Steam\steamapps\common\Slay the Spire 2`

**任务:**

- [ ] C# 改动后编译 bridge。
- [ ] 部署 DLL 到游戏 mod 目录。
- [ ] 启动 bridge health check。
- [ ] 用 live probe 检查：
  - player facing。
  - enemy back attack powers。
  - potion slot after use。
  - transient end_turn guard。

**验收:**

- [ ] `ResolvePlayerFacing` 输出 left/right/null 正确。
- [ ] `ComputeIncomingDamageMultiplier` 与 live 机制一致。
- [ ] use_potion 后 payload slot 正确 empty/null。
- [ ] legal actions 不在不稳定帧泄漏 only-end-turn。

---

## 10. 推荐落地顺序

### Phase A — 只加诊断，不改变策略行为

- [ ] A1. selected end_turn context dump。
- [ ] A2. action offender dump。
- [ ] A3. global/per-encounter 指标统一。
- [ ] A4. X 费动态诊断。
- [ ] A5. potion final-state dump。
- [ ] A6. 20k smoke，确认“人工观察的问题”能被日志抓到。

**为什么先做:** 现在最大风险是 detector 仍有盲区。先让问题可见，再改分类/奖励。

### Phase B — 修 end_turn taxonomy 与 action quality

- [ ] B1. forced / strategic_defer / bad 三分类。
- [ ] B2. strategic skip 收窄。
- [ ] B3. refund-no-followup 重算。
- [ ] B4. bad override 优先级高于 defer。
- [ ] B5. 50k 验证 Insatiable / 空过指标。

### Phase C — Bridge fast step 暂态修复

- [ ] C1. bridge 状态稳定 flags 确认。
- [ ] C2. only-end-turn transient guard。
- [ ] C3. Python fast step 短轮询，不用长 sleep。
- [ ] C4. 吞吐 + 泄漏率验证。

### Phase D — 卡牌生命周期与 action outcome token

- [ ] D1. CardEffectProfile 字段补齐。
- [ ] D2. observation token 补齐。
- [ ] D3. action outcome token。
- [ ] D4. future-world aux heads。
- [ ] D5. 全 Ironclad + colorless coverage 审计。

### Phase E — 机制 boss 专项

- [ ] E1. Kaiser facing action semantics。
- [ ] E2. Ceremonial one-card-lock high-impact objective。
- [ ] E3. Insatiable strategic skip offender + override。
- [ ] E4. per-encounter 中短跑验证。

### Phase F — direct planner 与课程调度

- [ ] F1. drift-gated planner blend。
- [ ] F2. Q-like lookahead 监督。
- [ ] F3. encounter-balanced replay scheduler。
- [ ] F4. temperature per-encounter 调度。
- [ ] F5. 100k+ 验证。

### Phase G — 人类模仿学习突破

- [ ] G1. 人类手打轨迹格式。
- [ ] G2. demo recorder/exporter。
- [ ] G3. BC/DAgger 混合训练。
- [ ] G4. 针对 Kaiser/Ceremonial/Insatiable 的 demo replay。

---

## 11. 任务依赖图

```text
A diagnostics
  ├─> B end_turn taxonomy
  ├─> B strategic/refund quality
  ├─> C transient end_turn guard
  └─> E boss offender analysis

D card lifecycle/action outcome
  ├─> B strategic/refund 精准化
  ├─> E boss mechanism action outcome
  └─> F Q-like lookahead supervision

C bridge fast step
  └─> 所有训练可信度

F planner/scheduler
  ├─> 依赖 A/B/D/E 指标
  └─> 支撑 search-free 目标态

G imitation learning
  └─> 在 A/B/D/E 可观测后接入，否则 demo 无法解释失败点
```

---

## 12. 每日检查面板

每次训练至少看以下几组，不再只看总 win：

### 12.1 全局健康

```text
256/win
256/boss_win
256/elite_win
256/normal_win
loss/total
loss/policy
loss/value
metric/predicted_legal_f1
direct_rollout_latent_drift_mean
planner_objective_q_mae
planner_risk_q_mae
sample_rate/boss
sample_rate/elite
sample_rate/normal
temperature
```

### 12.2 空过与费用

```text
boss_combat/family_end_turn_rate
boss_combat/forced_end_turn_selected_rate
boss_combat/strategic_defer_end_turn_selected_rate
boss_combat/bad_end_turn_selected_rate
boss_combat/wasteful_end_turn_available_rate
boss_combat/wasteful_end_turn_selected_rate
boss_combat/energy_mean
boss_combat/positive_action_count_mean
boss_combat/mandatory_positive_action_count_mean
boss_combat/x_cost_zero_energy_available_rate
boss_combat/x_cost_zero_energy_selected_rate
```

### 12.3 药水

```text
boss_combat/potion_available_rate
boss_combat/potion_selected_when_available_rate
boss_combat/potion_low_quality_selected_rate
boss_combat/potion_high_save_value_unused_rate
boss_combat/potion_unused_on_death_rate
boss_combat/potion_after_use_slot_not_empty_rate
```

### 12.4 机制 boss

```text
boss_combat/kaiser_crab_boss/win
boss_combat/kaiser_crab_boss/kaiser_back_attack_risk_mean
boss_combat/kaiser_crab_boss/kaiser_facing_change_candidate_count_mean
boss_combat/kaiser_crab_boss/kaiser_facing_change_selected_rate
boss_combat/kaiser_crab_boss/kaiser_risky_end_turn_selected_rate

boss_combat/ceremonial_beast_boss/win
boss_combat/ceremonial_beast_boss/ceremonial_one_card_lock_rate
boss_combat/ceremonial_beast_boss/ceremonial_high_impact_count_mean
boss_combat/ceremonial_beast_boss/ceremonial_high_impact_selected_rate
boss_combat/ceremonial_beast_boss/bad_end_turn_selected_rate

boss_combat/the_insatiable_boss/win
boss_combat/the_insatiable_boss/strategic_skip_selected_rate
boss_combat/the_insatiable_boss/refund_no_followup_selected_rate
boss_combat/the_insatiable_boss/bad_end_turn_selected_rate
```

---

## 13. 完成定义 Definition of Done

本轮目标态完成必须同时满足：

- [ ] 任何 selected end_turn 都能被解释为 forced / strategic / bad，并且 reason 可读。
- [ ] 人工观察到的空过能在 JSONL 中找到对应 offender，不再出现“日志全 0 但人眼看到”的矛盾。
- [ ] X 费牌使用基于当前能量动态计算，不再被初始 3 能量误导。
- [ ] 回费牌是否白打由后续可用行动/抽牌/生成/降费/遗物触发共同判断。
- [ ] 消耗/保留/虚无/重放/附魔/变化/复制/强化/降费进入 card profile、obs token、action outcome、transition aux。
- [ ] Kaiser 左右侧来自 powers id，转身由目标侧与当前 facing 判断，不再使用 enemy.side。
- [ ] Ceremonial one-card-lock 下高影响动作选择率可被监督和诊断。
- [ ] Direct planner 有 drift gate，不在 latent drift 高时盲信 rollout Q。
- [ ] Replay 采样不再 boss 过载导致 normal/elite 遗忘。
- [ ] Bridge fast step 不靠长 sleep，transient only-end-turn 不泄漏。
- [ ] 20k / 50k / 100k 三个验证阶段都有指标通过线。

---

## 14. 首批建议开工清单

如果现在立刻开始写代码，推荐第一批只做以下 8 个任务，避免同时改 reward/model/bridge 导致无法归因：

1. [ ] `train.py`：新增 selected end_turn context dump。
2. [ ] `train.py`：新增 action offender dump。
3. [ ] `train.py`：统一 global + per-encounter metric emit。
4. [ ] `train.py`：新增 forced/strategic/bad end_turn 分类，但先只 emit 不改变 action score。
5. [ ] `semantic_action.py` / `observation_v3.py`：X 费动态费用字段补齐。
6. [ ] `train.py`：Insatiable strategic/refund top offender 聚合。
7. [ ] `semantic_action.py`：Kaiser `target_back_attack_position` / `changes_facing` 从 powers id 解析。
8. [ ] smoke 20k：确认新指标和 JSONL 能抓住人工观察问题。

完成以上后，再进入 reward/score/model 结构改动。
