# MuZero Boss 机制恢复与训练解瓶颈执行计划（2026-05-03）

> 目的：把最新训练日志暴露出的 boss 机制瓶颈拆成可执行任务，避免继续靠长训碰运气。
> 适用范围：`packages/rl-agent` 的 MuZero planner / train / env / observation，STS2 bridge action semantic payload，diagnostics，targeted eval。
> 当前结论：**最新 checkpoint 可作为 warm-start 权重，但不能作为可用策略；不建议在当前 objective + 当前 buffer 上继续堆长训。**
> 推荐策略：完成 P0 修复后，新 run + fresh/reweighted buffer；只有 targeted boss/mechanism gate 过线后，再跑正式长训。

---

## 0. 最新训练状态摘要

### 0.1 最新 run

```text
LOG_DIR = E:\game\project\sts2_mcp\packages\rl-agent\logs_muzero\muzero_token_memory_combat_sandbox_2envs_20260503_082408
CKPT    = E:\game\project\sts2_mcp\packages\rl-agent\checkpoints_muzero\muzero_token_memory_combat_sandbox_2envs_20260503_082408\muzero_final
```

metadata 关键点：

```text
total_steps         = 501093
episode_count       = 16504
buffer_size         = 58707
replay_buffer_saved = true
mixed_precision     = bf16
amp_enabled         = true
amp_scaler_enabled  = false
```

### 0.2 质量判断

| 维度 | 当前现象 | 判断 |
|---|---:|---|
| loss 主体 | `loss/total` latest/mean20 约 11.x，policy/value/reward 正常 | 主体没有持续发散 |
| future-world/bank | 发生过 boss-heavy severe spike，`future_world_aux_loss` / `future_bank_state_loss` 同步爆 | 目标/观测 delta 仍有病态 batch |
| 普通/精英 | normal / elite 胜率尚可 | 不是纯执行崩坏 |
| boss | 256 boss win 明显弱；Kaiser / Ceremonial / Knowledge Demon 最新窗口为 0 | 当前瓶颈集中在 boss 机制决策 |
| planner | `root_top1_visit_share` 高，entropy 低；`search/root_bias_scale = 0` | 需要优先确认 root bias 是否真的影响最终选择 |
| 最后一局 | `episode/reward = -1456`，`episode/length = 1199` | 有极端长坏轨迹或 hang-like episode，需要单独追踪 |

### 0.3 关键机制诊断

| 机制 | 当前观测 | 结论 |
|---|---|---|
| 空过 / end_turn | legacy `wasteful_end_turn_rate = 0` 不可信；新 `true_wasteful_end_turn_selected_rate` 低但非零；`forced_end_turn` 大量存在 | transient leak 大概率已缓解，但 forced end_turn 需要把手牌 cost / playable reason dump 出来确认 |
| Kaiser facing | `kaiser_back_attack_risk_mean ≈ 1`；`facing_change_candidate_count > 0`；`facing_change_selected_rate` 低；Kaiser win 0 | 已不是“看不见机制”，而是 planner/value/policy 不把转身当高优先级 |
| Ceremonial | `missed_stun_window` / `low_impact_under_lock` offender 多，win 0 | 一卡锁 / stun window 的 action impact 排序没有学会 |
| Potion | 全局不是“有就必用”，但低质量用药和高保存价值未用都存在；slot after-use 状态仍有歧义 | 需要机会成本模型 + potion lifecycle，不是简单提高/降低用药率 |
| X-cost | 0 能量打 X 费低频但真实存在 | effective X 必须绑定当前 energy，不得被初始 3 费或 stale obs 误导 |
| HP-cost | immediate self-lethal 指标低，但 low-margin hp-cost 存在 | 需要 `hp_after_cost - incoming_after_block` 的生存边际，不只是 `hp-cost <= 0` |
| 消耗/保留/回费/选择类卡牌 | 合法可打不等于应该打；有些牌应保留或延迟 | wasteful / positive action classifier 必须引入 deferable / strategic skip 分类 |

---

## 1. 全局设计原则

### 1.1 不再让每个模块各猜字段

当前很多 bug 来自：

- train 侧读 action feature row；
- bridge 侧 semantic 字段不完整；
- detector 假设 `enemy.side = left/right`；
- X 费牌可能读到 stale energy；
- potion slot 是否已消耗不清楚；
- HP-cost / exhaust / retain / enchant 机制散落在文本或局部字段里。

必须收敛到一个统一的 action annotation 层：

```text
raw bridge obs + legal actions
  -> stable state validator
  -> ActionMechanicAnnotation builder
  -> MechanismContext builder
  -> ActionQualityScorer
  -> planner root bias / rollout value adjustment
  -> selected action
  -> diagnostics / replay priority / auxiliary targets
```

任何 critical 机制逻辑都优先读 typed internal fields；文本正则只能作为低置信 fallback diagnostics，不能驱动 hard mask / 高权重 reward。

### 1.2 合法可打 ≠ 应该打

以下情况必须从 “wasteful end_turn” 中剥离：

- 打出后会消耗，且当前收益不足的消耗牌；
- 打出后回费但没有后续可用动作的回费牌；
- X 费牌在 `current_energy = 0` 且无非 X side effect；
- 需要等 boss 窗口 / 下一回合 / 特定手牌组合的 setup 牌；
- 保留/附魔/复制/手牌选择类动作的机会成本高于当前收益；
- 仅 bridge 暂时未暴露手牌操作的 transient only-end_turn 窗口。

所以 “有 playable card” 不能直接推出 “结束回合 = 空过”。必须分成：

```text
urgent_positive_action     应立即执行的正收益动作
normal_positive_action     一般正收益动作
deferable_positive_action  合法但可合理延迟
strategic_skip_action      当前不打更合理
unsafe_action              自杀 / 机制灾难 / 低质量用药
forced_end_turn            没有任何合理动作
transient_only_end_turn    bridge 状态未稳定，不应暴露给 policy
```

### 1.3 Boss 战战损原则

用户已明确：**boss 战战损一般不重要，因为除 A10 外 boss 后会回满。**

因此 reward 不应强迫 boss 战用更低战损通过；但仍必须保留：

- 自杀 hard mask；
- 会导致本回合/敌方行动后死亡的 HP-cost 禁止或强惩罚；
- 会错过 boss 机制窗口的行动惩罚；
- A10 / 不回血特殊规则需要由 run context 显式区分。

换言之：

```text
normal/elite: 战损是核心目标之一
boss(non-A10): 胜利 > 机制正确 > 非致命战损
boss(A10 or no-full-heal): 胜利 + 战损都重要
```

### 1.4 所有修复都要带可证伪指标

每个机制修复必须同时新增：

- selected rate；
- available/candidate rate；
- bias applied rate；
- selected despite bias / missed opportunity；
- per-encounter namespace；
- JSONL offender dump。

否则训练到一半仍然无法判断是 observability、planner、reward 还是 policy 学习问题。

---

# P0 — 正式长训前必须完成

---

## P0-1. 确认并修复 root bias 真正进入最终 action selection

### 背景

最新日志中 metadata 显示 root bias enabled，但 TB 里：

```text
search/root_bias_scale = 0
search/root_objective_value = 0
```

同时 root top1 share 极高，policy 很集中。如果机制 bias 没真正改变最终 ranking，那么 Kaiser / Ceremonial / potion / X-cost 所有 action quality 逻辑都只是 diagnostics。

### 目标

确认 `_combat_action_quality_bias()` / root bias / rollout objective 是否实际改变：

- root priors；
- MCTS root score；
- direct rollout selected action；
- final executed action。

### 主要排查文件

- `packages/rl-agent/muzero/train.py`
- `packages/rl-agent/muzero/mcts.py`
- `packages/rl-agent/muzero/config*.py`
- `packages/rl-agent/sts2_env/combat_env.py`
- 任何写 `search/root_bias_scale` / `root_objective_value` 的位置。

### 实施任务

- [ ] 搜索 root bias 写入和读取链路：

```powershell
rg -n "root_bias|bias_scale|objective_value|combat_action_quality|quality_bias|root_objective|changed_top1" packages -S --glob "!**/logs_muzero/**"
```

- [ ] 在 action selection 前后记录每个 root action：

```python
{
  "action_id": str,
  "family": str,
  "prior_before": float,
  "value_before": float,
  "score_before": float,
  "quality_bias": float,
  "score_after": float,
  "rank_before": int,
  "rank_after": int,
  "selected_before": bool,
  "selected_after": bool,
  "mechanism_tags": [...]
}
```

- [ ] 如果现在只计算但没有加到最终 score，修复为至少作用在 root selection 排序上。
- [ ] 如果 `root_bias_scale` 被 config 置 0，明确配置项和默认值。
- [ ] 如果 drift gate / safety gate 把 bias 清零，新增指标说明原因。
- [ ] 如果 direct planner 和 MCTS path 不同，两个路径都必须接入。

### 新增指标

- [ ] `search/root_bias_nonzero_rate`
- [ ] `search/root_bias_abs_mean`
- [ ] `search/root_bias_max_abs`
- [ ] `search/root_bias_changed_top1_rate`
- [ ] `search/root_bias_selected_action_delta_mean`
- [ ] `search/root_bias_suppressed_by_gate_rate`
- [ ] `search/root_bias_scale_effective_mean`
- [ ] `boss_combat/root_bias_changed_top1_rate`
- [ ] `boss_combat/<encounter>/root_bias_changed_top1_rate`

### JSONL diagnostics

- [ ] 新增：`diagnostics/root_bias_decisions.jsonl`
- [ ] 只采样 boss / offender / changed_top1 / selected_bad_action，避免爆量。

### 验收标准

- [ ] targeted smoke 中 `root_bias_nonzero_rate > 0`。
- [ ] Kaiser/Ceremonial 场景中 `root_bias_changed_top1_rate > 0`。
- [ ] 出现机制候选时，JSONL 能展示 bias 如何改变 ranking。
- [ ] 如果 selected 仍错，能区分：bias 太小、value 抵消、legal action 标注错、candidate 不存在。

---

## P0-2. 建立统一 ActionMechanicAnnotation

### 背景

当前机制字段散落在：

- bridge legal action payload；
- obs_v3 action feature vector；
- train.py 的 `_combat_action_quality_bias()`；
- mcts.py 的 positive action / end_turn detector；
- env reward；
- diagnostics dump。

这会导致每加一个机制就有多个口径。需要一个中间层作为唯一事实来源。

### 推荐结构

新增 shared module，例如：

```text
packages/rl-agent/sts2_env/action_annotation.py
```

核心 pure function：

```python
def annotate_action(
    raw_obs: dict,
    action: dict,
    legal_actions: list[dict] | None = None,
    *,
    post_action_preview: dict | None = None,
) -> dict:
    ...
```

推荐输出：

```json
{
  "family": "play_card | use_potion | end_turn | choose | other",
  "action_id": "...",
  "card": {
    "id": "...",
    "title": "...",
    "cost_base": 1,
    "cost_current": 1,
    "is_x_cost": false,
    "effective_x": 0,
    "exhausts_on_play": false,
    "ethereal": false,
    "retain": false,
    "temporary": false,
    "upgraded": false,
    "replay_count_added": 0,
    "cost_reduction": 0,
    "refund_energy": 0,
    "requires_followup": false,
    "has_independent_value": false,
    "modifies_cards": false,
    "selection_effect": "none | choose_hand | transform | replace | retain | enchant | copy | discard | exhaust"
  },
  "potion": {
    "id": "...",
    "class": "damage | block | energy | draw | card_generation | power | debuff | heal | revive | special",
    "is_consumable": true,
    "is_auto_trigger": false,
    "requires_target": false,
    "requires_followup": false,
    "save_value": 0.0,
    "use_quality": 0.0,
    "waste_risk": 0.0
  },
  "target": {
    "combat_id": "...",
    "is_enemy": true,
    "enemy_position": "left | right | none | unknown",
    "changes_kaiser_facing": false,
    "target_intent_damage": 0.0,
    "target_is_lethal": false
  },
  "safety": {
    "hp_cost_unblockable": 0,
    "self_damage_blockable": 0,
    "max_hp_loss": 0,
    "hp_after_cost": 0,
    "incoming_after_action": 0,
    "survival_margin_after_enemy": 0,
    "self_lethal_now": false,
    "enemy_turn_lethal_after_action": false,
    "low_margin": false
  },
  "quality": {
    "urgent_positive": false,
    "normal_positive": false,
    "deferable_positive": false,
    "strategic_skip": false,
    "unsafe": false,
    "mechanism_answer": false,
    "mechanism_miss_if_not_selected": false,
    "reason_codes": []
  }
}
```

### 字段来源优先级

1. bridge typed runtime fields；
2. bridge effect preview / internal card object fields；
3. observation semantic fields；
4. curated card/potion metadata table；
5. text regex fallback，仅 diagnostics 标低置信。

### 必须覆盖的机制字段

- [ ] targeting：enemy combat id、是否敌人、是否 self、是否 AoE；
- [ ] Kaiser 位置：从 enemy powers 的 `BACK_ATTACK_LEFT_POWER` / `BACK_ATTACK_RIGHT_POWER` 推导；
- [ ] current facing：从 boss mechanism context / powers / bridge explicit field；
- [ ] X-cost：`current_energy` 下的 `effective_x`；
- [ ] HP-cost：unblockable / blockable / max HP / delayed；
- [ ] card lifecycle：exhaust、ethereal、retain、temporary、replay、void、enchant、copy、transform、replace、cost reduction；
- [ ] refund / cost manipulation：当前打出是否只在有 follow-up 时才有价值；
- [ ] selection/card-state actions：选择 X 张、保留、替换、复制、附魔；
- [ ] potion timing：lethal、prevent lethal、mechanism window、followup、save value、waste risk；
- [ ] strategic skip/defer：合法但当前不打更合理。

### 需要替换的旧口径

- [ ] mcts.py 中直接读 action feature row 判断 positive/end_turn；
- [ ] train.py 中各自解析 roles/block/damage/potion；
- [ ] env reward 中另写一套 potion/kaiser/hp-cost 判断；
- [ ] diagnostics 中不一致的 candidate 统计。

### 验收标准

- [ ] 同一个 action 在 planner、reward、diagnostics 中 reason_codes 一致。
- [ ] `kaiser_facing_change_candidate_count` 不再依赖 `enemy.side`。
- [ ] `positive_action_count` 能区分 urgent / deferable。
- [ ] replay dump 中可直接复现 annotation。

---

## P0-3. Kaiser Crab：从“看见风险”升级为“正确处理风险”

### 当前问题

最新指标显示：

```text
kaiser_back_attack_risk_mean              ≈ 0.99
kaiser_facing_change_candidate_count_mean > 0
kaiser_facing_change_selected_rate        很低
kaiser_pressure_selected_rate             偏高
kaiser_risky_end_turn_selected_rate       偏高
kaiser win rate                           0%
```

这说明 observability 和 candidate detection 基本不是主问题。现在主问题是：模型/规划器偏向 pressure，而不把 facing-change 当作高优先级机制动作。

### 正确转身定义

用户已明确：

> 转身 = 使用任何指向操作攻击那个方向的单位；卡牌或药水都可以转身。

实现不得用 `enemy.side` / `target.side` 当左右，因为 bridge 返回的 side 是敌我方 enum。左右来自 enemy powers：

```text
BACK_ATTACK_LEFT_POWER  -> left body part
BACK_ATTACK_RIGHT_POWER -> right body part
```

伪代码：

```python
def kaiser_enemy_position(enemy: dict) -> str | None:
    powers = enemy.get("powers") or []
    ids = {str(p.get("id") or p.get("power_id") or "") for p in powers}
    if "BACK_ATTACK_LEFT_POWER" in ids:
        return "left"
    if "BACK_ATTACK_RIGHT_POWER" in ids:
        return "right"
    return None

changes_facing = (
    action_is_targeted_card_or_potion
    and target_enemy_position in {"left", "right"}
    and target_enemy_position != current_player_facing
)
```

### Planner bias 规则

在 Kaiser high-risk 状态：

```text
if back_attack_risk high
and facing_change_candidate_count > 0
and selected action is not lethal
and selected action does not reduce risk
then non-facing pressure 应被强降分
```

推荐 action quality：

| 动作类型 | 条件 | quality bias |
|---|---|---:|
| facing-change + 指向高 intent 侧 | risk high，target intent ≥ 15 | 大正 |
| facing-change + 同时造成 lethal | lethal | 极大正 |
| defense / weak / block | 没有 facing candidate 或能防住 lethal | 中正 |
| pressure | 仅在 lethal 或显著降低风险时正向 | 条件正 |
| risky end_turn | risk high 且可转身/可防御 | 大负 |
| non-lethal mono-target pressure | risk high 且不转身 | 负 |

### Reward / target

- [ ] 保留 “打向对侧导致 facing 改变” 的稀疏奖励。
- [ ] 增加 “high-risk 下错过 facing candidate” 的 step-level 小惩罚。
- [ ] 不要给“保持正确 facing”做持续状态奖励，避免破坏“先削血、最后转身”的合理节奏。
- [ ] 对非 A10 boss，不要过度惩罚普通战损；但 Kaiser 背刺是机制失败，仍应额外惩罚。

### 指标

- [ ] `boss_combat/kaiser_crab_boss/kaiser_back_attack_risk_mean`
- [ ] `boss_combat/kaiser_crab_boss/kaiser_facing_change_candidate_count_mean`
- [ ] `boss_combat/kaiser_crab_boss/kaiser_facing_change_selected_rate`
- [ ] `boss_combat/kaiser_crab_boss/kaiser_pressure_selected_rate`
- [ ] `boss_combat/kaiser_crab_boss/kaiser_nonlethal_pressure_under_risk_rate`
- [ ] `boss_combat/kaiser_crab_boss/kaiser_risky_end_turn_selected_rate`
- [ ] `boss_combat/kaiser_crab_boss/kaiser_facing_bias_changed_top1_rate`
- [ ] `encounter/kaiser_crab_boss/win_rate_64`
- [ ] `encounter/kaiser_crab_boss/win_rate_256`

### Offender dump

`diagnostics/kaiser_facing_decisions.jsonl`：

```json
{
  "encounter": "kaiser_crab_boss",
  "risk": 1.0,
  "current_facing": "left",
  "enemy_positions": {"enemyA": "left", "enemyB": "right"},
  "legal_facing_candidates": [],
  "selected_action": {},
  "selected_changes_facing": false,
  "selected_lethal": false,
  "selected_pressure": true,
  "root_rank_before": 1,
  "root_rank_after": 4,
  "quality_bias": -0.8,
  "reason_codes": ["kaiser_nonlethal_pressure_under_back_attack_risk"]
}
```

### 验收标准

- [ ] targeted Kaiser eval 中 `facing_change_candidate_count_mean > 0`。
- [ ] `kaiser_facing_change_selected_rate > 20%` 作为第一阶段目标。
- [ ] `kaiser_risky_end_turn_selected_rate < 5%`。
- [ ] Kaiser 64-window win 从 0 提升到 ≥ 15%，再进入长训。

---

## P0-4. Ceremonial Beast：一卡锁 / stun window 的高影响动作排序

### 当前问题

Ceremonial 最新窗口 win 近似 0，并且：

```text
ceremonial_missed_stun_window        多
ceremonial_low_impact_under_lock     多
```

说明模型在机制窗口中没有选择高影响动作。

### 需要建模的上下文

- [ ] 是否处于 one-card lock；
- [ ] 是否存在 stun window；
- [ ] 当前回合最多/只应该打一张牌；
- [ ] 当前 action 是否本窗口最高 impact；
- [ ] 是否错过可造成 stun / break / interrupt 的动作；
- [ ] 是否低 impact 消耗了唯一行动机会。

### Action impact 评分

推荐为每个 action 计算：

```text
impact = lethal_score
       + stun_score
       + mechanism_progress_score
       + damage_to_threshold_score
       + survival_score
       + card_state_value
       - waste_cost
```

在 one-card lock 中，`impact` 不应只看 damage；还要看：

- 是否触发 stun；
- 是否阻止 boss 下一轮高伤；
- 是否保留关键牌；
- 是否浪费消耗牌；
- 是否使用药水打开 follow-up。

### Planner bias

| 场景 | 动作 | bias |
|---|---|---:|
| stun window 有可触发动作 | 触发 stun / break | 大正 |
| one-card lock | 当前最高 impact action | 中/大正 |
| one-card lock | low-impact 非机制动作 | 大负 |
| 有 lethal | lethal | 极大正 |
| 药水能制造 stun / lethal follow-up | potion | 条件正 |

### 指标

- [ ] `boss_combat/ceremonial_beast_boss/stun_window_available_rate`
- [ ] `boss_combat/ceremonial_beast_boss/stun_window_taken_rate`
- [ ] `boss_combat/ceremonial_beast_boss/missed_stun_window_rate`
- [ ] `boss_combat/ceremonial_beast_boss/one_card_lock_rate`
- [ ] `boss_combat/ceremonial_beast_boss/low_impact_under_lock_selected_rate`
- [ ] `boss_combat/ceremonial_beast_boss/highest_impact_selected_rate`
- [ ] `boss_combat/ceremonial_beast_boss/impact_bias_changed_top1_rate`

### Offender dump

`diagnostics/ceremonial_window_decisions.jsonl`：

```json
{
  "encounter": "ceremonial_beast_boss",
  "one_card_lock": true,
  "stun_window": true,
  "legal_actions_scored": [],
  "highest_impact_action": "...",
  "selected_action": "...",
  "selected_rank_by_impact": 4,
  "missed_stun": true,
  "reason_codes": ["low_impact_under_one_card_lock"]
}
```

### 验收标准

- [ ] `missed_stun_window_rate` 降低 ≥ 70%。
- [ ] `low_impact_under_lock_selected_rate` 降低 ≥ 70%。
- [ ] Ceremonial 64-window win ≥ 10%~15% 后再长训。

---

## P0-5. Potion：从“用不用”改为“契机 / 保存价值 / 生命周期”

### 当前问题

用户观测“有药基本就用”，但 aggregate use rate 不支持简单结论。更准确的问题是：

```text
低质量用药存在；
高保存价值药水可能没留到关键窗口；
某些 potion use 后 slot_after 仍非 empty，生命周期状态不清；
policy/value 没有学到药水机会成本。
```

### Potion 分类

不要只按 “potion available” 判断。每个药水需要分类：

| 类别 | 示例语义 | 合理契机 |
|---|---|---|
| lethal damage | 直接斩杀 | 能杀 / 接近杀 |
| prevent lethal | block / heal / weak / intangible | 否则会死或高风险 |
| energy | 回费 | 有明确 follow-up 可打 |
| draw | 抽牌 | 有能量 / 可形成机制答案 |
| card generation | 生成攻击/技能/能力 | 当前手牌缺答案，且生成后可用 |
| power / long-term | 力量、能力 | 长战、boss/elite、早期回合 |
| debuff | weak/vuln/poison 等 | 高 intent / lethal setup |
| revive / fairy | 自动救命 | 不应主动浪费 |
| special / chaos | 需要单独规则 | 按 metadata |

### 机会成本评分

```text
use_quality = lethal
            + prevent_lethal
            + mechanism_answer
            + high_value_followup
            + long_fight_scaling

waste_risk = low_urgency
           + no_followup
           + overkill
           + save_value_high
           + random_without_need
```

用药 reward / bias 原则：

- 高质量用药：小/中正；
- 低质量用药：中负；
- boss/elite 中长期药水：不能第一回合无脑丢；
- 能量/抽牌药水必须验证 follow-up；
- revive/fairy 类不要当普通 use_potion 学；
- lethal / prevent lethal / mechanism window 可以覆盖 save_value。

### Potion lifecycle 必须拆开

当前 `use_potion_transition` 后 slot 仍非空可能有多个原因，不能直接断定 bridge bug。需要记录：

```text
use_selected
execute_accepted
choice_surface_open
choice_resolved
effect_resolved
slot_cleared
still_usable_after_resolution
auto_trigger_potion
```

### 指标

- [ ] `boss_combat/potion_available_rate`
- [ ] `boss_combat/potion_selected_when_available_rate`
- [ ] `boss_combat/potion_use_quality_selected_mean`
- [ ] `boss_combat/potion_waste_risk_selected_mean`
- [ ] `boss_combat/potion_low_urgency_selected_rate`
- [ ] `boss_combat/potion_high_urgency_selected_rate`
- [ ] `boss_combat/potion_lethal_selected_rate`
- [ ] `boss_combat/potion_prevent_lethal_selected_rate`
- [ ] `boss_combat/potion_mechanism_selected_rate`
- [ ] `boss_combat/potion_high_save_value_unused_on_death_rate`
- [ ] `potion/lifecycle_execute_ok_slot_not_cleared_rate`
- [ ] `potion/lifecycle_choice_surface_rate`
- [ ] `potion/lifecycle_still_usable_after_resolution_rate`

### Offender dump

`diagnostics/potion_decisions.jsonl`：

```json
{
  "potion_id": "POTION.ENERGY_POTION",
  "potion_class": "energy",
  "hp": 30,
  "incoming": 0,
  "energy": 0,
  "followup_positive_count": 0,
  "mechanism_window": false,
  "lethal_available": false,
  "prevent_lethal": false,
  "save_value": 0.8,
  "use_quality": 0.0,
  "waste_risk": 0.6,
  "selected": true,
  "execute_accepted": true,
  "slot_cleared": false,
  "choice_surface_open": false,
  "reason_codes": ["energy_potion_no_followup", "high_save_value_low_urgency"]
}
```

### 验收标准

- [ ] `potion_low_urgency_selected_rate` 明显下降。
- [ ] `potion_high_urgency_selected_rate` / `mechanism_selected_rate` 上升。
- [ ] energy/draw potion 在 `followup_positive_count = 0` 时 selected rate 接近 0。
- [ ] lifecycle 能解释 slot_after 非空是 bug、choice 未完成、自动药水还是仍可用。

---

## P0-6. X-cost：effective X 必须动态绑定 current energy

### 当前问题

模型仍低频出现 0 费打 X 费牌。可能原因：

- action semantic 用了初始 3 费；
- hand/action feature 没随打牌后 energy 刷新；
- X 费牌的 `effective_x` 没进 annotation；
- 某些 X 费牌有非 X side effect，被误标为纯浪费或反过来。

### 规则

```python
effective_x = current_player_energy
bad_zero_x = (
    is_x_cost
    and effective_x <= 0
    and not has_non_x_side_effect
    and not is_free_due_to_special_rule
)
```

### 任务

- [ ] bridge action payload 中明确：

```json
{
  "cost": {
    "base": -1,
    "current": -1,
    "is_x_cost": true,
    "effective_x": 0,
    "energy_snapshot": 0,
    "source": "runtime_current_energy"
  }
}
```

- [ ] Python annotation 不再从旧 action feature row 推断 X 值。
- [ ] 每次 action 后 legal actions 必须基于最新 energy / hand 重新生成。
- [ ] X=0 但有 side effect 的牌要单独标 `has_non_x_side_effect=true`，不得一刀切。
- [ ] X 费 potion / card-generation 产生的 X 费牌也走同一逻辑。

### 指标

- [ ] `combat/x_cost_available_rate`
- [ ] `combat/x_cost_selected_rate`
- [ ] `combat/x_cost_zero_available_rate`
- [ ] `combat/x_cost_zero_bad_selected_rate`
- [ ] `combat/x_cost_zero_good_side_effect_selected_rate`
- [ ] `combat/x_cost_energy_mismatch_rate`
- [ ] `combat/action_energy_snapshot_stale_rate`

### 验收标准

- [ ] `x_cost_zero_bad_selected_rate < 0.1%`。
- [ ] diagnostics 能展示每次 X 费选择的 `current_energy` / `effective_x`。
- [ ] 打完一张牌后 action list 的 energy snapshot 不滞后。

---

## P0-7. HP-cost：从 immediate self-lethal 扩展到敌方行动后的生存边际

### 当前问题

`hp_cost_self_lethal_selected_rate = 0` 不足以证明安全。用户看到“扣血卡不看自身血条，偶尔自杀”，可能包括：

- 打出后当前 HP 仍 > 0，但敌方 intent 会杀；
- hp-cost 没有足够收益，不该在低血时打；
- 回费/抽牌类 hp-cost 没有 follow-up，白扣血；
- block 被错误用于抵消 unblockable hp loss。

### 生存边际

```python
hp_after_cost = current_hp - hp_cost_unblockable - max_hp_loss_effective
incoming_after_action = estimate_incoming_after_block_and_debuff(action)
survival_margin_after_enemy = hp_after_cost - incoming_after_action
```

分类：

| 类别 | 条件 | 处理 |
|---|---|---|
| immediate self lethal | `hp_after_cost <= 0` | hard mask |
| enemy-turn lethal | `survival_margin_after_enemy <= 0` 且 action 不 lethal / 不 prevent lethal | huge negative / mask 可配置 |
| low margin | margin 小于阈值 | soft negative |
| hp-cost no followup | 回费/抽牌后无后续收益 | negative / deferable |
| hp-cost enables lethal/prevent lethal | 能杀或防死 | allowed / positive |

### 指标

- [ ] `combat/hp_cost_self_lethal_available_rate`
- [ ] `combat/hp_cost_self_lethal_selected_rate`
- [ ] `combat/hp_cost_enemy_turn_lethal_selected_rate`
- [ ] `combat/hp_cost_low_margin_selected_rate`
- [ ] `combat/hp_cost_no_followup_selected_rate`
- [ ] `combat/hp_cost_followup_realized_rate`
- [ ] `death/self_hp_cost_lethal_count`
- [ ] `death/hp_cost_enabled_enemy_turn_death_count`

### 验收标准

- [ ] immediate self-lethal 永远不进入 legal mask。
- [ ] low-margin hp-cost selected rate 明显下降。
- [ ] 每个 hp-cost death 有 JSONL 可解释是 bug、估计误差还是策略失败。

---

## P0-8. End_turn / 空过：统一 classifier，避免 transient 和 strategic defer 混淆

### 当前问题

旧指标 `wasteful_end_turn_rate` 全程 0 曾经误导判断。新指标显示 bias 会应用，但 selected wasteful 低频存在。另一个问题是 forced_end_turn 中很多样本仍有 hand/energy，需要解释是：

- 手牌在动画期不可用；
- 全部不可打；
- 都是 strategic defer；
- action list stale；
- bridge 只暴露了 end_turn。

### 结束回合分类

每次 end_turn selected 前必须输出：

```text
forced_end_turn
transient_only_end_turn
true_wasteful_end_turn
strategic_defer_end_turn
unknown_end_turn
```

判定必须基于 annotation：

- urgent_positive_action_count；
- normal_positive_action_count；
- deferable_positive_action_count；
- unsafe_action_count；
- hand_count / playable_count；
- current_energy；
- bridge stable flags；
- draw/shuffle/animation/choice surface flags；
- legal action count。

### Bridge stable validator

不能靠长 sleep。设计原则：

```text
直接读状态 flag + 极短 short poll budget。
```

推荐 bridge payload 增加：

```json
{
  "stability": {
    "is_combat_actionable": true,
    "is_animating": false,
    "is_drawing_cards": false,
    "is_shuffling": false,
    "is_choice_open": false,
    "hand_version": 123,
    "legal_actions_version": 456,
    "only_end_turn_reason": "none | no_legal_actions | animation | drawing | shuffling | choice_pending | unknown"
  }
}
```

任务：

- [ ] 如果 `only_end_turn_reason` 是 animation/drawing/shuffling/choice_pending，不向 policy 暴露 end_turn，而是 env 内部短轮询等待。
- [ ] short poll 总 budget 必须远小于 combat reset 等待，例如 150~300ms 级，禁止累计到数秒级。
- [ ] 如果超时仍 only_end_turn，则暴露但标 `unknown_end_turn` 并 dump。

### 指标

- [ ] `combat/end_turn_selected_rate`
- [ ] `combat/true_wasteful_end_turn_selected_rate`
- [ ] `combat/strategic_defer_end_turn_selected_rate`
- [ ] `combat/forced_end_turn_selected_rate`
- [ ] `combat/transient_only_end_turn_available_rate`
- [ ] `combat/transient_leaked_selected_rate`
- [ ] `combat/only_end_turn_short_poll_count_mean`
- [ ] `combat/only_end_turn_short_poll_ms_mean`
- [ ] `combat/only_end_turn_short_poll_timeout_rate`
- [ ] `combat/forced_end_turn_with_hand_energy_rate`

### Offender dump

`diagnostics/end_turn_contexts.jsonl` 需要补充每张手牌：

```json
{
  "hand": [
    {
      "card_id": "...",
      "title": "...",
      "cost_current": 1,
      "is_playable": false,
      "unplayable_reason": "not_enough_energy | needs_target | animation | choice_pending | unknown",
      "annotation_quality": "deferable_positive",
      "reason_codes": ["exhaust_low_value", "refund_no_followup"]
    }
  ]
}
```

### 验收标准

- [ ] `transient_leaked_selected_rate = 0`。
- [ ] forced end_turn with hand/energy 的每个样本都有 reason。
- [ ] `true_wasteful_end_turn_selected_rate` 不是旧指标那样永远 0，而是可信地低。
- [ ] short poll 平均耗时不显著拖慢 training throughput。

---

## P0-9. 消耗 / 保留 / 复制 / 附魔 / 选择 / 回费类卡牌机制补齐

### 背景

用户明确指出：很多卡牌效果改变卡牌状态，例如：

- 当前牌添加重放；
- 虚无 / ethereal；
- 选择 X 张手牌替换；
- 保留一张卡；
- 保留手牌；
- 附魔；
- 复制；
- 降低耗能；
- 回费；
- 消耗。

这些不能用文本正则作为主逻辑，应从内部 id / card object / effect component / bridge typed fields 导出。

### 需要覆盖的 typed card mechanics

| 机制 | 字段 | 策略意义 |
|---|---|---|
| exhaust | `exhausts_on_play` | 合法但可能应保留，不应简单算 positive |
| ethereal / 虚无 | `ethereal` | 不打会失去，可能从 deferable 变 urgent |
| retain | `retain` / `retain_hand` | 可延迟价值高 |
| replay | `replay_count_added` | 影响未来 action count/value |
| enchant | `enchant_target`, `enchant_effect_id` | 改变牌长期价值 |
| transform/replace | `selection_effect=transform/replace` | 当前收益和未来 deck/hand 变化 |
| copy | `copy_count`, `copy_destination` | 增加未来资源 |
| cost reduction | `cost_reduction`, `duration` | 通常需要 follow-up 或长期价值 |
| refund energy | `refund_energy` | 没有 follow-up 时可能白打 |
| discard/exhaust selected | `selection_effect=discard/exhaust` | 可为正也可为负，取决于目标 |
| card generation | `creates_cards`, `created_card_pool` | 需要评估生成后是否可用 |

### Strategic defer 规则示例

```text
exhaust damage-only card:
  若当前不 lethal / 不解机制 / 不防死，可能是 deferable 或 skip。

refund card:
  若打出后没有 positive follow-up 且本身无独立收益，标 refund_no_followup。

retain card:
  若当前收益低且保留到机制窗口更好，标 strategic_skip。

ethereal card:
  若不打会消失，defer 成本高；需要更偏 urgent。

selection replace/transform:
  需要进入 choice flow 质量评估，不应只按 play_card 低伤害判断。
```

### 指标

- [ ] `combat/exhaust_deferable_available_rate`
- [ ] `combat/exhaust_low_value_selected_rate`
- [ ] `combat/ethereal_urgent_available_rate`
- [ ] `combat/ethereal_missed_rate`
- [ ] `combat/retain_strategic_skip_selected_rate`
- [ ] `combat/refund_action_available_rate`
- [ ] `combat/refund_no_followup_selected_rate`
- [ ] `combat/refund_followup_realized_rate`
- [ ] `combat/card_selection_action_rate`
- [ ] `combat/card_enchant_action_rate`
- [ ] `combat/card_state_mechanic_unknown_rate`

### 验收标准

- [ ] wasteful classifier 不再把所有 playable exhaust/refund 牌都算作必须打。
- [ ] 回费牌无 follow-up 的 bad selected rate 可见。
- [ ] 内部 typed fields 覆盖铁甲 88 张 + 无色牌的主要机制。
- [ ] `card_state_mechanic_unknown_rate` 在 curated coverage 中接近 0；未知项进入审计表。

---

## P0-10. future_world_aux / future_bank_state spike 根因定位与隔离

### 当前问题

最新 run 出现 boss-heavy spike：

```text
dominant = future_world_aux_loss
sample_tier: boss ≈ 0.8125
future_world_aux_loss ≈ 3061
future_bank_state_loss ≈ 3059
policy/value/reward 未同步爆
```

这说明 policy 主体还能训练，但 future target 或 token bank delta 在某些 boss batch 中病态。

### 任务

- [ ] 扩展 `diagnostics/loss_spikes.jsonl`，dump：

```json
{
  "encounter_id": "...",
  "boss_mechanics": {},
  "selected_actions": [],
  "prev_obs_summary": {},
  "next_obs_summary": {},
  "token_bank_before_stats": {},
  "token_bank_after_stats": {},
  "future_targets_minmax": {},
  "zone_deltas": {
    "hand": {},
    "draw": {},
    "discard": {},
    "exhaust": {},
    "limbo": {}
  },
  "potion_slot_deltas": [],
  "card_state_deltas": []
}
```

- [ ] 增加 spike quarantine：当 auxiliary loss 超过阈值时：
  - 可选择 skip aux update；
  - 或 clip aux target；
  - 但不要 skip policy/value/reward 全部更新，除非 total loss 会污染 optimizer。

- [ ] 增加 per-encounter spike count：
  - `loss_spike/kaiser_crab_boss_count`
  - `loss_spike/ceremonial_beast_boss_count`
  - `loss_spike/knowledge_demon_boss_count`

### 验收标准

- [ ] 下一次 spike 能离线复现到具体 encounter + action + card/potion/zone delta。
- [ ] spike 不会污染 optimizer momentum 或导致后续 median loss 上升。
- [ ] 如果 spike 来源于 observation schema，必须 bump replay schema 并丢弃旧 buffer。

---

# P1 — P0 后进入 targeted eval / curriculum

---

## P1-1. Per-encounter namespace 全量指标

### 背景

全 boss 聚合会被其他 boss 稀释。Kaiser / Ceremonial 的机制指标必须按 encounter 分桶。

### 统一命名

```text
boss_combat/<encounter_id>/<metric>
encounter/<encounter_id>/win_rate_64
encounter/<encounter_id>/win_rate_256
encounter/<encounter_id>/reward_mean_64
encounter/<encounter_id>/length_mean_64
```

### 必须覆盖 encounter

- [ ] `kaiser_crab_boss`
- [ ] `ceremonial_beast_boss`
- [ ] `knowledge_demon_boss`
- [ ] `the_kin_boss`
- [ ] `the_insatiable_boss`
- [ ] `phrog_parasite_elite`
- [ ] `knights_elite`
- [ ] `soul_nexus_elite`
- [ ] `construct_menagerie_normal`

### 验收标准

- [ ] 不再只依赖 `boss_combat/...` 全局均值判断机制。
- [ ] 每个 boss 的 candidate/selected/missed/offender 都能独立看。

---

## P1-2. Targeted boss/mechanism eval gate

### 背景

正式长训前必须先用短 eval 验证机制行为，否则会再次训练到中途才发现方向错。

### Eval suites

| Suite | 目的 | 样本建议 |
|---|---|---:|
| `eval_kaiser_facing` | facing risk + candidate + selected | 64~128 |
| `eval_ceremonial_window` | stun window / one-card lock | 64~128 |
| `eval_potion_timing` | low urgency / high urgency / save value | 128 |
| `eval_x_cost_dynamic_energy` | 0 energy X-cost / stale energy | 128 |
| `eval_hp_cost_safety` | immediate / enemy-turn lethal | 128 |
| `eval_end_turn_stability` | transient only-end_turn leak | 256 |
| `eval_card_state_mechanics` | exhaust/retain/refund/enchant/selection | 256 |

### Gate thresholds

进入 4h mixed boss 训练前：

- [ ] `transient_leaked_selected_rate = 0`
- [ ] `hp_cost_self_lethal_selected_rate = 0`
- [ ] `hp_cost_enemy_turn_lethal_selected_rate` 接近 0，offender 可解释
- [ ] `x_cost_zero_bad_selected_rate < 0.1%`
- [ ] `kaiser_facing_change_selected_rate > 20%` when candidate exists
- [ ] `kaiser_risky_end_turn_selected_rate < 5%`
- [ ] `ceremonial_missed_stun_window_rate` 下降 ≥ 70%
- [ ] `potion_low_urgency_selected_rate` 下降 ≥ 50%
- [ ] `root_bias_nonzero_rate > 0`
- [ ] `root_bias_changed_top1_rate > 0` in mechanism suites

进入正式 500k+ 长训前：

- [ ] Kaiser 64-window win ≥ 15%
- [ ] Ceremonial 64-window win ≥ 10%~15%
- [ ] boss 256-window win 不低于当前 baseline
- [ ] future-world/bank spike 有 quarantine 或 root cause dump

---

## P1-3. Replay / curriculum / imitation learning 训练策略

### 当前 checkpoint 怎么用

| 选择 | 建议 | 原因 |
|---|---|---|
| 从零训练 | 不优先 | 已有权重有普通/精英能力，完全丢弃成本高 |
| 从旧权重 + 旧 buffer 继续 | 不建议 | 旧 buffer 混有错误机制标签 / 旧 reward / 旧 schema |
| 从旧权重 + fresh buffer | 推荐 | 保留基本能力，避免旧错误样本污染 |
| 从旧权重 + reweighted curated buffer | 推荐 | 对 boss 机制样本增权，修复瓶颈 |
| schema 改动后加载旧 replay | 禁止或需迁移 | annotation/reward/action schema 变化会破坏 target 一致性 |

### Curriculum 建议

1. **Smoke 10~20min**：只验证 crash / schema / metrics。
2. **Targeted mechanism 30~60min**：Kaiser、Ceremonial、potion、X-cost、HP-cost。
3. **Mixed boss 2~4h**：确保 boss 指标不回退。
4. **Full mixed run 8~24h**：再看 256-window / boss win。
5. **正式长训**：只有 gate 过线才跑。

### 用户手打 imitation learning 可作为突破口

用户之前问“手打 AI 模仿学习是否能突破瓶颈”。建议作为 P1，不要取代 P0：

- [ ] 增加 human demonstration recorder：保存 raw_obs、legal_actions、chosen_action、post_obs、reason label 可选。
- [ ] 优先采集 boss 机制局面：Kaiser facing、Ceremonial stun、药水契机、HP-cost、X-cost。
- [ ] 用 behavior cloning warm-up 训练 policy head：
  - 不训练或低权重训练 value；
  - 加 action mask consistency；
  - 加 mechanism tag auxiliary target。
- [ ] 用 DAgger-lite：模型遇到 low-confidence / offender 局面时保存给用户手选。
- [ ] human data 权重不要过高，避免过拟合单一打法。

推荐数据格式：

```json
{
  "demo_id": "...",
  "encounter": "kaiser_crab_boss",
  "raw_obs": {},
  "legal_actions": [],
  "chosen_action_id": "...",
  "annotation": {},
  "human_reason_tags": ["turn_facing", "avoid_back_attack"],
  "post_obs": {}
}
```

---

# P2 — 机制覆盖审计与长期维护

---

## P2-1. 铁甲 88 张 + 无色牌机制覆盖表

### 目标

不要再靠训练中发现漏机制。生成并维护 coverage table：

```text
docs/generated/card_mechanic_coverage_YYYYMMDD.md
```

每张牌至少列：

- card id；
- title；
- cost base/current；
- type；
- target type；
- damage/block/magic/secondary values；
- exhaust/ethereal/retain/replay/void/enchant/copy/transform/replace；
- HP-cost / max HP cost；
- energy refund / cost reduction；
- selection flow；
- generated cards；
- current annotation coverage；
- unknown typed fields；
- fallback text regex used?。

### 验收标准

- [ ] ironclad 88 张主要机制 typed coverage = 100%。
- [ ] 无色牌 typed coverage 尽可能完整；unknown 列表可人工补。
- [ ] `card_state_mechanic_unknown_rate` 有基线并持续下降。

---

## P2-2. Potion 全量覆盖表

### 目标

统计所有药水并归类 timing model，避免“一种药水一个 bug”。

每个 potion 至少列：

- potion id；
- title；
- class；
- target requirement；
- consumable / auto-trigger / choice-surface；
- effect values；
- follow-up requirement；
- save_value baseline；
- good timing examples；
- bad timing examples；
- lifecycle expected behavior。

### 验收标准

- [ ] 所有 potion 都有 class。
- [ ] potion lifecycle bug 能按 potion id 聚合。
- [ ] low-quality potion offender 可回溯到具体 class rule。

---

# 3. 推荐落地顺序

## 3.1 一次性最小闭环

按以下顺序执行，避免修一半又训练：

1. **P0-1 root bias audit/fix**
   如果 bias 不进最终选择，后续机制 bias 都无效。

2. **P0-2 ActionMechanicAnnotation**
   把各模块口径统一，后面只接这一层。

3. **P0-8 end_turn classifier + stable validator dump**
   防止继续产生假空过样本。

4. **P0-6 X-cost + P0-7 HP-cost**
   这两类属于安全/明显错误，先挡住。

5. **P0-5 Potion timing/lifecycle**
   修“药水契机”而不是简单用药率。

6. **P0-3 Kaiser + P0-4 Ceremonial**
   针对当前 boss 0% 的主阻塞。

7. **P0-10 spike dump/quarantine**
   防止 boss-heavy OOD batch 再污染 future-world/bank。

8. **P1 targeted eval gates**
   过线后才开长训。

## 3.2 不建议拆开的项

以下项强相关，不建议分开长训验证：

- root bias audit 与 Kaiser/Ceremonial planner bias；
- ActionMechanicAnnotation 与 end_turn/positive classifier；
- potion timing 与 potion lifecycle；
- X-cost dynamic energy 与 action list refresh/stale detection；
- HP-cost immediate safety 与 enemy-turn survival margin。

---

# 4. 完成后的训练方案

## 4.1 新 run 启动原则

- [ ] 如果 action/observation/reward schema 变化：新 run + fresh buffer。
- [ ] 可以加载旧 checkpoint 权重作为 initialization。
- [ ] 不加载旧 replay buffer，除非有迁移脚本并写入 schema_version。
- [ ] 先 targeted eval，再 mixed boss，再 full run。

## 4.2 第一阶段观察指标

训练第 30~60 分钟必须看：

```text
search/root_bias_nonzero_rate
search/root_bias_changed_top1_rate
combat/transient_leaked_selected_rate
combat/x_cost_zero_bad_selected_rate
combat/hp_cost_self_lethal_selected_rate
boss_combat/kaiser_crab_boss/kaiser_facing_change_selected_rate
boss_combat/kaiser_crab_boss/kaiser_risky_end_turn_selected_rate
boss_combat/ceremonial_beast_boss/missed_stun_window_rate
boss_combat/potion_low_urgency_selected_rate
loss_spike/*
```

若这些指标没动，不要等 8 小时；直接停训查链路。

## 4.3 第一阶段预期

| 指标 | 预期 |
|---|---:|
| `root_bias_nonzero_rate` | > 0 |
| `root_bias_changed_top1_rate` | targeted suite > 0 |
| `transient_leaked_selected_rate` | 0 |
| `x_cost_zero_bad_selected_rate` | < 0.1% |
| `hp_cost_self_lethal_selected_rate` | 0 |
| `kaiser_facing_change_selected_rate` | > 20% when candidate exists |
| `kaiser_risky_end_turn_selected_rate` | < 5% |
| `ceremonial_missed_stun_window_rate` | -70% |
| `potion_low_urgency_selected_rate` | -50% |
| Kaiser 64 win | ≥ 15% first target |
| Ceremonial 64 win | ≥ 10%~15% first target |

---

# 5. Definition of Done

本计划视为完成，需要满足：

- [ ] 机制 annotation 成为 planner/reward/diagnostics 的统一入口。
- [ ] root bias 被证明能改变 root action ranking。
- [ ] Kaiser 风险可见、candidate 可见、selected 行为显著改善。
- [ ] Ceremonial window 行为显著改善。
- [ ] potion 低质量使用下降，高价值契机使用上升。
- [ ] X-cost 0 energy bad use 基本消失。
- [ ] HP-cost 自杀和敌方行动后死亡风险可解释且受控。
- [ ] end_turn 不再混淆 transient / forced / strategic defer / true wasteful。
- [ ] future-world/bank spike 有可定位 dump 或 quarantine。
- [ ] targeted eval gate 通过后再进行正式长训。

---

## 6. 给执行者的注意事项

- 不要只改 reward；如果 planner/root selection 不接 bias，短期行为不会变。
- 不要只看全局 boss_combat 均值；Kaiser/Ceremonial 必须 per-encounter。
- 不要把所有 playable card 都算 positive；消耗/保留/回费/选择类动作需要战略延迟分类。
- 不要把 potion use rate 当目标；目标是 use_quality 与 waste_risk。
- 不要用长 sleep 修 only-end_turn；必须用状态 flag + 短轮询。
- 不要用文本正则驱动 critical 机制；typed field 缺失时先补 bridge/metadata。
- 不要在 schema 变化后混旧 buffer 长训。
