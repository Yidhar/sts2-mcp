# MuZero Route / Deck / Long-horizon 改造方案审核与执行设计

日期：2026-05-08
范围：审核并细化 `Deck-quality 特征`、`Route heuristic`、`Long-horizon value head` 三项改造。
文档目的：给后续实现者提供可执行设计。本文**不是要求立即改代码**，而是规定正确的实施顺序、验收指标和风险边界。

---

## 0. 一句话结论

这套改造方向正确，但原方案不能一次性直接接进训练。

正确路线应该是：

```text
先验证字段和 schema，
再扩 deck/build 可观测性，
再基于现有 per-action route_summary 做 route heuristic dry-run，
确认 heuristic 与生存/胜率正相关后小权重接入 MCTS prior，
最后再加 long-horizon head，并且第一阶段只 train-only，不参与推理。
```

最重要的修正：

```text
Long-horizon value 是 state value，不是 action prior。
Route heuristic 必须是 per legal action score，不是 observation-level current path score。
Deck-quality 应扩展现有 run_memory/_build_profile，不要新建平行 deck source。
```

---

## 1. 背景问题

当前模型的主要瓶颈不像是单纯 combat 动作不会打，而是更偏向：

```text
1. 路线选择短视：弱牌组仍可能走高风险 elite / 怪堆路线。
2. 牌组质量感知不足：模型不知道自己当前 deck 是否有能力承受 elite / boss。
3. 长期生存目标弱：value 更容易学局部战斗收益，不一定学到“多活几层”。
4. route/build/combat 信号混杂：某个改动带来胜率变化后，难以归因。
```

因此，改造目标不是简单给模型加更多 feature，而是建立一套可诊断的长期决策链路：

```text
deck quality -> route risk/value estimate -> route action prior/value -> long-horizon survival supervision
```

但每一步都必须可观测、可关闭、可归因。

---

## 2. 当前仓库已有基础

下面这些是实现者必须先理解的现有链路。

### 2.1 Deck / build profile 已存在

相关文件：

```text
packages/rl-agent/sts2_env/run_memory.py
```

当前已有：

```python
RUN_MEMORY_DIM = 48
OBJECTIVE_CONTEXT_DIM = 16
```

当前 deck 来源不是 `run_memory.deck`，而是：

```python
obs["player"]["deck_cards"]
```

现有 `_build_profile(obs)` 已经计算了一批 build/deck profile：

```text
deck_size
frontload
block
draw
scaling
aoe
heal
curse_density
high_cost_density
zero_cost_density
x_cost_density
consistency
build_gap_risk
```

这些已经写进 `run_memory_vector`，例如：

```text
vector[14] = deck_size
vector[21] = frontload
vector[22] = block
vector[23] = draw
vector[24] = scaling
vector[25] = aoe
vector[26] = heal
vector[27] = curse_density
vector[28] = high_cost_density
vector[29] = zero_cost_density
vector[30] = x_cost_density
vector[46] = consistency
vector[47] = build_gap_risk
```

结论：

```text
Deck-quality v2 应扩展现有 _build_profile(obs)，不要新建 run_memory.deck。
```

---

### 2.2 Route summary 已存在

相关文件：

```text
packages/rl-agent/sts2_env/_sim_translate.py
packages/rl-agent/sts2_env/observation_common.py
```

当前系统已经对 map legal action 做 reachable subtree 摘要。每个 map action 已经有：

```text
action["route_summary"]
action["route_nodes"]
```

编码到 observation 后是：

```text
obs["route_summary"][action_index]
obs["route_nodes"][action_index]
```

现有 route summary 维度：

```python
ROUTE_SUMMARY_DIM = 20
```

slot 语义：

| index | 含义 |
|---:|---|
| 0 | reachable_node_count |
| 1 | max_depth |
| 2 | direct_child_count |
| 3 | forced_path_steps_before_branch |
| 4 | count_monster |
| 5 | count_elite |
| 6 | count_boss |
| 7 | count_event |
| 8 | count_question_mark |
| 9 | count_rest_site |
| 10 | count_shop |
| 11 | count_treasure |
| 12 | next_elite_steps |
| 13 | next_rest_steps |
| 14 | next_shop_steps |
| 15 | next_event_steps |
| 16 | next_question_mark_steps |
| 17 | next_treasure_steps |
| 18 | can_reach_rest_site_before_elite |
| 19 | can_reach_elite_then_rest_site |

结论：

```text
Route heuristic 第一版应利用现有 route_summary 做 per-action score。
不建议第一版重新实现完整 top-K path enumerator。
```

---

### 2.3 Route auxiliary target 已存在

相关文件：

```text
packages/rl-agent/sts2_env/aux_targets.py
```

已有函数：

```python
compute_route_targets(prev_obs, action, planner_context=None)
```

它已经基于：

```text
route_summary
hp
gold
objective_context
```

计算 route 相关 target，例如：

```text
safe route
elite route
rest value
shop value
event value
treasure value
branch/reach
aggregate
```

结论：

```text
Route heuristic 不是全新训练目标，而是补强现有 route_summary -> route target -> MCTS prior 的链路。
```

---

### 2.4 MCTS 已有 objective prior bias

相关文件：

```text
packages/rl-agent/muzero/sts2_env/mcts.py
```

已有：

```python
_objective_prior_bias(...)
```

它已经会读取：

```text
obs["actions"]
obs["semantic_actions"]
obs["objective_context"]
```

并对部分 map action 加 route/build objective bias。

当前不足：

```text
route bias 主要看 immediate point_type，尚未充分利用完整 route_summary。
```

结论：

```text
Route heuristic prior bias 的最小改动接入点就是 _objective_prior_bias()。
```

---

### 2.5 网络已有 domain / objective value 结构

相关文件：

```text
packages/rl-agent/muzero/sts2_env/muzero_model.py
packages/rl-agent/sts2_env/objective_heads.py
```

当前模型已经有：

```text
domain_gate
candidate_policy_heads
value_heads
value_component_heads
objective_value
```

objective heads 已经包括：

```text
survival
hp_preservation
build_progress
resource_efficiency
```

结论：

```text
Long-horizon head 应作为现有 value/objective system 的 auxiliary head，而不是孤立地强塞进 prior。
```

---

### 2.6 Replay buffer 固定 shape

相关文件：

```text
packages/rl-agent/muzero/sts2_env/muzero_buffer.py
```

固定 shape 包括：

```text
run_memory: (RUN_MEMORY_DIM,)
objective_context: (OBJECTIVE_DIM,)
route_summary: (MAX_ACTIONS, ROUTE_SUMMARY_DIM)
```

结论：

```text
只要改 RUN_MEMORY_DIM / ROUTE_SUMMARY_DIM / 新增 obs key，就会影响旧 replay buffer 和 checkpoint。
```

因此实现前必须明确：

```text
1. 是否丢弃旧 replay buffer；
2. 是否从新 checkpoint 开始；
3. 是否做 padding/migration；
4. checkpoint load 是否 strict=False；
5. tensorboard 指标是否带 schema/version 标记。
```

---

## 3. 原方案的主要问题

| 原方案点 | 问题 | 应改为 |
|---|---|---|
| 从 `run_memory.deck` 统计 deck quality | 当前真实 deck 来源不是这个 | 用 `obs["player"]["deck_cards"]`，扩展 `_build_profile(obs)` |
| Deck archetype 用 hard one-hot | 阈值错误会强误导模型 | 用连续 soft score，不要 hard 分类 |
| 新建 route planner 枚举 top-K paths | 第一版重复现有 route_summary | 先对每个 map legal action 的 route_summary 打分 |
| route observation 注入 current path score | 选择前没有 current_chosen | 改成 per-action score，选择后记录 selected vs best |
| long-horizon value blend 进 root prior | state scalar 对所有 action 相同，不能排序动作 | 先 train-only；后续接 value scalarization 或 successor value |
| bias 权重 0.3/0.5 起步 | 可能直接带崩路线习惯 | 先 0.0 dry-run，再 0.1/0.2/0.3 |
| 估计 train.py 30 行 + network.py 20 行 | 明显低估 | 实际还涉及 obs schema、replay、checkpoint、metrics、tests |

---

## 4. 总体实施原则

### 4.1 禁止猜字段

所有新增 feature 必须来自已核实字段。

禁止：

```text
根据名字猜 side、type、mechanic
用卡牌文本正则推断机制
默认某个字段一定存在
```

必须：

```text
字段缺失 safe-zero
记录 present_rate / metadata_hit_rate
有 fixture 测试
```

---

### 4.2 先观测，后干预

任何会改变 action 选择的逻辑，第一阶段都必须 dry-run。

流程：

```text
1. 计算 heuristic / target；
2. 记录它认为最优的动作；
3. 记录模型实际选择；
4. 观察两者差距和 episode 结果；
5. 确认正相关后，再启用小权重 bias。
```

---

### 4.3 每次只启用一个行为改动

不要同时启用：

```text
新 deck feature
route prior bias
long-horizon head
long-horizon planner blend
```

否则胜率变化无法归因。

建议每阶段至少观察：

```text
win_rate_64
win_rate_256
boss_win_rate_256
act1_boss_seen
act1_clear
route/elite_selected_when_weak_deck_rate
route/rest_selected_low_hp_rate
```

---

### 4.4 schema 改动必须显式版本化

如果扩维：

```text
RUN_MEMORY_DIM 48 -> 64/80
ROUTE_SUMMARY_DIM 20 -> 24
新增 deck_quality obs key
新增 route_heuristic obs key
```

必须同步处理：

```text
observation encoder
replay buffer fixed shapes
model input dim
checkpoint loading
unit tests
smoke tests
old buffer 清理策略
```

---

## 5. Phase 0：字段 / schema audit

### 5.1 目标

确认后续改造依赖的字段真实存在、稳定、覆盖率足够。

### 5.2 检查项

#### Deck 相关

```text
obs["player"]["deck_cards"] 是否存在
deck_cards 是否包含完整牌组
每张牌是否有 card_id / id / internal_id
是否能区分 upgrade
是否能命中 card metadata
是否有 cost / type / rarity / tags / effect profile
```

#### Route 相关

```text
map legal actions 是否都有 route_summary
route_summary 是否维度固定为 20
route_summary slots 是否符合 observation_common.py 定义
route_nodes 覆盖率是否正常
boss 节点 / elite / rest / shop 计数是否合理
```

#### Floor / episode metadata 相关

```text
当前 floor 是否能从每个 step obs 或 transition_state 提取
actual_max_floor 是否在 episode 结束后可靠
terminated / truncated 是否可靠
combat_sandbox 和 full_run 是否可区分
```

### 5.3 产物

建议产出一个 audit jsonl 或 TB 指标：

```text
deck_audit/present_rate
deck_audit/card_id_present_rate
deck_audit/metadata_hit_rate
deck_audit/upgrade_present_rate
route_audit/summary_present_rate
route_audit/summary_dim_ok_rate
route_audit/map_action_coverage_rate
floor_audit/current_floor_present_rate
floor_audit/episode_max_floor_present_rate
floor_audit/truncated_rate
```

### 5.4 验收标准

```text
deck_cards present_rate >= 0.98
card_id present_rate >= 0.98
metadata_hit_rate >= 0.95，最好 >= 0.99
route_summary map action coverage >= 0.98
current_floor present_rate >= 0.98
```

若不满足，不能进入 Phase 1/4。

---

## 6. Phase 1：Deck-quality v2

### 6.1 目标

让模型在 route/build 决策时知道：

```text
当前牌组攻击够不够
防御够不够
能不能快速轮转
有没有回费/抽牌引擎
有没有 scaling
有没有 curse/status 污染
能不能安全打 elite
是否需要 shop/remove/rest
```

### 6.2 修改位置

优先改：

```text
packages/rl-agent/sts2_env/run_memory.py
```

重点函数：

```text
_deck_cards(obs)
_build_profile(obs)
RunMemoryTracker
```

如果必须扩维，还要改：

```text
packages/rl-agent/sts2_env/observation_common.py
packages/rl-agent/muzero/sts2_env/muzero_buffer.py
packages/rl-agent/muzero/sts2_env/muzero_model.py
```

具体以实际代码为准。

### 6.3 推荐 feature 列表

#### 基础结构

| feature | 含义 | normalization |
|---|---|---|
| deck_size | 当前牌组大小 | clamp(deck_size / 40, 0, 1) |
| upgraded_ratio | 升级牌比例 | 0..1 |
| attack_density | attack 类卡比例 | 0..1 |
| skill_density | skill 类卡比例 | 0..1 |
| power_density | power 类卡比例 | 0..1 |
| curse_density | curse 比例 | 0..1 |
| status_density | status 比例 | 0..1 |

#### 费用与效率

| feature | 含义 | 注意 |
|---|---|---|
| avg_cost | 平均费用 | X 费单独处理，不要当普通 0 费 |
| avg_damage_per_energy | 每费攻击效率 | cost=0 要用安全分母 |
| avg_block_per_energy | 每费防御效率 | 同上 |
| high_cost_density | 高费牌比例 | 已有，可保留 |
| zero_cost_density | 0 费牌比例 | 已有，可保留 |
| x_cost_density | X 费牌比例 | 已有，可保留，并配合 X 费逻辑 |

#### 轮转 / 引擎

| feature | 含义 |
|---|---|
| draw_density | 抽牌/换牌卡比例 |
| expected_extra_draw_per_turn | 粗估额外抽牌能力 |
| energy_refund_density | 回费牌比例 |
| cost_reduce_density | 降费牌比例 |
| retain_density | 保留牌比例 |
| innate_density | 固有牌比例 |
| exhaust_density | 消耗牌比例 |
| ethereal_density | 虚无牌比例 |
| copy/replay_density | 复制/重放相关能力 |

#### scaling

| feature | 含义 |
|---|---|
| strength_scaling_density | 力量 scaling |
| dex_scaling_density | 敏捷 scaling |
| poison_scaling_density | 毒 scaling |
| block_scaling_density | 防御 scaling |
| power_scaling_density | power 型长期收益 |

#### 综合分数

建议用连续分数，不要 hard one-hot：

```text
frontload_score
block_score
scaling_score
draw_engine_score
energy_engine_score
pollution_score
consistency_score
elite_readiness_score
boss_readiness_score
```

### 6.4 特征来源要求

优先级：

```text
1. internal card_id / metadata
2. bridge semantic tags / roles
3. effect profile / preview
4. safe fallback zero
```

禁止第一版使用：

```text
卡牌描述文本 regex
中英文文本 contains
根据 display name 猜机制
```

### 6.5 X 费卡注意

X 费卡不能按固定 0 费或 3 费处理。

Deck profile 中建议：

```text
x_cost_density 单独暴露
x_cost_damage_potential 单独暴露
x_cost_block_potential 单独暴露
```

在 combat action 里仍需要动态根据当前 energy 判断。

### 6.6 验收指标

必须记录：

```text
deck_quality/present_rate
deck_quality/metadata_hit_rate
deck_quality/deck_size_mean
deck_quality/avg_cost
deck_quality/avg_damage_per_energy
deck_quality/avg_block_per_energy
deck_quality/draw_density
deck_quality/energy_refund_density
deck_quality/exhaust_density
deck_quality/ethereal_density
deck_quality/retain_density
deck_quality/innate_density
deck_quality/curse_density
deck_quality/status_density
deck_quality/elite_readiness_score
deck_quality/pollution_score
```

### 6.7 单元测试建议

构造 synthetic deck fixtures：

```text
纯 Strike deck -> frontload 高，block 低
纯 Defend deck -> block 高，frontload 低
带 Bash/高伤害 deck -> damage_per_energy 高
带 draw/energy refund deck -> draw/energy score 高
带 curse/status deck -> pollution 高
带 exhaust/ethereal/retain deck -> 对应 density 高
带 scaling power deck -> scaling 高
```

验收：

```text
所有 feature 非 NaN
所有 feature clamp 在预期范围
metadata 缺失时不 crash
空 deck 不 crash
```

---

## 7. Phase 2：Route heuristic dry-run

### 7.1 目标

不改变模型决策，只回答：

```text
基于当前 deck/hp/gold/route_summary，哪条路线更合理？
模型实际是否选择了合理路线？
heuristic 的高分路线是否真的带来更高 survival / win？
```

### 7.2 不建议第一版做 full path planner

原方案的：

```python
enumerate_candidate_paths(map_graph, current_node, policy=...)
```

可以作为第二版，但第一版不要做。因为现有系统已经有 per-action reachable subtree summary。

第一版应实现：

```python
score_route_action(route_summary, deck_quality, hp_ratio, gold, potion_count, act, floor) -> RouteScore
```

### 7.3 RouteScore 建议结构

```python
RouteScore = {
    "score": float,
    "unsafe_elite_penalty": float,
    "forced_elite_penalty": float,
    "rest_value": float,
    "shop_value": float,
    "event_value": float,
    "treasure_value": float,
    "branch_value": float,
    "boss_progress": float,
    "breakdown": {...},
}
```

### 7.4 评分公式建议

基础结构：

```text
score =
    + boss_progress
    + rest_value
    + shop_value
    + treasure_value
    + event_value
    + branch_value
    - unsafe_elite_penalty
    - forced_elite_penalty
    - no_rest_before_elite_penalty
    - low_hp_monster_chain_penalty
```

### 7.5 elite risk 动态计算

不要固定：

```text
-2.0 × n_elites
```

应根据 deck/hp/potion/rest 动态计算：

```text
elite_risk =
    + low_hp_factor
    + weak_frontload_factor
    + weak_block_factor
    + poor_rotation_factor
    + no_potion_factor
    + no_rest_before_elite_factor
    - strong_frontload_factor
    - strong_block_factor
    - has_rest_before_elite_factor
    - has_potion_factor
```

然后：

```text
unsafe_elite_penalty = count_elite * elite_risk
```

示例：

```text
高 HP + 高 frontload + 有 potion + elite 后有火堆：elite penalty 小。
低 HP + 防御差 + deck 污染 + elite 前没火堆：elite penalty 大。
```

### 7.6 rest value

```text
rest_value = count_rest_site * f(low_hp, important_upgrade_need, upcoming_elite_or_boss)
```

建议：

```text
低 HP -> rest value 高
即将 elite/boss -> rest value 高
满血且无关键升级 -> rest value 低
```

### 7.7 shop value

```text
shop_value = count_shop * f(gold, remove_need, potion_need, deck_gap)
```

建议：

```text
gold < 75 -> shop value 低
gold 高 -> shop value 高
curse/status 污染高 -> shop remove value 高
药水空且后面有 elite/boss -> shop potion value 高
牌组缺攻击/防御/scaling -> shop card value 高
```

### 7.8 event value

事件不要一律强正。

建议第一版小权重：

```text
event_value = 0.1 ~ 0.3 * event_count
```

低 HP 时事件风险可能更高，不能盲目正向。

### 7.9 branch value / boss progress

避免 heuristic 过度贪资源不推进。

应考虑：

```text
reachable_node_count
max_depth
direct_child_count
forced_path_steps_before_branch
count_boss
next_boss_steps
```

### 7.10 dry-run 指标

必须记录：

```text
route_heuristic/available_rate
route_heuristic/top1_selected_rate
route_heuristic/top2_selected_rate
route_heuristic/selected_score_mean
route_heuristic/best_score_mean
route_heuristic/best_minus_selected_mean
route_heuristic/unsafe_elite_selected_rate
route_heuristic/low_hp_elite_selected_rate
route_heuristic/forced_elite_selected_rate
route_heuristic/no_rest_before_elite_selected_rate
route_heuristic/rest_before_elite_available_rate
route_heuristic/rest_selected_low_hp_rate
route_heuristic/shop_with_gold_available_rate
route_heuristic/shop_selected_high_gold_rate
```

### 7.11 dry-run 验收

进入 Phase 3 前必须证明：

```text
1. heuristic score 非全 0；
2. best_minus_selected_mean 能揭示模型路线失误；
3. heuristic 高分路线与 survival / act1_boss_seen / act1_clear 正相关；
4. low_hp_elite_selected_rate 能被指标捕捉；
5. heuristic 没有明显过度保守，例如永远避 elite 导致发育不足。
```

---

## 8. Phase 3：Route heuristic prior bias

### 8.1 目标

在 route/map action 上给模型一个弱先验，但保留模型决策权。

### 8.2 接入位置

优先接入：

```text
packages/rl-agent/muzero/sts2_env/mcts.py
_objective_prior_bias(...)
```

### 8.3 接入逻辑

伪代码：

```python
if is_map_action(action_index):
    summary = obs["route_summary"][action_index]
    score = score_route_action(summary, deck_quality, hp_ratio, gold, potion_count, act, floor)
    norm_score = clamp(score.normalized, -1.0, 1.0)
    bias[action_index] += route_heuristic_bias * norm_score
```

### 8.4 flag 策略

默认关闭：

```text
--route-heuristic-bias 0.0
```

逐步启用：

```text
0.1 -> 0.2 -> 0.3
```

早期不建议：

```text
0.5
```

### 8.5 防崩限制

必须：

```text
只作用于 route/map domain
score clamp 到 [-1, 1]
final bias clamp
保留关闭开关
记录 bias_applied_rate
记录 selected action 的 raw score 和 final bias
```

### 8.6 启用后观察指标

重点看：

```text
win_rate_64
win_rate_256
boss_win_rate_256
act1_boss_seen
act1_clear
route_heuristic/top1_selected_rate
route_heuristic/best_minus_selected_mean
route_heuristic/low_hp_elite_selected_rate
route_heuristic/rest_selected_low_hp_rate
route_heuristic/shop_selected_high_gold_rate
```

如果出现：

```text
win_rate 下降
act1_boss_seen 下降
模型过度避战
elite_seen 过低导致发育不足
```

应立即降低或关闭 bias。

---

## 9. Phase 4：Long-horizon value head，train-only

### 9.1 目标

让 representation 学到：

```text
从当前 state 开始，预计还能活几层 / 能否到达关键 floor。
```

第一阶段只作为 auxiliary loss，不改变决策。

### 9.2 为什么不能直接加进 root prior

`remaining_floors_value(state)` 是 state-level scalar。

在 root 下所有 legal actions 共享同一个 root state，因此：

```python
prior[action_i] += same_remaining_floors_value
```

无法改变 action 排名。它不是 action-specific prior。

正确接入需要：

```text
1. train-only auxiliary representation；或
2. value scalarization；或
3. 对每个 candidate action 展开 successor，再比较 successor remaining_floors_value。
```

### 9.3 target 设计

基础 scalar target：

```python
remaining = max(0, actual_max_floor - current_floor)
target = clamp(remaining / max_span, 0.0, 1.0)
```

推荐 `max_span`：

```text
20 或 50，需按当前训练范围确定。
```

loss：

```text
Huber 优先于 MSE
```

### 9.4 bucket head 替代方案

更稳的做法是加 bucket heads：

```text
reach_floor_3
reach_floor_5
reach_floor_7
reach_floor_10
reach_floor_12
reach_floor_15
reach_floor_17
reach_floor_20
```

目标：

```python
target[k] = 1 if actual_max_floor >= k else 0
```

优点：

```text
比 scalar MSE 稳
更容易看 AUC / calibration
更适合判断“能否活到 Act1 boss / Act2”
```

建议第一版可以二选一：

```text
方案 A：只做 remaining_floors scalar；
方案 B：只做 reach bucket heads；
方案 C：两者都做，但 loss weight 很小。
```

更稳的是 B 或 A+B 小权重。

### 9.5 mask 规则

必须 mask：

```text
missing current_floor
missing actual_max_floor
truncated but not terminated
combat_sandbox episode
reset/interrupted episode
```

否则模型会学到错误 target。

### 9.6 loss weight

初始建议小权重：

```text
remaining_floors_loss_weight = 0.05 ~ 0.10
```

不要一开始给太大，避免压过 policy/value/reward 主 loss。

### 9.7 必须记录的指标

```text
loss/remaining_floors
long_horizon/target_valid_rate
long_horizon/truncated_skip_rate
long_horizon/combat_sandbox_mask_rate
long_horizon/pred_mean
long_horizon/target_mean
long_horizon/mae
long_horizon/corr_pred_target
long_horizon/reach_floor_5_auc
long_horizon/reach_floor_10_auc
long_horizon/reach_floor_15_auc
long_horizon/by_domain_route_mae
long_horizon/by_domain_build_mae
long_horizon/by_domain_combat_mae
```

### 9.8 接入 planner 前验收

必须满足：

```text
target_valid_rate 稳定
MAE 持续下降
corr_pred_target 明显为正
bucket AUC 高于随机
没有 loss spike
不影响 total loss 稳定性
```

否则不允许进入 Phase 5。

---

## 10. Phase 5：Long-horizon 接入 planner / value

### 10.1 目标

让长期生存预测影响 route/build 决策。

### 10.2 推荐接入优先级

优先级从低风险到高风险：

```text
1. 仅作为 diagnostic，不接入；
2. 混入 route/build domain value scalarization；
3. 混入 MCTS leaf/root value backup；
4. 对每个 candidate action 做 successor value 比较；
5. 不建议 raw state scalar 直接接 root prior。
```

### 10.3 successor-aware 方式

如果一定要影响 action prior/action score，应做 action-specific：

```text
for each candidate action:
    predict successor hidden/state
    predict successor remaining_floors_value
    action_score += long_horizon_weight * successor_remaining_value
```

这是正确的，但工程成本明显高于原方案估计。

### 10.4 启用方式

第一版权重建议：

```text
--long-horizon-planner-weight 0.0  # 默认
--long-horizon-planner-weight 0.05
--long-horizon-planner-weight 0.10
```

不要一开始大权重。

---

## 11. 具体任务清单

### Task 0.1：字段 audit

- [ ] 采样 full-run obs，统计 deck fields present_rate。
- [ ] 采样 map obs，统计 route_summary coverage。
- [ ] 采样 episode metadata，统计 current_floor / actual_max_floor / truncated。
- [ ] 输出 TB 或 jsonl。
- [ ] 不满足阈值时停止后续实现。

### Task 1.1：Deck-quality v2 helper

- [ ] 在 `run_memory.py` 中扩展或新增 helper。
- [ ] 输入只用 `obs["player"]["deck_cards"]` 和 verified metadata。
- [ ] 所有 feature safe-zero。
- [ ] 所有 feature clamp/normalize。
- [ ] 不使用文本 regex。

### Task 1.2：Deck-quality metrics

- [ ] 添加 present_rate。
- [ ] 添加 metadata_hit_rate。
- [ ] 添加各核心 feature mean/max。
- [ ] 添加异常计数：NaN、inf、missing metadata。

### Task 1.3：Deck-quality tests

- [ ] 空 deck。
- [ ] 缺 metadata。
- [ ] 纯攻击 deck。
- [ ] 纯防御 deck。
- [ ] draw/energy deck。
- [ ] curse/status deck。
- [ ] exhaust/ethereal/retain deck。

### Task 2.1：Route score helper

- [ ] 新增 `score_route_action(...)`。
- [ ] 输入使用 existing route_summary。
- [ ] 输出 score + breakdown。
- [ ] 不改 action selection。

### Task 2.2：Route dry-run metrics

- [ ] 记录 best_score。
- [ ] 记录 selected_score。
- [ ] 记录 best_minus_selected。
- [ ] 记录 top1/top2 selected rate。
- [ ] 记录 low_hp_elite_selected_rate。
- [ ] 记录 rest/shop 相关 rate。

### Task 2.3：Route score tests

- [ ] 低 HP + elite + 无 rest -> score 低。
- [ ] 低 HP + rest before elite -> score 较高。
- [ ] 高 gold + shop -> score 较高。
- [ ] 强 deck + elite -> penalty 降低。
- [ ] 弱 deck + elite -> penalty 增大。

### Task 3.1：Route prior bias

- [ ] 在 `_objective_prior_bias()` 接入。
- [ ] flag 默认 0.0。
- [ ] 只对 map action 生效。
- [ ] bias clamp。
- [ ] TB 记录 bias_applied_rate。

### Task 4.1：Long-horizon target builder

- [ ] 从每个 transition 提取 current_floor。
- [ ] episode 结束后回填 actual_max_floor。
- [ ] mask truncated / combat_sandbox / missing floor。
- [ ] target normalize。

### Task 4.2：Long-horizon head

- [ ] 新增 scalar 或 bucket head。
- [ ] 更新 forward output。
- [ ] 更新 loss。
- [ ] loss weight 小权重。
- [ ] train-only，不接 planner。

### Task 4.3：Long-horizon metrics/tests

- [ ] target_valid_rate。
- [ ] MAE / corr。
- [ ] bucket AUC。
- [ ] checkpoint load test。
- [ ] old buffer compatibility strategy。

### Task 5.1：Long-horizon planner integration

- [ ] 只在 Phase 4 验收通过后做。
- [ ] 默认关闭。
- [ ] 小权重。
- [ ] 优先接 value scalarization，不直接接 root prior。

---

## 12. 启动训练前 checklist

每次正式训练前确认：

```text
[ ] 是否改了 obs shape？如果是，旧 buffer 是否已清理？
[ ] 是否改了 model input/output？checkpoint load 是否测试？
[ ] route heuristic bias 是否默认 0.0？
[ ] long-horizon planner weight 是否默认 0.0？
[ ] dry-run 指标是否已经出现在 TB？
[ ] feature present_rate / metadata_hit_rate 是否正常？
[ ] 是否能跑 smoke episode？
[ ] 是否有 unit tests 覆盖空 deck / missing metadata / route edge case？
```

---

## 13. 训练观察指标

### 13.1 主指标

```text
episode/win_rate_64
episode/win_rate_256
episode/boss_win_rate_256
episode/act1_boss_seen
episode/act1_clear
episode/reward_avg20
episode/death_floor_avg20
```

### 13.2 Deck-quality 指标

```text
deck_quality/present_rate
deck_quality/metadata_hit_rate
deck_quality/elite_readiness_score
deck_quality/pollution_score
deck_quality/draw_density
deck_quality/energy_refund_density
deck_quality/avg_damage_per_energy
deck_quality/avg_block_per_energy
```

### 13.3 Route 指标

```text
route_heuristic/top1_selected_rate
route_heuristic/best_minus_selected_mean
route_heuristic/unsafe_elite_selected_rate
route_heuristic/low_hp_elite_selected_rate
route_heuristic/rest_selected_low_hp_rate
route_heuristic/shop_selected_high_gold_rate
```

### 13.4 Long-horizon 指标

```text
loss/remaining_floors
long_horizon/target_valid_rate
long_horizon/mae
long_horizon/corr_pred_target
long_horizon/reach_floor_10_auc
long_horizon/reach_floor_15_auc
```

---

## 14. 回滚策略

如果训练下降，需要按顺序回滚：

```text
1. 关闭 route_heuristic_bias，保留 dry-run metrics。
2. 关闭 long_horizon planner weight。
3. 保留 long_horizon train-only；若 loss spike，再关闭 long_horizon loss。
4. 如果 obs shape 改动导致不稳定，回退 schema 或从新 buffer 重训。
5. 如果 deck features 异常，先看 present_rate / metadata_hit_rate / NaN 指标。
```

任何时候都应能通过 flag 回到：

```text
只观测，不干预。
```

---

## 15. 最终建议

批准做：

```text
Deck-quality v2
Route heuristic dry-run
Route heuristic 小权重 bias
Long-horizon train-only head
```

不批准第一版直接做：

```text
完整重写 route planner
action 选择前的 current_path_score
long-horizon raw scalar 直接加 root prior
0.5 强 route bias
无 schema 计划地扩 obs/head
```

最终推荐执行顺序：

```text
Phase 0：字段 / schema audit
Phase 1：Deck-quality v2 + metrics + tests
Phase 2：Route heuristic dry-run
Phase 3：Route heuristic prior bias，小权重可关
Phase 4：Long-horizon head train-only
Phase 5：Long-horizon 稳定后接 value/planner
```

这能最大限度避免：

```text
训练到一半发现字段错
胜率下降无法归因
route 习惯被强 bias 带崩
long-horizon target 噪声污染主模型
旧 replay/checkpoint schema 不兼容
```

---

## 16. Full-run 数据分布偏移与训练设计

### 16.1 问题定义

Full run 的真实数据分布和当前 combat/boss sandbox 分布差异很大。

以 Act 1 为例，按当前模型水平，一个 full run 大致包含：

```text
weak combat:   2 ~ 3 场
normal combat: 5 ~ 8 场
elite combat:  0 ~ 2 场
boss combat:   0 ~ 1 场
event room:    若干
shop:          0 ~ 2 次
card reward:   约 8 ~ 10 次
```

进入 Act 2 / Act 3 后，样本结构继续变化：

```text
boss 总数固定最多 3 场
normal combat 数量远多于 boss / elite
route / reward / shop / rest 决策数量大量增加
Act 2/3 数据只有模型足够强时才会自然出现
```

因此，如果直接用自然 full-run 数据训练，会出现几个问题：

```text
1. normal combat 样本压倒 boss / elite / route / reward 样本；
2. Act 2/3 数据极少，模型难以学到后期策略；
3. boss 只有 3 场/局，boss 机制覆盖速度极慢；
4. reward/card-pick 决策很多，但 reward 结果延迟很长，credit assignment 很弱；
5. 当前模型越弱，越采不到后期数据，形成自举瓶颈；
6. combat sandbox 学到的策略和 full-run 真实牌组/血量/资源分布会发生偏移。
```

结论：

```text
不能只靠自然 full-run on-policy 数据。
必须使用“分层采样 + 场景重放 + full-run 校准 + 少量强先验”的训练设计。
```

---

### 16.2 训练目标拆分

Full-run 训练应拆成 5 类能力，而不是只看 win_rate：

```text
A. Combat execution：每场战斗内少战损获胜。
B. Boss/elite mechanics：处理稀有但高影响机制。
C. Reward/build planning：选卡、删卡、升级、买牌、买药。
D. Route planning：根据当前 deck/hp/gold/potion 选择路线。
E. Long-horizon survival：让 build/route/combat 的局部选择服务于整局通关。
```

每类能力的数据频率和监督方式不同，所以 replay / loss / curriculum 也应该不同。

---

### 16.3 Replay buffer 设计：不要一个池子混到底

建议把 replay 分成多个逻辑 bucket，即使物理上仍在同一个 buffer，也要记录 sample tags。

推荐 tags：

```text
phase:
  full_run
  combat_sandbox
  boss_sandbox
  elite_sandbox
  reward_sandbox
  route_sandbox
  human_demo

act:
  act1
  act2
  act3
  unknown

room_type:
  weak
  normal
  elite
  boss
  event
  shop
  rest
  reward
  map

encounter_id:
  concrete enemy/boss id

decision_type:
  combat_action
  end_turn
  potion
  card_reward
  shop_purchase
  card_remove
  smith/rest
  map_route
  event_choice
```

训练 batch 不应按自然频率采样，而应使用 stratified mixture。

---

### 16.4 推荐 batch mixture

第一版推荐每个训练 batch 的样本来源大致为：

```text
40% full_run natural distribution
20% normal/weak combat execution
15% boss/elite targeted sandbox
15% reward/build/route decisions
10% human/demo or high-quality trajectory replay
```

如果没有 human/demo，则暂时改成：

```text
40% full_run
25% combat execution
20% boss/elite targeted
15% reward/route/build
```

注意：这里的 full_run 仍然保留自然分布，用来防止 sandbox 过拟合；但稀有关键场景必须被 oversample。

---

### 16.5 Full-run 自然分布只用于校准，不应独占训练

Full-run 的价值：

```text
1. 提供真实 hp/gold/deck/potion 分布；
2. 提供真实 route -> reward -> combat 的长期耦合；
3. 校准 sandbox 是否偏离真实运行；
4. 产生真实 long-horizon target。
```

Full-run 的问题：

```text
1. 后期数据稀少；
2. boss/elite 样本太少；
3. 当前模型弱时数据质量差；
4. reward/build credit assignment 极弱。
```

所以 full-run 应作为“主分布校准源”，但不能作为唯一训练源。

---

### 16.6 Act coverage curriculum

需要显式保证 Act 1/2/3 覆盖，而不是等模型自然打到。

推荐阶段：

#### Stage A：Act 1 stabilization

目标：

```text
Act 1 boss_seen 稳定
Act 1 clear 稳定
low_hp_elite_selected_rate 下降
reward/build 不明显污染 deck
```

训练组成：

```text
full_run act1-heavy
act1 normal/elite/boss sandbox
act1 reward/route sandbox
```

#### Stage B：Act 2 injection

触发条件：

```text
Act 1 clear 达到稳定阈值，例如 40%~60%
```

加入：

```text
Act 2 seeded states
Act 2 normal/elite/boss sandbox
Act 2 reward/shop/rest decisions
```

这些 seeded states 可以来自：

```text
模型自己成功进入 Act 2 的 checkpoint
人工/规则生成的合理 Act 2 deck/hp/gold 状态
human demo trajectory
历史 high-quality run snapshot
```

#### Stage C：Act 3 injection

触发条件：

```text
Act 2 boss_seen / clear 有稳定样本
```

同理加入 Act 3 seeded states。

---

### 16.7 稀有场景必须 oversample

Full run 中 boss 只有最多 3 场，elite 也少，不能靠自然频率。

必须 oversample：

```text
boss fights by boss_id
elite fights by elite_id
low HP combat
potion timing decisions
card reward decisions with close alternatives
shop remove / buy potion / buy key card
rest vs smith
route fork with elite/rest/shop tradeoff
```

建议每个关键 encounter 至少有独立监控：

```text
per_encounter/win_rate
per_encounter/avg_hp_loss
per_encounter/potion_use_rate
per_encounter/mechanic_success_rate
per_encounter/wasteful_end_turn_rate
```

---

### 16.8 Reward/card-pick 训练要单独处理

Act 1 有约 8~10 次 reward 选卡。它们频率高，但反馈极延迟。

建议：

```text
1. reward decision 单独打 tag；
2. replay 中对 reward decision oversample；
3. 给 reward 选择增加 auxiliary target；
4. 使用 deck-quality delta 作为短期 shaping；
5. 最终仍用 full-run survival 校准。
```

Reward 选择的辅助目标可以包括：

```text
frontload_gap_improved
block_gap_improved
draw_engine_improved
energy_curve_improved
pollution_increased
curve_too_heavy
elite_readiness_delta
boss_readiness_delta
```

但要注意：

```text
这些只能作为辅助，不应强制替代最终胜率目标。
```

---

### 16.9 Route 训练要用 counterfactual / candidate scoring

Route 决策很少，但影响巨大。

每次 map fork 应记录：

```text
所有 legal route candidates 的 route_summary
模型选择的 route
heuristic best route
后续 N 层实际结果
是否死亡
是否进入 elite/boss/shop/rest
```

训练上应增加：

```text
route selected_score
route best_score
best_minus_selected
route outcome label
```

如果能做 offline counterfactual，可记录：

```text
同一 state 下未选择路线的 heuristic score
```

但不要伪造未走路线的真实 outcome。

---

### 16.10 Combat sandbox 与 full-run 的分布校准

Combat sandbox 的优势：

```text
样本快
可定向覆盖 boss/elite/机制
可测单场战损
```

Combat sandbox 的风险：

```text
初始牌组/hp/potion/手牌分布不等于 full-run
模型可能学到 sandbox 特有习惯
```

因此每个 sandbox scenario 应尽量从 full-run snapshot 派生：

```text
真实 deck
真实 relics
真实 hp/max_hp
真实 potion slots
真实 gold
真实 act/floor
真实升级状态
```

并记录分布差异：

```text
distribution_shift/deck_size_delta
distribution_shift/hp_ratio_delta
distribution_shift/relic_count_delta
distribution_shift/potion_count_delta
distribution_shift/card_upgrade_ratio_delta
distribution_shift/action_count_delta
```

---

### 16.11 Human demo / imitation 的位置

如果加入手打记录，建议不要让 imitation 直接长期压制 RL。

推荐用途：

```text
1. 冷启动 route/reward/shop/rest 策略；
2. 纠正药水时机；
3. 纠正 boss/elite 特殊机制；
4. 提供 Act 2/3 seeded states；
5. 提供高质量 trajectory 用于 prioritized replay。
```

训练方式：

```text
behavior cloning loss 小权重
只对 human_demo tagged samples 生效
随训练进展衰减
RL value/reward 仍为最终主导
```

建议权重：

```text
BC weight 初期 0.1~0.3
稳定后 decay 到 0.02~0.05
```

需要指标：

```text
imitation/demo_sample_rate
imitation/bc_loss
imitation/demo_action_agreement
imitation/demo_value_vs_policy_conflict
```

---

### 16.12 Prioritized replay 设计

优先采样不应只看 TD error，还应看机制稀缺性。

priority 可由以下组成：

```text
priority =
    td_error_priority
  + rare_room_bonus
  + rare_act_bonus
  + boss_or_elite_bonus
  + route_decision_bonus
  + reward_decision_bonus
  + low_hp_decision_bonus
  + mechanism_failure_bonus
```

关键 failure 包括：

```text
低 HP 打扣血牌
浪费药水/过早药水
低 HP 走 elite
有狂乱逃离但未在倒计时危险时打
Kaiser facing/back attack 处理失败
X 费 0 能量错误使用
空过/等待窗口误判
```

---

### 16.13 推荐训练配方

#### Phase FullRun-0：观测期

```text
route_heuristic_bias = 0
long_horizon_planner_weight = 0
收集 full_run + sandbox 混合数据
只记录分布和 dry-run 指标
```

目标：确认分布。

#### Phase FullRun-1：Act 1 稳定

Batch mixture：

```text
40% full_run act1
25% act1 combat sandbox
15% act1 boss/elite targeted
15% reward/route/shop/rest
5% demo/high-quality replay
```

目标：

```text
Act 1 boss_seen 提升
Act 1 clear 提升
normal/elite/boss 战损下降
```

#### Phase FullRun-2：Route/build 介入

启用：

```text
route_heuristic_bias = 0.1
Deck-quality v2 已上线
reward/build auxiliary 开小权重
```

目标：

```text
低 HP elite 下降
shop/rest 使用更合理
reward 选卡改善 deck gaps
```

#### Phase FullRun-3：Act 2 seeded injection

Batch mixture：

```text
35% full_run natural
20% act1/act2 combat sandbox
15% boss/elite targeted
20% route/reward/build
10% act2 seeded states / demo
```

目标：

```text
Act 2 数据不再完全依赖自然到达
模型开始学习 Act 2 分布
```

#### Phase FullRun-4：Long-horizon train-only

启用：

```text
remaining_floors / reach_floor buckets auxiliary loss
planner weight 仍为 0
```

目标：

```text
long_horizon MAE 下降
reach_floor AUC 上升
不引入 loss spike
```

#### Phase FullRun-5：Long-horizon 小权重接入

前提：Phase 4 指标过关。

启用：

```text
long_horizon_planner_weight = 0.05 ~ 0.10
```

优先接 value scalarization，不要接 raw root prior。

---

### 16.14 需要新增的分布指标

为了确认 full-run 分布偏移，必须新增：

```text
full_run_dist/act1_weak_count_mean
full_run_dist/act1_normal_count_mean
full_run_dist/act1_elite_count_mean
full_run_dist/act1_boss_count_mean
full_run_dist/act1_event_count_mean
full_run_dist/act1_shop_count_mean
full_run_dist/act1_card_reward_count_mean

full_run_dist/act2_seen_rate
full_run_dist/act2_normal_count_mean
full_run_dist/act2_elite_count_mean
full_run_dist/act2_boss_seen_rate

full_run_dist/act3_seen_rate
full_run_dist/act3_normal_count_mean
full_run_dist/act3_elite_count_mean
full_run_dist/act3_boss_seen_rate

sample_mix/full_run_rate
sample_mix/combat_sandbox_rate
sample_mix/boss_sandbox_rate
sample_mix/reward_route_rate
sample_mix/demo_rate

replay_coverage/by_act_act1_rate
replay_coverage/by_act_act2_rate
replay_coverage/by_act_act3_rate
replay_coverage/by_room_normal_rate
replay_coverage/by_room_elite_rate
replay_coverage/by_room_boss_rate
replay_coverage/by_decision_reward_rate
replay_coverage/by_decision_route_rate
```

---

### 16.15 最终原则

Full-run 训练不应追求“完全按自然频率采样”。

自然频率会导致：

```text
normal combat 过多
boss/elite/route/reward 信号过少
Act 2/3 自举困难
```

推荐原则：

```text
full_run 用来校准真实分布，
sandbox 用来覆盖稀有机制，
reward/route bucket 用来解决长期 credit assignment，
demo/high-quality replay 用来突破早期策略瓶颈，
prioritized replay 用来反复纠正高价值失败。
```

也就是：

```text
自然分布负责真实，
分层采样负责覆盖，
辅助目标负责 credit assignment，
full-run win_rate 负责最终裁判。
```
