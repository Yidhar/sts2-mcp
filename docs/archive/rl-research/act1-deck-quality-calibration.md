# Act1 Deck Quality Calibration — card reward / combo option / death-deck diagnostics

更新时间：2026-05-14

目标：把 full-run Act1 的构筑质量纳入训练诊断与轻量运行时保护，避免模型反复跳过奖励、死亡时卡组过薄/攻防质量不足；同时避免把“组合组件卡”误判成纯垃圾或纯强卡。

> 本阶段不改 observation schema，不混旧 replay，不强行改模型输入。
> 当前实现只做 diagnostics + card reward anti-skip guard + TensorBoard 指标。

---

## 1. 核心判断：一张卡不是只有“强/弱”两个状态

奖励选卡要拆成五个分量：

```text
card_value
  = immediate_standalone_value      # 当前单卡立刻能贡献多少
  + current_deck_deficit_fill       # 是否补当前卡组缺口：攻击/防御/过牌/缩费/成长
  + combo_component_option_value    # 是否是未来体系组件，有凑齐后的上限
  - unmet_dependency_risk           # 当前缺依赖，未来凑不齐会变成低收益/污染
  - energy_curve_opportunity_cost   # 费用、手牌占位、抽到却打不出去的代价
```

用户指出的情况对应 `combo_component_option_value` 和 `unmet_dependency_risk` 同时存在：

- 组件已被当前卡组支持：高价值卡。
- 组件暂时没被支持，但 Act1 还早：小幅期权价值，因为未来奖励/商店可能补齐。
- 组件暂时没被支持且 Act1 已晚：风险升高，除非它本身也是强单卡，否则不该强拿。
- 组件永远没凑齐：死亡回放里应表现为 orphan combo components。

因此不能用单一静态 tier 决定奖励卡，也不能只看当前一回合伤害。

---

## 2. 当前实现落点

### 2.1 `sts2_env/deck_quality.py`

新增/强化的 deck quality 维度：

| 维度 | 指标 | 用途 |
|---|---|---|
| 可打出质量 | `raw_expected_playable_cards_per_turn` | 典型回合在能量预算下能实际打出几张有用牌 |
| 可打出攻击 | `raw_expected_playable_attack_damage_per_turn` | 不是牌面攻击总量，而是能量约束下的可兑现伤害 |
| 可打出防御 | `raw_expected_playable_block_per_turn` | 不是牌面防御总量，而是能量约束下的可兑现格挡 |
| 能量利用率 | `expected_energy_utilization_score` | 是否有牌可打/能量是否被转化成有效行动 |
| 费用曲线 | `cost_curve_zero/one/two/three_plus/x_share` | 识别高费臃肿、0费/1费流畅度、X-cost 占比 |
| 组合组件密度 | `combo_component_density` | 卡组里有多少组合相关组件 |
| 组合启动器 | `combo_enabler_density` | 过牌、缩费、保留、消耗、弃牌等体系启动件 |
| 组合收益件 | `combo_payoff_density` | 格挡转伤害、消耗/弃牌收益、成长收益等 payoff |
| 组合期权价值 | `combo_option_value_score` | 当前 deck 对组合体系的支持程度/潜在上限 |
| 未满足依赖风险 | `combo_unmet_dependency_score` | payoffs/enablers 不平衡导致组件孤立的风险 |
| 成长期权 | `scaling_option_value_score` | 面向 boss/长战的成长潜力 |
| 跨回合收益密度 | `delayed_payoff_density` | power、成长、格挡转伤害、消耗/弃牌收益等需要未来回合兑现的 payoff 占比 |
| 跨回合启动件密度 | `delayed_enabler_density` | 过牌、回费、保留、消耗、弃牌、复制等能让 payoff 更快兑现的启动件占比 |
| 兑现速度 | `delayed_payoff_time_to_value_score` | 当前卡组是否能尽快抽到/打出/支持慢收益牌 |
| 成熟度 | `delayed_payoff_maturity_score` | 慢收益牌是否已经被当前卡组的费用、过牌、防御、成长环境支持 |
| 期权价值 | `delayed_payoff_option_value_score` | “未来可能凑齐体系”的小幅正价值 |
| 未兑现风险 | `delayed_payoff_unrealized_risk_score` | 拿了慢收益/组件但启动件不足，死亡前没兑现的风险 |

特殊修正：

- `CARD.BODY_SLAM` / 全身撞击：静态 damage 可能是 0，但它是 `block_payoff + combo_payoff`。
- `CARD.ENTRENCH` / 巩固、`CARD.BARRICADE` / 壁垒：作为 block-scaling / enabler 处理。

这解决了“全身撞击没有静态伤害所以像空白卡”的 deck-quality 盲点。

### 2.2 跨回合收益 / delayed payoff

用户指出的关键点是：有些卡不是当前回合兑现，而是在未来多个回合后才体现价值。典型例子包括：

- power / strength / dex / poison scaling；
- 全身撞击这类 block-to-damage payoff；
- 消耗、弃牌、复制、保留相关组件；
- 长 boss 战中才明显超过普通前期牌的成长卡。

因此一张“慢牌/组件牌”必须拆成两个方向：

```text
delayed payoff option = 当前还早，未来奖励/商店/路线可能补齐体系
unrealized risk       = 当前依赖不足，抽到也打不出收益，死亡时会变成污染
```

Act1 early 可以给小期权，但 late Act1 / boss 前更重视 maturity：

- `delayed_payoff_maturity_score` 高：说明 deck 已经有过牌、能量、防御或 partner，慢牌更可能兑现。
- `delayed_payoff_unrealized_risk_score` 高：说明 payoff/enabler 不平衡，死亡时可能看到“拿了体系件但没成体系”。
- death deck 如果 `delayed_payoff_density` 高、`unrealized_risk` 高、`maturity` 低，下一步要查奖励选择是否拿了过多孤立慢牌。

---

## 3. Card reward guard 的原则

文件：`muzero/training/card_reward_guard.py`

### 3.1 只在病态跳过时介入

guard 只处理：

- 当前 surface 是 card reward；
- 模型选择 skip/proceed/leave；
- Act1 或未知但早期卡组；
- 当前 deck 存在明显缺口：薄、低攻击、低防御、低过牌、低成长；
- 候选奖励中有非 junk 且分数超过阈值的卡。

它不会：

- 改 replay schema；
- 改 observation tensor；
- 在健康大卡组里强制拿牌；
- 强制拿 curse/status/junk；
- 代替模型做完整构筑搜索。

### 3.2 组合组件的三段式处理

对 Body Slam 这类牌，guard 现在区分：

| 场景 | 行为 |
|---|---|
| 有足够格挡支持 | 认为是可用 payoff，可以阻止 skip |
| 没有格挡支持但 Act1 还早 | 只给很小的 future option bonus |
| 没有格挡支持且本身无独立收益 | 不让它因为“便宜攻击牌 + combo tag”越过 anti-skip 阈值 |
| 已晚期且依赖没凑齐 | orphan risk 增加，除非当前就有用，否则不强拿 |

新增 telemetry：

```text
search/build/card_reward_guard_future_combo_window_mean
search/build/card_reward_guard_future_combo_floor_window_mean
search/build/card_reward_guard_future_combo_route_window_mean
search/build/card_reward_guard_future_reward_opportunity_mean
search/build/card_reward_guard_future_runway_observed_rate
```

含义：当前还有多少未来奖励/商店/事件窗口可以让组合组件被补齐。

fallback 规则仍按 Act1 floor 衰减：

```text
floor <= 5   -> 1.00
floor <= 8   -> 0.75
floor <= 11  -> 0.50
floor <= 14  -> 0.25
floor >= 15  -> 0.10
```

但如果 raw obs 里能看到 route/map DTO：

- 优先从 `route_summary` 或 `_sim_raw.map.nodes` 估算后续可用机会；
- normal/elite 视为高概率卡牌奖励窗口；
- shop/event/question/treasure 作为较弱的未来补强窗口；
- boss 距离 <= 3 时强制收缩 option window；
- route/map 分数只和 floor 分数做软 blend，不硬改 policy。

这样可以表达用户指出的关键情况：

```text
同一张组件卡
  early + 后面还有多次奖励/商店      => 有期权价值
  late  + boss 很近 + 当前依赖未成型  => 更像 orphan risk
  当前 deck 已经有 partner           => 立即变强
  当前 deck 缺 partner 但它单卡强     => 仍可按 standalone value 选
```

这不是最终价值函数，只是 anti-skip guard 的软信号。

---

## 4. Death-deck / 阶段日志必须看什么

阶段日志不应只看 win/loss。死亡时 deck 本身是主线诊断指标。

### 4.1 关键 TB 指标

Final deck：

```text
deck/final_size
deck/final_raw_avg_damage_per_energy
deck/final_raw_avg_block_per_energy
deck/final_raw_expected_cards_seen_per_turn
deck/final_raw_expected_playable_cards_per_turn
deck/final_raw_expected_playable_attack_damage_per_turn
deck/final_raw_expected_playable_block_per_turn
deck/final_expected_energy_utilization_score
deck/final_cost_curve_three_plus_share
deck/final_combo_option_value_score
deck/final_combo_unmet_dependency_score
deck/final_delayed_payoff_maturity_score
deck/final_delayed_payoff_option_value_score
deck/final_delayed_payoff_unrealized_risk_score
```

Death deck：

```text
death_deck/size
death_deck/raw_avg_damage_per_energy
death_deck/raw_avg_block_per_energy
death_deck/raw_expected_cards_seen_per_turn
death_deck/raw_expected_playable_cards_per_turn
death_deck/raw_expected_playable_attack_damage_per_turn
death_deck/raw_expected_playable_block_per_turn
death_deck/expected_energy_utilization_score
death_deck/cost_curve_three_plus_share
death_deck/combo_component_density
death_deck/combo_option_value_score
death_deck/combo_unmet_dependency_score
death_deck/delayed_payoff_density
death_deck/delayed_payoff_maturity_score
death_deck/delayed_payoff_unrealized_risk_score
```

Card reward guard：

```text
search/build/card_reward_guard_context_rate
search/build/card_reward_guard_applicable_rate
search/build/card_reward_guard_selected_skip_rate
search/build/card_reward_guard_applied_rate
search/build/card_reward_guard_override_rate
search/build/card_reward_guard_expected_playable_cards_mean
search/build/card_reward_guard_expected_playable_attack_damage_mean
search/build/card_reward_guard_expected_playable_block_mean
search/build/card_reward_guard_expected_energy_utilization_mean
search/build/card_reward_guard_combo_option_value_mean
search/build/card_reward_guard_combo_unmet_dependency_mean
search/build/card_reward_guard_future_combo_window_mean
search/build/card_reward_guard_future_combo_floor_window_mean
search/build/card_reward_guard_future_combo_route_window_mean
search/build/card_reward_guard_future_reward_opportunity_mean
search/build/card_reward_guard_future_runway_observed_rate
search/build/card_reward_guard_delayed_payoff_option_value_mean
search/build/card_reward_guard_delayed_payoff_unrealized_risk_mean
search/build/card_reward_guard_delayed_payoff_maturity_mean
search/build/card_reward_guard_delayed_payoff_time_to_value_mean
```

### 4.2 新增 advisory warnings

这些 warning 不直接让 gate FAIL，但用于判断训练是否偏离：

| warning | 触发含义 |
|---|---|
| `low_death_playable_cards` | 死亡卡组能打出的有效牌太少 |
| `low_death_playable_attack` | 死亡卡组可兑现攻击不足 |
| `low_death_playable_block` | 死亡卡组可兑现防御不足 |
| `low_death_energy_utilization` | 卡组有牌但能量转化效率低 |
| `death_cost_curve_bloat` | 高费牌过多且缺少过牌/引擎 |
| `death_orphan_combo_components` | 组件很多，但依赖没凑齐，体系孤立 |
| `death_unrealized_delayed_payoff` | 慢收益/跨回合 payoff 很多，但成熟度低、死亡前未兑现 |

---

## 5. 如何解读典型异常

### 5.1 连续跳过 4~5 次奖励

重点看：

```text
build/card_reward_skip_rate
build/card_reward_consecutive_skip_max
search/build/card_reward_guard_selected_skip_rate
search/build/card_reward_guard_applied_rate
```

如果 skip 高、guard applied 高：模型在病态跳过，guard 正在兜底。

如果 skip 高、guard applied 低：可能候选牌确实都低质量，或 card reward payload 没有正确暴露候选牌。

### 5.2 死亡时 deck 大但还是弱

看：

```text
death_deck/raw_expected_playable_cards_per_turn
death_deck/expected_energy_utilization_score
death_deck/cost_curve_three_plus_share
death_deck/raw_expected_extra_draw_per_turn
```

这说明不是“没拿牌”，而是“拿了但费用曲线/过牌/有效牌比例差”。

### 5.3 死亡时组件多但体系不成型

看：

```text
death_deck/combo_component_density
death_deck/combo_option_value_score
death_deck/combo_unmet_dependency_score
death_deck/delayed_payoff_density
death_deck/delayed_payoff_maturity_score
death_deck/delayed_payoff_unrealized_risk_score
```

如果 component high + option low + unmet high：模型在拿孤立组件，没补齐 enabler/payoff。

这时下一步不是简单禁止组件卡，而是用人类样本/成功 run 校准：

- 什么阶段值得拿组件；
- 已有几个 partner 时值得拿 payoff；
- 临近 boss 时未成型组件应降权多少；
- 哪些组件本身也是强单卡，哪些只是体系件。

---

## 6. 当前不是最终方案

这次改动解决的是“可观测、可兜底、可诊断”。下一阶段如果要让模型真正学会构筑，需要：

1. 用 human / successful full-run 样本拟合 card reward ranking。
2. 对每个 reward candidate 记录：standalone、deficit-fill、combo-option、dependency-risk、energy-cost。
3. 给 reward/shop/remove/upgrade 决策加入 successor-aware build value。
4. 在 replay scheduler 里增加 card reward / shop / upgrade 的分层采样权重。
5. full-run 验证仍为最终标准，combat sandbox 只能补战斗分布，不能替代构筑/路线学习。

---

## 7. 当前验收

已覆盖测试：

```text
tests/test_deck_quality_v2_phase1.py
tests/test_deck_build_metrics.py
tests/test_card_reward_guard.py
tests/test_monitor_fullrun_act1_gate.py
```

关键测试点：

- deck quality 输出全量 key，finite，无 NaN。
- 费用曲线、可打出牌数、可打出攻防、combo option/unmet 被暴露。
- Body Slam 没有格挡 partner 时不会被 anti-skip guard 强拿。
- Body Slam 有格挡 partner 时会作为有效 combo payoff 阻止病态 skip。
- future combo window 随 Act1 接近 boss 递减。
- death-deck playability / orphan combo warnings 只 advisory，不污染 gate verdict。
