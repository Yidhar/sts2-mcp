# Act1 主线验证清单（2026-05-15）

目标：接管 coding / training / monitoring，使 MuZero full-run 模型稳定突破 Act1。

本文件记录 EndTurn 收口后的主线验证口径。后续不要再用单局
`end_turn_contexts.jsonl` 作为主线阻塞判断；只有聚合红线变坏时才重开
EndTurn 线。

---

## 1. 当前结论

### EndTurn / 空过

结论：**不是当前 Act1 主 blocker**。

只保留聚合监控：

- `combat_quality/wasteful_end_turn_rate`
- `combat_quality/missed_lethal`
- `combat_quality/missed_damage`
- `combat_quality/avoidable_damage`
- `combat_quality/bad_pure_block_selected_rate`
- `combat_quality/insufficient_block_selected_rate`

重开条件：

- `wasteful_end_turn_rate > 0.002`，或
- `missed_lethal / missed_damage / avoidable_damage` 出现持续非零，或
- death slice 明确显示满能量、稳定 frontier、有合法高价值动作但 EndTurn。

否则不再追单局 EndTurn 日志。

### 当前主 blocker

当前主线问题是 **构筑质量不足**：

- 卡牌奖励跳过率过高；
- 死亡卡组 starter-heavy；
- boss / elite readiness 低；
- 商店打开、购买、删牌行为需要新 run 验证；
- 旧训练进程不会热加载新代码，因此当前 active run 只能看趋势，不能验证最新修复。

---

## 2. 当前 active run 基线

Run：

```text
muzero_pass_large_fullrun_random_act1_tempo_deckdiag_from_104457_20260514_205259
```

最新 gate 摘要：

```text
episode_count = 103
buffer/size ≈ 14515
recent_tail/64/act1_boss_seen_rate = 0.15625
recent_tail/64/act1_pass_rate = 0
episode/act1_clear = 0
verdict = FAIL
```

Loss / memory 健康：

```text
loss/total max_tail ≈ 13.74
loss/future_world_aux max_tail ≈ 0.205
loss/future_bank_state max_tail ≈ 0.031
loss/future_bank_delta max_tail ≈ 0.062
memory/max_allocated_gb ≈ 9.39
memory/reserved_gb ≈ 10.85
```

死亡卡组近期均值：

```text
reward pick/skip ≈ 0.29 / 0.71
deck size ≈ 15.1
starter_count ≈ 8.5
nonstarter_count ≈ 6.6
upgraded_count ≈ 0.4
dmg/E ≈ 3.20
block/E ≈ 1.92
boss_readiness ≈ 0.26
elite_readiness ≈ 0.27
```

---

## 3. 新代码需要新 run 验证的指标

### 3.1 Card reward

必须出现并观察：

```text
build/card_reward_seen
build/card_reward_pick_rate
build/card_reward_skip_rate
build/card_reward_consecutive_skip_max
search/build/card_reward_guard_skip_blocked_rate
search/build/post_search_hard_guard_policy_retargeted_rate
```

期望：

- `card_reward_skip_rate` 明显低于旧 run 的 `~0.70`；
- `post_search_hard_guard_policy_retargeted_rate` 不再 missing；
- 如果 skip guard firing，则 retarget rate 应接近 guard applied rate；
- death deck 的 starter ratio 下降，nonstarter count 上升。

诊断 JSONL：

```text
diagnostics/card_reward_choices.jsonl
```

重点检查字段：

```text
original_selected_is_skip=True
final_selected_is_skip=False
selected_is_skip=False
skip_blocked=True
policy_retargeted=True
original_action_idx != final_action_idx
```

### 3.2 Shop

必须出现并观察：

```text
build/shop_seen
build/shop_open_rate
build/shop_buy_any_rate
build/shop_buy_card_rate
build/shop_buy_relic_rate
build/shop_buy_potion_rate
build/shop_remove_rate
build/shop_leave_rate
build/shop_leave_with_gold_ge_100_rate
build/shop_leave_with_remove_affordable_rate
search/build/shop_action_guard_open_applied_rate
search/build/shop_action_guard_remove_applied_rate
```

红线：

- 有 `gold >= 100` 仍长期不打开商店；
- remove affordable 且 starter/junk-heavy 仍长期不删牌；
- shop seen 后 open/buy/remove 接近 0。

### 3.3 Death deck

每次看日志必须看死亡卡组。重点字段：

```text
death_deck/size
death_deck/starter_count
death_deck/starter_ratio
death_deck/nonstarter_count
death_deck/upgraded_count
death_deck/upgraded_ratio
death_deck/raw_avg_damage_per_energy
death_deck/raw_avg_block_per_energy
death_deck/raw_expected_cards_seen_per_turn
death_deck/raw_expected_playable_cards_per_turn
death_deck/boss_readiness_score
death_deck/elite_readiness_score
death_deck/combo_unmet_dependency_score
```

期望方向：

- starter ratio 下降；
- nonstarter count 上升；
- reward skip 下降；
- boss / elite readiness 上升；
- draw / playable / per-energy quality 不再长期低位。

---

## 4. Gate 命令

不要只看 TensorBoard 单项。每次主线检查同时跑：

```bash
cd /mnt/e/game/project/sts2_mcp/packages/rl-agent
./.venv-wsl-rocm/bin/python scripts/monitor_fullrun_act1_gate.py --tail 50
./.venv-wsl-rocm/bin/python scripts/report_fullrun_death_decks.py --tail 10 --show-cards
```

Windows / PowerShell：

```powershell
wsl bash -lc "cd /mnt/e/game/project/sts2_mcp/packages/rl-agent && ./.venv-wsl-rocm/bin/python scripts/monitor_fullrun_act1_gate.py --tail 50 && ./.venv-wsl-rocm/bin/python scripts/report_fullrun_death_decks.py --tail 10 --show-cards"
```

---

## 5. Act1 通过判据

早期新 run 最低 gate：

```text
recent_tail/64/act1_boss_seen_rate >= 0.30
recent_tail/64/act1_pass_rate > 0
episode/act1_clear > 0
loss/future_world_aux 不 spike
loss/future_bank_state 不 spike
memory/max_allocated_gb < 20GB
```

稳定突破 Act1 的目标：

```text
recent_tail/64/act1_boss_seen_rate >= 0.60
recent_tail/64/act1_pass_rate >= 0.20
recent_tail/256/act1_pass_rate > 0
death_deck/starter_ratio 明显低于旧 run
card_reward_skip_rate 明显低于 0.70
shop_open/buy/remove 指标非零且方向合理
combat redlines 不爆
```

---

## 6. 代码验证状态

当前聚焦回归：

```text
275 passed
```

覆盖：

- Body Slam 动态伤害；
- card reward guard；
- post-search policy retarget；
- shop action guard；
- shop metrics；
- async telemetry build/shop/death-deck metrics；
- death deck report；
- Act1 gate monitor；
- frontier recovery；
- combat hard guards。

文件拆分：

```text
muzero/diagnostics/trainer_dumps.py      2217 -> 1834 lines
muzero/diagnostics/death_slice_dumps.py  new, 402 lines
```

---

## 7. 下一步

不要继续追 EndTurn 单局日志。

下一轮新 run 的首要验证项：

1. `post_search_hard_guard_policy_retargeted_rate` 是否出现；
2. card reward skip 是否下降；
3. shop metrics 是否出现；
4. shop open / buy / remove 是否非零；
5. death deck starter-heavy 是否缓解；
6. `act1_boss_seen_rate` 是否从 `~0.16` 恢复到 `>=0.30`；
7. 是否出现 `act1_clear > 0`。
