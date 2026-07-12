# MuZero Boss 胜率 >50% 瓶颈修复 Runbook（2026-05-03）

> 本文是给后续执行代理使用的**新执行文档**。目标不是再做泛泛规划，而是把当前 boss 胜率瓶颈拆成可执行、可验收、可中止的任务。
> 主目标：以真实最新 TensorBoard / event log 证明 `recent_tail/256/boss_win_rate > 0.50`。
> 辅助目标：`recent_tail/64/boss_win_rate > 0.50` 只能作为短期突破信号，不能作为最终完成。
> 当前状态：截至本文写入时，目标尚未达成；不要把本文、测试通过、launcher 创建、或者单小时趋势改善当作完成。

---

## 1. 当前最新日志结论

最新被监控脚本读取到的 run：

```text
E:\game\project\sts2_mcp\packages\rl-agent\logs_muzero\muzero_token_memory_combat_sandbox_2envs_20260503_082408
```

关键 boss 指标：

| 指标 | 当前值 | 判断 |
|---|---:|---|
| `recent_tail/64/boss_win_rate` | `last=0.315789`，`mean20=0.317124` | 短窗只有约 31.6%，未达标 |
| `recent_tail/256/boss_win_rate` | `last=0.166667`，`mean20=0.159487` | 主验收严重未达标 |
| `boss/win` | `last=0.0`，`mean20=0.30` | 最近 boss 胜率不稳定 |
| `boss/loss` | `last=1.0`，`mean20=0.70` | boss 仍以失败为主 |

机制/行为指标：

| 指标 | 当前值 | 判断 |
|---|---:|---|
| `search/root_bias_nonzero_rate` | missing | 旧 run 不能证明 root bias 真进入 search/action ranking |
| `search/root_bias_changed_top1_rate` | missing | 旧 run 不能证明 bias 改变最终动作 |
| `boss_combat/root_bias_nonzero_rate` | missing | 旧 run 无法做机制 bias 成效判断 |
| `boss_combat/true_wasteful_end_turn_selected_rate` | `mean20=0.01275` | 真空过不高，但旧 `wasteful_*` 不再可信，要看 true 口径 |
| `boss_combat/forced_end_turn_selected_rate` | `mean20=0.20877` | forced/transient end_turn 仍偏高，需检查桥接稳定面 |
| `boss_combat/kaiser_crab_boss/kaiser_facing_change_candidate_count_mean` | `mean20=0.894` | Kaiser 转身候选已被识别到 |
| `boss_combat/kaiser_crab_boss/kaiser_facing_change_selected_rate` | `mean20=0.0444` | 候选存在但选择率极低 |
| `boss_combat/kaiser_crab_boss/kaiser_risky_end_turn_selected_rate` | `mean20=0.186` | Kaiser 背刺风险下结束回合仍偏多 |
| `boss_combat/ceremonial_beast_boss/ceremonial_high_impact_selected_rate` | `mean20=0.293` | Ceremonial 高价值窗口处理不足 |
| `boss_combat/ceremonial_beast_boss/ceremonial_low_impact_selected_rate` | `mean20=0.0772` | 低价值行动仍存在 |
| `boss_combat/potion_low_urgency_selected_rate` | `mean20=0.0266` | 低紧急度用药尚未完全消失 |
| `boss_combat/potion_high_urgency_selected_rate` | `mean20=0.0046` | 高紧急度用药几乎没有，说明用药时机仍没有学好 |
| `boss_combat/x_cost_zero_bad_selected_rate_p0` | `mean20=0.0081` | 0 能量 X 费误用大幅降低但仍需动态校验 |
| `boss_combat/hp_cost_low_margin_selected_rate` | `mean20=0.0288` | 低血量/低安全边际自损卡仍有风险 |
| `loss/future_world_aux` | `mean20=0.3288` | 当前旧 run 短窗稳定 |
| `loss/future_bank_state` | `mean20=0.0166` | 当前旧 run 短窗稳定 |

结论：当前瓶颈不是“模型需要再多训一点”这么简单，而是：

1. **新 root-bias 可观测指标没有在当前最新 run 中出现**，所以不能确认机制修复是否进入最终动作选择。
2. **Kaiser 已经看到转身候选但不选**，说明不是状态输入完全缺失，而是 action ranking / value / bias 权重不足。
3. **Ceremonial 的高价值窗口选择率不足**，需要 boss-specific 行动价值约束。
4. **药水不是“完全不用”，而是 urgency calibration 错：低紧急度偶尔用，高紧急度几乎不用。**
5. **forced/transient end_turn 仍偏高**，需要验证桥接是否仍在只暴露 `end_turn` 的不稳定窗口让 agent 误选。
6. **主目标距离很远：256-window boss win 只有约 16.7%。**

---

## 2. 不要继续走的错误路线

### 2.1 不要把 overall win 当成 boss 成功

普通/精英 encounter 打得好，不代表 boss 机制处理好了。最终验收必须是：

```text
recent_tail/256/boss_win_rate > 0.50
```

并且至少核查以下 boss 不出现单点崩盘：

```text
kaiser_crab_boss
ceremonial_beast_boss
knowledge_demon_boss
the_kin_boss
the_insatiable_boss
```

### 2.2 不要用旧 launcher 继续长训

旧 launcher 存在几个问题：

- 不保证 boss-heavy encounter distribution；
- 可能复用旧 replay buffer / optimizer momentum；
- 旧 run 缺失 root-bias effectiveness metrics；
- boss 样本不足时，global metric 会被 normal / elite 稀释。

### 2.3 不要再用“有 playable card 就一定不能 end_turn”的粗糙判断

有些牌合法可打，但策略上应该保留或延迟：

- 消耗牌：打出后进入消耗堆；
- 回费牌：如果后续没有可打牌，白打可能亏；
- X 费牌：有效能量是动态的，不能被初始 3 费误导；
- 保留/重放/虚无/复制/替换/附魔类牌：改变的是未来手牌质量；
- 自损牌：需要看当前血量、安全边际和是否会自杀。

因此 end_turn detector 必须使用 `true_wasteful_*` / `strategic_defer_*` / `forced_end_turn_*` 这类分桶，而不是简单看 playable count。

### 2.4 不要用长 sleep 修桥接瞬时窗口

桥接层如果在抽卡、洗牌、动画、状态刷新中只暴露 `end_turn`，应通过**稳定状态判定 + 短轮询**来延迟暴露，而不是长时间固定等待。固定等待会拖慢训练吞吐，并可能超过 combat reset 等待开销。

---

## 3. 总体修复策略

分 4 条主线并行推进，但每条都有独立验收：

1. **启动正确的新 boss-recovery run**
   warm-start 最新 checkpoint，但丢弃旧 optimizer 和 replay buffer；显式 boss-heavy；确认 event 在更新。

2. **先证明机制 bias 真的影响最终动作**
   新 run 必须 emit `root_bias_nonzero_rate`、`root_bias_changed_top1_rate`、per-encounter root-bias 指标。没有这些就不要长训。

3. **修 boss-specific 行动排序**
   Kaiser、Ceremonial、Knowledge Demon、The Kin、Insatiable 分 encounter 追踪，不再只看 global boss aggregate。

4. **修通用高风险行动 calibration**
   药水时机、X 费动态有效能量、自损卡安全边际、true wasteful end_turn、forced end_turn、future-bank spike dump。

---

## 4. P0：先让新训练真的跑起来

### 4.1 启动方式

优先使用新的 boss recovery launcher：

```text
packages/rl-agent/launch_muzero_boss_recovery_20260503.sh
```

如果 PowerShell/WSL 后台启动失败，使用 WSL 侧 wrapper，避免嵌套 quoting 问题：

```bash
cat > /mnt/e/game/project/sts2_mcp/packages/rl-agent/start_boss_recovery_background.sh <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
cd /mnt/e/game/project/sts2_mcp/packages/rl-agent
mkdir -p train_logs
LOG="train_logs/boss_recovery_launch_$(date +%Y%m%d_%H%M%S).log"
nohup bash ./launch_muzero_boss_recovery_20260503.sh > "$LOG" 2>&1 < /dev/null &
PID=$!
echo "$PID" > train_logs/boss_recovery_latest.pid
echo "$LOG" > train_logs/boss_recovery_latest.logpath
echo "linux_pid=$PID"
echo "launch_log=$LOG"
EOF
chmod +x /mnt/e/game/project/sts2_mcp/packages/rl-agent/start_boss_recovery_background.sh
```

PowerShell 启动：

```powershell
wsl bash /mnt/e/game/project/sts2_mcp/packages/rl-agent/start_boss_recovery_background.sh
```

### 4.2 启动后 60 秒内必须验证

```powershell
wsl bash -lc "ps aux | grep -E 'muzero.train|launch_muzero_boss_recovery' | grep -v grep || true"

Get-ChildItem E:\game\project\sts2_mcp\packages\rl-agent\logs_muzero -Directory |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 5 Name,LastWriteTime

Get-ChildItem E:\game\project\sts2_mcp\packages\rl-agent\train_logs -Filter 'boss_recovery*.log' |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 5 Name,Length,LastWriteTime
```

期望出现：

```text
logs_muzero/muzero_boss_recovery_<timestamp>
train_logs/boss_recovery_launch_<timestamp>.log
```

如果没有新 run dir，也没有 launch log，说明训练根本没启动，不要进入监控阶段。

---

## 5. P0：新 run 第一验收——root bias 是否真的生效

### 5.1 必须出现的 exact tags

新 run 10-20 分钟内必须出现：

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
boss_combat/kaiser_crab_boss/root_bias_changed_top1_rate
boss_combat/ceremonial_beast_boss/root_bias_changed_top1_rate
```

监控命令：

```powershell
cd E:\game\project\sts2_mcp\packages\rl-agent
.\venv\Scripts\python.exe scripts\monitor_boss_recovery.py --tail 20
```

### 5.2 判定逻辑

| 观测 | 判定 | 下一步 |
|---|---|---|
| root-bias tags missing | instrumentation 未进入当前训练路径，或者 logger key 不一致 | 停止训练，检查 `mcts.py` / `train.py` emit 路径 |
| `root_bias_nonzero_rate = 0` | bias 没被计算或被 gate 全压掉 | 查 feature/action semantic/encounter context |
| `root_bias_nonzero_rate > 0` 但 `changed_top1_rate = 0` | bias 太弱或只影响非竞争动作 | 调整 scale、gate、候选优先级 |
| `changed_top1_rate > 0` 但 boss win 不升 | bias 方向可能错，或者 policy/value 与机制目标冲突 | 看 per-encounter action delta 和失败回合 dump |
| `suppressed_by_gate_rate` 高 | drift/uncertainty gate 过严 | 检查 gate 条件是否错误压制 boss-critical bias |

P0 原则：**没有 root-bias effectiveness，就不要相信任何 boss-specific 修复已经实际影响 policy。**

---

## 6. P1：Kaiser Crab 专项修复

### 6.1 当前症状

```text
kaiser_facing_change_candidate_count_mean mean20 ≈ 0.894
kaiser_facing_change_selected_rate        mean20 ≈ 0.044
kaiser_risky_end_turn_selected_rate       mean20 ≈ 0.186
```

说明：

- 左右部位解析已经基本进入 candidate；
- 不是完全没有转身候选；
- 但模型极少选择转身；
- 带背刺风险结束回合仍偏多。

### 6.2 机制定义

Kaiser 转身不应依赖 `enemy.side` / `target.side` 的 `left/right` 字段，因为 bridge 中 `side` 是敌我阵营，不是左右。左右位置来自 enemy powers：

```text
BACK_ATTACK_LEFT_POWER
BACK_ATTACK_RIGHT_POWER
```

转身动作定义：

```text
任意指向操作，只要 target_combat_id 指向当前 facing 的相反 Kaiser 部位，即视为 facing-change action。
```

包括：

- 攻击牌；
- 指向药水；
- debuff / weak / vulnerable 牌；
- 其他 target enemy 的合法动作。

### 6.3 要补齐的指标

```text
boss_combat/kaiser_crab_boss/kaiser_back_attack_risk_mean
boss_combat/kaiser_crab_boss/kaiser_facing_change_available_rate
boss_combat/kaiser_crab_boss/kaiser_facing_change_candidate_count_mean
boss_combat/kaiser_crab_boss/kaiser_facing_change_selected_rate
boss_combat/kaiser_crab_boss/kaiser_risky_end_turn_available_rate
boss_combat/kaiser_crab_boss/kaiser_risky_end_turn_selected_rate
boss_combat/kaiser_crab_boss/root_bias_changed_top1_rate
boss_combat/kaiser_crab_boss/root_bias_selected_action_delta_mean
```

### 6.4 行动修复规则

优先级建议：

1. 若有 lethal：lethal > 转身防守。
2. 若背刺风险高且无 lethal：
   - 选择能转身且不显著亏节奏的目标动作；
   - 若同时有高 pressure，允许 pressure 方案，但必须降低 `risky_end_turn_selected_rate`。
3. 若当前动作会导致背刺风险回合结束仍存在：
   - 对 `end_turn` 加显式负 bias；
   - 对 non-lethal damage-only 且不转身的动作谨慎评分。

最低验收：

```text
kaiser_facing_change_selected_rate >= 0.15
kaiser_risky_end_turn_selected_rate <= 0.10
kaiser boss win trend 不低于旧 run
```

---

## 7. P1：Ceremonial Beast 专项修复

### 7.1 当前症状

```text
ceremonial_high_impact_selected_rate mean20 ≈ 0.293
ceremonial_low_impact_selected_rate  mean20 ≈ 0.077
ceremonial_missed_stun_window_rate   missing
```

说明：

- 已经有 high/low impact 分类；
- 但缺少 missed stun-window exact metric；
- 不知道模型是否在关键窗口错过爆发/解锁/保命。

### 7.2 要补齐的指标

```text
boss_combat/ceremonial_beast_boss/ceremonial_stun_window_rate
boss_combat/ceremonial_beast_boss/ceremonial_missed_stun_window_rate
boss_combat/ceremonial_beast_boss/ceremonial_high_impact_available_rate
boss_combat/ceremonial_beast_boss/ceremonial_high_impact_selected_rate
boss_combat/ceremonial_beast_boss/ceremonial_low_impact_selected_rate
boss_combat/ceremonial_beast_boss/root_bias_changed_top1_rate
```

### 7.3 行动修复规则

关键原则：

- 在 stun / one-card-lock / burst window 中，不能只按当前 immediate reward 选牌；
- 高 impact 动作应包括：lethal setup、强 block、关键 debuff、破机制、抽牌/回费后可继续展开；
- 低 impact 动作包括：低伤害、无机制收益、无后续展开的消耗资源动作。

最低验收：

```text
ceremonial_high_impact_selected_rate 上升
ceremonial_low_impact_selected_rate 不上升
ceremonial_missed_stun_window_rate 出现且下降
```

---

## 8. P1：药水时机修复

### 8.1 当前症状

```text
potion_low_urgency_selected_rate  mean20 ≈ 0.0266
potion_high_urgency_selected_rate mean20 ≈ 0.0046
potion_lethal_selected_rate       mean20 = 0
potion_prevent_lethal_selected_rate mean20 = 0
```

这不是简单的“有药水就用”或“完全不用”。更准确地说：

```text
模型没有学会 urgency 分层。
```

表现为：

- 低紧急度偶尔用药；
- 真正高紧急度、斩杀、防死场景几乎不用；
- potion action 可能被 immediate damage / policy prior 错误驱动。

### 8.2 药水分类

按用途分桶：

1. **Lethal potion**：本回合直接造成击杀或稳定击杀。
2. **Prevent-lethal potion**：若不用药，敌方意图/机制可能造成死亡；用药能防死。
3. **Swing potion**：不能立刻斩杀/防死，但显著改变战局，例如强 block、强 debuff、强抽牌/能量。
4. **Setup potion**：为后续回合准备，只有在 boss 长战且风险可控时合理。
5. **Low-urgency potion**：没有斩杀、防死、机制窗口、资源雪崩价值时使用。
6. **Bad potion**：目标错误、overkill 过大、浪费稀缺药水、或抢占更好动作。

### 8.3 要补齐的指标

```text
boss_combat/potion_low_urgency_available_rate
boss_combat/potion_low_urgency_selected_rate
boss_combat/potion_high_urgency_available_rate
boss_combat/potion_high_urgency_selected_rate
boss_combat/potion_lethal_available_rate
boss_combat/potion_lethal_selected_rate
boss_combat/potion_prevent_lethal_available_rate
boss_combat/potion_prevent_lethal_selected_rate
boss_combat/potion_bad_selected_rate
boss_combat/<encounter>/potion_*_selected_rate
```

### 8.4 修复规则

- 低紧急度药水：加负 bias，但不要完全禁用，因为某些 setup potion 在 boss 长战合理。
- 斩杀药水：强正 bias。
- 防死药水：强正 bias，优先级高于普通攻击。
- boss 机制药水：如果药水能触发 Kaiser 转身、Ceremonial 关键窗口、或防止 boss lethal，应归入 high urgency。
- 使用后必须确认 bridge potion slot 清空或 metric 端识别已消耗，避免 `potion_unused_on_death` 假阳性。

最低验收：

```text
potion_low_urgency_selected_rate 下降
potion_high_urgency_selected_rate 在 high_urgency_available 时明显上升
potion_lethal/prevent_lethal selected 不再长期为 0
```

---

## 9. P1：X 费牌动态有效能量

### 9.1 问题定义

X 费牌不能用回合初始能量判断。有效 X 值应是动作发生时的实时能量：

```text
effective_x = current_energy_at_action_resolution
```

如果前面打了回费牌、药水、降费、保留/附魔效果，X 费价值会变化。

### 9.2 错误模式

- 模型被初始 3 费误导，以为 X 费有价值，但当前实际能量为 0；
- 回费牌之后未刷新 legal/action features，导致 X 费估值滞后；
- 0 能量 X 费有特殊收益时被误罚，例如某些不依赖 X 的副效果，需要白名单/semantic utility 判断。

### 9.3 要补齐的指标

```text
boss_combat/x_cost_available_rate_p0
boss_combat/x_cost_selected_rate_p0
boss_combat/x_cost_zero_bad_available_rate_p0
boss_combat/x_cost_zero_bad_selected_rate_p0
boss_combat/x_cost_effective_energy_mean
boss_combat/x_cost_selected_effective_energy_mean
boss_combat/<encounter>/x_cost_zero_bad_selected_rate_p0
```

### 9.4 修复规则

- 每次动作后必须刷新 hand/legal/action feature；
- X 费 action feature 使用当前 energy，不使用 turn-start energy；
- 只有在 `effective_x == 0` 且没有独立正收益 semantic 时，才标为 bad zero-X；
- 如果 X 费可斩杀、防死、触发机制，即使 X 小也不能简单惩罚。

最低验收：

```text
x_cost_zero_bad_selected_rate_p0 接近 0
x_cost_selected_effective_energy_mean 合理大于 0
```

---

## 10. P1：自损 / 扣血成本卡安全

### 10.1 问题定义

模型偶尔使用扣血卡不看自身血量，可能造成自杀或低血量进入不可恢复状态。需要把 HP cost 从普通负 reward 改成 action-level safety constraint。

### 10.2 卡牌/效果范围

需要覆盖：

- 直接支付 HP 的牌；
- 造成自伤的牌；
- 以 HP 换能量/抽牌/伤害的牌；
- 会触发后续 HP loss 的附魔/状态；
- 回费牌：若支付 HP 后无后续可打动作，实际是白亏血。

### 10.3 要补齐的指标

```text
boss_combat/hp_cost_available_rate
boss_combat/hp_cost_selected_rate
boss_combat/hp_cost_self_lethal_available_rate
boss_combat/hp_cost_self_lethal_selected_rate
boss_combat/hp_cost_low_margin_available_rate
boss_combat/hp_cost_low_margin_selected_rate
boss_combat/hp_cost_no_followup_selected_rate
boss_combat/<encounter>/hp_cost_*_selected_rate
```

### 10.4 修复规则

硬约束：

```text
若 hp_after_cost <= 0，除非动作本身确定先结算胜利，否则禁止/极强负 bias。
```

软约束：

```text
若 hp_after_cost <= enemy_next_damage_or_mechanic_risk + safety_margin，降低优先级。
```

回费/抽牌自损牌额外规则：

```text
如果打出后没有可用 follow-up action，且不能防死/斩杀/破机制，则标为 hp_cost_no_followup_bad。
```

最低验收：

```text
hp_cost_self_lethal_selected_rate = 0
hp_cost_low_margin_selected_rate 下降
hp_cost_no_followup_selected_rate 出现并下降
```

---

## 11. P1：end_turn / 桥接瞬时窗口

### 11.1 当前症状

```text
true_wasteful_end_turn_selected_rate mean20 ≈ 0.0128
forced_end_turn_selected_rate        mean20 ≈ 0.2088
```

`true_wasteful` 不高，但 `forced_end_turn` 偏高。重点不是再调旧 `wasteful_end_turn_rate`，而是检查 bridge 是否在短暂动画/刷新窗口只暴露 `end_turn`。

### 11.2 分类要求

end_turn 必须拆成：

1. `forced_end_turn`：确实没有任何其他合法动作。
2. `transient_only_end_turn`：短暂不稳定窗口只暴露 end_turn，等待后会出现动作。
3. `true_wasteful_end_turn`：有正收益动作且非战略保留却结束回合。
4. `strategic_defer_end_turn`：合法动作存在但打出会亏，例如消耗牌/回费牌无 follow-up/保留更优。
5. `bad_end_turn`：风险明显未处理，例如 Kaiser 背刺、敌方 lethal、boss 机制窗口错过。

### 11.3 桥接修复原则

- 不用长 sleep；
- 使用直接状态判定：queue empty、animation settled、hand stable、legal actions stable、draw/discard transition stable；
- 若当前只暴露 `end_turn`，执行短轮询；
- 若短轮询内出现非 end_turn 动作，则不向模型暴露 transient end_turn；
- 若状态稳定后仍只有 end_turn，才暴露。

建议短轮询预算：

```text
poll interval: 25-50 ms
max polls: 4-8
max total: 100-300 ms
```

超过这个范围就可能显著拖慢训练，不能接受。

### 11.4 要补齐的指标

```text
bridge/transient_only_end_turn_detected_rate
bridge/transient_only_end_turn_suppressed_rate
bridge/end_turn_validation_poll_count_mean
bridge/end_turn_validation_wait_ms_mean
bridge/end_turn_validation_wait_ms_p95
boss_combat/forced_end_turn_selected_rate
boss_combat/true_wasteful_end_turn_selected_rate
boss_combat/strategic_defer_end_turn_selected_rate
boss_combat/bad_end_turn_selected_rate
```

最低验收：

```text
transient_only_end_turn_suppressed_rate > 0 when detected
end_turn_validation_wait_ms_p95 <= 300 ms
forced_end_turn_selected_rate 不再异常高
true_wasteful_end_turn_selected_rate 不上升
```

---

## 12. P2：future-world / future-bank spike 防护

### 12.1 问题定义

此前 boss-heavy batch 曾出现 `future_world_aux` / `future_bank_state` 数百到上千量级 spike。即使 fresh optimizer 能恢复，spike 触发源仍可能污染训练。

### 12.2 必须保留的防护

- loss spike threshold dump；
- dump prev/next obs、action、aux targets、encounter id、card/action features；
- spike batch quarantine；
- 不加载旧 optimizer momentum；
- `future_bank_token_slot_source_weight = 0` 直到该 head 完全验证。

### 12.3 指标

```text
loss/total
loss/future_world_aux
loss/future_bank_state
loss/future_bank_delta
loss_spike/dump_count
loss_spike/quarantine_count
```

停止条件：

```text
loss/future_world_aux 或 loss/future_bank_state 连续 spike，且 median 被抬高
loss/total median 持续上升而不是单点 spike
policy/value/reward loss 同时劣化
```

---

## 13. 训练 Gate 计划

### 13.1 T+10 分钟 Gate：训练是否真实有效启动

必须满足：

```text
新 run_dir 出现
TensorBoard event mtime 持续更新
boss/win 或 boss/loss 开始出现
root-bias exact tags 出现
```

失败动作：

- 如果没有新 run：修 launcher/WSL 启动；
- 如果没有 event：看 launch log；
- 如果 root-bias tags missing：停训，修 instrumentation。

### 13.2 T+30 分钟 Gate：机制 bias 是否进入动作选择

必须检查：

```text
search/root_bias_nonzero_rate
search/root_bias_changed_top1_rate
boss_combat/root_bias_changed_top1_rate
boss_combat/kaiser_crab_boss/root_bias_changed_top1_rate
boss_combat/ceremonial_beast_boss/root_bias_changed_top1_rate
```

期望：

```text
root_bias_nonzero_rate > 0
boss-specific changed_top1_rate > 0 on targeted encounters
```

失败动作：

- nonzero = 0：查 context/semantic；
- changed_top1 = 0：调 scale/gate；
- suppressed_by_gate 高：查 drift/uncertainty gate。

### 13.3 T+60 分钟 Gate：boss 短窗是否改善

观察：

```text
recent_tail/64/boss_win_rate
boss/win mean20
kaiser_facing_change_selected_rate
kaiser_risky_end_turn_selected_rate
ceremonial_high_impact_selected_rate
potion_high_urgency_selected_rate
x_cost_zero_bad_selected_rate_p0
hp_cost_low_margin_selected_rate
loss/future_world_aux
loss/future_bank_state
```

继续条件：

```text
recent_tail/64/boss_win_rate 明显上升，且机制指标方向正确，loss median 稳定
```

停止/回滚条件：

```text
boss_win_rate 不动或下降
root_bias_changed_top1_rate 长期为 0
Kaiser/Ceremonial 指标恶化
low_urgency potion 上升而 high_urgency 不动
future-bank spike 污染 median
```

### 13.4 主验收 Gate

只有以下条件同时满足，才算达到目标：

```text
recent_tail/256/boss_win_rate > 0.50
recent_tail/64/boss_win_rate  > 0.50
per-boss 不出现单 boss 近似 0% 崩盘
loss median 稳定
root-bias metrics 存在且非空
```

---

## 14. 执行任务清单

### P0-A：启动/监控新 boss-recovery run

- [ ] 确认 `launch_muzero_boss_recovery_20260503.sh` 存在且 `bash -n` 通过。
- [ ] 用 WSL wrapper 后台启动。
- [ ] 60 秒内确认 Linux 进程存在。
- [ ] 确认新 `logs_muzero/muzero_boss_recovery_*` 目录出现。
- [ ] 确认 launch log 非空且无 fatal error。
- [ ] 跑 `scripts/monitor_boss_recovery.py --tail 20`。

### P0-B：root-bias effectiveness

- [ ] 新 run 出现 root-bias exact tags。
- [ ] global `root_bias_nonzero_rate > 0`。
- [ ] boss_combat `root_bias_nonzero_rate > 0`。
- [ ] Kaiser / Ceremonial per-encounter `root_bias_changed_top1_rate` 出现。
- [ ] 若 changed_top1 长期为 0，调 bias scale/gate，而不是继续长训。

### P1-A：Kaiser

- [ ] 用 powers id 解析 left/right 部位。
- [ ] 所有指向相反部位的 enemy-target action 都算转身候选，包括攻击、药水、debuff。
- [ ] 风险下 end_turn 加强负 bias。
- [ ] target side 错误/不转身的低价值动作降权。
- [ ] 指标验收：`facing_change_selected_rate` 上升，`risky_end_turn_selected_rate` 下降。

### P1-B：Ceremonial

- [ ] 补 `ceremonial_missed_stun_window_rate`。
- [ ] 定义 high-impact / low-impact 的 boss-specific 规则。
- [ ] 在 stun/lock/window 中提高 high-impact bias。
- [ ] 指标验收：high-impact 上升，missed-window 下降。

### P1-C：Potion timing

- [ ] 区分 lethal / prevent-lethal / swing / setup / low-urgency / bad potion。
- [ ] 高紧急度动作强正 bias。
- [ ] 低紧急度动作负 bias，但不绝对禁用。
- [ ] 修 `potion_unused_on_death` 与 bridge slot 清空/已消耗识别的一致性。
- [ ] 指标验收：high-urgency selected 上升，low-urgency selected 下降。

### P1-D：X-cost dynamic

- [ ] action feature 使用当前实时 energy。
- [ ] 每次动作后刷新 hand/legal/action feature。
- [ ] 0 能量 X 费只在无独立正收益时标 bad。
- [ ] 指标验收：`x_cost_zero_bad_selected_rate_p0` 接近 0。

### P1-E：HP-cost safety

- [ ] 覆盖支付 HP、自伤、HP 换能量/抽牌/伤害、附魔后续自伤。
- [ ] self-lethal 硬约束。
- [ ] low-margin 软约束。
- [ ] hp-cost no-followup bad 检测。
- [ ] 指标验收：self-lethal selected = 0，low-margin selected 下降。

### P1-F：end_turn transient / strategic defer

- [ ] 拆分 forced / transient / true wasteful / strategic defer / bad end_turn。
- [ ] 只在状态稳定后暴露 end_turn。
- [ ] 短轮询总预算控制在 100-300 ms。
- [ ] 指标验收：transient suppressed > 0，wait p95 <= 300 ms。

### P2：spike dump / quarantine

- [ ] 保留 loss spike dump。
- [ ] spike 时 dump encounter、obs、action、aux targets。
- [ ] 若 median 被污染，停训并 quarantine batch。

---

## 15. 最终 Definition of Done

本文对应任务只有在以下证据满足时才算完成：

```text
latest_run_dir 是新的 boss recovery run
recent_tail/256/boss_win_rate > 0.50
recent_tail/64/boss_win_rate  > 0.50
boss/win mean20 稳定高于 0.50 附近或继续上升
Kaiser / Ceremonial / Knowledge Demon / The Kin / Insatiable 没有单点崩盘
root-bias metrics 存在且显示 bias 非零、能改变 top1
potion/X-cost/HP-cost/end_turn 指标不出现明显回归
loss/future_world_aux 与 loss/future_bank_state median 稳定
```

如果只满足下面任意一项，不算完成：

- 文档写完；
- 单元测试通过；
- launcher 创建；
- 训练进程启动；
- 64-window 短暂超过 50%；
- overall win 超过 50%；
- 单个 boss 短暂变好但 256-window 未达标。

---

## 16. 推荐下一步

立即执行顺序：

1. 启动 `launch_muzero_boss_recovery_20260503.sh` 对应的新 boss-heavy run。
2. 10-20 分钟内用 `monitor_boss_recovery.py` 验证 root-bias exact tags。
3. 若 tags missing，停训修 instrumentation。
4. 若 tags 存在但 changed-top1 为 0，调 root-bias scale/gate。
5. 若 root-bias 生效，再看 Kaiser/Ceremonial 指标是否按预期改善。
6. 只有机制指标和 boss 短窗都改善时，才继续长训等待 256-window boss win 过 50%。
