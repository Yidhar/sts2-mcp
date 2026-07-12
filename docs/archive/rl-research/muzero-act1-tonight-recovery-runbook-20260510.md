# MuZero Act1 今晚恢复 Runbook（2026-05-10）

> 目标：**今晚把当前 MuZero full-run 训练从“Act1 路线崩溃 / 到不了 boss”恢复到可推进状态**。
> 本文档写给执行 agent（Claude / Codex / 人工均可），要求按顺序执行、每一步有明确 pass/fail。
> 当前优先级不是做长期架构，而是：**先让模型稳定见到 Act1 boss，并争取出现 Act1 clear**。

---

## 0. 当前结论（一句话）

**不要继续调 Phase 3 route heuristic bias。**

`--route-heuristic-bias 0.0` 的 baseline 已经同样崩溃，说明根因在更上游：

1. checkpoint / buffer / optimizer 可能已被污染；
2. full-run 线路/动作索引/观测 schema 可能有回归；
3. Phase2→Phase3 从 combat-sandbox 切 full-run 后，路线选择数据分布变了；
4. bridge runtime 还有 401 token/reset 崩溃风险；
5. route heuristic 当前只能做 **诊断/安全护栏**，不能作为在线 prior 强行混入。

今晚的主线是：

```text
P0 运行时健康检查
→ P1 checkpoint/config bisect 找回健康基点
→ P2 route action 对齐测试 + route 安全护栏
→ P3 从健康基点 fresh optimizer / empty buffer 重新 full-run
→ P4 用硬指标决定继续/停机/回滚
```

---

## 1. 这次要达成什么

### 1.1 今晚最低成功标准

在 full-run smoke / training 中达到：

```text
recent_tail/64/act1_boss_seen_rate >= 0.30
```

解释：最近 64 个 episode 里至少约 19 次见到 Act1 boss。
这说明 route/build/combat 基本能支撑 Act1 前半段，不再卡在 floor 6–11。

### 1.2 今晚进阶成功标准

任一满足即可：

```text
recent_tail/128/act1_pass_rate > 0.00
```

或：

```text
至少出现 1 次 Act1 clear
```

### 1.3 必须立即停止的失败标准

任一满足就停，不要继续浪费训练：

```text
50 episode 后 episode/act1_boss_seen_rate < 0.10
recent_tail/64/act1_boss_seen_rate 继续下降
episode/max_act_id = 0 且 death_floor 中位数 <= 10
route_heuristic/top1_selected_rate_multi 最近 5 个 episode 全为 0
route_heuristic/best_minus_selected_mean_multi 持续上升并 > 4.0
route/action_index_alignment_error_rate > 0
bridge reset 出现重复 401 / missing_or_invalid_token
loss/future_world_aux 或 loss/future_bank_state 持续 spike，不是孤立 spike
```

---

## 2. 已知坏状态：不要从这些地方继续

### 2.1 不要继续当前 bad baseline

当前坏 run：

```text
packages/rl-agent/logs_muzero/muzero_phase3_bias0_baseline_20260509
packages/rl-agent/checkpoints_muzero/muzero_phase3_bias0_baseline_20260509
```

最新读数：

```text
50 episodes, 496 route decisions
route bias confirmed off
episode/act1_boss_seen = 0.04
recent_tail/64/act1_boss_seen_rate 0.0435 → 0.040
recent_tail/64/act1_pass_rate = 0.0
episode/max_floor last5 = [11, 6, 6, 11, 11]
top1_selected_rate_multi = 0.123, last5 = [0,0,0,0,0]
best_minus_selected_mean_multi = 4.18, last5 rising
forced_elite_selected_rate = 0.085
```

结论：

```text
Phase3 online route bias 不是根因；
bias=0 baseline 自己也坏；
不要从 phase3_bias0_baseline 的 ckpt / buffer / optimizer 继续。
```

### 2.2 不要继续调 route heuristic weight

下面这些今晚都不要做：

```text
--route-heuristic-bias 0.05
--route-heuristic-bias 0.10
继续 v2/v3 heuristic score 微调
让 heuristic 参与 visit/prior 分布但没有强诊断
```

原因：

1. `bias_applied_rate=1.0` 只能说明 plumbing 通了；
2. `root_bias_changed_top1_rate=0` 时指标仍恶化，说明 tiny bias 也可能扭曲搜索分布；
3. 当前上游 baseline 已坏，调 bias 只能污染判断。

---

## 3. 今晚总策略

### 3.1 用 “heuristic system” 而不是盲目梯度训练

参考 `learning-beyond-gradients` 的核心思想：
不要只靠神经网络权重自己学，先把系统改成可诊断、可回放、可防回归。

今晚落地为：

```text
detector > shadow score > safety guard > smoke matrix > train
```

即：

1. route heuristic 先当 shadow detector；
2. 只对明显自杀路线做 deterministic safety guard；
3. 每个修改必须有 TB metric / jsonl dump；
4. 失败后能知道是 checkpoint 坏、runtime 坏、route index 坏，还是模型策略坏；
5. 不允许“改完就训 6 小时才发现 schema 又错了”。

### 3.2 今晚不要做的长期项目

这些可以以后做，但今晚不做：

```text
MoE boss experts
long-horizon value head
full route planner 大重构
从 0 冷启动完整重训
大规模 imitation learning pipeline
继续加入大量新特征
重写 MCTS
```

原因：这些都不能在今晚快速验证 Act1 pass，且会引入新变量。

---

## 4. P0：运行时 / bridge 健康检查（必须先做）

> 目标：排除 bridge token / reset 崩溃，不把运行时问题误判成模型退化。

### 4.1 检查 bridge session / health

在 WSL 中执行：

```bash
cd /mnt/e/game/project/sts2_mcp/packages/rl-agent

./.venv-wsl-rocm/bin/python - <<'PY'
from sts2_env.bridge_client import BridgeClient

session = "/mnt/c/Users/yidhar/AppData/Roaming/SlayTheSpire2/bridge/session.json"
b = BridgeClient(session_file=session)

print("health:")
print(b.health())

try:
    s = b.get_state()
    print("state ok:", isinstance(s, dict), list(s.keys())[:10] if isinstance(s, dict) else type(s))
except Exception as e:
    print("state failed:", repr(e))
    raise
PY
```

### 4.2 P0 pass/fail

PASS：

```text
/health ok
/state ok
没有 401
没有 missing_or_invalid_token
```

FAIL：

```text
HTTP 401
missing_or_invalid_token
reset 连续失败
bridge session 文件不存在或 token 过期
```

FAIL 处理：

1. 重启 bridge/watchdog；
2. 确认 session.json 已刷新；
3. 重新跑 4.1；
4. P0 不过，不允许启动训练。

### 4.3 训练中 runtime stop gate

训练日志若出现：

```text
Bridge returned HTTP 401 for POST /env/reset
missing_or_invalid_token
```

连续 2 次以上，立即停止当前 smoke。
不要继续写入 bad run，不要把它算入模型评估。

---

## 5. P1：checkpoint / launch config bisect

> 目标：找出今晚可以继续的健康基点。
> 不要凭感觉用最新 ckpt。

### 5.1 候选 checkpoint

优先顺序如下：

| Case | Resume from | 目的 | 预期 |
|---|---|---|---|
| A | `muzero_phase3_bias0_baseline_20260509/muzero_step_00071735` | 只确认坏状态，不建议继续 | 应该 FAIL |
| B | `muzero_phase3_20260508_full_run_seed/muzero_step_00065730` | 判断 phase3 seed 末 ckpt 是否已坏 | 不确定 |
| C | `muzero_phase3_20260508_full_run_seed/muzero_step_00063520` | rollback 候选 | 可能好于 B |
| D | `muzero_phase3_20260508_full_run_seed/muzero_step_00061551` | 更早 rollback 候选 | 可能好于 C |
| E | `muzero_phase2_20260507_full_pool/muzero_step_00049158` | 最可能健康基点 | 首选恢复点 |
| F | `muzero_phase2_20260507_full_pool/muzero_step_00047125` | fallback | E 失败时用 |
| G | `muzero_phase2_20260507_full_pool/muzero_step_00045062` | fallback | F 失败时用 |

Windows 路径：

```text
E:\game\project\sts2_mcp\packages\rl-agent\checkpoints_muzero\...
```

WSL 路径：

```text
/mnt/e/game/project/sts2_mcp/packages/rl-agent/checkpoints_muzero/...
```

### 5.2 bisect smoke 原则

每个候选只跑 enough episodes，不训练太久：

```text
30 episodes：快速淘汰明显坏 ckpt
50 episodes：判断是否可用
64 episodes：进入正式 recent_tail 判断
```

每个候选必须：

```text
fresh optimizer
empty replay buffer
route_heuristic_bias = 0.0
same seed pool if possible: K8R3LFN7ZQ
full-run, not combat-sandbox
```

### 5.3 推荐今晚第一选择

除非 P1 smoke 证明 `00065730` 是健康的，否则今晚直接从：

```text
packages/rl-agent/checkpoints_muzero/muzero_phase2_20260507_full_pool/muzero_step_00049158
```

重启 full-run：

```text
resume-without-optimizer
resume-without-buffer
route_heuristic_bias=0.0
```

理由：

1. phase3/bias0 近端 ckpt 已表现为 route collapse；
2. phase2 full_pool 是进入 full-run 前较干净的基点；
3. 从 0 开始今晚太慢，无法覆盖 Act1 route/build/combat 数据分布；
4. fresh optimizer/buffer 可以去掉旧坏习惯和 bad replay。

### 5.4 P1 pass/fail 阈值

某个候选 ckpt 可用，必须满足：

```text
50 episode 内 act1_boss_seen_rate >= 0.20
death_floor median >= 12
best_minus_selected_mean_multi 不持续上升
top1_selected_rate_multi 不连续 5 ep 为 0
forced_elite_selected_rate <= 0.10
low_hp_elite_selected_rate <= 0.10
bridge/reset 无重复 401
```

强 PASS：

```text
64-window act1_boss_seen_rate >= 0.30
```

FAIL：

```text
50 episode 内 act1_boss_seen_rate < 0.10
last5 max_floor 多次 <= 6
best_minus_selected_mean_multi > 4.0 且 rising
```

---

## 6. P2：route action 对齐与 schema 回归检查

> 目标：确认模型选择的 action index、legal action、route summary、heuristic candidate 是同一个东西。
> 之前多次问题都来自字段语义误解，所以这一步必须做。

### 6.1 相关文件

```text
packages/rl-agent/muzero/train.py
packages/rl-agent/muzero/sts2_env/mcts.py
packages/rl-agent/sts2_env/route_heuristic.py
packages/rl-agent/tests/test_route_heuristic_phase2.py
packages/rl-agent/tests/test_route_heuristic_phase3_bias.py
```

### 6.2 必须新增/确认的测试

#### Test A：bias=0 完全 inert

断言：

```text
--route-heuristic-bias 0.0 时：
search/route/route_heuristic_bias_applied_rate = 0
MCTS 不设置 _pending_route_heuristic_bias
root prior / visit distribution 不受 route heuristic 影响
```

#### Test B：mixed legal actions index 对齐

构造 legal actions：

```python
[
  {"kind": "play_card", ...},
  {"kind": "travel", "coord": [1, 5], "action_id": "map_1"},
  {"kind": "end_turn", ...},
  {"kind": "travel", "coord": [3, 5], "action_id": "map_2"},
]
```

要求 route ranking 输出保持全量 index 对齐：

```python
[
  None,
  {"action_id": "map_1", "coord": [1, 5], ...},
  None,
  {"action_id": "map_2", "coord": [3, 5], ...},
]
```

禁止只返回 route-only compact list 后又拿 full legal index 去索引。

#### Test C：selected action 与 heuristic selected 是同一个动作

在 route decision 上记录：

```text
selected_full_index
selected_action_id
selected_coord
selected_kind
heuristic_rank_for_selected
heuristic_action_id_at_selected_index
heuristic_coord_at_selected_index
```

断言：

```text
selected_action_id == heuristic_action_id_at_selected_index
selected_coord == heuristic_coord_at_selected_index
selected_kind == "travel/map"
```

#### Test D：route_summary 字段存在但不能伪 100%

之前 Phase 0 出现过所有字段 100% 的 audit，必须确认不是“默认填零也算 present”。

测试逻辑：

```text
present_rate：字段真实来自 bridge/run_memory，而不是默认空 dict
dim_ok_rate：长度对，但还要 nonzero/variation rate
metadata_hit_rate：真实 card_id 命中 metadata，不是 unknown fallback
```

新增指标建议：

```text
route/summary_nonzero_rate
route/summary_variation_rate
deck/metadata_unknown_rate
deck/card_id_unknown_rate
```

### 6.3 建议直接跑的测试命令

WSL：

```bash
cd /mnt/e/game/project/sts2_mcp/packages/rl-agent

./.venv-wsl-rocm/bin/python -m pytest -q \
  tests/test_route_heuristic_phase2.py \
  tests/test_route_heuristic_phase3_bias.py
```

若新增了 route alignment test：

```bash
./.venv-wsl-rocm/bin/python -m pytest -q \
  tests/test_route_heuristic_phase2.py \
  tests/test_route_heuristic_phase3_bias.py \
  tests/test_route_action_alignment.py
```

### 6.4 P2 pass/fail

PASS：

```text
所有 route heuristic/unit tests 通过
route_action_index_alignment_error_rate = 0
full_vs_compact_legal_action_mismatch_rate = 0
bias=0 applied_rate = 0
route summary nonzero/variation 正常
```

FAIL：

```text
任何 action_id/coord/index mismatch
任何 bias=0 仍影响 MCTS
route summary 只有 present 没有 variation
```

FAIL 时：

```text
不允许训练；
先修 P2；
修完重跑 tests；
然后再 P1/P3。
```

---

## 7. P3：route safety guard（今晚允许做的最小代码改动）

> 目标：不要让模型在 full-run 中因为明显错误路线过早死在 floor 6–11。
> 注意：这不是 route bias，不是 prior 混合；这是 deterministic safety override，只在明显坏选择时介入。

### 7.1 为什么需要 guard

当前观察：

```text
act1_boss_seen 接近 0
death floor 常见 6–11
best_minus_selected_mean_multi 高且 rising
top1_selected_rate_multi collapse
forced_elite_selected_rate 不低
low_hp_elite_selected_rate 曾显著上升
```

这说明模型不是“boss 打不过”，而是：

```text
路线/早期资源管理导致根本见不到 boss
```

由于 tonight 目标是让训练分布覆盖 Act1 boss，必须避免 obvious route suicide。

### 7.2 风险分级

对每个 route candidate 打 `risk_class`：

| risk_class | 含义 | 例子 |
|---:|---|---|
| 0 | 安全路线 | 无 forced/immediate elite，有 rest/shop/event 支撑 |
| 1 | 可接受路线 | 有 optional elite，但可绕；或小怪链不长 |
| 2 | 中风险路线 | forced elite 但不是 immediate，且前面有 rest/shop |
| 3 | 高风险路线 | immediate elite，或 no-rest-before-elite |
| 4 | 极高风险路线 | low HP 下 forced/immediate elite，或连续怪后接 elite |

建议在 `route_heuristic.py` 的 breakdown 中增加：

```text
risk_class
risk_reason
safe_alternative_available
```

### 7.3 override 规则

只在 `decision_domain == "route"` 时启用。

伪代码：

```python
ranked = rank_legal_route_actions(...)
selected = ranked[selected_action_index]

safe_candidates = [
    c for c in ranked
    if c is not None and c["risk_class"] <= 1
]

if selected["risk_class"] >= 3 and safe_candidates:
    override_to = best_by_score_among_lowest_risk_class(safe_candidates)

if hp_ratio < 0.40:
    if selected has forced/immediate elite:
        candidates = no_elite_or_rest_shop_routes(ranked)
        if candidates:
            override_to = best_by_lowest_risk_then_score(candidates)
```

硬约束：

```text
只 override route action；
不 override combat/build/card reward；
不在只有一条路时 override；
不在所有候选都同样高风险时伪装安全；
override 后必须验证 action_id/coord/index 对齐；
```

### 7.4 guard 指标

必须写 TB scalar：

```text
route_safety_guard/applicable_rate
route_safety_guard/override_rate
route_safety_guard/safe_available_rate
route_safety_guard/selected_risk_class_mean
route_safety_guard/guarded_to_risk_class_mean
route_safety_guard/forced_when_safe_available_rate
route_safety_guard/immediate_elite_when_safe_available_rate
route_safety_guard/low_hp_forced_when_safe_available_rate
route_safety_guard/action_index_alignment_error_rate
```

### 7.5 guard pass/fail

PASS：

```text
action_index_alignment_error_rate = 0
forced_elite_selected_rate 下降
low_hp_elite_selected_rate 下降
act1_boss_seen_rate 上升
override_rate <= 0.25
```

FAIL：

```text
override_rate > 0.25
act1_boss_seen 没提升
death_floor 没提升
best_minus_selected_mean_multi 继续 rising
```

解释：

```text
override_rate > 25% 表示模型/route scoring 本身坏得太厉害；
guard 只能止血，不能长期替代策略学习。
```

---

## 8. P4：今晚推荐启动配置

### 8.1 首选配置

若 P0/P2 通过，且没有证据证明 `00065730` 健康，则用：

```text
resume_from = checkpoints_muzero/muzero_phase2_20260507_full_pool/muzero_step_00049158
fresh optimizer
empty replay buffer
route_heuristic_bias = 0.0
full-run
seed_pool = K8R3LFN7ZQ
route safety guard = enabled only after P2 pass
```

### 8.2 不要用的配置

```text
不要 resume phase3_bias0_baseline_20260509 的任何 ckpt
不要加载旧 optimizer
不要加载旧 replay buffer
不要 route_heuristic_bias > 0
不要在 P2 未过时启用 route guard
```

### 8.3 建议新建 launch script

复制：

```text
packages/rl-agent/launch_muzero_phase3_bias0_baseline_20260509.sh
```

新建：

```text
packages/rl-agent/launch_muzero_act1_recovery_20260510.sh
```

必须确认脚本里这些值：

```bash
RUN_NAME=muzero_act1_recovery_phase2_49158_20260510
RESUME_FROM=/mnt/e/game/project/sts2_mcp/packages/rl-agent/checkpoints_muzero/muzero_phase2_20260507_full_pool/muzero_step_00049158
ROUTE_HEURISTIC_BIAS_WEIGHT=0.0
```

训练参数必须包含：

```bash
--resume-from "$RESUME_FROM"
--resume-without-optimizer
--resume-without-buffer
--route-heuristic-bias 0.0
```

如果实现了 route safety guard，再加：

```bash
--route-safety-guard
```

或者等价环境变量/flag。
如果没有实现，不要伪装开启。

### 8.4 手动启动示例

WSL：

```bash
cd /mnt/e/game/project/sts2_mcp/packages/rl-agent
bash launch_muzero_act1_recovery_20260510.sh
```

如临时复用旧脚本，必须显式覆盖：

```bash
cd /mnt/e/game/project/sts2_mcp/packages/rl-agent

RUN_NAME=muzero_act1_recovery_phase2_49158_20260510 \
RESUME_FROM=/mnt/e/game/project/sts2_mcp/packages/rl-agent/checkpoints_muzero/muzero_phase2_20260507_full_pool/muzero_step_00049158 \
ROUTE_HEURISTIC_BIAS_WEIGHT=0.0 \
bash launch_muzero_phase3_bias0_baseline_20260509.sh
```

启动后第一件事检查 `run_config.txt`：

```bash
cat logs_muzero/muzero_act1_recovery_phase2_49158_20260510/run_config.txt
```

必须看到：

```text
resume_from = ...phase2_20260507_full_pool...00049158
route_heuristic_bias = 0.0
resume_without_optimizer = true
resume_without_buffer = true
```

---

## 9. P5：监控指标清单

### 9.1 每 30 分钟必须看

```text
episode/reward
episode/max_floor
episode/death_floor
episode/act1_boss_seen
episode/act1_pass_rate
recent_tail/64/act1_boss_seen_rate
recent_tail/64/act1_pass_rate
recent_tail/256/win_rate
```

### 9.2 route 必看

```text
route_heuristic/top1_selected_rate_multi
route_heuristic/top2_selected_rate_multi
route_heuristic/best_minus_selected_mean_multi
route_heuristic/forced_elite_selected_rate
route_heuristic/low_hp_elite_selected_rate
route_heuristic/no_rest_before_elite_selected_rate
route_heuristic/selected_forced_elite_count_mean
route_heuristic/best_forced_elite_count_mean
```

如果启用 safety guard：

```text
route_safety_guard/applicable_rate
route_safety_guard/override_rate
route_safety_guard/safe_available_rate
route_safety_guard/forced_when_safe_available_rate
route_safety_guard/action_index_alignment_error_rate
```

### 9.3 确认 route bias 没偷偷开启

必须持续为 0：

```text
search/route/route_heuristic_bias_applied_rate = 0
search/route/bias_* = 0
route_heuristic/bias_applied_rate = 0
```

### 9.4 combat 机制回归必看

这些是之前修过的坑，今晚不能回归：

```text
boss_combat/wasteful_end_turn_selected_rate = 0
x_cost_zero_bad_p0 = 0
hp_cost_self_lethal_selected_rate = 0
hp_cost_low_margin_selected_rate 不应持续升高
kaiser/facing_change_candidate 非 0 时 selected 不应长期 0
insatiable/frantic_escape missed_at_1 = 0
potion_timing/bad_selected_rate 不应升高
```

### 9.5 loss 健康

```text
loss/total
loss/policy
loss/value
loss/reward
loss/future_world_aux
loss/future_bank_state
loss/future_bank_delta
loss/jepa_next_hidden
loss/surprise
loss/latent_gaussian_reg
```

允许：

```text
孤立 spike，随后恢复
```

不允许：

```text
future_world_aux / future_bank_state 连续 spike
median 明显抬升
policy/value/reward 同时恶化
```

---

## 10. full-run 数据分布问题：为什么必须先恢复 Act1 coverage

当前 full-run 一个 Act1 大致分布：

```text
weak combat: 2–3
normal combat: 5–8
elite: 0–2
boss: 0–1
event: 若干
shop: 0–2
card reward: 约 8–10 次
```

进入 Act2/Act3 后，normal combat 仍是大头，而 boss 总共固定只有 3 场。

这意味着：

1. 如果模型死在 floor 6–11，几乎没有 Act1 boss 数据；
2. boss-specific 修得再多也没有训练分布覆盖；
3. route/build/card reward 的错误会放大，因为它们决定后面能不能见到 boss；
4. replay buffer 很容易被 early-death normal/combat 数据淹没；
5. 继续从坏 buffer 学，会强化“早死但局部战斗还行”的策略。

今晚所以要：

```text
fresh buffer
rollback healthy ckpt
route safety guard 防早死
先把 act1_boss_seen 拉回 >=0.30
```

等 Act1 coverage 恢复后，再谈：

```text
boss experts
long-horizon value
imitation learning
Act2/Act3 curriculum
```

---

## 11. 如果今晚要加最小 imitation 辅助

这不是今晚主线，但如果 route 仍不恢复，可以做最小版：

### 11.1 只录 route/build，不先录全 combat

优先录：

```text
map route decisions
card reward picks
shop buy/remove
rest/smith decisions
```

这些决定 Act1 是否见 boss，且比 combat imitation 更稳定。

### 11.2 不直接混入在线训练，先做验证集

先产出：

```text
human_demo/act1_route_build_*.jsonl
```

每条记录：

```json
{
  "screen": "map/card_reward/shop/rest",
  "floor": 7,
  "hp": 31,
  "max_hp": 75,
  "deck_summary": "...",
  "legal_actions": [...],
  "chosen_action_id": "...",
  "chosen_index": 3,
  "reason": "avoid forced elite before rest"
}
```

用途：

```text
离线检查模型 top-k 是否包含人工动作
训练 route/build auxiliary imitation loss
生成 golden regression tests
```

不要今晚一边录一边直接强行训练，变量太多。

---

## 12. 执行任务清单（给 Claude）

### Task 0：确认当前状态，不改代码

- [ ] 读取最近 logs：
  - `muzero_phase3_bias0_baseline_20260509`
  - `muzero_phase3_bias005_v2heur_smoke_20260509`
  - `muzero_phase3_20260508_full_run_seed`
- [ ] 确认 bad baseline 指标与本文一致。
- [ ] 确认没有正在运行的训练会写入同一 run dir。

输出：

```text
当前 active run / PID / 最新 event mtime / 最新 ckpt
```

### Task 1：P0 bridge health

- [ ] 跑 4.1 health check。
- [ ] 若 401，修 bridge/watchdog/session。
- [ ] 通过前不允许训练。

输出：

```text
P0 PASS/FAIL
health payload 摘要
```

### Task 2：P2 route alignment tests

- [ ] 检查 `route_heuristic.py` ranking 是否保持 full legal action index。
- [ ] 检查 `train.py` selected/best 统计是否用同一 index 空间。
- [ ] 检查 `mcts.py` bias=0 是否完全 inert。
- [ ] 新增或补齐 unit tests。
- [ ] 跑 pytest。

输出：

```text
测试文件
pytest 结果
是否发现 index/schema mismatch
```

### Task 3：实现 route safety guard（如果 P2 通过）

- [ ] 在 `route_heuristic.py` 增加 risk_class。
- [ ] 在 action select 后、env step 前增加 guard hook。
- [ ] 只对 route decision 生效。
- [ ] 增加 TB metrics。
- [ ] 增加 unit test：有 safe candidate 时高风险路线会被 override。

输出：

```text
修改文件列表
新增指标列表
pytest 结果
```

### Task 4：建立 Act1 recovery launch script

- [ ] 新建 `launch_muzero_act1_recovery_20260510.sh`。
- [ ] resume from phase2 `00049158`。
- [ ] fresh optimizer / empty buffer。
- [ ] bias=0。
- [ ] guard flag 只在实现并测试通过后开启。

输出：

```text
脚本路径
run_name
resume_from
确认 run_config 关键字段
```

### Task 5：50 episode smoke

- [ ] 启动 recovery run。
- [ ] 等 30–50 episodes。
- [ ] 读取 TB / stdout。
- [ ] 按 5.4 判断。

输出表：

| metric | value | pass/fail |
|---|---:|---|
| act1_boss_seen_rate_50 | | |
| recent_tail/64/act1_boss_seen_rate | | |
| death_floor median | | |
| top1_selected_rate_multi | | |
| best_minus_selected_mean_multi | | |
| forced_elite_selected_rate | | |
| low_hp_elite_selected_rate | | |
| bridge 401 count | | |

### Task 6：今晚继续/回滚决策

若 PASS：

```text
继续训练到 64/128 window；
目标 recent_tail/64/act1_boss_seen_rate >= 0.30；
观察是否出现 Act1 clear。
```

若 FAIL：

```text
停止当前 run；
不要保存为新基线；
回滚到 00047125 或 00045062；
重复 P1/P5；
如果 phase2 全失败，则检查 full-run env/schema，而不是继续训。
```

---

## 13. 今晚推荐决策树

```text
Start
 |
 |-- P0 bridge health fail?
 |      |-- yes -> fix bridge/session, no training
 |      '-- no
 |
 |-- P2 route index/schema tests fail?
 |      |-- yes -> fix tests/code, no training
 |      '-- no
 |
 |-- phase3 00065730 smoke >= 0.20 boss_seen?
 |      |-- yes -> can resume from 00065730 fresh opt/buffer, bias=0
 |      '-- no
 |
 |-- phase2 00049158 smoke >= 0.20 boss_seen?
 |      |-- yes -> main recovery run from 00049158
 |      '-- no
 |
 |-- phase2 00047125 / 00045062 smoke?
 |      |-- pass -> use earliest healthy
 |      '-- fail -> full-run env/schema regression, stop training and debug
```

---

## 14. 训练继续后的判断方式

### 14.1 不要只看 win_rate

当前 `win_rate_256` 容易混入：

```text
normal combat win
elite win
boss win
short episode survival artifact
```

今晚主要看：

```text
act1_boss_seen_rate
act1_pass_rate
death_floor
max_floor
route bad-choice metrics
```

### 14.2 win_rate 解释

如果：

```text
normal/elite win_rate 下降
act1_boss_seen 上升
death_floor 上升
episode length 上升
```

可能不是坏事：模型从短战早死转向更长路线，遇到更难战斗。
但如果：

```text
normal/elite win_rate 下降
act1_boss_seen 下降
death_floor 下降
max_floor 下降
```

就是整体退化，必须停。

---

## 15. 最终今晚要交付的东西

### 15.1 必交付

```text
1. P0 bridge health 结果
2. route alignment / bias=0 inert tests 结果
3. recovery launch script
4. 50 episode smoke 表格
5. 继续/回滚决策
```

### 15.2 如果代码有修改，必须列出

```text
modified files
new files
tests run
metrics added
known risk
```

### 15.3 不允许的交付

```text
只说“看起来可能是噪声”
只继续训练不给 stop gate
没有确认 run_config 就启动
没有 route index/schema test 就启用 guard
继续调 route heuristic bias
从 bad baseline ckpt 继续
```

---

## 16. 一页版行动摘要

今晚不要从 0 重训，也不要从最新坏 ckpt 继续。
按下面做：

```text
1. P0：确认 bridge health，无 401。
2. P2：确认 route action index/schema 对齐，bias=0 完全 inert。
3. 禁用 route heuristic bias。
4. 可选启用 deterministic route safety guard，只拦明显自杀路线。
5. 从 phase2_full_pool/muzero_step_00049158 fresh optimizer + empty buffer 启 full-run。
6. 50 episode 检查 act1_boss_seen_rate。
7. >=0.20 继续，>=0.30 算今晚恢复成功；<0.10 停止并回滚 00047125/00045062。
```

核心判断：

```text
今晚不是“让模型突然学会所有 boss”；
今晚是先修复 full-run 分布，让 replay 重新覆盖 Act1 boss。
```
