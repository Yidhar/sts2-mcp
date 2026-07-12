# Phase 2 Route Heuristic 代码 Review

日期：2026-05-08
范围：Phase 2 route heuristic dry-run 实现审查
结论：**Phase 2 主体方向正确，但当前不能视为完成，也不能进入 Phase 3 prior bias。**

---

## 1. 总结结论

当前实现已经满足最基础的 dry-run 安全要求：

- `route_heuristic.py` 只计算 score / breakdown。
- `train.py` 是在模型已经选出 `action_idx` 后记录 dry-run 结果。
- 当前没有修改：
  - `action_idx`
  - `action_mask`
  - policy prior
  - MCTS bias

所以 **Phase 2 不会直接改变模型决策**。

但是当前版本最大问题是：**诊断指标口径不可靠**。

主要问题：

1. 缺少 `route_heuristic/available_rate`。
2. 多个名字叫 `available` 的指标实际统计的是 selected action。
3. 缺少设计文档要求的关键指标：
   - `unsafe_elite_selected_rate`
   - `forced_elite_selected_rate`
   - `no_rest_before_elite_selected_rate`
4. 单候选路线会污染 `top1_selected_rate / top2_selected_rate`。
5. 缺 route_summary runtime 使用率诊断。
6. `try/except Exception: pass` 会让 Phase 2 失效时静默消失。
7. 当前测试在 Windows Python 环境下无法跑通，原因是 `sts2_env/__init__.py` 强制 import torch。

因此当前状态应判定为：

```text
核心 helper：部分通过
dry-run 安全性：通过
per-action scoring：基本通过
TB 指标完整性：不通过
TB 指标口径：不通过
测试可运行性：不通过，当前环境被 torch import 阻塞
可进入 Phase 3 prior bias：不可以
```

---

## 2. 审查文件

重点检查了：

```text
packages/rl-agent/sts2_env/route_heuristic.py
packages/rl-agent/muzero/train.py
packages/rl-agent/sts2_env/deck_quality.py
packages/rl-agent/sts2_env/_sim_translate.py
packages/rl-agent/sts2_env/observation_common.py
packages/rl-agent/scripts/phase0_audit.py
packages/rl-agent/tests/test_route_heuristic_phase2.py
packages/rl-agent/tests/test_deck_quality_v2_phase1.py
```

相关设计文档：

```text
docs/muzero-route-deck-long-horizon-review-20260508.md
```

---

## 3. 已验证通过的部分

### 3.1 `route_heuristic.py` 是纯 dry-run helper

文件：

```text
packages/rl-agent/sts2_env/route_heuristic.py
```

主要函数：

```python
score_route_action(...)
rank_legal_route_actions(...)
```

当前行为：

- 对 route action 计算 score。
- 返回 breakdown。
- 不修改输入 action。
- 不修改 action mask。
- 不接入 policy prior。
- 不接入 MCTS bias。

这符合 Phase 2 的要求。

---

### 3.2 没有实现 full path planner，符合设计要求

设计文档要求 Phase 2 第一版不要写 full path planner，而是使用现有 per-action `route_summary`。

当前实现使用：

```python
route_summary=action.get("route_summary")
```

位置：

```text
packages/rl-agent/sts2_env/route_heuristic.py
```

这一点正确。

---

### 3.3 `train.py` 目前没有影响真实选择

文件：

```text
packages/rl-agent/muzero/train.py
```

当前逻辑是：

1. 模型 / planner 先选出 `action_idx`。
2. 再根据当前 legal map actions 计算 route heuristic ranking。
3. 最后把 selected 与 heuristic best 做对比，写入 dry-run records。

相关代码位置：

```text
packages/rl-agent/muzero/train.py:7347-7418
```

因此当前 Phase 2 没有改变模型行为。

---

## 4. 测试情况

### 4.1 语法检查通过

执行：

```powershell
python -m py_compile sts2_env\route_heuristic.py sts2_env\deck_quality.py
```

结果：通过。

---

### 4.2 pytest 当前无法运行到逻辑测试

执行：

```powershell
python -m pytest tests/test_route_heuristic_phase2.py tests/test_deck_quality_v2_phase1.py -q
```

结果：collection 阶段失败。

错误原因：

```text
ModuleNotFoundError: No module named 'torch'
```

触发路径：

```text
tests/test_route_heuristic_phase2.py
  -> from sts2_env.route_heuristic import ...
  -> sts2_env/__init__.py
  -> from .aux_maskable_ppo import AuxMaskablePPO
  -> import torch
```

问题不是 Phase 2 测试逻辑失败，而是：

```text
sts2_env/__init__.py 无条件 import torch-heavy 模块，
导致轻量 helper 单测也必须依赖 torch。
```

建议后续处理：

1. 在有 torch 的训练环境中重新跑测试；或
2. 把 `sts2_env/__init__.py` 改成 lazy import，避免轻量模块测试强依赖 torch。

---

## 5. P0 阻塞问题

这些问题必须修，否则 Phase 2 的 TensorBoard 指标会误导判断。

---

### P0-1：缺少 `route_heuristic/available_rate`

设计文档要求：

```text
route_heuristic/available_rate
```

但当前 `train.py` 没有 emit 这个指标。

当前 emit 的是：

```text
route_heuristic/decision_count
route_heuristic/top1_selected_rate
route_heuristic/top2_selected_rate
route_heuristic/selected_score_mean
route_heuristic/best_score_mean
route_heuristic/best_minus_selected_mean
route_heuristic/low_hp_elite_selected_rate
route_heuristic/low_hp_elite_seen_rate
route_heuristic/rest_before_elite_available_rate
route_heuristic/rest_selected_when_low_hp_rate
route_heuristic/shop_with_gold_available_rate
route_heuristic/shop_selected_high_gold_rate
```

相关代码：

```text
packages/rl-agent/muzero/train.py:8116-8128
```

问题：

如果 Phase 2 整体没有工作，或者 route_summary 缺失，当前逻辑可能只是没有 records / 没有 scalar，而不是明确告诉你：

```text
available_rate = 0
```

这会导致排查困难。

#### 修复要求

需要显式统计：

```python
route_decision_total
route_map_decision_total
route_heuristic_available_count
route_summary_used_count
route_multi_candidate_count
```

然后 emit：

```text
route_heuristic/available_rate
route_heuristic/summary_used_rate
route_heuristic/map_candidate_count_mean
route_heuristic/multi_candidate_rate
```

---

### P0-2：`available` 指标实际统计的是 selected

当前代码：

```python
rest_before_elite_avail = sum(
    1 for r in route_dry_run_records
    if r["selected_rest_before_elite_available"] > 0.5
)

shop_with_gold_avail = sum(
    1 for r in route_dry_run_records
    if r["selected_has_shop"] > 0.5 and r["gold"] >= 75
)
```

位置：

```text
packages/rl-agent/muzero/train.py:8110-8113
```

然后 emit：

```text
route_heuristic/rest_before_elite_available_rate
route_heuristic/shop_with_gold_available_rate
```

问题：

指标名里的 `available` 应该表示：

```text
当前 legal map candidates 中是否存在这种路线。
```

但当前统计的是：

```text
selected action 自己是不是这种路线。
```

这会造成严重误判。

#### 示例

当前有三条路线：

```text
A: elite, no rest
B: rest before elite
C: shop
```

模型选择 A。

正确统计应该是：

```text
rest_before_elite_available_rate = 1
shop_with_gold_available_rate = 1 if gold enough
rest_selected_low_hp_rate = 0
shop_selected_high_gold_rate = 0
```

当前实现会统计成：

```text
rest_before_elite_available_rate = 0
shop_with_gold_available_rate = 0
```

这会让人误以为好路线不存在。

#### 修复要求

每个 route decision record 需要同时记录：

```python
candidate_rest_before_elite_available = any(...)
candidate_shop_with_gold_available = any(...)

selected_rest_before_elite = ...
selected_has_shop = ...
```

指标应拆开：

```text
route_heuristic/rest_before_elite_available_rate
route_heuristic/rest_before_elite_selected_rate
route_heuristic/shop_with_gold_available_rate
route_heuristic/shop_selected_high_gold_rate
```

---

### P0-3：缺少 `unsafe_elite_selected_rate`

设计文档要求：

```text
route_heuristic/unsafe_elite_selected_rate
```

当前没有。

当前只有：

```text
route_heuristic/low_hp_elite_selected_rate
route_heuristic/low_hp_elite_seen_rate
```

位置：

```text
packages/rl-agent/muzero/train.py:8122-8123
```

问题：

`low_hp_elite_selected_rate` 只能说明：

```text
低 HP 时模型选了带 elite 的路线。
```

但不能说明：

```text
模型是否选择了 heuristic 认为 unsafe 的 elite 路线。
```

强 deck、高血、有药水时选择 elite 可能是正确的。所以必须区分：

```text
elite selected
unsafe elite selected
low HP elite selected
forced elite selected
no-rest-before-elite selected
```

#### 修复要求

新增：

```text
route_heuristic/unsafe_elite_selected_rate
route_heuristic/forced_elite_selected_rate
route_heuristic/no_rest_before_elite_selected_rate
```

每条 record 需要保存 selected breakdown：

```python
selected_unsafe_elite_penalty
selected_forced_elite_penalty
selected_no_rest_before_elite_penalty
selected_elite_risk_factor
```

---

### P0-4：`rest_selected_low_hp_rate` 指标名字不匹配

设计文档要求：

```text
route_heuristic/rest_selected_low_hp_rate
```

当前代码 emit：

```text
route_heuristic/rest_selected_when_low_hp_rate
```

位置：

```text
packages/rl-agent/muzero/train.py:8126
```

这会导致 dashboard / 分析脚本按设计文档找不到指标。

#### 修复要求

建议双写：

```text
route_heuristic/rest_selected_low_hp_rate
route_heuristic/rest_selected_when_low_hp_rate
```

后一项用于兼容已有日志，前一项与文档保持一致。

---

### P0-5：单候选路线污染 `top1_selected_rate / top2_selected_rate`

当前逻辑：

```python
if len(scores) >= 2:
    ...
else:
    top2_idx = top1_idx
```

位置：

```text
packages/rl-agent/muzero/train.py:7392-7398
```

如果当前地图只有一个 legal map action，那么：

```text
top1_selected_rate = 1
top2_selected_rate = 1
best_minus_selected = 0
```

这会把 forced route 决策伪装成：

```text
模型与 heuristic 完全一致。
```

#### 修复要求

新增：

```text
route_heuristic/candidate_count_mean
route_heuristic/multi_candidate_rate
route_heuristic/top1_selected_rate_multi
route_heuristic/top2_selected_rate_multi
route_heuristic/best_minus_selected_mean_multi
```

全量指标可以保留，但进入 Phase 3 前主要看 multi-candidate 指标。

---

### P0-6：缺 route_summary 使用率诊断

`score_route_action()` 对缺失 route_summary 会 safe-zero：

```python
if not isinstance(route_summary, dict):
    return _zero_score()
```

`_zero_score()` 返回：

```python
"summary_used": False
```

但当前 `train.py` 没有把 `summary_used` 聚合成指标。

问题：

即使所有 route_summary 都缺失，`rank_legal_route_actions()` 仍可能返回 safe-zero breakdown。这样 dry-run 指标看起来存在，但实际 score 全部没有使用 route_summary。

#### 修复要求

每条 record 应记录：

```python
summary_used_candidate_count
summary_used_selected
summary_used_best
```

并 emit：

```text
route_heuristic/summary_used_rate
route_heuristic/selected_summary_used_rate
route_heuristic/best_summary_used_rate
```

---

### P0-7：`try/except Exception: pass` 会静默吞掉 Phase 2 失效

当前有两处：

```python
except Exception:
    pass
```

位置：

```text
packages/rl-agent/muzero/train.py:7417-7418
packages/rl-agent/muzero/train.py:8129-8130
```

dry-run 不应该中断训练，这点正确。但完全静默会导致：

- import 错误无日志
- schema 错误无日志
- route_summary 错误无日志
- deck_quality 错误无日志
- TensorBoard 指标消失但原因未知

#### 修复要求

保留不 crash，但至少 emit：

```text
route_heuristic/error_count
route_heuristic/record_exception_count
route_heuristic/emit_exception_count
```

建议加 throttled log：

```python
if self.episode_count % 100 == 0:
    print("[route_heuristic] dry-run failed:", repr(exc))
```

---

## 6. P1 中等风险问题

这些不是立即阻塞 Phase 2 dry-run，但进入 Phase 3 prior bias 前需要修或校准。

---

### P1-1：完全线性 elite 路线可能漏判 forced penalty

`_sim_translate.py` 中：

```python
forced_steps = None
...
if forced_steps is None and child_count >= 2:
    forced_steps = max(0, depth - 1)
```

如果整条路线完全没有分叉：

```text
forced_path_steps_before_branch = None
```

但 `_forced_elite_penalty()` 中：

```python
forced = _safe_int(summary.get("forced_path_steps_before_branch"))
```

`None` 会变成 0，导致完全线性 elite 路线不触发 forced penalty。

它仍然可能被：

```text
unsafe_elite_penalty
no_rest_before_elite_penalty
```

惩罚，但 `forced_elite_penalty` 这个诊断会漏。

#### 修复建议

如果：

```python
forced_path_steps_before_branch is None
and next_elite_steps is not None
```

应视作 locked path。

可以在 heuristic 里处理：

```python
if forced_raw is None and next_elite_steps is not None:
    return 0.6
```

或者在 route_summary 构建时把完全线性路径的 forced steps 设为 `max_depth`。

---

### P1-2：`count_elite` 是 subtree elite 数，可能过度惩罚可选 elite

当前：

```python
n_elites = _safe_float(summary.get("count_elite"))
return n_elites * elite_risk * 1.0
```

问题：

`route_summary` 是从候选下一节点开始 BFS 整个 reachable subtree。`count_elite` 不是必经 elite 数，而是未来子树里所有可达 elite 数。

这可能导致：

```text
高分叉、高资源路线因为 subtree 有 elite 被过度扣分。
```

Phase 2 dry-run 可以接受，但 Phase 3 不能直接把这个分数接成 prior bias。

#### Phase 3 前建议拆分

```text
immediate_elite_risk
next_elite_steps_risk
forced_elite_risk
optional_elite_count
subtree_elite_count
no_rest_before_elite_risk
```

真正影响 prior 的应该更偏向：

```text
immediate / forced / no-rest-before-elite
```

而不是裸 `count_elite`。

---

### P1-3：`branch_value` 可能过大

当前：

```python
raw = 0.05 * reachable + 0.20 * direct_children
```

问题：

`direct_child_count` 当前来自：

```python
map_parent_child_count = len(map_options_by_index)
```

这很可能是当前节点的 map options 数量，而不是选择某个 child 后的分支数量。对同一步的所有 candidate 来说，它大概率是常数。

另外：

```text
reachable_node_count 越大，branch_value 越大。
```

这可能奖励巨大但危险的 subtree。

#### 建议

Phase 2 先观察，不急着删。

但需要新增 breakdown 指标：

```text
route_heuristic/selected_branch_value_mean
route_heuristic/best_branch_value_mean
route_heuristic/selected_unsafe_elite_penalty_mean
route_heuristic/best_unsafe_elite_penalty_mean
```

如果发现 heuristic 总是偏向大 subtree，需要对 branch value 加 cap，例如：

```python
branch_value = min(branch_value, 0.8)
```

---

### P1-4：low HP 阈值不统一

`route_heuristic.py` 使用：

```python
low_hp_flag = 1.0 if hp_ratio < 0.40 else 0.0
```

`train.py` 的 rest metric 使用：

```python
hp_ratio < 0.50
```

这会让：

```text
low_hp_elite_selected_rate
rest_selected_when_low_hp_rate
```

使用不同 low HP 定义。

这不是必然错误，但需要明确。

#### 建议

定义常量：

```python
ROUTE_LOW_HP_RATIO = 0.40
ROUTE_REST_URGENCY_HP_RATIO = 0.50
```

或者统一阈值。

---

### P1-5：Phase 2 potion count 判断偏弱

当前：

```python
p.get("title") and p.get("title") != "[empty]"
```

问题：

之前 potion unused 问题已经证明 potion slot 可能有：

```text
{empty: true}
"empty"
"none"
"null"
""
"[empty]"
```

当前逻辑只排除 `[empty]`，可能把空 slot 当作 potion。

这会影响 elite risk：

```python
if potion_count <= 0:
    risk += 0.2
```

如果空 slot 被误算成药水，heuristic 会低估 elite 风险。

#### 修复建议

复用统一 helper：

```python
def _count_non_empty_potions(potions):
    ...
```

规则应排除：

```text
empty: true
title/id/name in {"", "empty", "none", "null", "[empty]"}
```

---

## 7. deck_quality 相关问题

Phase 2 依赖 `deck_quality_v2_from_obs()`，因此也检查了 `deck_quality.py`。

---

### P1-6：文档说接入 run_memory，但实际没有

`deck_quality.py` 文件头写：

```text
This module is consumed by run_memory._build_profile()
```

但实际 `run_memory.py` 没有 import/use `deck_quality_v2`。

当前 `deck_quality_v2` 主要用于：

1. Phase 2 route heuristic dry-run；
2. episode end `deck_quality/*` logging。

不是模型输入特征。

#### 建议

修改注释，避免误导：

```text
Phase 1/2 only computes + logs deck_quality_v2 and feeds route heuristic dry-run.
It is not yet wired into run_memory/model observation.
```

---

### P1-7：`_DRAW_TAGS` 定义了但没有使用

当前有：

```python
_DRAW_TAGS = frozenset({"draw", "card_draw", "draw_cards"})
```

但 draw 只读取：

```python
signals.get("draw")
```

如果某些卡只有 tag 没有 signal，会低估 draw。

#### 修复建议

增加 fallback：

```python
if draw > 0:
    draw_total += draw
    draw_cards += 1
elif tags & _DRAW_TAGS:
    draw_total += 1.0
    draw_cards += 1
```

这仍然使用 typed metadata tag，不是文本正则。

---

### P1-8：`consistency_score` 注释和公式相反

注释含义：

```text
10 unique cards has perfect 1.0
```

公式：

```python
consistency_score = 1.0 - (unique_count - 1) / deck_size
```

实际效果：

```text
unique 越少，score 越高。
重复越多，score 越高。
```

当前 route heuristic 没用 `consistency_score`，所以不是 Phase 2 阻塞。但后续如果把 deck_quality 接入模型，需要修正注释或公式。

---

## 8. 关于 `_current_raw_combat_obs()`

函数名看起来像只返回 combat obs：

```python
def _current_raw_combat_obs(self) -> dict[str, Any] | None:
```

但实际实现是：

```python
raw_obs = getattr(getattr(self.env, "unwrapped", self.env), "_last_obs_raw", None)
return raw_obs if isinstance(raw_obs, dict) else None
```

也就是说它只是返回当前 `_last_obs_raw`，不是真的 combat-only。

因此当前 route dry-run 使用它 **不是立即 bug**。

但建议重命名为：

```python
_current_raw_obs()
```

否则后续维护者容易误判。

---

## 9. 关于 Phase 0 100% 是否有写漏洞

检查了：

```text
packages/rl-agent/scripts/phase0_audit.py
```

当前有样本量 gate：

```python
MIN_MAP_ACTIONS_FOR_PASS = 150
MIN_MAP_STEPS_FOR_PASS = 50
MIN_UNIQUE_CARDS_FOR_PASS = 12
```

并且 route / unique deck 相关 pass 会检查 sample sufficient。

所以从代码看，Phase 0 audit 不是简单的零分母 100%。

但是：

```text
Phase 0 PASS 只能证明 route_summary/schema 字段存在。
它不能证明 Phase 2 dry-run 指标口径正确。
```

当前 Phase 2 的指标聚合仍然需要修。

---

## 10. 建议修复任务清单

### Task A：重构 route dry-run record

每条 route decision 至少记录：

```python
{
    "route_decision": 1,
    "map_candidate_count": int,
    "summary_used_candidate_count": int,
    "available": bool,
    "multi_candidate": bool,

    "selected_idx": int,
    "top1_idx": int | None,
    "top2_idx": int | None,

    "selected_score": float | None,
    "best_score": float | None,
    "best_minus_selected": float | None,

    "selected_unsafe_elite": bool,
    "selected_low_hp_elite": bool,
    "selected_forced_elite": bool,
    "selected_no_rest_before_elite": bool,

    "candidate_rest_before_elite_available": bool,
    "selected_rest_before_elite": bool,

    "candidate_shop_with_gold_available": bool,
    "selected_shop_high_gold": bool,

    "hp_ratio": float,
    "gold": float,
}
```

---

### Task B：补齐 TB 指标

必须新增或修正：

```text
route_heuristic/available_rate
route_heuristic/summary_used_rate
route_heuristic/candidate_count_mean
route_heuristic/multi_candidate_rate

route_heuristic/top1_selected_rate
route_heuristic/top2_selected_rate
route_heuristic/top1_selected_rate_multi
route_heuristic/top2_selected_rate_multi

route_heuristic/selected_score_mean
route_heuristic/best_score_mean
route_heuristic/best_minus_selected_mean
route_heuristic/best_minus_selected_mean_multi

route_heuristic/unsafe_elite_selected_rate
route_heuristic/low_hp_elite_selected_rate
route_heuristic/forced_elite_selected_rate
route_heuristic/no_rest_before_elite_selected_rate

route_heuristic/rest_before_elite_available_rate
route_heuristic/rest_before_elite_selected_rate
route_heuristic/rest_selected_low_hp_rate

route_heuristic/shop_with_gold_available_rate
route_heuristic/shop_selected_high_gold_rate

route_heuristic/error_count
route_heuristic/record_exception_count
route_heuristic/emit_exception_count
```

---

### Task C：修正 available / selected 口径

必须确保：

```text
available = legal candidates 里存在
selected = 模型实际选择了
```

不能再用 selected action 字段去命名 available 指标。

---

### Task D：修 potion count

新增统一 helper，排除：

```text
empty: true
title/id/name in {"", "empty", "none", "null", "[empty]"}
```

然后 route heuristic dry-run 使用这个 helper。

---

### Task E：异常可观测化

把：

```python
except Exception:
    pass
```

改成：

```python
except Exception as exc:
    route_heuristic_error_count += 1
    maybe_throttled_log(exc)
```

训练不能 crash，但错误必须可见。

---

### Task F：修 forced linear elite path

如果：

```python
forced_path_steps_before_branch is None
and next_elite_steps is not None
```

应视作 forced / locked route。

---

### Task G：Phase 3 前校准 subtree elite penalty

进入 Phase 3 前，不要直接把当前 score 作为 prior bias。

需要先确认：

```text
subtree count_elite 是否过度惩罚 optional elite。
branch_value 是否过度奖励大 subtree。
```

建议新增 breakdown 均值指标。

---

### Task H：修测试环境问题

二选一：

1. 在有 torch 的训练环境中跑 Phase 2 tests；或
2. 改 `sts2_env/__init__.py` 为 lazy import，让 `route_heuristic.py / deck_quality.py` 这种轻量模块可独立测试。

---

## 11. Phase 2 修完后的验收标准

修完后必须满足：

```text
1. route_heuristic/available_rate 能稳定出现。
2. route_heuristic/summary_used_rate > 0.98。
3. route_heuristic/candidate_count_mean 合理。
4. route_heuristic/multi_candidate_rate 非 0。
5. top1/top2 指标区分全量与 multi-candidate。
6. available 指标真正来自 legal candidates，不是 selected action。
7. unsafe_elite / forced_elite / no_rest_before_elite 指标全部出现。
8. best_minus_selected_mean 能揭示模型路线失误。
9. error_count 长期为 0。
10. Phase 2 单测能在目标环境跑通。
```

进入 Phase 3 prior bias 前，还需要证明：

```text
heuristic 高分路线与 survival / act1_boss_seen / act1_clear 正相关。
```

如果没有这个正相关，不应该把 heuristic 接入 prior。

---

## 12. 最重要的修复顺序

建议按以下顺序处理：

```text
1. 修 available vs selected 指标口径。
2. 新增 available_rate / summary_used_rate / multi_candidate_rate。
3. 补 unsafe_elite / forced_elite / no_rest_before_elite 指标。
4. 修 rest_selected_low_hp_rate 命名。
5. 修 potion count helper。
6. 给 Phase 2 try/except 加 error_count。
7. 修 forced linear elite path。
8. 跑 Phase 2 tests。
9. 观察至少数小时 dry-run 指标。
10. 再决定是否进入 Phase 3 prior bias。
```

---

## 13. 一句话结论

当前 Phase 2 **不是策略层危险**，因为它还没有影响 action selection；但它是 **观测层危险**，因为多个指标口径会误导你判断模型路线能力。

必须先把 dry-run metrics 修准，再考虑 Phase 3。
