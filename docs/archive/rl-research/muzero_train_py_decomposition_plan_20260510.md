# MuZero `train.py` 拆分方案（2026-05-10）

## 目标

`packages/rl-agent/muzero/train.py` 曾经膨胀到 **16k+ 行**，同时承担 CLI、训练主循环、搜索调度、战斗策略启发式、Boss 特化规则、route heuristic、TensorBoard 指标、异常 dump 等职责。当前已完成多批低风险拆分，`train.py` 已降到 **25 行**，只保留兼容入口与旧导入 re-export；Trainer core 已迁往 `muzero/training/trainer.py`，CLI 主循环已迁往 `muzero/training/cli_main.py`，CLI 参数声明已迁往 `muzero/training/cli_args.py`，async actor TensorBoard replay 已迁往 `muzero/training/async_telemetry.py`，combat/action runtime adapter 已迁往 `muzero/training/combat_runtime_features.py`。后续仍需拆 `muzero/sts2_env/*` 等历史债务，但禁止再把策略逻辑写回 `train.py`。继续把策略逻辑写进 `train.py` 会导致：

1. **策略互相污染**：例如 Kaiser 风险处理 helper 曾经在全局把纯格挡卡误判为 urgent，影响普通怪战斗。
2. **测试粒度过大**：小策略改动必须 import 巨型 `train.py`，启动慢且容易牵连无关逻辑。
3. **指标命名分散**：同一类策略的 availability / selected / bias / dump 指标散落在多个位置。
4. **后续维护困难**：Claude/Codex 接手时容易继续向 `train.py` 堆代码。

本方案要求后续改动逐步把策略和启发式搜索移入专用目录和专用类；截至 2026-05-11，`train.py` 已只保留兼容入口，Trainer core 已迁到 `muzero/training/trainer.py`。

---

## 硬约束

1. **单文件目标上限：2000 行**
   - 新文件必须 `< 2000` 行。
   - 迁移后的旧文件也应逐步压到 `< 2000` 行；如果某模块超过 1500 行，需要提前规划二级拆分。
2. **新策略禁止直接写入 `train.py`**
   - `train.py` 只能保留兼容 entrypoint / re-export；薄 adapter / wiring / CLI 参数传递也应放入 `muzero/training/*`。
   - 纯策略判断、bias 计算、指标 key 定义、offender 诊断必须放到专用模块。
3. **纯逻辑先写模块，后接入 Trainer**
   - 策略模块尽量不依赖 `MuZeroTrainer` 实例。
   - 需要 trainer helper 的场景，用 adapter 将 raw_obs/action 转成 typed context。
4. **每个迁移 PR/patch 要可测试**
   - 纯 refactor 不允许改变行为。
   - 行为变更必须同时增加单测和 TensorBoard 指标。
5. **route heuristic bias 默认关闭**
   - 近期 Phase 3 route bias 出现回归，默认保持 `--route-heuristic-bias 0.0`。
   - route safety guard 可以保留，但必须与 route score/bias 分离。
6. **seed 默认随机**
   - 固定 seed 只允许 opt-in，用于复现实验。

---

## 当前状态快照

| 文件 | 当前规模 | 状态 |
|---|---:|---|
| `muzero/train.py` | 25 行 | 仅兼容入口 + re-export；`python -m muzero.train` 和旧测试导入继续可用 |
| `muzero/training/trainer.py` | ~1,120 行 | `MuZeroTrainer` core wiring、默认常量、AMP/target encoder/fast-path helper；新策略禁止写入 |
| `muzero/training/cli_main.py` | ~1,091 行 | 训练主循环、env/network/replay/trainer wiring、async actor lifecycle、启动/关闭 orchestration；CLI 参数与 async telemetry 已拆出 |
| `muzero/training/cli_args.py` | ~428 行 | `build_arg_parser()`；只放 CLI flag/default/help，不放训练逻辑、策略逻辑或路径拼接 |
| `muzero/training/async_telemetry.py` | ~286 行 | async actor 的 learner-side TensorBoard metric replay；集中维护 `search_suffix_map`，不要写回 `cli_main.py` |
| `muzero/training/combat_runtime_features.py` | ~1,259 行 | raw_obs/action/energy/HP/enemy/encounter adapter、Kaiser/Insatiable wrapper、positive action classification；超过 1.5k 前继续拆 `combat_state.py` / `encounter_runtime.py` |
| `muzero/training/build_route_hard_guards.py` | ~512 行 | build/campfire 与 route hard guard；route safety guard 与 Phase3 soft bias 分离 |
| `muzero/training/self_play.py` | ~1,250 行 | self-play rollout；同步 metric map 已补齐 route heuristic 与 route safety 指标；禁止继续堆策略 |
| `muzero/sts2_env/mcts.py` | ~1,700 行 | 暂未超限但已接近 2k；route/search 特化必须落到 `muzero/search/` 或 `muzero/route_heuristics/` |
| `sts2_env/route_heuristic.py` | ~790 行 | 现有 route heuristic 专用文件；新增 MuZero prior/search glue 优先放 `muzero/route_heuristics/` |
| `muzero/combat_quality/block_waste.py` | <200 行 | no-pressure pure-block 专用策略模块，符合目标结构 |
| `muzero/combat_quality/action_bias.py` | <200 行 | 已抽出纯格挡 waste root-prior bias，`train.py` 只做调用和指标接线 |
| `muzero/combat_quality/guard_metrics.py` | <200 行 | 已抽出 hard guard 默认 TensorBoard metric key 合约，`train.py` 只循环初始化 |
| `muzero/combat_quality/potion_guard.py` | <500 行 | 已抽出药水 hard-guard 的 boss race / Liquid Memories / Fortifier 例外判断 |
| `muzero/combat_quality/hard_guard_orchestrator.py` | <250 行 | 已抽出 `_apply_combat_action_hard_guards` 调度顺序、合法性检查、默认 key 初始化 |
| `muzero/combat_quality/basic_hard_guards.py` | <400 行 | discard-potion priority、Kaiser facing、Insatiable escape、X-cost zero、HP-cost margin、elite/boss lethal EndTurn |
| `muzero/combat_quality/boss_survival_hard_guards.py` | <500 行 | boss race/setup potion、critical survival potion、elite/boss survival block |
| `muzero/combat_quality/late_normal_hard_guards.py` | <550 行 | late-Act1 normal lethal EndTurn、生存 block/potion、race/setup potion |
| `muzero/combat_quality/survival_non_endturn_guard.py` | <750 行 | 非 EndTurn 动作在危险窗口下的保护性替换，保留 race/scaling enemy 例外 |
| `muzero/combat_quality/potion_bad_use_guard.py` | <700 行 | 低时机药水使用 hard guard，保留 critical/boss/late-normal fail-open 例外 |
| `muzero/combat_quality/selection_loop_guard.py` | <120 行 | card-selection / 净化类重复选同一项的死循环保护 |
| `muzero/strategy/encounters/kaiser.py` | <400 行 | 已抽出 Kaiser/back-attack/facing-change 位置解析；禁止再用 `enemy.side` 判断左右 |
| `muzero/strategy/encounters/insatiable.py` | <150 行 | 已抽出 Insatiable Sandpit countdown 与 Frantic Escape 候选识别 |
| `muzero/training/paths.py` | ~351 行 | 统一路径 dataclass，后续替代 `train.py` 内硬编码路径 |
| `muzero/training/file_budget.py` | <200 行 | 文件行数预算 guard；阻止新增 >2000 行 Python 文件 |
| `muzero/training/monitoring.py` | <300 行 | recent-tail 监控、Null writer、episode capture buffer |
| `muzero/training/cli_parsing.py` | <150 行 | encounter/session/tier/weight/int-list CLI 解析 helper |
| `muzero/training/env_factory.py` | <150 行 | train env 构造、live encounter 查询、session file 解析 |
| `muzero/training/checkpointing.py` | <400 行 | resume load、obs tensor 转换、checkpoint save/prune mixin |
| `muzero/training/losses.py` | ~760 行 | policy/value/reward/objective/future-bank/surface loss mixin |
| `muzero/training/train_step.py` | ~934 行 | 单步训练与优化器/AMP 接线 |

---

## 2026-05-11 已落地拆分

本轮完成的是**纯迁移 / 零行为变更**拆分，目标是把后续策略修改挡在 `train.py` 外面；`train.py` 已降为薄入口：

1. `muzero/train.py`
   - 只保留兼容 entrypoint 与 `from muzero.train import MuZeroTrainer` 旧导入 re-export。
   - 不再持有 `MuZeroTrainer` 类体、训练主循环、策略 helper 或路径拼接。
2. `muzero/training/trainer.py`
   - 从 `train.py` 迁出 `MuZeroTrainer` core wiring、默认常量、AMP/target encoder/fast-path helper。
   - 该文件仍是 trainer core，但必须低于 2000 行；新策略/搜索逻辑继续放到专用目录。
3. `muzero/training/cli_main.py`
   - 从 `train.py` 迁出完整 `main()`。
   - 当前只保留训练 orchestration；CLI 参数和 async TB replay 已继续拆走，避免 `cli_main.py` 再接近 2k。
4. `muzero/training/cli_args.py`
   - 从 `cli_main.py` 迁出 `build_arg_parser()`。
   - route heuristic bias 默认仍是 `0.0`；route safety guard CLI 注册在这里。
   - 该文件只允许新增参数声明/默认值/help 文案，不允许塞策略判断。
5. `muzero/training/async_telemetry.py`
   - 从 `cli_main.py` 迁出 async actor 的 `log_async_episode_scalars()`。
   - route heuristic / route safety / combat quality 的 async TB suffix map 在这里集中维护。
6. `muzero/training/combat_runtime_features.py`
   - 从 `MuZeroTrainer` 迁出 combat/action runtime adapter：
     - decision domain 修正；
     - action semantic family / roles / metric wrapper；
     - HP、energy、incoming damage、enemy HP、discard pile；
     - positive/deferable action classification；
     - Kaiser facing/back-attack 与 Insatiable Frantic Escape wrapper。
   - 纯策略仍在 `muzero/strategy/*` 与 `muzero/combat_quality/*`，这里只做 bridge payload 到策略函数的 adapter。
7. `muzero/training/build_route_hard_guards.py`
   - build/campfire REST/SMITH guard 与 route safety guard 已独立于 combat hard guard。
   - route safety guard 是 hard guard；Phase3 route heuristic bias 是 soft prior，两者不得混写。
8. 测试已随拆分更新：
   - CLI flag source 断言改读 `muzero/training/cli_args.py`；
   - async writer metric source 断言改读 `muzero/training/async_telemetry.py`；
   - 主循环/wiring source 断言仍读 `muzero/training/cli_main.py`；
   - Trainer constructor/wiring source 断言改读 `muzero/training/trainer.py`；
   - 同步 self-play metric map 仍读 `muzero/training/self_play.py`；
   - 兼容测试继续从 `muzero.train import MuZeroTrainer`。

当前关键行数：

```text
  25  muzero/train.py
1120  muzero/training/trainer.py
1091  muzero/training/cli_main.py
 428  muzero/training/cli_args.py
 286  muzero/training/async_telemetry.py
1259  muzero/training/combat_runtime_features.py
1250  muzero/training/self_play.py
 512  muzero/training/build_route_hard_guards.py
 934  muzero/training/train_step.py
 760  muzero/training/losses.py
1166  muzero/combat_quality/trainer_quality.py
1299  muzero/diagnostics/episode_metrics.py
1168  muzero/diagnostics/trainer_dumps.py
```

本轮验证：

```text
python -m py_compile muzero/train.py
                    muzero/training/trainer.py
                    muzero/training/cli_main.py
                    muzero/training/cli_args.py
                    muzero/training/async_telemetry.py
                    muzero/training/file_budget.py
                    muzero/training/paths.py
                    muzero/diagnostics/trainer_dumps.py

python -m muzero.train --help

pytest tests/test_muzero_training_paths.py
       tests/test_route_heuristic_phase2.py
       tests/test_route_heuristic_phase3_bias.py
       tests/test_route_safety_guard.py
       tests/test_build_action_hard_guards.py
       tests/test_combat_action_hard_guards.py
       tests/test_selection_loop_guard_p2_2.py
       tests/test_strategy_encounter_modules.py
       tests/test_combat_quality_potion_guard.py
       tests/test_combat_quality_guard_metrics.py
       tests/test_combat_quality_action_bias.py
       tests/test_combat_quality_metrics.py
       tests/test_combat_quality_typed_card_effect_profile.py

170 passed, 13 subtests passed
```

行数预算检查：

```text
python scripts/check_muzero_file_budget.py --quiet-ok
```

检查通过；当前 >2000 行只剩 allowlist 中的历史债务：

```text
sts2_env/observation_v3.py
tests/test_combat_action_hard_guards.py
sts2_env/env_v2.py
muzero/sts2_env/token_memory.py
sts2_env/combat_env.py
muzero/sts2_env/muzero_model.py
sts2_env/_sim_translate.py
```

---

## 文件与路径治理规则

从现在开始，新增策略、启发式搜索、诊断 dump、训练 artifact 路径不得直接在 `train.py` 中临时拼接。统一使用：

```text
packages/rl-agent/muzero/training/paths.py
```

已落地的路径类：

```python
RunPaths
  repo_root
  package_root
  log_dir
  checkpoint_dir
  resume_from
  diagnostics_dir
  checkpoint_step_dir(step)
  replay_buffer_path(...)
  diagnostic_jsonl(name)

StrategyModulePaths
  strategy_dir
  strategy_encounters_dir
  combat_quality_dir
  diagnostics_dir
  strategy_file(name)
  encounter_file(name)
  combat_quality_file(name)
  diagnostics_file(name)

HeuristicSearchModulePaths
  route_heuristics_dir
  search_dir
  route_heuristic_file(name)
  search_file(name)

PolicyModulePaths
  # backward-compatible aggregate:
  strategy: StrategyModulePaths
  heuristic_search: HeuristicSearchModulePaths
```

使用原则：

1. **训练运行文件**（log、checkpoint、diagnostics、replay buffer、人类示范数据）走 `RunPaths`。
2. **通用策略源码目录**（action feature、combat state、HP/X-cost/card-state/potion timing、encounter policy）走 `StrategyModulePaths.strategy_file(...)` / `StrategyModulePaths.encounter_file(...)`。
3. **战斗质量/guard 源码目录**（pure-block、end-turn、hard guard、action offender）仍走 `StrategyModulePaths.combat_quality_file(...)`。
4. **启发式搜索源码目录**（route graph/candidate/scoring/bias/guard、root-prior/search glue）走 `HeuristicSearchModulePaths`。
5. 旧代码或过渡期需要“一把拿全路径”时才走 `PolicyModulePaths`。
6. `train.py` 只允许：
   - 从 CLI args 构造 `RunPaths.from_args(args)`；
   - 把 `RunPaths` 注入 trainer / diagnostics / checkpointing；
   - 不再出现新的 `Path(self.log_dir) / "diagnostics" / ...` 形式。
7. 旧的硬编码路径迁移时必须保持文件名不变，避免打断已有日志分析脚本。

推荐替换模式：

```python
# 旧：散落在 train.py
path = Path(self.log_dir) / "diagnostics" / "loss_spikes.jsonl"

# 新：集中路径契约
path = self.run_paths.diagnostic_jsonl("loss_spikes")
```

行数预算：

| 模块类别 | 单文件上限 | 超过 1500 行时处理 |
|---|---:|---|
| strategy/policy | 2000 | 按机制拆子文件 |
| route/search heuristic | 2000 | graph/candidates/scoring/bias/guard 分离 |
| diagnostics/metrics | 2000 | key registry、dump writer、readout 分离 |
| training loop/loss | 2000 | loop/losses/replay/checkpoint/tensorboard 分离 |
| `train.py` entrypoint | 100 | 只保留兼容 CLI entrypoint / re-export；不再接收新 wiring |

检查命令：

```bash
cd packages/rl-agent
python scripts/check_muzero_file_budget.py
```

该检查当前采用“**遗留大文件 allowlist + 新文件强制失败**”模式：

- `combat_env.py`、`env_v2.py`、`observation_v3.py`、`muzero_model.py` 等历史债务会打印为 `LEGACY`；`muzero/train.py` 已不在 allowlist，若再次超过 2000 行会直接失败；
- 任何新策略/搜索/诊断文件超过 2000 行会打印为 `BLOCK` 并返回非零退出码；
- 等拆分完成后再用 `--fail-on-legacy` 切到严格模式。

---

## 目标目录结构

### 1. 通用策略：`muzero/strategy/`

```text
packages/rl-agent/muzero/strategy/
  __init__.py
  action_features.py             # action/card identity、cost、roles、family、exhaust/ethereal/retain 等纯特征
  combat_state.py                # HP、energy、incoming pressure、enemy list、pile count 等 combat state 摘要
  decision_domain.py             # combat/build/route domain 修正与识别
  potions.py                     # potion profile、timing、keep value、death unused count 等
  card_quality.py                # 正收益动作/无效动作分类；调用 combat_quality/block_waste.py
  hp_cost_policy.py              # 放血/扣血卡 self-lethal、low-margin、blocked/unblocked HP cost
  x_cost_policy.py               # X 费动态能量、0 费 X 牌、star-X/non-energy side effect
  card_state_policy.py           # 保留/虚无/重放/升级/复制/附魔/替换/手牌状态 mutation
  encounters/
    __init__.py
    kaiser.py                    # facing/back-attack/facing-change 策略（已落地）
    insatiable.py                # sandpit countdown / Frantic Escape urgency（已落地）
    boss_potions.py              # boss 机制药水与 race 逻辑
```

原则：

- 这里放“该不该做某类动作”的纯判断与特征，不直接写 TensorBoard，也不直接读写 checkpoint/log。
- `strategy/*` 可以被 combat guards、search prior、diagnostics 复用。
- `strategy/*` 不 import `muzero.train`；需要 Trainer 状态时由 adapter 组装 `Context` 后传入。

---

### 2. 战斗策略质量：`muzero/combat_quality/`

```text
packages/rl-agent/muzero/combat_quality/
  __init__.py
  block_waste.py                 # 纯格挡卡在无伤害压力下的 waste 判断（已落地）
  end_turn_quality.py            # End Turn 分类：bad / forced / strategic_defer / unknown
  hp_cost_policy.py              # 放血/扣血卡自杀、低血边际、blocked/unblocked HP cost
  x_cost_policy.py               # X 费动态能量、0 能量 X 牌、star X/non-energy side effect
  card_state_policy.py           # 保留/虚无/重放/升级/复制/附魔/替换/手牌状态 mutation
  potion_timing_policy.py        # 药水使用窗口、save value、overkill、follow-up、mechanism answer
  boss_mechanic_policy.py        # Kaiser/Insatiable/Ceremonial/KnowledgeDemon 等 boss 机制入口
  action_bias.py                 # 汇总各 combat policy 的 root prior bias
  metrics.py                     # combat_quality_* 指标 key、aggregation white-list
  guard_metrics.py               # hard guard 默认 metric key 合约；每个 combat decision 初始化为 0
  potion_guard.py                # 药水 hard guard 例外窗口：boss race、Lagavulin、Liquid Memories、Fortifier
  offenders.py                   # action offender reason flags + dump schema
```

建议类/函数边界：

```python
@dataclass(frozen=True)
class CombatActionContext:
    action: Mapping[str, Any]
    raw_obs: Mapping[str, Any]
    legal_actions: Sequence[Mapping[str, Any]]
    action_index: int
    energy: float
    mask: np.ndarray

@dataclass(frozen=True)
class CombatPolicyResult:
    availability: dict[str, float]
    selected_flags: dict[str, float]
    bias: np.ndarray | None
    reasons: tuple[str, ...]

class CombatQualityPolicy:
    def classify_action(self, ctx: CombatActionContext) -> CombatPolicyResult: ...
    def compute_bias(self, root_ctx: CombatRootContext) -> CombatPolicyResult: ...
```

Trainer 层保留内容（位于 `muzero/training/trainer.py` 或 mixin；**不回写 `train.py`**）：

```python
# Trainer / mixin 内只做 adapter
ctx = self._make_combat_root_context(obs, raw_obs, legal_actions, mask)
bias, stats, debug = self.combat_quality_policy.compute_bias(ctx)
```

当前已完成的薄 adapter 迁移：

- Kaiser facing/back-attack：
  - 源码：`muzero/strategy/encounters/kaiser.py`
  - 保留 wrapper：`MuZeroTrainer._is_kaiser_facing_change_action(...)`、`_find_kaiser_facing_candidates(...)`
  - 关键契约：只信 `BACK_ATTACK_LEFT_POWER` / `BACK_ATTACK_RIGHT_POWER`；`enemy.side == "Enemy"` 不是左右。
- Insatiable Sandpit：
  - 源码：`muzero/strategy/encounters/insatiable.py`
  - 保留 wrapper：`_find_insatiable_frantic_escape_candidates(...)`、`_insatiable_sandpit_countdown_from_context(...)`
  - 关键契约：`Frantic Escape` 先匹配内部 id，再用中英文标题兜底。
- 药水 guard：
  - 源码：`muzero/combat_quality/potion_guard.py`
  - 保留 wrapper/import alias：Trainer/mixin 只负责传入 encounter/HP/energy/threat context。
  - 关键契约：不要把药水“有就用”；只给 boss race、0 费无非药水替代、真实压力/关键 HP、Lagavulin setup 等窄窗口放行。

---

### 3. Route / 地图启发式：`muzero/route_heuristics/`

当前已有 `sts2_env/route_heuristic.py`，后续建议迁移或包装到 MuZero 侧专用目录，避免 route search 逻辑继续混进 `train.py` / `mcts.py`。

```text
packages/rl-agent/muzero/route_heuristics/
  __init__.py
  graph.py                       # map graph / reachable route forest 解析
  candidates.py                  # top-K path enumeration
  scoring.py                     # rest/shop/elite/event/boss 路径评分
  safety_guard.py                # low HP elite、forced elite、rest-before-elite hard guard
  bias.py                        # 将 route score 转成 root prior bias；默认 off
  metrics.py                     # route_heuristic/* 指标 key 和聚合
  dry_run.py                     # dry-run record / baseline compare
```

建议类：

```python
@dataclass(frozen=True)
class RouteDecisionContext:
    map_state: Mapping[str, Any]
    current_floor: int
    hp_ratio: float
    gold: int
    deck_quality: Mapping[str, float]
    legal_actions: Sequence[Mapping[str, Any]]

@dataclass(frozen=True)
class RouteCandidateScore:
    action_index: int
    score: float
    breakdown: Mapping[str, float]
    unsafe_elite: bool
    forced_elite_count: int
    rest_before_elite: bool

class RouteHeuristicPolicy:
    def enumerate_candidates(self, ctx: RouteDecisionContext) -> list[RouteCandidateScore]: ...
    def compute_prior_bias(self, ctx: RouteDecisionContext, weight: float) -> np.ndarray: ...
    def safety_override(self, ctx: RouteDecisionContext) -> int | None: ...
```

注意：

- `bias.py` 只做 soft prior，不做 hard override。
- `safety_guard.py` 才能做 hard override。
- route bias 默认 `0.0`；所有实验必须通过 CLI 明确打开。

---

### 4. 搜索和 planning glue：`muzero/search/`

```text
packages/rl-agent/muzero/search/
  __init__.py
  root_prior.py                  # policy logits + heuristic bias + legal mask 合成
  direct_policy.py               # combat-policy-mode direct 分支
  rollout_policy.py              # combat rollout steps / beam / q blend
  route_search.py                # route-num-simulations / route-specific MCTS glue
  build_search.py                # build/card reward/shop/rest search glue
```

目标：`mcts.py` 保持通用搜索实现，不包含具体 STS2 combat/boss/route 业务规则。

---

### 5. 训练循环：`muzero/training/`

```text
packages/rl-agent/muzero/training/
  __init__.py
  cli.py                         # argparse 参数定义与配置校验（目标态）
  cli_parsing.py                 # 已落地：encounter/session/tier/weight/int-list 解析 helper
  trainer.py                     # MuZeroTrainer 主类的瘦身后版本
  loop.py                        # collect/train/eval/checkpoint 主循环
  losses.py                      # 已落地：value/reward/policy/future_world/future_bank/surface loss mixin
  optimizer.py                   # AMP GradScaler、grad clip、optimizer/scheduler resume 策略
  monitoring.py                  # 已落地：recent-tail monitor / Null writer / episode capture buffer
  env_factory.py                 # 已落地：train env/session/pool 构造
  replay_io.py                   # buffer load/save、resume-without-buffer、tier quota
  checkpointing.py               # 已落地：checkpoint load/save/rotation + obs tensor helper
  tensorboard.py                 # TB emit helper、metric aggregation
  paths.py                       # RunPaths/CheckpointPaths/LogPaths dataclass
```

建议路径类：

```python
@dataclass(frozen=True)
class RunPaths:
    repo_root: Path
    package_root: Path
    log_dir: Path
    checkpoint_dir: Path
    resume_from: Path | None = None

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "RunPaths": ...
    def ensure_dirs(self) -> None: ...

@dataclass(frozen=True)
class PolicyModulePaths:
    combat_quality_dir: Path
    route_heuristics_dir: Path
    diagnostics_dir: Path
```

目的：避免脚本、trainer、diagnostics 各自硬编码路径；日志、checkpoint、replay、human demo 文件统一从 `RunPaths` 派生。

---

### 5. 诊断与指标：`muzero/diagnostics/`

```text
packages/rl-agent/muzero/diagnostics/
  __init__.py
  metric_keys.py                 # 全局 key registry，避免字符串散落
  boss_episode.py                # boss_combat/* per-encounter/global emit
  action_offenders.py            # bad action dump/offender reason taxonomy
  spike_dump.py                  # loss spike batch dump schema
  log_readout.py                 # 训练日志/TB readout 脚本复用逻辑
```

指标规范：

- availability：`*_available`, `*_count`, `*_candidate`
- selected：`*_selected`, `*_selected_rate`
- bias：`*_bias_count`, `*_bias_abs_mean`, `*_bias_min/max`
- per-boss：`boss_combat/<encounter_key>/*`
- global combat guard：`combat_guards/*`

---

## 拆分优先级

### Phase A — 立即约束新增代码（已开始）

目标：停止继续污染 `train.py`。截至 2026-05-11，`train.py` 已经只剩兼容入口；后续 adapter 也应放在 `muzero/training/*` mixin 或专用策略模块中。

- [x] 新增 `muzero/combat_quality/block_waste.py`。
- [x] `MuZeroTrainer` 只保留 `_card_block_waste_profile()` adapter；`train.py` 不再持有 adapter。
- [x] 新增 no-pressure pure-block 单测。
- [x] 修复 Kaiser risk helper 污染全局 block urgent 的问题。
- [ ] 后续任何 combat 策略必须先放入 `muzero/combat_quality/`。
- [ ] 后续任何 route 策略必须先放入 `muzero/route_heuristics/` 或现有 `sts2_env/route_heuristic.py`，不得直接写入 `train.py`。

验收：

```bash
python -m py_compile muzero/train.py muzero/combat_quality/*.py
python -m pytest tests/test_combat_quality_typed_card_effect_profile.py tests/test_combat_action_hard_guards.py -q
```

---

### Phase B — 抽出 Combat action quality bias

目标：把 `_raw_end_turn_context()`、`_combat_action_quality_bias()`、`_selected_combat_quality_stats()` 中的业务策略搬出巨型 Trainer；`train.py` 已不再承载这些逻辑。

建议步骤：

1. 新建 `muzero/combat_quality/context.py`
   - `CombatRootContext`
   - `CombatActionContext`
   - action/raw_obs adapter 的输入结构
2. 新建 `muzero/combat_quality/end_turn_quality.py`
   - 搬迁 End Turn taxonomy：bad / forced / strategic_defer / unknown。
3. 新建 `muzero/combat_quality/action_bias.py`
   - 输入 `CombatRootContext`，输出 `(bias, stats, debug)`。
4. Trainer/mixin 中 `_combat_action_quality_bias()` 改成 wrapper。
5. 保持所有现有指标 key 不变。

验收：

- `tests/test_combat_action_hard_guards.py` 全通过。
- `tests/test_combat_quality_typed_card_effect_profile.py` 全通过。
- 旧 run 的 TB tag 不断档。
- `train.py` 保持薄入口；Trainer/mixin 行数继续受 2000 行预算约束。

风险控制：

- 先搬纯函数，不改阈值。
- 每搬一个函数就跑目标测试。

---

### Phase C — 抽出 Boss mechanics policy

目标：Kaiser / Insatiable / Ceremonial 等 boss 特化规则与普通战斗策略隔离，避免再次污染全局。

建议文件：

```text
muzero/combat_quality/boss_mechanic_policy.py
```

拆分内容：

- Kaiser:
  - facing-change candidate
  - back-attack risk handling
  - pressure vs defense selected metrics
- Insatiable:
  - Sandpit countdown
  - Frantic Escape urgency curve
  - missed_lt3 / missed_at_1 metrics
- Ceremonial:
  - one-card lock
  - stun window
  - high/low impact action classification
- Knowledge Demon:
  - selection runtime state / internal id usage

验收：

- 非 Kaiser raw_obs 下任何 Kaiser helper 不得把普通 block/debuff 标记为 mechanism urgent。
- per-boss 指标只在对应 encounter emit。
- `boss_combat/<encounter>/*` 与 global `boss_combat/*` 都有可读汇总。

---

### Phase D — 抽出 Potion / HP cost / Card state policies

目标：把高风险策略模块化，避免互相影响。

文件：

```text
muzero/combat_quality/potion_timing_policy.py
muzero/combat_quality/hp_cost_policy.py
muzero/combat_quality/card_state_policy.py
muzero/combat_quality/x_cost_policy.py
```

关注机制：

- 药水：save value、lethal/prevent lethal、overkill、hand-context、long-term potion。
- HP cost：自杀、低血边际、unblockable HP cost、是否能靠 block/kill 抵消风险。
- card state：虚无、保留、重放、升级、复制、附魔、替换、选择循环、follow-up。
- X cost：当前动态 energy，而不是初始 3 费；0 费 X 牌是否有 non-energy side effect。

验收：

- 每个文件有独立测试。
- selected-side 与 availability-side 使用同一套 classification。
- 指标完整：available / selected / bias / offender dump。

---

### Phase E — Route heuristic 解耦

目标：`train.py` 不再直接计算 route heuristic dry-run、bias、safety guard；现状是 route/build hard guard 已在 `muzero/training/build_route_hard_guards.py`，后续 route soft-prior/search glue 应继续迁往 `muzero/route_heuristics/` 或 `muzero/search/`。

步骤：

1. 将 `sts2_env/route_heuristic.py` 保留为底层评分函数，或迁移到 `muzero/route_heuristics/scoring.py`。
2. 新建 `RouteHeuristicPolicy` 聚合：candidate enumeration + score + bias + safety override。
3. Trainer/mixin 只调用：

```python
route_result = self.route_policy.evaluate(ctx)
if route_result.override_idx is not None:
    action_idx = route_result.override_idx
root_prior += route_result.bias
stats.update(route_result.metrics)
```

4. route bias 默认 0；safety guard 单独 flag。

验收：

- `tests/test_route_heuristic_phase2.py`
- `tests/test_route_heuristic_phase3_bias.py`
- `tests/test_route_safety_guard.py`
- route bias off 时 `search/route/bias_* == 0`。

---

### Phase F — 训练循环和 loss 拆分

目标：让 `train.py` 最终只做 CLI entrypoint。该目标已达成；下面保留为目标形态/防回归标准。

最终形态：

```python
# muzero/train.py
from muzero.training.trainer import *  # compatibility re-export

if __name__ == "__main__":
    from muzero.training.cli_main import main
    main()
```

迁移顺序：

1. `training/cli.py`
2. `training/paths.py`
3. `training/checkpointing.py`
4. `training/replay_io.py`
5. `training/losses.py`
6. `training/loop.py`
7. `training/tensorboard.py`

验收：

- `train.py < 100` 行。
- 每个 `training/*.py < 2000` 行。
- checkpoint resume / resume-without-optimizer / resume-without-buffer 行为不变。
- 旧启动脚本无需改 main module：仍可 `python -m muzero.train`。

---

## 防回归检查清单

每次拆分或策略改动后必须执行：

```bash
cd /mnt/e/game/project/sts2_mcp/packages/rl-agent
./.venv-wsl-rocm/bin/python -m py_compile \
  muzero/train.py \
  muzero/combat_quality/*.py \
  muzero/strategy/encounters/*.py \
  sts2_env/route_heuristic.py

./.venv-wsl-rocm/bin/python -m pytest \
  tests/test_strategy_encounter_modules.py \
  tests/test_combat_quality_potion_guard.py \
  tests/test_combat_quality_typed_card_effect_profile.py \
  tests/test_combat_action_hard_guards.py \
  tests/test_route_heuristic_phase2.py \
  tests/test_route_heuristic_phase3_bias.py \
  tests/test_route_safety_guard.py \
  -q

# 新文件不得超过 2000 行；历史巨型文件只允许作为 LEGACY 债务存在。
./.venv-wsl-rocm/bin/python scripts/check_muzero_file_budget.py --quiet-ok
```

如果改动训练 loop / replay / checkpoint，再加：

```bash
./.venv-wsl-rocm/bin/python -m pytest \
  tests/test_replay_tier_quota.py \
  tests/test_env_v2_frontier_recovery.py \
  tests/test_build_action_hard_guards.py \
  -q
```

---

## 今晚训练相关原则

1. 先解决全局 combat intent：
   - 非攻击意图时，不应把纯格挡牌视为 urgent。
   - 只有真实进攻/setup/draw/scaling/debuff/card-state mutation 或 boss mechanism answer 才让 End Turn 变 bad。
2. seed 默认随机：
   - 让模型看到更丰富的怪物、卡牌、奖励、路线分布。
   - 需要复现时才 `USE_FIXED_SEED=1 FIXED_SEED=...`。
3. route bias 保持 off：
   - 不把 Phase 3 route bias 回归混进 combat policy 观察。
4. 每次训练 readout 必看：
   - `combat_quality_card_block_waste_count`
   - `combat_quality_card_block_waste_selected`
   - `combat_quality_wasteful_end_turn_selected`
   - `combat_quality_bad_end_turn_selected`
   - normal/elite/boss 分桶 win rate
   - `episode/act1_boss_seen`, `episode/act1_clear`

---

## 已完成的本次落地项

- 新增 `muzero/combat_quality/block_waste.py`，把“无伤害压力下纯格挡是 waste”的判断移出 `train.py`。
- 新增 `muzero/combat_quality/action_bias.py`，把纯格挡 waste 的 root-prior bias 计算移出 `train.py`；当存在攻击/过牌/设置等 progress alternative 时使用 hard penalty，并额外 emit `card_block_waste_hard_bias_applied/progress_*` 指标。
- 新增 `tests/test_combat_quality_action_bias.py`，覆盖 hard penalty、soft penalty、mask/out-of-range 过滤、1-D 输入校验。
- 新增 `muzero/training/paths.py`，提供 `RunPaths` / `PolicyModulePaths`，后续路径拼接不得继续散落进 `train.py`。
- `muzero/training/cli_main.py` 的 main setup 和 `MuZeroTrainer.__init__` 已接入 `RunPaths`；log/checkpoint/diagnostics/async actor scratch 目录统一从 `RunPaths` 派生。
- `MuZeroTrainer` 已持有 `policy_module_paths = PolicyModulePaths.from_package_root(...)`，后续策略/route/search 源码路径必须从该聚合路径类或其 `strategy` / `heuristic_search` 子路径类派生。
- 新增 `muzero/route_heuristics/`、`muzero/search/`、`muzero/diagnostics/` 包入口，作为后续 route/search/metric 迁移的固定落点。
- 新增 `muzero/training/file_budget.py` 与 `scripts/check_muzero_file_budget.py`：任何非 allowlist Python 文件超过 2000 行会失败；`train.py` 已从 allowlist 移除，若再次膨胀会直接失败。
- `MuZeroTrainer` 对 pure-block waste 仅保留 `_card_block_waste_profile()` adapter 与 `apply_card_block_waste_bias(...)` 接线；具体判断、bias 强度、聚合指标 key 均在 `muzero/combat_quality/`。
- 新增 `muzero/combat_quality/guard_metrics.py`，把 `_apply_combat_action_hard_guards()` 开头 93 个默认 metric key 从 `train.py` 移出。
- 修复 `_classify_positive_combat_action()` 中 Kaiser helper 的全局污染：只有 Kaiser back-attack risk > 0 时，block/debuff 才能作为 Kaiser mechanism urgent。
- 增加单测覆盖：
  - no incoming + Defend => deferable, not urgent
  - incoming + Defend => urgent
  - no incoming + block/draw => not block waste
  - only no-pressure Defend + End Turn => End Turn 不算 wasteful，Defend 被压制
  - Strike + no-pressure Defend + End Turn => End Turn 仍算 wasteful，Defend 被压制
- 增加路径类单测 `tests/test_muzero_training_paths.py`，确保相对 log/checkpoint/resume 路径统一按 `packages/rl-agent` 解析。
- `tests/test_muzero_training_paths.py` 同时覆盖文件预算 guard：新巨型策略文件会被识别为 blocking violation，历史巨型文件只走 documented legacy allowlist。
- 启动脚本默认随机 seed；固定 seed 改为 opt-in。
- 新增 `muzero/combat_quality/potion_guard.py`，把 boss race / Lagavulin setup / 0 能量 Liquid Memories / Fortifier 真实防御窗口从 `_apply_combat_action_hard_guards()` 中抽出。
- 新增 `tests/test_combat_quality_potion_guard.py`，覆盖强度药、伤害药、空 Liquid Memories、Lagavulin setup、0 能量 boss 压力、Fortifier no-op。
- 新增 `muzero/strategy/encounters/kaiser.py`，把 Kaiser facing-change、target combat id、BACK_ATTACK 左右解析、candidate 搜索从 `train.py` 抽出；保留 Trainer wrapper 兼容旧测试 patch。
- 新增 `muzero/strategy/encounters/insatiable.py`，把 Frantic Escape 内部 id / 中英文 fallback 和 Sandpit countdown 解析从 `train.py` 抽出。
- 新增 `tests/test_strategy_encounter_modules.py`，直接覆盖 encounter 纯模块，不再只通过巨型 `MuZeroTrainer` 间接测试。
- 新增 `muzero/training/monitoring.py`，把 `NullSummaryWriter`、`RecentCombatMonitor`、`EpisodeCaptureBuffer` 移出 `train.py`。
- 新增 `muzero/training/cli_parsing.py`，把 encounter pool/session/tier/weight/int-list 解析 helper 移出 `train.py`。
- 新增 `muzero/training/env_factory.py`，把 `build_train_env()`、live encounter 查询和 session file resolution 移出 `train.py`。
- 扩展 `muzero/training/checkpointing.py`，把 checkpoint save/prune 逻辑移入 `CheckpointingMixin`；`train.py` 只继承并调用。
- 新增 `muzero/training/losses.py`，把 policy/value/reward/objective/future-bank/surface loss helper 移入 `TrainingLossMixin`。
- 新增 `muzero/training/self_play.py`、`muzero/training/train_step.py`，把 self-play rollout 与 optimizer train-step 从 `train.py` 移入 mixin；`self_play.py` 当前约 1,250 行，超过 1,500 前需要继续拆 rollout metric/step adapter。
- 新增 `muzero/diagnostics/episode_metrics.py`、`muzero/diagnostics/trainer_dumps.py`，把 episode/TensorBoard 聚合与 hard-guard/death-slice dump 写出从 `train.py` 移出。
- 新增 combat hard-guard 分桶模块：
  - `hard_guard_orchestrator.py`：统一合法性检查、默认 key 初始化、顺序调度与 override-any 指标；
  - `basic_hard_guards.py`：discard/Kaiser/Insatiable/X-cost/HP-cost/lethal EndTurn；
  - `boss_survival_hard_guards.py`：elite/boss potion/block 生存窗口；
  - `late_normal_hard_guards.py`：late-Act1 normal lethal/survival/race；
  - `survival_non_endturn_guard.py`：危险窗口下非 EndTurn 动作保护性替换；
  - `potion_bad_use_guard.py`：低时机药水使用拦截与 fail-open 例外；
  - `selection_loop_guard.py`：净化/选卡重复选择死循环保护。
- 当前 `train.py` 行数约 **25**，已不再是 legacy monolith；`MuZeroTrainer` core 位于 `muzero/training/trainer.py`（约 1,120 行）。下一批优先拆 `trainer.py` 中的 AMP/target encoder/fast-path helper，以及 `combat_runtime_features.py` / `self_play.py` 超过 1,500 行前的二级拆分；不要再把策略阈值写回 `train.py`。
