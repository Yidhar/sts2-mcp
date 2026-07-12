# MuZero 代码拆分与文件预算约束

本文档用于约束后续 MuZero / STS2 训练代码改动，避免再次把所有训练、
策略、启发式搜索、诊断逻辑堆回 `muzero/train.py`。

## 当前结论

- `muzero/train.py` 当前按 `scripts/check_muzero_file_budget.py` 的计数方式是
  **25 个物理行**，只做兼容入口和 legacy re-export。
- `muzero/train.py` 只允许是**薄兼容入口**：
  - 保持 `python -m muzero.train` 可用；
  - 保持历史导入 `from muzero.train import MuZeroTrainer` 可用；
  - 不再承载策略阈值、boss 机制、route scoring、hard guard、训练 loss 实现。
- 新增 Python 文件默认必须 `< 2000` 行。
- 文件超过 `1500` 行时，先拆分职责，再继续添加逻辑。
- 路径处理必须使用 `muzero/training/paths.py` 中的专用路径类，不能从
  `train.py` 反向拼路径；策略源码路径走 `StrategyModulePaths`，route/search
  启发式源码路径走 `HeuristicSearchModulePaths`，聚合使用 `PolicyModulePaths`。

## 目录职责边界

```text
packages/rl-agent/muzero/
├── train.py
│   └── 薄入口 / legacy import compatibility；禁止继续写业务逻辑。
├── training/
│   ├── trainer.py
│   │   └── MuZeroTrainer 训练编排主类；只做 orchestration，不写具体策略。
│   ├── cli_main.py / cli_args.py / cli_parsing.py
│   │   └── CLI 入口、参数定义、参数解析 helper。
│   ├── env_factory.py
│   │   └── 训练环境/session/pool 构造。
│   ├── self_play.py
│   │   └── self-play episode loop 与 trajectory 收集。
│   ├── train_step.py / losses.py
│   │   └── optimizer step、policy/value/reward/objective/future-bank loss。
│   ├── checkpointing.py
│   │   └── checkpoint save/load/prune、resume 相关逻辑。
│   ├── monitoring.py / async_telemetry.py
│   │   └── TensorBoard、recent-tail、异步 actor telemetry。
│   ├── combat_runtime_features.py
│   │   └── 从 raw obs / legal actions 抽运行期特征；不直接决定策略。
│   ├── route_heuristic_telemetry.py
│   │   └── route heuristic 指标聚合。
│   ├── paths.py
│   │   └── 运行期路径与源码模块路径的唯一入口。
│   └── file_budget.py
│       └── 单文件行数预算检查逻辑。
├── strategy/
│   ├── action_features.py
│   │   └── 通用动作/卡牌特征判定。
│   └── encounters/
│       ├── kaiser.py
│       │   └── Kaiser back-attack / facing-change 机制。
│       └── insatiable.py
│           └── Sandpit countdown / Frantic Escape 机制。
├── combat_quality/
│   ├── block_waste.py / action_bias.py / metrics.py
│   │   └── 无伤害压力下纯格挡浪费、root-prior bias、combat quality 指标。
│   ├── hard_guard_orchestrator.py
│   │   └── hard guard 调度顺序。
│   ├── meaningful_damage_guard.py
│   │   └── 无/低压力且安全进展牌存在时，禁止把 End Turn 当作空收益选择。
│   ├── no_pressure_block_guard.py
│   │   └── 无/极低压力下把纯防御/格挡浪费改写为安全伤害或战斗进展牌。
│   ├── basic_hard_guards.py
│   ├── boss_survival_hard_guards.py
│   ├── late_normal_hard_guards.py
│   ├── survival_non_endturn_guard.py
│   ├── potion_bad_use_guard.py
│   ├── potion_guard.py / potion_timing.py
│   └── selection_loop_guard.py
│       └── 各类 combat 策略保护与指标；禁止回填到 train.py。
├── route_heuristics/
│   └── route 候选、评分、安全性、bias 逻辑。
├── search/
│   └── root prior/search adapter；不写 STS2 卡牌/怪物专用策略。
├── diagnostics/
│   ├── episode_metrics.py
│   └── trainer_dumps.py
│       └── 诊断 dump schema、episode metric 聚合。
└── sts2_env/
    └── MuZero 网络、token memory、MCTS、buffer 等环境/模型兼容层。
```

## 路径类使用规范

所有运行期 artifact 路径使用：

```python
from muzero.training.paths import RunPaths

run_paths = RunPaths.from_args(args)
run_paths.ensure_dirs()
diagnostic_path = run_paths.diagnostic_jsonl("combat_offenders")
checkpoint_path = run_paths.checkpoint_step_dir(step)
```

所有策略源码路径使用：

```python
from muzero.training.paths import StrategyModulePaths

paths = StrategyModulePaths.from_package_root()
paths.strategy_file("action_features")
paths.encounter_file("kaiser")
paths.combat_quality_file("block_waste")
paths.diagnostics_file("episode_metrics")
```

所有 route/search 启发式源码路径使用：

```python
from muzero.training.paths import HeuristicSearchModulePaths

paths = HeuristicSearchModulePaths.from_package_root()
paths.route_heuristic_file("root_bias")
paths.search_file("some_search_adapter")
```

过渡期如必须一次拿到全部路径，使用：

```python
from muzero.training.paths import PolicyModulePaths

paths = PolicyModulePaths.from_package_root()
```

## 文件预算检查

常规检查命令：

```bash
cd packages/rl-agent
./.venv-wsl-rocm/bin/python scripts/check_muzero_file_budget.py --quiet-ok
```

默认扫描范围包含 `muzero/`、`sts2_env/`、`scripts/`、`tests/`、`legacy/`，
并跳过虚拟环境、日志、checkpoint、cache 目录。若要临时做全包扫描，可以传
`--root .`；检查器同样会跳过上述非源码目录。

严格模式（包含历史遗留文件也失败）：

```bash
./.venv-wsl-rocm/bin/python scripts/check_muzero_file_budget.py --fail-on-legacy
```

当前策略是：

- 新增/改造文件超过 2000 行：**失败**。
- 既有历史大文件暂时 allowlist，但不能继续扩大职责。
- 历史大文件后续应按优先级逐步拆：
  1. `sts2_env/observation_v3.py`
  2. `sts2_env/env_v2.py`
  3. `sts2_env/combat_env.py`
  4. `muzero/sts2_env/muzero_model.py`
  5. `muzero/sts2_env/token_memory.py`
  6. `legacy/train_offline_multitask.py`
  7. `sts2_env/_sim_translate.py`
  8. `tests/test_combat_action_hard_guards.py`

## 剩余历史大文件拆分任务

`scripts/check_muzero_file_budget.py` 当前默认只阻止新的超大文件；严格模式
`--fail-on-legacy` 会继续失败，直到下表全部拆完。这个 allowlist 只能减少，
不能新增。拆分时保持原 public import 兼容：老文件降为 facade / re-export，
真实实现迁移到职责更小的模块。

| 当前文件 | 当前行数量级 | 目标结构 | 拆分原则 |
| --- | ---: | --- | --- |
| `sts2_env/observation_v3.py` | 4.4k | `sts2_env/observation_v3/` 或 `sts2_env/observation_*` 子模块 | token schema、player/enemy/card/relic/potion 编码、map/run context、mask/feature 校验分开；原文件只保留 builder facade。 |
| `sts2_env/env_v2.py` | 3.1k | `sts2_env/env_v2/` 或 `sts2_env/full_run_*` 子模块 | reset/step 主流程、reward、room transition、map/event/shop/rest/card reward 处理分开；训练策略不能写入这里。 |
| `muzero/sts2_env/token_memory.py` | 3.0k | `muzero/sts2_env/token_memory_components/` | token state、encoder、bank update、aux target、serialization 分开；`token_memory.py` 只保留兼容类/函数。 |
| `sts2_env/combat_env.py` | 2.7k | `sts2_env/combat/` 或 `sts2_env/combat_*` 子模块 | combat reset/step、legal-action selection、transition_state、reward info、diagnostics 分开；hard guard 不写进环境。 |
| `muzero/sts2_env/muzero_model.py` | 2.6k | `muzero/sts2_env/model_components/` | representation/dynamics/prediction heads、objective heads、latent regularizers、factory 分开；外部仍从 `muzero_model.py` 导入模型。 |
| `legacy/train_offline_multitask.py` | 2.4k | `legacy/offline_multitask/` 或删除归档 | 如果仍需使用，按 dataset、model、loss、CLI 拆；如果不再使用，转移到归档并从默认检查移除。 |
| `sts2_env/_sim_translate.py` | 2.1k | `sts2_env/sim_translate/` | card/action/enemy/potion/reward payload 翻译分开；不要再追加新桥接协议字段到单文件。 |
| `tests/test_combat_action_hard_guards.py` | 4.3k | 多个 `tests/test_*_guard.py` | 按 guard family 拆：HP cost、X cost、end-turn、Kaiser、Insatiable、potion、selection loop、survival。 |

### 拆分顺序建议

1. 先拆 `tests/test_combat_action_hard_guards.py`：风险最低，能降低后续改 guard
   时的认知负担。
2. 再拆 `muzero/sts2_env/muzero_model.py` 与
   `muzero/sts2_env/token_memory.py`：新特征越来越多，网络/输入结构必须清楚。
3. 再拆 `sts2_env/observation_v3.py`：这是 observation 契约核心，拆前先补
   fixture/golden shape tests。
4. 最后拆 `sts2_env/env_v2.py`、`sts2_env/combat_env.py`、
   `_sim_translate.py`：这些和桥接状态机耦合最大，必须每拆一步跑 smoke。

每拆一个文件的验收：

```bash
cd packages/rl-agent
./.venv-wsl-rocm/bin/python scripts/check_muzero_file_budget.py --quiet-ok
./.venv-wsl-rocm/bin/python -m pytest <对应测试> -q
./.venv-wsl-rocm/bin/python -m muzero.train --help >/tmp/muzero_train_help.txt
```

当上表清空后，把 CI/本地检查切成严格模式：

```bash
./.venv-wsl-rocm/bin/python scripts/check_muzero_file_budget.py --fail-on-legacy
```

## Act1 恢复训练种子与 combat-quality guard 规范

当前 Act1 恢复的主 blocker 是全局战斗动作质量：模型在普通/精英战斗中仍会
在敌人无攻击或只有极低伤害压力时打纯防御/无伤害进展牌，导致战斗拖长、
战损上升、篝火被迫回血而不是升级。这个问题不应通过固定路线、固定 seed
或 boss-only 特化掩盖，必须在 combat quality 层正面治理。

### 训练 seed 规则

- 正式训练使用随机 seed：不要传固定 `--seed`，也不要使用固定 seed pool。
- 固定 seed 只允许用于 smoke、评估、回归定位、checkpoint bisect。
- Act1 恢复训练阶段默认保持：
  - `--route-heuristic-bias 0.0`
  - `--route-safety-guard`
  - `--disable-potion-reward-fast-path`
  - `--resume-without-optimizer`
  - `--resume-without-buffer`

### 全局 hard guard 规则

`meaningful_damage_guard.py` 与 `no_pressure_block_guard.py` 都属于全局
combat-quality 保护，不是 boss 专项：

- `meaningful_damage_guard.py`
  - 只在策略/search 已选 End Turn 时触发。
  - 若没有 meaningful incoming pressure，且存在安全、可支付、即时伤害或
    进展牌，则把 End Turn 改写为该牌。
  - 必须拒绝坏的 0 能量 X 费、非 lethal HP-cost、自身扣血、纯防御、
    无 follow-up 的回费/设置牌。
- `no_pressure_block_guard.py`
  - 只在策略/search 已选 `card_block_waste` 或 `card_pure_block` 时触发。
  - meaningful pressure 下必须跳过，让生存 guard 决策。
  - 无压力或 1~3 点左右的极低非致命压力下，若存在安全伤害/进展牌，则改写。
  - block+draw、block+debuff、power、hand mutation、机制答案等有进展的牌
    不应被误判为纯防御。

### 必须观察的指标

新训练启动后，TensorBoard 必须出现以下 tag 才能证明新 guard 被部署：

```text
combat_quality/meaningful_damage_endturn_guard_applied_rate
combat_quality/no_pressure_block_guard_applied_rate
normal_combat/no_pressure_block_guard_applied_rate
elite_combat/no_pressure_block_guard_applied_rate
combat_guards/meaningful_damage_endturn_guard_applied_rate
combat_guards/no_pressure_block_guard_applied_rate
```

如果 `meaningful` / `no_pressure` tag 不出现，说明当前 run 不是有效验证目标。
同时持续观察：

```text
combat_quality/card_pure_block_selected_rate
combat_quality/card_no_damage_pressure_selected_rate
combat_quality/wasteful_end_turn_rate
recent_tail/64/normal_win_rate
recent_tail/64/elite_win_rate
recent_tail/64/boss_win_rate
recent_tail/64/act1_pass_rate
```

成功方向是：`card_pure_block_selected_rate` 与
`card_no_damage_pressure_selected_rate` 下降，normal/elite 战损改善，
boss_seen 与 act1_pass 逐步上升。

## 禁止事项

- 禁止把 boss 机制、药水时机、X 费/HP cost/空过判断直接写进 `train.py`。
- 禁止把 route heuristic scoring 写进 `trainer.py` 或 `train.py`。
- 禁止用临时字符串路径在多个模块里散落拼接 log/checkpoint/diagnostic 路径。
- 禁止为绕过检查把新文件加入 legacy allowlist；新文件必须拆分。
- 禁止仅按卡牌文本正则实现核心机制；优先使用内部 id、effect profile、
  bridge runtime fields，再用标题/文本作为兜底。

## 新需求落点速查

| 需求类型 | 首选目录/文件 |
| --- | --- |
| 新 boss/怪物特化机制 | `muzero/strategy/encounters/<encounter>.py` |
| 卡牌/动作通用 feature | `muzero/strategy/action_features.py` 或拆子模块 |
| combat hard guard | `muzero/combat_quality/*_hard_guards.py` |
| 药水使用时机 | `muzero/combat_quality/potion_guard.py` / `potion_timing.py` |
| 无效防御/空过/pressure 指标 | `muzero/combat_quality/metrics.py` / `block_waste.py` |
| route candidate/scoring | `muzero/route_heuristics/` |
| search/root prior adapter | `muzero/search/` |
| TB/recent-tail/episode 聚合 | `muzero/diagnostics/episode_metrics.py` 或 `training/monitoring.py` |
| CLI 参数 | `muzero/training/cli_args.py` |
| 环境/session 构造 | `muzero/training/env_factory.py` |
| checkpoint/resume | `muzero/training/checkpointing.py` |

## 验收标准

一次文件治理改动至少满足：

1. `muzero/train.py` 仍为薄入口。
2. 新增文件 `< 2000` 行。
3. `scripts/check_muzero_file_budget.py --quiet-ok` 通过。
4. `python -m muzero.train --help` 通过。
5. 相关单元测试通过。
