# MuZero Combat Regression Execution Pack

日期：2026-04-29
项目根目录：`E:\game\project\sts2_mcp`
RL 目录：`E:\game\project\sts2_mcp\packages\rl-agent`
用途：这是一组可以直接发给 Claude / 其他代码执行 agent 的 Markdown 任务包。

本执行包不是新的 orchestration 框架，不包含 worker JSON、不包含调度脚本、不要求创建额外项目脚手架。
它的目标是把当前 MuZero combat sandbox 回归问题拆成清晰、可执行、可验收的工程任务。

---

## 1. 先读顺序

给任何执行 agent 时，建议按下面顺序读取：

1. `README.md`：总目标、禁止事项、并行规则。
2. `00-repo-map-and-evidence.md`：代码地图、当前证据、已知陷阱。
3. `08-worker-task-cards.md`：所有可分派任务卡片。
4. 根据被分配的 Task ID 阅读对应专题文档：
   - A 系列：`01-diagnostics-first.md`
   - B 系列：`02-end-turn-action-quality.md`
   - C 系列：`03-bridge-fast-step.md`
   - D 系列：`04-card-lifecycle-action-outcome.md`
   - E 系列：`05-boss-mechanics.md`
   - F 系列：`06-direct-planner-replay-imitation.md`
   - 验收：`07-tests-validation-rollout.md`

如果只发给一个 Claude worker，让它先实现 `08-worker-task-cards.md` 里标记为 P0 的任务，并要求严格按该卡片输出“改了哪些文件 / 如何验证 / 还剩哪些风险”。

---

## 2. 总目标

当前训练已经能打出部分基础能力，但进入瓶颈和回归期，尤其是：

- 观测到空过、能量浪费、0 能量打 X 费牌。
- boss 胜率波动大，部分 boss 机制处理不稳定。
- Kaiser Crab 面向 / 背刺机制的 candidate detector 曾经基于错误字段语义。
- `wasteful_end_turn_rate` 与人类观察、bias 应用指标口径不一致。
- exhaust / refund / retain / transform / enchant 等卡牌生命周期策略没有被完整建模。
- direct planner 的 Q-like / lookahead value 在 latent drift 高时可能反而带偏。
- bridge fast step 不能靠长固定等待，否则 combat sandbox 吞吐会被直接打爆。

目标态不是“最小 patch”，而是形成完整的 combat 决策建模闭环：

1. **可观测**：selected action 的原因、当时状态、候选动作、机制风险可以被 JSONL/TB 精确定位。
2. **可分类**：end_turn 分成 forced / strategic defer / bad，而不是单一 wasteful。
3. **可动态**：X-cost、回费、抽牌、费用降低、卡牌变化、消耗/保留/复制都基于当前状态和内部效果 profile，而不是静态文本猜测。
4. **可机制化**：Kaiser / Ceremonial / Insatiable 等 boss 的关键机制进入 observation/action semantics/aux heads/planner 评分。
5. **可高速**：bridge fast step 使用状态稳定性判断，不使用长 sleep。
6. **可训练**：direct planner 在模型不可靠时降权 lookahead，replay 调度不过度 boss 采样，允许人类 demo 做模仿学习突破窗口。

---

## 3. 总禁止事项

执行时请遵守：

1. **不要只做最小兜底 patch 然后留下 TODO**。每个任务至少要完成：
   - 数据字段
   - 训练/诊断指标
   - 测试或离线验证
   - 文档/注释更新
2. **不要用长固定 sleep 解决 bridge 状态刷新问题**。combat sandbox 是快速战斗训练器，等待必须依赖状态版本、frontier、animation/queue 状态。
3. **不要用 `enemy.side` / `action.target.side` 判断 Kaiser 左右**。实际 `side` 是敌我阵营，不是左右位置。左右应由敌人 `powers[].id` 中的 `BACK_ATTACK_LEFT_POWER` / `BACK_ATTACK_RIGHT_POWER` 推断。
4. **不要把所有 exhaust/refund/special cards 都粗暴标成“应该跳过”**。能打不代表该打，但不打必须有明确未来理由；不能让模型学成怂。
5. **不要用卡牌文本正则作为主路径**。文本正则只能是 fallback。优先从游戏内部 id、card data、powers、modifiers、card effect profile 导出结构化字段。
6. **不要只看全局 boss 聚合指标**。Kaiser/Ceremonial/Insatiable 必须有 encounter-specific namespace，否则会被其他 boss 稀释。
7. **不要把 potion 问题归因于“模型不愿意用”或“模型乱用”而不做时机/消耗/slot 刷新诊断**。
8. **不要把 direct planner 的 Q 当成永远可信**。latent drift / Q MAE / legal F1 / branch disagreement 高时必须降权。
9. **不要并行改同一个高冲突文件**，特别是 `packages/rl-agent/muzero/train.py`。如果多 worker 并行，必须明确 ownership。

---

## 4. 推荐阶段顺序

### Phase 0：只加诊断，不改变策略

先做 A1/A2/A3/A4：

- selected end_turn context dump
- action offender dump
- X-cost dynamic diagnostics
- potion transition diagnostics

验收：能在一次短训练中解释“为什么选了 end_turn / X-cost / potion”。

### Phase 1：修正 end_turn 与 action quality 口径

做 B1/B2/B3：

- EndTurnTaxonomy
- strategic skip 缩窄
- refund-no-followup 重算

验收：`bad_end_turn_selected_rate`、`strategic_defer_end_turn_selected_rate`、`forced_end_turn_selected_rate` 能同时存在且口径合理。

### Phase 2：bridge fast step 正确化

做 C1/C2：

- transient only-end-turn guard
- Python fast step short polling

验收：等待时间 p95 不明显高于原 fast step； transient only-end-turn 不再泄漏成可选动作。

### Phase 3：卡牌生命周期和动作后果建模

做 D1/D2/D3：

- CardEffectProfile 覆盖 lifecycle/cost/hand/pile/effect/mechanism
- observation/action token 增强
- future-world aux heads

验收：武装、保留、消耗、回费、复制、变化、附魔等不再只能靠文本猜。

### Phase 4：boss 机制专项

做 E1/E2/E3：

- Kaiser facing semantics
- Ceremonial one-card/stun/lock 机制
- Insatiable strategic skip/refund offenders

验收：每个 boss 都有独立 namespace 和可解释 offender dump。

### Phase 5：planner/replay/imitation

做 F1/F2/F3：

- drift-gated direct planner
- encounter-balanced replay
- human demo imitation dataset

验收：模型进入瓶颈后可以用 demo 和更稳的 planner 突破，而不是靠 MCTS 或随机训练硬磨。

---

## 5. 并行规则

可以多 worker，但请按文件 ownership 分组，避免冲突：

- Worker A：`train.py` 诊断/指标，只做 A/B 类，不碰 bridge C#。
- Worker B：bridge C# fast step 和 payload，只做 C1，不碰 `train.py` 策略。
- Worker C：card effect profile / observation / semantic action，只做 D 类。
- Worker D：boss mechanics，只做 E 类，必要时只对 `train.py` 增加 metrics，不改 planner。
- Worker E：planner/replay/imitation，只做 F 类，必须等待 A/B 诊断字段稳定后再改权重。

高冲突文件：

```text
packages/rl-agent/muzero/train.py
packages/rl-agent/sts2_env/observation_v3.py
packages/rl-agent/sts2_env/semantic_action.py
mods/sts2-bridge/Scripts/BridgeGameApi.cs
mods/sts2-bridge/Scripts/BridgeGameApi.EnvPayloads.cs
```

并行时同一时间最好只有一个 worker 写 `train.py`。

---

## 6. Worker 输出格式

每个 worker 完成后必须输出：

```md
## 完成内容
- ...

## 修改文件
- path1
- path2

## 新增/变更指标
- namespace/name: meaning

## 测试命令
```powershell
...
```

## 验证结果
- pass/fail
- 关键日志片段或指标

## 风险与后续
- ...
```

不能只说“已完成”。必须给出可复现验证路径。

---

## 7. 与旧单文件 task list 的关系

旧文件：

```text
docs/muzero-combat-regression-task-list-20260429.md
```

仍保留，作为长版背景。
本目录是更适合分发给 Claude worker 的多文件执行包。
