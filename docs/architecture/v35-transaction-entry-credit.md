# v35 事务入口吸收态恢复方案

## 1. 问题与证据

v34 已证明事务执行能力本身正常：训练行为策略可以进入锻造或商店删牌页面，
并在事务页内按 `Select -> Confirm` 得到权威牌组突变。问题发生在事务入口，
不是页面协议或样本数量：多个无引导 held-out gate 中，休息点的非休息动作和
商店删牌服务的模型概率都饱和到数值零；增加入口 epsilon 和完成引导只改变
行为策略 `mu`，没有改变模型策略 `pi`。

原有入口学习通道都在这个状态下失效：

```text
V-trace actor:        rho * A * grad(log pi),  rho = pi / mu
episodic imitation:   min(rho, 1) * target
entropy:              term proportional to pi
```

当 `pi(entry)` 下溢为零而课程仍令 `mu(entry)>0` 时，`rho` 也为零；因此即使
收集到大量成功完成轨迹，入口策略仍可能得到精确零梯度。事务页内部没有这个
问题，所以 v34 能学会 Confirm，却不能自主选择 Smith 或 card removal。

## 2. 非目标与边界

v35 不规定何时一定要升级或删牌，也不把某张牌写成规则。以下方案被明确禁止：

- 给锻造、删牌或 Confirm 增加任务奖励；
- 从合法动作中屏蔽休息、取消或其他商品；
- 在评估时启用事务引导；
- 用辅助 Q 头直接重排、替换或强制执行策略动作；
- 把页面退出或一次点击当成成功。

长期通关、前进距离、血损和复活成本仍由原有版本化任务目标决定。v35 只修复
“已观测成功动作无法重新获得任何策略支持”的优化病态。

## 3. 权威生命周期证据

事务 replay v4 为 upgrade/remove 记录一个不可变 lifecycle：

```text
entry state/action
  -> selection transitions
  -> exit action
  -> authoritative post-state or terminal
```

证据包含：

- operation（仅 `upgrade` 或 `remove`）；
- entry/exit 在完整事务 trace 中的精确索引；
- entry 的行为 log-probability、采集时模型概率、policy version 与动作指纹；
- `COMMITTED / CANCELLED / UNRESOLVED / DEADLOCK` 结果；
- 权威牌组突变是否验证；
- 非终局成功后的编码快照，或成功终局标志；
- 从入口到退出的事实 reward return、累计 discount 与 option 步数。

只有牌组升级等级确实增加或牌组总数确实减少一张时，结果才能为
`COMMITTED + effect_verified`。取消、无法证明的退出和 deadlock 不能携带
成功 post-state，也不能生成正入口标签。DTO、checkpoint loader 与 replay
restore 都重复校验这些约束，避免 pickle 绕过构造器后静默注入错误证据。

入口是 burn-in 后第一个学习步，退出是 trace 最后一个事实步。若极端长事务
把入口挤出有界上下文，该 trace 仍可作为 liveness/value 证据，但自动失去
lifecycle 资格；系统不会猜一个新的入口。

## 4. 单边策略支持目标

对每条 `COMMITTED + effect_verified` 证据，在当前模型下重放精确入口，并令：

```text
g = relu(log(p_floor) - log pi(entry|state))
L_support = smooth_l1(g, 0)
```

v35 首轮取 `p_floor=0.05`、权重 `0.25`。

这个目标有四个关键性质：

1. 直接作用于归一化 legal-candidate log probability，不依赖 `rho=pi/mu`；
2. 即使 softmax 概率下溢，有限的 log-softmax 仍给 entry logit 非零、有界梯度；
3. 达到 5% 后损失严格为零，不会成为永久行为克隆；
4. 只恢复探索/学习支持，不宣称 entry 优于 rest 或其他购买。

恢复到非零支持后，普通 V-trace 与完整 episode 信用才能重新根据后续通关、
血损和复活结果决定概率是否继续上升或下降。取消、未决和 deadlock 证据只计入
诊断，不产生这个正向目标。

## 5. 事务级 SMDP 入口 Q

同一条成功生命周期还训练候选独立的事务 Q 头：

```text
R_option = sum[t=entry..exit] product(discount before t) * reward_t
Gamma_option = product[t=entry..exit] discount_t
target = R_option + Gamma_option * stop_gradient(V(post_state))
```

终局成功的 `Gamma_option` 必须为零；非终局必须有权威 post-state。目标不含
“升级奖励”或“删牌奖励”，只是把既有任务 reward 按实际事务长度做正确的
semi-Markov bootstrapping。该 Q 头训练共享表征和入口价值，但不直接改写 actor。

## 6. Replay 采样与容量

成功入口是稀有标签。transaction replay v4 维护两个独立派生 strata：

- committed upgrade lifecycle；
- committed remove lifecycle。

常规 `sample_traces=4` 时，每个存在的 operation 至少保留一个采样席位，然后
再分配 selection cycle、单调完成、纠正性 deselect 和普通/deadlock 样本。
索引完全由不可变 trace 重建，不额外写入 checkpoint；restore 后采样器 RNG、
计数与 strata 必须精确连续。首轮容量为 4096 条、1 GiB，超限按既有确定性
淘汰契约执行。

## 7. ABI 与血统

- curriculum config：v14 -> v15；
- transaction trace：v3 -> v4；
- transaction replay：v3 -> v4；
- lifecycle evidence：v1；
- exact resume 必须匹配 lifecycle ABI，并拒绝 v34 replay；
- v34 只能通过受测试的 v14 -> v15 `model_parameter_initialization` 迁移。

v35 固定从下列健康 held-out 锚点初始化：

```text
run_id:        795ad740-bcf6-4707-86a0-8a3b3301a13c
checkpoint:    healthy-validation-step-000050320
checkpoint_id: 2a3cccf8-09f5-4808-9991-86af6e6f3cfc
source step:   50,320
policy/update: 814 / 814
```

继承形状兼容的网络权重与成熟 schedule phase；optimizer、rollout queue、RNG、
transaction/episodic/failure-credit replay 和本地计数全部重置。manifest 必须固定
源绝对路径、checkpoint ID、manifest/metadata SHA256、run ID、step、policy 与
update 计数，禁止 latest/fallback 查找。

## 8. 遥测与验收

训练遥测必须分开报告：

- online / transaction / liveness / episodic objective；
- support label 数、已达到 floor 数、平均 log-support gap；
- 当前模型概率、采集时模型概率、行为概率；
- upgrade/remove 各自 label 数和模型/行为概率；
- committed/cancelled/unresolved/deadlock lifecycle 数；
- SMDP Q label 与 loss；
- replay 中 upgrade/remove committed strata 的存量。

历史 `loss` 是上述有正有负目标的有符号复合和，不应按普通监督 loss 解读为
必须单调下降。

### 首轮门禁

在 0/5k/10k/25k/50k/75k 做 16-seed 无引导评估，100k 做 64-seed final audit。
通过顺序为：

1. 训练 replay 中 upgrade/remove 都持续产生 verified committed evidence；
2. entry support loss 有标签、有梯度，模型概率脱离字面零并朝 floor 收敛；
3. held-out 中出现自主 Smith/card removal，而非仅训练 guidance 完成；
4. Act 1、完整通关、deadlock、血损与复活相对固定同 seed 健康锚点不退化；
5. 关闭事务 guidance 的后续 lineage 仍保留自主完成能力。

若只满足 1 而 2 不满足，视为学习通路实现错误；若 2 满足但 3 长期不满足，
再检查入口长期 advantage/宏观信用，而不是继续提高 guidance；若 3 满足但主任务
显著退化，则回滚策略权重并调整 support floor/weight，不能用评估规则掩盖。
