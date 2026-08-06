# v37 复活经济与 Liveness 稳定性恢复契约

## 1. 事故边界与固定证据

v36 运行 `d62931b7-8dbb-4df8-bf53-09e3c290acc0` 在约 27k--30k 环境步发生了
与 v32 同族的共享躯干位移。503 次更新按时间分桶后，最后两桶出现：

- `liveness_cost_actor_loss` 从长期约 `-0.1` 扩大到 `-11.7/-7.2`；
- `liveness_credit_loss` 从约 `0.04` 翻转到约 `-1.1`；
- liveness 裁剪前梯度范数从 `0.157` 上升到 `2.539`，主梯度范数从
  `2.87` 上升到 `11.1`；
- 策略熵从 `0.163` 下跌到 `0.061`，旧 `one-hot-v1` 断路器没有识别这种
  非严格 one-hot 的软塌缩；
- 事务走廊的支持损失、入口概率和 SMDP-Q 保持稳定，因此事务恢复机制不是
  这次退化的起点。

失控运行已人工停止。事故保存点
`periodic-step-000030101` 只用于取证，禁止作为训练父本。v37 唯一允许的模型参数
父本是 v36 的 `healthy-validation-step-000025474`：

- checkpoint id：`0824662c-7801-43c7-8d97-6ca1c49af25d`
- source run id：`d62931b7-8dbb-4df8-bf53-09e3c290acc0`
- environment steps / policy version / learner updates：`25474 / 416 / 416`
- manifest SHA256：`b2c5d3d93ea43425074c552308a0173653e89ccbe552e82bd7b140b6d4586567`
- metadata SHA256：`3c26655815a20628371c17a43c0c747dacad03a7e5344d51dcbcfc982b0f13f8`

## 2. 根因

旧复活成本的边际分辨率过低，`revival_budget=64` 又允许大量低质量失败状态进入
liveness replay。正风险 actor 项持续执行 `log(pi_selected) * centered_risk`。当某个
高风险动作已经被策略压到极低概率后，继续沿相同方向更新几乎没有行为收益，却仍会
通过共享编码器/策略躯干形成高体量、方向一致的位移。战斗策略随后退化，产生更多死亡
与复活数据，再进一步放大 liveness 平面，构成正反馈。

成功 episode 的 CE 式模仿还会在稀疏战略入口形成单向棘轮。它对休息点和商店的每次
成功选择重复施加直接 logit 信号，可能压过真正需要按 HP、牌组、楼层和金币学习的
上下文偏好。该问题与事务内部 `select -> confirm` 能力不同，不能通过取消整个事务走廊
来解决。

## 3. 版本化修复

### 3.1 奖励 v6 与有限复活课程

累计复活成本改为：

```text
C_revive(k) = min(0.40, 0.010*k + 0.0009*k^2)
```

v37 的课程上限为 16 次复活。第 `k` 次复活的边际成本为
`0.0109 + 0.0018*(k-1)`；在 16 次课程范围内尚未触及 0.40 上限，因此严格递增，
不会像旧双曲线奖励在长尾处失去分辨率。复活、HP 损失和
步数成本的最大预算分别为 `0.40 + 0.55 + 0.04 = 0.99 < 1`，所以终局成功仍严格
优先于任何效率优化：模型不能通过早死来换取较少复活。

### 3.2 Liveness risk-actor 的已满足资格

仅对“正 centered risk、因此要继续压低已选动作”的 actor 标签应用 1% 概率地板：

- 当前 `P(selected) < 0.01` 时，停止该 actor 方向；
- liveness value/cost critic 标签不受影响；
- centered risk 为负的恢复方向不受影响；
- direct/cycle/completion/contrast 事实信用不受影响。

这不是把 `log(pi)` 数值裁掉，也不是屏蔽危险动作；它只终止已经实现行为目标后仍会
移动共享躯干的冗余 actor 梯度。新增遥测
`liveness_risk_actor_saturation_suppressed_labels` 用于证明通道实际命中。

### 3.3 成功模仿棘轮表面豁免

完整成功 episode 在 `rest_site` 与 `shop` 上仍提供全部 run/Act/combat value 和
复活/效率事实目标，但不再重复提供 selected-action CE 模仿，也不再占用
`fresh_policy_sequences` 的保留槽。入口偏好继续由在线 V-trace、事实 SMDP option-Q
和最终回报学习。其他表面保持原行为。

### 3.4 `policy-collapse-v2` 断路器

v2 按非强制决策的归一化熵 `H(pi)/log(|A_legal|)` 检测软塌缩：至少 16 个有效决策，
平均归一化熵不高于 0.10，且至少 75% 的决策不高于该阈值，连续 8 个 learner batch
后，将熵权重临时抬至至少 0.012，共 16 次更新。

旧 importance-ratio 的 `rho≈1 且未裁剪` 条件只保留为诊断遥测，不参与 v2 触发；
否则健康的全新 on-policy batch 也会被误判。新增 `normalized_entropy` 和
`entropy_breaker_soft_collapse_condition` 记录触发依据。

## 4. 明确不变项

v37 的 `[transaction_learning]` 与 `[transaction_exploration]` 必须和 v36 完全相等：

- 两侧 5% 支持走廊；
- 入口探索 `0.50`；
- 面内完成引导 `0.95`；
- completion CE `0.15`；
- lifecycle SMDP-Q `0.10`；
- replay 容量、burn-in 与所有事务资格不变。

不得新增锻造、删牌、休息的手工奖励，不得 mask/改写模型动作，也不得把恢复策略写成
固定 HP 阈值。

## 5. Lineage 与门禁

配置版本升为 `sts2-relational-curriculum-config-v17`，learner dynamics 升为 v2，奖励
升为 v6。这些都改变训练语义，因此从 v36 健康锚点只能进行显式 **model-init**：

- 继承兼容网络参数与成熟 schedule clocks；
- optimizer、actor snapshot、rollout queue、episodic replay、failure-credit replay、
  transaction replay、RNG 和 lineage 计数全部重新开始；
- 禁止伪装成 exact resume，禁止自动寻找 latest checkpoint。

评估从 0 到 90k 最长间隔为 10k，避免再次出现 25k 到 50k 的观测盲区。门禁永不
回滚：10k 前只记录并固定证据，10k 后若确认塌缩则停止并保留失败候选；绝不重放已经
丢弃的训练前缀。

## 6. 验收

启动前必须通过：

1. reward v6 的凸边际、上界和终局优先级测试；
2. 低概率正风险 actor 被豁免、负风险恢复 actor 保留、critic 保留的梯度测试；
3. `rest_site/shop` 仅豁免成功 CE、value 标签仍存在，fresh sampler 不浪费槽位；
4. soft-collapse 触发和健康高熵/健康 on-policy 不误触发测试；
5. v16 -> v17 只能 model-init、exact-resume 仍 fail-closed；
6. v37 launcher 对 checkpoint id、两份 SHA、source run/step/policy/update 全部固定；
7. v37 与 v36 的事务配置对象严格相等。

运行中每个 gate 至少报告：复活数/预算耗尽率、HP loss、Act/run 通过率、deadlock，
`liveness_cost_actor_loss`、liveness/main 梯度范数、饱和豁免标签数、归一化熵、断路器
状态，以及事务入口/完成/取消率。若 liveness actor 再次数量级放大或归一化熵持续低于
阈值，应停止 lineage，而不是继续用事故权重训练。
