# v36 事务策略恢复契约

## 问题

v34/v35 已证明训练行为策略能进入并完成锻造与商店删牌事务，但无引导策略曾先后
陷入两个相反的吸收态：`P(锻造)=0`，以及引入单向入口支撑后
`P(锻造)≈1`。这不是“模型已经学会何时锻造”的证据，而是归一化策略在边界处
失去有效梯度。与此同时，休息点/商店的 `entry -> cancel -> entry` 周期会终止
episode，却可能让首次事务入口带来的后续收益证据丢失。

## 正式语义

1. **双向支持走廊**：仅对已验证完成的 upgrade/remove lifecycle 生效。事务入口
   与其余合法动作的聚合概率均须不低于 `p_floor`；任一侧低于 floor 时使用直接
   log-probability barrier，进入走廊后损失严格为零。
2. **策略选择仍由模型学习**：支持走廊不奖励锻造、删牌或休息，也不规定谁应当
   更大。上下文选择继续由完整 episode 的事实回报、V-trace 与 SMDP option-Q
   目标学习，输入包含当前 HP、牌组、楼层、遗物、金币和合法候选。
3. **事务内部直接信用**：成功路径的 select/confirm 与精确重复 cancel 周期继续
   使用事实构造的归一化策略 CE；入口步被排除，避免将 guidance 变成永久模仿。
4. **无行为规则**：不 mask、改写、强制确认或按卡牌/事件 ID 写死选择。

## 门禁与控制流

v36 不允许回滚：`evaluation_guard_failure_action="stop"` 且
`evaluation_guard_max_rollbacks=0`。0 与 5k 的失败是不可变诊断检查点，消费该
gate 后继续向前，绝不重放相同 0→5k 前缀；从计划 gate 10k 起，失败才会停止
lineage，并保留精确失败候选。比较使用**计划 gate**而不是评估实际落地的 episode
边界，避免长 episode 意外改变门禁语义。

## 初始化与验收

v36 是从 v35 第三次 5k 的固定 `guard_failure_evidence` 保存点进行明确
model-init 的新 lineage，不伪装 exact resume。网络参数和成熟调度时钟继承；
optimizer、rollout、replay、RNG 与本地计数全部重置。验收至少检查：

- 两侧策略概率都不再为字面 0/1，support loss 在走廊内归零；
- 无引导评估中 rest/forge 的选择随 HP/牌组等状态变化，而非固定比例；
- forge/removal 的完成率上升，entry/cancel 周期下降；
- 0/5k 失败不产生任何 rollback 事件，10k 以后失败只停止并保留候选；
- Act 1、run、复活、血损与 deadlock 必须用固定 held-out seed 单独报告。
