# v34 锻造升级与商店删牌事务课程

## 目标

v33 已经暴露锻造和商店删牌动作，但模型没有稳定走通事务：锻造曾进入
32 次却全部取消。单纯追加训练很难解决，因为稀有入口、选择页取消偏置和
终局信用延迟共同导致“成功升级/删牌”的有效样本接近零。

v34 的目标不是规定“必须升级/必须删牌”，而是先让策略得到足够的完整成功
轨迹，学习以下动作语法：

```text
进入服务 -> 选择一张牌 -> 确认 -> 观察权威牌组突变
```

长期完整局回报仍负责学习：

- 什么时候在休息与锻造之间选择锻造；
- 什么时候花钱删牌；
- 应升级或删除哪一张牌；
- 这次投资对后续战斗、血损、复活和通关的真实价值。

## 训练期数据收集

### 稀有入口定向探索

锻造入口和商店删牌服务各自获得一个与候选数量无关的探索分支。这样商店
有十几个商品时，删牌入口不会因候选数量被稀释。v34 训练配方只在这些入口
把 epsilon 下限提高到 `0.50`，普通战斗、选路和其他商店动作不受影响。

### 事务完成引导

在权威 Selection 事务中，行为策略与 forward proposal 做显式混合：

```text
mu(a|s) = (1-q) * pi_base(a|s) + q * pi_forward(a|s), q=0.95
```

- 一旦 Confirm 合法，forward proposal 只指向 Confirm；
- Confirm 尚不合法时，只在 Select 候选间分配概率，并保留模型原有相对排序；
- Cancel/Deselect 仍通过 `1-q` 保持非零支持；
- journal 保存精确 `mu(a|s)`，V-trace 不会把引导动作伪装成 on-policy；
- deterministic held-out evaluation 完全关闭该引导。

因此这是可审计的数据收集课程，而不是环境规则替模型操作。

## 权威成功判定

UI 页面退出、点击 Confirm、金币变化都不足以记成功。

### 锻造成功

同一牌组定义计数保持不变，且至少一张既有牌的升级等级上升。

### 删牌成功

牌组总数严格减少 1；不能出现新定义；任何牌定义计数都不能增加。

只有满足上述事实，事务才进入 `completed/PREFER`。取消进入 `cancelled`，
无可证明突变进入 `unresolved`，避免错误正标签。

## 信用与奖励边界

不新增“升级 +x”“删牌 +y”的手工奖励。这类奖励会诱导模型无条件升级或
删除，而忽视 HP、金币、卡组强度和路线。v34 使用已有两条正式学习通路：

1. completion failure-credit 让策略学会完成已经进入的事务；
2. 完整 episode/Act/run return 学习入口选择和具体卡牌选择的长期价值。

## 评估

训练 journal 报告：

- 定向事务入口探索次数；
- completion guidance 次数、forward 命中和 fallback；
- 锻造 started/completed/cancelled/unresolved；
- 商店删牌 started/completed/cancelled/unresolved。

held-out 评估不使用引导，必须同时报告：

- 锻造实际牌组升级次数；
- 商店删牌实际牌组减少次数；
- Act 1/完整通关率、deadlock、血损与复活；
- 与 v33 相同 seed 的逐 seed 配对结果。

训练期能完成事务只证明数据通路已打通；只有 held-out 自主完成且完整局指标
不退化，才能宣称模型学会。

## 血统

v34 从 v33 在 75,714 环境步的 healthy evaluation anchor 做明确 model-init：

- 继承网络权重与成熟 schedule phase；
- 不继承 optimizer、replay、RNG、rollout queue 或局部计数；
- config-v13 只通过受测试的 v13 -> v14 model-init 迁移读取；
- 禁止伪装成 exact resume。

下一阶段在 held-out 事务完成率稳定后，应关闭该课程并建立新的无引导验证
lineage，而不是永久依赖 completion guidance。
