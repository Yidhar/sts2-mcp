# 有界长训与固定种子整局评估

## 本轮实验边界

- 入口: `packages/rl-agent/sts2_rl/macro_training_campaign.py`。
- 课程: `packages/rl-agent/config/experiments/hybrid_macro_credit_v2_reward_v9_long.toml`。
- 从旧宏观 publication update 20 **仅继承模型权重**，建立新 lineage；优化器、随机状态、update 编号归零。不是旧 checkpoint 的精确续训。
- 修复动作角色哈希碰撞: 哈希只用于学习特征；策略分组改用独立的精确语义标签绑定。编码身份 v18 / snapshot v5，旧快照不能冒充新快照。
- 保持旧课程的训练参数: 每 cohort 1 个同根样本、12 个后续宏观决策、lambda 0.95、KL 0.02、复活预算 64。不同时改变算法和采样长度。
- 训练种子为偶数，从 index offset 20，即 13400040 开始；失败 cohort 不跳种子、不推进已提交游标。
- 这是实验续训，不表示单次分支回报已经具备统计置信度，也不表示策略已被证明改善。

## 自动执行顺序

1. 将父 publication 复制为本轮只读基线，记录 SHA256；冻结配置和声明范围内的源码哈希。
2. 基线依次自然贪心运行固定种子 13401001、13401003、13401005、13401007。
3. 只有基线至少一局完整结束且无执行错误，才进入训练。
4. 每次子进程只提交 1 个完整 cohort；后续从本轮 checkpoint 严格恢复。保存每次调用后的 publication 快照。
5. 遇到训练/协议错误立即停训，不通过换种子绕过。预算到期先 SIGINT，允许保存状态和关闭模拟器；未提交 cohort 丢弃。
6. 对最后一个有效 publication 跑同一组完整游戏评估，生成逐种子对比。失败的训练仍保留失败状态，评估成功不会掩盖错误。
7. 不自动替换基线，不按训练 loss 或最佳单局挑选模型，不自动开启下一轮。

默认 **8 小时为整轮总预算**，包括训练前后评估。每局最多 1200 秒或 8000 个原生动作；为最后 4 局预留时间。若基线很快结束，剩余时间可用于更多训练；最多 64 个新 update。预算限制可能打断最后一个 cohort，只有之前已提交的更新有效。所有子进程由一个前台 WSL 进程监管；不是计划任务或隐藏常驻服务。

## 成绩解释

评估使用同一个模拟器验证入口、模型 actor、语义执行器和 reward-v9；不强制根动作、不使用分支搜索、不更新权重、不在终点补 V 值。

记录从 reset 开始的实际累计折扣回报，以及最终幕、楼层、HP、复活次数、累计损血、原生动作数、typed run_result 和 terminal_reason。

- `complete`: 环境事实真正终止、未截断，且 typed run_result 为 victory/defeat。
- `censored`: 时间/步数限制或环境截断。楼层和回报只是部分轨迹，不能算作通关、失败或完整配对差值。
- `failed`: 编码、协议、执行错误或缺少可信终止结果。

**64 次复活辅助模式不等于普通零复活通关。** 同样通关但复活更多可能更差，必须同时看实际回报与复活次数。4 个固定种子只是小样本描述性配对，不代表统计显著或泛化认证。

## 证据文件

本轮 campaign 目录与 mutable run 目录互相独立，均位于仓库外 artifact root。

| 文件 | 作用 |
|---|---|
| `plan.json` | 总预算、种子、父模型哈希、源码与评估协议身份 |
| `status.json` | 当前阶段、活动对局、已提交更新、错误状态 |
| `progress.md` | 可直接阅读的逐局对比表与当前对局事实 |
| `course.toml` | 冻结配置 |
| `models/` | 原始基线与每次训练调用后的 publication 快照 |
| `evaluations/{baseline,final}/seed-*/summary.json` | 逐局结果与策略来源 |
| `evaluations/{baseline,final}/seed-*/final-state.json` | 最后真实环境状态 |
| `evaluations/{baseline,final}/seed-*/trajectory.jsonl` | 紧凑事实轨迹；不是完整重放/同根证明 |
| `invocations/*/{process.json,stdout.log,stderr.log,exit.json}` | 子进程身份、输出、退出与预算中断证据 |
| `comparison.json` | 逐种子完整配对和删失计数；不自动给质量认证 |

旧 monitor 中的短分支截止楼层仍属于训练窗口，不要把它与这里的整局 `ending_floor` 混用。运行后不得编辑冻结的训练/评估源码；需要修复时停下当前实验，建立新的协议身份与 lineage。

## 启动示例 (WSL)

在 `packages/rl-agent` 目录，用已配置的 ROCm Python 执行。路径使用 Linux/WSL 绝对路径，`STS2_ARTIFACT_ROOT` 指向仓库外 runtime；为输出目录选择尚不存在的新名字。

```bash
python -m sts2_rl.macro_training_campaign \
  --config /absolute/checkout/packages/rl-agent/config/experiments/hybrid_macro_credit_v2_reward_v9_long.toml \
  --parent /absolute/runtime/hybrid_runs/old-run/publication.pt \
  --output-dir /absolute/runtime/analysis/new-campaign \
  --run-dir /absolute/runtime/hybrid_runs/new-lineage \
  --lineage-id new-lineage \
  --sim-exe /absolute/runtime/dependencies/HeadlessSim.exe \
  --device cuda --hours 8 --episode-seconds 1200 --max-native-steps 8000 --max-updates 64
```

CLI 拒绝复用已有 campaign/run 目录。外部手动中止不会自动续跑；先查看已提交 checkpoint 和原因，不能盲删 writer lock 后重启。
