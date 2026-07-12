# STS2 RL Agent — 效果差的根因诊断 (2026-06-02)

> 方法：8 路并行深挖（学习循环 / 奖励 / 世界模型与planner / PPO / 探索 / guard泛滥 / 环境与观测 / 日志取证）→ 综合 → 对抗式复核。复核独立读码核实，6 条根因中 5 条直接确认、1 条部分修正。所有 file:line 均经二次核对。

---

## 0. 先纠正之前的解读偏差（已确认）

- **`RecentTail win=0 / normal=0 / elite=0` 不是"从不打赢单场"**：full run 模式不记录单房间胜率，这几个值在全流程里是**指标缺失**。但 TensorBoard 里 `act1_pass_rate / boss_win / elite_win` 是**真实记录**且 34 个 fullrun 全是 0.000 —— 即 agent **确实从未清过 Act 1 / 打过 boss/elite**。两件事都成立。
- **"狂点 end_turn" 大部分是日志假象**：`end_turn_pre_dispatch_audit.jsonl` 有选择偏差；用无偏的 `intent_combat_quality` 看，单局战斗 **出牌:结束回合 ≈ 405:133（≈3:1）**，agent 确实在出牌。后期被动化是 reward + 无探索导致的局部最优**症状**，不是字面意义的空过刷屏。
- **死亡卡组不是"只有 4 打击 4 防御"**：模型会选奖励牌、卡组有改善；但**质量偏弱、升级几乎不生效**（~72% 死亡卡组 0 张升级牌；74/99 次升级决策实际 `upgraded_delta=0`）。
- **死亡中位楼层 ≈ 11**，多死于 Act-1 普通战，很少摸到 Act-1 boss。

**一句话定性**：不是"从没学会"，而是**全流程训练机制本身无法让学习累积 + 奖励无法教会获胜 + guard 在破坏信用分配** —— 是机制性/退化性失败，不是单点 bug。

---

## 1. 排序后的真·根因（code-verified）

### RC-1【critical】每次重启都丢弃 replay buffer + optimizer，且每 1–3h 就 kill 重启
- `cli_main.py:496-497` `load_buffer = not args.resume_without_buffer`、`load_optimizer = not args.resume_without_optimizer`；启动脚本两个 flag 都传了。
- **关键新证据**：`muzero_step_00150405/metadata.json` 里 `buffer_size=23395, replay_buffer_saved=true`；磁盘上 `replay_buffer.pkl=2.3GB`、`optimizer.pt=395MB` **都在、都可重载**，却每次重启被丢掉。
- 取证：5/14–5/16 连续 14 次重启，total_steps +62k 但 episode_count 只 +416，每次 buffer=0；最长一段才 488 个 optimizer step，目标段只有 29。某次出现过 ~3% 普通战胜率，**下次重启就清零**。
- `cli_main.py:1056` 训练门槛 `len(buffer) >= min_buffer_size(=500)`，所以每次空 buffer 重启先白跑 ~500 transition 才开始有梯度。
- **为什么会有这俩 flag（复核发现）**：buffer 重载是 **schema-gated**（`checkpointing.py:177-200`，要求 `allow_exact_resume`、网络/replay schema 完全兼容）。dev 频繁改 guard/aux-head → schema 变 → buffer 被静默拒绝/崩溃 → 于是加了 `--resume-without-buffer` 绕过。**这正是你那条 "prefer root fix over workaround" 记忆的情景**：根治是 schema 版本化迁移，而不是丢 buffer；光删 flag 仍可能撞 schema gate 静默空启。

### RC-2【critical】54M 参数模型，batch=8 / n-envs=1 / 每 50 步才 1 次更新 → 梯度饿死
- 启动：`--batch-size 8 --n-envs 1 --updates-per-train 1 --unroll-steps 5`；`cli_args.py:124` `train-every` 默认 50。
- `network.pt≈217MB(fp32) ≈ 54M 参数`。param:有效更新 比 > 3M:1。
- 单环境样本时序强相关，违反 Adam 的 i.i.d 假设；梯度是噪声而非可用信号。
- ⚠️ 注意：`cli_args.py` 默认其实是 batch=32 / updates=3 —— **饿死是启动脚本覆盖出来的**。

### RC-3【high，部分修正】combat direct 模式缺真正的策略改进算子 + 无探索
- `--combat-policy-mode direct` 下不跑 MCTS；`planner_memory_profile.py:73-80` 强制 `steps=1, beam=1`（`branch_disagreement_mean=0` 即无前瞻）。
- **value target 用网络自身未训练的 value 头**：`self_play.py:613` `root_value = initial.value`，n-step bootstrap 自我引用、无搜索回填。
- **无熵正则**：`train_step.py` loss 求和里没有 entropy 项，`cli_args.py` 没有 `--entropy-coef`；`losses.py:41-42` 纯 CE。
- 温度：`self_play.py:484-487` `T=1-progress*0.9`；resume 在 150405、total=180000，progress=0.836 → **T≈0.248**（低但非贪婪；argmax 仅 T≤0.05）。
- **复核修正**：policy target 不是"纯网络 softmax"——`self_play.py:702-724` 在 softmax 前混入 1-step rollout Q（q0.75/objQ0.50/risk0.35/-unc0.35）+ 手写 quality_bias，存在一个**很弱的**改进算子。真正的缺口是 **value 端自我引用 + 无熵/温度地板**。

### RC-4【critical】guard 在自博弈时同时改写"执行动作"和"训练目标"（信用分配被腐蚀）
- `self_play.py:940-957`：先存 `pre_guard_action_idx`，再用 `_apply_post_search_action_hard_guards` 替换 `action_idx`，然后 `retarget_search_policy_after_hard_guard` 改写 `search_policy`。
- `post_search_policy_retarget.py:119-130`：override 时 `rewritten=zeros; rewritten[final_idx]=1.0` —— **硬 one-hot**。
- `self_play.py:1446-1452` 存的是 guard 动作 + one-hot；`:1480` `env.step()` 执行的也是 guard 动作。`cli_args.py:65-66` 默认 `'full'`。
- **后果**：**被训练的策略 ≠ 实际执行的策略**。网络被训成模仿 ~25 个手写启发式，永远收不到"自己选错"的梯度（如 card_reward 100% `skip_blocked`，是 guard 逼的，策略本身想 skip）。这是把强化学习降级成了"模仿一堆补丁"。

### RC-5【critical】奖励无法教会获胜：无胜利奖励 + 打架净负 + 正向塑形门槛在死亡楼层之上
- `env_v2.py:549-589` 全流程 reward **没有 win/act-clear 项**；sandbox 的胜利奖励 (`headless_sim_bridge_client.py:788-818`) 只在 `_sandbox_encounter_id` 存在时触发，**full-run 永不触发** → 0 胜利信号 → 朝"获胜"方向没有任何梯度。
- HP 不对称 `reward_constants.py:12-13`：`ENEMY_HP_DELTA=0.01` vs `PLAYER_HP_LOSS=0.03`（3:1）→ 主动交火局部为负。（复核补充：boss 有 tier 软化 0.15、elite 1.5，并非全程死板 3:1，但普通战仍是 3:1。）
- `FLOOR_CLEAR_MIN_FLOOR=11`（`reward_constants.py:131`），而 agent 活在 1–14 层、中位死亡 11 层 → **几乎整段生命周期只有惩罚**。
- 死亡惩罚 `BASE 6.0 + 2.0*missing_hp + 3.0*floor_norm`（≈ -8..-11），与实测 reward 吻合 → return 分布**单边负、低方差**，value 头预测一个常数负值就能压低 loss，没有动作区分度。

### RC-6【判定】PPO 与 MuZero 同样的方式失败 → 根因在环境/奖励/机制，不在算法
- git：PPO（`train_attention_policy.py`/`aux_maskable_ppo.py`）最后改于 **2026-04-22**；MuZero HEAD **2026-05-14**。**PPO 已废弃，CLAUDE.md 说它 "active" 是过时的**。
- PPO `step_819200`：starter/full-run **0/100 胜、平均 1.0 步/局**（瞬间坍缩）。
- 同一 PPO `step_170240/180224` 在**精选中期卡组**上 `resolved_win_rate=0.68–0.71`、`boss_wr=0.375`、`normal_wr=1.0`。
- 复核确认 PPO 的 masking/GAE/优势归一化**都正确**。
- **结论**：两个独立学习器、同一套 env/reward/guard，**同样坍缩** → 调算法是南辕北辙；且 PPO 在好卡组+sandbox 下能赢 → **学习机器本身是好的，坏的是 full-run-from-starter 的训练设置**。

---

## 2. 症状 vs 根因（别再打症状的补丁）

| 观察到的"症状" | 真实归因 |
|---|---|
| end_turn 偏多 | 大部分日志假象（实际 3:1 出牌）；剩余被动化 = RC-3(无探索)+RC-5(打架净负) 的局部最优 |
| 死亡卡组偏弱/不升级 | RC-4(card_reward 被 guard 逼) + 升级管线 bug（74/99 次升级 0 生效）+ RC-5(无 build 奖励) |
| episode reward 负 | RC-5：无胜利奖励、死亡惩罚主导 |
| 795 步超长局 | `env_v2._make_frontier_refreshed_response` 的 no-op 刷新被计入 step（基本无害，但污染指标） |
| 0% 通关 | RC-1/2/5 叠加：学习不累积 + 饿死 + 奖励教不会获胜 |

---

## 3. guard 泛滥的判决：不是浪费，是**在主动破坏学习**

- ~24K 行 guard/诊断在 ~1 周内加入；每个 run 都以新 guard 命名（symptom-chasing treadmill）。
- 致命点是 RC-4 的 **one-hot retarget**：guard 不只在运行时覆盖动作，还把训练目标改写成 guard 动作的 one-hot → 网络被训成模仿启发式，永远学不到自己选择的后果。
- 这些 guard 编码的恰恰是 agent 本应学会的战术（survival_non_endturn / no_pressure_block / meaningful_damage / strategic_defer / potion_bad_use / late_normal_* …）。
- **判决**：把 ~25 个 guard 收敛到 2–3 个真正的"安全/死循环恢复"guard（如 selection_loop_guard），删掉所有战术偏好类 guard；**且无论如何不让 `retarget_search_policy_after_hard_guard` 改写学习目标**（guard 只当行为策略包装，离线 off-policy 用模型自身分布做 target）。

---

## 4. 算法 vs 环境：合并到一条路线（你的第 5 点）

- **保留 MuZero**（HEAD、在积极开发、token_memory_v1 是目标架构）。
- **归档/删除 PPO 路径**（自 04-22 废弃）：把 `train_attention_policy.py`、`aux_maskable_ppo.py`、`omni_attention_policy.py` 等移入 `legacy/`，并在 `CLAUDE.md` 标注"AuxMaskablePPO 已被 MuZero 取代"。
- **从 PPO 继承两个教训**：(a) 它在精选卡组 sandbox 能到 68% → **学习机器没问题**；(b) 它和 MuZero 同样坍缩 → **先修共享的 env/reward/regimen/guard，别再双线调算法**。
- **采纳 PPO 的课程化**：strong-deck sandbox → `snapshot_starter_early_boost` 过渡 → full-run（管线已存在 `train_attention_policy.py:621-631`），别指望 full-run-from-starter 直接学会。

---

## 5. 整改路线图（按优先级）

### Tier 0 — 立刻停止（这些在杀死学习）
1. **停止"每 1–3h kill 重启"的开发循环**：改成一次长跑（数天），中途不改代码；guard/诊断改动要么热加载，要么排到计划 checkpoint 之间落地。
2. **默认不再传 `--resume-without-buffer --resume-without-optimizer`**（buffer/optimizer 已在磁盘、可重载）。**配套修 schema-compat gate**（版本化迁移），否则光删 flag 仍会撞 `checkpointing.py:177-200` 静默空启。
3. **不再用 guard 改写的目标训练**：自博弈数据生成时关 guard（`--combat-hard-guard-policy off`）验证存的是原始策略；若 guard 必须跑，就**只执行 guard 动作、但 target 存 pre-guard 动作与原 `search_policy`**（离线 off-policy）。
4. **冻结新增 `*_guard.py`**，直到 guards-off 的裸策略出现非零胜率。

### Tier 1 — 高性价比配置（本周可做）
5. 吞吐：`--batch-size ≥128`（VRAM 不够就 micro-batch 8 累积 16 步）、`--updates-per-train ≥4`、`--n-envs 4–8`。
6. **重平衡 loss**：把 ~15 个 world-model/aux 权重临时压低（future-world-aux≈0.05、surface-*≈0.1、semantic/state-consistency≈0.25），让 policy+value+reward 主导；或按各项幅度归一化后再求和（`train_step.py:681-708`）。看 `loss_ratio/*` 验证 policy/value 占比上升。
7. **加探索**：加 `--entropy-coef`（接到 `train_step.py:345-357`）；采集温度地板抬到 0.5–0.7 且 resume 后与 total_timesteps 解耦（`self_play.py:484-487`）；可对 direct_probs 注 Dirichlet/ε-uniform（排除 end_turn）。
8. **奖励修复**：env_v2 full-run 加显式**胜利奖励（≥+15，> 死亡惩罚量级）**并镜像 sandbox 的胜利奖励；修 3:1 HP 不对称（普通战把 PLAYER_HP_LOSS 降到 ≈0.01）；`FLOOR_CLEAR_MIN_FLOOR` 从 11 降到 2–3 并加"每清一房"正奖励；保证有能量时 end_turn 不是零成本。

### Tier 2 — 更深的修复
9. **停止自蒸馏（最高杠杆之一）**：combat target 改用 rollout-planner Q 的 softmax（或重开 MCTS 生成 visit-count target）；`root_value` 用 planner Q 而非裸 value 头（`self_play.py:613`）。
10. **采集时重开前瞻**：`--planner-memory-profile eval/max`（steps≥4/beam≥4）或 MCTS；验证 `planner_q` 把"攻击/斩杀"排到 end_turn 之上。
11. 修 latent 坍缩：`--latent-gaussian-reg-weight` 0.005 → 0.1–0.5，监控 `latent_reg_var_mean` 回到 ~1.0。
12. MuZero 可逆 value 变换 `h(x)=sign(x)(√(|x|+1)-1)+εx`（`muzero_model.py:72-138`），压缩 -10..-17 的终局尖峰。
13. **要么右尺寸化模型**（d_model 128 / world_layers 3 / 更少 slot → ~10–15M），**要么真正提吞吐**，二选一。
14. **课程化**：sandbox 强卡组 → snapshot 过渡 → full-run。
15. **修升级管线**：查清 74/99 次 deck-upgrade `upgraded_delta=0` 的原因（no-op？错屏？策略拒选）。

### Tier 3 — 合并与卫生
16. 归档 PPO；把 ~25 guard 收敛到 2–3 个安全 guard，删战术偏好 guard。
17. repo 卫生：39 个 `_tmp_*.py`、102 个 `.sh` 归整到 `scripts/`；commit 别再写 "save"。
18. 加 **win-rate 回归看板**（按 total_steps 串联 run lineage），重启后胜率清零要告警 —— 这本可第一时间发现 buffer/optimizer 重置抹掉了 3% 增益。

### 先验证（每项 <30 min，定症 vs 因）
- 无偏 action-family 直方图（确认 play:end ≈3:1，破"狂点 end_turn"伪命题）。
- **guards OFF vs FULL 跑 ~500 局 A/B**（已有 56 个 guard 遥测键），量化策略对 guard 的依赖。
- planner sanity probe：固定战斗态，断言 `planner_q` 把斩杀/攻击和 end_turn 拉开 margin。
- `train_step.py:913-922` 的 loss-spike skip 率打到 TensorBoard（确认不是在静默丢弃本就稀少的更新）。

---

## 6. 还需验证 / 开放问题
- **schema-compat gate 是 buffer 被丢的真正动因**（复核认为被低估）：根治是 replay/network schema 版本化 + 迁移，否则一切"保留 buffer"的修复都可能静默失效。
- aux 占 loss ~0.58、`latent_reg_var_mean=0.178` 来自 TB 标量，复核未独立打开 → 中等置信，建议跑前先核。
- `headless_sim_bridge_client.py:797` `combat_won = cur_hp>0.0` 把"截断仍活着"判为胜（**潜在地雷，当前未启用**）—— 重开 sim 课程前必须改成显式胜负标志，否则会教出"拖到截断=赢"。
- `observation_common.py:1848` `_parse_act` 靠尾随数字解析，live 的 `ACT.OVERGROWTH` 无数字 → act 特征恒 0（sim/live 不一致），建议改成名称→索引映射。

---

### 最高杠杆 3 件事（如果只做三件）
1. **保住学习状态**：长跑、不丢 buffer/optimizer（+ 修 schema gate）。
2. **停止 guard one-hot 改写训练目标**（恢复一致 MDP）。
3. **给 full-run 加胜利奖励 + 修 HP 不对称 + 降 floor-clear 门槛**（造出朝获胜的梯度）。
其余（吞吐、熵、planner Q-target、latent 正则、课程化）是让上面三件真正收敛的放大器。
