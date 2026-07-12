# MuZero Token Memory Pass-Large-v1 容量升级执行说明（2026-05-12）

> 目标：不要继续在 `128/8` 或 `128/12` 上叠补丁；为 STS2 Act1→通关训练一次性留足 route/build/history/combat/boss 余量，同时避免旧 replay / checkpoint 静默污染新 schema。

---

## 1. 当前瓶颈与本次升级边界

### 1.1 原瓶颈

旧默认配置接近：

```text
token_d_model               = 128
token_memory_slots          = 8
token_world_bank_top_k      = 3
token_bank_token_slots      = 4
action_embed_dim            = 64
support_size                = 25
MAX_ACTIONS                 = 80
MAX_ROUTE_NODES             = 24
MAX_DECK                    = 40
MAX_RELICS                  = 20
MAX_WORLD_TOKENS            = 412
MAX_CANDIDATE_LOCAL_TOKENS  = 32
```

这套容量对短程 combat tactic 尚可，但对完整 STS2 训练明显偏紧：

- `runtime/build/route/history/enemy/support/powers` 每个 bank 基本只有 1 个 memory slot。
- `token_world_bank_top_k=3` 会在 route/build/reward/shop 决策时丢掉 `support/history/runtime` 等关键上下文。
- deck/relic/route/action/world-token caps 对 full run 后期和复杂选择面存在截断风险。
- `action_embed_dim=64` 对 card target、route target、shop/reward action 语义偏窄。
- `support_size=25` 对长程 value/reward 可能截断偏早。

### 1.2 本次做什么

本次升级做的是 **结构容量与 schema 定版**：

1. 增加 Pass-Large 目标模型容量参数入口。
2. 增加显式 `pass_large_v1` memory slot layout，避免 24 slots 中 17 个被默认塞进 global。
3. 同步扩大 token_v3 observation caps。
4. 将 replay/checkpoint metadata 升级到 pass-large schema，避免旧 replay 静默混入。
5. 补齐 CLI / latent probe / tests / 文档。

### 1.3 本次不做什么

本次不直接解决：

- route heuristic bias 策略质量；默认仍建议 `--route-heuristic-bias 0.0`。
- long-horizon head 直接控制 policy；它仍应先 train-only 或只做 shadow metric，除非 successor action value 已验证。
- combat 行为习惯本身（无压力打防、无伤害意图空过、低 HP 打扣血牌等）；这些要靠 replay 分布、hard guard、diagnostics 和 targeted curriculum 继续治理。

---

## 2. 推荐目标规格：STS2-Pass-Large-v1

训练新 run 时使用：

```yaml
model:
  obs_mode: token_v3
  model_arch: token_memory_v1

  token_d_model: 192
  token_memory_slots: 24
  token_memory_slot_layout: pass_large_v1
  hidden_dim: 4608              # 192 * 24

  token_n_heads: 6              # head_dim = 32
  token_ffn_dim: 768            # 4x d_model

  token_world_layers: 6
  token_local_layers: 2
  token_decoder_layers: 3
  token_candidate_set_layers: 2

  token_world_bank_top_k: 7     # 7 个显式 world bank 全可见
  token_bank_token_slots: 8

  action_embed_dim: 128
  dynamics_res_blocks: 6
  support_size: 31              # 63 bins

  token_internal_planner_blend: 0.7
  token_internal_planner_q_blend: 0.5
  token_internal_planner_objective_q_blend: 0.35
  token_internal_planner_risk_blend: 0.25

  token_dropout: 0.0
  activation_checkpointing: on

observation:
  max_actions: 96
  max_route_nodes: 48
  max_deck: 64
  max_relics: 40
  max_world_tokens: 640
  max_candidate_local_tokens: 40
```

> 注意：`hidden_dim` 不需要单独传；token 模式下由 `token_d_model * token_memory_slots` 自动得到。

---

## 3. 关键改动：显式 memory slot layout

### 3.1 为什么不能只把 slots 改成 24

旧 `legacy` 逻辑是：

```python
if slot_count >= bank_count:
    return list(range(bank_count)) + [GLOBAL] * (slot_count - bank_count)
```

如果仅设置：

```text
token_memory_slots = 24
token_memory_slot_layout = legacy
```

实际 layout 会变成：

| bank | slots |
|---|---:|
| runtime | 1 |
| support | 1 |
| enemy | 1 |
| build | 1 |
| route | 1 |
| powers | 1 |
| history | 1 |
| global | 17 |

这会让 `build/route/history/enemy` 仍然被压成单 slot，不能解决 long-horizon 信息瓶颈。

### 3.2 Pass-Large-v1 layout

新增 `--token-memory-slot-layout pass_large_v1`，24 slots 时固定为：

| bank | slots | 用途 |
|---|---:|---|
| runtime | 4 | 玩家状态、能量、手牌/抽弃消耗循环、当前战斗节奏 |
| enemy | 3 | 敌人 intent、buff、boss 机制、multi-enemy threat |
| build | 4 | deck curve、攻防/过牌/scaling、reward/shop/upgrade 估值 |
| route | 3 | 路径、休息/商店/精英/事件时序、风险/收益 horizon |
| support | 2 | relic/potion 支持图、资源兜底 |
| powers | 2 | power slot、keyword 状态 |
| history | 2 | 最近行为、已选方向、战损/高风险 run 轨迹 |
| global | 4 | 跨 bank 汇总、长期价值、风险/规划融合 |
| **total** | **24** |  |

内部模板顺序为：

```text
runtime, support, enemy, build, route, powers, history, global,
runtime, support, enemy, build, route, powers, history, global,
runtime, build, route, global,
runtime, enemy, build, global
```

### 3.3 必须共享 layout 的模块

已将同一个 `memory_slot_layout` 传入以下 token-memory 模块，避免 representation / dynamics / prediction slot identity 不一致：

- `TokenMemoryEncoder`
- `TokenDynamicsNetwork`
- `TokenPredictionNetwork`
- `TokenTransitionSurfaceHead`
- `TokenFutureWorldBankHead`
- `TokenLatentProjector`
- semantic dynamics / semantic prediction path

---

## 4. Observation caps 定版

本次把 token_v3 关键上限提高到：

| 项目 | 旧值 | 新值 | 原因 |
|---|---:|---:|---|
| `MAX_ACTIONS` | 80 | 96 | reward/shop/route/multi-target surface 留余量 |
| `MAX_ROUTE_NODES` | 24 | 48 | full route/path horizon 不要过早截断 |
| `MAX_DECK` | 40 | 64 | Act2/Act3 构筑和大 deck 更安全 |
| `MAX_RELICS` | 20 | 40 | 通关后期 relic 截断风险降低 |
| `MAX_WORLD_TOKENS` | 412 | 640 | route/build/history/world token 合并后留余量 |
| `MAX_CANDIDATE_LOCAL_TOKENS` | 32 | 40 | 复杂候选 action local context 留余量 |

保持不变：

```text
MAX_HAND    = 12
MAX_ENEMIES = 5
MAX_POTIONS = 5
```

新的 observation API version：

```text
attention_obs_v5_pass_large
```

---

## 5. Schema / replay / checkpoint 兼容策略

### 5.1 为什么必须 bump schema

本次同时改变：

- obs shape caps；
- `hidden_dim`；
- memory slot identity/layout；
- action embedding dim；
- scalar support bins；
- token future bank capacity。

因此旧 replay / optimizer state / old checkpoint 不应静默混入新 run。否则会出现隐蔽的：

- action index alignment 污染；
- route_summary/action/logit/Q index 错位；
- value target support 不一致；
- replay obs shape 不一致；
- slot identity 语义漂移。

### 5.2 当前策略

已写入 checkpoint/replay metadata：

```text
format = muzero-v2
obs_schema_version = token_v3_attention_obs_v5_pass_large
replay_schema_version = muzero_replay_v2_token_v3_pass_large_caps
checkpoint_compatibility_version = 2
observation_shape_caps = {...}
model_schema_version = obs/model/layout/slots/d_model/action/support 的组合字符串
```

加载行为：

- network：仍允许 shape-compatible partial warm-start，但 Pass-Large 目标训练不推荐依赖旧权重。
- optimizer：只有 network exact resume 才加载；partial warm-start 会跳过 optimizer。
- replay buffer：必须 metadata 和 replay_buffer.pkl schema/caps 全部匹配才加载，否则跳过。

### 5.3 推荐训练原则

如果目标是干净评估 Pass-Large：

- 不要加载旧 replay buffer。
- 不要加载旧 optimizer。
- 最稳是完全从零开始新 run。
- 如果必须 warm-start，只允许作为短期对照实验，并且记录为 `partial network warm-start / empty replay / fresh optimizer`。

---

## 6. 大训练前必须过的验收项

### 6.1 代码级测试

在 WSL 中运行：

```bash
cd /mnt/e/game/project/sts2_mcp/packages/rl-agent
./.venv-wsl-rocm/bin/python -m pytest tests/test_token_memory_slot_layout.py -q
./.venv-wsl-rocm/bin/python -m pytest tests/test_observation_v3.py -q
```

最低要求：

- `pass_large_v1` 24-slot count 精确匹配 4/3/4/3/2/2/2/4。
- MuZeroNetwork token modules 全部持有同一 layout。
- token 模式 `hidden_dim == token_d_model * token_memory_slots`。
- `MAX_CANDIDATE_LOCAL_TOKENS == 40`。
- `OBSERVATION_API_VERSION == attention_obs_v5_pass_large`。

### 6.2 smoke 检查

```bash
cd /mnt/e/game/project/sts2_mcp/packages/rl-agent

./.venv-wsl-rocm/bin/python - <<'PY'
from collections import Counter
from muzero.sts2_env.token_memory import build_memory_slot_bank_ids, MEMORY_BANK_NAMES
ids = build_memory_slot_bank_ids(24, "pass_large_v1")
print(ids)
print(Counter(MEMORY_BANK_NAMES[i] for i in ids))
PY

./.venv-wsl-rocm/bin/python -m muzero.train --help | grep -E "token-memory-slot-layout|action-embed-dim|support-size|dynamics-res-blocks|action-rollout-buckets"
```

可选：实例化 Pass-Large 模型但不训练：

```bash
./.venv-wsl-rocm/bin/python - <<'PY'
from muzero.sts2_env.muzero_model import MuZeroNetwork
m = MuZeroNetwork(
    obs_mode="token_v3",
    model_arch="token_memory_v1",
    token_d_model=192,
    token_memory_slots=24,
    token_memory_slot_layout="pass_large_v1",
    token_n_heads=6,
    token_ffn_dim=768,
    token_world_layers=6,
    token_local_layers=2,
    token_decoder_layers=3,
    token_candidate_set_layers=2,
    token_world_bank_top_k=7,
    token_bank_token_slots=8,
    action_embed_dim=128,
    dynamics_res_blocks=6,
    support_size=31,
    activation_checkpointing=True,
)
print({
    "hidden_dim": m.hidden_dim,
    "action_embed_dim": m.action_embed_dim,
    "support_size": m.support_size,
    "slot_layout": m.constructor_spec()["token_memory_slot_layout"],
})
PY
```

期望：

```text
hidden_dim = 4608
action_embed_dim = 128
support_size = 31
slot_layout = pass_large_v1
```

---

## 7. 推荐新训练命令模板

> 不要在已有旧 run 目录上覆盖。新建 run/log/checkpoint 目录，旧 buffer/optimizer 不要加载。

```bash
cd /mnt/e/game/project/sts2_mcp/packages/rl-agent

./.venv-wsl-rocm/bin/python -m muzero.train \
  --obs-mode token_v3 \
  --model-arch token_memory_v1 \
  --token-d-model 192 \
  --token-memory-slots 24 \
  --token-memory-slot-layout pass_large_v1 \
  --token-n-heads 6 \
  --token-ffn-dim 768 \
  --token-world-layers 6 \
  --token-local-layers 2 \
  --token-decoder-layers 3 \
  --token-candidate-set-layers 2 \
  --token-world-bank-top-k 7 \
  --token-bank-token-slots 8 \
  --action-embed-dim 128 \
  --dynamics-res-blocks 6 \
  --support-size 31 \
  --activation-checkpointing on \
  --token-dropout 0.0 \
  --action-rollout-buckets 8,16,32,64,96 \
  --route-heuristic-bias 0.0 \
  --mixed-precision auto \
  --batch-size 8 \
  --unroll-steps 5 \
  --n-envs 2
```

说明：

- `--batch-size 8` 是保守起步值；显存稳定后再上调。
- 当前代码未看到正式 gradient accumulation CLI；不要在文档里假设可用。
- full-run 不传 `--seed-pool` / `--seed-pool-file`，保持随机种子，样本更丰富。
- route heuristic bias 先关闭，避免此前 `bias=0.05` 导致路径选择退化的问题重复出现。

---

## 8. 训练分布建议

自然 full-run replay 会严重偏向 normal combat 和 early death：

- Act1 大致包含：2–3 场 weak、5–8 场 normal、0–2 场 elite、0–1 场 boss、若干 event/shop/rest/reward。
- 全流程到 Act3 的 boss 固定只有 3 场，但 normal/奖励/事件样本远多于 boss/elite。
- 如果仅靠自然 full-run，模型会长期缺 boss/elite 和高质量 route/build 样本。

建议 replay / curriculum 长期目标：

| replay 来源 | 目标占比 | 作用 |
|---|---:|---|
| full-run / act progression | 40% | 学真实 route/build/resource/hp 分布 |
| combat tactical execution | 25% | 纠正攻防、空过、无压力打防、战损等基础行为 |
| boss + elite targeted | 20% | 补 boss/elite 稀缺样本与特定机制 |
| route + reward + shop + upgrade + build decisions | 15% | 补长程 credit assignment 与构筑估值 |

combat sandbox 可以继续使用：

```text
curated_combat_ironclad_mixed_provenance
subset = bootstrap_human_plus_local_all_minus_combat_reset_failures
```

但要记住：sandbox 注入的是战斗前状态（遗物、卡组、药水、升级/附魔等），full-run 里这些资源仍要模型自己学会通过 route/reward/shop/rest/smith 获得。

---

## 9. 必须持续监控的指标

### 9.1 Observation truncation

目标全部 `< 0.5%`，最好 `< 0.1%`：

```text
obs/world_token_truncation_rate
obs/deck_truncation_rate
obs/relic_truncation_rate
obs/action_truncation_rate
obs/route_node_truncation_rate
obs/candidate_local_truncation_rate
```

若这些升高，说明模型容量再大也在吃缺失信息。

### 9.2 Combat behavior

重点看行为质量，不只看 episode reward：

```text
weak/normal/elite/boss win_rate
normal_hp_loss_mean
avoidable_damage_rate
no_pressure_pure_block_selected_rate
pure_block_waste_selected_rate
meaningful_damage_endturn_bad_rate
wasteful_end_turn_selected_rate
hp_cost_self_lethal_selected_rate
hp_cost_low_margin_selected_rate
x_cost_zero_bad_p0
potion_low_urgency_selected_rate
```

当前历史问题：

- 敌方非攻击意图时仍打无效防御。
- 无伤害压力时空过 / 结束回合，导致少打伤害、拖长战斗、战损增加。
- 低 HP 边际打扣血牌偶发自杀或进入高风险。
- potion 使用过早或时机错误。

### 9.3 Route/build alignment

```text
top1_selected_rate_multi
best_minus_selected_mean_multi
forced_elite_selected_rate
unsafe_elite_selected_rate
low_hp_elite_selected_rate
act1_boss_seen_rate
act1_pass_rate
reward_pick_quality/top1_selected_rate_multi
build_gap_risk_corr
```

如果 `best_minus_selected_mean_multi` 持续 > 2 且 `top1_selected_rate_multi` 低迷，说明 route/build scoring 和 action alignment 仍有问题，不能用更大模型硬赌。

### 9.4 Long-horizon / future-world

```text
loss/future_world_aux
loss/future_bank_state
loss/future_bank_delta
loss/future_bank_token_presence
loss/jepa_next_hidden
loss/surprise
loss/latent_gaussian_reg
long_horizon/target_valid_rate
long_horizon/mae
long_horizon/corr_pred_target
long_horizon/reach_floor_10_auc
long_horizon/reach_floor_15_auc
```

若 future-world 又出现 100x/1000x spike，需要继续 dump 触发 batch；不要让旧 optimizer momentum 污染新 run。

---

## 10. 通关最低阶段性目标

### Combat

| 指标 | 初始目标 | 后续目标 |
|---|---:|---:|
| weak/normal combat win rate | >= 90% | >= 95% |
| elite combat win rate | >= 55% | >= 65% |
| boss win rate | >= 45% | >= 55% |
| average hp loss normal | 持续下降 | 稳定低位 |
| avoidable damage rate | 持续下降 | 接近 0 |

### Route / Act1

| 指标 | 初始目标 | 后续目标 |
|---|---:|---:|
| act1_boss_seen_rate | >= 60% | >= 80% |
| act1_pass_rate | > 0 | 20–30%+ |
| forced_elite_bad_pick_rate | < 10% | < 5% |
| rest_before_elite_detection | > 80% | > 90% |

### Build / reward

| 指标 | 初始目标 | 后续目标 |
|---|---:|---:|
| reward_pick_quality top1_selected_rate_multi | >= 45% | >= 60% |
| best_minus_selected_mean_multi | 下降 | 稳定 <= 2 |
| deck/relic/action/route truncation | < 0.5% | < 0.1% |

---

## 11. 风险与回滚

### 11.1 显存/吞吐风险

Pass-Large 约为原 23M baseline 的数倍，吞吐一定下降。优先措施：

1. `--activation-checkpointing on`
2. `--mixed-precision auto`
3. 降低 `--batch-size`
4. 先 `--unroll-steps 5`，稳定后再考虑 7
5. 不要同时提高 combat rollout depth/beam 到过高值

### 11.2 数据需求风险

模型变大后更依赖 replay 质量。不能只跑自然 full-run 等它自己碰 boss：

- full-run 负责真实分布；
- combat sandbox 负责战斗基础和 boss/elite 稀缺样本；
- route/reward/shop/upgrade targeted 负责长程 credit。

### 11.3 策略污染风险

- route heuristic bias 默认 0；只做 shadow metric 和小样本 A/B。
- long-horizon head 不直接加到 root prior；除非做 successor-aware per-action value 并验证相关性。
- 旧 replay/optimizer 不混入新 schema。

---

## 12. 当前代码落点

主要文件：

```text
packages/rl-agent/muzero/sts2_env/token_memory.py
packages/rl-agent/muzero/sts2_env/muzero_model.py
packages/rl-agent/muzero/sts2_env/muzero_buffer.py
packages/rl-agent/muzero/training/cli_args.py
packages/rl-agent/muzero/training/cli_main.py
packages/rl-agent/muzero/training/checkpointing.py
packages/rl-agent/muzero/eval_latent_probes.py
packages/rl-agent/sts2_env/observation_common.py
packages/rl-agent/sts2_env/observation_v3.py
packages/rl-agent/tests/test_token_memory_slot_layout.py
packages/rl-agent/tests/test_observation_v3.py
```

文档：

```text
docs/muzero-token-memory-pass-large-v1-20260512.md
```

---

## 13. 执行结论

我建议把 `STS2-Pass-Large-v1` 作为下一次从零训练的正式容量基线：

```text
d_model = 192
memory_slots = 24
slot_layout = pass_large_v1
hidden_dim = 4608
world_bank_top_k = 7
bank_token_slots = 8
action_embed_dim = 128
support_size = 31
dynamics_res_blocks = 6
observation caps = pass-large caps
```

但判断它是否能通过 Act1，不应只看 reward，而要同时看：

1. normal 战斗战损是否下降；
2. 无压力打防 / 空过是否被压下去；
3. boss/elite targeted 胜率是否回升；
4. route/build alignment 是否改善；
5. truncation 是否接近 0；
6. future-world loss 是否无 spike。

如果这些指标没有改善，下一步不应继续加容量，而应回到 replay 分布、action alignment、route/build target、combat quality diagnostics 上修。
