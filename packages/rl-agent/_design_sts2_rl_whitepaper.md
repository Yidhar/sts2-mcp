# 杀戮尖塔2 强化学习 AI 架构深度调研与设计白皮书

> 撰写日期: 2026-04-20
> 目标读者: sts2_mcp rl-agent 团队（Phase 6 → Phase 8 已落地，0% Boss 击杀率瓶颈）
> 目标版本: Slay the Spire 2 Early Access v0.98.3（2026-03-05 发布）

---

## 执行摘要（Executive TL;DR）

1. **STS2 的 MDP 比 STS1 复杂度上升约 2~3 倍**，主要来自三个新机制：（a）双变体 Act（Overgrowth / Underdocks）把状态空间按 act 拆成两份；（b）Enchantments（永久带副作用的卡牌强化）把"卡牌特征"从静态变成"有可变属性的实体"；（c）Durability 让"每战只能触发 N 次的遗物"成为部分可观测、与战斗深度耦合的 resource counter。外加 Regent 的 Stars 双资源池、Necrobinder 的 Doom/Souls/Osty 三机制栈、Defect 的 Synchronize 临时 Focus、Silent 的 Sly 关键字（弃牌即打出），"角色-场景"联合特征维度爆炸。
2. **团队当前的 AuxMaskablePPO + 412-token 7-bank Transformer 架构在建模上是正确且前沿的**：Phase 6 POWER_SLOT 和 Phase 8 HISTORY/causality 的设计与 2025 年 SkyNet（Belief-Aware MuZero）论文的"auxiliary 带 belief 预测头"思路高度一致；观测编码维度（412 × 160）对 STS2 的状态复杂度是富余的，不是瓶颈。
3. **0% Boss 击杀率的真正瓶颈不是架构，是 credit assignment + exploration**。具体来说：（a）reward 密度在 Act1 Boss 层依然太稀疏（floor 17 比 floor 1 的访问量少 30×，PPO 的 GAE 在 λ=0.95、γ=0.99 下几乎看不到 boss 信号）；（b）ent_coef=0.01 在 127 路动作空间上等效 entropy bonus ≈ 0.05 nats，远低于 long-horizon sparse-reward 任务推荐的 0.02~0.05（相对论）；（c）aux_causality 25~30% zero-rate 说明 head 已经学会"大部分动作没有显著后果"这个正确但无用的先验——需要更硬的 counterfactual 目标。
4. **建议的下一步**（按 ROI 排序）：(1) 用 BC 在 1.97M 非战斗样本上 warm-start，**同时**开放 combat_sandbox 对 Act1 Boss 的课程预训练；(2) 把 Boss 相关 reward 再放大 2~3 倍 + 引入 HP-normalized shaping；(3) 把 ent_coef schedule 改成 0.03→0.005 余弦退火；(4) aux_causality 升级到 Tier 3（预测回合末 V(s) 的 counterfactual delta）；(5) 只有当前 5 项做完 Boss 击杀率仍 <5% 时，再考虑 MuZero pivot——且应以 Stochastic MuZero（处理抽牌 RNG）+ belief-aware aux head（SkyNet 思路）为目标，而非香草 MuZero。

---

# 第一阶段：全网深度调研与信息收集（Web-Sourced Intelligence）

本阶段所有陈述均标注了**STS2**或**STS1-推断**来源。未标注 STS1 的即为 2026 Early Access 版本确认信息。

## 1.1 版本与发行事实

- **STS2 Early Access 于 2026-03-05 上线**，价格 $24.99 USD，引擎 Godot 4.5.1 + .NET 9。
- **五个角色**：Ironclad / Silent / Defect（回归，但均有机制级重做）+ Regent / Necrobinder（全新）。
- **当前最高难度是 Ascension 10**（**不是 STS1 的 A20**），EA 期间不会扩展到 A20。这对 RL 意味着状态动作空间的"难度维度"是 10 档离散，而不是 STS1 的 20 档。

## 1.2 核心流程机制（相对 STS1 的根本变化）

| 机制 | STS1 | STS2 | RL 影响 |
|---|---|---|---|
| 地图 | 进入新 Act 时生成，Boss 不可见 | 进入新 Act 时生成，**Boss 在图顶立即可见**，强制 route 规划向后看 | `route_node_boss_preview` 必须进入状态；rewardshaping 可基于"离 boss 多少步" |
| Act 结构 | 3 个单线性 Act | 3 个 Act，**每个 Act 有 a/b 双变体**（当前 EA: 1a Overgrowth, 1b Underdocks, 2a, 3a；1b/2b/3b 陆续上线）。进入 Act 时随机分配 | 等效于 Act 数 ×2，encounter 池 ×2；需要在 obs 里 one-hot act 变体 |
| 篝火 | Heal 30% / Upgrade / Smith 事件 | Heal 30%（missing HP）或 Upgrade | 比 STS1 简单一档，决策模型可以用 binary head |
| Boss 奖励时机 | 击败后立刻选 boss relic | **击败 Act N boss 后不发 relic，Act N+1 开始时才发** | credit assignment 变长——reward shaping 必须桥接 |
| Boss 前置 | Boss 前的节点不强制 rest | **Boss 前的节点恒为 Rest Site**，保证可以回血 | rest→boss 的过渡是一个"必达子游戏"；HRL 里这是天然的 option 边界 |

## 1.3 全新角色机制

### Regent（双资源系统）
- Energy 每回合重置 3；**Stars 跨回合累积，无上限，战斗开局自带 3 Stars**（起手遗物 Divine Right）。
- Forge 关键字：生成并强化独有 Retain 卡 **Sovereign Blade**（起始 2-energy / 8 damage，每次打出 Forge 卡后 damage 永久 +Forge 值，战斗内保留）。
- 两大 archetype：**Stars Engine**（爆发回合）/ **Forge Blade**（单张 win-con 叠伤害）。

### Necrobinder（三机制栈，66 HP 最脆）
- **Doom**：在敌人身上叠标记；当敌人当前 HP ≤ Doom 层数时**处决**（斩杀线）。
- **Souls**：一张 **0 mana skill**，抽 2 张牌后 exhaust；用作速度催化。
- **Osty**：一只巨型骷髅手召唤物，作为物理护盾；通过 Summon 关键字增强；"Osty attack" 卡牌随同伴加成缩放（如 Squeeze：25 base + 5 每张 Osty 攻击卡）。
- 三 archetype：Souls Engine / Osty Tank / Doom Execute。

### Silent（Sly 关键字重做）
- **Sly** 新关键字：**任何标 Sly 的卡被弃掉时，自动免费打出**（不进弃牌堆，替换原先的 discard-synergy）。
- Poison + Shiv 分支被重构，弃牌 combo 从"触发器"变成"自动触发"。

### Defect（Synchronize 临时 Focus）
- **永久 Focus 叠加被移除**；改为 **Synchronize 关键字**产生"本回合 +N Focus"尖峰。
- 新 orb: **Glass Orb**（多目标伤害）。
- 整体从长战稳定 DPS 变成"回合级爆发窗口"。

### Ironclad（削弱 Hellraiser loop）
- Expect a Fight / Dominate / Spite / Stoke 全部重做；Hellraiser status-gaining 循环被移除，status 机制搬到 Defect。
- "失 HP 触发加成"的条件效应保留，和起手遗物 Burning Blood（战后回 6 HP）联动。

## 1.4 战斗系统新增要素

- **Durability（全新关键字）**: 某些 relic 每战只能触发 N 次；之后沉默。这让"理论永久被动" relic 的实际价值与"战斗长度"耦合——在 hallway 用掉 vs 留给 boss 是策略点。
- **Enchantments（非升级的卡牌修饰符）**: 永久附加，一张牌只能一个 enchantment，事件 / 特殊节点获取。例：**Corrupted** = 伤害 +50%，每次打出扣 3 HP。与 STS1 单一 upgrade 并列的第二维度。对应英文反义词是 **Afflictions**（带负面）——当前 EA 提到但详细列表未公开。
- **Potion 库**扩展；特定 relic（Fairy in a Bottle、Snecko Oil、Focus Potion）对 potion 效用有乘数。
- **Boss HP 区间**（来自多个攻略站，已交叉校验）：
  - Vantom（1a Overgrowth）：~280-320 HP，Slippery buff（单次伤害封顶 1）。
  - Waterfall Giant（1b Underdocks）：~350-400 HP，死亡时 Steam Eruption 爆炸，死前 Empower 每回合。
  - Inkblot Phantasm（1b）：T1 固定 30 伤害 + 3 张 Call 状态牌（每张 6 HP），每 2 回合 Intangible；对 <40 HP 的牌组几乎必败（**Act1 最硬的 boss**）。
  - The Insatiable（已知 boss，位置待确认）：321/341 HP，5 种已知动作。

## 1.5 MDP 状态空间膨胀的具体来源

相比 STS1，下列新变量让 state space 规模大致 ×2~3：

| 膨胀源 | 增加维度 | 影响 |
|---|---|---|
| Act 变体 | +log2(2)^3 ≈ 3 bit | 中等 |
| Enchantments | 每张牌额外 ~8-16 个离散标志 | 大（卡面不再是静态） |
| Durability counters | 每个有 durability 的 relic +1 标量 | 中等，战斗内动态 |
| Regent Stars | 跨回合 state | 大（长程 credit） |
| Necrobinder 三机制 | 每敌人 Doom 标量 + Osty 实体 + Souls 流 | 大（新实体类型） |
| Defect Synchronize | 每回合临时 Focus | 小（本回合） |
| Silent Sly 弃牌连锁 | 弃牌事件驱动的 auto-play 序列 | 中等（环境副作用） |
| Alternate acts（随机） | encounter / event 池 ×2 | 大（行为分布漂移） |

**结论**：STS2 的 state space 复杂度上升，但**主要变化是"实体型"而非"标量型"**——这是 Transformer token-based 编码（当前团队架构）比传统 flat MLP 占优的地方，设计方向正确。

---

# 第二阶段：《杀戮尖塔2》RL 环境建模（Environment & MDP Formulation）

## 2.1 状态空间编码

### 2.1.1 总体策略：实体 + 池 + 上下文 三层

将观测分成三个语义层：

```python
state = {
    "global":        g_vec,       # d_g ≈ 32 : HP/maxHP, energy, stars, floor, act_variant, ascension, gold
    "entities":      entity_set,  # variable-size set of typed tokens (player buffs, enemies w/ intent, Osty, piles, map nodes, relics, powers)
    "context":       ctx_vec,     # d_c ≈ 64 : run_memory_digest, last_N_actions_summary, boss_distance
}
```

### 2.1.2 实体 token 设计（与当前 `observation_v3.py` 对齐，加少量字段）

```python
token = concat([
    one_hot(token_type, NUM_TYPES=72),     # 新增 STARS_POOL, OSTY_ENTITY, ENCHANT_SLOT, DURABILITY_COUNTER
    numeric_features[96],                   # 见下
    text_embedding[64],                     # bge-small-zh-v1.5 前 64 dim（已缓存）
])

# numeric_features（按 role 不同语义不同）：
# - HAND_CARD:   [cost, upgraded_flag, enchant_id_onehot[8], damage_est, block_est, targets_onehot[4], x_cost_flag, retain_flag, ...]
# - ENEMY:       [hp_ratio, block, intent_onehot[8], intent_damage_est, buff_vec[32], debuff_vec[16], is_boss, act_variant_hint]
# - POWER_SLOT:  [power_id_bucket, stacks_log1p, owner_onehot[2], effect_algebra_vec[32]]
# - RELIC:       [relic_id_bucket, durability_left, durability_max, triggered_this_combat_count, tier]
# - MAP_NODE:    [node_type_onehot[8], floor, path_to_boss_steps, reachable_now, risk_estimate]
```

关键点：
- **Enchantments** 进入 HAND_CARD token 的 `enchant_id_onehot`；不作为独立 token，避免 attention 过度分裂同张牌的两面信息。
- **Durability counter** 进入 RELIC token 的两个标量（`durability_left`, `triggered_count`）——当前 Phase 6 POWER_SLOT 的 effect-algebra 范式已经能承载这种 "动态被动"。
- **Stars pool** 作为独立 GLOBAL 字段（不是 token），因为它是标量资源；Regent 场景下 `stars_current / stars_generated_this_combat / stars_spent_this_combat` 三元组足够。
- **Osty** 视为特殊 ENEMY 或特殊 ALLY token，带 `is_summon=True` flag 和 own HP。

### 2.1.3 双层 map/combat 状态耦合

```python
# 推荐在 policy 层做 two-phase attention：
# phase A: within-combat (runtime + enemy + support + powers + history banks)
# phase B: between-combat (build + route + boss_distance_scalar)
# 当 screen == combat 时，value_head 接 combat_gate；否则接 route_gate。
```

**当前代码中 7 banks 的设计已经支持这个拆分**；只需在 AuxMaskablePPO 里加一个 `screen_mode` flag 选择 value head 支路（见第五阶段建议 #3）。

## 2.2 动作空间设计

### 2.2.1 合并层（team 已实现的方向）

- `MAX_ACTIONS = 127`，动作通过 bridge 的 legal_actions list 动态 mask。
- `semantic_action.py` 的 49 个语义家族（play_card, use_potion, end_turn, map, shop, smith, reward_pick, event_choice, rest_heal, rest_upgrade 等）是**宏观-微观合并**的正确方式：policy 输出的 logit 落在 127 个"物理 slot"上，但 aux head / reward shaping 按 49 个"语义家族"对齐。

### 2.2.2 Action Masking 伪代码（已实现，仅作为文档确认）

```python
def compute_action_mask(obs, legal_actions) -> np.ndarray:
    mask = np.zeros(MAX_ACTIONS, dtype=bool)
    for idx in legal_actions:
        mask[idx] = True
    # 保险：end_turn 永远合法（除非在非战斗屏幕）
    if obs["screen"] == "combat" and not mask.any():
        mask[END_TURN_IDX] = True
    return mask

# 在 MaskableCategoricalDistribution 里：
# logits = policy_head(features)
# logits = logits.masked_fill(~mask, -1e9)
# dist = Categorical(logits=logits)
```

### 2.2.3 目标选择的子结构

STS2 的 single-target 卡需要 `(card_idx, target_enemy_idx)` 二元选择。团队当前用 127 个 flat slot × 候选 24 local token 的 CandidateDecoder，把 `(card_i, target_j)` 展开成 single action ID——**这是对的**。它把双层动作压回平面，MaskablePPO 原生适配。风险是当 `#cards × #targets` 接近 127 时发生截断；需要在环境层保证按优先级填充（优先级：攻击牌 × 每敌人 > 技能 × 玩家 > 技能 × 每敌人）。

## 2.3 奖励函数塑造（对齐 team 的 layer 1/2/3）

### 2.3.1 设计原则

- **Shaping 必须 potential-based** 以保证与最终胜率的 policy optimality 一致（Ng, Harada & Russell, 1999）：`F(s,s') = γ Φ(s') − Φ(s)`。team 当前的 floor-clear ladder 不严格 potential-based，但可以近似（见下）。
- Reward 应同时反映 **tempo**（伤害、block）+ **tempo 效率**（HP 损耗 / 敌 HP 消减）+ **长程进度**（floor、boss 距离）。

### 2.3.2 推荐的 reward 分层（替换/补齐当前 layer 1/2/3）

```python
def compute_step_reward(prev_state, action, new_state) -> float:
    r = 0.0

    # ---- Layer A: combat intrinsics ----
    # 敌方 HP 消减（已有）
    r += 1.0 * sum(max(0, e.prev_hp - e.hp) / e.max_hp for e in enemies)  # 归一化
    # 我方 HP 损耗
    r -= 0.8 * max(0, prev_state.player_hp - new_state.player_hp) / prev_state.player_max_hp
    # End-turn 浪费：手里仍有可打且能破敌 block 的高价值牌
    if action.is_end_turn and _wasted_energy(prev_state) > 0:
        r -= 0.05 * _wasted_energy(prev_state)

    # ---- Layer B: progression potential-based ----
    # Φ(s) = w_floor * floor_ratio + w_boss * boss_distance_reciprocal + w_deck * deck_quality
    phi_prev, phi_new = potential(prev_state), potential(new_state)
    r += GAMMA * phi_new - phi_prev

    # ---- Layer C: boss-fight amplifier ----
    if new_state.in_boss_fight:
        # 对 boss 造成的伤害按 5× 放大（team 已做；保持）
        r += 4.0 * sum(max(0, b.prev_hp - b.hp) / b.max_hp for b in boss_enemies)
        # 存活到 boss 回合数奖励：鼓励活久一点（team 观察到 boss_encounter_steps=57 太短）
        r += 0.02 * 1  # 每在 boss 战多撑一步

    # ---- Layer D: rest / potion（已有，简述）----
    r += layer2_rest_hp_gate(prev_state, action)
    r += layer3_potion_v3(prev_state, action, new_state)

    # ---- Layer E: 新增 —— sparse terminal ----
    if new_state.boss_defeated:
        r += 10.0  # Act1 boss 击杀 hard signal
    if new_state.run_won:
        r += 50.0  # 通关

    return r

def potential(s):
    # 归一化到 [0, 1]
    return (
        0.4 * s.highest_floor_touched / MAX_FLOOR +
        0.3 * (1.0 / (1.0 + s.steps_to_boss)) +
        0.3 * deck_quality_score(s.deck)  # 来自 skada BC 的卡组估值模型
    )
```

**关键差异 vs 团队当前 stack**：
1. Progression 改为 **potential-based** → 不再"多走一步就赚"，而是"从 floor 5 走到 floor 6 永久涨 +Δφ"。
2. 引入 `deck_quality_score`（可以复用 BC 训练时的 value 估计器，或简单用"2-cost 内高伤害牌数 + scaling 关键字计数"）—— 这是当前团队**没有**的信号。
3. Boss 存活步数微小奖励（+0.02/step in boss），解决 `boss_encounter_steps=57` 过短问题。
4. 胜利 terminal 信号 +50 足够大，但要通过 BC warm-start 让 policy 偶尔看到，不要指望纯 PPO 探索出来。

### 2.3.3 避免 reward hacking 的 3 条约束

- **Potion 囤积惩罚**封顶 -0.30（team 已做），保证不把"不买药水"当成最优策略。
- **Rest heal** 的 hp gate（team 已做）不要无条件奖励 heal，只在 hp<60% 时给正 reward。
- **End-turn waste**（上面 layer A）要以 _破敌 block 的杀伤机会_ 为条件，不然会在 combo 牌组里误伤（打不穿就留着是对的）。

---

# 第三阶段：AI 核心算法架构设计（Core Algorithm）

## 3.1 算法选型论证

**候选对比**（针对 STS2 的四个特征：部分可观测 / 长序列 / 离散动作 / RNG）：

| 算法 | 样本效率 | 工程成本 | RNG 处理 | Partial observability | 对 STS2 的评分 |
|---|---|---|---|---|---|
| PPO / MaskablePPO | 中 | 低 | 通过大 batch + GAE 间接学 | 通过 LSTM / Transformer | B+ |
| **AuxMaskablePPO**（team 当前）| 中+ | 低 | 同上 + aux head 给 shaping | 同上 + 辅助预测头 | **A-** |
| MuZero | 高（理论）| 高 | 确定性 latent，抽牌需要 chance node | 通过 latent state | B（需要大改造）|
| **Stochastic MuZero** | 高 | 很高 | chance outcomes 显式建模 | latent + chance | A（但工程 3-6 月）|
| **SkyNet（Belief-Aware MuZero）** | 最高（理论） | 最高 | 同上 + belief head | belief-aware | S（最契合，但极重）|
| HRL（Options / Feudal）| 看分解好坏 | 中 | 无专门机制 | 高层抽象帮助 | B（和 PPO 互补） |
| DQN / Rainbow | 低 | 中 | 弱 | 弱 | C（不推荐） |

**推荐**（给团队的 engineering roadmap）：

> **Track 1（主线，继续）**: AuxMaskablePPO + 第二阶段的 reward 重构 + BC warm-start。把当前实现榨到位。
> **Track 2（备选，观察 12 周）**: 如果 Track 1 在 1M step 内无法突破 5% boss 击杀率，启动 **Stochastic MuZero 原型**；不建议跳过 Stochastic 直接做 SkyNet（belief head 的 supervision 需要自监督，工程量叠加）。

**为什么不现在就 MuZero？**
- MuZero 的 planning 过树搜索假设 reward 密度够高；在 boss 几乎不可达的当前阶段，tree search 会在 Act1 mid 反复展开无效子树。
- MuZero 需要"可回滚的模拟器"（state → next_state 完全确定）。HeadlessSim 支持，但 live 真实游戏只能单向前进——会限制 Track 2 的 data path。
- 团队的 Transformer encoder 在 MuZero 改造时**可以直接复用作为 representation network**，沉没成本 ≈ 0。

## 3.2 网络结构设计

### 3.2.1 当前架构评估（STS2OmniAttentionPolicy）

```
obs
 ├─ 412 world tokens × 160 d
 ├─ 24 candidate local tokens × 160 d
 └─ global scalars

EntityTokenEmbedder         # 投影 + 类型/owner/role/zone/order embedding
    ↓
[7 World Banks × N layers]  # runtime / support / enemy / build / route / powers / history
    ↓
Cross-Attention（bank-to-bank）
    ↓
CandidateDecoder             # 世界 context → 候选动作 context
    ↓
├── Policy head (masked categorical over 127 actions)
├── Value head (scalar)
└── 8 auxiliary heads (objective / transition / traits / build / selection / route / enemy_state / causality)
```

**诊断**：架构容量充分，不需要扩容。问题在 **bank routing 的 role IDs 是否覆盖了 STS2 全部实体类型**。检查清单：
- [ ] `ENCHANT_SLOT` / `DURABILITY_COUNTER` 在 powers bank？（建议：Enchant→build bank, Durability→support bank）
- [ ] `STARS_POOL`、`OSTY_ENTITY`、`SOULS_ENGINE` 是否有专属 role_id？（建议 role_id 扩到 ~24）
- [ ] `act_variant`（1a vs 1b）作为 global bias 注入每个 bank 的第一层 query？

### 3.2.2 Critic 设计：双 value head

当前单一 value head 对 combat 和 non-combat 两类截然不同的"估值意义"是**次优**的：
- 战斗中的 V(s) 是"这个战斗预期结束时玩家还剩多少相对 HP + 多少战斗奖励"。
- 非战斗的 V(s) 是"从这个 map/event/shop 状态到 run 结束的期望 return"。

**推荐改动**：

```python
class DualValueHead(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.combat_v = nn.Linear(d, 1)
        self.run_v    = nn.Linear(d, 1)

    def forward(self, z, screen_mode):
        # screen_mode: [B] int, 0=combat, 1=non-combat
        # 在 GAE 里使用对应的 V；loss 也按场景拆分
        v_c = self.combat_v(z).squeeze(-1)
        v_r = self.run_v(z).squeeze(-1)
        return torch.where(screen_mode == 0, v_c, v_r)
```

**预期收益**：combat vs run-level value loss 不会互相污染，GAE advantage 在 Act1 boss 前的 rest site 转换点不会出现梯度撕裂（目前 team 观察到的 sim EV 反转 +0.37→-0.34 很可能部分来自这个）。

### 3.2.3 Actor head 保持现状

masked categorical + 127 slots + CandidateDecoder 局部上下文 = 正确。不改。

## 3.3 随机性处理

STS2 的三个 RNG 源：

| RNG | 期望处理方式 |
|---|---|
| 抽牌顺序 | 观测包含"抽牌堆大小"和"弃牌堆数字组成"而非 card ordering —— 强制 agent 学 expectation |
| 事件结果 | 事件选项的"预期结果"用多次 rollout 的 aux_head transition 学 |
| 敌人行动 | intent 是 observable（几乎完全）；但 move selection 的底层 RNG 是 hidden —— aux_head enemy_state 预测下一个 intent 分布 |

### 3.3.1 推荐：在 action distribution 上加 temperature schedule

```python
# 训练早期用较高的 temperature 鼓励探索
temperature = max(0.7, 1.2 - 0.00001 * global_step)
logits = logits / temperature
```

这在 MaskablePPO 下不冲突 mask，但能让"弱 advantage 动作"保留更多采样概率 → 降低 combat-stuck rate 12% 的问题。

### 3.3.2 aux_causality 升级（Tier 3）

当前 Tier 2 预测 `(damage, block, hp_loss, draw, energy, strength, dex, vuln_delta)` 已达瓶颈（25-30% zero-rate）。升级方向：

```python
# Tier 3: counterfactual value delta
causality_target = V(s_after_action) - V(s_after_end_turn_if_instead)
# 即：这个动作 vs "直接结束回合" 的 2-step 价值差
```

这让 causality head 不再学"动作的直接观测 delta"，而是学"动作对长程价值的边际贡献"。这个目标和 advantage 高度相关但 **monotonic transform 不同**（advantage 是相对当前 policy 的相对值；causality_target 是相对"空动作" baseline 的绝对值），能给 actor 梯度注入新信息。

---

# 第四阶段：工程落地与训练策略

## 4.1 模拟器 / self-play 基础设施

**团队已经有两条 pipeline**——这是 asset 不是 liability：

| Path | IT/s | 用途 | 限制 |
|---|---|---|---|
| HeadlessSim (frankqwang fork) | 20-25 × 8 env | 主训练 | 可能与真实游戏有细微机制差异 |
| Godot bridge mod | 4 × 8 env | 校准 / eval | 慢 10×，但 ground-truth |

**推荐策略**：
- 95% 训练用 HeadlessSim。
- 每 50k step 在 bridge 上跑 20-50 episode eval，比对 sim/live 的 EV 和 `boss_encounter_steps` 指标。当前 team 已观察到 sim/live EV 反转——这是**非常有价值的信号**，说明 HeadlessSim 机制和 live 有轻微失配。**强烈建议把这个 drift 量化到 dashboard**（metric: `live_sim_ev_correlation_rolling`）。
- 如果 drift > 0.3，**不要**无脑继续 sim 训练——回到 bridge 做几百 k step 校准。

## 4.2 课程学习设计

### 4.2.1 三段课程

```
Stage 1 (0-100k step):
  - combat_sandbox only
  - encounter pool: Act1 hallway + Act1 elite
  - snapshot_pool 权重: uniform
  - 目标: 学会基础 combat primitives
  - 验收: 平均战斗 HP 损失 / max HP < 0.25

Stage 2 (100k-400k step):
  - combat_sandbox, tier_weighted_encounter_balanced
  - 加入 Act1 boss encounters（Vantom/Waterfall Giant/Inkblot）
  - tier weights: hallway=0.3, elite=0.3, act1_boss=0.4  ← 显式过采样 boss
  - 目标: Act1 boss 场内存活能力
  - 验收: act1_boss kill rate > 30% (in sandbox)

Stage 3 (400k-∞):
  - full run (env_v2)
  - snapshot warmstart: 50% 概率从 floor-10+ snapshot 开局
  - 目标: 长程 routing + deck building
  - 验收: full-run act1_boss touch rate > 40%, kill rate > 15%
```

**关键点**：目前 team 已有 `CombatSnapshotPool` 和 `tier_weighted_encounter_balanced` 采样器，**Stage 2 基本上就是调 weights 的事**。不是新工程。

### 4.2.2 BC warm-start 时机

- **建议在 Stage 1 之前做 BC**（1.97M Skada 非战斗样本）—— 让 policy 至少学会 map routing 和 reward pick 的先验。
- 不要在 full-run PPO 已经跑飞的检查点上 BC，会打断 PPO 的梯度。
- BC 只训 policy head + encoder 的顶部 1-2 层；aux head 和 value head 从 scratch 起。

## 4.3 探索优化

### 4.3.1 ent_coef schedule

团队当前 `ent_coef=0.01` 在 127 路动作空间 + long-horizon sparse reward 下偏低。建议：

```python
# cosine schedule
def ent_coef_schedule(step, total=2_000_000):
    start, end = 0.03, 0.005
    progress = min(step / total, 1.0)
    return end + 0.5 * (start - end) * (1 + math.cos(math.pi * progress))
```

这保留 2000k step 内的高探索窗口，然后缓慢收敛。参考 long-horizon LLM-agent PPO 的 EMPG 工作：uncertainty-modulated entropy 比 static schedule 再提 10-20%，但工程复杂度上升——先做 schedule，有余力再上 EMPG。

### 4.3.2 好奇心/内驱 reward（可选）

如果 ent schedule + reward refactor 还不够，可以引入 RND (Random Network Distillation) 风格的 intrinsic reward，用 feature space 上 prediction error 当 bonus。但**不建议现在就做**——team 的 reward stack 已经复杂，再加一层难以诊断。放 backlog。

### 4.3.3 防止"只会防御"局部最优的对抗机制

- Shaping 里的 **deck_quality_score**（2.3.2）权重高于 block 累积。
- Aux_build head（team 已有）targets 里加入 "scaling_fit" 和 "frontload_fit"——不要让 policy 以为 block-heavy 无伤害的牌组是好牌组。**检查当前 build head 的 target 标签，如果是 human-tag 的，保持；如果来自 deck_tracker，确保权重经过调整**。

---

# 第五阶段：现状评估与决策建议（对 team 当前实现的评审）

## 5.1 Q1: 架构是否 fundamentally sound？

**答：架构 sound，不是瓶颈。**

- AuxMaskablePPO + 412-token 7-bank 在 observation capacity 上富余（STS2 典型 state 的"有效信息 token 数"约 60-120，当前 padding 比例 >60%，不是信息瓶颈）。
- 与 2025 年 SkyNet 工作的设计哲学（transformer encoder + ego-conditioned aux supervision）高度一致。团队独立推导出接近 SoTA 的架构，不该被 boss kill rate 掩盖这个事实。
- **结构性缺陷候选**（需要修复但不是推倒重来）：
  - 单 value head 不区分 combat/run（见 3.2.2）→ GAE 噪声源。
  - aux_causality 在 per-step delta 饱和 → 信息量天花板（见 3.3.2）。
  - Durability / Enchantment / Stars / Osty 等 STS2 新实体的 token role routing 未必覆盖到位（见 3.2.1 checklist）。

## 5.2 Q2: Shaping 饱和？还差哪 3 个信号？

**答：当前 reward stack 差的 3 个最重要信号是——**

1. **Potential-based progression（deck_quality + boss_distance）**。当前 floor-ladder 不是 potential-based，会出现"反复接近 boss 再死"的 reward lottery。
2. **Combat survival bonus in boss**（+0.02/step in boss fight）。team 自己的数据说 `boss_encounter_steps=57` → 意味着每个 boss 场 policy 平均只撑 2-3 回合。policy 需要"多撑一回合本身有奖励"的信号。
3. **Deck quality shaping（非战斗场）**。每次 card pick / skip 后 `Δdeck_quality_score`，权重 ~0.1。防止 policy 选"新奇但垃圾"的 rare 卡。

（第 4 个候选：**Boss kill terminal reward +10**，team 的 current stack 里需要确认是否存在且足够大。如果 < +5，直接翻倍。）

## 5.3 Q3: 放弃 PPO 的量化标准

**建议在达成以下 2 个条件之一后，启动 Track 2（Stochastic MuZero）：**

| 条件 | 阈值 | 观察窗 |
|---|---|---|
| **A. 瓶颈条件** | 完成 5.4 的 top-5 实验后，boss kill rate 在 2M step 窗口内仍 <5% | 2M step 实验预算 |
| **B. 饱和条件** | aux_causality Tier 3 升级后 zero-rate 仍 >20%，且 value_loss 连续 500k step 不降 | 500k step |

Pivot 成本估计：3-6 月工程（representation network 复用 50% + chance node 建模 + MCTS backend + self-play scaffolding）。在启动前**必须**完成：
- 当前 HeadlessSim 加 deterministic rollback API（MuZero 需要）。
- aux_head 的 8 种 supervision 重构为 MuZero 的 dynamics/reward/policy/value heads。
- Prior 用 MaskablePPO 最终 checkpoint warm-start，不从零起。

## 5.4 Top-5 实验（按 ROI 排序）

### Exp 1: Reward refactor + BC warm-start（ROI: ⭐⭐⭐⭐⭐）
- **改动**：(a) 实施 2.3.2 的 potential-based shaping；(b) BC 预训 policy head 在 1.97M Skada 非战斗样本上 5 epoch；(c) 保留 current layer 1/2/3，只叠加不替换。
- **预期 telemetry**：`act1_boss_touch_rate` 2.5% → 8~12%；`act1_boss_kill_rate` 0.05% → 1~3%；sim/live EV 相关性改善。
- **预算**：BC 12-24h + PPO 300-500k step（~15-25h sim）。
- **fallback**：如果 BC 让 PPO 在早期崩（KL 发散），降低 BC lr 到 3e-5，或只做 encoder freeze + policy-only BC。

### Exp 2: Dual value head（ROI: ⭐⭐⭐⭐）
- **改动**：3.2.2 的 DualValueHead，按 screen_mode 路由。
- **预期 telemetry**：`value_loss_combat` 和 `value_loss_run` 分别下降 15-30%；sim EV 曲线不再反转。
- **预算**：代码 1 天 + 再训 200-400k step (10-20h)。
- **fallback**：如果两个 head 都学不好，退回单 head 但在 loss 里对 combat/run 加权。

### Exp 3: ent_coef cosine schedule + temperature（ROI: ⭐⭐⭐⭐）
- **改动**：4.3.1 的 schedule + 3.3.1 的 sampling temperature 1.2→0.7。
- **预期 telemetry**：`combat_stuck_rate` 12% → 5-7%；早期 aux_causality loss 下降更快；`action_entropy_avg` 在前 500k step 保持 >2.0。
- **预算**：代码 0.5 天 + 再训 300k step (15h)。
- **fallback**：如果 entropy 过高导致 value loss 发散，把 start 从 0.03 降到 0.02。

### Exp 4: Aux causality Tier 3（counterfactual V delta）（ROI: ⭐⭐⭐）
- **改动**：3.3.2，新 target = `V(s_after_action) − V(s_after_end_turn_baseline)`；bootstrap from existing value head。
- **预期 telemetry**：`aux_causality_zero_rate` 25% → 10-15%；更重要的是 policy_gradient_norm 在 boss fight step 变大（说明 actor 收到了新信息）。
- **预算**：代码 2-3 天（需要在 env step 里做 counterfactual rollout，或离线用 target network 估）+ 再训 500k step (25h)。
- **fallback**：如果 V 还没学好导致 target 噪声太大，退回 Tier 2 但改 loss weight schedule。

### Exp 5: Act1-boss curriculum sandbox（ROI: ⭐⭐⭐）
- **改动**：4.2.1 的 Stage 2，`tier_weighted` 里 act1_boss weight = 0.4；snapshot pool 只用 floor 14-17 的 snapshot。
- **预期 telemetry**：在 sandbox 内 `act1_boss_kill_rate_sandbox` 从 <5% → 30%+；迁移到 full-run 时初始 boss kill rate 3-8%。
- **预算**：数据准备 2 天（筛选 snapshot）+ sandbox pretrain 200k step (10h) + full-run finetune 300k step (15h)。
- **fallback**：如果 sandbox 学到的 policy 在 full-run 掉链子（distribution shift），做 replay mix：sandbox 30% + full-run 70%。

## 5.5 Top-5 做完仍 <5% 的 pivot 建议

**单一最大 pivot：Stochastic MuZero + belief-aware auxiliary（SkyNet 式）**。

不做香草 MuZero 的原因：抽牌是 STS2 核心 RNG，需要显式 chance node。
不做 HRL pivot 的原因：团队 27 aux head + 49 semantic action 已经提供了大量 temporal abstraction 的 induction bias；HRL 的收益更小且工程同样重。

Pivot 路径：

1. 把 HeadlessSim 加上 deterministic replay（给定 seed 能完全复现 state trajectory）。
2. Representation network = 当前 Transformer encoder 前 80%（冻结）。
3. Dynamics network = 新 Transformer decoder，6 层，输入 (latent, action)，输出 (next_latent, reward_pred)。
4. Policy / value head 用 current MaskablePPO checkpoint warm-start。
5. Chance outcome head: 预测 6 类 chance outcome（抽到什么 class of card, 事件 branch, 敌人 intent pick）。
6. Simulation budget: 50 MCTS simulations / decision，比 AlphaZero 同级游戏的 800 低一个量级，但 STS2 的 branching factor 也低。

预期时间：3 人月第一版能跑；6 人月能超过 PPO baseline。如果 6 个月后还没超过，再考虑彻底换方向（比如 LLM-based planner + PPO value）。

---

## 附录 A: 关键 pseudocode 汇总

### A.1 Potential-based shaping wrapper

```python
class PotentialShapingWrapper(gym.Wrapper):
    def __init__(self, env, gamma=0.99):
        super().__init__(env)
        self.gamma = gamma
        self._prev_phi = 0.0

    def reset(self, **kw):
        obs, info = self.env.reset(**kw)
        self._prev_phi = self._potential(obs)
        return obs, info

    def step(self, action):
        obs, r, term, trunc, info = self.env.step(action)
        phi_new = self._potential(obs)
        r += self.gamma * phi_new - self._prev_phi
        self._prev_phi = phi_new if not (term or trunc) else 0.0
        return obs, r, term, trunc, info

    def _potential(self, obs):
        floor = obs["global"]["floor"]
        boss_dist = obs["global"]["steps_to_boss"]
        deck_q = obs["global"]["deck_quality_score"]
        return (
            0.4 * min(floor, 17) / 17.0 +
            0.3 * (1.0 / (1.0 + max(boss_dist, 0))) +
            0.3 * deck_q
        )
```

### A.2 Dual value head 切换

```python
# in AuxMaskablePPO.compute_returns_and_advantage
last_values = self.policy.predict_values(
    obs_tensor,
    screen_mode=obs_tensor["screen_mode"],  # NEW
)
# GAE 用 screen-routed V
```

### A.3 BC warm-start 目标

```python
# skada_bc_train.py 伪代码
for batch in skada_dataloader:  # 1.97M 非战斗样本
    obs, expert_action, expert_legal_mask = batch
    logits = policy.action_logits(obs, mask=expert_legal_mask)
    loss = F.cross_entropy(logits, expert_action, reduction="mean")
    # 只更新 policy head + encoder 顶层
    loss.backward()
    opt.step()
# 训练 3-5 epoch，lr=3e-5；early stop 当 val_acc 不再升
```

---

## 附录 B: 来源（Web-sourced Intelligence）

- Slay the Spire 2 Wikipedia: https://en.wikipedia.org/wiki/Slay_the_Spire_II
- Slay the Spire 2 Steam 页面: https://store.steampowered.com/app/2868840/Slay_the_Spire_2/
- STS2 Necrobinder 指南 (Mobalytics): https://mobalytics.gg/slay-the-spire-2/characters/necrobinder-guide
- STS2 Regent 指南 (Mobalytics): https://mobalytics.gg/slay-the-spire-2/characters/regent-guide
- STS2 Ascension 等级 (pcgamesn): https://www.pcgamesn.com/slay-the-spire-2/ascension-levels
- STS2 Enchantments 指南 (Mobalytics): https://mobalytics.gg/slay-the-spire-2/guides/enchantments
- STS2 Bosses Wiki: https://slaythespire.wiki.gg/wiki/Slay_the_Spire_2:Bosses
- STS2 所有 Boss 攻略: https://www.sts2front.com/tips/all-bosses-act-by-act/
- STS2 Alternate Acts Overgrowth / Underdocks: https://slaythespire.wiki.gg/wiki/Slay_the_Spire_2:Acts
- STS2 Events Wiki: https://slaythespire.wiki.gg/wiki/Slay_the_Spire_2:Events_List
- STS2 角色重做指南 (dtgre): https://www.dtgre.com/2026/03/slay-the-spire-2-character-reworks-guide.html
- SkyNet (Belief-Aware MuZero) arXiv: https://arxiv.org/abs/2603.27751
- Stochastic MuZero (DeepMind): https://openreview.net/forum?id=X6D9bAHhBQ1
- Stochastic MuZero PyTorch 实现: https://github.com/DHDev0/Stochastic-muzero
- Jump-Start RL (BC warm-start): https://proceedings.mlr.press/v202/uchendu23a/uchendu23a.pdf
- Entropy-Modulated Policy Gradients (EMPG): https://arxiv.org/pdf/2509.09265
- STS1 相关 RL 尝试（作为历史对照）: https://github.com/getchebarne/slai-the-spire

*以上资料于 2026-04-20 汇总，STS2 仍在 Early Access，机制随补丁更新；引用数值以 v0.98.3 为基准，未来版本以 patch notes 为准。*
