# STS2 当前特征工程设计说明

本文描述当前项目主线实现中的**特征工程设计**，重点说明原始游戏状态如何被加工为模型输入，以及这些输入如何服务于当前的 `omni_attention_v1_frozen` 架构。

---

## 1. 一句话总结

当前项目采用的是一套**分层语义增强的 token 化特征工程系统**：

- 先从原始 bridge 状态中提取基础数值特征和语义文本；
- 再补充跨时间步的 run memory / objective context；
- 再把每个合法动作编码成语义动作表示；
- 最终组织成 `world / candidate query / candidate local` 三路输入；
- 然后交给共享 trunk + omni-attention 做关系建模。

它不是传统的“单个大向量”方案，也不是纯 raw JSON end-to-end，而是：

> **手工工程负责把状态拆对、压对、对齐对；注意力模型负责学习跨对象、跨表面、跨时序的组合关系。**

---

## 2. 整体流水线

```text
bridge/raw obs + legal_actions + planner_context
    -> observation_common.py   基础数值/文本/preview 特征
    -> run_memory.py           长程记忆 / 构筑画像 / objective
    -> semantic_action.py      动作语义签名
    -> observation_v3.py       world/query/local 三路 token 化
    -> omni_attention_policy.py
         - shared numeric trunk
         - shared text trunk
         - world self-attention
         - query -> local bridge
         - banked world cross-attention
         - policy/value/aux heads
```

当前相关版本标记：

- observation API: `attention_obs_v5_pass_large`
- attention architecture: `omni_attention_v1_frozen`

---

## 3. 基础特征层：`observation_common.py`

这一层是当前系统的**底层特征工具箱**。虽然旧 dense 路径已经不是主输入，但其大量底层特征提取逻辑仍被 `observation_v3.py` 复用。

### 3.1 关键维度

```text
SCALAR_DIM        = 61
CARD_FEAT_DIM     = 41
DECK_FEAT_DIM     = 41
ENEMY_FEAT_DIM    = 27
POWER_DIM         = 24
RELIC_SIGNAL_DIM  = 12
ACTION_FEAT_DIM   = 49
RUN_MEMORY_DIM    = 48
OBJECTIVE_DIM     = 16
```

### 3.2 关键上限

```text
MAX_HAND        = 12
MAX_DECK        = 64
MAX_ENEMIES     = 5
MAX_RELICS      = 40
MAX_POTIONS     = 5
MAX_ACTIONS     = 96
MAX_ROUTE_NODES = 48
```

### 3.3 这一层负责什么

它负责把原始状态里的低层信息规范化成可复用的数值特征，包括但不限于：

- **全局标量**
  - phase
  - act / floor / room_type
  - 玩家 hp / max_hp / block / gold
  - combat energy / stars
  - combat 中 action 的总体统计
- **卡牌基础特征**
  - cost / star / type
  - preview damage / block / draw / weak / vulnerable / heal / hp_loss
  - summon / strength / dexterity / energy / hits
  - exhaust / ethereal / retain / innate
  - rarity
  - damage/block per energy
  - upgrade level
  - x_cost_value
- **敌人基础特征**
  - hp / max_hp / block
  - intent 相关数值
  - power amount map
  - 典型负面/正面状态归一化
- **玩家 power 特征**
  - strength / dexterity / weak / vulnerable / frail
  - plating / ritual / metallicize / barricade / rage
  - vigor / intangible / thorns / regen / artifact
  - poison / calamity / buffer / entangled / lockon
- **遗物信号**
  - energy / draw / strength / dexterity / vigor / thorns
  - intangible / artifact / poison / calamity / defense / regen
- **动作基础数值**
  - kind / target / skip / end_turn
  - item cost / reward type
  - route point type / coordinate
  - upgrade preview / slot index
  - source card / potion 的 preview 统计
- **路线摘要与路线节点特征**
  - route summary
  - route node type / depth / row / col / child_count / leaf

换句话说，这一层做的是：

> **把原始 bridge 状态标准化为稳定、可拼装、可复用的底层特征部件。**

---

## 4. 长程特征层：`run_memory.py`

这一层不是“当前帧状态”，而是**跨时间步、跨战斗、跨整局**的持久上下文。

### 4.1 `RUN_MEMORY_DIM = 48`

`RunMemoryTracker` 会持续累计如下信息：

- combats / elites / bosses seen
- rests / shops / events seen
- card reward picks
- potion uses
- smith / rest 次数
- map choices
- gold spent
- cumulative hp loss
- recent hp loss
- lowest hp ratio seen
- 当前 floor / act
- last semantic action
- 当前 episode mode
  - `combat_sandbox`
  - `full_run`
- 当前环境是否启用 potion mechanics

### 4.2 构筑画像 `_build_profile(obs)`

这一层会生成 deck-level profile：

- `deck_size`
- `frontload`
- `block`
- `draw`
- `scaling`
- `aoe`
- `heal`
- `curse_density`
- `high_cost_density`
- `zero_cost_density`
- `x_cost_density`
- `consistency`
- `build_gap_risk`

这不是模型自己从 deck token 生推的隐藏变量，而是**显式手工提炼出的构筑画像**。

### 4.3 `OBJECTIVE_CONTEXT_DIM = 16`

在 run memory 基础上继续派生出 objective context，例如：

- `survival_priority`
- `hp_loss_priority`
- `build_priority`
- `resource_priority`
- `preserve_hp_bias`
- `save_potion_mode`
- `force_rest_mode`
- `greed_upgrade_mode`
- `elite_pressure`
- `boss_pressure`
- `safe_route_bias`
- `shop_value_bias`
- `rest_value_bias`
- `smith_value_bias`
- `zero_damage_desire`
- `long_horizon_mode`

也就是说，模型输入里不仅有“现在是什么状态”，还有“现在这局整体应当偏什么目标”。

---

## 5. 动作语义层：`semantic_action.py`

这一层把 live legal action 从“临时槽位”提升为“稳定语义动作”。

### 5.1 动作 family

当前包含：

- `play_card`
- `use_potion`
- `discard_potion`
- `card_selection`
- `end_turn`
- `proceed`
- `map`
- `reward`
- `card_reward`
- `shop`
- `rest`
- `smith`
- `deck_upgrade`
- `event_option`
- `treasure_relic`
- `startup`
- `other`

### 5.2 target scope

- `none`
- `self`
- `single_enemy`
- `all_enemies`
- `choice`
- `map`
- `shop`
- `event`
- `other`

### 5.3 semantic roles

- `attack`
- `block`
- `draw`
- `debuff`
- `buff`
- `heal`
- `aoe`
- `x_cost`
- `setup`
- `scaling`
- `resource`
- `terminal`

### 5.4 语义 numeric

`encode_semantic_action_numeric(...)` 会把动作签名编码成固定向量，覆盖：

- family one-hot
- target scope one-hot
- role one-hot
- cost
- price
- damage / block / draw / heal / hp_loss
- hits / damage_per_hit
- x_cost_value
- target_index / choice_index
- star
- upgrade_level
- is_zero_cost

因此当前动作表征已经不是单纯的 `action_id + mask`，而是：

> **动作基础数值 + 动作语义签名 + 动作/语义文本嵌入**

---

## 6. 主输入层：`observation_v3.py`

这是当前特征工程真正的核心落地点。

### 6.1 token 规格

```text
MAX_WORLD_TOKENS            = 320
MAX_CANDIDATE_LOCAL_TOKENS  = 24
TOKEN_NUMERIC_DIM           = 96
TOKEN_TEXT_DIM              = 64
TOKEN_FEAT_DIM              = 160
ENTITY_HASH_BUCKETS         = 8192
```

每个 token 的载荷为：

```text
[96 维 numeric | 64 维 text]
```

### 6.2 每个 token 的结构标签

除了数值和文本，每个 token 还带有结构化元信息：

- `type_id`
- `role_id`
- `owner_id`
- `entity_id`
- `zone_id`
- `order_id`

其中 query token 还附带：

- `target_owner_id`
- `target_entity_id`

也就是说，输入不是“无标签向量列表”，而是：

> **数值/文本载荷 + 类型/角色/归属/实体/区域/顺序的结构化标记**

---

## 7. 三路输入设计：`world / query / local`

当前主输入被拆成三路：

1. `world_tokens`
2. `candidate_query_tokens`
3. `candidate_local_tokens`

这三路并不是冗余重复，而是三个不同的建模表面。

---

## 8. `world_tokens`：全局世界记忆面

world 负责承载完整的全局状态。

### 8.1 全局与玩家状态 token

主要包括：

- `CLS_WORLD`
- `CLS_COMBAT`
- `CLS_BUILD`
- `CLS_ROUTE`
- `PLAYER_SURVIVAL`
- `RESOURCE_BUDGET`
- `THREAT_SUMMARY`
- `OBJECTIVE_CONTEXT`
- `RUN_CONTEXT`

这些 token 覆盖了：

- 玩家血量 / 最大血量 / 格挡
- incoming damage
- 当前能量 / 最大能量 / stars
- 金币
- 全局 threat
- objective context
- run memory

因此你关心的：

- **血条**
- **能量**
- **整体局势目标**
- **保命 / 贪 / 省药偏置**

都已经是 world surface 的显式输入。

### 8.2 卡牌相关 world token

当前 world 中直接包含多个牌堆：

- `HAND_CARD`
- `DRAW_PREVIEW_CARD`
- `DISCARD_CARD`
- `EXHAUST_CARD`
- `PLAY_PILE_CARD`
- `DECK_CARD`

也就是说，以下信息都在 world 中显式存在：

- 手牌
- 抽牌堆
- 弃牌堆
- 消耗堆
- 出牌堆
- 整体牌库

每张卡 token 都已经带有数值特征与文本特征，而不是单独的卡 ID。

### 8.3 遗物 / 药水 world token

world 里直接有：

- `RELIC`
- `POTION`

并且遗物还有 numeric signal，例如：

- energy
- draw
- strength
- dexterity
- vigor
- thorns
- intangible
- artifact
- poison
- defense
- regen

这意味着模型在 world 上天然可见：

- 当前遗物系统提供了什么类型的增益
- 当前药水槽里有什么语义资源

### 8.4 敌人 world token

敌人不是一个粗糙 token，而是被拆成多个语义子 token：

- `ENEMY_CORE`
- `ENEMY_INTENT`
- `ENEMY_POWER`
- `ENEMY_REACTIVE_TRAIT`
- `ENEMY_PHASE_RULE`

这样做的意义是把“敌人的不同信息面”拆开，便于注意力做结构化组合。

已覆盖的敌人信息面包括：

- 血量 / 格挡
- 伤害意图 / hits / repeats
- power amount
- thorn / artifact / buffer / intangible
- reactive traits
- phase rules / threshold mechanics

### 8.5 路线 world token

路线信息也直接进入 world：

- `ROUTE_SUMMARY_TOKEN`
- `ROUTE_NODE`

其中 route summary 包括：

- reachable node count
- max depth
- direct child count
- forced path steps before branch
- 各种节点计数
  - monster / elite / boss / event / question / rest / shop / treasure
- 到下一个关键节点的步数
  - next elite / rest / shop / event / question / treasure
- `can_reach_rest_site_before_elite`
- `can_reach_elite_then_rest_site`

而 route node 还会给出：

- node type one-hot
- depth
- row / col
- child_count
- is_leaf

---

## 9. `candidate_query_tokens`：每个合法动作一个 query

每个 legal action 都被编码成一个 query token，对应四种决策域：

- `COMBAT_CANDIDATE`
- `BUILD_CANDIDATE`
- `SELECTION_CANDIDATE`
- `ROUTE_CANDIDATE`

### 9.1 query token 的组成

一个 query token 通常包含：

- 49 维动作基础特征
- 压缩后的 semantic action numeric
- action text embedding
- semantic action text embedding
- source / target 结构绑定信息
  - owner
  - entity
  - zone
  - order

### 9.2 关键意义

这意味着当前系统不是“先做外部搜索，再把搜索结论塞给 policy”，而是：

> **把每个动作本身当作一个需要被建模的查询对象。**

同时，selection surface 也被单独建模，不再粗暴混进 build。

---

## 10. `candidate_local_tokens`：每个动作的局部上下文

这层是当前特征工程最关键的设计之一。

每个 action 最多可附带 `24` 个 local token，用来描述：

> **“如果我要评估这个动作，我最应该额外看到哪些局部上下文？”**

这使得模型不必每次都从整个 world 里盲扫所有信息，而可以先看一组与当前动作强相关的局部上下文。

---

## 11. Combat 决策时的特征工程

这是当前系统最丰富的一块。

### 11.1 source / target 相关 token

- `SOURCE_CARD_LOCAL`
- `SOURCE_POTION_LOCAL`
- `TARGET_LOCAL`

这类 token 提供：

- 当前动作的源对象
- 当前目标敌人
- 源对象与目标对象的局部绑定

### 11.2 玩家与资源 token

- `PLAYER_STATE_LOCAL`
- `ENERGY_CONTEXT_LOCAL`
- `ENERGY_BUDGET_LOCAL`

这些 token 会考虑：

- 当前能量 / 最大能量
- source cost
- 是否 zero-cost
- 是否 x-cost
- source 自身回能
- best energy potion
- relic energy signal
- relic draw signal
- 当前能否出牌
- 加 support 后能否出牌
- x 费牌的可扩张 spend

因此当前模型输入已经显式覆盖了：

- **费用**
- **能量**
- **X 费**
- **药水/遗物提供的能量扩张**

### 11.3 牌堆上下文 token

combat local 会显式补：

- `DRAW_CONTEXT_LOCAL`
- `DISCARD_CONTEXT_LOCAL`
- `EXHAUST_CONTEXT_LOCAL`
- `PLAY_PILE_CONTEXT_LOCAL`

这些 token 会概括 pile 的结构，例如：

- 张数
- attack / skill / power / curse/status 比例
- zero-cost 密度
- exhaust / ethereal / retain / innate 密度
- same-source 出现率
- total damage / block / draw / energy / hits

### 11.4 pile binding token

还会继续补：

- `DRAW_BINDING_LOCAL`
- `DISCARD_BINDING_LOCAL`
- `EXHAUST_BINDING_LOCAL`
- `PLAY_BINDING_LOCAL`

它们描述“当前 source card 与各 pile 的关系”，例如：

- 同名卡是否在 pile 里出现
- 最靠前的位置
- 是否 top match / near match
- pile 的 zero-cost / draw / exhaust / retain 密度

这对循环、回手、洗牌、同名卡跟踪非常重要。

### 11.5 循环规划 token

- `CYCLE_PLAN_LOCAL`

它会看：

- hand / draw / discard / exhaust / play pile 的规模
- 是否接近 reshuffle
- source 是否带 cycle keyword
  - draw
  - discard
  - shuffle
  - return to hand
- source 在各 pile 中的位置
- 当前动作是否有利于循环或回收

这意味着当前输入已经不仅仅是“看手牌”，而是把：

- **抽牌堆**
- **弃牌堆**
- **消耗堆**
- **出牌堆**
- **循环关系**

都纳入了动作局部特征工程。

### 11.6 support token：遗物 / 药水

combat local 中还会补：

- `RELIC_TRIGGER_LOCAL`
- `POTION_OPTION_LOCAL`
- `RELIC_POTION_GRAPH_LOCAL`

#### `RELIC_TRIGGER_LOCAL`

会对当前 action 与各遗物做 relevance 打分，取最相关的遗物进入 local。

#### `POTION_OPTION_LOCAL`

会对当前 action 与当前药水槽内容做 relevance 打分，取最相关的药水进入 local。

#### `RELIC_POTION_GRAPH_LOCAL`

会把 support 系统进一步压成图式摘要，例如：

- top relic relevance
- avg relic relevance
- top potion relevance
- avg potion relevance
- best energy / damage / block / draw potion
- relic_signals
- support 与敌方反制的耦合
- x_cost 与额外能量 support 的耦合
- exhaust / multi-hit 与 relic/potion 的耦合

因此当前 combat 特征工程已经显式支持：

- **玩家出牌时考虑药水**
- **玩家出牌时考虑遗物**
- **药水/遗物和 source card 的联动**

### 11.7 目标反制 token

- `TARGET_REACTION_LOCAL`

它会显式编码：

- `thorns`
- `contact_punish`
- `split`
- `threshold`
- `threshold_value`
- 当前 source 是否 attack-like
- 是否 multi-hit
- 预估是否能击杀
- 敌方 intent 伤害
- 敌方 block / hp
- 是否 x_cost
- 是否有 artifact / buffer / intangible
- 是否带 weak / vulnerable
- 是否 aoe

这正是你关心的那一类：

- 荆棘
- 反伤
- threshold / 分裂 / phase rule
- 某些受击后触发的特殊敌人机制

### 11.8 目标敌人上下文补充

如果动作带 target，还会把目标敌人的局部上下文补回 local：

- `ENEMY_INTENT`
- `ENEMY_POWER`
- `ENEMY_REACTIVE_TRAIT`
- `ENEMY_PHASE_RULE`

因此 target-aware 决策并不只是 query 上的 target index，而是：

> **目标敌人状态 + 目标敌人反制 + source card profile 的联合局部建模**

---

## 12. Build / 选牌 / 路线决策时的特征工程

### 12.1 Build 决策

build candidate 会补：

- `BUILD_STATE_LOCAL`
- `DECK_SYNERGY_LOCAL`
- `SHOP_ECON_LOCAL`（当动作来自商店）

#### `BUILD_STATE_LOCAL`

汇总：

- run memory 前 24 维
- objective 前 24 维
- 当前 deck size
- gold
- relic count
- potion count
- 部分 action 数值

#### `DECK_SYNERGY_LOCAL`

这不是“卡本体特征”，而是：

> **候选卡 × 当前牌库画像**

它会比较：

- deck 中 attack / skill / power 比例
- avg cost
- avg damage / block / draw / energy / exhaust
- 同名卡数量
- 候选卡是否补 deck 短板
- 候选卡是否是 zero-cost / x-cost / retain / exhaust 等

#### `SHOP_ECON_LOCAL`

补充：

- 当前 gold
- item cost
- 是否买得起
- 价格占金币比例
- item 类型
  - card / relic / potion / remove

### 12.2 Selection 决策

selection 现在单独建模，不再只是 build 的附属物。

query 侧：

- `SELECTION_CANDIDATE`

local 侧：

- `SELECTION_OPERATOR_LOCAL`
- `SELECTION_SEMANTICS_LOCAL`
- `DECK_SYNERGY_LOCAL`（如果选项关联卡）

#### `SELECTION_OPERATOR_LOCAL`

编码：

- confirm / cancel / close / skip
- upgrade / smith
- transform / mutate
- remove / purge
- exhaust / consume
- discard
- discover / draft / reward / choose / pick
- 是否发生在 combat 中

#### `SELECTION_SEMANTICS_LOCAL`

编码：

- 作用于 hand / draw / discard / exhaust / deck / play pile
- upgrade / transform / remove
- retain / bottle / duplicate
- reward / discover

如果 selection 发生在 combat 中，还会额外补：

- pile context
- relic triggers
- potion options

所以当前“强化 / 变形 / remove / exhaust / 战斗内选牌”已经有自己的表面，而不是全部压成构筑通道。

### 12.3 Route 决策

route candidate 会补：

- `ROUTE_RISK_LOCAL`
- `ROUTE_VALUE_LOCAL`

两者都基于 route summary，但强调角度不同：

#### `ROUTE_RISK_LOCAL`

偏重：

- elite / boss 风险
- 某些高风险结构
- 当前 hp ratio
- 当前 gold

#### `ROUTE_VALUE_LOCAL`

偏重：

- rest / shop / treasure / question / event 等收益结构
- 到关键节点的步数

同时 route action 本身还携带：

- `route_summary`
- `route_nodes`

因此路线决策不是一维评分，而是：

> **路线全局摘要 + 路线节点序列 + 风险局部视角 + 收益局部视角**

---

## 13. 文本特征工程：`text_encoder.py`

### 13.1 文本编码模型

当前文本编码器使用：

- `BAAI/bge-small-zh-v1.5`
- `TEXT_DIM = 512`

特点：

- 冻结
- L2 normalize
- 内存缓存
- 磁盘缓存

### 13.2 输入侧文本压缩

在 `observation_v3.py` 中，512 维原始文本 embedding 会被压到：

- `TOKEN_TEXT_DIM = 64`

### 13.3 文本去重与缓存复用

当前已经有两级文本复用：

#### 第一级：step 内 text registry

同一步内相同文本只 encode 一次，再分发给多个 token。

#### 第二级：全局 TextEncoder cache

跨 step / 跨 episode / 跨进程共享磁盘与内存缓存。

因此当前文本输入并不是“每个地方重复编码一次”，而是有明确的**去重与缓存复用**。

---

## 14. 模型入口前的共享 trunk：`omni_attention_policy.py`

在 token 化之后，world / query / local 不再各自维护独立投影器，而是先走共享 trunk。

### 14.1 共享 trunk

当前有：

- `shared_numeric_trunk`
- `shared_text_trunk`

三路输入都会复用这两套 trunk。

### 14.2 重复行复用

在 `_project_shared_modal_trunk_with_reuse(...)` 中：

- 会把 world/query/local 的 numeric 或 text 先拼起来；
- 对完全相同的行做 `torch.unique`；
- 只投影一次；
- 再按原 shape 还原。

因此在模型入口前还做了一次**跨表面的投影复用**。

---

## 15. 注意力如何消费这些特征

当前 frozen 模型结构是：

1. `world` 先做 self-attention
2. `query` 先通过 `query -> local bridge` 读取自身 local context
3. `candidate_x` 再去跨注意力读取 world

### 15.1 world bank 显式分组

world 被分成 5 个 bank：

- `runtime`
- `support`
- `enemy`
- `build`
- `route`

大致对应：

- **runtime**
  - player state
  - resource
  - threat
  - objective
  - run memory
  - hand / draw / discard / exhaust / play
  - pile link / cycle plan / energy budget
- **support**
  - relic
  - potion
  - support graph
- **enemy**
  - enemy core
  - intent
  - power
  - trait
  - reaction
- **build**
  - deck_card
  - build_state
  - deck_synergy
  - reward / shop / upgrade
- **route**
  - route_summary
  - route_node
  - route_risk
  - route_value

### 15.2 banked world cross-attention

candidate 不是无脑 attend 全部 world token，而是：

- 先根据 `world_bank_summaries` 做 routing
- 选 top-k bank
- 再到选中的 bank 上做 cross-attention

这意味着当前特征工程不只是“构造 token”，还已经**按 bank-aware world memory** 的思路组织好了输入表面。

---

## 16. 从关心点来看，当前输入覆盖了什么

| 关注点 | 当前输入覆盖方式 |
|---|---|
| 手牌 | `HAND_CARD` + `SOURCE_CARD_LOCAL` |
| 抽牌堆 | `DRAW_PREVIEW_CARD` + `DRAW_CONTEXT_LOCAL` + `DRAW_BINDING_LOCAL` |
| 弃牌堆 | `DISCARD_CARD` + `DISCARD_CONTEXT_LOCAL` + `DISCARD_BINDING_LOCAL` |
| 消耗堆 | `EXHAUST_CARD` + `EXHAUST_CONTEXT_LOCAL` + `EXHAUST_BINDING_LOCAL` |
| 出牌堆 | `PLAY_PILE_CARD` + `PLAY_PILE_CONTEXT_LOCAL` + `PLAY_BINDING_LOCAL` |
| 打循环 / 控牌 | `CYCLE_PLAN_LOCAL` + pile binding + pile summary |
| 全局牌库构筑 | `DECK_CARD` + `BUILD_STATE_LOCAL` + `DECK_SYNERGY_LOCAL` + build profile |
| 药水 | `POTION` + `POTION_OPTION_LOCAL` + semantic action |
| 遗物 | `RELIC` + `RELIC_TRIGGER_LOCAL` + `RELIC_POTION_GRAPH_LOCAL` + relic signals |
| 能量 / X费 | `RESOURCE_BUDGET` + `ENERGY_CONTEXT_LOCAL` + `ENERGY_BUDGET_LOCAL` + semantic x_cost |
| 玩家血条 | `PLAYER_SURVIVAL` + `PLAYER_STATE_LOCAL` |
| 敌人状态 | `ENEMY_CORE` + `ENEMY_POWER` + `ENEMY_REACTIVE_TRAIT` + `ENEMY_PHASE_RULE` |
| 敌人意图 | `ENEMY_INTENT` |
| 荆棘 / 反制 / 阈值 | `TARGET_REACTION_LOCAL` + enemy power / trait / phase rule |
| 选牌 / 强化 / 变形 / remove / exhaust | `SELECTION_CANDIDATE` + `SELECTION_OPERATOR_LOCAL` + `SELECTION_SEMANTICS_LOCAL` |
| 路线抉择 | `ROUTE_SUMMARY_TOKEN` + `ROUTE_NODE` + `ROUTE_RISK_LOCAL` + `ROUTE_VALUE_LOCAL` |

---

## 17. 当前设计的本质定位

### 17.1 它不是纯 raw end-to-end

当前仍然存在大量手工工程：

- preview metric
- 数值归一化
- relic signal
- action semantic signature
- run memory
- objective context
- build profile
- route summary
- pile summary / pile binding / cycle plan
- target reaction
- selection semantics

### 17.2 它也不是传统硬规则打分器

这些手工工程的主要作用不是直接决定动作，而是：

- 先把信息拆成更容易对齐的对象
- 给模型提供稳定的关系 scaffold
- 然后把组合关系交给 attention 学

所以更准确的说法是：

> **这是一个“强语义 scaffold + 注意力主导建模”的特征工程系统。**

---

## 18. 最终结论

如果用一句话总结当前项目的特征工程：

> **当前系统已经把手牌、牌堆、遗物、药水、敌人状态、敌人反制、能量、X 费、全局构筑、路线、选择面、长期 run memory 全部拆成 `world/query/local` 三层语义 token，并让 attention 在这些对象之间学习交互。**

它的优势在于：

- 覆盖面已经很完整
- 不是只看手牌
- 不是只看单步动作
- 不是只做外部搜索
- 已经能把 combat / build / selection / route 统一到同一套输入结构下
- 已经显式对接了 banked world cross-attention 的结构需求

它的边界在于：

- 输入端仍然有较强的手工偏置
- 还不是完全去工程化的纯内在建模

但从当前主线来看，这套设计已经明确站在：

> **“用工程把对象表面搭好，再让大部分关系交给注意力去学”**

这一边。
