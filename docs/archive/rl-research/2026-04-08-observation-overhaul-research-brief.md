# STS2 MuZero RL Agent — Observation 编码层全面重构 实施任务书

> **文档性质**：这是一份交给实施型 AI agent 的自包含改动清单。
> 阅读者无法访问本项目的对话历史，所有必要背景信息、当前代码结构、具体改动点和验证标准都已包含在本文档中。
>
> **目标**：一次性完成 observation_v2.py 的特征维度扩展、归一化方案升级、build_aux 槽位分离，以及下游网络层的兼容性验证。所有改动在同一个 commit 中完成。

> **归档说明（后续更新）**：本文编写时 MuZero 文件仍分散在顶层与 `sts2_env/` 下。
> 当前仓库中这套实现已经统一归档到 `packages/rl-agent/muzero/`，因此文中凡是提到
> `train_muzero.py` / `sts2_env/muzero_model.py` / `sts2_env/muzero_buffer.py` /
> `sts2_env/mcts.py`，应对应理解为：
> - `python -m muzero.train`
> - `muzero/sts2_env/muzero_model.py`
> - `muzero/sts2_env/muzero_buffer.py`
> - `muzero/sts2_env/mcts.py`

---

## 第一部分：项目全景

### 1.1 项目目标

构建一个能自主游玩 Slay the Spire 2（杀戮尖塔2）的 RL AI agent。STS2 是一个 roguelike 卡牌构筑游戏，agent 需要在三个决策域中做出选择：战斗出牌（combat）、卡牌构筑（build：拿牌/升级/商店）、路线规划（route：选择地图路径）。

核心挑战：
- 动态动作空间：每步最多 80 个合法动作，每步内容不同
- 高随机性：抽牌、敌人行为、事件都有随机成分
- 长期规划：构筑决策影响几十回合后的战斗能力
- 泛化需求：希望模型能适应 mod 新卡/新角色，不依赖硬编码规则

当前方案：从 Maskable PPO 迁移到 MuZero（MCTS + 学习的世界模型），核心实现现已统一归档为 4 个主文件：`muzero/sts2_env/muzero_model.py`、`muzero/sts2_env/mcts.py`、`muzero/sts2_env/muzero_buffer.py`、`muzero/train.py`。

### 1.2 技术架构

```
C# Bridge Mod (STS2 游戏内)
    ↓ HTTP API: obs JSON + legal_actions JSON
Python RL Agent
    ├── bridge_client.py        — HTTP 客户端
    ├── combat_env.py           — Gymnasium 环境包装
    ├── observation_v2.py       — obs JSON → 结构化张量（★ 本次改动核心）
    ├── model.py                — PPO 策略网络（SharedContextEncoder + 域编码器 + CandidateScorer）
    ├── muzero/sts2_env/muzero_model.py  — MuZero 网络（RepresentationNetwork + DynamicsNetwork + PredictionNetwork）
    ├── muzero/sts2_env/mcts.py          — Sampled MuZero MCTS 搜索
    ├── muzero/sts2_env/muzero_buffer.py — 优先级 replay buffer
    ├── muzero/train.py                  — MuZero 训练循环
    ├── content_registry.py     — 卡牌/遗物/药水静态元数据注册表
    └── text_encoder.py         — BAAI/bge-small-zh-v1.5 文本嵌入（512d）
```

### 1.3 当前编码架构（改动前的基线）

**observation_v2.py 产出的 Dict observation：**

| 键名 | 形状 | 说明 |
|------|------|------|
| `scalars` | `(61,)` | 全局标量（阶段/运行状态/玩家/战斗/决策/构筑/动作摘要） |
| `decision_domain` | `(3,)` | one-hot: combat / build / route |
| `hand` | `(12, 20)` | 手牌数值特征 |
| `hand_text` | `(12, 512)` | 手牌文本嵌入 |
| `hand_mask` | `(12,)` | 手牌有效掩码 |
| `deck` | `(40, 20)` | 牌库数值特征 |
| `deck_text` | `(40, 512)` | 牌库文本嵌入 |
| `deck_mask` | `(40,)` | 牌库有效掩码 |
| `enemies` | `(5, 10)` | 敌人数值特征 |
| `enemy_text` | `(5, 512)` | 敌人文本嵌入 |
| `enemy_mask` | `(5,)` | 敌人有效掩码 |
| `player_powers` | `(20,)` | 玩家 power 状态 |
| `relics` | `(20, 512)` | 遗物文本嵌入 |
| `relic_mask` | `(20,)` | 遗物有效掩码 |
| `potions` | `(5, 512)` | 药水文本嵌入 |
| `potion_mask` | `(5,)` | 药水有效掩码 |
| `context_text` | `(512,)` | 上下文文本嵌入 |
| `actions` | `(80, 32)` | 动作数值特征 |
| `action_text` | `(80, 512)` | 动作文本嵌入 |
| `route_summary` | `(80, 20)` | 路线摘要 |
| `route_nodes` | `(80, 24, 13)` | 路线节点 |
| `route_node_mask` | `(80, 24)` | 路线节点掩码 |
| `action_mask` | `(80,)` | 合法动作掩码 |

### 1.4 网络消费方式

**model.py (PPO) 和 `muzero/sts2_env/muzero_model.py` (MuZero) 的编码器都使用同一套模式：**

```python
# 卡牌编码器（combat 手牌、build 牌库共用）
self.card_enc = nn.Sequential(
    nn.Linear(CARD_FEAT_DIM + text_proj_dim, embed_dim),  # 20+32=52 → 64
    nn.ReLU(),
)

# 敌人编码器
self.enemy_enc = nn.Sequential(
    nn.Linear(ENEMY_FEAT_DIM + text_proj_dim, embed_dim),  # 10+32=42 → 64
    nn.ReLU(),
)

# 牌库编码器
self.deck_enc = nn.Sequential(
    nn.Linear(DECK_FEAT_DIM + text_proj_dim, embed_dim),  # 20+32=52 → 64
    nn.ReLU(),
)

# 动作编码器
self.action_net = nn.Sequential(
    nn.Linear(ACTION_FEAT_DIM + text_proj_dim, action_dim),  # 32+32=64 → action_dim
    nn.ReLU(),
)
```

**关键：所有 Linear 层使用变量（CARD_FEAT_DIM 等），不是硬编码数字。改动常量后，网络层在下次实例化时自动适配新维度。已有 checkpoint 会不兼容，但项目处于重新训练阶段，这不是问题。**

### 1.5 双编码设计（已有，需增强）

当前每个实体（卡牌/敌人/动作）已经是 **结构化数值 + 文本嵌入** 双路拼接：

```
每张卡 = CARD_FEAT_DIM(20d) 数值特征 ⊕ text_proj(32d) 文本投影
         ↓
       拼接 52d → Linear → 64d embedding → Self-Attention / Cross-Attention
```

文本嵌入来自 BAAI/bge-small-zh-v1.5 (512d, L2-normalized, frozen)。内容由 `content_registry.py` 组装，格式如：`"Strike+ | 1e Attack AnyEnemy | sig dmg=6 hits=1 | tag attack"`。

**问题**："给予 2 层易伤"和"给予 3 层易伤"在文本嵌入空间中几乎相同。数值精度靠结构化特征承担，但当前只有 8 个效果槽位（damage/block/draw/weak/vulnerable/heal/hp_loss/summon），大量效果（strength/dexterity/exhaust/ethereal 等）只存在于文本嵌入中，缺乏数值表达。

---

## 第二部分：具体改动清单

> **致实施者的说明**：
>
> 以下改动应当在一个 commit 中全部完成。每个改动点都给出了：
> - 目标文件和精确的行号范围
> - 改动前的代码
> - 改动后的代码（或设计规格）
> - 验证标准
>
> 改动之间有依赖关系，请按编号顺序执行。

---

### 改动 1：新增 log1p 归一化工具函数

**文件**：`sts2_env/observation_v2.py`
**位置**：在 `_metric()` 函数（第 109-112 行）之后新增

**新增代码**：
```python
import math

# log1p 归一化锚点（控制上界）
_LOG1P_200 = math.log1p(200.0)   # damage, block, heal, hp_loss 的归一化锚点
_LOG1P_1200 = math.log1p(1200.0) # enemy HP 的归一化锚点
_LOG1P_500 = math.log1p(500.0)   # gold, item cost 的归一化锚点
_LOG1P_200_F = math.log1p(200.0) # enemy block 的归一化锚点

def _log_norm(value: float, anchor: float) -> float:
    """Log1p 归一化：保留大数值区分度，输出范围 [0, ~1.0]。

    与线性 min(x/max, 1.0) 相比：
      线性: 50→1.0, 100→1.0, 200→1.0 (截断)
      log:  50→0.74, 100→0.87, 200→1.0 (保留区分度)
    """
    if value <= 0:
        return 0.0
    return min(math.log1p(value) / anchor, 1.0)
```

**验证**：`_log_norm(50, _LOG1P_200)` ≈ 0.74, `_log_norm(100, _LOG1P_200)` ≈ 0.87, `_log_norm(200, _LOG1P_200)` ≈ 1.0

---

### 改动 2：扩展维度常量

**文件**：`sts2_env/observation_v2.py`
**位置**：第 58-63 行

**改动前**：
```python
SCALAR_DIM = 61
CARD_FEAT_DIM = 20
DECK_FEAT_DIM = 20
ENEMY_FEAT_DIM = 10
POWER_DIM = 20
ACTION_FEAT_DIM = 32
```

**改动后**：
```python
SCALAR_DIM = 61          # 不变
CARD_FEAT_DIM = 32       # 20 → 32 (+12: 4效果 + 4关键词 + 1稀有度 + 3构筑专用)
DECK_FEAT_DIM = 32       # 20 → 32 (与 CARD_FEAT_DIM 对齐)
ENEMY_FEAT_DIM = 14      # 10 → 14 (+4: intent type one-hot)
POWER_DIM = 20           # 不变
ACTION_FEAT_DIM = 40     # 32 → 40 (+8: 4效果 + 4关键词)
```

---

### 改动 3：从 content_registry 获取 keywords 和 semantic_signals

**文件**：`sts2_env/observation_v2.py`
**位置**：在 `_metric()` 之后、`DictObservationEncoder` 类之前新增

**新增代码**：
```python
from content_registry import get_card_metadata  # 需确认函数名

# 关键词到索引的映射
_CARD_KEYWORDS = {"exhaust": 0, "ethereal": 1, "retain": 2, "innate": 3}

# 稀有度到数值的映射
_RARITY_MAP = {"Common": 0.33, "Uncommon": 0.67, "Rare": 1.0}

def _get_card_keywords(card: dict) -> tuple[list[bool], float]:
    """从 bridge 运行时数据或静态元数据中提取关键词和稀有度。

    Returns:
        (keyword_flags[4], rarity_value)
    """
    flags = [False, False, False, False]
    rarity_val = 0.0

    # 优先从运行时 card dict 读取 keywords
    keywords = card.get("keywords")
    if isinstance(keywords, list):
        for kw in keywords:
            kw_lower = (kw or "").lower()
            if kw_lower in _CARD_KEYWORDS:
                flags[_CARD_KEYWORDS[kw_lower]] = True

    # 读取稀有度
    rarity = card.get("rarity", "")
    if rarity:
        rarity_val = _RARITY_MAP.get(rarity, 0.0)

    # 如果运行时数据不含 keywords/rarity，尝试从静态元数据获取
    if not any(flags) and rarity_val == 0.0:
        card_id = card.get("id", "")
        if card_id:
            try:
                from content_registry import _resolve_card_metadata
                metadata = _resolve_card_metadata(card_id)
                if metadata:
                    static_kws = metadata.get("keywords") or []
                    for kw in static_kws:
                        kw_lower = (kw or "").lower()
                        if kw_lower in _CARD_KEYWORDS:
                            flags[_CARD_KEYWORDS[kw_lower]] = True
                    if not rarity_val:
                        rarity_val = _RARITY_MAP.get(metadata.get("rarity", ""), 0.0)
            except (ImportError, Exception):
                pass

    return flags, rarity_val


def _get_card_extra_metrics(card: dict) -> tuple[float, float, float, float]:
    """提取当前 8 个效果槽之外的额外数值效果。

    Returns:
        (strength, dexterity, energy_gain, multi_hit)
    """
    strength = _metric(card, "strength") or _metric(card, "strengthGain")
    dexterity = _metric(card, "dexterity") or _metric(card, "dexterityGain")
    energy = _metric(card, "energyGain") or _metric(card, "energy")
    hits = _metric(card, "hits", default=1.0)

    # 也尝试从 semantic_signals 读取（静态元数据可能更全）
    signals = card.get("semantic_signals")
    if isinstance(signals, dict):
        if not strength:
            strength = _float(signals.get("strengthGain"))
        if not dexterity:
            dexterity = _float(signals.get("dexterityGain"))
        if not energy:
            energy = _float(signals.get("energyGain"))
        if hits <= 1.0:
            hits = _float(signals.get("hits", 1.0))

    return strength, dexterity, energy, hits
```

**注意**：`content_registry.py` 中的元数据解析函数名需要根据实际代码确认。关键路径是 `content_registry.py` 加载 `content/cards.static.generated.json` 的 `@lru_cache` 函数。搜索 `def _resolve_card` 或 `def get_card_metadata` 或 `def _load_cards` 确定实际函数名并调整 import。

---

### 改动 4：重写 `_enc_card_collection()` — CARD_FEAT_DIM 20→32

**文件**：`sts2_env/observation_v2.py`
**位置**：第 371-428 行，完整替换 `_enc_card_collection` 方法

**新的 32 维布局**：

```
索引    内容                     归一化方式          来源
────────────────────────────────────────────────────────────
[0]     cost                    /5.0               card.cost
[1]     is_attack               binary             card.type
[2]     is_skill                binary             card.type
[3]     is_power                binary             card.type
[4]     x_cost                  binary             card.x_cost
[5]     star                    /5.0               card.star
[6]     star_x                  binary             card.star_x
[7]     target_single           binary             card.target
[8]     target_all              binary             card.target
[9]     target_self             binary             card.target
[10]    is_status               binary             card.type
[11]    is_curse                binary             card.type
─── 以上 [0-11] 不变 ──────────────────────────────────────
[12]    damage                  log1p(/200)        card.damage        ← 归一化改 log1p
[13]    block                   log1p(/200)        card.block         ← 归一化改 log1p
[14]    draw                    /5.0               card.draw          （小整数，保持线性）
[15]    weak                    /5.0               card.weak
[16]    vulnerable              /5.0               card.vulnerable
[17]    heal                    log1p(/200)        card.heal          ← 归一化改 log1p
[18]    hp_loss                 log1p(/200)        card.hp_loss       ← 归一化改 log1p
[19]    summon                  /5.0               card.summon
─── 以上 [12-19] 归一化升级，不再被 build_aux 覆盖 ──────────
[20]    strength                /10.0              card.strength / semantic_signals.strengthGain
[21]    dexterity               /10.0              card.dexterity / semantic_signals.dexterityGain
[22]    energy_gain             /5.0               card.energyGain / semantic_signals.energyGain
[23]    multi_hit               /10.0              card.hits / semantic_signals.hits
─── 以上 [20-23] 新增效果数值 ─────────────────────────────
[24]    kw_exhaust              binary             keywords contains "Exhaust"
[25]    kw_ethereal             binary             keywords contains "Ethereal"
[26]    kw_retain               binary             keywords contains "Retain"
[27]    kw_innate               binary             keywords contains "Innate"
[28]    rarity                  0.33/0.67/1.0      card.rarity
─── 以上 [24-28] 新增关键词和稀有度 ───────────────────────
[29]    build_rank_norm         /option_total      build_aux.remove_rank（仅构筑时填充）
[30]    build_keep_norm         /option_total      build_aux.keep_rank
[31]    build_gap_delta         (+5)/10            build_aux.starter_gap_after
─── 以上 [29-31] 构筑专用槽（不覆盖战斗数值）──────────────
```

**新代码**：
```python
def _enc_card_collection(
    self,
    cards: list,
    numeric: np.ndarray,
    text: np.ndarray,
    mask: np.ndarray,
) -> None:
    max_items = numeric.shape[0]
    texts: list[str] = []
    text_slots: list[int] = []
    for index, card in enumerate(cards[:max_items]):
        if not isinstance(card, dict):
            continue
        mask[index] = 1.0
        row = numeric[index]

        # [0-11] 静态属性 — 不变
        row[0] = min(_float(card.get("cost")) / 5.0, 1.0)
        card_type = (card.get("type") or "").capitalize()
        row[1] = 1.0 if card_type == "Attack" else 0.0
        row[2] = 1.0 if card_type == "Skill" else 0.0
        row[3] = 1.0 if card_type == "Power" else 0.0
        row[4] = _bool(card.get("x_cost"))
        row[5] = min(_float(card.get("star")) / 5.0, 1.0) if card.get("star") is not None else 0.0
        row[6] = _bool(card.get("star_x"))
        target = (card.get("target") or "").lower()
        row[7] = 1.0 if "single" in target or "anyenemy" in target else 0.0
        row[8] = 1.0 if "all" in target else 0.0
        row[9] = 1.0 if "self" in target else 0.0
        row[10] = 1.0 if card_type == "Status" else 0.0
        row[11] = 1.0 if card_type == "Curse" else 0.0

        # [12-19] 战斗效果数值 — log1p 归一化，永远不被覆盖
        row[12] = _log_norm(_metric(card, "damage"), _LOG1P_200)
        row[13] = _log_norm(_metric(card, "block"), _LOG1P_200)
        row[14] = min(_metric(card, "draw") / 5.0, 1.0)
        row[15] = min(_metric(card, "weak") / 5.0, 1.0)
        row[16] = min(_metric(card, "vulnerable") / 5.0, 1.0)
        row[17] = _log_norm(_metric(card, "heal"), _LOG1P_200)
        row[18] = _log_norm(_metric(card, "hp_loss"), _LOG1P_200)
        row[19] = min(_metric(card, "summon") / 5.0, 1.0)

        # [20-23] 额外效果数值
        strength, dexterity, energy, hits = _get_card_extra_metrics(card)
        row[20] = min(strength / 10.0, 1.0)
        row[21] = min(dexterity / 10.0, 1.0)
        row[22] = min(energy / 5.0, 1.0)
        row[23] = min(hits / 10.0, 1.0)

        # [24-28] 关键词和稀有度
        kw_flags, rarity_val = _get_card_keywords(card)
        row[24] = 1.0 if kw_flags[0] else 0.0  # exhaust
        row[25] = 1.0 if kw_flags[1] else 0.0  # ethereal
        row[26] = 1.0 if kw_flags[2] else 0.0  # retain
        row[27] = 1.0 if kw_flags[3] else 0.0  # innate
        row[28] = rarity_val

        # [29-31] 构筑专用槽（独立区域，不覆盖战斗数值）
        build_aux = card.get("build_aux")
        if isinstance(build_aux, dict):
            option_total = max(_float(build_aux.get("option_total")), 1.0)
            row[29] = min(_float(build_aux.get("remove_rank")) / option_total, 1.0)
            row[30] = min(_float(build_aux.get("keep_rank")) / option_total, 1.0)
            row[31] = min((_float(build_aux.get("starter_gap_after")) + 5.0) / 10.0, 1.0)

        # 文本嵌入
        text_value = self._build_live_card_text(card)
        if text_value and self.use_text:
            texts.append(text_value)
            text_slots.append(index)

    if texts:
        embeddings = self._get_encoder().encode_batch(texts)
        for embedding_index, slot in enumerate(text_slots):
            text[slot] = embeddings[embedding_index]
```

**注意**：`_enc_hand` 和 `_enc_deck` 都调用 `_enc_card_collection`，所以改一处即覆盖两者。hand 用的是 `(MAX_HAND, CARD_FEAT_DIM)` = `(12, 32)`，deck 用的是 `(MAX_DECK, DECK_FEAT_DIM)` = `(40, 32)`。

**build_aux 信息缩减说明**：原来 build_aux 会覆盖 `row[12:19]` 共 8 个槽位写入详细的构筑评估数据（remove_rank, keep_rank, count_before, count_after, junk_after, starter_attack_after, starter_defend_after, starter_gap_after）。现在只保留 3 个最关键的到独立槽位 `[29-31]`。丢掉的 5 个字段（count_before, count_after, junk_after, starter_attack_after, starter_defend_after）可以后续按需扩展 CARD_FEAT_DIM，但当前阶段的核心目标是让战斗数值永远可见，所以先精简 build_aux。

---

### 改动 5：重写 `_enc_enemies()` — ENEMY_FEAT_DIM 10→14

**文件**：`sts2_env/observation_v2.py`
**位置**：第 430-472 行，完整替换 `_enc_enemies` 方法

**新的 14 维布局**：

```
索引    内容                     归一化方式          来源
────────────────────────────────────────────────────────────
[0]     hp_ratio                hp/max_hp          enemy.hp / enemy.max_hp
[1]     hp_absolute             log1p(/1200)       enemy.hp                ← 改 log1p
[2]     max_hp                  log1p(/1200)       enemy.max_hp            ← 改 log1p
[3]     block                   log1p(/200)        enemy.block             ← 改 log1p
[4]     intent_damage           log1p(/200)        intent.total_damage     ← 改 log1p（原/80截断）
[5]     intent_repeats          /5.0               intent.repeats
[6]     power_count             /5.0               len(enemy.powers)
[7]     has_vulnerable          binary             powers 中含 vulnerable
[8]     has_weak                binary             powers 中含 weak
[9]     has_strength            binary             powers 中含 strength
─── 以上 [0-9] 归一化升级 ─────────────────────────────────
[10]    intent_is_attack        binary             intent 含伤害 (total_damage > 0)
[11]    intent_is_defend        binary             intent 含防御信号
[12]    intent_is_buff          binary             intent 含 buff 信号
[13]    intent_is_debuff        binary             intent 含 debuff 信号
─── 以上 [10-13] 新增 intent 类型 ─────────────────────────
```

**新代码**：
```python
def _enc_enemies(self, enemies: np.ndarray, enemy_text: np.ndarray, enemy_mask: np.ndarray, obs: dict) -> None:
    combat = obs.get("combat") or {}
    entries = combat.get("enemies") or []
    texts: list[str] = []
    text_slots: list[int] = []

    for index, enemy in enumerate(entries[:MAX_ENEMIES]):
        if not isinstance(enemy, dict):
            continue
        enemy_mask[index] = 1.0
        row = enemies[index]

        hp = _float(enemy.get("hp"))
        max_hp = _float(enemy.get("max_hp"))
        row[0] = min(hp / max_hp, 1.0) if max_hp > 0 else 0.0
        row[1] = _log_norm(hp, _LOG1P_1200)
        row[2] = _log_norm(max_hp, _LOG1P_1200)
        row[3] = _log_norm(_float(enemy.get("block")), _LOG1P_200_F)

        intent = enemy.get("intent") or {}
        total_damage = _float(intent.get("total_damage"))
        row[4] = _log_norm(total_damage, _LOG1P_200)
        row[5] = min(_float(intent.get("repeats")) / 5.0, 1.0)

        powers = enemy.get("powers") or []
        row[6] = min(len(powers) / 5.0, 1.0)
        for power in powers:
            if not isinstance(power, dict):
                continue
            title = (power.get("title") or "").lower()
            if "vulnerable" in title:
                row[7] = 1.0
            elif "weak" in title:
                row[8] = 1.0
            elif "strength" in title:
                row[9] = 1.0

        # [10-13] intent 类型 one-hot
        row[10] = 1.0 if total_damage > 0 else 0.0
        # intent_is_defend / buff / debuff：从 intent dict 的其他字段推断
        intent_type = (intent.get("type") or intent.get("id") or "").lower()
        row[11] = 1.0 if any(kw in intent_type for kw in ("defend", "block", "shield")) else 0.0
        row[12] = 1.0 if any(kw in intent_type for kw in ("buff", "strength", "ritual", "enrage")) else 0.0
        row[13] = 1.0 if any(kw in intent_type for kw in ("debuff", "weak", "vulnerable", "frail")) else 0.0

        if self.use_text:
            text_value = build_live_enemy_semantic_text(enemy)
            if text_value:
                texts.append(text_value)
                text_slots.append(index)

    if texts:
        embeddings = self._get_encoder().encode_batch(texts)
        for embedding_index, slot in enumerate(text_slots):
            enemy_text[slot] = embeddings[embedding_index]
```

**注意**：intent 类型的具体字段名（`intent.type` vs `intent.id`）需要根据 bridge 实际返回的 JSON 结构确认。如果 bridge 的 intent 只有 `total_damage` 和 `repeats`，那么 `[11-13]` 可能需要从 intent 的文本描述或其他字段推断。最坏情况下这三个字段填 0.0，等 bridge 端补充后再启用。`[10]` (intent_is_attack) 始终可用因为只依赖 `total_damage > 0`。

---

### 改动 6：重写 `_enc_action_numeric()` — ACTION_FEAT_DIM 32→40

**文件**：`sts2_env/observation_v2.py`
**位置**：第 718-826 行，完整替换 `_enc_action_numeric` 方法

**新的 40 维布局**：

```
索引    内容                     来源                   备注
────────────────────────────────────────────────────────────
[0]     action_kind             action.kind            不变
[1]     has_card                card exists            不变
[2]     card_cost               card.cost /5           不变
[3]     card_star               card.star /5           不变
[4]     card_is_attack          card.type              不变
[5]     card_is_skill           card.type              不变
[6]     card_is_power           card.type              不变
[7]     has_target_name         target.name            不变
[8]     target_is_player        target.side            不变
[9]     is_end_turn             action_id              不变
[10]    is_proceed              kind==proceed           不变
[11]    is_skip                 action_id              不变
[12]    item_cost               item.cost              log1p(/500) ← 归一化改 log1p
[13]    reward_is_gold          reward.type            不变
[14]    reward_is_card          reward.type            不变
[15]    point_type              action.point_type      不变
[16]    coord_row               coord.row /15          不变
[17]    coord_col               coord.col /7           不变
[18]    has_upgrade_preview     upgrade_preview        不变
[19]    option_index            index /20              不变
[20]    is_status               card.type              不变
[21]    is_curse                card.type              不变
─── 以上 [0-21] 布局不变（build_aux 不再覆盖 [7-21]）────
[22]    damage                  source.damage          log1p(/200) ← 归一化改
[23]    block                   source.block           log1p(/200) ← 归一化改
[24]    draw                    source.draw /5         不变
[25]    weak                    source.weak /5         不变
[26]    vulnerable              source.vulnerable /5   不变
[27]    heal                    source.heal            log1p(/200) ← 归一化改
[28]    hp_loss                 source.hp_loss         log1p(/200) ← 归一化改
[29]    summon                  source.summon /5       不变
─── 以上 [22-29] 战斗效果（不再被 build_aux 覆盖）────────
[30]    strength                source.strength /10    新增
[31]    dexterity               source.dexterity /10   新增
[32]    energy_gain             source.energyGain /5   新增
[33]    multi_hit               source.hits /10        新增
[34]    kw_exhaust              binary                 新增
[35]    kw_ethereal             binary                 新增
[36]    kw_retain               binary                 新增
[37]    kw_innate               binary                 新增
[38]    has_positive_effect     aggregated binary       原 row[30] 移至此
[39]    is_free_card            cost==0 binary          原 row[31] 移至此
─── 以上 [30-39] 新增效果 + 关键词 + 迁移的旗标 ──────────
```

**新代码（核心部分，省略 [0-21] 不变的部分）**：

```python
def _enc_action_numeric(self, row: np.ndarray, action: dict) -> None:
    kind = action.get("kind", "")
    row[0] = _KIND_TO_ORD.get(kind, 0) / _NUM_KINDS

    card = action.get("card")
    if isinstance(card, dict):
        row[1] = 1.0
        row[2] = min(_float(card.get("cost")) / 5.0, 1.0)
        row[3] = min(_float(card.get("star")) / 5.0, 1.0) if card.get("star") is not None else 0.0
        card_type = (card.get("type") or "").capitalize()
        row[4] = 1.0 if card_type == "Attack" else 0.0
        row[5] = 1.0 if card_type == "Skill" else 0.0
        row[6] = 1.0 if card_type == "Power" else 0.0
        row[20] = 1.0 if card_type == "Status" else 0.0
        row[21] = 1.0 if card_type == "Curse" else 0.0

    target = action.get("target")
    row[7] = 1.0 if isinstance(target, dict) and target.get("name") else 0.0
    row[8] = 1.0 if isinstance(target, dict) and target.get("side") == "Player" else 0.0
    row[9] = 1.0 if action.get("action_id") == "end_turn" else 0.0
    row[10] = 1.0 if kind == "proceed" and not action.get("skip") else 0.0
    row[11] = 1.0 if action.get("skip") or "skip" in (action.get("action_id") or "") else 0.0

    item = action.get("item")
    if isinstance(item, dict):
        row[12] = _log_norm(_float(item.get("cost")), _LOG1P_500)

    reward = action.get("reward")
    if isinstance(reward, dict):
        reward_type = reward.get("type", "")
        row[13] = 1.0 if reward_type == "gold" else 0.0
        row[14] = 1.0 if reward_type == "card" else 0.0

    point_type = self._normalize_route_point_type(action.get("point_type_norm") or action.get("point_type", ""))
    row[15] = _PT_TO_ORD.get(point_type, 0) / _NUM_PT

    coord = action.get("coord")
    if isinstance(coord, dict):
        row[16] = min(_float(coord.get("row")) / 15.0, 1.0)
        row[17] = min(_float(coord.get("col")) / 7.0, 1.0)

    row[18] = 1.0 if isinstance(action.get("upgrade_preview"), dict) else 0.0

    option_index = action.get("index")
    if option_index is None:
        option_index = action.get("hand_index")
    if option_index is None:
        option_index = action.get("slot_index")
    row[19] = min(_float(option_index) / 20.0, 1.0) if option_index is not None else 0.0

    # ---- 效果数值 [22-37] ----
    source = None
    if isinstance(card, dict):
        source = card
    else:
        potion = action.get("potion")
        if isinstance(potion, dict):
            source = potion

    if isinstance(source, dict):
        # [22-29] 基础效果 — log1p 归一化
        row[22] = _log_norm(_metric(source, "damage"), _LOG1P_200)
        row[23] = _log_norm(_metric(source, "block"), _LOG1P_200)
        row[24] = min(_metric(source, "draw") / 5.0, 1.0)
        row[25] = min(_metric(source, "weak") / 5.0, 1.0)
        row[26] = min(_metric(source, "vulnerable") / 5.0, 1.0)
        row[27] = _log_norm(_metric(source, "heal"), _LOG1P_200)
        row[28] = _log_norm(_metric(source, "hp_loss"), _LOG1P_200)
        row[29] = min(_metric(source, "summon") / 5.0, 1.0)

        # [30-33] 新增效果数值
        strength, dexterity, energy, hits = _get_card_extra_metrics(source)
        row[30] = min(strength / 10.0, 1.0)
        row[31] = min(dexterity / 10.0, 1.0)
        row[32] = min(energy / 5.0, 1.0)
        row[33] = min(hits / 10.0, 1.0)

        # [34-37] 关键词
        kw_flags, _ = _get_card_keywords(source)
        row[34] = 1.0 if kw_flags[0] else 0.0  # exhaust
        row[35] = 1.0 if kw_flags[1] else 0.0  # ethereal
        row[36] = 1.0 if kw_flags[2] else 0.0  # retain
        row[37] = 1.0 if kw_flags[3] else 0.0  # innate

        # [38] has_positive_effect (原 row[30])
        row[38] = 1.0 if (
            _metric(source, "damage") > 0
            or _metric(source, "block") > 0
            or _metric(source, "draw") > 0
            or _metric(source, "weak") > 0
            or _metric(source, "vulnerable") > 0
            or _metric(source, "heal") > 0
            or strength > 0
            or dexterity > 0
            or _metric(source, "summon") > 0
        ) else 0.0

        # [39] is_free_card (原 row[31])
        row[39] = 1.0 if isinstance(card, dict) and _float(card.get("cost")) == 0 else 0.0

    # ---- build_aux: 不再覆盖任何已填充的槽位 ----
    # build_aux 信息现在通过 action_text 的文本嵌入传递，
    # 以及通过 card 级别的 build_aux 槽位 [29-31]（在 _enc_card_collection 中处理）。
    # 如果后续需要在 action 级别恢复 build_aux 数值特征，
    # 可以将 ACTION_FEAT_DIM 进一步扩展到 48+，在 [40:] 添加专用槽位。
```

**关于 build_aux 在 action 编码中的处理**：

原来 `_enc_action_numeric` 的 build_aux 分支会覆盖 `row[7:31]` 共 25 个槽位。这些 build_aux 特征包括：is_starter_attack, is_starter_defend, is_curse_or_status, shop_remove_rate, gap_improves, remove_any_rate, reward_rate, smith_rate, transform_rate, primary_rate, balanced_after, remove_rank_is_one, rate_rank, is_largest_stack, count_share, remove_rank, keep_rank, primary_score, keep_score, count_before, count_after, junk_after, starter_attack_after, starter_defend_after, starter_gap_delta。

**处理方案（二选一，实施者根据项目阶段选择）**：

**方案 A（推荐，本次采用）：彻底移除 action 级 build_aux 覆盖。** build_aux 信息通过以下渠道传递：
1. `_enc_card_collection` 中卡牌级别的 `row[29-31]`（remove_rank, keep_rank, gap_delta）
2. `action_text` 的文本嵌入中包含构筑语义
3. `decision_domain` one-hot 告诉模型当前处于 build 域

这意味着构筑决策会暂时丢失一些精细的启发式特征（如 smith_rate, primary_score）。但核心目标是让模型通过 MCTS 搜索自主学习构筑策略，而不是依赖预计算的启发式。

**方案 B（保守）：ACTION_FEAT_DIM 扩展到 56，在 `[40:55]` 添加 build_aux 专用槽位。** 如果实施者判断构筑性能回退不可接受，采用此方案。

---

### 改动 7：更新 `_enc_scalars` 中的归一化

**文件**：`sts2_env/observation_v2.py`
**位置**：`_enc_scalars` 方法内的若干行

**逐点改动**：

```python
# 第 252 行 — player block
# 改前: vector[offset + 3] = min(_float(player.get("block")) / 100.0, 1.0)
# 改后:
vector[offset + 3] = _log_norm(_float(player.get("block")), _LOG1P_200)

# 第 253 行 — player gold
# 改前: vector[offset + 4] = min(_float(player.get("gold")) / 500.0, 1.0)
# 改后:
vector[offset + 4] = _log_norm(_float(player.get("gold")), _LOG1P_500)

# 第 303 行 — build state gold (重复的 gold)
# 改前: vector[offset + 5] = min(_float(player.get("gold")) / 999.0, 1.0)
# 改后:
vector[offset + 5] = _log_norm(_float(player.get("gold")), _LOG1P_500)
```

SCALAR_DIM 保持 61 不变，只改归一化方式。

---

### 改动 8：更新 obs_space 形状声明

**文件**：`sts2_env/observation_v2.py`
**位置**：`obs_space` property 内（第 130-160 行）

这部分使用的是变量（`CARD_FEAT_DIM`、`ENEMY_FEAT_DIM`、`ACTION_FEAT_DIM`），所以**改动常量后会自动适配**。无需手动修改。

同理，`encode()` 方法中的 `np.zeros` 调用也使用变量：
```python
hand = np.zeros((MAX_HAND, CARD_FEAT_DIM), dtype=np.float32)     # 自动变成 (12, 32)
enemies = np.zeros((MAX_ENEMIES, ENEMY_FEAT_DIM), dtype=np.float32)  # 自动变成 (5, 14)
actions = np.zeros((MAX_ACTIONS, ACTION_FEAT_DIM), dtype=np.float32) # 自动变成 (80, 40)
```

**验证**：确认没有任何地方硬编码了 `20`、`10`、`32` 这些旧维度数字。

---

### 改动 9：验证 model.py 网络层兼容性

**文件**：`sts2_env/model.py`

**需检查的关键行**：

```python
# 第 96 行 — card_enc 输入
self.card_enc = nn.Sequential(nn.Linear(CARD_FEAT_DIM + text_proj_dim, embed_dim), nn.ReLU())
# CARD_FEAT_DIM=32, text_proj_dim=32 → 输入 64, 输出 64 ✓ 自动适配

# 第 97 行 — enemy_enc 输入
self.enemy_enc = nn.Sequential(nn.Linear(ENEMY_FEAT_DIM + text_proj_dim, embed_dim), nn.ReLU())
# ENEMY_FEAT_DIM=14, text_proj_dim=32 → 输入 46, 输出 64 ✓ 自动适配

# 第 144 行 — deck_enc 输入
self.deck_enc = nn.Sequential(nn.Linear(DECK_FEAT_DIM + text_proj_dim, embed_dim), nn.ReLU())
# DECK_FEAT_DIM=32, text_proj_dim=32 → 输入 64, 输出 64 ✓ 自动适配

# 第 181、200 行 — action_enc 输入
nn.Linear(ACTION_FEAT_DIM + text_proj_dim, action_dim)
# ACTION_FEAT_DIM=40, text_proj_dim=32 → 输入 72, 输出 action_dim ✓ 自动适配

# 第 67 行 — scalar_net 输入
nn.Linear(SCALAR_DIM + POWER_DIM + context_text_dim + 16 + 16 + NUM_DOMAINS, hidden_dim)
# SCALAR_DIM=61 不变 ✓
```

**结论**：model.py 无需改动。所有 Linear 层使用导入的变量，自动适配新维度。

**唯一需要检查的硬编码**：搜索 model.py 中的数字字面量 `20`、`10`、`32`（不在注释中的），确认没有被用作维度常量。

---

### 改动 10：验证 `muzero/sts2_env/muzero_model.py` 网络层兼容性

**文件**：`muzero/sts2_env/muzero_model.py`

**需检查的关键行**：

```python
# 第 249 行 — ActionEncoder
nn.Linear(ACTION_FEAT_DIM + text_proj_dim, action_embed_dim)
# ACTION_FEAT_DIM=40, text_proj_dim=32 → 输入 72 ✓ 自动适配

# 第 324 行 — card_enc
nn.Linear(CARD_FEAT_DIM + text_proj_dim, embed_dim)
# CARD_FEAT_DIM=32 ✓ 自动适配

# 第 328 行 — enemy_enc
nn.Linear(ENEMY_FEAT_DIM + text_proj_dim, embed_dim)
# ENEMY_FEAT_DIM=14 ✓ 自动适配

# 第 353 行 — deck_enc
nn.Linear(DECK_FEAT_DIM + text_proj_dim, embed_dim)
# DECK_FEAT_DIM=32 ✓ 自动适配

# 第 364 行 — final_net shared_size
shared_size = SCALAR_DIM + POWER_DIM + text_proj_dim + 16 + 16 + NUM_DOMAINS
# SCALAR_DIM=61 不变 ✓
```

**结论**：`muzero/sts2_env/muzero_model.py` 无需改动。

---

### 改动 11：验证 combat_env.py 兼容性

**文件**：`sts2_env/combat_env.py`

`action_masks()` 方法返回 `[MAX_ACTIONS]` binary mask，与 observation 维度无关。`step()` 和 `reset()` 调用 `obs_encoder.encode()`，维度变化对其透明。

**结论**：combat_env.py 无需改动。

---

### 改动 12：验证 `muzero/train.py` 兼容性

**文件**：`muzero/train.py`

训练循环使用 `DictObservationEncoder` 的输出，通过 dict key 访问各张量。维度变化对其透明，因为 batch 维度的 shape 来自 obs_space。

**唯一潜在问题**：如果训练代码中有硬编码 `action_mask_batch[:, :, :80]` 之类的切片，需要检查。但 MAX_ACTIONS=80 没有变化，所以这不是问题。

**结论**：`muzero/train.py` 无需改动。

---

### 改动 13：验证 `muzero/sts2_env/muzero_buffer.py` 兼容性

**文件**：`muzero/sts2_env/muzero_buffer.py`

buffer 存储的是完整的 obs dict + action_mask(80,)。obs dict 中各张量的维度变化对 buffer 透明（它只做 list append 和 batch stack）。

**结论**：`muzero/sts2_env/muzero_buffer.py` 无需改动。

---

## 第三部分：已知信息（请勿重复调研）

### 已有的数据来源

| 数据 | 文件 | 状态 |
|------|------|------|
| 卡牌静态元数据（keywords, semantic_signals, rarity, semantic_tags） | `content/cards.static.generated.json` (1.39MB) | ✓ 可用 |
| 卡牌动态数据（运行时效果数值 damage/block/...） | bridge HTTP API 返回的 card dict | ✓ 可用 |
| 24 种 semantic_signals | content_registry.py `_CARD_SIGNAL_ALIASES` | ✓ 可用 |
| 文本嵌入 | text_encoder.py (bge-small-zh-v1.5, 512d) | ✓ 可用，不需改动 |
| intent 类型字段 | bridge 返回的 `enemy.intent` dict | ⚠️ 需确认具体字段名 |

### 已验证的兼容性

| 文件 | 是否使用变量 | 是否需要改动 |
|------|------------|------------|
| model.py | 是，全部用 CARD_FEAT_DIM 等变量 | **不需要** |
| `muzero/sts2_env/muzero_model.py` | 是 | **不需要** |
| combat_env.py | 不直接引用维度常量 | **不需要** |
| `muzero/train.py` | 通过 DictObservationEncoder 间接使用 | **不需要** |
| `muzero/sts2_env/muzero_buffer.py` | 存储 dict，维度透明 | **不需要** |
| train_v2.py (PPO) | 通过 DictObservationEncoder 间接使用 | **不需要** |

### 参数量影响估算

```
CARD_FEAT_DIM +12 → card_enc: +12×64 = +768 params (× 两处 = +1536)
DECK_FEAT_DIM +12 → deck_enc: +12×64 = +768 params
ENEMY_FEAT_DIM +4 → enemy_enc: +4×64 = +256 params
ACTION_FEAT_DIM +8 → action_enc: +8×64 = +512 params (× 多处)
总计增量: ~3000-4000 params（可忽略）
```

---

## 第四部分：验证清单

实施完成后，执行以下验证：

### 4.1 维度一致性测试

```python
# 构造一个 mock bridge obs，验证编码后的张量形状
from sts2_env.observation_v2 import DictObservationEncoder, CARD_FEAT_DIM, DECK_FEAT_DIM, ENEMY_FEAT_DIM, ACTION_FEAT_DIM, SCALAR_DIM

encoder = DictObservationEncoder(use_text=False)  # 跳过文本编码加速测试

# 验证常量值
assert CARD_FEAT_DIM == 32
assert DECK_FEAT_DIM == 32
assert ENEMY_FEAT_DIM == 14
assert ACTION_FEAT_DIM == 40
assert SCALAR_DIM == 61

# 验证 obs_space 形状
space = encoder.obs_space
assert space["hand"].shape == (12, 32)
assert space["deck"].shape == (40, 32)
assert space["enemies"].shape == (5, 14)
assert space["actions"].shape == (80, 40)
assert space["scalars"].shape == (61,)
```

### 4.2 log1p 归一化范围测试

```python
from sts2_env.observation_v2 import _log_norm, _LOG1P_200

# 验证典型值
assert abs(_log_norm(0, _LOG1P_200)) < 0.001
assert 0.70 < _log_norm(50, _LOG1P_200) < 0.78
assert 0.84 < _log_norm(100, _LOG1P_200) < 0.90
assert 0.97 < _log_norm(200, _LOG1P_200) <= 1.0
assert _log_norm(500, _LOG1P_200) == 1.0  # 超出锚点被 clamp
```

### 4.3 build_aux 不覆盖战斗数值测试

```python
# 构造一张有 build_aux 的卡牌
card_with_build_aux = {
    "type": "Attack", "cost": 1, "damage": 15, "block": 0,
    "target": "AnyEnemy", "draw": 0, "weak": 0, "vulnerable": 0,
    "heal": 0, "hp_loss": 0, "summon": 0,
    "build_aux": {"remove_rank": 2, "keep_rank": 3, "option_total": 5,
                  "starter_gap_after": 1, "deck_after_size": 20}
}

# 编码
import numpy as np
row = np.zeros(CARD_FEAT_DIM)
# 调用 _enc_card_collection 的逻辑

# 验证 damage 没有被覆盖
assert row[12] > 0  # damage = 15, log1p(15)/log1p(200) ≈ 0.52
# 验证 build_aux 在独立槽位
assert row[29] > 0  # remove_rank / option_total = 2/5 = 0.4
```

### 4.4 网络实例化测试

```python
import torch
from sts2_env.model import STS2CandidateScoringPolicy  # 或具体的网络类
from sts2_env.muzero_model import MuZeroNetwork

# 验证 PPO 网络能实例化
# ppo_net = ... (依赖 sb3 接口，可能需要完整 obs_space)

# 验证 MuZero 网络能实例化
muzero_net = MuZeroNetwork()
print(f"MuZero total params: {sum(p.numel() for p in muzero_net.parameters())}")
```

### 4.5 关键词提取测试

```python
from sts2_env.observation_v2 import _get_card_keywords

# 测试运行时 keywords
card_exhaust = {"keywords": ["Exhaust", "Sly"], "rarity": "Uncommon"}
flags, rarity = _get_card_keywords(card_exhaust)
assert flags[0] == True   # exhaust
assert flags[1] == False   # ethereal
assert flags[2] == False   # retain
assert flags[3] == False   # innate
assert rarity == 0.67      # Uncommon

# 测试无 keywords 的卡
card_basic = {"type": "Attack", "cost": 1}
flags, rarity = _get_card_keywords(card_basic)
assert not any(flags)
assert rarity == 0.0
```

---

## 第五部分：改动文件清单总结

| # | 文件 | 改动类型 | 改动量 |
|---|------|---------|--------|
| 1 | `sts2_env/observation_v2.py` | **核心重写** | ~200 行修改 |
| 2 | `sts2_env/model.py` | **仅验证** | 0 行（确认无硬编码） |
| 3 | `muzero/sts2_env/muzero_model.py` | **仅验证** | 0 行（确认无硬编码） |
| 4 | `sts2_env/combat_env.py` | **仅验证** | 0 行 |
| 5 | `muzero/train.py` | **仅验证** | 0 行 |
| 6 | `muzero/sts2_env/muzero_buffer.py` | **仅验证** | 0 行 |

**实际需要写代码的只有 observation_v2.py 一个文件。**

---

## 第六部分：实施顺序建议

1. 在文件顶部添加 `import math` 和 log1p 归一化工具函数
2. 修改 6 个维度常量
3. 添加 `_get_card_keywords()` 和 `_get_card_extra_metrics()` 辅助函数
4. 重写 `_enc_card_collection()`（同时影响 hand 和 deck 编码）
5. 重写 `_enc_enemies()`
6. 重写 `_enc_action_numeric()`（注意移除 build_aux 覆盖逻辑）
7. 更新 `_enc_scalars()` 中的 3 处归一化
8. 运行第四部分的所有验证测试
9. 在 `model.py` 和 `muzero/sts2_env/muzero_model.py` 中 grep 数字字面量确认无硬编码

---

*本文档生成日期：2026-04-08*
*项目阶段：MuZero 迁移 Phase 1 — 训练前特征层重构*
