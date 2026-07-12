# Ironclad / Colorless 卡牌机制覆盖审计

Source: `repo://game-data/generated/cards.static.generated.json`

## 结论摘要

1. 规范化游戏数据中的 **Ironclad 基础牌是 90 张**，不是 88 张。这里没有强行丢牌：`Ancient`/`Event`/`Basic` 也一起导出。若后续要严格对齐“奖励池 88 张”，需要再按奖励池/掉落池规则过滤。
2. 规范化游戏数据中的 **Colorless 基础牌是 128 张**；其中剔除 Status/Curse/Quest/Token 后的可主动使用无色牌为 **81 张**，Uncommon/Rare 常规无色牌为 **64 张**。
3. 基础伤害/格挡/抽牌/回费/生命代价/X 费/消耗/虚无/保留/运行时附魔，已经有明确 bridge + observation 特征路径。
4. 最大风险不是“模型完全看不到”，而是当前 `semanticTags/semanticSignals` 和 source catalog 只能给粗粒度内部族群（例如 `Upgrade`/`Exhaust`/`Transform`/`AddGeneratedCardToCombat`/`EnergyCost.Set*`），还缺 **zone/scope/filter/selection/count/destination/duration/modifier/result_card** 等 typed operation 参数。文本匹配只保留离线告警，不能作为训练主契约。

## 输出文件

- `repo://docs/generated/ironclad-cards-base.json`：Ironclad 基础牌完整导出 + 分类。
- `repo://docs/generated/colorless-cards-base.json`：Colorless 基础牌完整导出 + 分类。
- `repo://docs/generated/card-mechanism-coverage.csv`：Ironclad + Colorless 全量机制覆盖表，适合 Excel/TB 外部检查。
- `repo://docs/generated/ironclad-card-roster.md`：Ironclad 全表。
- `repo://docs/generated/colorless-card-roster.md`：Colorless 全表。
- `repo://packages/rl-agent/tools/audit_card_mechanism_coverage.py`：可重复生成脚本。

Internal source catalog: `artifact://dependencies/sts2-ai/Assets/datasets/game_knowledge_catalog/cards.jsonl`；本次审计中有 source profile 的卡：**218/218**。

Checked-in 文件只能通过固定边界发布；该模式拒绝 `--items`、`--out-dir` 和 `--report` 覆盖：

```powershell
python tools/third_party/restore_sts2_ai.py
python packages/rl-agent/tools/audit_card_mechanism_coverage.py --publish-checked-in-docs
```

不带该开关的运行时审计始终写入 `STS2_ARTIFACT_ROOT/reports/card-mechanism`，不能写回源码树。

## 牌数与分布

| 项目 | 数量 |
| --- | --- |
| Ironclad base cards | 90 |
| Colorless base cards | 128 |
| Colorless playable non Status/Curse/Quest/Token | 81 |
| Colorless Uncommon/Rare | 64 |
| Total audited base cards | 218 |
| Internal catalog cards loaded | 577 |
| Audited cards with source profile | 218 |

### 稀有度分布

| 颜色 | 稀有度 | 数量 |
| --- | --- | --- |
| ironclad | Ancient | 2 |
| ironclad | Basic | 3 |
| ironclad | Common | 20 |
| ironclad | Event | 3 |
| ironclad | Rare | 26 |
| ironclad | Uncommon | 36 |
| colorless | Ancient | 8 |
| colorless | Curse | 18 |
| colorless | Event | 9 |
| colorless | Quest | 3 |
| colorless | Rare | 25 |
| colorless | Status | 16 |
| colorless | Token | 10 |
| colorless | Uncommon | 39 |

### 类型分布

| 颜色 | 类型 | 数量 |
| --- | --- | --- |
| ironclad | Attack | 39 |
| ironclad | Power | 21 |
| ironclad | Skill | 30 |
| colorless | Attack | 31 |
| colorless | Curse | 18 |
| colorless | None | 1 |
| colorless | Power | 13 |
| colorless | Quest | 3 |
| colorless | Skill | 46 |
| colorless | Status | 16 |

## 覆盖状态汇总

这里的 `gap` 指“typed effect profile / operation 参数还没进入稳定契约”，不是说 transformer 完全没有文本 token 可看；也不是要求继续堆中文/英文正则。

| 颜色 | covered | partial | gap | unclassified |
| --- | --- | --- | --- | --- |
| ironclad | 21 | 35 | 34 | 0 |
| colorless | 28 | 19 | 81 | 0 |
| all | 49 | 54 | 115 | 0 |

## 机制覆盖矩阵

| 机制分类 | 总数 | Ironclad | Colorless | 覆盖状态 | 当前通道 | 例子 |
| --- | --- | --- | --- | --- | --- | --- |
| effect_profile_granularity_gap | 88 | 34 | 54 | gap | semanticTags/internal command ids are still too coarse | 涅奥之怒(CARD.NEOWS_FURY)、神化(CARD.APOTHEOSIS)、灵体(CARD.APPARITION)、许愿(CARD.WISH)、进阶之灾(CARD.ASCENDERS_BANE)、霉运(CARD.BAD_LUCK) |
| damage | 82 | 43 | 39 | covered | effect_preview/static semantic_signals/action profile | 撕咬(CARD.MAUL)、涅奥之怒(CARD.NEOWS_FURY)、吹哨(CARD.WHISTLE)、腐朽(CARD.DECAY)、异鸟扑击(CARD.BYRD_SWOOP)、杀灭(CARD.EXTERMINATE) |
| global_rule_power | 80 | 42 | 38 | partial | powers/text + world tokens | 灵体(CARD.APPARITION)、放松(CARD.RELAX)、疑虑(CARD.DOUBT)、羞耻(CARD.SHAME)、压扁(CARD.SQUASH)、疯狂科学(CARD.MAD_SCIENCE) |
| turn_or_delayed_rule | 65 | 28 | 37 | partial | powers/text + state transitions | 放松(CARD.RELAX)、霉运(CARD.BAD_LUCK)、债务(CARD.DEBT)、腐朽(CARD.DECAY)、疑虑(CARD.DOUBT)、凡庸(CARD.NORMALITY) |
| draw | 54 | 17 | 37 | covered | effect_preview/static semantic_signals/action profile | 涅奥之怒(CARD.NEOWS_FURY)、至亮之焰(CARD.BRIGHTEST_FLAME)、放松(CARD.RELAX)、许愿(CARD.WISH)、疯狂科学(CARD.MAD_SCIENCE)、羽化(CARD.METAMORPHOSIS) |
| exhaust_self | 51 | 12 | 39 | covered | keyword/card_flow/action quality | 涅奥之怒(CARD.NEOWS_FURY)、吹哨(CARD.WHISTLE)、神化(CARD.APOTHEOSIS)、灵体(CARD.APPARITION)、放松(CARD.RELAX)、许愿(CARD.WISH) |
| block | 41 | 23 | 18 | covered | effect_preview/static semantic_signals/action profile | 放松(CARD.RELAX)、疯狂科学(CARD.MAD_SCIENCE)、坚韧之环(CARD.TORIC_TOUGHNESS)、希望灯塔(CARD.BEACON_OF_HOPE)、永恒铠甲(CARD.ETERNAL_ARMOR)、拟态(CARD.MIMIC) |
| random_or_choose | 34 | 14 | 20 | partial | semantic tags/text/card-selection surfaces | 涅奥之怒(CARD.NEOWS_FURY)、疯狂科学(CARD.MAD_SCIENCE)、羽化(CARD.METAMORPHOSIS)、大奖(CARD.JACKPOT)、灾祸(CARD.CALAMITY)、炼制药水(CARD.ALCHEMIZE) |
| status_or_curse_penalty_gap | 34 | 0 | 34 | gap | raw text + keywords only | 进阶之灾(CARD.ASCENDERS_BANE)、霉运(CARD.BAD_LUCK)、笨拙(CARD.CLUMSY)、铃铛的诅咒(CARD.CURSE_OF_THE_BELL)、债务(CARD.DEBT)、腐朽(CARD.DECAY) |
| exhaust_other_or_hand | 32 | 15 | 17 | partial | semantic tags/internal Exhaust command + hand_mutation local | 许愿(CARD.WISH)、开悟(CARD.ENLIGHTENMENT)、羽化(CARD.METAMORPHOSIS)、炼制药水(CARD.ALCHEMIZE)、天选(CARD.ANOINTED)、潦草急就(CARD.SCRAWL) |
| hp_or_self_cost | 31 | 16 | 15 | covered | effect_preview.hp_loss/static semantic_signals + risk classifier | 至亮之焰(CARD.BRIGHTEST_FLAME)、霉运(CARD.BAD_LUCK)、债务(CARD.DEBT)、腐朽(CARD.DECAY)、悔恨(CARD.REGRET)、击倒(CARD.KNOCKDOWN) |
| multi_hit_or_aoe | 30 | 18 | 12 | covered | effect_preview hits/target + semantic tags | 撕咬(CARD.MAUL)、杀灭(CARD.EXTERMINATE)、啄击(CARD.PECK)、滚石(CARD.ROLLING_BOULDER)、狠揍(CARD.BEAT_DOWN)、扫荡凝视(CARD.SWEEPING_GAZE) |
| whole_hand_state | 30 | 15 | 15 | partial | internal PileType.Hand access + card_flow | 涅奥之怒(CARD.NEOWS_FURY)、许愿(CARD.WISH)、开悟(CARD.ENLIGHTENMENT)、大奖(CARD.JACKPOT)、箭雨(CARD.SALVO)、天选(CARD.ANOINTED) |
| cost_modify | 26 | 13 | 13 | partial | internal EnergyCost / power ids + hand_mutation local | 神化(CARD.APOTHEOSIS)、开悟(CARD.ENLIGHTENMENT)、羽化(CARD.METAMORPHOSIS)、灾祸(CARD.CALAMITY)、乱战(CARD.MAYHEM)、怀旧(CARD.NOSTALGIA) |
| hard_rule_constraint_gap | 26 | 0 | 26 | gap | mostly raw text/legal mask | 进阶之灾(CARD.ASCENDERS_BANE)、霉运(CARD.BAD_LUCK)、笨拙(CARD.CLUMSY)、铃铛的诅咒(CARD.CURSE_OF_THE_BELL)、债务(CARD.DEBT)、腐朽(CARD.DECAY) |
| pile_fetch_reorder | 24 | 6 | 18 | partial | pile tokens + internal PileType/CardPileCmd ids | 涅奥之怒(CARD.NEOWS_FURY)、许愿(CARD.WISH)、疯狂科学(CARD.MAD_SCIENCE)、羽化(CARD.METAMORPHOSIS)、流星锤(CARD.BOLAS)、乱战(CARD.MAYHEM) |
| add_or_generate_card | 22 | 6 | 16 | partial | semantic tags/internal command ids + future hand snapshot | 涅奥之怒(CARD.NEOWS_FURY)、许愿(CARD.WISH)、羽化(CARD.METAMORPHOSIS)、大奖(CARD.JACKPOT)、灾祸(CARD.CALAMITY)、天选(CARD.ANOINTED) |
| card_modifier_or_enchantment | 22 | 2 | 20 | partial | runtime modifier_summary + internal modifier ids | 灵体(CARD.APPARITION)、许愿(CARD.WISH)、进阶之灾(CARD.ASCENDERS_BANE)、霉运(CARD.BAD_LUCK)、笨拙(CARD.CLUMSY)、铃铛的诅咒(CARD.CURSE_OF_THE_BELL) |
| debuff | 17 | 13 | 4 | covered | effect_preview weak/vulnerable/poison + semantic tags | 疑虑(CARD.DOUBT)、压扁(CARD.SQUASH)、疯狂科学(CARD.MAD_SCIENCE)、震荡波(CARD.SHOCKWAVE)、破击(CARD.BREAK)、痛击(CARD.BASH) |
| buff_stats | 16 | 10 | 6 | covered | effect_preview strength/dexterity + semantic tags | 灵体(CARD.APPARITION)、疯狂进食(CARD.FEEDING_FRENZY)、永恒铠甲(CARD.ETERNAL_ARMOR)、非凡技艺(CARD.PROWESS)、协同配合(CARD.COORDINATE)、黑暗镣铐(CARD.DARK_SHACKLES) |
| energy_gain | 13 | 5 | 8 | covered | static semantic_signals + action quality resource timing | 至亮之焰(CARD.BRIGHTEST_FLAME)、放松(CARD.RELAX)、燃料(CARD.FUEL)、冷光(CARD.LUMINESCE)、自动化(CARD.AUTOMATION)、相信着你(CARD.BELIEVE_IN_YOU) |
| retain | 11 | 0 | 11 | covered | keyword/card_flow/action quality | 许愿(CARD.WISH)、睡眠不佳(CARD.POOR_SLEEP)、金斧(CARD.GOLD_AXE)、箭雨(CARD.SALVO)、天选(CARD.ANOINTED)、潦草急就(CARD.SCRAWL) |
| ethereal | 7 | 0 | 7 | covered | keyword/card_flow/action quality | 灵体(CARD.APPARITION)、进阶之灾(CARD.ASCENDERS_BANE)、笨拙(CARD.CLUMSY)、愚行(CARD.FOLLY)、晕眩(CARD.DAZED)、虚空(CARD.VOID) |
| upgrade_card | 7 | 4 | 3 | partial | semantic tags/internal Upgrade command + hand_mutation local | 神化(CARD.APOTHEOSIS)、大奖(CARD.JACKPOT)、飞溅(CARD.SPLASH)、武装(CARD.ARMAMENTS)、好勇斗狠(CARD.AGGRESSION)、原始力量(CARD.PRIMAL_FORCE) |
| exhaust_pile_dependency | 6 | 5 | 1 | partial | pile binding tokens + text/static tags | 君王之剑(CARD.SOVEREIGN_BLADE)、契约终结(CARD.PACTS_END)、灰烬打击(CARD.ASHEN_STRIKE)、彼岸咆哮(CARD.HOWL_FROM_BEYOND)、邪眼(CARD.EVIL_EYE)、被遗忘的仪式(CARD.FORGOTTEN_RITUAL) |
| play_top_or_autoplay | 6 | 2 | 4 | partial | text/static tags + next state | 羽化(CARD.METAMORPHOSIS)、乱战(CARD.MAYHEM)、狠揍(CARD.BEAT_DOWN)、横祸(CARD.CATASTROPHE)、破灭(CARD.HAVOC)、倾泻(CARD.CASCADE) |
| discard_pile_dependency | 5 | 3 | 2 | partial | pile binding tokens + text/static tags | 涅奥之怒(CARD.NEOWS_FURY)、狠揍(CARD.BEAT_DOWN)、愤怒(CARD.ANGER)、头槌(CARD.HEADBUTT)、好勇斗狠(CARD.AGGRESSION) |
| potion_or_gold_gain | 5 | 0 | 5 | covered | static semantic tags/signals + reward/potion profile | 债务(CARD.DEBT)、藏宝图(CARD.SPOILS_MAP)、金斧(CARD.GOLD_AXE)、贪婪之手(CARD.HAND_OF_GREED)、炼制药水(CARD.ALCHEMIZE) |
| next_card_modifier | 4 | 2 | 2 | partial | internal power/modifier ids + runtime modifiers | 未掘宝石(CARD.HIDDEN_GEM)、双打组合(CARD.TAG_TEAM)、连环拳(CARD.ONE_TWO_PUNCH)、无情猛攻(CARD.UNRELENTING) |
| copy_card | 3 | 3 | 0 | partial | internal source ids/static tags | 愤怒(CARD.ANGER)、双持(CARD.DUAL_WIELD)、杂耍(CARD.JUGGLING) |
| x_cost | 3 | 2 | 1 | covered | Bridge resolved x_cost_value + zero-energy X metrics | 连射(CARD.VOLLEY)、倾泻(CARD.CASCADE)、旋风斩(CARD.WHIRLWIND) |
| heal_or_max_hp | 2 | 1 | 1 | covered | effect_preview/static semantic_signals | 至亮之焰(CARD.BRIGHTEST_FLAME)、狂宴(CARD.FEED) |
| remove_card | 2 | 0 | 2 | partial | text/static tags | 愧疚(CARD.GUILTY)、弃用卡牌(CARD.DEPRECATED_CARD) |
| transform_card | 2 | 1 | 1 | partial | semantic tags/internal Transform command + hand_mutation local | 熵(CARD.ENTROPY)、原始力量(CARD.PRIMAL_FORCE) |
| text_only_mechanism_warning | 1 | 0 | 1 | partial | audit fallback only; not a training contract | 毒素(CARD.TOXIC) |

## 高风险 / 需要重点复核的卡

只列前 80 张；全量见 CSV。

| 颜色 | ID | 名称 | 稀有度 | 类型 | 费用 | 状态 | gap 类别 | partial 类别 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| colorless | CARD.APOTHEOSIS | 神化 | Ancient | Skill | 2 | gap | effect_profile_granularity_gap | cost_modify, upgrade_card |
| colorless | CARD.APPARITION | 灵体 | Ancient | Skill | 1 | gap | effect_profile_granularity_gap | card_modifier_or_enchantment, global_rule_power |
| colorless | CARD.NEOWS_FURY | 涅奥之怒 | Ancient | Attack | 1 | gap | effect_profile_granularity_gap | add_or_generate_card, discard_pile_dependency, pile_fetch_reorder, random_or_choose, whole_hand_state |
| colorless | CARD.WISH | 许愿 | Ancient | Skill | 0 | gap | effect_profile_granularity_gap | add_or_generate_card, card_modifier_or_enchantment, exhaust_other_or_hand, pile_fetch_reorder, whole_hand_state |
| colorless | CARD.ASCENDERS_BANE | 进阶之灾 | Curse | Curse | None | gap | effect_profile_granularity_gap, hard_rule_constraint_gap, status_or_curse_penalty_gap | card_modifier_or_enchantment |
| colorless | CARD.BAD_LUCK | 霉运 | Curse | Curse | None | gap | effect_profile_granularity_gap, hard_rule_constraint_gap, status_or_curse_penalty_gap | card_modifier_or_enchantment, turn_or_delayed_rule |
| colorless | CARD.CLUMSY | 笨拙 | Curse | Curse | None | gap | effect_profile_granularity_gap, hard_rule_constraint_gap, status_or_curse_penalty_gap | card_modifier_or_enchantment |
| colorless | CARD.CURSE_OF_THE_BELL | 铃铛的诅咒 | Curse | Curse | None | gap | effect_profile_granularity_gap, hard_rule_constraint_gap, status_or_curse_penalty_gap | card_modifier_or_enchantment |
| colorless | CARD.DEBT | 债务 | Curse | Curse | None | gap | hard_rule_constraint_gap, status_or_curse_penalty_gap | turn_or_delayed_rule |
| colorless | CARD.DECAY | 腐朽 | Curse | Curse | None | gap | hard_rule_constraint_gap, status_or_curse_penalty_gap | turn_or_delayed_rule |
| colorless | CARD.DOUBT | 疑虑 | Curse | Curse | None | gap | hard_rule_constraint_gap, status_or_curse_penalty_gap | global_rule_power, turn_or_delayed_rule |
| colorless | CARD.ENTHRALLED | 执迷 | Curse | Curse | 2 | gap | effect_profile_granularity_gap, hard_rule_constraint_gap, status_or_curse_penalty_gap | card_modifier_or_enchantment |
| colorless | CARD.FOLLY | 愚行 | Curse | Curse | None | gap | effect_profile_granularity_gap, hard_rule_constraint_gap, status_or_curse_penalty_gap | card_modifier_or_enchantment |
| colorless | CARD.GREED | 贪婪 | Curse | Curse | None | gap | effect_profile_granularity_gap, hard_rule_constraint_gap, status_or_curse_penalty_gap | card_modifier_or_enchantment |
| colorless | CARD.GUILTY | 愧疚 | Curse | Curse | None | gap | hard_rule_constraint_gap, status_or_curse_penalty_gap | remove_card |
| colorless | CARD.INJURY | 受伤 | Curse | Curse | None | gap | hard_rule_constraint_gap, status_or_curse_penalty_gap |  |
| colorless | CARD.NORMALITY | 凡庸 | Curse | Curse | None | gap | hard_rule_constraint_gap, status_or_curse_penalty_gap | turn_or_delayed_rule |
| colorless | CARD.POOR_SLEEP | 睡眠不佳 | Curse | Curse | None | gap | hard_rule_constraint_gap, status_or_curse_penalty_gap |  |
| colorless | CARD.REGRET | 悔恨 | Curse | Curse | None | gap | hard_rule_constraint_gap, status_or_curse_penalty_gap | turn_or_delayed_rule |
| colorless | CARD.SHAME | 羞耻 | Curse | Curse | None | gap | hard_rule_constraint_gap, status_or_curse_penalty_gap | global_rule_power, turn_or_delayed_rule |
| colorless | CARD.SPORE_MIND | 孢子心灵 | Curse | Curse | 1 | gap | status_or_curse_penalty_gap |  |
| colorless | CARD.WRITHE | 苦恼 | Curse | Curse | None | gap | hard_rule_constraint_gap, status_or_curse_penalty_gap |  |
| colorless | CARD.ENLIGHTENMENT | 开悟 | Event | Skill | 0 | gap | effect_profile_granularity_gap | cost_modify, exhaust_other_or_hand, whole_hand_state |
| colorless | CARD.MAD_SCIENCE | 疯狂科学 | Event | None | 1 | gap | effect_profile_granularity_gap | card_modifier_or_enchantment, global_rule_power, pile_fetch_reorder, random_or_choose |
| colorless | CARD.METAMORPHOSIS | 羽化 | Event | Skill | 2 | gap | effect_profile_granularity_gap | add_or_generate_card, cost_modify, exhaust_other_or_hand, pile_fetch_reorder, play_top_or_autoplay, random_or_choose |
| colorless | CARD.BYRDONIS_EGG | 多尼斯异鸟蛋 | Quest | Quest | None | gap | hard_rule_constraint_gap |  |
| colorless | CARD.LANTERN_KEY | 灯火钥匙 | Quest | Quest | None | gap | hard_rule_constraint_gap |  |
| colorless | CARD.SPOILS_MAP | 藏宝图 | Quest | Quest | None | gap | hard_rule_constraint_gap |  |
| colorless | CARD.ALCHEMIZE | 炼制药水 | Rare | Skill | 1 | gap | effect_profile_granularity_gap | cost_modify, exhaust_other_or_hand, random_or_choose |
| colorless | CARD.ANOINTED | 天选 | Rare | Skill | 1 | gap | effect_profile_granularity_gap | add_or_generate_card, card_modifier_or_enchantment, exhaust_other_or_hand, pile_fetch_reorder, whole_hand_state |
| colorless | CARD.BEACON_OF_HOPE | 希望灯塔 | Rare | Power | 1 | gap | effect_profile_granularity_gap | card_modifier_or_enchantment, global_rule_power, turn_or_delayed_rule |
| colorless | CARD.BEAT_DOWN | 狠揍 | Rare | Skill | 3 | gap | effect_profile_granularity_gap | discard_pile_dependency, pile_fetch_reorder, play_top_or_autoplay, random_or_choose |
| colorless | CARD.BOLAS | 流星锤 | Rare | Attack | 0 | gap | effect_profile_granularity_gap | pile_fetch_reorder, turn_or_delayed_rule |
| colorless | CARD.CALAMITY | 灾祸 | Rare | Power | 3 | gap | effect_profile_granularity_gap | add_or_generate_card, cost_modify, global_rule_power, random_or_choose, turn_or_delayed_rule |
| colorless | CARD.ENTROPY | 熵 | Rare | Power | 1 | gap | effect_profile_granularity_gap | card_modifier_or_enchantment, global_rule_power, transform_card, turn_or_delayed_rule |
| colorless | CARD.ETERNAL_ARMOR | 永恒铠甲 | Rare | Power | 3 | gap | effect_profile_granularity_gap | card_modifier_or_enchantment, global_rule_power |
| colorless | CARD.GOLD_AXE | 金斧 | Rare | Attack | 1 | gap | effect_profile_granularity_gap | card_modifier_or_enchantment |
| colorless | CARD.HIDDEN_GEM | 未掘宝石 | Rare | Skill | 1 | gap | effect_profile_granularity_gap | add_or_generate_card, card_modifier_or_enchantment, next_card_modifier, pile_fetch_reorder, random_or_choose |
| colorless | CARD.JACKPOT | 大奖 | Rare | Attack | 3 | gap | effect_profile_granularity_gap | add_or_generate_card, random_or_choose, upgrade_card, whole_hand_state |
| colorless | CARD.MAYHEM | 乱战 | Rare | Power | 2 | gap | effect_profile_granularity_gap | cost_modify, global_rule_power, pile_fetch_reorder, play_top_or_autoplay, turn_or_delayed_rule |
| colorless | CARD.NOSTALGIA | 怀旧 | Rare | Power | 1 | gap | effect_profile_granularity_gap | cost_modify, global_rule_power, pile_fetch_reorder |
| colorless | CARD.SALVO | 箭雨 | Rare | Attack | 1 | gap | effect_profile_granularity_gap | global_rule_power, turn_or_delayed_rule, whole_hand_state |
| colorless | CARD.SCRAWL | 潦草急就 | Rare | Skill | 1 | gap | effect_profile_granularity_gap | card_modifier_or_enchantment, exhaust_other_or_hand |
| colorless | CARD.SECRET_TECHNIQUE | 秘密技法 | Rare | Skill | 0 | gap | effect_profile_granularity_gap | add_or_generate_card, exhaust_other_or_hand, pile_fetch_reorder, random_or_choose, whole_hand_state |
| colorless | CARD.SECRET_WEAPON | 秘密武器 | Rare | Skill | 0 | gap | effect_profile_granularity_gap | add_or_generate_card, exhaust_other_or_hand, pile_fetch_reorder, random_or_choose, whole_hand_state |
| colorless | CARD.BECKON | 呼唤 | Status | Status | 1 | gap | status_or_curse_penalty_gap | turn_or_delayed_rule |
| colorless | CARD.BURN | 灼伤 | Status | Status | None | gap | hard_rule_constraint_gap, status_or_curse_penalty_gap | turn_or_delayed_rule |
| colorless | CARD.DAZED | 晕眩 | Status | Status | None | gap | effect_profile_granularity_gap, hard_rule_constraint_gap, status_or_curse_penalty_gap | card_modifier_or_enchantment |
| colorless | CARD.DEBRIS | 碎屑 | Status | Status | 1 | gap | status_or_curse_penalty_gap |  |
| colorless | CARD.DEPRECATED_CARD | 弃用卡牌 | Status | Status | 0 | gap | effect_profile_granularity_gap, status_or_curse_penalty_gap | exhaust_other_or_hand, remove_card |
| colorless | CARD.DISINTEGRATION | 瓦解 | Status | Status | None | gap | status_or_curse_penalty_gap | global_rule_power, turn_or_delayed_rule |
| colorless | CARD.FRANTIC_ESCAPE | 狂乱逃离 | Status | Status | 1 | gap | status_or_curse_penalty_gap | global_rule_power |
| colorless | CARD.INFECTION | 感染 | Status | Status | None | gap | hard_rule_constraint_gap, status_or_curse_penalty_gap | turn_or_delayed_rule |
| colorless | CARD.MIND_ROT | 心灵腐化 | Status | Status | None | gap | status_or_curse_penalty_gap | global_rule_power |
| colorless | CARD.SLIMED | 黏液 | Status | Status | 1 | gap | status_or_curse_penalty_gap |  |
| colorless | CARD.SLOTH | 懒惰 | Status | Status | None | gap | status_or_curse_penalty_gap | global_rule_power |
| colorless | CARD.SOOT | 煤灰 | Status | Status | None | gap | hard_rule_constraint_gap, status_or_curse_penalty_gap |  |
| colorless | CARD.TOXIC | 毒素 | Status | Status | 1 | gap | status_or_curse_penalty_gap | text_only_mechanism_warning, turn_or_delayed_rule |
| colorless | CARD.VOID | 虚空 | Status | Status | None | gap | effect_profile_granularity_gap, hard_rule_constraint_gap, status_or_curse_penalty_gap | card_modifier_or_enchantment, turn_or_delayed_rule |
| colorless | CARD.WASTE_AWAY | 衰朽 | Status | Status | None | gap | status_or_curse_penalty_gap | global_rule_power |
| colorless | CARD.WOUND | 伤口 | Status | Status | None | gap | hard_rule_constraint_gap, status_or_curse_penalty_gap |  |
| colorless | CARD.SHIV | 小刀 | Token | Attack | 0 | gap | effect_profile_granularity_gap | add_or_generate_card |
| colorless | CARD.SOUL | 灵魂 | Token | Skill | 0 | gap | effect_profile_granularity_gap | add_or_generate_card |
| colorless | CARD.SOVEREIGN_BLADE | 君王之剑 | Token | Attack | 2 | gap | effect_profile_granularity_gap | cost_modify, exhaust_pile_dependency |
| colorless | CARD.SWEEPING_GAZE | 扫荡凝视 | Token | Attack | 0 | gap | effect_profile_granularity_gap | card_modifier_or_enchantment, exhaust_other_or_hand, random_or_choose |
| colorless | CARD.AUTOMATION | 自动化 | Uncommon | Power | 1 | gap | effect_profile_granularity_gap | cost_modify, global_rule_power |
| colorless | CARD.CATASTROPHE | 横祸 | Uncommon | Skill | 2 | gap | effect_profile_granularity_gap | pile_fetch_reorder, play_top_or_autoplay, random_or_choose |
| colorless | CARD.DISCOVERY | 发现 | Uncommon | Skill | 1 | gap | effect_profile_granularity_gap | add_or_generate_card, cost_modify, exhaust_other_or_hand, random_or_choose, turn_or_delayed_rule, whole_hand_state |
| colorless | CARD.DRAMATIC_ENTRANCE | 闪亮登场 | Uncommon | Attack | 0 | gap | effect_profile_granularity_gap | exhaust_other_or_hand |
| colorless | CARD.EQUILIBRIUM | 均衡 | Uncommon | Skill | 2 | gap | effect_profile_granularity_gap | global_rule_power, turn_or_delayed_rule, whole_hand_state |
| colorless | CARD.HUDDLE_UP | 抱团 | Uncommon | Skill | 1 | gap | effect_profile_granularity_gap | exhaust_other_or_hand |
| colorless | CARD.JACK_OF_ALL_TRADES | 花样百出 | Uncommon | Skill | 0 | gap | effect_profile_granularity_gap | add_or_generate_card, exhaust_other_or_hand, random_or_choose, whole_hand_state |
| colorless | CARD.MIND_BLAST | 心灵震慑 | Uncommon | Attack | 1 | gap | effect_profile_granularity_gap | cost_modify, pile_fetch_reorder |
| colorless | CARD.PURITY | 净化 | Uncommon | Skill | 0 | gap | effect_profile_granularity_gap | exhaust_other_or_hand, random_or_choose, whole_hand_state |
| colorless | CARD.SEEKER_STRIKE | 探寻打击 | Uncommon | Attack | 1 | gap | effect_profile_granularity_gap | add_or_generate_card, pile_fetch_reorder, random_or_choose, whole_hand_state |
| colorless | CARD.SHOCKWAVE | 震荡波 | Uncommon | Skill | 2 | gap | effect_profile_granularity_gap | exhaust_other_or_hand, global_rule_power |
| colorless | CARD.SPLASH | 飞溅 | Uncommon | Skill | 1 | gap | effect_profile_granularity_gap | add_or_generate_card, cost_modify, random_or_choose, turn_or_delayed_rule, upgrade_card, whole_hand_state |
| colorless | CARD.STRATAGEM | 计策 | Uncommon | Power | 1 | gap | effect_profile_granularity_gap | add_or_generate_card, cost_modify, global_rule_power, pile_fetch_reorder, random_or_choose, turn_or_delayed_rule, whole_hand_state |
| colorless | CARD.TAG_TEAM | 双打组合 | Uncommon | Attack | 2 | gap | effect_profile_granularity_gap | global_rule_power, next_card_modifier |
| colorless | CARD.THINKING_AHEAD | 深谋远虑 | Uncommon | Skill | 0 | gap | effect_profile_granularity_gap | exhaust_other_or_hand, pile_fetch_reorder, random_or_choose |

## 对当前模型结构的判断

### 已经能覆盖得比较好的部分

- **基础数值战斗效果**：伤害、格挡、抽牌、治疗、弱/易伤/毒、力量/敏捷/无实体等，来自 `BridgeGameApi.BuildCardPayload().effect_preview`、静态 `semanticSignals`、`observation_v3` 的 action/source profile。
- **X 费动态**：Bridge 暴露 `costs_x` 与 `effect_preview.x_cost_value/x_cost_semantics`，obs 与训练侧已有 0 能量 X 费指标；理论上不会再把 X 费固定理解成初始 3 费。
- **自身消耗 / 虚无 / 保留 / 运行时附魔**：Bridge 暴露 `keywords`、`card_flow`、`afflictions/enchantments`、`modifier_summary`，obs 也聚合了 `adds_exhaust/adds_retain/adds_ethereal/replay/cost_randomizes_on_draw/sets_cost_zero` 等 modifier。
- **牌堆可见性**：手牌、抽牌堆、弃牌堆、消耗牌堆已经有 token/binding/density 信息，注意力有机会把它们连接起来。

### 仍然不够硬的部分

1. **内部 ID 已经存在，但粒度还不够**
   `artifact://dependencies/sts2-ai/.../cards.jsonl` 与 C# 源码能提供 `commands`、`powers`、`PileType.Hand/Draw/Discard/Exhaust`、`CardSelectCmd.FromHand*`、`CardCmd.Upgrade/Exhaust/Transform`、`CardPileCmd.AddGeneratedCardToCombat`、`EnergyCost.Set*`、`CreateClone`、`Replay`、`AddKeyword/AddEnchantment/AddAffliction` 等内部事实。这比文本正则可靠得多。当前缺口不是“识别不到升级/消耗/变化这些词”，而是还没把这些内部调用编译成可训练的 `card_effect_profile.operations`。

2. **action -> pile transition 还不够结构化**
   头槌、破灭、倾泻、秘密武器/技法、探寻打击、战鼓、好勇斗狠、怀旧等会读/写抽牌堆或弃牌堆。现在模型可以通过 pile tokens 和 source facts 知道访问了哪些 pile，但缺少统一的 `source_zone/destination_zone/topdeck_target/play_top_count/fetch_filter`。

3. **消耗牌堆依赖需要显式条件特征**
   灰烬打击、契约终结、被遗忘的仪式、邪眼、黑暗之拥、腐化、恶魔之焰、添柴、重振精神等都要求模型理解“当前消耗堆数量 / 本回合是否消耗过 / 消耗后触发”。现在有消耗堆 token 和 `PileType.Exhaust` 访问事实，但每张牌的条件依赖还没有变成 typed feature。

4. **未来规则 / 下一张牌修饰需要 temporal rule token**
   腐化、无情猛攻、连环拳、怀旧、神气制胜、自动化/地狱狂徒等会改变后续出牌规则。search-free planner 要可靠，需要把“下一张攻击免费/重放/技能 0 费并消耗/每回合第一张置顶”等规则从文本提升为 rule token。

5. **Status/Curse 硬约束缺 typed profile**
   虚空、灼伤、遗憾、腐朽、普通、懒惰、执迷等不是普通收益牌；有抽到触发、回合末触发、出牌数限制、必须优先打出等硬规则。法律动作 mask 会处理“能不能打”，但策略层需要提前知道“为什么必须处理/为什么不能拖”。

## 建议的目标态补齐顺序

1. **生成 `card_effect_profile.operations`**：从 `cards.jsonl` + C# source facts 编译 typed operations。每个 operation 至少包含 `op/source_zone/destination_zone/scope/selection/count/min_count/max_count/target_filter/duration/modifier/power_id/result_card/created_card/upgraded_override/per_card_scaling`。
2. **Bridge `BuildCardPayload` 暴露 profile**：和 potion `effect_profile` 一样，把卡牌内部 profile 放进 runtime payload；`EnvCompact` / `observation_v3` 只保留压缩后的关键 operation token。
3. **`hand_mutation.py` 主路径改读 operations**：`upgrade_card/exhaust_card/transform_card/copy_card/modify_cost/move_card/add_modifier/add_keyword/set_replay` 等先读 typed op；文本只作为 `text_only_mechanism_warning` 与离线审计 fallback。
4. **把 pile transition / future rule 做成 token**：例如 `FETCH_FROM_DRAW_SKILL`、`TOPDECK_FROM_HAND`、`EXHAUST_HAND_ALL_NON_ATTACK`、`COPY_HAND_ATTACK_OR_POWER`、`NEXT_ATTACK_COST_ZERO`、`SKILL_COST_ZERO_AND_EXHAUST_ON_PLAY`。
5. **给 resource/exhaust deferability 加目标监督**：继续保留“合法但不该打”的策略空间，特别是回费牌没有后续动作、一次性消耗牌当前收益低、保留/虚无/end-turn 去向变化等场景。
6. **Status/Curse/硬约束单独做 aux head**：预测本回合/下回合由状态牌导致的 HP/energy/play-limit 风险，避免只从 reward 后验学习。
