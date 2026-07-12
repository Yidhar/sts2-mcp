# Colorless base cards

| ID | 名称 | 稀有度 | 类型 | 费用 | 覆盖 | 机制分类 | 基础描述 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CARD.MAUL | 撕咬 | Ancient | Attack | 1 | covered | damage, multi_hit_or_aoe | 造成5点伤害两次。 / 在这场战斗中，将所有“撕咬”牌的伤害增加1。 |
| CARD.NEOWS_FURY | 涅奥之怒 | Ancient | Attack | 1 | gap | add_or_generate_card, damage, discard_pile_dependency, draw, effect_profile_granularity_gap, exhaust_self, pile_fetch_reorder, random_or_choose, whole_hand_state | 造成10点伤害。 / 将你弃牌堆中的2张随机牌放入你的手牌。 / 消耗。 |
| CARD.WHISTLE | 吹哨 | Ancient | Attack | 3 | covered | damage, exhaust_self | 造成33点伤害。 / 击晕该敌人。 / 消耗。 |
| CARD.APOTHEOSIS | 神化 | Ancient | Skill | 2 | gap | cost_modify, effect_profile_granularity_gap, exhaust_self, upgrade_card | 固有。 / 升级你的全部卡牌。 / 消耗。 |
| CARD.APPARITION | 灵体 | Ancient | Skill | 1 | gap | buff_stats, card_modifier_or_enchantment, effect_profile_granularity_gap, ethereal, exhaust_self, global_rule_power | 虚无。 / 获得1层无实体。 / 消耗。 |
| CARD.BRIGHTEST_FLAME | 至亮之焰 | Ancient | Skill | 0 | covered | draw, energy_gain, heal_or_max_hp, hp_or_self_cost | 获得2点能量。 / 抽2张牌。 / 失去1点最大生命。 |
| CARD.RELAX | 放松 | Ancient | Skill | 3 | partial | block, draw, energy_gain, exhaust_self, global_rule_power, turn_or_delayed_rule | 获得15点格挡。 / 下个回合，抽2张牌并获得2点能量。 / 消耗。 |
| CARD.WISH | 许愿 | Ancient | Skill | 0 | gap | add_or_generate_card, card_modifier_or_enchantment, draw, effect_profile_granularity_gap, exhaust_other_or_hand, exhaust_self, pile_fetch_reorder, retain, whole_hand_state | 将你抽牌堆中的一张牌放入你的手牌。 / 消耗。 |
| CARD.ASCENDERS_BANE | 进阶之灾 | Curse | Curse | None | gap | card_modifier_or_enchantment, effect_profile_granularity_gap, ethereal, hard_rule_constraint_gap, status_or_curse_penalty_gap | 不能被打出。 / 虚无。 / 永恒。 |
| CARD.BAD_LUCK | 霉运 | Curse | Curse | None | gap | card_modifier_or_enchantment, effect_profile_granularity_gap, hard_rule_constraint_gap, hp_or_self_cost, status_or_curse_penalty_gap, turn_or_delayed_rule | 不能被打出。 / 在你的回合结束时，如果这张牌在你的手牌中，则失去13点生命。 / 永恒。 |
| CARD.CLUMSY | 笨拙 | Curse | Curse | None | gap | card_modifier_or_enchantment, effect_profile_granularity_gap, ethereal, hard_rule_constraint_gap, status_or_curse_penalty_gap | 不能被打出。 / 虚无。 |
| CARD.CURSE_OF_THE_BELL | 铃铛的诅咒 | Curse | Curse | None | gap | card_modifier_or_enchantment, effect_profile_granularity_gap, hard_rule_constraint_gap, status_or_curse_penalty_gap | 不能被打出。 / 永恒。 |
| CARD.DEBT | 债务 | Curse | Curse | None | gap | hard_rule_constraint_gap, hp_or_self_cost, potion_or_gold_gain, status_or_curse_penalty_gap, turn_or_delayed_rule | 不能被打出。 / 在你的回合结束时，如果这张牌在你的手牌中，则失去10金币。 |
| CARD.DECAY | 腐朽 | Curse | Curse | None | gap | damage, hard_rule_constraint_gap, hp_or_self_cost, status_or_curse_penalty_gap, turn_or_delayed_rule | 不能被打出。 / 在你的回合结束时，如果这张牌在你的手牌中, 你受到2点伤害。 |
| CARD.DOUBT | 疑虑 | Curse | Curse | None | gap | debuff, global_rule_power, hard_rule_constraint_gap, status_or_curse_penalty_gap, turn_or_delayed_rule | 不能被打出。 / 在你的回合结束时，如果这张牌在你的手牌中，获得1层虚弱。 |
| CARD.ENTHRALLED | 执迷 | Curse | Curse | 2 | gap | card_modifier_or_enchantment, effect_profile_granularity_gap, hard_rule_constraint_gap, status_or_curse_penalty_gap | 如果这张牌在你的手牌中，你必须优先打出这张牌。 / 永恒。 |
| CARD.FOLLY | 愚行 | Curse | Curse | None | gap | card_modifier_or_enchantment, effect_profile_granularity_gap, ethereal, hard_rule_constraint_gap, status_or_curse_penalty_gap | 不能被打出。 / 固有。 / 虚无。 / 永恒。 |
| CARD.GREED | 贪婪 | Curse | Curse | None | gap | card_modifier_or_enchantment, effect_profile_granularity_gap, hard_rule_constraint_gap, status_or_curse_penalty_gap | 不能被打出。 / 永恒。 |
| CARD.GUILTY | 愧疚 | Curse | Curse | None | gap | hard_rule_constraint_gap, remove_card, status_or_curse_penalty_gap | 不能被打出。 / 在5场战斗后从你的牌组中移除。 |
| CARD.INJURY | 受伤 | Curse | Curse | None | gap | hard_rule_constraint_gap, status_or_curse_penalty_gap | 不能被打出。 |
| CARD.NORMALITY | 凡庸 | Curse | Curse | None | gap | hard_rule_constraint_gap, status_or_curse_penalty_gap, turn_or_delayed_rule | 不能被打出。 / 你在本回合不能打出超过3张牌。 |
| CARD.POOR_SLEEP | 睡眠不佳 | Curse | Curse | None | gap | hard_rule_constraint_gap, retain, status_or_curse_penalty_gap | 不能被打出。 / 保留。 |
| CARD.REGRET | 悔恨 | Curse | Curse | None | gap | hard_rule_constraint_gap, hp_or_self_cost, status_or_curse_penalty_gap, turn_or_delayed_rule | 不能被打出。 / 在你的回合结束时，如果这张牌在你的手牌中，失去相当于手牌数量的生命。 |
| CARD.SHAME | 羞耻 | Curse | Curse | None | gap | global_rule_power, hard_rule_constraint_gap, status_or_curse_penalty_gap, turn_or_delayed_rule | 不能被打出。 / 在你的回合结束时，如果这张牌在你的手牌中，则获得1层脆弱。 |
| CARD.SPORE_MIND | 孢子心灵 | Curse | Curse | 1 | gap | exhaust_self, status_or_curse_penalty_gap | 消耗。 |
| CARD.WRITHE | 苦恼 | Curse | Curse | None | gap | hard_rule_constraint_gap, status_or_curse_penalty_gap | 不能被打出。 / 固有。 |
| CARD.BYRD_SWOOP | 异鸟扑击 | Event | Attack | 0 | covered | damage | 造成14点伤害。 |
| CARD.EXTERMINATE | 杀灭 | Event | Attack | 1 | covered | damage, multi_hit_or_aoe | 对所有敌人造成3点伤害4次。 |
| CARD.PECK | 啄击 | Event | Attack | 1 | covered | damage, multi_hit_or_aoe | 造成2点伤害3次。 |
| CARD.SQUASH | 压扁 | Event | Attack | 1 | partial | damage, debuff, global_rule_power | 造成10点伤害。 / 给予2层易伤。 |
| CARD.MAD_SCIENCE | 疯狂科学 | Event | None | 1 | gap | block, card_modifier_or_enchantment, damage, debuff, draw, effect_profile_granularity_gap, global_rule_power, pile_fetch_reorder, random_or_choose | {CardType:choose(Attack\|Skill\|Power):造成{Damage:diff()}点伤害{Violence:{ViolenceHits:diff()}次\|}。\|获得{Block:diff()}点格挡。\|}能量 |
| CARD.ENLIGHTENMENT | 开悟 | Event | Skill | 0 | gap | cost_modify, effect_profile_granularity_gap, exhaust_other_or_hand, exhaust_self, whole_hand_state | 在这个回合，你当前手牌中所有牌的耗能降低至1。 / 消耗。 |
| CARD.FEEDING_FRENZY | 疯狂进食 | Event | Skill | 0 | partial | buff_stats, global_rule_power, turn_or_delayed_rule | 在本回合内获得5点力量。 |
| CARD.METAMORPHOSIS | 羽化 | Event | Skill | 2 | gap | add_or_generate_card, cost_modify, draw, effect_profile_granularity_gap, exhaust_other_or_hand, exhaust_self, pile_fetch_reorder, play_top_or_autoplay, random_or_choose | 在你的抽牌堆中加入3张随机攻击牌。它们在本场战斗中可以被免费打出。 / 消耗。 |
| CARD.TORIC_TOUGHNESS | 坚韧之环 | Event | Skill | 2 | partial | block, global_rule_power, turn_or_delayed_rule | 获得5点格挡。 / 在接下来的2个回合开始时，获得5点格挡。 |
| CARD.BYRDONIS_EGG | 多尼斯异鸟蛋 | Quest | Quest | None | gap | hard_rule_constraint_gap | 不能被打出。 / 能在休息处被孵化。 |
| CARD.LANTERN_KEY | 灯火钥匙 | Quest | Quest | None | gap | hard_rule_constraint_gap | 不能被打出。 / 在下一阶段解锁一个特殊事件。 |
| CARD.SPOILS_MAP | 藏宝图 | Quest | Quest | None | gap | hard_rule_constraint_gap, potion_or_gold_gain | 不能被打出。 / 在下一阶段的地图上，标记一个有600额外金币的地点。 |
| CARD.BOLAS | 流星锤 | Rare | Attack | 0 | gap | damage, effect_profile_granularity_gap, pile_fetch_reorder, turn_or_delayed_rule | 造成3点伤害。 / 在你的下个回合开始时，将此卡返回你的手牌。 |
| CARD.GOLD_AXE | 金斧 | Rare | Attack | 1 | gap | card_modifier_or_enchantment, damage, effect_profile_granularity_gap, potion_or_gold_gain, retain | 造成本场战斗中所打出牌数的伤害。 |
| CARD.HAND_OF_GREED | 贪婪之手 | Rare | Attack | 2 | covered | damage, potion_or_gold_gain | 造成20点伤害。 / 斩杀时，获得20金币。 |
| CARD.JACKPOT | 大奖 | Rare | Attack | 3 | gap | add_or_generate_card, damage, draw, effect_profile_granularity_gap, random_or_choose, upgrade_card, whole_hand_state | 造成25点伤害。 / 将3张随机0点能量的牌加入你的手牌。 |
| CARD.KNOCKDOWN | 击倒 | Rare | Attack | 3 | partial | damage, global_rule_power, hp_or_self_cost, turn_or_delayed_rule | 造成10点伤害。 / 该敌人在本回合受到的来自其他玩家的伤害变为两倍。 |
| CARD.REND | 撕碎 | Rare | Attack | 2 | covered | damage | 造成15点伤害。 / 该名敌人身上每有一种负面效果，就额外造成5点伤害。 |
| CARD.SALVO | 箭雨 | Rare | Attack | 1 | gap | damage, effect_profile_granularity_gap, global_rule_power, retain, turn_or_delayed_rule, whole_hand_state | 造成12点伤害。 / 在本回合保留你的手牌。 |
| CARD.BEACON_OF_HOPE | 希望灯塔 | Rare | Power | 1 | gap | block, card_modifier_or_enchantment, effect_profile_granularity_gap, global_rule_power, turn_or_delayed_rule | 每当你在你的回合获得格挡时，其他玩家获得相应一半的格挡。 |
| CARD.CALAMITY | 灾祸 | Rare | Power | 3 | gap | add_or_generate_card, cost_modify, effect_profile_granularity_gap, global_rule_power, random_or_choose, turn_or_delayed_rule | 每当你打出一张攻击牌时，将一张随机攻击牌添加到你的手牌。 |
| CARD.ENTROPY | 熵 | Rare | Power | 1 | gap | card_modifier_or_enchantment, draw, effect_profile_granularity_gap, global_rule_power, transform_card, turn_or_delayed_rule | 在你的回合开始时，变化你手牌中的1张牌。 |
| CARD.ETERNAL_ARMOR | 永恒铠甲 | Rare | Power | 3 | gap | block, buff_stats, card_modifier_or_enchantment, effect_profile_granularity_gap, global_rule_power | 获得9层覆甲。 |
| CARD.MAYHEM | 乱战 | Rare | Power | 2 | gap | cost_modify, draw, effect_profile_granularity_gap, global_rule_power, pile_fetch_reorder, play_top_or_autoplay, turn_or_delayed_rule | 在你的回合开始时，打出你抽牌堆顶部的牌。 |
| CARD.NOSTALGIA | 怀旧 | Rare | Power | 1 | gap | cost_modify, draw, effect_profile_granularity_gap, global_rule_power, pile_fetch_reorder | 将你每回合打出第一张攻击或技能牌，置于你的抽牌堆顶端。 |
| CARD.ROLLING_BOULDER | 滚石 | Rare | Power | 3 | partial | damage, global_rule_power, multi_hit_or_aoe, turn_or_delayed_rule | 在你的回合开始时，对所有敌人造成5点伤害，然后将该伤害增加5点。 |
| CARD.ALCHEMIZE | 炼制药水 | Rare | Skill | 1 | gap | cost_modify, effect_profile_granularity_gap, exhaust_other_or_hand, exhaust_self, potion_or_gold_gain, random_or_choose | 获得一瓶随机药水。 / 消耗。 |
| CARD.ANOINTED | 天选 | Rare | Skill | 1 | gap | add_or_generate_card, card_modifier_or_enchantment, draw, effect_profile_granularity_gap, exhaust_other_or_hand, exhaust_self, pile_fetch_reorder, retain, whole_hand_state | 将你抽牌堆中的所有稀有牌放入你的手牌。 / 消耗。 |
| CARD.BEAT_DOWN | 狠揍 | Rare | Skill | 3 | gap | discard_pile_dependency, draw, effect_profile_granularity_gap, multi_hit_or_aoe, pile_fetch_reorder, play_top_or_autoplay, random_or_choose | 打出你弃牌堆中的3张随机攻击牌。 |
| CARD.HIDDEN_GEM | 未掘宝石 | Rare | Skill | 1 | gap | add_or_generate_card, card_modifier_or_enchantment, draw, effect_profile_granularity_gap, next_card_modifier, pile_fetch_reorder, random_or_choose | 你抽牌堆中的一张没有重放的随机牌获得2层重放。 |
| CARD.MASTER_OF_STRATEGY | 战略大师 | Rare | Skill | 0 | covered | draw, exhaust_self | 抽3张牌。 / 消耗。 |
| CARD.MIMIC | 拟态 | Rare | Skill | 1 | covered | block, exhaust_self | 获得等同于另一位玩家格挡值的格挡。 / 消耗。 |
| CARD.RALLY | 集结 | Rare | Skill | 2 | covered | block | 所有玩家获得12点格挡。 |
| CARD.SCRAWL | 潦草急就 | Rare | Skill | 1 | gap | card_modifier_or_enchantment, draw, effect_profile_granularity_gap, exhaust_other_or_hand, exhaust_self, retain | 抽牌直到抽满手牌。 / 消耗。 |
| CARD.SECRET_TECHNIQUE | 秘密技法 | Rare | Skill | 0 | gap | add_or_generate_card, draw, effect_profile_granularity_gap, exhaust_other_or_hand, exhaust_self, pile_fetch_reorder, random_or_choose, whole_hand_state | 从抽牌堆中选择一张技能牌放入你的手牌。 / 消耗。 |
| CARD.SECRET_WEAPON | 秘密武器 | Rare | Skill | 0 | gap | add_or_generate_card, draw, effect_profile_granularity_gap, exhaust_other_or_hand, exhaust_self, pile_fetch_reorder, random_or_choose, whole_hand_state | 从抽牌堆中选择一张攻击牌放入你的手牌。 / 消耗。 |
| CARD.THE_GAMBIT | 孤注一掷 | Rare | Skill | 0 | partial | block, global_rule_power, hp_or_self_cost | 获得50点格挡。 / 如果你在本场战斗中受到未被格挡的伤害，则立刻死亡。 |
| CARD.BECKON | 呼唤 | Status | Status | 1 | gap | hp_or_self_cost, status_or_curse_penalty_gap, turn_or_delayed_rule | 在你的回合结束时，如果这张牌在你的手牌中， / 则失去6点生命。 |
| CARD.BURN | 灼伤 | Status | Status | None | gap | damage, hard_rule_constraint_gap, hp_or_self_cost, status_or_curse_penalty_gap, turn_or_delayed_rule | 不能被打出。 / 在你的回合结束时，如果这张牌在你的手牌中，你受到2点伤害。 |
| CARD.DAZED | 晕眩 | Status | Status | None | gap | card_modifier_or_enchantment, effect_profile_granularity_gap, ethereal, hard_rule_constraint_gap, status_or_curse_penalty_gap | 不能被打出。 / 虚无。 |
| CARD.DEBRIS | 碎屑 | Status | Status | 1 | gap | exhaust_self, status_or_curse_penalty_gap | 消耗。 |
| CARD.DEPRECATED_CARD | 弃用卡牌 | Status | Status | 0 | gap | draw, effect_profile_granularity_gap, exhaust_other_or_hand, exhaust_self, remove_card, status_or_curse_penalty_gap | 这张卡牌在近期的一次更新中被移除。 / 抽1张牌。 / 将这张牌从你的牌组中移除。 / 消耗。 |
| CARD.DISINTEGRATION | 瓦解 | Status | Status | None | gap | global_rule_power, hp_or_self_cost, status_or_curse_penalty_gap, turn_or_delayed_rule | 在你的回合结束时，受到6点伤害。 |
| CARD.FRANTIC_ESCAPE | 狂乱逃离 | Status | Status | 1 | gap | global_rule_power, status_or_curse_penalty_gap | 远离。 / 将沙坑的计数加1。 / 这张牌的耗能加1。 |
| CARD.INFECTION | 感染 | Status | Status | None | gap | damage, hard_rule_constraint_gap, hp_or_self_cost, status_or_curse_penalty_gap, turn_or_delayed_rule | 不能被打出。 / 在你的回合结束时，如果这张牌在你的手牌中，则受到3点伤害。 |
| CARD.MIND_ROT | 心灵腐化 | Status | Status | None | gap | draw, global_rule_power, status_or_curse_penalty_gap | 每回合少抽1张牌。 |
| CARD.SLIMED | 黏液 | Status | Status | 1 | gap | draw, exhaust_self, status_or_curse_penalty_gap | 抽1张牌。 / 消耗。 |
| CARD.SLOTH | 懒惰 | Status | Status | None | gap | global_rule_power, status_or_curse_penalty_gap | 你在每个回合不能打出超过3张牌。 |
| CARD.SOOT | 煤灰 | Status | Status | None | gap | hard_rule_constraint_gap, status_or_curse_penalty_gap | 不能被打出。 |
| CARD.TOXIC | 毒素 | Status | Status | 1 | gap | damage, exhaust_self, hp_or_self_cost, status_or_curse_penalty_gap, text_only_mechanism_warning, turn_or_delayed_rule | 在你的回合结束时，这张牌在你的手牌中，则受到5点伤害。 / 消耗。 |
| CARD.VOID | 虚空 | Status | Status | None | gap | card_modifier_or_enchantment, draw, effect_profile_granularity_gap, ethereal, hard_rule_constraint_gap, hp_or_self_cost, status_or_curse_penalty_gap, turn_or_delayed_rule | 不能被打出。 / 虚无。 / 每当你抽到这张牌时，失去1点能量。 |
| CARD.WASTE_AWAY | 衰朽 | Status | Status | None | gap | global_rule_power, hp_or_self_cost, status_or_curse_penalty_gap | 每回合失去1点1点能量。 |
| CARD.WOUND | 伤口 | Status | Status | None | gap | hard_rule_constraint_gap, status_or_curse_penalty_gap | 不能被打出。 |
| CARD.GIANT_ROCK | 巨石 | Token | Attack | 1 | covered | damage | 造成16点伤害。 |
| CARD.MINION_DIVE_BOMB | 仆从俯冲 | Token | Attack | 0 | covered | damage, exhaust_self | 造成13点伤害。 / 消耗。 |
| CARD.MINION_STRIKE | 仆从打击 | Token | Attack | 0 | covered | damage, draw, exhaust_self | 造成6点伤害。 / 抽1张牌。 / 消耗。 |
| CARD.SHIV | 小刀 | Token | Attack | 0 | gap | add_or_generate_card, damage, effect_profile_granularity_gap, exhaust_self | 造成4点伤害。 / 消耗。 |
| CARD.SOVEREIGN_BLADE | 君王之剑 | Token | Attack | 2 | gap | cost_modify, damage, effect_profile_granularity_gap, exhaust_pile_dependency, retain | 保留。 / 造成10点伤害。 |
| CARD.SWEEPING_GAZE | 扫荡凝视 | Token | Attack | 0 | gap | card_modifier_or_enchantment, damage, effect_profile_granularity_gap, ethereal, exhaust_other_or_hand, exhaust_self, multi_hit_or_aoe, random_or_choose | 虚无。 / 奥斯提对随机一名敌人造成10点伤害。 / 消耗。 |
| CARD.FUEL | 燃料 | Token | Skill | 0 | covered | draw, energy_gain, exhaust_self | 获得1点能量。 / 抽1张牌。 / 消耗。 |
| CARD.LUMINESCE | 冷光 | Token | Skill | 0 | covered | energy_gain, exhaust_self, retain | 保留。 / 获得2点能量。 / 消耗。 |
| CARD.MINION_SACRIFICE | 仆从捐躯 | Token | Skill | 0 | covered | block, exhaust_self | 获得9点格挡。 / 消耗。 |
| CARD.SOUL | 灵魂 | Token | Skill | 0 | gap | add_or_generate_card, draw, effect_profile_granularity_gap, exhaust_self | 抽2张牌。 / 消耗。 |
| CARD.DRAMATIC_ENTRANCE | 闪亮登场 | Uncommon | Attack | 0 | gap | damage, effect_profile_granularity_gap, exhaust_other_or_hand, exhaust_self, multi_hit_or_aoe | 固有。 / 对所有敌人造成11点伤害。 / 消耗。 |
| CARD.FISTICUFFS | 拳斗 | Uncommon | Attack | 1 | covered | block, damage | 造成7点伤害。 / 获得等量于所造成伤害的格挡。 |
| CARD.FLASH_OF_STEEL | 亮剑 | Uncommon | Attack | 0 | covered | damage, draw | 造成5点伤害。 / 抽1张牌。 |
| CARD.GANG_UP | 群起攻之 | Uncommon | Attack | 1 | partial | damage, turn_or_delayed_rule | 造成5点伤害。 / 本回合其他玩家每攻击过一次该敌人，该牌造成的伤害就额外增加5点。 |
| CARD.MIND_BLAST | 心灵震慑 | Uncommon | Attack | 1 | gap | cost_modify, damage, draw, effect_profile_granularity_gap, pile_fetch_reorder | 固有。 / 造成你抽牌堆中剩余牌数的伤害。 |
| CARD.OMNISLICE | 万向斩 | Uncommon | Attack | 0 | covered | damage | 造成8点伤害。 / 对所有其他敌人造成等量的伤害。 |
| CARD.SEEKER_STRIKE | 探寻打击 | Uncommon | Attack | 1 | gap | add_or_generate_card, damage, draw, effect_profile_granularity_gap, pile_fetch_reorder, random_or_choose, whole_hand_state | 造成9点伤害。 / 从抽牌堆的随机3张牌中选择一张加入你的手牌。 |
| CARD.TAG_TEAM | 双打组合 | Uncommon | Attack | 2 | gap | damage, effect_profile_granularity_gap, global_rule_power, multi_hit_or_aoe, next_card_modifier | 造成11点伤害。 / 其他玩家的下一张攻击牌将在该名敌人身上额外生效一次。 |
| CARD.THRUMMING_HATCHET | 无休手斧 | Uncommon | Attack | 1 | gap | damage, effect_profile_granularity_gap, pile_fetch_reorder, turn_or_delayed_rule | 造成11点伤害。 / 在你的下个回合开始时，将此卡返回你的手牌。 |
| CARD.ULTIMATE_STRIKE | 究极打击 | Uncommon | Attack | 1 | covered | damage | 造成14点伤害。 |
| CARD.VOLLEY | 连射 | Uncommon | Attack | X | partial | damage, multi_hit_or_aoe, random_or_choose, x_cost | 随机对敌人造成10点伤害X次。 |
| CARD.AUTOMATION | 自动化 | Uncommon | Power | 1 | gap | cost_modify, draw, effect_profile_granularity_gap, energy_gain, global_rule_power | 你每抽10张牌，获得1点能量。 |
| CARD.FASTEN | 勒紧 | Uncommon | Power | 1 | partial | block, global_rule_power | 从“防御”牌中额外获得5点格挡。 |
| CARD.PANACHE | 神气制胜 | Uncommon | Power | 0 | partial | damage, global_rule_power, multi_hit_or_aoe, turn_or_delayed_rule | 每当你在一回合内打出五张牌时，对所有敌人造成10点伤害。 |
| CARD.PREP_TIME | 准备时间 | Uncommon | Power | 1 | partial | global_rule_power, turn_or_delayed_rule | 在你的回合开始时，获得4点活力。 |
| CARD.PROWESS | 非凡技艺 | Uncommon | Power | 1 | partial | buff_stats, global_rule_power | 获得1点力量。 / 获得1点敏捷。 |
| CARD.STRATAGEM | 计策 | Uncommon | Power | 1 | gap | add_or_generate_card, cost_modify, draw, effect_profile_granularity_gap, global_rule_power, pile_fetch_reorder, random_or_choose, turn_or_delayed_rule, whole_hand_state | 每当你的抽牌堆打乱洗牌时，选择一张牌放入你的手牌。 |
| CARD.BELIEVE_IN_YOU | 相信着你 | Uncommon | Skill | 0 | covered | energy_gain | 另一名玩家获得2点能量。 |
| CARD.CATASTROPHE | 横祸 | Uncommon | Skill | 2 | gap | draw, effect_profile_granularity_gap, pile_fetch_reorder, play_top_or_autoplay, random_or_choose | 从你的抽牌堆中随机打出2张牌。 |
| CARD.COORDINATE | 协同配合 | Uncommon | Skill | 1 | partial | buff_stats, global_rule_power, turn_or_delayed_rule | 在本回合给予其他玩家5点力量。 |
| CARD.DARK_SHACKLES | 黑暗镣铐 | Uncommon | Skill | 0 | partial | buff_stats, exhaust_self, global_rule_power, hp_or_self_cost, turn_or_delayed_rule | 使一名敌人在本回合失去9点力量。 / 消耗。 |
| CARD.DISCOVERY | 发现 | Uncommon | Skill | 1 | gap | add_or_generate_card, cost_modify, effect_profile_granularity_gap, exhaust_other_or_hand, exhaust_self, random_or_choose, turn_or_delayed_rule, whole_hand_state | 从3张随机牌中选择1张加入你的手牌。这张牌在本回合可以免费打出。 / 消耗。 |
| CARD.EQUILIBRIUM | 均衡 | Uncommon | Skill | 2 | gap | block, effect_profile_granularity_gap, global_rule_power, retain, turn_or_delayed_rule, whole_hand_state | 获得13点格挡。 / 在本回合保留你的手牌。 |
| CARD.FINESSE | 妙计 | Uncommon | Skill | 0 | covered | block, draw | 获得4点格挡。 / 抽1张牌。 |
| CARD.HUDDLE_UP | 抱团 | Uncommon | Skill | 1 | gap | draw, effect_profile_granularity_gap, exhaust_other_or_hand, exhaust_self | 所有玩家抽2张牌。 / 消耗。 |
| CARD.IMPATIENCE | 急躁 | Uncommon | Skill | 0 | covered | draw | 如果你的手牌中没有攻击牌，抽2张牌。 |
| CARD.INTERCEPT | 拦截 | Uncommon | Skill | 1 | partial | block, global_rule_power, turn_or_delayed_rule | 获得9点格挡。 / 将本回合所有要对另一名玩家发起的攻击转移到你的身上。 |
| CARD.JACK_OF_ALL_TRADES | 花样百出 | Uncommon | Skill | 0 | gap | add_or_generate_card, draw, effect_profile_granularity_gap, exhaust_other_or_hand, exhaust_self, random_or_choose, whole_hand_state | 将1张随机无色牌加入你的手牌。 / 消耗。 |
| CARD.LIFT | 托举 | Uncommon | Skill | 1 | covered | block | 给另一名玩家11点格挡。 |
| CARD.PANIC_BUTTON | 应急按钮 | Uncommon | Skill | 0 | partial | block, exhaust_self, global_rule_power | 获得30点格挡。 / 你在接下来的2回合内无法再从卡牌中获得格挡。 / 消耗。 |
| CARD.PRODUCTION | 生产制造 | Uncommon | Skill | 0 | covered | energy_gain, exhaust_self | 获得2点能量。 / 消耗。 |
| CARD.PROLONG | 延伸 | Uncommon | Skill | 0 | partial | block, exhaust_self, global_rule_power, turn_or_delayed_rule | 在下个回合获得等量于你当前格挡值的格挡。 / 消耗。 |
| CARD.PURITY | 净化 | Uncommon | Skill | 0 | gap | draw, effect_profile_granularity_gap, exhaust_other_or_hand, exhaust_self, random_or_choose, retain, whole_hand_state | 保留。 / 从手牌中选择最多3张牌消耗。 / 消耗。 |
| CARD.RESTLESSNESS | 心神不宁 | Uncommon | Skill | 0 | covered | draw, energy_gain, retain | 保留。 / 如果你的手牌为空，则抽2张牌并获得2点能量。 |
| CARD.SHOCKWAVE | 震荡波 | Uncommon | Skill | 2 | gap | debuff, effect_profile_granularity_gap, exhaust_other_or_hand, exhaust_self, global_rule_power, multi_hit_or_aoe | 给予所有敌人3层虚弱和易伤。 / 消耗。 |
| CARD.SPLASH | 飞溅 | Uncommon | Skill | 1 | gap | add_or_generate_card, cost_modify, effect_profile_granularity_gap, random_or_choose, turn_or_delayed_rule, upgrade_card, whole_hand_state | 从3张其他角色的攻击牌中选择1张加入你的手牌。这张牌在本回合免费打出。 |
| CARD.THE_BOMB | 炸弹 | Uncommon | Skill | 2 | partial | damage, global_rule_power, multi_hit_or_aoe, turn_or_delayed_rule | 在3回合结束后，对所有敌人造成40点伤害。 |
| CARD.THINKING_AHEAD | 深谋远虑 | Uncommon | Skill | 0 | gap | draw, effect_profile_granularity_gap, exhaust_other_or_hand, exhaust_self, pile_fetch_reorder, random_or_choose | 抽2张牌。 / 将手牌中的一张牌放到你的抽牌堆的顶端。 / 消耗。 |
| CARD.ULTIMATE_DEFEND | 究极防御 | Uncommon | Skill | 1 | covered | block | 获得11点格挡。 |
