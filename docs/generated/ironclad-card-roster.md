# Ironclad base cards

| ID | 名称 | 稀有度 | 类型 | 费用 | 覆盖 | 机制分类 | 基础描述 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CARD.BREAK | 破击 | Ancient | Attack | 1 | partial | damage, debuff, global_rule_power | 造成20点伤害。 / 给予5层易伤。 |
| CARD.CORRUPTION | 腐化 | Ancient | Power | 3 | gap | cost_modify, effect_profile_granularity_gap, exhaust_other_or_hand, exhaust_self, global_rule_power, turn_or_delayed_rule | 技能牌消耗变为0点能量。 / 每当你打出一张技能牌时，将其消耗。 |
| CARD.BASH | 痛击 | Basic | Attack | 2 | partial | damage, debuff, global_rule_power | 造成8点伤害。 / 给予2层易伤。 |
| CARD.STRIKE_IRONCLAD | 打击 | Basic | Attack | 1 | covered | damage | 造成6点伤害。 |
| CARD.DEFEND_IRONCLAD | 防御 | Basic | Skill | 1 | covered | block | 获得5点格挡。 |
| CARD.ANGER | 愤怒 | Common | Attack | 0 | gap | add_or_generate_card, copy_card, damage, discard_pile_dependency, effect_profile_granularity_gap, pile_fetch_reorder | 造成6点伤害。 / 将一张此牌的复制品加入你的弃牌堆。 |
| CARD.BODY_SLAM | 全身撞击 | Common | Attack | 1 | gap | block, cost_modify, damage, effect_profile_granularity_gap | 造成你当前格挡值的伤害。 |
| CARD.BREAKTHROUGH | 突破 | Common | Attack | 1 | covered | damage, hp_or_self_cost, multi_hit_or_aoe | 失去1点生命。 / 对所有敌人造成9点伤害。 |
| CARD.CINDER | 余烬 | Common | Attack | 2 | gap | damage, effect_profile_granularity_gap, exhaust_other_or_hand, random_or_choose, whole_hand_state | 造成18点伤害。 / 随机消耗1张牌。 |
| CARD.HEADBUTT | 头槌 | Common | Attack | 1 | gap | damage, discard_pile_dependency, draw, effect_profile_granularity_gap, pile_fetch_reorder | 造成9点伤害。 / 将你弃牌堆中的一张牌放到抽牌堆顶部。 |
| CARD.IRON_WAVE | 铁斩波 | Common | Attack | 1 | covered | block, damage | 获得5点格挡。 / 造成5点伤害。 |
| CARD.MOLTEN_FIST | 熔融之拳 | Common | Attack | 1 | partial | damage, debuff, exhaust_self, global_rule_power | 造成10点伤害。 / 将该敌人身上的易伤层数翻倍。 / 消耗。 |
| CARD.PERFECTED_STRIKE | 完美打击 | Common | Attack | 2 | covered | damage | 造成6点伤害。 / 你每有一张名字中含有“打击”的牌，伤害+2。 |
| CARD.POMMEL_STRIKE | 剑柄打击 | Common | Attack | 1 | covered | damage, draw | 造成9点伤害。 / 抽1张牌。 |
| CARD.SETUP_STRIKE | 预备打击 | Common | Attack | 1 | partial | buff_stats, damage, global_rule_power, turn_or_delayed_rule | 造成7点伤害。 / 在本回合内获得2点力量。 |
| CARD.SWORD_BOOMERANG | 飞剑回旋镖 | Common | Attack | 1 | partial | damage, multi_hit_or_aoe, random_or_choose | 随机对敌人造成3点伤害3次。 |
| CARD.THUNDERCLAP | 闪电霹雳 | Common | Attack | 1 | partial | damage, debuff, global_rule_power, multi_hit_or_aoe | 对所有敌人造成4点伤害，给予1层易伤。 |
| CARD.TWIN_STRIKE | 双重打击 | Common | Attack | 1 | covered | damage, multi_hit_or_aoe | 造成5点伤害两次。 |
| CARD.ARMAMENTS | 武装 | Common | Skill | 1 | gap | block, effect_profile_granularity_gap, random_or_choose, upgrade_card, whole_hand_state | 获得5点格挡。 / 升级你手牌中的一张牌。 |
| CARD.BLOODLETTING | 放血 | Common | Skill | 0 | covered | energy_gain, hp_or_self_cost | 失去3点生命。 / 获得2点能量。 |
| CARD.BLOOD_WALL | 血墙 | Common | Skill | 2 | covered | block, hp_or_self_cost | 失去2点生命。 / 获得16点格挡。 |
| CARD.HAVOC | 破灭 | Common | Skill | 1 | gap | cost_modify, draw, effect_profile_granularity_gap, exhaust_other_or_hand, exhaust_self, pile_fetch_reorder, play_top_or_autoplay | 打出抽牌堆顶部的牌并将其消耗。 |
| CARD.SHRUG_IT_OFF | 耸肩无视 | Common | Skill | 1 | covered | block, draw | 获得8点格挡。 / 抽1张牌。 |
| CARD.TREMBLE | 战栗 | Common | Skill | 1 | partial | debuff, exhaust_self, global_rule_power | 给予3层易伤。 / 消耗。 |
| CARD.TRUE_GRIT | 坚毅 | Common | Skill | 1 | gap | block, effect_profile_granularity_gap, exhaust_other_or_hand, random_or_choose, whole_hand_state | 获得7点格挡。 / 随机消耗1张牌。 |
| CARD.CLASH | 交锋 | Event | Attack | 0 | covered | damage | 只有在手牌中每一张牌都是攻击牌时才能被打出。 / 造成14点伤害。 |
| CARD.DUAL_WIELD | 双持 | Event | Skill | 1 | gap | add_or_generate_card, copy_card, draw, effect_profile_granularity_gap, random_or_choose, whole_hand_state | 选择一张攻击牌或能力牌。将一张此牌的复制品加入你的手牌。 |
| CARD.ENTRENCH | 巩固 | Event | Skill | 2 | gap | block, cost_modify, effect_profile_granularity_gap | 将你当前的格挡翻倍。 |
| CARD.CONFLAGRATION | 焚烧 | Rare | Attack | 1 | partial | damage, multi_hit_or_aoe, turn_or_delayed_rule | 对所有敌人造成8点伤害。 / 你在本回合中每打出过一张其他攻击牌，这张牌的伤害就提升2点。 |
| CARD.FEED | 狂宴 | Rare | Attack | 1 | covered | damage, exhaust_self, heal_or_max_hp, hp_or_self_cost | 造成10点伤害。 / 斩杀时，永久获得3点最大生命值。 / 消耗。 |
| CARD.FIEND_FIRE | 恶魔之焰 | Rare | Attack | 2 | gap | damage, effect_profile_granularity_gap, exhaust_other_or_hand, exhaust_self, whole_hand_state | 消耗所有手牌。 / 每张被消耗的牌造成7点伤害。 / 消耗。 |
| CARD.MANGLE | 凌虐 | Rare | Attack | 3 | partial | buff_stats, damage, global_rule_power, hp_or_self_cost, turn_or_delayed_rule | 造成15点伤害。 / 敌人在本回合失去10点力量。 |
| CARD.PACTS_END | 契约终结 | Rare | Attack | 0 | partial | damage, draw, exhaust_pile_dependency, multi_hit_or_aoe | 只有在你的消耗牌堆拥有大于等于3张牌的时候才能被打出。 / 对所有敌人造成17点伤害。 |
| CARD.TEAR_ASUNDER | 扯碎 | Rare | Attack | 2 | covered | damage, hp_or_self_cost | 造成5点伤害。 / 在本场战斗中，你每失去过一次生命值，这张牌就额外造成一次伤害。 |
| CARD.THRASH | 痛殴 | Rare | Attack | 1 | gap | damage, effect_profile_granularity_gap, exhaust_other_or_hand, multi_hit_or_aoe, random_or_choose, whole_hand_state | 造成4点伤害两次。 / 消耗你的手牌中随机一张攻击牌，并将它的伤害添加给这张牌。 |
| CARD.AGGRESSION | 好勇斗狠 | Rare | Power | 1 | gap | add_or_generate_card, card_modifier_or_enchantment, discard_pile_dependency, effect_profile_granularity_gap, global_rule_power, pile_fetch_reorder, random_or_choose, turn_or_delayed_rule, upgrade_card, whole_hand_state | 在你的回合开始时，将你弃牌堆的一张随机攻击牌放入你的手牌并将其升级。 |
| CARD.BARRICADE | 壁垒 | Rare | Power | 3 | gap | block, cost_modify, effect_profile_granularity_gap, global_rule_power, turn_or_delayed_rule | 格挡不再在你的回合开始时消失。 |
| CARD.CRIMSON_MANTLE | 绯红披风 | Rare | Power | 1 | partial | block, global_rule_power, hp_or_self_cost, turn_or_delayed_rule | 在你的回合开始时，失去1点生命并获得8点格挡。 |
| CARD.CRUELTY | 残酷 | Rare | Power | 1 | partial | debuff, global_rule_power, hp_or_self_cost | 有易伤状态的敌人额外受到25%的伤害。 |
| CARD.DARK_EMBRACE | 黑暗之拥 | Rare | Power | 2 | gap | cost_modify, draw, effect_profile_granularity_gap, exhaust_other_or_hand, global_rule_power, turn_or_delayed_rule | 每当有一张牌被消耗时， / 抽1张牌。 |
| CARD.DEMON_FORM | 恶魔形态 | Rare | Power | 3 | partial | buff_stats, global_rule_power, turn_or_delayed_rule | 在你的回合开始时，获得2点力量。 |
| CARD.HELLRAISER | 地狱狂徒 | Rare | Power | 2 | gap | cost_modify, draw, effect_profile_granularity_gap, global_rule_power, multi_hit_or_aoe, random_or_choose, turn_or_delayed_rule | 每当你抽到名字中有“打击”的牌时，对一名随机敌人打出这张牌。 |
| CARD.JUGGERNAUT | 势不可当 | Rare | Power | 2 | partial | block, damage, global_rule_power, multi_hit_or_aoe, random_or_choose, turn_or_delayed_rule | 每当你获得格挡时，对随机敌人造成5点伤害。 |
| CARD.PYRE | 薪火之源 | Rare | Power | 2 | partial | energy_gain, global_rule_power, turn_or_delayed_rule | 在回合开始时，获得1点能量。 |
| CARD.TANK | 肉盾 | Rare | Power | 1 | gap | cost_modify, effect_profile_granularity_gap, global_rule_power | 承受双倍来自敌人的伤害。 / 敌人的伤害对盟友减半。 |
| CARD.UNMOVABLE | 坚定不移 | Rare | Power | 2 | gap | block, cost_modify, effect_profile_granularity_gap, global_rule_power | 翻倍你每回合第一次从卡牌中获得的格挡。 |
| CARD.BRAND | 烙印 | Rare | Skill | 0 | gap | buff_stats, effect_profile_granularity_gap, exhaust_other_or_hand, global_rule_power, hp_or_self_cost, random_or_choose, whole_hand_state | 失去1点生命。 / 消耗1张牌。 / 获得1点力量。 |
| CARD.CASCADE | 倾泻 | Rare | Skill | X | gap | draw, effect_profile_granularity_gap, pile_fetch_reorder, play_top_or_autoplay, x_cost | 打出你抽牌堆顶部的X张牌。 |
| CARD.COLOSSUS | 巨像 | Rare | Skill | 1 | partial | block, damage, debuff, global_rule_power, turn_or_delayed_rule | 获得5点格挡。 / 在本回合中，有易伤状态的敌人对你造成的伤害降低50%。 |
| CARD.IMPERVIOUS | 岿然不动 | Rare | Skill | 2 | covered | block, exhaust_self | 获得30点格挡。 / 消耗。 |
| CARD.OFFERING | 祭品 | Rare | Skill | 0 | covered | draw, energy_gain, exhaust_self, hp_or_self_cost | 失去6点生命。 / 获得2点能量。 / 抽3张牌。 / 消耗。 |
| CARD.ONE_TWO_PUNCH | 连环拳 | Rare | Skill | 1 | gap | effect_profile_granularity_gap, global_rule_power, multi_hit_or_aoe, next_card_modifier | 在这个回合，你打出的下1张攻击牌会被额外打出一次。 |
| CARD.PRIMAL_FORCE | 原始力量 | Rare | Skill | 0 | gap | buff_stats, effect_profile_granularity_gap, transform_card, upgrade_card, whole_hand_state | 将手牌中的所有攻击牌变化为巨石。 |
| CARD.STOKE | 添柴 | Rare | Skill | 1 | gap | add_or_generate_card, draw, effect_profile_granularity_gap, exhaust_other_or_hand, random_or_choose, upgrade_card, whole_hand_state | 消耗所有手牌。 / 每消耗一张牌，将1张随机牌加入你的手牌。 |
| CARD.ASHEN_STRIKE | 灰烬打击 | Uncommon | Attack | 1 | gap | damage, effect_profile_granularity_gap, exhaust_other_or_hand, exhaust_pile_dependency | 造成6点伤害。 / 你的消耗牌堆中每有一张牌，伤害增加3。 |
| CARD.BLUDGEON | 重锤 | Uncommon | Attack | 3 | covered | damage | 造成32点伤害。 |
| CARD.BULLY | 欺凌 | Uncommon | Attack | 0 | partial | damage, debuff, global_rule_power | 造成4点伤害。 / 该敌人身上每有一层易伤就额外造成2点伤害。 |
| CARD.DISMANTLE | 拆卸 | Uncommon | Attack | 1 | partial | damage, debuff, global_rule_power, multi_hit_or_aoe | 造成8点伤害。 / 如果该敌人有易伤状态，则攻击两次。 |
| CARD.FIGHT_ME | 与我一战！ | Uncommon | Attack | 2 | partial | buff_stats, damage, global_rule_power, multi_hit_or_aoe | 造成5点伤害两次。 / 获得3点力量。 / 该敌人获得1点力量。 |
| CARD.GRAPPLE | 擒拿 | Uncommon | Attack | 1 | partial | block, damage, global_rule_power, turn_or_delayed_rule | 造成7点伤害。 / 当你在本回合获得格挡时，对该敌人造成5点伤害。 |
| CARD.HEMOKINESIS | 御血术 | Uncommon | Attack | 1 | covered | damage, hp_or_self_cost | 失去2点生命。 / 造成15点伤害。 |
| CARD.HOWL_FROM_BEYOND | 彼岸咆哮 | Uncommon | Attack | 3 | gap | damage, effect_profile_granularity_gap, exhaust_other_or_hand, exhaust_pile_dependency, multi_hit_or_aoe, turn_or_delayed_rule | 对所有敌人造成16点伤害。 / 在你的回合开始时，如果这张牌在你的消耗牌堆中，则将其打出。 |
| CARD.PILLAGE | 劫掠 | Uncommon | Attack | 1 | covered | damage, draw | 造成6点伤害。 / 抽牌直到你抽到一张非攻击牌。 |
| CARD.RAMPAGE | 暴走 | Uncommon | Attack | 1 | covered | damage | 造成9点伤害。 / 将这张牌在本场战斗中的伤害增加5。 |
| CARD.SPITE | 怨恨 | Uncommon | Attack | 0 | partial | damage, draw, hp_or_self_cost, multi_hit_or_aoe, turn_or_delayed_rule | 造成5点伤害。 / 如果你在本回合失去过生命值， / 则攻击2次。 |
| CARD.STOMP | 踩踏 | Uncommon | Attack | 3 | partial | damage, multi_hit_or_aoe, turn_or_delayed_rule | 对所有敌人造成12点伤害。 / 你在本回合中每打出过一张攻击牌，其耗能减少1点能量。 |
| CARD.UNRELENTING | 无情猛攻 | Uncommon | Attack | 2 | gap | cost_modify, damage, effect_profile_granularity_gap, global_rule_power, next_card_modifier | 造成12点伤害。 / 你打出的下一张攻击牌耗能变为0点能量。 |
| CARD.UPPERCUT | 上勾拳 | Uncommon | Attack | 2 | partial | damage, debuff, global_rule_power | 造成13点伤害。 / 给予1层虚弱。 / 给予1层易伤。 |
| CARD.WHIRLWIND | 旋风斩 | Uncommon | Attack | X | covered | damage, multi_hit_or_aoe, x_cost | 对所有敌人造成5点伤害X次。 |
| CARD.DRUM_OF_BATTLE | 战鼓 | Uncommon | Power | 0 | gap | draw, effect_profile_granularity_gap, global_rule_power, pile_fetch_reorder, turn_or_delayed_rule | 抽2张牌。 / 在你的回合开始时，消耗你的抽牌堆顶部的牌。 |
| CARD.FEEL_NO_PAIN | 无惧疼痛 | Uncommon | Power | 1 | gap | block, effect_profile_granularity_gap, exhaust_other_or_hand, global_rule_power, turn_or_delayed_rule | 每当有一张牌被消耗时，获得3点格挡。 |
| CARD.INFERNO | 狱火 | Uncommon | Power | 1 | partial | damage, global_rule_power, hp_or_self_cost, multi_hit_or_aoe, turn_or_delayed_rule | 在你的回合开始时，失去1点生命。 / 每当你在你的回合内失去生命时，对所有敌人造成6点伤害。 |
| CARD.INFLAME | 燃烧 | Uncommon | Power | 1 | partial | buff_stats, global_rule_power | 获得2点力量。 |
| CARD.JUGGLING | 杂耍 | Uncommon | Power | 1 | gap | add_or_generate_card, card_modifier_or_enchantment, copy_card, effect_profile_granularity_gap, global_rule_power, whole_hand_state | 将你在每回合打出的第三张攻击牌的复制品加入你的手牌。 |
| CARD.RUPTURE | 撕裂 | Uncommon | Power | 1 | partial | buff_stats, global_rule_power, hp_or_self_cost, turn_or_delayed_rule | 每当你在你的回合失去生命值时, 获得1点力量。 |
| CARD.STAMPEDE | 惊逃 | Uncommon | Power | 2 | gap | cost_modify, effect_profile_granularity_gap, global_rule_power, multi_hit_or_aoe, random_or_choose, turn_or_delayed_rule | 在你的回合结束时，随机打出你手牌中的1张攻击牌攻击随机敌人。 |
| CARD.STONE_ARMOR | 岩石铠甲 | Uncommon | Power | 1 | partial | block, buff_stats, global_rule_power | 获得4层覆甲。 |
| CARD.VICIOUS | 凶恶 | Uncommon | Power | 1 | partial | debuff, draw, global_rule_power, turn_or_delayed_rule | 每当你给予易伤时，抽1张牌。 |
| CARD.BATTLE_TRANCE | 战斗专注 | Uncommon | Skill | 0 | partial | draw, global_rule_power, turn_or_delayed_rule | 抽3张牌。 / 你在本回合内不能再抽任何牌。 |
| CARD.BURNING_PACT | 燃烧契约 | Uncommon | Skill | 1 | gap | draw, effect_profile_granularity_gap, exhaust_other_or_hand, random_or_choose, whole_hand_state | 消耗1张牌。 / 抽2张牌。 |
| CARD.DEMONIC_SHIELD | 恶魔护盾 | Uncommon | Skill | 0 | covered | block, exhaust_self, hp_or_self_cost | 失去1点生命。 / 给予另一位玩家你当前格挡值的格挡。 / 消耗。 |
| CARD.DOMINATE | 主宰 | Uncommon | Skill | 1 | partial | buff_stats, debuff, exhaust_self, global_rule_power | 给予1层易伤。 / 敌人身上每有一层易伤，就获得1点力量。 / 消耗。 |
| CARD.EVIL_EYE | 邪眼 | Uncommon | Skill | 1 | partial | block, exhaust_pile_dependency, turn_or_delayed_rule | 获得8点格挡。 / 如果你在本回合消耗过卡牌，则额外获得8点格挡。 |
| CARD.EXPECT_A_FIGHT | 跃跃欲试 | Uncommon | Skill | 2 | gap | cost_modify, effect_profile_granularity_gap, energy_gain, turn_or_delayed_rule, whole_hand_state | 你的手牌中每有一张攻击牌，就获得1点能量。 / 你在本回合内 / 不能再获得1点能量。 |
| CARD.FLAME_BARRIER | 火焰屏障 | Uncommon | Skill | 2 | partial | block, damage, global_rule_power, hp_or_self_cost | 获得12点格挡。 / 你在这个回合每受到一次攻击，都会对攻击者造成4点伤害。 |
| CARD.FORGOTTEN_RITUAL | 被遗忘的仪式 | Uncommon | Skill | 1 | partial | energy_gain, exhaust_pile_dependency, exhaust_self, turn_or_delayed_rule | 如果你在本回合消耗过卡牌，则获得3点能量。 / 消耗。 |
| CARD.INFERNAL_BLADE | 地狱之刃 | Uncommon | Skill | 1 | gap | add_or_generate_card, cost_modify, effect_profile_granularity_gap, exhaust_other_or_hand, exhaust_self, random_or_choose, turn_or_delayed_rule, whole_hand_state | 将一张随机攻击牌加入你的手牌。那张牌在本回合内可以免费打出。 / 消耗。 |
| CARD.RAGE | 狂怒 | Uncommon | Skill | 0 | partial | block, global_rule_power | 打出此牌后，你在这个回合内每打出一张攻击牌，获得3点格挡。 |
| CARD.SECOND_WIND | 重振精神 | Uncommon | Skill | 1 | gap | block, effect_profile_granularity_gap, exhaust_other_or_hand, whole_hand_state | 消耗手牌中所有非攻击牌，每张获得5点格挡。 |
| CARD.TAUNT | 挑衅 | Uncommon | Skill | 1 | partial | block, debuff, global_rule_power | 获得7点格挡。 / 给予1层易伤。 |
