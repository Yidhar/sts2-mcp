# Kaiser Facing & Potion Timing — 待办设计笔记

更新: 2026-04-27
状态: 待落地, 等当前 typed-effect run 跑出 4-6h 数据后再上

---

## 1. Kaiser Crab Facing 行为修复

### 现象
模型在 kaiser_crab_boss 战中长期猛攻一只手 (mono-target),
不会主动 targeted attack 另一侧来翻转 facing 应对高伤背刺.
TB 数据: `kaiser_facing_change_selected_rate` mean ≈ 0.008,
但 `kaiser_back_attack_risk_mean` 经常 ≥ 0.5.

### 当前奖励三条路径 (`combat_env.py:891-959`)
| 信号 | 触发条件 | 当前权重 |
|---|---|---|
| `KAISER_FACING_CHANGE_BONUS` | `primary_back_attack_active` 1→0 边沿 + play_card/use_potion | 0.80 |
| `KAISER_PRESSURE_KILL_BONUS` | enemy_hp_delta>5 + risk_drop>0.20 (无需转身) | 0.20 |
| `KAISER_BACK_ATTACK_HP_LOSS_PENALTY_SCALE` | 受背刺伤害 (按 hp_loss 缩放) | 见 reward_constants |

### 问题根因
1. Pressure-kill 路径让"猛打"经济上自洽: 反复 +0.20 比一次性 +0.80 facing 更稳
2. Facing-change 检测是边沿触发, 模型必须**精确在某一帧**完成翻转才得分;
   "前面集中削血, 最后一击转身" 这种合理策略也能拿 bonus, 不能改成密集状态信号
3. HP_LOSS_PENALTY 与 enemy_hp_delta 奖励不对等, PPO 短视吃眼前

### 计划 (用户决议)

#### A. Facing bonus 与目标意图伤害挂钩 [P0]
不再固定 0.80. 改为按所翻向的敌人 `intent.total_damage` 缩放:
```python
# pseudo
target_intent_dmg = primary_threat_intent_damage  # already in boss_mechanics
threat_norm = min(target_intent_dmg / 20.0, 2.0)  # 20 dmg 算基线
bonus = KAISER_FACING_CHANGE_BONUS_BASE * threat_norm
# 低伤目标 (<5 dmg) → 几乎不给奖励, 鼓励集中输出
# 高伤目标 (≥30 dmg) → 1.5-2.0 倍奖励, 强烈推动转身
```
新增 `KAISER_FACING_CHANGE_BONUS_BASE = 0.60` (比当前 0.80 略低做基础锚点)

#### B. 不采用 facing 状态信号 [REJECT]
理由: 模型可能采用"前几回合先削血, 最后一击转身"的合理策略.
连续状态奖励会让模型只追求保持朝向, 失去 DPS 节奏.
保留边沿触发, 但用 A 让"转向高伤"激励远大于"转向低伤".

#### C. 加大背身受高伤惩罚, 让 HP loss 自然叠加 [P0]
当前 `KAISER_BACK_ATTACK_HP_LOSS_PENALTY_SCALE` 让背刺伤害成倍计算,
但仍是按 hp_loss 缩放. 计划改为:
```python
# 已有: reward -= hp_loss * SCALE * (1+before_risk) * soften
# 新增: 按 (intent_dmg, hp_loss) 双重缩放, 让模型看到背刺的真正成本
hi_threat_back = before_primary > 0.5 and primary_threat_intent_damage > 15.0
if hi_threat_back:
    reward -= float(KAISER_BACK_ATTACK_HI_THREAT_EXTRA) * (hp_loss / max(player_max_hp, 1.0))
```
新增 `KAISER_BACK_ATTACK_HI_THREAT_EXTRA = 1.50`
(以 max_hp 比例计算, 跨 boss/HP 不同的战斗仍然平稳)

#### 落地顺序
1. 先观察 typed-effect run 跑 4-6h, 看 kaiser 行为是否自然改善
2. 若 kaiser_facing_change_selected_rate 仍 < 0.02 且 boss WR 仍 < 0.35,
   一次性落地 A + C, 重启训练
3. 不要单独上 A 或 C, 两者协同: A 给"转身的甜头", C 给"不转的痛"

### 落地状态 (§1)
- ✅ A: `KAISER_FACING_CHANGE_BONUS_BASE=0.60` + `KAISER_FACING_INTENT_DMG_REF=20.0`
  + `KAISER_FACING_INTENT_DMG_SCALE_MAX=2.0` 落地. `combat_env._boss_mechanic_reward()`
  kaiser 分支用 `primary_threat_intent_damage / 20.0` 缩放, 上限 2x.
- ✅ C: `KAISER_BACK_ATTACK_HI_THREAT_DMG=15.0` + `KAISER_BACK_ATTACK_HI_THREAT_EXTRA=1.50`
  落地. 当被攻击者 intent ≥ 15 dmg 且玩家受到 hp_loss > 0 时, 按 `hp_loss/max_hp` 比例
  追加罚分.
- ✅ B (REJECT) 仍未做且不会做 (会破坏"末回合转身"合理策略).

---

## 2. 药水时机 (Phase 4) 信号未进 reward 通道 — 关键缺陷

### 现象
模型仍然第一回合丢完所有药水, 部分药水的使用收益为零或负面.
TB 数据: `boss_combat/potion_use_quality_selected_mean` ≈ 0.10-0.20 (很低),
`waste_risk_selected_mean` ≈ 0.15-0.20 (一直存在大额浪费),
`family_potion_rate` 第一回合远高于后续回合.

### 根因 — Phase 4 的 timing 信号根本没进 reward
`_potion_timing_profile()` 在 `muzero/train.py` 计算了 use_quality / waste_risk /
save_value, 但只用在两个地方:
- `_combat_action_quality_bias()` — 只调整 inference 时 action 排序, **不影响 return**
- TB metrics — 只可视化

`combat_env.py` 完全没有引用任何 timing 字段, 现在的药水 reward 路径:
```python
def _encounter_potion_use_reward(action):  # combat_env.py:794
    if tier == "boss":   return POTION_USE_BOSS_BONUS = 0.0   # 已置零
    if tier == "elite":  return POTION_USE_ELITE_BONUS = 0.0  # 已置零
    return POTION_USE_MONSTER_BONUS + POTION_USE_MONSTER_PENALTY = 0.0  # 已置零
```
+ `POTION_HOARDING_PENALTY_PER_POTION = 0.0` (终局也没罚).

**所以模型在策略梯度里看到的状态是**:
- 用药水 = 0 reward (不论时机好坏)
- 不用药水留到死 = 0 reward
- 全丢光 vs 全留着, **return 完全没差别**

planner bias 只能在 search 时偏向不用, 但 PPO/MuZero 学到的 (state, action, value) 三元组中,
value 完全不区分药水时机. 网络当然学不到.

### 修复方案 (递进, 不引入硬规则)

#### Phase 4b: timing → reward shaping [P0, 必须]
在 `combat_env.py` 加新方法, 从 `_potion_timing_profile` 拿 use_quality/waste_risk:
```python
def _potion_timing_step_reward(self, action, before_obs, ...) -> float:
    if self._action_family(action) not in {"use_potion", "potion"}:
        return 0.0
    # 调用与 train.py 相同的 timing profile (从环境侧近似计算或缓存)
    profile = self._compute_potion_timing(action, before_obs)
    use_q = profile["use_quality"]
    waste = profile["waste_risk"]
    # 软奖惩: 用得好 +小额, 用得废 -小额, 让 return 反映 timing
    reward = (
        POTION_TIMING_QUALITY_SCALE * use_q
        - POTION_TIMING_WASTE_SCALE * waste
    )
    return reward
```
新常量:
- `POTION_TIMING_QUALITY_SCALE = 0.40` (use_quality=1 给 +0.40)
- `POTION_TIMING_WASTE_SCALE = 0.60` (waste_risk=1 罚 -0.60)
- 罚比奖大 → 鼓励"宁可不用"

#### 实施细节
1. `_potion_timing_profile` 当前在 train.py 上, 需要抽到 `sts2_env/potion_timing.py` 公共模块
   两边共享
2. 环境侧 timing 计算 input 是 `before_obs + action + legal_actions`,
   不需要 raw_obs 也能算 (env 自己有 raw_obs)
3. 第一回合所有药水的 `use_quality` 普遍很低 (没 followup, 不是斩杀, 没机制对应),
   所以加这条 reward 后第一回合丢完会被拉低 return ~ -2.4 (4 药水 × -0.6)
   足够触发模型学习

#### Phase 4c: 用 metric 验证修复有效 [P1]
跑 2-3h 后看:
- `family_potion_rate` 第一回合 vs 后续回合的差异是否缩小
- `potion_waste_risk_selected_mean` 是否从 0.18 降到 < 0.10
- `boss/win` 是否提升 (药水保留到关键时刻应该提高 boss 通过率)

### 落地状态 (§2 Phase 4b)
- ✅ `sts2_env/potion_timing.py` 新建 — pure-function `compute_potion_timing()` 共享
  use_quality / waste_risk / save_value 计算. 不依赖 trainer instance.
- ✅ `combat_env._potion_timing_step_reward()` 调用上述函数, 按
  `+POTION_TIMING_QUALITY_SCALE * use_q - POTION_TIMING_WASTE_SCALE * waste`
  注入到每 step reward. 计数器 `_potion_timing_quality_events / _potion_timing_waste_events`.
- ✅ `POTION_TIMING_QUALITY_SCALE=0.40` / `POTION_TIMING_WASTE_SCALE=0.60` 加入 reward_constants.
- 已验证 sanity: FIRE_POTION on normal → use_q=1.0, waste=0; ENERGY_POTION 无 followup
  → use_q=0, waste=0.45 → 实际罚分 -0.27. 第一回合丢光 4 药水 ≈ -1.08 return 损失.

---

## 3. 落地顺序 (整体)

| 步骤 | 内容 | 触发条件 |
|---|---|---|
| 当前 | 让 typed-effect run 跑 4-6h, 观察 baseline | — |
| 1 | 落地 §2 Phase 4b (potion timing → reward) | 必做, 优先级最高 |
| 2 | 观察 2-3h, 验证 potion_waste_risk 下降 | 数据确认 |
| 3 | 落地 §1 A + C (kaiser facing + back-attack) | 若 kaiser 行为仍未自愈 |
| 4 | 落地 §4 多选烧牌 confirm 终止信号 | 与 §3 同一 reward 周期一起做 |
| 5 | 再观察 4-6h, 评估 boss WR 趋势 | 全局健康度 |
| 6 | 给 hourly cron 加游戏存活监测 (§5) | 立即, 解耦于训练改动 |

---

## 4. 多选烧牌 (Card Selection) — 选满循环 + 不主动 confirm

### 现象
当玩家进入多选消耗界面 (例如 POTION.GLOWWATER / 净化效果牌 / TOUCH_OF_INSANITY 等
"choose up to N cards to ..."), 模型行为:
1. 永远把候选数选到上限 (5/5), 不会主动选 K<N 然后 confirm
2. 反复换最后一张被选中的牌, 选了又换, 形成循环
3. 必须等到 max 限制阻止再选才被动 confirm

### 根因
当前 card_selection 决策的奖励/价值结构里:
- "选一张" 是有 reward 的 (作为 selection 进度的隐式正向)
- "confirm" 在选数 < N 时**没有任何 reward 或 value 信号** 区别于 "继续选"
- 状态相似的 confirm vs reselect 在 policy 头看几乎等价
- 替换最后一张本质上是无意义的 "重新选" — model 学不到这是无效循环
- 终止信号缺失: episode reward 不区分 "选了 3 张就 confirm" vs "硬塞满 5 张然后 confirm"

类似于 Phase 4 药水问题: timing 信号没进 reward, 模型学不到时机.
这里是 **selection 终止信号没进 reward**, 模型学不到适可而止.

### 修复方案

#### A. card_selection step 加无效循环检测 [P0]
在 `combat_env.py` step 或专门的 selection reward 路径里:
```python
# 状态追踪: 在 reset 重置, step 累加
self._selection_replace_count: int = 0
self._selection_last_picked_id: str | None = None

def _selection_step_reward(self, action, before_obs, after_obs):
    if self._action_family(action) != "card_selection":
        return 0.0
    # 检测 "选一张又换掉" 模式
    picked = action.get("card", {}).get("id") or ""
    if picked and picked == self._selection_last_picked_id:
        self._selection_replace_count += 1
    else:
        self._selection_replace_count = 0
    self._selection_last_picked_id = picked

    reward = 0.0
    if self._selection_replace_count >= 2:
        # 同一张牌被反复选-换 ≥ 2 次 → 显著惩罚
        reward -= float(SELECTION_LOOP_PENALTY) * self._selection_replace_count
    return reward
```
新常量:
- `SELECTION_LOOP_PENALTY = 0.30` (每次循环往复 -0.30, 累计很快变痛)

#### B. confirm 早停奖励 [P0]
在 selection 触发 finalize 时根据"选数 / 上限"给微正向:
```python
def _selection_confirm_reward(self, picked_count, max_count, ...):
    if picked_count >= max_count:
        return 0.0  # 选满不奖励
    # 选了 K < N 张主动 confirm → 给小额正向, 越早越多
    saved = max_count - picked_count
    return float(SELECTION_EARLY_CONFIRM_BONUS) * (saved / max(max_count, 1))
```
新常量:
- `SELECTION_EARLY_CONFIRM_BONUS = 0.40` (选半数 confirm ≈ +0.20)

#### C. 长期: candidate token 给 "已选过" 标志 [P1]
`observation_v3.py` 的 candidate local token 应该把"本次 selection 已选过的 card id"
做成数值 slot, 让模型 attention 能直接看到"这张已选/未选". 现在 model 要靠 history
token 推断, 太隔了一层.

新增 token numeric slot (复用现有未占用槽位):
```
candidate slot[K] = 1.0 if card_id in current_selection else 0.0
```

#### 实施细节
- 多选烧牌的合法 action 列表里通常包括 "confirm" / "skip" / "done" 类 terminator;
  需要先确认 `_action_family` 里能区分 "选一张" vs "confirm"
- 如果 confirm 是 `kind == "card_selection"` 但 selection_semantics == "confirm"
  之类的子类型, A/B 实现要分开处理

#### 验收
- TB 新指标 `combat_quality_selection_loop_count` / `selection_early_confirm_count`
- 行为指标: 净化平均选 5/5 → 应降到 ≈ 3/5 (经验值, 看实战)
- 不应破坏正常多选场景 (例如确实需要选满 N 时仍能选满)

### 落地状态 (§4)
- ✅ §4A 已落地: `combat_env._card_selection_step_reward()` + `SELECTION_LOOP_PENALTY=0.30`,
  追踪 `_selection_last_picked_id` / `_selection_flip_count` / `_selection_loop_events`,
  flip ≥ 2 时按 flip 数线性扣分.
- ⚠️ §4B 部分落地: `SELECTION_EARLY_CONFIRM_BONUS=0.40` 已加, 但当前 bridge **不输出
  `max_pick / selection_pick_limit`**, 所以 confirm 时无法计算 (max-K)/max — 该字段
  未来在 bridge 加 `MaxSelections` 反射后 reward 才会非零. 目前 confirm 不奖也不罚
  (loop penalty 仍能阻止循环).
- ✅ §4C 已落地: bridge `card_selection:select` action payload 加 `is_selected` 字段
  (`BridgeGameApi.cs:2703` 附近), observation_v3 `SELECTION_OPERATOR_LOCAL` token
  slot 14 读取该字段. 模型可直接看到候选是否已选.

---

## 5. 监控盲区: 游戏崩溃 hourly cron 看不到

### 现象
2026-04-27 19:28 游戏崩溃, 看门狗自动重启了, 训练侧 (新 bridge_client 补丁后) 透明
吃下了 outage 没报错. 我的 hourly cron 报告**完全没监测到**, 用户必须自己发现.

### 根因
hourly cron 当前数据源:
1. `events.out.tfevents.*` — 训练 metrics (训练没崩就一切正常)
2. 训练 stdout — 只筛 slow_bridge_step + Traceback (游戏崩溃但训练没崩 → 无信号)

游戏自身的崩溃日志 / watchdog action 完全在监控之外.

### 修复方案

#### A. 监控游戏 PID + 重启次数 [P0]
hourly cron 步骤 5 加:
```bash
# 当前游戏 PID + 启动时间
tasklist | grep SlayTheSpire2.exe
# 看 session_0.json mtime — 每次重启会刷新
stat -c "%y" /c/Users/yidhar/AppData/Roaming/SlayTheSpire2/bridge/session_0.json
```
若 mtime 在过去 1h 内变化, 报告 "游戏在 H 期间重启 N 次".

#### B. 监控 watchdog 日志 [REJECT — 死路]
计划 grep launcher.py stdout (任务 b970qplix.output 等) 抓 `[launcher] killing` /
`[launcher] restarted` 行, **但实测发现 harness 任务 .output 始终为 0 bytes** —
launcher 输出走 TTY, harness 不 tail 这种 stream. 此路不通.
替代方案: §5A 已经覆盖 — session.json mtime 移动 = watchdog 重启了游戏.

#### C. 监控 bridge 端 mod 日志 [P2]
`%APPDATA%/SlayTheSpire2/bridge/bridge-debug-N.log` 有 mod side trace, 比 stdout 更全;
但量大且非结构化, 优先级低.

### 落地状态
- ✅ §5A: 已落地. `packages/rl-agent/scripts/game_health_snapshot.sh` 输出
  `game_pid=N;session_mtime=...;launchers=N;train_procs=N` 单行, 可被 cron 直接 grep.
  cron 简报每小时调用一次, 比对 session_mtime 与上次, mtime 变化 = 游戏重启.
- ❌ §5B: 验证失败 (launcher 输出黑洞). 由 §5A 覆盖.
- ⏸️ §5C: 暂搁置.

不要同时改 §1 + §2, 保持因果可分析.
