# 03 — Bridge Fast Step

combat sandbox 是必须保留的快速战斗训练器。
解决“抽卡/洗牌/动画短窗口只暴露 end_turn”时，不能靠长 sleep。

---

## TASK-C1 — Bridge Transient Only-End-Turn Guard

### 目标

bridge payload 必须告诉 Python：

- 当前 action frontier 是否稳定。
- 只有 end_turn 是因为真的没动作，还是动画/队列/抽牌洗牌未完成的短暂状态。
- state_version/state_hash 是否已经推进。

### Owned paths

```text
mods/sts2-bridge/Scripts/BridgeGameApi.cs
mods/sts2-bridge/Scripts/BridgeGameApi.EnvPayloads.cs
mods/sts2-bridge/Scripts/BridgeGameApi.Actions.cs
```

实际文件名以 repo 当前结构为准，但 C# bridge action payload 必须覆盖。

### payload 目标结构

在 observation 或 legal action response 中新增：

```json
{
  "actionability": {
    "frontier_stable": true,
    "transient_only_end_turn": false,
    "only_end_turn_reason": "stable_no_actions|queue_pending|animation_pending|draw_shuffle_pending|unknown",
    "state_version": 123,
    "state_hash": "abc...",
    "queue_pending": false,
    "animation_pending": false,
    "draw_shuffle_pending": false,
    "legal_non_end_turn_count": 3
  }
}
```

### 判定逻辑

当 legal actions 只有 end_turn 时：

1. 如果队列/动画/抽牌/洗牌仍 pending：
   - `frontier_stable=false`
   - `transient_only_end_turn=true`
   - reason 对应 pending 类型
2. 如果 pending 全部 false，连续状态稳定：
   - `frontier_stable=true`
   - `transient_only_end_turn=false`
   - reason=`stable_no_actions`
3. 如果无法判断：
   - reason=`unknown`
   - Python 端短轮询 1-2 次，不长等。

### 禁止做法

不要：

```csharp
Thread.Sleep(500);
Thread.Sleep(1000);
固定等待 N 秒后再返回
```

如果必须等待，只能是非常短的状态轮询，并且暴露 wait_ms/poll_count 指标。

### 测试/验证

1. 人造 payload：queue pending + only end_turn => transient true。
2. 人造 payload：无 pending + only end_turn => stable no actions。
3. 人造 payload：有 non-end_turn legal action => transient false。
4. 实机短跑：`transient_only_end_turn_count` 非零时，不应直接选择 end_turn。

---

## TASK-C2 — Python Fast Step Short Polling

### 目标

Python 端收到 `transient_only_end_turn=true` 时短轮询 bridge，等待稳定 action frontier，但不能使用长 sleep。

### Owned paths

```text
packages/rl-agent/sts2_env/combat_env.py
packages/rl-agent/sts2_env/headless_sim_bridge_client.py
packages/rl-agent/muzero/train.py        # metrics only
packages/rl-agent/tests/test_bridge_transient_end_turn_guard.py
```

### 策略

伪代码：

```python
def wait_for_actionable_frontier(obs, max_wait_ms=120, poll_interval_ms=10):
    if not obs.actionability.transient_only_end_turn:
        return obs

    deadline = now + max_wait_ms
    polls = 0
    while now < deadline:
        sleep(poll_interval_ms)
        polls += 1
        obs2 = bridge.get_state()
        if is_stable_action_frontier(obs2):
            return obs2.with_wait_stats(polls=polls)

    return obs.with_wait_timeout_flag()
```

默认建议：

```text
max_wait_ms: 80-120
poll_interval_ms: 8-16
```

具体数值要通过指标验证，而不是拍脑袋加长。

### stable 条件

满足任一：

1. 出现 non-end_turn legal action。
2. actionability.frontier_stable=true 且 transient_only_end_turn=false。
3. state_version/hash 连续两次稳定，且 pending false。

### 指标

```text
bridge_fast_step/wait_ms_mean
bridge_fast_step/wait_ms_p95
bridge_fast_step/poll_count_mean
bridge_fast_step/transient_only_end_turn_count
bridge_fast_step/transient_delayed_count
bridge_fast_step/transient_leaked_count
bridge_fast_step/wait_timeout_count
bridge_fast_step/stable_no_actions_count
```

### 验收标准

1. `wait_ms_p95` 不应接近 combat reset 等待时间。
2. `transient_leaked_count` 应接近 0。
3. `stable_no_actions_count` 仍允许正常 end_turn。
4. throughput 不明显下降。

### 测试

1. fake bridge 第 1 次 only end_turn transient，第 2 次出现 play_card => 返回第 2 次。
2. fake bridge 一直 transient 到 timeout => 标记 timeout，但不无限等待。
3. fake bridge stable no actions => 不等待。
4. fake bridge state hash 稳定 + pending false => 返回 stable。
