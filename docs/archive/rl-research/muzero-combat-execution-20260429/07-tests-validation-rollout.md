# 07 — Tests, Validation and Rollout

本文件定义所有任务完成后的验证方式。
不要只跑一条 smoke test 就认为完成；需要离线单测、短训练指标、长训练趋势三层验收。

---

## 1. 单元测试清单

建议新增或更新：

```text
packages/rl-agent/tests/test_end_turn_context_dump.py
packages/rl-agent/tests/test_action_offender_metrics.py
packages/rl-agent/tests/test_x_cost_dynamic_energy.py
packages/rl-agent/tests/test_potion_transition_diagnostics.py
packages/rl-agent/tests/test_end_turn_taxonomy.py
packages/rl-agent/tests/test_strategic_skip_narrowing.py
packages/rl-agent/tests/test_refund_no_followup.py
packages/rl-agent/tests/test_bridge_transient_end_turn_guard.py
packages/rl-agent/tests/test_card_lifecycle_tokens.py
packages/rl-agent/tests/test_future_world_aux_card_lifecycle.py
packages/rl-agent/tests/test_kaiser_facing_semantics.py
packages/rl-agent/tests/test_ceremonial_mechanics.py
packages/rl-agent/tests/test_insatiable_offenders.py
packages/rl-agent/tests/test_direct_planner_drift_gate.py
packages/rl-agent/tests/test_encounter_balanced_replay.py
packages/rl-agent/tests/test_human_demo_dataset.py
```

如果 repo 现有测试结构不同，可以放在对应目录，但文件名要可搜索。

---

## 2. 基础命令

```powershell
cd E:\game\project\sts2_mcp\packages\rl-agent

# 全量测试
python -m pytest tests -q

# 关键测试
python -m pytest tests/test_end_turn_taxonomy.py -q
python -m pytest tests/test_x_cost_dynamic_energy.py -q
python -m pytest tests/test_kaiser_facing_semantics.py -q
python -m pytest tests/test_bridge_transient_end_turn_guard.py -q
python -m pytest tests/test_direct_planner_drift_gate.py -q
```

如果项目使用其它测试入口，请在 PR/输出中写明实际命令。

---

## 3. 短跑验证：2k-5k steps

目的：验证没有 crash，指标 namespace 出现，dump 可读。

必须检查：

```text
diagnostics/end_turn_contexts.jsonl exists
diagnostics/action_offenders.jsonl exists

boss_combat/bad_end_turn_selected_rate
boss_combat/forced_end_turn_selected_rate
boss_combat/strategic_defer_end_turn_selected_rate
boss_combat/kaiser_crab_boss/kaiser_facing_change_candidate_count_mean
boss_combat/ceremonial_beast_boss/one_card_lock_active_rate
bridge_fast_step/wait_ms_p95
planner/drift_gate_mean
```

通过标准：

- 无 crash。
- JSONL 每行是合法 JSON。
- global 和 per-encounter namespace 都出现。
- `wait_ms_p95` 没有异常长。

---

## 4. 中跑验证：20k steps

目的：验证修复方向没有明显负反馈。

重点看：

```text
bad_end_turn_selected_rate
strategic_defer_end_turn_selected_rate
zero_energy_x_cost_selected_rate
refund_no_followup_low_value_selected_rate
kaiser_facing_change_selected_rate
ceremonial_low_impact_under_lock_selected_rate
latent_drift
drift_gate
```

通过标准：

- `bad_end_turn_selected_rate` 不应持续高位。
- `zero_energy_x_cost_selected_rate` 应下降，除非有合理 mechanism tags。
- `strategic_defer_end_turn_selected_rate` 可以存在，但 offender dump 中必须能解释。
- `latent_drift` 高时 `drift_gate` 应下降。
- Kaiser facing candidate 不再长期为 0。

---

## 5. 长跑验证：50k-100k steps

目的：验证胜率趋势。

重点表：

```text
256/win
256/boss_win
256/elite_win
256/normal_win
64/boss_win
reward_mean
loss/total
loss/future_lifecycle_aux
```

per encounter：

```text
kaiser_crab_boss
ceremonial_beast_boss
the_insatiable_boss
the_kin_boss
knowledge_demon_boss
knights_elite
phrog_parasite_elite
```

通过标准：

- boss overall 不继续明显下降。
- Kaiser/Ceremonial/Insatiable 至少一个从低位恢复，且没有其它 boss 大幅崩。
- normal/elite 不因 boss 过采样退化。
- reward mean 不下降。

---

## 6. Definition of Done

### P0 DoD

完成 A/B/C 后：

- 能解释每一次 selected end_turn。
- end_turn 三分类指标存在。
- transient only-end-turn 不靠长 sleep。
- X-cost 0 能量行为可诊断。
- Kaiser facing candidate 不再因为错误字段一直为 0。

### P1 DoD

完成 D/E 后：

- 卡牌生命周期字段覆盖 Ironclad 88 张和无色牌关键机制。
- observation/action token 包含 lifecycle/cost/hand/pile/boss mechanism。
- future-world aux 能预测关键 pile/hand/energy/boss state。
- Ceremonial 和 Insatiable 有独立 offender dump。

### P2 DoD

完成 F 后：

- direct planner 有 drift gate。
- replay 不再 boss 过采样到挤压基础能力。
- human demo 可以进入训练。
- 瓶颈期可以通过 demo + gated planner 改善，而不是恢复高延迟 MCTS。

---

## 7. 回滚策略

每个阶段都要可单独关闭：

```text
--enable-action-offender-dump
--enable-end-turn-taxonomy
--enable-bridge-transient-guard
--enable-card-lifecycle-aux
--enable-boss-mechanic-aux
--enable-drift-gated-planner
--demo-dataset
```

如果实际项目不使用 CLI flag，也要有 config 开关。

回滚优先级：

1. 先关 planner 权重改动。
2. 再关 aux loss 权重。
3. 最后才关 observation 字段。

不要为了回滚删除诊断字段；诊断字段应尽量保留。
