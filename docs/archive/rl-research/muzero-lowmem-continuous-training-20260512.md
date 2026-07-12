# MuZero Pass-Large 低显存持续训练状态（2026-05-12）

## 当前决策

**不重启，继续当前低显存训练。**

原因：

- 低显存优化已生效，显存峰值稳定在约 **7.62 GB**，reserved 约 **8.1–8.3 GB**。
- loss 无 spike，future world/bank 辅助头稳定。
- buffer 过 5k 后，normal combat 正在恢复：64-window normal 已到达/略过 gate，256-window normal 仍低但持续上行。
- 当前主要瓶颈不是显存，而是 normal encounter 战术质量，尤其仍有少量过度防御/低 tempo 防御。

## Active run

```text
run_id = muzero_pass_large_sandbox_weaknormal_lowmem_bucket_guardlock_v7_20260512_110850
log_dir = packages/rl-agent/logs_muzero/muzero_pass_large_sandbox_weaknormal_lowmem_bucket_guardlock_v7_20260512_110850
ckpt_dir = packages/rl-agent/checkpoints_muzero/muzero_pass_large_sandbox_weaknormal_lowmem_bucket_guardlock_v7_20260512_110850
train_pid = 3076
monitor_pid = 3079
```

## 已确认生效的低显存优化

### 1. Train-time planner 降级

```text
--planner-memory-profile train
combat_rollout_steps = 1
combat_rollout_beam_width = 1
action_rollout_buckets = 4,8,16,32,96
action_rollout_chunk_size = 16
```

目的：训练期减少 rollout/action 批量展开导致的显存峰值。eval/max profile 后续仍可拉高 planner。

### 2. Activation checkpointing

```text
--activation-checkpointing on
```

已覆盖 token_memory / model 中重模块路径，包括 world/local/candidate/memory/prediction 相关 block。

### 3. Dynamic token length bucketing

```text
STS2_TOKEN_LENGTH_BUCKETING=1
```

作用：不是完全固定 MAX_WORLD_TOKENS，而是按 active token 长度 trim 到 bucket，减少 attention shape 显存，同时避免 ROCm 上过多动态 shape 抖动。

### 4. SDPA backend 固定

```text
STS2_SDPA_BACKEND=math
STS2_SDPA_STRICT=1
```

当前选择是为了稳定和可控显存峰值；本机 ROCm/torch 下没有继续赌 flash/efficient 后端。

### 5. Allocator 配置

```text
PYTORCH_ALLOC_CONF=garbage_collection_threshold:0.6,max_split_size_mb:256
```

当前 observed：

```text
memory/max_allocated_gb ≈ 7.62
memory/reserved_gb      ≈ 8.1–8.3
memory/empty_cache avg  <= 0.05
```

## 最新 gate 摘要（buffer≈5487）

```text
verdict: FAIL，但正在接近 pass
buffer/size                          5487
recent_tail/64/win_rate              0.9219   fail, target 0.95
recent_tail/64/weak_win_rate         1.0000   pass
recent_tail/64/normal_win_rate       0.9091   pass, target 0.90
recent_tail/64/hard_normal_win_rate  1.0000   pass
recent_tail/256/normal_win_rate      0.8505   fail, target 0.88
recent_tail/256/hard_normal_win_rate 0.7857   pass
bad_pure_block avg_tail              0.00217  just above target 0.002
refund_no_followup_with_progress     0.00250  pass
potion_low_urgency                   0.00972  pass, very close
loss/total max_tail                  14.76    pass
future_world_aux max_tail            0.33     pass
future_bank_state max_tail           0.083    pass
future_bank_delta max_tail           0.253    pass
```

## 当前战术瓶颈

Normal encounter 仍有少量过度防御/低 tempo 问题。diagnostics recent window 中主要 offender：

```text
card_pure_block_selected
pure_block_no_alternative_selected
pure_block_low_value_pressure_selected
pure_block_progress_alternative_selected
card_block_waste_selected
card_no_damage_pressure_selected
```

主要 encounter 候选：

```text
encounter.fogmog_normal
encounter.construct_menagerie_normal
encounter.sewer_clam_normal
encounter.corpse_slugs_normal
encounter.the_lost_and_forgotten_normal
encounter.exoskeletons_normal
encounter.scrolls_of_biting_normal
encounter.the_obscura_normal
```

## 后续自动/人工决策线

继续当前 run 到至少 **8k buffer**，不因 5k 初期 fail 立刻重启。

### 继续当前 run 的条件

满足任意明显改善趋势即继续：

```text
64 normal_win_rate >= 0.90 持续
256 normal_win_rate 继续向 0.88 上行
bad_pure_block avg_tail <= 0.002 附近或下降
loss/memory 稳定
```

### 切 targeted normal sandbox 的条件

到 **buffer 8k 左右** 如果仍满足：

```text
256 normal_win_rate < 0.86–0.87 且无上升趋势
或 64 normal_win_rate 再次跌破 0.90
或 bad_pure_block avg_tail 明显 > 0.002
```

则停止当前 train/monitor，只从最新 checkpoint 启动 targeted normal sandbox，重点采样：

```text
encounter.the_obscura_normal=4.0
encounter.fogmog_normal=4.0
encounter.construct_menagerie_normal=3.5
encounter.sewer_clam_normal=3.0
encounter.corpse_slugs_normal=3.0
encounter.the_lost_and_forgotten_normal=3.0
encounter.exoskeletons_normal=3.0
encounter.scrolls_of_biting_normal=3.0
encounter.bowlbugs_normal=3.0
encounter.frog_knight_normal=3.0
encounter.mytes_normal=3.0
encounter.ovicopter_normal=3.0
```

低显存参数必须保持不变。

## 检查命令

```bash
cd /mnt/e/game/project/sts2_mcp/packages/rl-agent
RUN_ID=$(cat logs_muzero/latest_lowmem_run_id.txt)
./.venv-wsl-rocm/bin/python scripts/monitor_combat_sandbox_gate.py \
  --run-dir logs_muzero/$RUN_ID \
  --tail 20 \
  --min-buffer 5000 \
  --max-reserved-gb 22 \
  --max-peak-allocated-gb 20 \
  --max-empty-cache-rate 0.2 \
  --diagnostics-window 1500
```


## 2026-05-12 13:24 CST — low-VRAM optimization deployed; targeted continuous training restarted

### Decision
The previous weak/normal lowmem run was stopped because the 8k gate remained failed while memory/loss were healthy. The failure was tactical/distributional rather than VRAM-related:

- previous run: `muzero_pass_large_sandbox_weaknormal_lowmem_bucket_guardlock_v7_20260512_110850`
- stopped PIDs: train `3076`, monitor `3079`
- latest checkpoint used: `checkpoints_muzero/muzero_pass_large_sandbox_weaknormal_lowmem_bucket_guardlock_v7_20260512_110850/muzero_step_00008200`
- final check before stop: buffer `8400`; 64-window win/weak/normal/hard-normal still below gate; bad pure block / x-cost / low-urgency potion still above thresholds.

### New run
Started targeted normal/hard-normal repair run:

- run id: `muzero_pass_large_sandbox_targeted_normal_lowmem_v1_20260512_131738`
- log dir: `packages/rl-agent/logs_muzero/muzero_pass_large_sandbox_targeted_normal_lowmem_v1_20260512_131738`
- ckpt dir: `packages/rl-agent/checkpoints_muzero/muzero_pass_large_sandbox_targeted_normal_lowmem_v1_20260512_131738`
- resume mode: weights from `muzero_step_00008200`, fresh optimizer, fresh buffer
- sampling: weak/normal with normal-heavy targeted offender weights
- marker updated: `packages/rl-agent/logs_muzero/latest_lowmem_run_id.txt`

### Low-VRAM controls active

- `STS2_SDPA_BACKEND=math`
- `STS2_SDPA_STRICT=1`
- `STS2_TOKEN_LENGTH_BUCKETING=1`
- `--planner-memory-profile train`
- `--activation-checkpointing on`
- batch size `8`, unroll `5`, planner rollout train profile/chunking retained

### Early smoke at 13:24 CST

This is still a cold buffer and should not be used as a real quality gate yet.

- train PID: `6033` alive
- monitor PID: `6052` alive
- buffer: `474`
- event file active
- memory healthy: max allocated `7.45 GB`, reserved `7.97 GB`, empty-cache avg `0`
- loss healthy: total max-tail `16.77`; future_world_aux `0.258`; future_bank_state `0.082`; future_bank_delta `0.265`; token_slot_source `0`
- tactical metrics still cold/unstable: normal win `0.80`, bad_pure_block avg-tail `0.00796`, zero-energy X-cost `0.00999`, potion_low_urgency `0.0140`

### Next monitoring gates

Do not judge before buffer has enough samples.

1. buffer `3000`: health gate only — memory/loss must remain stable; check offender ranking.
2. buffer `5000`: trend gate — normal/hard-normal and bad pure block should be improving.
3. buffer `8000`: real sandbox gate — if 64-window collapse persists, collect offender/death slices and patch tactical guards or sampling.


## 2026-05-12 13:37 CST — low-VRAM controls verified; continuous training left running

### Runtime status

Current active run remains:

```text
muzero_pass_large_sandbox_targeted_normal_lowmem_v1_20260512_131738
```

Process check:

- train PID `6033`: alive, GPU run active
- monitor PID `6052`: alive
- event file mtime: active at `13:36:59 CST`
- `stderr.log`: only WSL screen-size warning; no Python exception/OOM

### Verified active low-VRAM controls

Process environment confirms:

```text
PYTORCH_ALLOC_CONF=garbage_collection_threshold:0.6,max_split_size_mb:256
STS2_SDPA_BACKEND=math
STS2_SDPA_STRICT=1
STS2_TOKEN_LENGTH_BUCKETING=1
```

Startup log confirms:

```text
Activation checkpointing: cli=on enabled=True
Planner memory profile: train (steps=1, beam=1, buckets=4,8,16,32, chunk=16)
Token-memory: d_model=192 slots=24 slot_layout=pass_large_v1 bank_token_slots=8 world_layers=6 decoder_layers=3 bank_top_k=7 rollout_buckets=(4, 8, 16, 32, 96)
Mixed precision: bf16
```

### 13:37 cold-buffer metrics

Still below the minimum buffer for a quality verdict, so current verdict is **WAIT_BUFFER**, not fail.

```text
buffer/size                         1403 / 3000 health-gate minimum
recent_tail/64 win_rate             0.889
recent_tail/64 weak_win_rate        1.000
recent_tail/64 normal_win_rate      0.870
bad_pure_block avg_tail             0.000
zero_energy_x_cost avg_tail         0.00357
potion_low_urgency avg_tail         0.00876
hp_cost_low_margin avg_tail         0.000
loss/total max_tail                 15.61
future_world_aux max_tail           0.256
future_bank_state max_tail          0.0996
future_bank_delta max_tail          0.273
memory/max_allocated_gb max_tail    7.45
memory/reserved_gb max_tail         7.98
memory/empty_cache_called avg_tail  0
```

### Interpretation

- VRAM optimization is effective for the current Pass-Large config: peak allocation is under `8 GB`, far below the `20 GB` guard; reserved memory is under `8 GB`, far below the `22 GB` guard.
- Loss is stable; no future-world/bank spike is present.
- The run should **continue**. Do not restart before the buffer reaches at least `3000`, and preferably evaluate trend at `5000` / real gate at `8000`.
- Main remaining tactical offender is still block-card selection frequency, especially `防御` / `防御+` / `血墙`, but the stricter progress/waste metrics are currently passing in the cold window.

### Current offender/death focus

Recent diagnostics top offenders/deaths still point to these encounters for later targeted repair if the 5k/8k gates fail:

```text
encounter.chompers_normal
encounter.two_tailed_rats_normal
encounter.fogmog_normal
encounter.frog_knight_normal
encounter.owl_magistrate_normal
encounter.nibbits_normal
encounter.vine_shambler_normal
encounter.the_obscura_normal
```

Recent deaths:

```text
ENCOUNTER.FOGMOG_NORMAL
ENCOUNTER.CULTISTS_NORMAL
ENCOUNTER.RUBY_RAIDERS_NORMAL
ENCOUNTER.SLUMBERING_BEETLE_NORMAL
ENCOUNTER.THE_OBSCURA_NORMAL
```

### Next action

Leave current training running. Next meaningful checks:

1. buffer `>=3000`: health gate only; memory/loss must remain stable.
2. buffer `>=5000`: check whether normal win and offender metrics are trending better.
3. buffer `>=8000`: decide pass/fail for this targeted sandbox phase.
