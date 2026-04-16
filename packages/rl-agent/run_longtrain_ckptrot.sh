#!/usr/bin/env bash
set -euo pipefail
cd "/mnt/e/game/project/sts2_mcp/packages/rl-agent"
run_name="longtrain_ironclad_async_bf16_s64_b8_e1_ckptrot_20260416_180416"
log_dir="/mnt/e/game/project/sts2_mcp/packages/rl-agent/logs_attention/longtrain_ironclad_async_bf16_s64_b8_e1_ckptrot_20260416_180416"
ckpt_dir="/mnt/e/game/project/sts2_mcp/packages/rl-agent/checkpoints_attention/longtrain_ironclad_async_bf16_s64_b8_e1_ckptrot_20260416_180416"
mkdir -p "$log_dir" "$ckpt_dir"
export PYTORCH_ALLOC_CONF=expandable_segments:True
printf 'RUN_NAME=%s\nLOG_DIR=%s\nCKPT_DIR=%s\n' "$run_name" "$log_dir" "$ckpt_dir" > "$log_dir/launcher_meta.txt"
exec ./.venv-wsl-rocm/bin/python -u ./train_attention_policy.py \
  --total-timesteps 200000 \
  --n-envs 4 \
  --session-files "/mnt/c/Users/yidhar/AppData/Roaming/SlayTheSpire2/bridge/session_0.json,/mnt/c/Users/yidhar/AppData/Roaming/SlayTheSpire2/bridge/session_1.json,/mnt/c/Users/yidhar/AppData/Roaming/SlayTheSpire2/bridge/session_2.json,/mnt/c/Users/yidhar/AppData/Roaming/SlayTheSpire2/bridge/session_3.json" \
  --collector-mode async \
  --combat-sandbox \
  --snapshot-pool /mnt/e/game/project/sts2_mcp/datasets/curated_combat_ironclad_mixed_provenance \
  --snapshot-curated-subset bootstrap_human_plus_local_act1clear_roomwin_only_minus_combat_reset_failures \
  --snapshot-sample-mode tier_weighted_encounter_balanced \
  --snapshot-tier-weights "weak:0.45,normal:0.35,elite:0.15,boss:0.05" \
  --device cuda \
  --text-device cuda \
  --amp --amp-dtype bf16 \
  --n-steps 64 \
  --batch-size 8 \
  --n-epochs 1 \
  --reset-timeout-ms 90000 \
  --step-timeout-ms 20000 \
  --checkpoint-interval-timesteps 10000 \
  --checkpoint-keep-last 3 \
  --log-dir "$log_dir" \
  --checkpoint-dir "$ckpt_dir" \
  > "$log_dir/stdout.log" 2> "$log_dir/stderr.log"