#!/usr/bin/env bash
set -euo pipefail
sleep 15
log_dir=/mnt/e/game/project/sts2_mcp/packages/rl-agent/logs_attention/longtrain_ironclad_async_bf16_s64_b8_e1_20260416_170003
printf '%s\n' '=== WSL TRAIN PROCS ==='
ps -eo pid,ppid,etime,cmd | grep -E 'train_attention_policy.py|longtrain_ironclad_async_bf16_s64_b8_e1_20260416_170003' | grep -v grep || true
printf '%s\n' '=== LOG DIR ==='
ls -lah "" || true
printf '%s\n' '=== STDOUT TAIL ==='
if [ -f "/stdout.log" ]; then tail -n 60 "/stdout.log"; else echo 'stdout missing'; fi
printf '%s\n' '=== STDERR TAIL ==='
if [ -f "/stderr.log" ]; then tail -n 120 "/stderr.log"; else echo 'stderr missing'; fi
printf '%s\n' '=== ASYNC STATUS ==='
if [ -f "/async_status.json" ]; then cat "/async_status.json"; else echo 'async_status missing'; fi