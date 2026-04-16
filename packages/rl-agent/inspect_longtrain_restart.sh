#!/usr/bin/env bash
set -euo pipefail
sleep 12
run_name="longtrain_ironclad_async_bf16_s64_b8_e1_20260416_165711"
log_dir="/mnt/e/game/project/sts2_mcp/packages/rl-agent/logs_attention/$run_name"
printf '%s\n' '=== PS ==='
ps -p 269 -o pid=,ppid=,etime=,cmd= || true
printf '%s\n' '=== STDOUT ==='
if [ -f "$log_dir/stdout.log" ]; then tail -n 40 "$log_dir/stdout.log"; else echo 'stdout missing'; fi
printf '%s\n' '=== STDERR ==='
if [ -f "$log_dir/stderr.log" ]; then tail -n 80 "$log_dir/stderr.log"; else echo 'stderr missing'; fi
printf '%s\n' '=== ASYNC STATUS ==='
if [ -f "$log_dir/async_status.json" ]; then cat "$log_dir/async_status.json"; else echo 'async_status missing'; fi