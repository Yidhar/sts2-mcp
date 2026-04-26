"""Filter boss episodes by potion availability and compare win rates.

Uses TB trajectory per-episode scalars to correlate:
- episode/boss_entry_potion_count
- boss/win, boss/loss (per-boss-episode summaries)
- boss_combat/potion_unused_on_death_rate
- boss_combat/family_use_potion_rate
"""
import os
import sys
from tensorboard.backend.event_processing import event_accumulator

run = sys.argv[1] if len(sys.argv) > 1 else None
if run is None:
    runs = sorted([d for d in os.listdir("logs_muzero")
                   if d.startswith("muzero_token_memory_combat_sandbox_2envs_")])
    run = f"logs_muzero/{runs[-1]}"
print(f"run: {run}\n")

ea = event_accumulator.EventAccumulator(run, size_guidance={"scalars": 0})
ea.Reload()


def series(t):
    try:
        return [(v.step, v.value) for v in ea.Scalars(t)]
    except Exception:
        return []


# Align by episode_count step: boss/* + episode/boss_entry_* + boss_combat/*
potion_count = dict(series("episode/boss_entry_potion_count"))
boss_win = dict(series("boss/win"))
boss_loss = dict(series("boss/loss"))
boss_reward = dict(series("boss/reward"))
potion_unused = dict(series("boss_combat/potion_unused_on_death_rate"))
use_potion_rate = dict(series("boss_combat/family_use_potion_rate"))

# Collect only steps that have boss summary (boss/attempt_count present)
attempts = dict(series("boss/attempt_count"))
print(f"total boss episode summaries: {len(attempts)}")

# Bucket by potion count at boss entry
buckets = {"0": [], "1": [], "2": [], "3": [], "4+": []}
for step in sorted(attempts.keys()):
    pc = int(potion_count.get(step, 0))
    win = int(boss_win.get(step, 0))
    unused = potion_unused.get(step, 0)
    rew = boss_reward.get(step, 0)
    use_rate = use_potion_rate.get(step, 0)
    key = "4+" if pc >= 4 else str(pc)
    buckets[key].append((step, win, unused, rew, use_rate))


def summarize(name, rows):
    if not rows:
        print(f"  {name}: (0 samples)")
        return
    n = len(rows)
    wins = sum(r[1] for r in rows)
    unused_sum = sum(r[2] for r in rows)
    rew_sum = sum(r[3] for r in rows)
    use_sum = sum(r[4] for r in rows)
    print(f"  {name}: n={n} win={wins}/{n} ({wins/n*100:.1f}%) "
          f"potion_unused_on_death={unused_sum/n:.2f} reward={rew_sum/n:+.2f} "
          f"use_potion_rate={use_sum/n:.3f}")


print("\n=== Boss eps bucketed by entry_potion_count ===")
for k in ("0", "1", "2", "3", "4+"):
    summarize(f"potion_count={k}", buckets[k])

# Sanity: aggregate
all_rows = [r for b in buckets.values() for r in b]
summarize("ALL boss eps", all_rows)

# Also dump last 15 boss eps detail
print("\n=== Last 15 boss eps (step, pot_count, win, unused, reward, use_rate) ===")
for step in sorted(attempts.keys())[-15:]:
    pc = int(potion_count.get(step, 0))
    w = int(boss_win.get(step, 0))
    u = potion_unused.get(step, 0)
    r = boss_reward.get(step, 0)
    up = use_potion_rate.get(step, 0)
    print(f"  ep={step}  pot_count={pc}  win={w}  unused={u:.1f}  reward={r:+.2f}  use_rate={up:.3f}")
