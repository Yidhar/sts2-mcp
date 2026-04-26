import os
from tensorboard.backend.event_processing import event_accumulator

runs = sorted([d for d in os.listdir("logs_muzero")
               if d.startswith("muzero_token_memory_combat_sandbox_2envs_")])
new = runs[-1]
print(f"current run: {new}\n")
ea = event_accumulator.EventAccumulator(f"logs_muzero/{new}",
                                        size_guidance={"scalars": 0})
ea.Reload()


def show(t):
    try:
        v = ea.Scalars(t)[-1].value
        c = len(ea.Scalars(t))
        return v, c
    except Exception:
        return None, 0


base = {
    "boss_combat/playable_cards_left_mean": 3.31,
    "boss_combat/potion_unused_on_death_rate": 1.00,
    "boss_combat/family_use_potion_rate": 0.069,
    "boss_combat/direct_rollout_uncertainty_mean": 53.62,
    "boss_combat/direct_rollout_objective_q_mean": -0.37,
    "episode/boss_entry_potion_count": 3.0,
    "loss_ratio/aux_total": 0.4985,
    "recent_tail/256/boss_win_rate": 0.0,
}
print(f"{'metric':<55} {'baseline':>10} {'now':>10} {'delta':>10} {'count':>6}")
print("-" * 96)
for k, b in base.items():
    v, c = show(k)
    if v is None:
        print(f"{k:<55} {b:>10.4f} {'MISSING':>10} {'N/A':>10} {c:>6}")
    else:
        print(f"{k:<55} {b:>10.4f} {v:>10.4f} {v - b:>+10.4f} {c:>6}")

print("\n=== boss sample volume check ===")
for k in ["recent_tail/64/boss_sample_share", "recent_tail/64/boss_sample_count",
          "recent_tail/256/boss_sample_share", "recent_tail/256/boss_sample_count",
          "buffer/sample_boss_rate"]:
    v, c = show(k)
    print(f"  {k}: {v if v is None else f'{v:.4f}'} (count={c})")
