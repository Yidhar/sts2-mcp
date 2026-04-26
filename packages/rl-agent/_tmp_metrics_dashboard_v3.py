"""Dashboard v3: per-encounter kaiser namespace + wasteful diagnosis."""
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
all_tags = set(ea.Tags().get("scalars", []))


def stats(t):
    try:
        vals = ea.Scalars(t)
        n = len(vals)
        if not n:
            return None
        values = [v.value for v in vals]
        last = values[-1]
        mx = max(values)
        mean_all = sum(values) / n
        nz = sum(1 for v in values if v > 1e-9)
        return last, mx, mean_all, nz, n
    except Exception:
        return None


def show(t, width=64):
    r = stats(t)
    if r is None:
        if t in all_tags:
            print(f"  {t:<{width}} EMPTY")
        else:
            print(f"  {t:<{width}} MISSING")
        return
    last, mx, mean_all, nz, n = r
    print(f"  {t:<{width}} last={last:.4f} max={mx:.4f} mean={mean_all:.4f} nz={nz}/{n}")


print("=== Global boss + win rate ===")
for k in ["recent_tail/256/boss_win_rate", "recent_tail/64/boss_win_rate",
          "recent_tail/256/win_rate", "recent_tail/256/reward_mean"]:
    show(k)

print("\n=== Per-encounter win (256) ===")
for k in sorted(all_tags):
    if k.startswith("recent_tail/256/encounter_") and k.endswith("_win_rate"):
        show(k)

print("\n=== Wasteful/空过 (global boss_combat) ===")
for k in ["boss_combat/wasteful_end_turn_rate",
          "boss_combat/family_end_turn_rate",
          "boss_combat/direct_end_turn_selected_rate",
          "boss_combat/wasteful_end_turn_bias_applied_rate",
          "boss_combat/energy_mean",
          "boss_combat/positive_action_count_mean",
          "boss_combat/playable_cards_left_mean",
          "boss_combat/zero_energy_x_cost_available_mean",
          "boss_combat/zero_energy_x_cost_selected_rate"]:
    show(k)

print("\n=== Kaiser (global boss_combat) ===")
for k in ["boss_combat/kaiser_back_attack_risk_mean",
          "boss_combat/kaiser_defense_candidate_count_mean",
          "boss_combat/kaiser_defense_selected_rate",
          "boss_combat/kaiser_facing_change_candidate_count_mean",
          "boss_combat/kaiser_facing_change_selected_rate",
          "boss_combat/kaiser_pressure_candidate_count_mean",
          "boss_combat/kaiser_pressure_selected_rate",
          "boss_combat/kaiser_risky_end_turn_selected_rate"]:
    show(k)

# Per-encounter kaiser (new namespace)
print("\n=== Kaiser per-encounter (boss_combat/kaiser_crab_boss/*) ===")
kaiser_tags = sorted(t for t in all_tags if t.startswith("boss_combat/kaiser_crab_boss/"))
if not kaiser_tags:
    print("  (no kaiser_crab_boss/* tags yet - run needs more kaiser episodes)")
else:
    for k in kaiser_tags:
        show(k)

# Per-encounter ceremonial
print("\n=== Ceremonial per-encounter (boss_combat/ceremonial_beast_boss/*) ===")
c_tags = sorted(t for t in all_tags if t.startswith("boss_combat/ceremonial_beast_boss/"))
if not c_tags:
    print("  (no ceremonial_beast_boss/* tags yet)")
else:
    for k in c_tags:
        show(k)

print("\n=== Direct rollout ===")
for k in ["search/combat/direct_rollout_objective_q_mean",
          "search/combat/direct_rollout_q_mean",
          "search/combat/direct_rollout_uncertainty_mean",
          "search/combat/direct_rollout_branch_disagreement_mean",
          "search/combat/direct_rollout_latent_drift_mean"]:
    show(k)

print("\n=== Loss + VRAM ===")
for k in ["loss/total", "loss/policy", "loss/value", "loss/future_world_aux",
          "loss_ratio/aux_total", "memory/reserved_gb", "buffer/size"]:
    show(k)
