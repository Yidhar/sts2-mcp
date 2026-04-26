"""Dashboard v2: (last, max, mean_all, nonzero) four-value summary.

Counters baseline-read bias when a metric fires sparsely (e.g. ceremonial
shaping only non-zero on ceremonial episodes).
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


def stats(t):
    try:
        vals = ea.Scalars(t)
        n = len(vals)
        if n == 0:
            return None
        values = [v.value for v in vals]
        last = values[-1]
        mx = max(values)
        mean_all = sum(values) / n
        nonzero = sum(1 for v in values if v > 1e-9)
        return last, mx, mean_all, nonzero, n
    except Exception:
        return None


def show(t, width=56):
    r = stats(t)
    if r is None:
        print(f"  {t:<{width}}  MISSING")
        return
    last, mx, mean_all, nz, n = r
    print(f"  {t:<{width}} last={last:.4f} max={mx:.4f} mean={mean_all:.4f} nz={nz}/{n}")


print("=== 1. boss win rate ===")
for k in ["recent_tail/256/boss_win_rate", "recent_tail/64/boss_win_rate",
          "recent_tail/256/win_rate", "recent_tail/256/reward_mean"]:
    show(k)

print("\n=== 2. per-encounter (256) ===")
for k in sorted(ea.Tags().get("scalars", [])):
    if k.startswith("recent_tail/256/encounter_") and k.endswith("_win_rate"):
        show(k)

print("\n=== 3. boss_combat kaiser ===")
for k in ["boss_combat/kaiser_back_attack_risk_mean",
          "boss_combat/kaiser_defense_candidate_count_mean",
          "boss_combat/kaiser_defense_selected_rate"]:
    show(k)

print("\n=== 4. boss_combat ceremonial ===")
for k in ["boss_combat/ceremonial_one_card_lock_rate",
          "boss_combat/ceremonial_stun_window_rate",
          "boss_combat/ceremonial_low_impact_count_mean",
          "boss_combat/ceremonial_high_impact_count_mean",
          "boss_combat/ceremonial_low_impact_selected_rate",
          "boss_combat/ceremonial_high_impact_selected_rate"]:
    show(k)

print("\n=== 5. boss_combat behavior ===")
for k in ["boss_combat/family_end_turn_rate",
          "boss_combat/wasteful_end_turn_rate",
          "boss_combat/family_use_potion_rate",
          "boss_combat/family_play_card_rate",
          "boss_combat/playable_cards_left_mean",
          "boss_combat/potion_unused_on_death_rate",
          "boss_combat/decision_count",
          "boss_combat/direct_policy_used_rate"]:
    show(k)

print("\n=== 6. direct rollout ===")
for k in ["search/combat/direct_rollout_objective_q_mean",
          "search/combat/direct_rollout_q_mean",
          "search/combat/direct_rollout_risk_q_mean",
          "search/combat/direct_rollout_uncertainty_mean",
          "search/combat/direct_rollout_surprise_mean",
          "search/combat/direct_rollout_branch_disagreement_mean",
          "search/combat/direct_rollout_surface_entropy_mean",
          "search/combat/direct_rollout_latent_drift_mean"]:
    show(k)

print("\n=== 7. loss ===")
for k in ["loss/total", "loss/policy", "loss/value", "loss/reward",
          "loss/surprise", "loss/jepa_next_hidden", "loss/future_world_aux",
          "loss_ratio/aux_total"]:
    show(k)

print("\n=== 8. VRAM + buffer ===")
for k in ["memory/reserved_gb", "memory/allocated_gb",
          "buffer/size", "buffer/sample_boss_rate"]:
    show(k)
