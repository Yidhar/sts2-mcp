"""Dashboard print for the 10 substitute metrics."""
from tensorboard.backend.event_processing import event_accumulator
import os, sys

run = sys.argv[1] if len(sys.argv) > 1 else "logs_muzero/muzero_token_memory_combat_sandbox_2envs_20260424_042022"
ea = event_accumulator.EventAccumulator(run, size_guidance={"scalars": 0})
ea.Reload()


def last(t):
    try:
        return ea.Scalars(t)[-1].value
    except Exception:
        return None


def show(t, fmt="{:.4f}"):
    v = last(t)
    if v is None:
        print(f"  {t}: MISSING")
    else:
        print(f"  {t}: " + fmt.format(v))


print("=== 1. boss 整体趋势 ===")
show("recent_tail/256/boss_win_rate")
show("recent_tail/64/boss_win_rate")

print("\n=== 2. 具体 boss/elite 名字 win rate (256) ===")
for k in sorted(ea.Tags().get("scalars", [])):
    if k.startswith("recent_tail/256/encounter_") and k.endswith("_win_rate"):
        show(k)

print("\n=== 3. 整体能力 ===")
for k in ["recent_tail/256/win_rate", "recent_tail/256/reward_mean",
          "recent_tail/256/length_mean", "recent_tail/256/loss_rate"]:
    show(k)

print("\n=== 4. 药水使用 ===")
show("decision/combat/family_use_potion_rate")
show("decision/combat/family_play_card_rate")
show("decision/combat/family_end_turn_rate")

print("\n=== 5. Q 趋势 (rollout planner) ===")
for k in ["search/combat/direct_rollout_objective_q_mean",
          "search/combat/direct_rollout_q_mean",
          "search/combat/direct_rollout_risk_q_mean"]:
    show(k)

print("\n=== 6. 新 aux 收敛 ===")
for k in ["search/combat/direct_rollout_uncertainty_mean",
          "search/combat/direct_rollout_surprise_mean",
          "search/combat/direct_rollout_branch_disagreement_mean",
          "search/combat/direct_rollout_latent_drift_mean",
          "search/combat/direct_rollout_surface_entropy_mean"]:
    show(k)

print("\n=== 7. 损失健康 ===")
for k in ["loss/total", "loss/policy", "loss/value", "loss/reward",
          "loss/surprise", "loss/jepa_next_hidden",
          "loss/objective_value", "loss/planner_q",
          "loss/future_world_aux", "loss/latent_gaussian_reg"]:
    show(k)

print("\n=== 8. surprise head 学到了吗 ===")
for k in ["metric/surprise_target_mean", "metric/surprise_pred_mean",
          "metric/surprise_mae", "metric/surprise_target_offset"]:
    show(k)

print("\n=== 9. policy 探索 ===")
for k in ["metric/student_policy_entropy", "metric/teacher_policy_entropy",
          "metric/latent_policy_distill_kl",
          "metric/latent_policy_distill_top1_agreement"]:
    show(k)

print("\n=== 10. VRAM ===")
for k in ["memory/reserved_gb", "memory/allocated_gb",
          "memory/max_reserved_gb", "memory/reserved_minus_allocated_gb"]:
    show(k)

print("\n=== bonus: buffer 采样 ===")
for k in ["buffer/sample_boss_rate", "buffer/sample_elite_rate",
          "buffer/sample_normal_rate", "buffer/sample_weak_rate",
          "buffer/sample_hard_encounter_rate", "buffer/size"]:
    show(k, "{:.0f}" if k == "buffer/size" else "{:.4f}")
