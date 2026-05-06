from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import statistics, time, re, math
RUN=Path('logs_muzero/muzero_boss_recovery_20260504_132317')
ea=EventAccumulator(str(RUN), size_guidance={'scalars':0}); ea.Reload(); tags=set(ea.Tags()['scalars'])

def st(tag,n=50):
    if tag not in tags: return None
    xs=ea.Scalars(tag)
    if not xs: return None
    tail=xs[-n:]; vs=[x.value for x in tail]
    return dict(count=len(xs), step=xs[-1].step, last=xs[-1].value, mean=sum(vs)/len(vs), med=statistics.median(vs), max=max(vs), min=min(vs), first=xs[0].value, first_step=xs[0].step)

def print_stats(title, taglist, n=50):
    print('\n## '+title)
    for tag in taglist:
        s=st(tag,n)
        if not s:
            print(f'MISSING {tag}')
        else:
            print(f"{tag:72s} count={s['count']:5d} step={s['step']:7d} first={s['first']:.6g}@{s['first_step']} last={s['last']:.6g} med{n}={s['med']:.6g} mean{n}={s['mean']:.6g} min{n}={s['min']:.6g} max{n}={s['max']:.6g}")

print('event_mtime', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(next(RUN.glob('events.out.tfevents.*')).stat().st_mtime)), 'tags', len(tags))
# discover some names
for pat in ['buffer','optimizer','total_steps','loss/','root_top','entropy','decision_count']:
    ms=sorted([t for t in tags if pat in t])[:80]
    print('DISCOVER', pat, len(ms), ms[:40])

print_stats('training_recent_tail', [
'recent_tail/64/win_rate','recent_tail/64/boss_win_rate','recent_tail/64/elite_win_rate','recent_tail/64/normal_win_rate','recent_tail/64/reward_mean','recent_tail/64/length_mean',
'recent_tail/256/win_rate','recent_tail/256/boss_win_rate','recent_tail/256/elite_win_rate','recent_tail/256/normal_win_rate','recent_tail/256/reward_mean','recent_tail/256/length_mean',
'recent_tail/256/boss_sample_count','recent_tail/256/elite_sample_count','recent_tail/256/normal_sample_count'
],20)
print_stats('loss_tail50', [
'loss/total','loss/future_world_aux','loss/future_bank_state','loss/future_bank_delta','loss/future_bank_occupancy','loss/future_bank_token_presence','loss/policy','loss/value','loss/reward','loss/surprise','loss/jepa_next_hidden','loss/latent_gaussian_reg'
],50)
print_stats('planner_tail50', [
'planning/rollout_steps_used','planning/root_top1_share','planning/root_policy_entropy','planning/root_valid_actions','boss_combat/direct_rollout_objective_q_mean','boss_combat/direct_rollout_uncertainty_mean','boss_combat/direct_rollout_risk_q_mean'
],50)
print_stats('global_boss_combat_tail50', [
'boss_combat/decision_count','boss_combat/family_end_turn_rate','boss_combat/direct_end_turn_selected_rate',
'boss_combat/forced_end_turn_selected_rate','boss_combat/wasteful_end_turn_available_rate','boss_combat/wasteful_end_turn_bias_applied_rate','boss_combat/wasteful_end_turn_rate','boss_combat/wasteful_end_turn_selected_rate','boss_combat/true_wasteful_end_turn_available_rate','boss_combat/true_wasteful_end_turn_selected_rate','boss_combat/bad_end_turn_selected_rate','boss_combat/prior_transient_only_end_turn_rate','boss_combat/transient_leaked_selected_rate','boss_combat/end_turn_unknown_selected_rate','boss_combat/strategic_defer_end_turn_selected_rate',
'boss_combat/energy_mean','boss_combat/playable_cards_left_mean','boss_combat/positive_action_count_mean','boss_combat/urgent_positive_action_count_mean','boss_combat/deferable_positive_action_count_mean',
'boss_combat/zero_energy_x_cost_available_mean','boss_combat/zero_energy_x_cost_selected_rate','boss_combat/x_cost_zero_bad_selected_rate_p0',
'boss_combat/potion_selected_rate','boss_combat/potion_selected_when_available_rate','boss_combat/potion_low_urgency_selected_rate','boss_combat/potion_save_recommended_selected_rate','boss_combat/potion_unused_on_death_rate','boss_combat/potion_unused_on_death_raw_rate','boss_combat/potion_use_quality_selected_mean',
'boss_combat/hp_cost_self_lethal_selected_rate','boss_combat/hp_cost_low_margin_selected_rate','boss_combat/hp_cost_unblockable_value_mean',
'boss_combat/kaiser_back_attack_risk_mean','boss_combat/kaiser_facing_change_candidate_count_mean','boss_combat/kaiser_facing_change_selected_rate','boss_combat/kaiser_risky_end_turn_selected_rate','boss_combat/kaiser_pressure_selected_rate','boss_combat/kaiser_defense_selected_rate',
'boss_combat/ceremonial_stun_window_rate','boss_combat/ceremonial_high_impact_selected_rate','boss_combat/ceremonial_low_impact_selected_rate'
],50)
encs=['kaiser_crab_boss','ceremonial_beast_boss','the_kin_boss','the_insatiable_boss','knowledge_demon_boss','phrog_parasite_elite','knights_elite','soul_nexus_elite','construct_menagerie_normal','slumbering_beetle_normal','ovicopter_normal']
metrics=['decision_count','family_end_turn_rate','forced_end_turn_selected_rate','wasteful_end_turn_available_rate','wasteful_end_turn_bias_applied_rate','wasteful_end_turn_rate','true_wasteful_end_turn_selected_rate','bad_end_turn_selected_rate','energy_mean','playable_cards_left_mean','positive_action_count_mean','zero_energy_x_cost_selected_rate','potion_selected_when_available_rate','potion_low_urgency_selected_rate','potion_unused_on_death_rate','hp_cost_self_lethal_selected_rate','kaiser_back_attack_risk_mean','kaiser_facing_change_candidate_count_mean','kaiser_facing_change_selected_rate','kaiser_risky_end_turn_selected_rate','kaiser_pressure_selected_rate','ceremonial_stun_window_rate','ceremonial_high_impact_selected_rate','ceremonial_low_impact_selected_rate']
print('\n## per_encounter_last_med20')
print('encounter metric last med20 count step')
for enc in encs:
    for m in metrics:
        tag=f'boss_combat/{enc}/{m}'
        s=st(tag,20)
        if s:
            # only print nonzero-ish or critical metrics
            if (abs(s['last'])>1e-9 or abs(s['med'])>1e-9 or m in ['family_end_turn_rate','decision_count','wasteful_end_turn_rate','true_wasteful_end_turn_selected_rate','kaiser_back_attack_risk_mean']):
                print(f'{enc:32s} {m:45s} last={s["last"]:.6g} med20={s["med"]:.6g} count={s["count"]} step={s["step"]}')

print('\n## per_encounter_recent_tail_256')
for enc in encs:
    for m in ['sample_count','win_rate']:
        tag=f'recent_tail/256/encounter_{enc}_{m}'
        s=st(tag,20)
        if s: print(f'{tag:75s} last={s["last"]:.6g} med20={s["med"]:.6g} count={s["count"]} step={s["step"]}')
