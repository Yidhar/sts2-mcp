from pathlib import Path
import statistics, math, time, re, os
try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except Exception as e:
    print('IMPORT_ERROR', repr(e)); raise
RUN = Path('logs_muzero/muzero_boss_recovery_20260504_132317')
events = sorted(RUN.glob('events.out.tfevents.*'), key=lambda p:p.stat().st_mtime)
print('events', [(p.name,p.stat().st_size,time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(p.stat().st_mtime))) for p in events])
if not events:
    raise SystemExit
# Load all event files by directory
acc = EventAccumulator(str(RUN), size_guidance={'scalars':0})
acc.Reload()
tags = acc.Tags().get('scalars', [])
print('scalar_tag_count', len(tags))
# print tags containing interesting terms
interesting_terms = ['loss/', 'boss_combat', 'episode/', 'recent', 'buffer', 'reward', 'win', 'kaiser', 'ceremonial', 'end_turn', 'potion', 'x_cost', 'offender', 'curriculum']
print('interesting_tags')
for t in sorted(tags):
    if any(term in t for term in interesting_terms):
        print(' ', t)

def vals(tag):
    if tag not in tags: return []
    return acc.Scalars(tag)

def stat(tag, n=50):
    xs=vals(tag)
    if not xs: return None
    tail=xs[-n:]
    vs=[x.value for x in tail]
    return {
        'tag': tag, 'count': len(xs), 'first_step': xs[0].step, 'last_step': xs[-1].step,
        'first': xs[0].value, 'last': xs[-1].value,
        'mean_tail': sum(vs)/len(vs), 'median_tail': statistics.median(vs), 'max_tail': max(vs), 'min_tail': min(vs),
    }

wanted = [
 'train/buffer_size','train/optimizer_steps','train/total_steps','episode/reward','episode/length','episode/win','episode/max_floor','episode/death_floor',
 'loss/total','loss/future_world_aux','loss/future_bank_state','loss/future_bank_delta','loss/policy','loss/value','loss/reward','loss/jepa_next_hidden','loss/surprise','loss/latent_gaussian_reg',
 'planning/root_top1_share','planning/root_policy_entropy','planning/root_valid_actions','planning/rollout_steps_used',
 'boss_combat/family_end_turn_rate','boss_combat/direct_end_turn_selected_rate','boss_combat/wasteful_end_turn_rate','boss_combat/wasteful_end_turn_bias_applied_rate',
 'boss_combat/wasteful_end_turn_available_rate','boss_combat/wasteful_end_turn_selected_rate',
 'boss_combat/energy_mean','boss_combat/positive_action_count_mean','boss_combat/playable_cards_left_mean',
 'boss_combat/kaiser_back_attack_risk_mean','boss_combat/kaiser_facing_change_candidate_count_mean','boss_combat/kaiser_facing_change_selected_rate','boss_combat/kaiser_risky_end_turn_selected_rate','boss_combat/kaiser_pressure_selected_rate','boss_combat/kaiser_defense_selected_rate',
 'boss_combat/ceremonial_stun_window_mean','boss_combat/ceremonial_high_impact_selected_rate','boss_combat/ceremonial_low_impact_selected_rate',
 'boss_combat/zero_energy_x_cost_available_mean','boss_combat/zero_energy_x_cost_selected_rate',
 'boss_combat/potion_unused_on_death_rate','boss_combat/family_use_potion_rate','boss_combat/potion_low_urgency_selected_rate','boss_combat/potion_save_recommended_selected_rate',
]
print('\nselected_stats_tail50')
for t in wanted:
    s=stat(t,50)
    if s:
        print(f"{t:75s} count={s['count']:5d} step={s['last_step']:7d} last={s['last']:.6g} med50={s['median_tail']:.6g} mean50={s['mean_tail']:.6g} max50={s['max_tail']:.6g}")

# encounter-specific tags containing selected metrics
print('\nencounter_specific_boss_combat_tags')
for t in sorted(tags):
    if t.startswith('boss_combat/') and t.count('/')>=2 and any(k in t for k in ['end_turn','kaiser','ceremonial','win','potion','x_cost','energy','positive','playable']):
        s=stat(t,20)
        if s:
            print(f"{t:95s} last={s['last']:.6g} med20={s['median_tail']:.6g} count={s['count']} step={s['last_step']}")

# Recent tail logs if scalar names match
print('\nrecent_tail_tags')
for t in sorted(tags):
    if 'recent' in t.lower() or 'tail' in t.lower() or '256' in t or '64' in t:
        s=stat(t,20)
        if s:
            print(f"{t:75s} last={s['last']:.6g} med20={s['median_tail']:.6g} count={s['count']} step={s['last_step']}")
