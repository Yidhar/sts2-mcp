import json
from pathlib import Path
p=Path('logs_muzero/muzero_boss_recovery_20260504_132317/diagnostics/action_offenders.jsonl')
rows=[]
with p.open(encoding='utf-8') as f:
    for line in f:
        try: r=json.loads(line)
        except: continue
        if r.get('offender_type') in ['kaiser_facing_missed','kaiser_risky_end_turn','ceremonial_missed_stun_window','ceremonial_low_impact_under_lock','low_quality_potion_selected','zero_energy_x_cost_selected','high_save_value_potion_unused']:
            rows.append(r)
for typ in ['kaiser_facing_missed','kaiser_risky_end_turn','ceremonial_missed_stun_window','ceremonial_low_impact_under_lock','low_quality_potion_selected','zero_energy_x_cost_selected','high_save_value_potion_unused']:
    print('\n###', typ)
    subset=[r for r in rows if r.get('offender_type')==typ]
    for r in subset[-5:]:
        print('step',r.get('global_step'),'ep',r.get('episode_id'),'enc',r.get('encounter_id'),'turn',r.get('turn'),'sel',r.get('selected_family'),r.get('selected_title'),r.get('selected_target'))
        print(' flags',r.get('reason_flags'))
        print(' state',r.get('state_summary'))
        print(' alt',r.get('alternative_actions')[:4] if isinstance(r.get('alternative_actions'),list) else r.get('alternative_actions'))
