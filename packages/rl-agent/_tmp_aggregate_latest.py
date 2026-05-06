import json, collections, statistics, math, os, time
from pathlib import Path
RUN = Path('logs_muzero/muzero_boss_recovery_20260504_132317')
POOL = {
    'encounter.kaiser_crab_boss', 'encounter.ceremonial_beast_boss', 'encounter.the_kin_boss',
    'encounter.the_insatiable_boss', 'encounter.knowledge_demon_boss', 'encounter.phrog_parasite_elite',
    'encounter.knights_elite', 'encounter.soul_nexus_elite', 'encounter.construct_menagerie_normal',
    'encounter.slumbering_beetle_normal', 'encounter.ovicopter_normal'
}

def norm_enc(x):
    x = str(x or '').strip().lower()
    if not x:
        return ''
    if x.startswith('encounter.'):
        return x
    if x.startswith('encounter_'):
        return 'encounter.' + x.split('encounter_', 1)[1]
    if x.startswith('enounter.'):
        return 'encounter.' + x.split('.', 1)[1]
    if x.startswith('enounter_'):
        return 'encounter.' + x.split('enounter_', 1)[1]
    if '.' in x:
        return 'encounter.' + x.split('.')[-1]
    return x

def load_jsonl(name):
    p = RUN / 'diagnostics' / name
    rows=[]; bad=0
    if not p.exists():
        return rows, bad, None
    with p.open('r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                bad += 1
    return rows, bad, p

def gp(r, path, default=0):
    x=r
    for p in path.split('.'):
        if isinstance(x, dict):
            x=x.get(p, default)
        else:
            return default
    return x

def num(x):
    try:
        if x is None or x == '': return 0.0
        return float(x)
    except Exception:
        return 0.0

def avg(rows, path):
    return sum(num(gp(r,path)) for r in rows)/len(rows) if rows else 0.0

def pct(a,b):
    return (100.0*a/b) if b else 0.0

def q(values, p):
    if not values: return 0.0
    values=sorted(values)
    idx=min(len(values)-1, max(0, int(round((len(values)-1)*p))))
    return values[idx]

print('RUN', RUN)
print('RUN_EXISTS', RUN.exists(), 'mtime', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(RUN.stat().st_mtime)) if RUN.exists() else None)

# end turn contexts
et,bad,p = load_jsonl('end_turn_contexts.jsonl')
print('\n## END_TURN_CONTEXTS', 'rows', len(et), 'bad', bad, 'size', p.stat().st_size if p else None, 'mtime', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(p.stat().st_mtime)) if p else None)
print('global_step range', (min((r.get('global_step',0) for r in et), default=None), max((r.get('global_step',0) for r in et), default=None)), 'episodes', (min((r.get('episode_id',0) for r in et), default=None), max((r.get('episode_id',0) for r in et), default=None)))
print('offpool', sum(norm_enc(r.get('encounter_id')) not in POOL for r in et))
class_counter = collections.Counter(r.get('end_turn_class') for r in et)
print('class_counts', dict(class_counter), 'class_rates', {k: round(pct(v,len(et)),2) for k,v in class_counter.items()})
# reason flags total
reason_counter=collections.Counter()
for r in et:
    for k,v in (r.get('reason_flags') or {}).items():
        if v: reason_counter[k]+=1
print('reason_flags_top', dict(reason_counter.most_common(20)))

by=collections.defaultdict(list)
for r in et:
    by[norm_enc(r.get('encounter_id'))].append(r)
print('\nencounter                                      rows offpool forced waste strategic unknown transient avgE avgLegal avgPlayable avgPositive avgUrgent avgDefer avgX0 avgPot avgKaiserRisk')
for enc, rows in sorted(by.items(), key=lambda kv: len(kv[1]), reverse=True):
    c=collections.Counter(r.get('end_turn_class') for r in rows)
    trans=sum(1 for r in rows if gp(r,'reason_flags.transient_only_end_turn'))
    print(f'{enc:44s} {len(rows):5d} {str(enc not in POOL):7s} {c.get("forced_end_turn",0):6d} {c.get("wasteful_end_turn",0):5d} {c.get("strategic_defer_end_turn",0):9d} {c.get("unknown",0):7d} {trans:9d} {avg(rows,"player.energy"):5.2f} {avg(rows,"counts.legal_action_count"):8.2f} {avg(rows,"counts.playable_cards_left"):11.2f} {avg(rows,"counts.positive_action_count"):11.2f} {avg(rows,"counts.urgent_action_count"):9.2f} {avg(rows,"counts.deferable_action_count"):8.2f} {avg(rows,"counts.zero_energy_x_cost_candidate_count"):6.2f} {avg(rows,"counts.potion_available_count"):6.2f} {avg(rows,"boss_context.kaiser_back_attack_risk"):12.3f}')

# suspicious end-turn: energy and legal non-endturn or positive/playable. Include all flags.
sus=[]
for r in et:
    energy=num(gp(r,'player.energy'))
    legal=num(gp(r,'counts.legal_action_count'))
    pos=num(gp(r,'counts.positive_action_count'))
    play=num(gp(r,'counts.playable_cards_left'))
    urgent=num(gp(r,'counts.urgent_action_count'))
    if energy > 0.05 and legal > 1.05 and (pos > 0.05 or urgent > 0.05 or play > 0.05):
        sus.append(r)
print('\nSUSPICIOUS_END_TURN energy>0 & legal>1 & (positive/urgent/playable)>0:', len(sus), f'({pct(len(sus),len(et)):.2f}% of end_turn contexts)')
sby=collections.defaultdict(list)
for r in sus: sby[norm_enc(r.get('encounter_id'))].append(r)
for enc, rows in sorted(sby.items(), key=lambda kv: len(kv[1]), reverse=True):
    c=collections.Counter(r.get('end_turn_class') for r in rows)
    print(f'  {enc:44s} n={len(rows):4d} class={dict(c)} avgE={avg(rows,"player.energy"):.2f} avgLegal={avg(rows,"counts.legal_action_count"):.2f} avgPlay={avg(rows,"counts.playable_cards_left"):.2f} avgPos={avg(rows,"counts.positive_action_count"):.2f} avgDef={avg(rows,"counts.deferable_action_count"):.2f}')
print('last_suspicious_samples:')
for r in sus[-12:]:
    tops=[]
    for a in r.get('top_legal_actions') or []:
        tops.append(f"{a.get('rank')}:{a.get('family')}:{a.get('title')}:{a.get('score')}")
    print(' ', r.get('global_step'), r.get('episode_id'), norm_enc(r.get('encounter_id')), r.get('end_turn_class'), 'E', gp(r,'player.energy'), 'legal', gp(r,'counts.legal_action_count'), 'play', gp(r,'counts.playable_cards_left'), 'pos', gp(r,'counts.positive_action_count'), 'defer', gp(r,'counts.deferable_action_count'), 'reasons', r.get('reason_flags'), 'top', tops[:5])

# action offenders
ao,bad,p = load_jsonl('action_offenders.jsonl')
print('\n## ACTION_OFFENDERS', 'rows', len(ao), 'bad', bad, 'size', p.stat().st_size if p else None, 'mtime', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(p.stat().st_mtime)) if p else None)
print('global_step range', (min((r.get('global_step',0) for r in ao), default=None), max((r.get('global_step',0) for r in ao), default=None)), 'episodes', (min((r.get('episode_id',0) for r in ao), default=None), max((r.get('episode_id',0) for r in ao), default=None)))
print('offpool', sum(norm_enc(r.get('encounter_id')) not in POOL for r in ao))
print('offender_type_counts', dict(collections.Counter(r.get('offender_type') for r in ao).most_common(30)))
aoby=collections.defaultdict(collections.Counter)
for r in ao:
    aoby[norm_enc(r.get('encounter_id'))][r.get('offender_type')] += 1
print('offenders_by_encounter:')
for enc,c in sorted(aoby.items(), key=lambda kv: sum(kv[1].values()), reverse=True):
    print(f'  {enc:44s} total={sum(c.values()):5d} offpool={str(enc not in POOL):5s} {dict(c.most_common(10))}')
# recent offenders last 1000 rows by type/enc
recent = ao[-1000:]
print('recent1000 offender counts', dict(collections.Counter(r.get('offender_type') for r in recent).most_common(20)))
recent_by=collections.defaultdict(collections.Counter)
for r in recent:
    recent_by[norm_enc(r.get('encounter_id'))][r.get('offender_type')] += 1
print('recent1000 by_encounter:')
for enc,c in sorted(recent_by.items(), key=lambda kv: sum(kv[1].values()), reverse=True):
    print(f'  {enc:44s} total={sum(c.values()):4d} {dict(c.most_common(8))}')

# kaiser facing legal actions
kf,bad,p = load_jsonl('kaiser_facing_legal_actions.jsonl')
print('\n## KAISER_FACING_LEGAL_ACTIONS', 'rows', len(kf), 'bad', bad, 'size', p.stat().st_size if p else None, 'mtime', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(p.stat().st_mtime)) if p else None)
if kf:
    change_counts=[]; explicit_counts=[]; target_counts=[]; risk=[]
    for r in kf:
        acts=r.get('actions') or []
        change_counts.append(sum(1 for a in acts if a.get('changes_facing')))
        explicit_counts.append(sum(1 for a in acts if a.get('explicit_facing')))
        target_counts.append(sum(1 for a in acts if a.get('target_back_attack_position') or a.get('target_side_or_back_attack_position') in ('left','right')))
        risk.append(num(r.get('risk')))
    print('risk_avg/max', round(sum(risk)/len(risk),3), max(risk), 'change_candidates avg/p50/max', round(sum(change_counts)/len(change_counts),3), q(change_counts,.5), max(change_counts), 'explicit avg/max', round(sum(explicit_counts)/len(explicit_counts),3), max(explicit_counts), 'target_position avg/max', round(sum(target_counts)/len(target_counts),3), max(target_counts))
    print('last_kaiser_sample', {k:kf[-1].get(k) for k in ['episode_count','total_steps','risk','player_facing','combat_facing_raw']}, 'actions', len(kf[-1].get('actions') or []), 'changes', sum(1 for a in (kf[-1].get('actions') or []) if a.get('changes_facing')))

# potions
pt,bad,p = load_jsonl('potion_transitions.jsonl')
print('\n## POTION_TRANSITIONS', 'rows', len(pt), 'bad', bad, 'size', p.stat().st_size if p else None, 'mtime', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(p.stat().st_mtime)) if p else None)
if pt:
    events=collections.Counter(r.get('event') for r in pt)
    print('events', dict(events))
    use=[r for r in pt if r.get('event')=='use_potion_transition']
    sticky=[]; fail=[]
    for r in use:
        if not r.get('execute_ok'): fail.append(r)
        after=r.get('potion_slot_after') or {}
        before_id=r.get('potion_id_before')
        if before_id and (not after.get('empty')) and str(after.get('id'))==str(before_id):
            sticky.append(r)
    print('use_count', len(use), 'execute_fail', len(fail), 'slot_still_same_after_use', len(sticky), f'({pct(len(sticky),len(use)):.1f}%)')
    print('potion_before_top', dict(collections.Counter(r.get('potion_id_before') for r in use).most_common(20)))
    for r in sticky[-5:]:
        print(' sticky_sample', r.get('global_step'), r.get('episode_id'), r.get('action_id'), r.get('potion_id_before'), 'after', r.get('potion_slot_after'))

# loss spikes
ls,bad,p = load_jsonl('loss_spikes.jsonl')
print('\n## LOSS_SPIKES', 'rows', len(ls), 'bad', bad, 'size', p.stat().st_size if p else None, 'mtime', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(p.stat().st_mtime)) if p else None)
for r in ls:
    losses=r.get('losses') or {}
    trig=r.get('trigger') or {}
    print(' spike', 'steps', r.get('total_steps') or r.get('step'), 'step_k', r.get('step_k'), 'dominant', trig.get('dominant_key') or r.get('dominant'), 'value', trig.get('dominant_value') or r.get('value'), 'tier', r.get('sample_tier'), 'action_dist', r.get('action_index_dist'), 'quarantine', r.get('quarantine_reason') or r.get('reasons'))
    # summarize batch snapshots if present
    for key in ['batch_summary','samples','sample_summary']:
        if key in r:
            print('  ', key, str(r[key])[:500])
