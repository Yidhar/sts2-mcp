import json, glob, os, collections, math
base='logs_muzero/muzero_act1_sandbox_hallway_cq_xcostguard_20260511_201009/diagnostics/death_slices'
rows=[]
for fp in glob.glob(base+'/*.jsonl'):
    enc=os.path.basename(fp).replace('.jsonl','')
    with open(fp,encoding='utf-8') as f:
        for ln in f:
            if ln.strip():
                try: rows.append((enc,json.loads(ln)))
                except Exception as e: print('bad',fp,e)
print('total_deaths',len(rows))
print('by_file')
for enc,c in collections.Counter(r[0] for r in rows).most_common(20): print(enc,c)
print('by_encounter_id')
for enc,c in collections.Counter((d.get('encounter_id') or enc) for enc,d in rows).most_common(20): print(enc,c)
print('last_12')
for enc,d in rows[-12:]:
    print('\n---',enc, {k:d.get(k) for k in ['episode_id','encounter_id','encounter_tier','episode_steps','snapshot_floor_number','metadata_subset']})
    for i,step in enumerate(d.get('tail_steps') or []):
        ss=step.get('search_stats') or {}
        ai=step.get('action_info') or {}
        vals={k:ss[k] for k in sorted(ss) if ss[k] and any(s in k for s in ['potion','late_normal','x_cost','wasteful_end','survival','block_waste','no_damage','hp_cost'])}
        if vals:
            print(i, ai.get('action_id'), vals)
