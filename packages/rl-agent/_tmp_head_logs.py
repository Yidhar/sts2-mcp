from pathlib import Path
RUN=Path('logs_muzero/muzero_boss_recovery_20260504_132317')
for f in ['diagnostics/end_turn_contexts.jsonl','diagnostics/action_offenders.jsonl','diagnostics/loss_spikes.jsonl','diagnostics/potion_transitions.jsonl','diagnostics/kaiser_facing_legal_actions.jsonl']:
    p=RUN/f
    print('---', f, 'exists', p.exists(), 'size', p.stat().st_size if p.exists() else None)
    if p.exists() and p.stat().st_size:
        with p.open('r', encoding='utf-8') as fh:
            print(fh.readline()[:2000].rstrip())
