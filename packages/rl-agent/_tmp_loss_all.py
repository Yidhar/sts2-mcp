from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import statistics
RUN=Path('logs_muzero/muzero_boss_recovery_20260504_132317')
ea=EventAccumulator(str(RUN), size_guidance={'scalars':0}); ea.Reload(); tags=set(ea.Tags()['scalars'])
for tag in ['loss/total','loss/future_world_aux','loss/future_bank_state','loss/future_bank_delta','loss/surprise','loss/policy','loss/value','loss/reward']:
    xs=ea.Scalars(tag); vals=[x.value for x in xs]
    im=max(range(len(vals)), key=lambda i: vals[i])
    print(f'{tag:28s} count={len(vals)} all_max={vals[im]:.6g}@{xs[im].step} all_med={statistics.median(vals):.6g} last={vals[-1]:.6g}@{xs[-1].step}')
