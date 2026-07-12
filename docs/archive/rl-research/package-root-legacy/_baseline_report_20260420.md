# Phase 8 Baseline Report
Generated: 2026-04-20 from reset_events.jsonl across all sim_phase8_* training runs.

## Summary table

| Run | Episodes | Mean floor | Max floor | Boss touch | Boss kill | Stuck rate |
|---|---|---|---|---|---|---|
| smoke50k | 160 | 5.67 | 14 | 0 (0.0%) | 0 (0.0%) | 74 (46.2%) |
| smoke10k | 79 | 7.73 | 17 | 1 (1.3%) | 0 (0.0%) | 0 (0.0%) |
| long820k | 2230 | 7.82 | 18 | 45 (2.0%) | 1 (0.0%) | 142 (6.4%) |
| rshape200k | 288 | 8.29 | 18 | 9 (3.1%) | 1 (0.3%) | 12 (4.2%) |
| potionv3 | 49 | 8.73 | 17 | 2 (4.1%) | 0 (0.0%) | 2 (4.1%) |
| live2k | 16 | 5.81 | 8 | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |

## Per-run detail

### Phase 8 smoke 50k (pre-patch) (`smoke50k`)
- episodes: **160**
- mean episode reward: **-0.485**, max: +0.000
- floor distribution (mean=5.67, max=14):
  - 1:27, 2:2, 3:8, 4:9, 5:28, 6:29, 7:19, 8:17, 9:5, 11:7, 12:6, 13:2, 14:1
- floor ≥10: **16/160 (10.0%)**, ≥17 (boss touch): **0/160 (0.0%)**, ≥18 (boss kill): **0/160 (0.0%)**
- truncation breakdown:
  - natural: 86/160 (53.8%)
  - phase_stuck_watchdog: 74/160 (46.2%)
- stuck_phase breakdown:
  - card_selection: 58
  - combat: 16
- telemetry: ❌ not captured (run pre-dates commit 64e7417)


### Phase 8 smoke 10k (cd791a0 card_sel fix) (`smoke10k`)
- episodes: **79**
- mean episode reward: **-0.871**, max: +0.170
- floor distribution (mean=7.73, max=17):
  - 2:1, 3:1, 5:11, 6:16, 7:15, 8:14, 9:9, 11:4, 13:3, 14:3, 15:1, 17:1
- floor ≥10: **12/79 (15.2%)**, ≥17 (boss touch): **1/79 (1.3%)**, ≥18 (boss kill): **0/79 (0.0%)**
- truncation breakdown:
  - natural: 79/79 (100.0%)
- telemetry: ❌ not captured (run pre-dates commit 64e7417)


### Phase 8 longtrain 820k baseline (`long820k`)
- episodes: **2230**
- mean episode reward: **-0.827**, max: +1.050
- floor distribution (mean=7.82, max=18):
  - 2:33, 3:18, 4:36, 5:276, 6:479, 7:420, 8:315, 9:248, 11:124, 12:92, 13:49, 14:57, 15:38, 17:44, 18:1
- floor ≥10: **405/2230 (18.2%)**, ≥17 (boss touch): **45/2230 (2.0%)**, ≥18 (boss kill): **1/2230 (0.0%)**
- truncation breakdown:
  - natural: 2088/2230 (93.6%)
  - phase_stuck_watchdog: 142/2230 (6.4%)
- stuck_phase breakdown:
  - combat: 141
  - actions: 1
- telemetry: ❌ not captured (run pre-dates commit 64e7417)


### Phase 8 resume 200k reward-shape v1 (`rshape200k`)
- episodes: **288**
- mean episode reward: **-0.698**, max: +7.540
- floor distribution (mean=8.29, max=18):
  - 2:2, 4:1, 5:25, 6:68, 7:45, 8:51, 9:31, 11:19, 12:17, 13:7, 14:8, 15:5, 17:8, 18:1
- floor ≥10: **65/288 (22.6%)**, ≥17 (boss touch): **9/288 (3.1%)**, ≥18 (boss kill): **1/288 (0.3%)**
- truncation breakdown:
  - natural: 276/288 (95.8%)
  - phase_stuck_watchdog: 12/288 (4.2%)
- stuck_phase breakdown:
  - combat: 12
- **telemetry** (Phase 8.2 counter fields):
  | field | sum | mean/ep | max | n |
  |---|---|---|---|---|
  | `potion_use_count` | sum=443.0 | mean/ep=2.120 | max=6.0 | n=209 |
  | `potion_use_boss_count` | sum=2.0 | mean/ep=0.010 | max=1.0 | n=209 |
  | `potion_use_elite_count` | sum=0.0 | mean/ep=0.000 | max=0.0 | n=209 |
  | `potion_use_bonus_total` | sum=44.7 | mean/ep=0.214 | max=0.8 | n=209 |
  | `potion_discard_count` | sum=0.0 | mean/ep=0.000 | max=0.0 | n=209 |
  | `rest_site_encounters` | sum=6106.0 | mean/ep=29.215 | max=441.0 | n=209 |
  | `rest_heal_chosen` | sum=6106.0 | mean/ep=29.215 | max=441.0 | n=209 |
  | `rest_skip_heal_chosen` | sum=0.0 | mean/ep=0.000 | max=0.0 | n=209 |
  | `rest_skip_heal_at_low_hp` | sum=0.0 | mean/ep=0.000 | max=0.0 | n=209 |
  | `rest_penalty_total` | sum=0.0 | mean/ep=0.000 | max=0.0 | n=209 |
  | `boss_damage_dealt_raw` | sum=1311.0 | mean/ep=6.273 | max=265.0 | n=209 |
  | `boss_damage_bonus_total` | sum=52.4 | mean/ep=0.251 | max=10.6 | n=209 |
  | `boss_encounter_steps` | sum=238.0 | mean/ep=1.139 | max=60.0 | n=209 |
  | `floor_clear_reward_total` | sum=48.3 | mean/ep=0.231 | max=4.1 | n=209 |
  | `floor_clear_events` | sum=121.0 | mean/ep=0.579 | max=7.0 | n=209 |
  | `boss_floor_entry_events` | sum=6.0 | mean/ep=0.029 | max=1.0 | n=209 |


### Phase 8 resume potion v3 (84f8ca6) (`potionv3`)
- episodes: **49**
- mean episode reward: **-0.581**, max: +7.390
- floor distribution (mean=8.73, max=17):
  - 4:1, 5:1, 6:7, 7:10, 8:10, 9:8, 11:4, 12:4, 14:1, 15:1, 17:2
- floor ≥10: **12/49 (24.5%)**, ≥17 (boss touch): **2/49 (4.1%)**, ≥18 (boss kill): **0/49 (0.0%)**
- truncation breakdown:
  - natural: 47/49 (95.9%)
  - phase_stuck_watchdog: 2/49 (4.1%)
- stuck_phase breakdown:
  - combat: 2
- **telemetry** (Phase 8.2 counter fields):
  | field | sum | mean/ep | max | n |
  |---|---|---|---|---|
  | `potion_use_count` | sum=107.0 | mean/ep=2.184 | max=6.0 | n=49 |
  | `potion_use_boss_count` | sum=2.0 | mean/ep=0.041 | max=1.0 | n=49 |
  | `potion_use_elite_count` | sum=0.0 | mean/ep=0.000 | max=0.0 | n=49 |
  | `potion_use_bonus_total` | sum=5.9 | mean/ep=0.119 | max=0.6 | n=49 |
  | `potion_discard_count` | sum=0.0 | mean/ep=0.000 | max=0.0 | n=49 |
  | `rest_site_encounters` | sum=1769.0 | mean/ep=36.102 | max=195.0 | n=49 |
  | `rest_heal_chosen` | sum=1769.0 | mean/ep=36.102 | max=195.0 | n=49 |
  | `rest_skip_heal_chosen` | sum=0.0 | mean/ep=0.000 | max=0.0 | n=49 |
  | `rest_skip_heal_at_low_hp` | sum=0.0 | mean/ep=0.000 | max=0.0 | n=49 |
  | `rest_penalty_total` | sum=0.0 | mean/ep=0.000 | max=0.0 | n=49 |
  | `boss_damage_dealt_raw` | sum=487.0 | mean/ep=9.939 | max=265.0 | n=49 |
  | `boss_damage_bonus_total` | sum=19.5 | mean/ep=0.398 | max=10.6 | n=49 |
  | `boss_encounter_steps` | sum=63.0 | mean/ep=1.286 | max=38.0 | n=49 |
  | `floor_clear_reward_total` | sum=13.9 | mean/ep=0.284 | max=3.8 | n=49 |
  | `floor_clear_events` | sum=33.0 | mean/ep=0.673 | max=6.0 | n=49 |
  | `boss_floor_entry_events` | sum=2.0 | mean/ep=0.041 | max=1.0 | n=49 |
  | `potion_hoarding_unused_at_end` | sum=0.0 | mean/ep=0.000 | max=0.0 | n=49 |
  | `potion_hoarding_penalty_total` | sum=0.0 | mean/ep=0.000 | max=0.0 | n=49 |


### Phase 8 live smoke (real game) (`live2k`)
- episodes: **16**
- mean episode reward: **-2.598**, max: -1.999
- floor distribution (mean=5.81, max=8):
  - 4:4, 5:1, 6:7, 7:2, 8:2
- floor ≥10: **0/16 (0.0%)**, ≥17 (boss touch): **0/16 (0.0%)**, ≥18 (boss kill): **0/16 (0.0%)**
- truncation breakdown:
  - natural: 16/16 (100.0%)
- telemetry: ❌ not captured (run pre-dates commit 64e7417)
