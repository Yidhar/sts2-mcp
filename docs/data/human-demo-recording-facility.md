# Human Manual-Play Recording Facility

## Goal

When MuZero gets stuck in a bad habit window, we need a low-friction way to
record high-quality human decisions and use them for imitation / replay
curriculum.  The recorder must capture **what the model actually sees**:

- raw bridge observation before the action
- full legal-action list before the action
- normalized internal action id and index (the v2 command itself uses the
  selected record's opaque `action_handle`)
- runtime card state and semantic fields already present on the action payload
- reward / terminal outcome metadata

This avoids training on vague text replays and prevents the common failure mode
where a human label points to an action id that was not legal in that state.

## Files

| File | Purpose |
|---|---|
| `sts2_env/human_demo_recorder.py` | Append-only JSONL writer; reusable from env wrappers and scripts. |
| `scripts/record_human_combat_demo.py` | Terminal manual-play tool: shows legal actions, asks for action index, records demos. |
| `scripts/validate_human_demos.py` | Strict schema validator plus runtime-card-state coverage report. |
| `docs/data/human-demo-format.md` | Decision JSONL schema consumed by `muzero.demo_dataset`. |
| `muzero/demo_dataset.py` | Existing loader/batch builder for imitation data. |

## Basic usage

Start / keep the STS2 bridge running, then run:

```powershell
Set-Location '<REPOSITORY_ROOT>'

if (-not $env:STS2_ARTIFACT_ROOT) { throw 'Set STS2_ARTIFACT_ROOT to an absolute external directory.' }
$python = Join-Path $env:STS2_ARTIFACT_ROOT 'environments\windows-python\Scripts\python.exe'
& $python .\packages\rl-agent\scripts\record_human_combat_demo.py `
  --encounter-id kaiser_crab_boss `
  --character ironclad `
  --current-hp 70 `
  --max-hp 80 `
  --max-energy 3
```

The script prints the current combat state and numbered legal actions:

```text
Legal actions:
  00  play_card | play_card:... | card=Strike | target=...
  01! play_card | play_card:... | card=Bloodletting | ...
  02  end_turn | end_turn
```

- Enter a number to choose that action.
- `r` redraws the state.
- `tag avoid_back_attack,block_incoming` attaches reason tags to the next
  decision.
- `note ...` attaches a free-form note to the next decision.
- `q` exits the manual session.

`!` means the current action mask blocks the action (for example self-lethal
HP-cost card).  It is still shown because the legal-action contract may expose
it, but human demos should normally avoid selecting it unless we intentionally
want a negative/edge-case replay.

## Output

Default output directory:

```text
<STS2_ARTIFACT_ROOT>/human_demos/<session_id>/
```

Files:

```text
manifest.json
decisions.jsonl
episodes.jsonl
```

Only `decisions.jsonl` is fed to `muzero.demo_dataset`.  `episodes.jsonl` is a
sidecar for operator audits and does not interfere with strict decision-row
loading.

## Validate a session

```powershell
if (-not $env:STS2_ARTIFACT_ROOT) { throw 'Set STS2_ARTIFACT_ROOT to an absolute external directory.' }
$python = Join-Path $env:STS2_ARTIFACT_ROOT 'environments\windows-python\Scripts\python.exe'
$demo = Join-Path $env:STS2_ARTIFACT_ROOT 'human_demos\<session_id>'
& $python .\packages\rl-agent\scripts\validate_human_demos.py $demo
```

The validator confirms:

1. every row is valid JSON;
2. every required field exists;
3. `selected_action_id` is present in `legal_actions`;
4. selected and legal card actions expose runtime fields often needed for
   exhaust / retain / replay / enchantment / dynamic-cost mechanics.

## Env-var auto recorder

For custom human-policy wrappers that already call `CombatSandboxEnv.step()`,
enable recording without changing wrapper code:

```powershell
if (-not $env:STS2_ARTIFACT_ROOT) { throw 'Set STS2_ARTIFACT_ROOT to an absolute external directory.' }
$env:STS2_HUMAN_DEMO_RECORD = "1"
$env:STS2_HUMAN_DEMO_DIR = Join-Path $env:STS2_ARTIFACT_ROOT 'human_demos'
$env:STS2_HUMAN_DEMO_SESSION = "human_kaiser_focus_001"
```

Then instantiate and drive `CombatSandboxEnv` normally.  The env hook records
each `step()` transition.  Do **not** set this env var during MuZero self-play
training unless you intentionally want the model's actions recorded as a
`source=human` dataset.

## Recommended collection protocol

To break the current plateau, collect focused sessions instead of random full
runs:

1. **Kaiser / facing**: force human decisions around back-attack risk,
   pressure kill vs turn-facing actions, and risky end-turn avoidance.
2. **Insatiable / sandpit**: demonstrate `Frantic Escape` timing when countdown
   is `1`, `<3`, and when extending the countdown is better than immediate
   escape.
3. **HP-cost cards**: demonstrate when Bloodletting-like refund cards are worth
   paying HP and when they should be skipped because there is no follow-up.
4. **Potion timing**: demonstrate high-urgency potion use and deliberate potion
   saving in low-urgency states.
5. **Card-selection screens**: demonstrate non-looping select/confirm behavior
   for Purge-like, replace-hand, retain-card, exhaust/transform/copy effects.

After each session:

```powershell
if (-not $env:STS2_ARTIFACT_ROOT) { throw 'Set STS2_ARTIFACT_ROOT to an absolute external directory.' }
$python = Join-Path $env:STS2_ARTIFACT_ROOT 'environments\windows-python\Scripts\python.exe'
$demo = Join-Path $env:STS2_ARTIFACT_ROOT 'human_demos\<session_id>'
& $python .\packages\rl-agent\scripts\validate_human_demos.py $demo
```

Only promote validated `decisions.jsonl` files into imitation training.
