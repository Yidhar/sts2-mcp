# Human Demo JSONL Format

This file documents the decision-row schema consumed by
`muzero.demo_dataset`.  A human demonstration dataset is an append-only JSONL
file where **each line is one decision** made from the exact legal-action list
visible to the policy.

The recorder added in `sts2_env/human_demo_recorder.py` writes this format to:

```text
<STS2_ARTIFACT_ROOT>/human_demos/<session_id>/decisions.jsonl
```

Sidecar episode lifecycle rows are written separately to `episodes.jsonl` so
`decisions.jsonl` remains strict-loader compatible.

The Bridge v2 wire contract executes the opaque `action_handle`; it does not
accept `action_id`. The recorder sees the environment's normalized legal-action
records after the v2 response is decoded and stores `action_id` as an internal
policy/replay identity. `selected_action_id` therefore labels the chosen
normalized record; the live environment resolves that record's
`action_handle` when it sends a command to the Bridge.

## Required decision fields

```json
{
  "version": 1,
  "source": "human",
  "timestamp": "2026-05-07T05:20:00.000000+00:00",
  "session_id": "human_20260507_132000_ab12cd34",
  "episode_id": "bridge_episode_id",
  "encounter_id": "kaiser_crab_boss",
  "tier": "boss",
  "turn": 3,
  "step_in_turn": 2,
  "step_in_episode": 12,
  "obs": {},
  "legal_actions": [
    {
      "action_id": "play_card:...",
      "family": "play_card",
      "card": {},
      "semantic": {},
      "runtime": {}
    }
  ],
  "selected_action_id": "play_card:...",
  "selected_action_index": 0,
  "selected_action": {},
  "reason_tags": ["avoid_back_attack", "block_incoming"],
  "comment": "optional human note",
  "reward": 0.0,
  "done": false,
  "truncated": false,
  "outcome": {}
}
```

`muzero.demo_dataset` requires:

- `version` is absent or `1`.
- `episode_id` is present.
- `encounter_id` is present.
- `obs` is a JSON object.
- `legal_actions` is a non-empty array.
- `selected_action_id` appears in `legal_actions[*].action_id`.

## Runtime card-state expectations

Recent card-mechanism fixes rely on bridge/runtime identity fields.  Demos are
most useful when selected card actions preserve:

- `card.instance_uuid` or equivalent runtime identity
- modified/current cost fields
- exhaust / ethereal / retain / replay flags
- selection-effect metadata
- enchantment metadata

The validator reports coverage for these fields but does not fail by default:

```powershell
if (-not $env:STS2_ARTIFACT_ROOT) { throw 'Set STS2_ARTIFACT_ROOT to an absolute external directory.' }
$python = Join-Path $env:STS2_ARTIFACT_ROOT 'environments\windows-python\Scripts\python.exe'
$demo = Join-Path $env:STS2_ARTIFACT_ROOT 'human_demos\<session>\decisions.jsonl'
& $python .\packages\rl-agent\scripts\validate_human_demos.py $demo
```

Use `--strict-runtime` only when auditing a bridge build that is expected to
emit runtime state on every selected card action.

## Reason tags

The loader accepts arbitrary strings, but the current known tags are:

- `lethal`
- `block_lethal`
- `avoid_back_attack`
- `change_facing`
- `use_stun_window`
- `save_exhaust_card`
- `play_exhaust_now`
- `refund_followup`
- `avoid_refund_no_followup`
- `use_potion_now`
- `save_potion`
- `setup_next_turn`
- `cycle_control`
- `block_incoming`

Tags are advisory supervision signals; the hard policy-imitation label remains
`selected_action_id`.
