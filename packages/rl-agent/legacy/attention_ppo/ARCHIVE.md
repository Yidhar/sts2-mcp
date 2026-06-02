# legacy/attention_ppo — archived PPO / omni-attention path

This directory holds the **AuxMaskablePPO + `STS2OmniAttentionPolicy`** training
path, archived 2026-06-02. It is **superseded by the MuZero token-memory
search-free path** under `muzero/`, which is the single active training target.

## Why it was archived

- Last meaningful code change: 2026-04-22. MuZero has been the active path since.
- `CLAUDE.md` previously (incorrectly) called this the "active" path; it is not.
- Control-experiment value: this path reached **~0.68 resolved win-rate /
  0.375 boss win on curated mid-run combat decks**, but **0/100 on starter
  full-run** — the *same* passive-collapse / 0%-win failure as MuZero. Two
  independent learners failing identically on the same env is the key evidence
  that the root cause is the **shared env / reward / training regimen / guards**,
  not the learning algorithm. See `../../_analysis_rl_diagnosis_20260602.md`.

## Contents

- `train_attention_policy.py`, `evaluate_attention_policy.py` — entrypoints
- `aux_maskable_ppo.py` — `AuxMaskablePPO` + async rollout buffer
- `omni_attention_policy.py` — `STS2OmniAttentionPolicy`
- `test_aux_attention_training.py`, `test_omni_attention_policy_static.py` —
  frozen-contract tests (no longer collected by `pytest tests/`)

## What stayed in `sts2_env/` (shared, still active)

- `attention_blocks.py` — imported by `muzero/sts2_env/token_memory.py`
- `aux_targets.py` — shared auxiliary-target builder
- `checkpoint.py` — attention-format "online checkpoint" loader, still used by
  several `probe_*.py` / `eval_*.py` diagnostic scripts; its PPO-class imports
  now point here (`legacy.attention_ppo.*`).

To run anything here, invoke from the repo package root
(`packages/rl-agent`) so that `legacy.attention_ppo.*` and `sts2_env.*`
both resolve.
