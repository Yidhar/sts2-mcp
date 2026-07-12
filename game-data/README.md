# STS2 Game Data

This package is the canonical offline source of policy-free static card, relic,
and potion facts. Runtime Bridge observations come directly from game state.
Version 2 deleted the scored upstream card catalog and the hand-curated card
override file; generated card records retain factual IDs/text/type/cost/keyword
and upgrade data only.

- `raw/` is reserved for authoritative, non-policy source inputs; it is empty in
  the current baseline.
- `generated/` contains deterministic generated artifacts.
- `manifest.json` deterministically records the exact file set, byte sizes,
  hashes, generator identity, and pinned upstream data commit. Wall-clock time,
  Git dirty state, and the release commit are intentionally kept out of this
  self-contained source manifest; release CI records them in an external
  attestation instead.

Consumers must resolve this package through `STS2_GAME_DATA_ROOT` or the
repository layout. They must not import data through an RL package path.

The following fields are forbidden at every nesting level: scores, priors,
quality/keep decisions, task summaries, strategy labels, semantic tags, and
semantic signals. They are neither model inputs nor checkpoint compatibility
requirements for the grounded baseline.

Run `python tools/game_data/build_manifest.py` after changing any data file.
`python tools/game_data/verify_manifest.py` also rejects unmanifested extra files.
