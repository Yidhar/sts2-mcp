# STS2 Game Data

This package is the only supported source of generated card, relic, potion,
enemy, and effect-profile data used by the bridge, MCP server, and RL trainer.

- `raw/` contains small curated inputs and overrides.
- `generated/` contains deterministic generated artifacts.
- `manifest.json` deterministically records the exact file set, byte sizes,
  hashes, generator identity, and pinned upstream data commit. Wall-clock time,
  Git dirty state, and the release commit are intentionally kept out of this
  self-contained source manifest; release CI records them in an external
  attestation instead.

Consumers must resolve this package through `STS2_GAME_DATA_ROOT` or the
repository layout. They must not import data through an RL package path.

Run `python tools/game_data/build_manifest.py` after changing any data file.
`python tools/game_data/verify_manifest.py` also rejects unmanifested extra files.
