# ADR 0004: Reward is owned exclusively by the RL package

- Status: accepted
- Date: 2026-07-11

The bridge and headless adapters emit canonical transition facts and never add
training reward. A pure, versioned reward calculator in the RL package consumes
those facts. Replay and checkpoint metadata record its version and hash.
The replay decision payload carries the complete reward/projection fingerprint;
standalone replay files are unsupported, and atomic checkpoints additionally bind the
fingerprint to hashed replay bytes.
