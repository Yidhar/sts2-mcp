# Documentation archive

This directory preserves obsolete architecture, implementation, experiment, and
migration context. Archived material is **non-normative** and may describe APIs,
entry points, paths, metrics, or safety assumptions that no longer exist.

`v1/architecture.md` and `v1/implementation-plan.md` are the pre-architecture-v2
documents preserved during the 2026-07-11 cleanup. They must not be used to:

- implement new newline-delimited hand-written MCP transport;
- extend legacy Bridge endpoints;
- select PPO `train_pipeline.py` as the maintained trainer;
- expose hidden/debug state through the normal player capability;
- retry non-idempotent game mutations;
- store runtime artifacts in the checkout.

For current behavior, start at [`../README.md`](../README.md),
[`../architecture.md`](../architecture.md), and
[`../migration/README.md`](../migration/README.md).
