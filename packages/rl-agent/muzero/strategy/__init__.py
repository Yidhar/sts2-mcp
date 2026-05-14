"""Pure MuZero combat/build decision strategy helpers.

This package is for reusable, testable policy logic that should not live in the
training orchestrator:

* action/card feature extraction
* combat-state summaries used by multiple guards
* potion timing and HP/X-cost/card-state policies
* encounter-specific strategy adapters under ``strategy.encounters``

Modules here should be leaf logic: they may depend on ``sts2_env`` payload
types and standard libraries, but should not import ``muzero.train``.
"""

__all__ = ["action_features"]
