"""Legacy wrapper for the archived v2 evaluator.

Mainline online evaluation now lives in ``evaluate_attention_policy.py``.
"""

from __future__ import annotations

from legacy.evaluate import *  # noqa: F401,F403


if __name__ == "__main__":
    from legacy.evaluate import main

    main()
