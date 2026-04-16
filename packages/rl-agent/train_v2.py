"""Legacy wrapper for the archived routed PPO trainer.

Mainline online training now lives in ``train_attention_policy.py``.
"""

from __future__ import annotations

from legacy.train_v2 import *  # noqa: F401,F403
from legacy.train_v2 import main


if __name__ == "__main__":
    main()
