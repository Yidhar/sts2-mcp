"""Legacy wrapper for the archived multitask offline trainer.

This path is archived because it still targets the routed v2 policy stack.
"""

from __future__ import annotations

from legacy.train_offline_multitask import *  # noqa: F401,F403
from legacy.train_offline_multitask import main


if __name__ == "__main__":
    main()
