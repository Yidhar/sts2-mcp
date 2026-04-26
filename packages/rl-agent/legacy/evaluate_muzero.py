"""Legacy import path wrapper for the MuZero evaluator.

Prefer new automation to use ``python -m muzero.evaluate``.
"""

from __future__ import annotations

from muzero.evaluate import *  # noqa: F401,F403
from muzero.evaluate import main


if __name__ == "__main__":
    main()
