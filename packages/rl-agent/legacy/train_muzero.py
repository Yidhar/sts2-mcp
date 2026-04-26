"""Legacy import path wrapper for the MuZero training stack.

Prefer new automation to use ``python -m muzero.train``.
"""

from __future__ import annotations

from muzero.train import *  # noqa: F401,F403
from muzero.train import main


if __name__ == "__main__":
    main()
