"""Compatibility wrapper for the MuZero replay analysis utility.

Prefer new automation to use ``python -m muzero.analyze_replay``.
"""

from __future__ import annotations

from muzero.analyze_replay import *  # noqa: F401,F403
from muzero.analyze_replay import main


if __name__ == "__main__":
    main()
