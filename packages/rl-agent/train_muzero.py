"""Legacy wrapper for the archived MuZero training stack."""

from __future__ import annotations

from legacy.train_muzero import *  # noqa: F401,F403
from legacy.train_muzero import main


if __name__ == "__main__":
    main()
