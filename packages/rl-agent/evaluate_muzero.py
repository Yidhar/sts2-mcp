"""Legacy wrapper for the archived MuZero evaluator."""

from __future__ import annotations

from legacy.evaluate_muzero import *  # noqa: F401,F403


if __name__ == "__main__":
    from legacy.evaluate_muzero import main

    main()
