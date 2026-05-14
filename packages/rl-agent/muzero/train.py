"""Compatibility entrypoint for MuZero STS2 training.

The former monolithic trainer was split into ``muzero.training.trainer`` and
smaller ``muzero.training.*`` modules.  Keep this file thin so
``python -m muzero.train`` and legacy imports such as
``from muzero.train import MuZeroTrainer`` keep working without reopening the
10k+ line monolith.
"""

from __future__ import annotations

# Re-export the historical public namespace used by older tests/scripts.
from muzero.training.trainer import *  # noqa: F401,F403
from muzero.training.trainer import _get_live_supported_encounter_ids as _get_live_supported_encounter_ids


def main():
    """Run the MuZero training CLI."""
    from muzero.training.cli_main import main as _cli_main

    return _cli_main()


if __name__ == "__main__":
    main()
