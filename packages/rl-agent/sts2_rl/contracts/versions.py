"""Contract versions loaded from the repository source or generated package copy."""

from __future__ import annotations

import runpy
from pathlib import Path

_packaged = Path(__file__).with_name("_generated_versions.py")
_repository = Path(__file__).resolve().parents[4] / "contracts" / "generated" / "python" / "versions.py"
_source = _repository if _repository.is_file() else _packaged
_values = runpy.run_path(str(_source))

API_VERSION = str(_values["API_VERSION"])
SCHEMA_VERSION = str(_values["SCHEMA_VERSION"])
ACTION_SCHEMA_VERSION = str(_values["ACTION_SCHEMA_VERSION"])
LEGAL_ACTION_ORDERING_VERSION = str(_values["LEGAL_ACTION_ORDERING_VERSION"])
OBSERVATION_SCHEMA_VERSION = str(_values["OBSERVATION_SCHEMA_VERSION"])
REWARD_SCHEMA_VERSION = str(_values["REWARD_SCHEMA_VERSION"])

__all__ = [
    "ACTION_SCHEMA_VERSION",
    "API_VERSION",
    "LEGAL_ACTION_ORDERING_VERSION",
    "OBSERVATION_SCHEMA_VERSION",
    "REWARD_SCHEMA_VERSION",
    "SCHEMA_VERSION",
]
