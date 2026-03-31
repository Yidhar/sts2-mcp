"""Compatibility wrapper for legacy imports.

The live online model source of truth now lives in ``sts2_env.model``.
"""

from .model import STS2CandidateScoringPolicy

__all__ = ["STS2CandidateScoringPolicy"]
