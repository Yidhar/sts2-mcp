"""Isolated macro-domain control: semantic transitions, replay, Double-Q.

Stage two of the semantic decision graph reset.  This package owns macro
preference learning exclusively; it never imports the legacy training stack,
and the legacy stack never imports it.
"""

from .authority import (
    MACRO_AUTHORITY_VERSION,
    MacroCollectionAuthority,
)
from .learner import (
    MACRO_Q_LEARNER_VERSION,
    MacroQConfig,
    MacroQLearner,
    MacroQMetrics,
)
from .replay import (
    MACRO_REPLAY_CONTRACT_VERSION,
    MacroSequenceReplay,
    MacroWindow,
)
from .transitions import (
    MACRO_TRANSITION_CONTRACT_VERSION,
    MacroEpisode,
    MacroStep,
    n_step_targets,
    summarize_counts,
)

__all__ = [
    "MACRO_AUTHORITY_VERSION",
    "MACRO_Q_LEARNER_VERSION",
    "MACRO_REPLAY_CONTRACT_VERSION",
    "MACRO_TRANSITION_CONTRACT_VERSION",
    "MacroCollectionAuthority",
    "MacroEpisode",
    "MacroQConfig",
    "MacroQLearner",
    "MacroQMetrics",
    "MacroSequenceReplay",
    "MacroStep",
    "MacroWindow",
    "n_step_targets",
    "summarize_counts",
]
