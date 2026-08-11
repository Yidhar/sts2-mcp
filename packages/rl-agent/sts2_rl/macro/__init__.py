"""Domain-owned semantic control: transitions, replay, and Double-Q.

The macro and combat challengers use the same factual contracts but separate
model/recurrent/optimizer instances.  Joined evaluation composes those owners
through a parameter-free router.
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
from .loading import (
    MACRO_LOADING_CONTRACT_VERSION,
    TOLERATED_HEAD_GROUPS,
    load_trunk_state,
)
from .replay import (
    MACRO_REPLAY_CONTRACT_VERSION,
    MacroSequenceReplay,
    MacroWindow,
)
from .router import (
    JOINED_AUTHORITY_ROUTER_VERSION,
    JoinedCollectionAuthority,
)
from .transitions import (
    MACRO_TRANSITION_CONTRACT_VERSION,
    MacroEpisode,
    MacroStep,
    n_step_targets,
    summarize_counts,
)

__all__ = [
    "JOINED_AUTHORITY_ROUTER_VERSION",
    "MACRO_AUTHORITY_VERSION",
    "MACRO_LOADING_CONTRACT_VERSION",
    "MACRO_Q_LEARNER_VERSION",
    "MACRO_REPLAY_CONTRACT_VERSION",
    "MACRO_TRANSITION_CONTRACT_VERSION",
    "TOLERATED_HEAD_GROUPS",
    "JoinedCollectionAuthority",
    "MacroCollectionAuthority",
    "MacroEpisode",
    "MacroQConfig",
    "MacroQLearner",
    "MacroQMetrics",
    "MacroSequenceReplay",
    "MacroStep",
    "MacroWindow",
    "load_trunk_state",
    "n_step_targets",
    "summarize_counts",
]
