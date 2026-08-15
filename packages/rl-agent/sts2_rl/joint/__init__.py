"""Unified hybrid candidate-Q control path."""

from .authority import (
    HYBRID_AUTHORITY_VERSION,
    DomainExplorationSchedule,
    HybridCollectionAuthority,
)
from .death_price import (
    DEATH_PRICE_CONTROLLER_VERSION,
    DEATH_PRICE_COURSE_VERSION,
    AbsoluteDeathPriceController,
    DeathPriceCourse,
    DeathPriceEpisodeOutcome,
    DeathPriceMetrics,
    death_price_course,
    validate_discounted_outcome_margin,
)
from .learner import (
    HYBRID_Q_LEARNER_VERSION,
    HybridQConfig,
    HybridQLearner,
    HybridQMetrics,
)
from .replay import (
    HYBRID_REPLAY_VERSION,
    DomainView,
    HybridRunReplay,
    HybridViewPair,
)
from .transitions import (
    HYBRID_TRANSITION_VERSION,
    HybridRun,
    HybridStep,
    next_domain_index,
)

__all__ = [
    "DEATH_PRICE_CONTROLLER_VERSION",
    "DEATH_PRICE_COURSE_VERSION",
    "HYBRID_AUTHORITY_VERSION",
    "HYBRID_Q_LEARNER_VERSION",
    "HYBRID_REPLAY_VERSION",
    "HYBRID_TRANSITION_VERSION",
    "AbsoluteDeathPriceController",
    "DeathPriceCourse",
    "DeathPriceEpisodeOutcome",
    "DeathPriceMetrics",
    "DomainExplorationSchedule",
    "DomainView",
    "HybridCollectionAuthority",
    "HybridQConfig",
    "HybridQLearner",
    "HybridQMetrics",
    "HybridRun",
    "HybridRunReplay",
    "HybridStep",
    "HybridViewPair",
    "death_price_course",
    "next_domain_index",
    "validate_discounted_outcome_margin",
]
