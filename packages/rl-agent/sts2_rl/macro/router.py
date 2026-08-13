"""Joined evaluation routing for independently owned decision domains.

The combat and macro candidate-Q controllers deliberately do not share a
model, recurrent state, replay, or optimizer.  This router is only the thin
collection-time composition needed to evaluate both controllers in one run.
It delegates each observation to exactly one domain authority and forwards
factual rewards to both domain views.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Final

import numpy as np

from sts2_rl.encoding import EncodedDecisionSnapshot

from .authority import MacroCollectionAuthority
from .transitions import MacroEpisode

JOINED_AUTHORITY_ROUTER_VERSION: Final = "sts2-joined-authority-router-v1"


class JoinedCollectionAuthority:
    """Route a live run across independent macro and combat authorities.

    The class owns no policy parameters and creates no additional learning
    signal.  It exists so joined held-out evaluation cannot accidentally use
    one recurrent/model instance for both control domains.
    """

    def __init__(
        self,
        *,
        macro: MacroCollectionAuthority,
        combat: MacroCollectionAuthority,
    ) -> None:
        if macro.control_domain != "macro":
            raise ValueError("joined router macro authority has the wrong domain")
        if combat.control_domain != "combat":
            raise ValueError("joined router combat authority has the wrong domain")
        self.macro = macro
        self.combat = combat
        self.evaluation_ownership = bool(
            macro.evaluation_ownership or combat.evaluation_ownership
        )

    @property
    def authorities(self) -> tuple[MacroCollectionAuthority, MacroCollectionAuthority]:
        return self.macro, self.combat

    @property
    def record_decisions(self) -> bool:
        return all(authority.record_decisions for authority in self.authorities)

    @record_decisions.setter
    def record_decisions(self, value: bool) -> None:
        for authority in self.authorities:
            authority.record_decisions = bool(value)

    @property
    def decision_log(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for authority in self.authorities:
            rows.extend(
                {"control_domain": authority.control_domain, **row}
                for row in authority.decision_log
            )
        return rows

    @decision_log.setter
    def decision_log(self, value: list[dict[str, Any]]) -> None:
        # Evaluation callers clear the diagnostic log by assigning ``[]``.
        if value:
            raise ValueError("joined decision log can only be cleared")
        for authority in self.authorities:
            authority.decision_log = []

    @property
    def overrides(self) -> int:
        return sum(authority.overrides for authority in self.authorities)

    @property
    def mechanical_dispatches(self) -> int:
        return sum(authority.mechanical_dispatches for authority in self.authorities)

    @property
    def declined(self) -> int:
        return sum(authority.declined for authority in self.authorities)

    def begin_episode(self, episode_id: str) -> None:
        self.macro.begin_episode(f"{episode_id}:macro")
        self.combat.begin_episode(f"{episode_id}:combat")

    def choose(
        self,
        *,
        observation: Mapping[str, Any],
        semantic_actions: Sequence[Mapping[str, Any]],
        snapshot: EncodedDecisionSnapshot,
        valid: np.typing.NDArray[np.bool_],
    ) -> int | None:
        # Both controllers must observe domain boundaries even though only one
        # of them is asked to choose.  In particular this resets tactical
        # recurrence between two combats separated by macro decisions.
        for current in self.authorities:
            current.observe_observation(observation)
        combat = observation.get("combat")
        authority = (
            self.combat
            if isinstance(combat, Mapping) and combat.get("in_progress") is True
            else self.macro
        )
        override = authority.choose(
            observation=observation,
            semantic_actions=semantic_actions,
            snapshot=snapshot,
            valid=valid,
        )
        if authority is self.macro and self.combat.has_pending_bridge:
            # Cross-domain bootstrap bridge: the first OWNED macro decision
            # after an encounter closes the combat domain's pending bridge
            # with the macro authority's greedy legal value of that exact
            # surface, evaluated at collection time under its real run-scale
            # recurrent state.  Freshness: the macro authority publishes
            # ``last_decision_value`` only when this very call owned a
            # semantic decision; the router reads-and-clears it here.  A
            # DECLINED (champion-handled) surface leaves the bridge pending —
            # boundary rewards keep folding into the open combat transition
            # until the next owned macro decision.  The router stays
            # policy-free: it forwards the decision snapshot it was already
            # given and a value the partner already computed.
            bridge_value = self.macro.last_decision_value
            self.macro.last_decision_value = None
            if bridge_value is not None:
                self.combat.close_encounter(snapshot, bridge_value)
        return override

    def observe_step(self, *, reward: float, floor: int | None, terminal: bool) -> None:
        for authority in self.authorities:
            authority.observe_step(reward=reward, floor=floor, terminal=terminal)

    def finish_episodes(
        self,
        episode_id: str,
    ) -> dict[str, MacroEpisode | None]:
        return {
            "macro": self.macro.finish_episode(f"{episode_id}:macro"),
            "combat": self.combat.finish_episode(f"{episode_id}:combat"),
        }

    def metrics(self) -> dict[str, Any]:
        return {
            "version": JOINED_AUTHORITY_ROUTER_VERSION,
            "macro": self.macro.metrics(),
            "combat": self.combat.metrics(),
        }


__all__ = ["JOINED_AUTHORITY_ROUTER_VERSION", "JoinedCollectionAuthority"]
