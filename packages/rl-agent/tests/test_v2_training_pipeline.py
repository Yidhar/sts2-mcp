from __future__ import annotations

import hashlib
import json
import math
import random
import time
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

import sts2_rl.macro_evaluation as macro_evaluation_module
from sts2_rl.checkpoints import CheckpointIntegrityError, ValidatedResumeCheckpoint
from sts2_rl.contracts import (
    BackendCapabilities,
    CombatResetRequest,
    EnvironmentResult,
    EnvironmentTransition,
    ResetRequest,
    StepRequest,
)
from sts2_rl.encoding import grounding_encoding_identity
from sts2_rl.training import (
    ENGINE_REVIVAL_MECHANISM,
    CurriculumConfig,
    DiagnosticsConfig,
    EnvironmentConfig,
    EvaluationGateState,
    FailureCreditConfig,
    ModelConfig,
    OptimizationConfig,
    RolloutConfig,
    RuntimeConfig,
    TrainingConfig,
    TransactionLearningConfig,
    TransactionOutcome,
    TransactionPolicyTarget,
    build_training_resources,
    evaluate_policy,
    factual_transaction_policy_targets,
    initialize_model_from_checkpoint,
    inspect_baseline,
    load_training_checkpoint,
    preflight_model_initialization,
    preflight_training_checkpoint,
    run_training,
    save_training_checkpoint,
)
from sts2_rl.training import checkpointing as checkpointing_module
from sts2_rl.training import factory as factory_module
from sts2_rl.training import runtime as runtime_module
from sts2_rl.training.checkpoint_evaluation import evaluate_checkpoint_policy
from sts2_rl.training.checkpointing import ActorSupervisorState, TrainingState
from sts2_rl.training.collector import (
    CollectionProtocolError,
    _CombatNetProgressTracker,
    _NonCombatEventCycleTracker,
)
from sts2_rl.training.failure_credit import EvidenceStratum
from sts2_rl.training.pipeline import ActorLearnerPipeline, RecoverableActorIncident
from sts2_rl.training.runtime import EvaluationInfrastructureError
from sts2_rl.training.trajectory import TrajectoryJournal


class FakeCombatBackend:
    def __init__(self, *, terminal_step: int = 2) -> None:
        self._capabilities = BackendCapabilities(
            backend_name="fake",
            session_id="fake-session",
        )
        self._state_version = 0
        self._step = 0
        self._episode = 0
        self.terminal_step = terminal_step
        self.closed = False
        self.reset_seeds: list[int | str | None] = []

    @property
    def capabilities(self) -> BackendCapabilities:
        return self._capabilities

    @property
    def session_id(self) -> str:
        return self._capabilities.session_id

    @property
    def is_connected(self) -> bool:
        return not self.closed

    def health(self) -> dict[str, Any]:
        return {"ok": True}

    def get_spec(self) -> dict[str, Any]:
        return {"ok": True}

    def get_state(self) -> dict[str, Any]:
        return {"ok": True, "state_version": self._state_version}

    @staticmethod
    def _actions() -> tuple[dict[str, Any], ...]:
        return (
            {
                "action_handle": "attack",
                "kind": "play_card",
                "model_action_kind": "play_card",
                "card": {"id": "attack", "cost": 1},
                "target": {"id": "enemy", "side": "enemy"},
            },
            {
                "action_handle": "end",
                "kind": "end_turn",
                "model_action_kind": "end_turn",
            },
        )

    def _observation(self, *, terminal: bool) -> dict[str, Any]:
        return {
            "phase": "combat",
            "decision_domain": "combat",
            "player": {"id": "player", "hp": 50, "max_hp": 80},
            "combat": {
                "in_progress": not terminal,
                "enemies": [
                    {
                        "id": "enemy",
                        "hp": 0 if terminal else max(1, 30 - 5 * self._step),
                        "max_hp": 30,
                    }
                ],
            },
            "run": {"act": 1, "floor": 1},
        }

    def reset(self, request: ResetRequest) -> EnvironmentResult:
        raise AssertionError("combat fake must use combat_reset")

    def combat_reset(self, request: CombatResetRequest) -> EnvironmentResult:
        assert request.expected_state_version == self._state_version
        self.reset_seeds.append(request.seed)
        before = self._state_version
        self._state_version += 1
        self._step = 0
        self._episode += 1
        episode_id = f"episode-{self._episode}"
        return EnvironmentResult(
            episode_id=episode_id,
            step_index=0,
            observation=self._observation(terminal=False),
            legal_actions=self._actions(),
            transition=EnvironmentTransition(
                episode_id=episode_id,
                step_index=0,
                before_state_version=before,
                after_state_version=self._state_version,
                facts={"combat_result": "none", "terminal_reason": None},
            ),
            info={"reward_authority": "external-rl"},
        )

    def step(self, request: StepRequest) -> EnvironmentResult:
        assert request.expected_step_index == self._step
        before = self._state_version
        self._state_version += 1
        self._step += 1
        terminal = self._step >= self.terminal_step
        reason = "combat_victory" if terminal else None
        return EnvironmentResult(
            episode_id=f"episode-{self._episode}",
            step_index=self._step,
            observation=self._observation(terminal=terminal),
            legal_actions=() if terminal else self._actions(),
            transition=EnvironmentTransition(
                episode_id=f"episode-{self._episode}",
                step_index=self._step,
                before_state_version=before,
                after_state_version=self._state_version,
                facts={
                    "combat_result": "victory" if terminal else "none",
                    "terminal_reason": reason,
                },
            ),
            terminated=terminal,
            terminal_reason=reason,
            info={"reward_authority": "external-rl"},
        )

    def close(self) -> None:
        self.closed = True


class StaleRevisionBackend(FakeCombatBackend):
    def step(self, request: StepRequest) -> EnvironmentResult:
        result = super().step(request)
        assert result.transition is not None
        return replace(
            result,
            transition=replace(
                result.transition,
                before_state_version=result.transition.before_state_version - 1,
            ),
        )


class RecoverableTestProtocolError(RuntimeError):
    recoverable = True
    incident_kind = "test_committed_unsettled"
    fingerprint = "test:step:committed-unsettled"
    quarantine_path = "tests/quarantine/incident.json.gz"


class RecoverableIncidentBackend(FakeCombatBackend):
    def __init__(self, *, fail_step: int) -> None:
        super().__init__(terminal_step=100)
        self.fail_step = fail_step

    def step(self, request: StepRequest) -> EnvironmentResult:
        if self._step + 1 == self.fail_step:
            raise RecoverableTestProtocolError("simulator committed but did not settle")
        return super().step(request)


class WideRecoverableIncidentBackend(RecoverableIncidentBackend):
    """Expose a 111-candidate accepted tail before a recoverable failure."""

    def _actions(self) -> tuple[dict[str, Any], ...]:
        if self._step == 0:
            return FakeCombatBackend._actions()
        return tuple(
            {
                "action_handle": f"attack-{index}",
                "kind": "play_card",
                "model_action_kind": "play_card",
                "card": {"id": "attack", "cost": 1},
                "target": {"id": "enemy", "side": "enemy"},
            }
            for index in range(111)
        )


class StrictGroupedSelectionBackend(FakeCombatBackend):
    """Two strict card groups whose learned and dispatch indexes differ."""

    def __init__(self) -> None:
        super().__init__(terminal_step=1)
        self.step_requests: list[StepRequest] = []

    @staticmethod
    def _actions() -> tuple[dict[str, Any], ...]:
        actions: list[dict[str, Any]] = []
        for card_id, enabled in (("CARD.WOUND", False), ("CARD.BURN", True)):
            for copy_index in range(3):
                actions.append(
                    {
                        "action_handle": f"select-{card_id}-{copy_index}",
                        "action": "combat_select_card",
                        "kind": "combat_select_card",
                        "model_action_kind": "card_selection",
                        "model_action_variant": "select",
                        "selection_operation": "select",
                        "enabled": enabled,
                        "card_index": len(actions),
                        "card": {
                            "id": card_id,
                            "pile": "Discard",
                            "cost": -2,
                            "is_selected": False,
                        },
                    }
                )
        return tuple(actions)

    def _observation(self, *, terminal: bool) -> dict[str, Any]:
        observation = super()._observation(terminal=terminal)
        observation["phase"] = "selection"
        observation["card_selection"] = {
            "mode": "SimpleGrid",
            "prompt_id": "TEST.STRICT_GROUPING",
            "selected_count": 0,
            "min_select": 1,
            "max_select": 1,
            "requires_manual_confirmation": False,
            "can_confirm": False,
            "cards": [
                {
                    "card_instance_id": f"instance-{index}",
                    "id": "CARD.WOUND" if index < 3 else "CARD.BURN",
                    "pile": "Discard",
                    "cost": -2,
                    "is_selected": False,
                }
                for index in range(6)
            ],
        }
        return observation

    def step(self, request: StepRequest) -> EnvironmentResult:
        self.step_requests.append(request)
        return super().step(request)


class OscillatingDamageBackend(FakeCombatBackend):
    """Deals damage every other step, then heals it all back."""

    def _observation(self, *, terminal: bool) -> dict[str, Any]:
        return {
            "phase": "combat",
            "decision_domain": "combat",
            "player": {"id": "player", "hp": 50, "max_hp": 80},
            "combat": {
                "in_progress": not terminal,
                "enemies": [
                    {
                        "id": "enemy",
                        "hp": 95 if self._step % 2 else 100,
                        "max_hp": 100,
                    }
                ],
            },
            "run": {"act": 1, "floor": 1},
        }


class ExhaustedOnlyEndTurnBackend(FakeCombatBackend):
    """Combat tail with cards in hand but no legal play and no draw/discard recovery."""

    @staticmethod
    def _actions() -> tuple[dict[str, Any], ...]:
        return (
            {
                "action_handle": "end",
                "kind": "end_turn",
                "model_action_kind": "end_turn",
            },
        )

    def _observation(self, *, terminal: bool) -> dict[str, Any]:
        hand = [
            {
                "id": "CARD.BURN" if index % 2 else "CARD.STATUS",
                "is_playable": False,
            }
            for index in range(10)
        ]
        return {
            "phase": "combat",
            "decision_domain": "combat",
            "player": {
                "id": "player",
                "hp": 50,
                "max_hp": 80,
                "hand": {"cards": hand},
                "draw_pile": {"cards": []},
                "discard_pile": {"cards": []},
                "exhaust_pile": {"cards": [{"id": "CARD.EXHAUSTED"}]},
            },
            "combat": {
                "in_progress": not terminal,
                "enemies": [{"id": "enemy", "hp": 100, "max_hp": 100}],
            },
            "run": {"act": 3, "floor": 46},
        }


class DynamicPreviewLoopBackend(FakeCombatBackend):
    """Non-combat event whose preview counter changes but run state does not."""

    def __init__(
        self,
        *,
        durable_change_step: int | None = None,
        durable_change_kind: str = "gold",
        durable_resource_cycle: tuple[int, ...] = (),
        alternate_ui_actions: bool = False,
        live_deck_shape: bool = False,
    ) -> None:
        super().__init__(terminal_step=100_000)
        self.durable_change_step = durable_change_step
        self.durable_change_kind = durable_change_kind
        self.durable_resource_cycle = durable_resource_cycle
        self.alternate_ui_actions = alternate_ui_actions
        self.live_deck_shape = live_deck_shape

    def _actions(self) -> tuple[dict[str, Any], ...]:
        if self.alternate_ui_actions:
            if self._step % 2 == 0:
                return (
                    {
                        "action_handle": f"select:{self._step}",
                        "action": "select_card",
                        "kind": "select_card",
                        "model_action_kind": "card_selection",
                        "model_action_variant": "select",
                        "card": {
                            "id": "CARD.STRIKE",
                            "pile": "Selectable",
                            "is_selected": False,
                        },
                    },
                )
            return (
                {
                    "action_handle": f"cancel:{self._step}",
                    "action": "cancel_selection",
                    "kind": "cancel_selection",
                    "model_action_kind": "card_selection",
                    "model_action_variant": "cancel_prompt",
                },
            )
        return (
            {
                "action_handle": f"hold:{self._step}",
                "action": "choose_event_option",
                "kind": "choose_event_option",
                "model_action_kind": "event_option",
                "transport_kind": "event_option",
                "index": 1,
                "label": f"EVENT.LOOP.options.HOLD_{self._step}",
                "option": {
                    "index": 1,
                    "text_key": f"EVENT.LOOP.options.HOLD_{self._step}",
                    "is_locked": False,
                    "is_chosen": False,
                    "is_proceed": False,
                },
            },
        )

    def _observation(self, *, terminal: bool = False) -> dict[str, Any]:
        durable_changed = bool(self.durable_change_step is not None and self._step >= self.durable_change_step)
        if self.durable_resource_cycle:
            gold = self.durable_resource_cycle[self._step % len(self.durable_resource_cycle)]
        else:
            gold = 100 if durable_changed and self.durable_change_kind == "gold" else 99
        deck = [
            {"id": "CARD.STRIKE", "is_upgraded": False},
            {"id": "CARD.DEFEND", "is_upgraded": False},
        ]
        if durable_changed and self.durable_change_kind == "deck":
            deck.append({"id": "CARD.BASH", "is_upgraded": True})
        room_changed = durable_changed and self.durable_change_kind == "room"
        observation: dict[str, Any] = {
            "phase": "selection" if self._step % 2 else "event",
            "decision_domain": "build",
            "state_type": "event",
            "screen": "SELECTION" if self._step % 2 else "EVENT",
            "terminated": terminal,
            "truncated": False,
            "player": {
                "character": "IRONCLAD",
                "hp": 33,
                "max_hp": 67,
                "gold": gold,
                "open_potion_slots": 2,
                "deck": len(deck) if self.live_deck_shape else deck,
                "relics": [
                    {"id": "RELIC.LIZARD_TAIL", "is_used_up": False},
                ],
                "potions": [{"id": "POTION.FIRE", "slot_index": 0}],
            },
            "combat": {"in_progress": False, "enemies": []},
            "run": {
                "active": True,
                "act": 1,
                "floor": 10 if room_changed else 9,
                "room_type": "event",
                "room_model_id": "EVENT.NEXT" if room_changed else "EVENT.LOOP",
            },
            "event": {
                "event_id": "EVENT.NEXT" if room_changed else "EVENT.LOOP",
                "description_key": f"EVENT.LOOP.pages.PREVIEW_{self._step}",
                "is_finished": False,
                "dynamic_vars": [
                    {
                        "name": "HpLoss",
                        "family": "value",
                        "int_value": self._step,
                        "preview_value": self._step,
                    }
                ],
                "options": [
                    {
                        "index": 1,
                        "text_key": "EVENT.LOOP.options.HOLD",
                        "is_locked": False,
                        "is_chosen": False,
                        "is_proceed": False,
                    }
                ],
            },
        }
        if self.live_deck_shape:
            player = observation["player"]
            assert isinstance(player, dict)
            player["deck_cards"] = deck
        if self.alternate_ui_actions:
            observation["card_selection"] = {
                "mode": "SimpleGrid",
                "prompt_id": "EVENT.LOOP.select_or_cancel",
                "selected_count": self._step % 2,
                "min_select": 0,
                "max_select": 1,
                "requires_manual_confirmation": True,
                "can_confirm": bool(self._step % 2),
                "options": [
                    {
                        "option_index": 0,
                        "is_selected": bool(self._step % 2),
                        "card": {"id": "CARD.STRIKE", "pile": "Selectable"},
                    }
                ],
            }
        return observation

    def reset(self, request: ResetRequest) -> EnvironmentResult:
        assert request.expected_state_version == self._state_version
        self.reset_seeds.append(request.seed)
        before = self._state_version
        self._state_version += 1
        self._step = 0
        self._episode += 1
        episode_id = f"event-episode-{self._episode}"
        return EnvironmentResult(
            episode_id=episode_id,
            step_index=0,
            observation=self._observation(),
            legal_actions=self._actions(),
            transition=EnvironmentTransition(
                episode_id=episode_id,
                step_index=0,
                before_state_version=before,
                after_state_version=self._state_version,
                facts={"combat_result": "none", "terminal_reason": None},
            ),
            info={"reward_authority": "external-rl"},
        )

    def combat_reset(self, request: CombatResetRequest) -> EnvironmentResult:
        raise AssertionError("event fake must use reset")

    def step(self, request: StepRequest) -> EnvironmentResult:
        assert request.expected_step_index == self._step
        before = self._state_version
        self._state_version += 1
        self._step += 1
        episode_id = f"event-episode-{self._episode}"
        return EnvironmentResult(
            episode_id=episode_id,
            step_index=self._step,
            observation=self._observation(),
            legal_actions=self._actions(),
            transition=EnvironmentTransition(
                episode_id=episode_id,
                step_index=self._step,
                before_state_version=before,
                after_state_version=self._state_version,
                facts={"combat_result": "none", "terminal_reason": None},
            ),
            info={"reward_authority": "external-rl"},
        )


class VitalityEventCycleBackend(DynamicPreviewLoopBackend):
    """Two-option event cycle with unbounded HP/max-HP/revival churn."""

    def _actions(self) -> tuple[dict[str, Any], ...]:
        return tuple(
            {
                "action_handle": f"event-option:{index}:{self._step}",
                "action": "choose_event_option",
                "kind": "choose_event_option",
                "model_action_kind": "event_option",
                "transport_kind": "event_option",
                "index": index,
                "option": {
                    "index": index,
                    "text_key": f"EVENT.VITALITY.options.{index}",
                    "is_locked": False,
                    "is_chosen": False,
                    "is_proceed": bool(index),
                },
            }
            for index in range(2)
        )

    def _observation(self, *, terminal: bool = False) -> dict[str, Any]:
        observation = super()._observation(terminal=terminal)
        player = observation["player"]
        assert isinstance(player, dict)
        player["hp"] = 1 if self._step % 2 == 0 else 40 + self._step
        player["max_hp"] = 80 + 2 * self._step
        observation["phase"] = "event"
        observation["screen"] = "EVENT"
        observation["_training"] = {
            "revivals_used": self._step // 2,
            "player_hp_lost": self._step * (self._step + 1),
        }
        event = observation["event"]
        assert isinstance(event, dict)
        event["event_id"] = "EVENT.VITALITY"
        event["description_key"] = (
            "EVENT.VITALITY.pages.LINGER9" if self._step % 2 == 0 else "EVENT.VITALITY.pages.DEATH_WARNING"
        )
        event["dynamic_vars"] = [
            {
                "name": "Damage",
                "family": "value",
                "int_value": self._step * self._step,
            }
        ]
        return observation


class ChangingAlternativeEventBackend(VitalityEventCycleBackend):
    """Reuses option indexes but changes the alternative's stable semantics."""

    def _actions(self) -> tuple[dict[str, Any], ...]:
        selected, alternative = super()._actions()
        alternative = {
            **alternative,
            "action_handle": f"changing-alternative:{self._step}",
            "option": {
                **alternative["option"],
                "text_key": f"EVENT.VITALITY.options.ALTERNATIVE_{self._step}",
                "is_proceed": bool(self._step % 2),
            },
        }
        return selected, alternative


class SparseRepeatEventBackend(VitalityEventCycleBackend):
    """Repeats an old decision only after it has left the learnable tail."""

    def _actions(self) -> tuple[dict[str, Any], ...]:
        selected, alternative = super()._actions()
        # The original action surface occurs three times globally, but never
        # three times inside the 32-step actor-learnable liveness tail.
        if self._step not in {0, 17, 35}:
            alternative = {
                **alternative,
                "option": {
                    **alternative["option"],
                    "text_key": f"EVENT.VITALITY.options.NOVEL_{self._step}",
                },
            }
        return selected, alternative

    def _observation(self, *, terminal: bool = False) -> dict[str, Any]:
        observation = super()._observation(terminal=terminal)
        event = observation["event"]
        assert isinstance(event, dict)
        event["description_key"] = "EVENT.VITALITY.pages.CONSTANT"
        return observation


class ChangingMultiplicityEventBackend(VitalityEventCycleBackend):
    """Changes only the number of strictly equal copies of the chosen option."""

    def _actions(self) -> tuple[dict[str, Any], ...]:
        selected, alternative = super()._actions()
        copies = tuple(
            {
                **selected,
                "action_handle": f"event-option:0:{self._step}:{copy_index}",
            }
            for copy_index in range(self._step + 1)
        )
        return (*copies, alternative)


class LockedAlternativeEventBackend(VitalityEventCycleBackend):
    """Publishes a locked option beside the only enabled event action."""

    def _actions(self) -> tuple[dict[str, Any], ...]:
        selected, alternative = super()._actions()
        return selected, {
            **alternative,
            "enabled": False,
            "is_enabled": False,
            "option": {**alternative["option"], "is_locked": True},
        }


class TerminalWithoutObservationFlagsBackend(DynamicPreviewLoopBackend):
    """Terminates at the stall boundary without mirroring result flags in obs."""

    def __init__(self, *, terminal_step: int) -> None:
        super().__init__()
        self.result_terminal_step = terminal_step

    def _observation(self, *, terminal: bool = False) -> dict[str, Any]:
        observation = super()._observation(terminal=terminal)
        observation.pop("terminated", None)
        observation.pop("truncated", None)
        return observation

    def step(self, request: StepRequest) -> EnvironmentResult:
        assert request.expected_step_index == self._step
        before = self._state_version
        self._state_version += 1
        self._step += 1
        episode_id = f"event-episode-{self._episode}"
        terminal = self._step >= self.result_terminal_step
        terminal_reason = "run_victory" if terminal else None
        return EnvironmentResult(
            episode_id=episode_id,
            step_index=self._step,
            observation=self._observation(terminal=terminal),
            legal_actions=() if terminal else self._actions(),
            transition=EnvironmentTransition(
                episode_id=episode_id,
                step_index=self._step,
                before_state_version=before,
                after_state_version=self._state_version,
                facts={
                    "combat_result": "none",
                    "run_result": "victory" if terminal else "none",
                    "terminal_reason": terminal_reason,
                },
            ),
            terminated=terminal,
            terminal_reason=terminal_reason,
            info={"reward_authority": "external-rl"},
        )


class RunTerminalWithoutRunResultBackend(TerminalWithoutObservationFlagsBackend):
    """Emulates the old bug: a run terminal mislabeled only as combat victory."""

    def step(self, request: StepRequest) -> EnvironmentResult:
        result = super().step(request)
        if not result.terminated:
            return result
        assert result.transition is not None
        return replace(
            result,
            transition=replace(
                result.transition,
                facts={
                    "combat_result": "victory",
                    "terminal_reason": "run_victory",
                },
            ),
        )


class RunDefeatBackend(TerminalWithoutObservationFlagsBackend):
    def step(self, request: StepRequest) -> EnvironmentResult:
        result = super().step(request)
        if not result.terminated:
            return result
        assert result.transition is not None
        return replace(
            result,
            transition=replace(
                result.transition,
                facts={
                    "combat_result": "none",
                    "run_result": "defeat",
                    "terminal_reason": "run_defeat",
                },
            ),
            terminal_reason="run_defeat",
        )


class Act1ContradictoryTerminalReasonBackend(RunDefeatBackend):
    def step(self, request: StepRequest) -> EnvironmentResult:
        result = super().step(request)
        if not result.terminated:
            return result
        assert result.transition is not None
        return replace(
            result,
            transition=replace(
                result.transition,
                facts={**result.transition.facts, "terminal_reason": "run_victory"},
            ),
        )


class StaticEventLoopBackend(DynamicPreviewLoopBackend):
    """Exact semantic loop used to make both detectors fire together."""

    def _actions(self) -> tuple[dict[str, Any], ...]:
        return (
            {
                "action_handle": f"hold:{self._step}",
                "action": "choose_event_option",
                "kind": "choose_event_option",
                "model_action_kind": "event_option",
                "transport_kind": "event_option",
                "index": 1,
                "label": "EVENT.LOOP.options.HOLD",
            },
        )

    def _observation(self, *, terminal: bool = False) -> dict[str, Any]:
        observation = super()._observation(terminal=terminal)
        observation["phase"] = "event"
        event = observation["event"]
        assert isinstance(event, dict)
        event["description_key"] = "EVENT.LOOP.pages.HOLD"
        event["dynamic_vars"] = []
        return observation


def _config(*, total_steps: int = 4) -> TrainingConfig:
    return TrainingConfig(
        profile="v2-test",
        model=ModelConfig(
            token_feature_dim=224,
            d_model=32,
            n_heads=4,
            ffn_dim=64,
            world_layers=1,
            latent_slots=4,
            latent_layers=1,
            local_layers=1,
            candidate_layers=1,
            recurrent_hidden_dim=32,
            dropout=0.0,
            type_vocab_size=32,
            role_vocab_size=32,
            owner_vocab_size=16,
            entity_vocab_size=128,
            zone_vocab_size=16,
            order_vocab_size=32,
            domain_count=8,
            max_world_tokens=24,
            max_candidates=6,
            max_candidate_local_tokens=5,
        ),
        optimization=OptimizationConfig(batch_unrolls=1),
        rollout=RolloutConfig(
            unroll_length=2,
            queue_capacity=8,
            minimum_unrolls=1,
            policy_sync_interval_unrolls=1,
            max_policy_lag=32,
        ),
        environment=EnvironmentConfig(
            backend="headless",
            scenario="combat",
            max_episode_steps=4,
        ),
        curriculum=CurriculumConfig(
            reward_objective="combat",
            epsilon_start=0.2,
            epsilon_end=0.1,
            epsilon_decay_steps=10,
        ),
        runtime=RuntimeConfig(
            device="cpu",
            collector_device="cpu",
            total_environment_steps=total_steps,
            seed=7,
            log_dir="tests/v2-run",
            checkpoint_dir="tests/v2-checkpoints",
            checkpoint_interval_steps=100,
            evaluation_steps=(),
            evaluation_episodes=0,
        ),
    )


def test_evaluation_gate_state_mapping_is_versioned_and_fail_closed() -> None:
    state = EvaluationGateState(
        completed_validation_steps=(0, 100),
        completed_early_validation_steps=(5,),
        completed_final_audit_steps=(250,),
    )
    assert EvaluationGateState.from_mapping(state.to_mapping()) == state

    malformed = state.to_mapping()
    malformed["version"] = "unknown-evaluation-state"
    with pytest.raises(ValueError, match="unsupported checkpoint evaluation_state"):
        EvaluationGateState.from_mapping(malformed)

    malformed = state.to_mapping()
    malformed["completed_final_audit_steps"] = [250, 250]
    with pytest.raises(ValueError, match="strictly increasing"):
        EvaluationGateState.from_mapping(malformed)

    malformed = state.to_mapping()
    malformed["completed_validation_steps"] = (0, 100)
    with pytest.raises(TypeError, match="must be an array"):
        EvaluationGateState.from_mapping(malformed)


def test_checkpoint_rejects_completed_evaluation_gate_beyond_training_horizon(
    tmp_path: Path,
) -> None:
    config = _config(total_steps=4)
    resources = build_training_resources(config, backend=FakeCombatBackend())
    target = tmp_path / "future-evaluation-gate"
    try:
        with pytest.raises(
            ValueError,
            match=r"cannot complete gates beyond training_state.environment_steps",
        ):
            save_training_checkpoint(
                target,
                config=config,
                resources=resources,
                state=TrainingState(environment_steps=2),
                evaluation_state=EvaluationGateState(
                    completed_final_audit_steps=(4,),
                ),
            )
    finally:
        resources.close()
    assert not target.exists()


def test_checkpoint_rejects_completed_gate_absent_from_its_schedule(
    tmp_path: Path,
) -> None:
    config = _config(total_steps=4)
    resources = build_training_resources(config, backend=FakeCombatBackend())
    target = tmp_path / "unconfigured-evaluation-gate"
    try:
        with pytest.raises(
            ValueError,
            match="contains gates absent from its training_config schedule",
        ):
            save_training_checkpoint(
                target,
                config=config,
                resources=resources,
                state=TrainingState(environment_steps=2),
                evaluation_state=EvaluationGateState(
                    completed_final_audit_steps=(2,),
                ),
            )
    finally:
        resources.close()
    assert not target.exists()


def _rewrite_checkpoint_runtime_identity(
    checkpoint: Path,
    *,
    mismatch: str,
) -> dict[str, Any]:
    manifest_path = checkpoint / "checkpoint.manifest.json"
    metadata_path = checkpoint / "metadata.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    contract = dict(manifest["contract"])
    provenance = json.loads(json.dumps(manifest["provenance"]))

    if mismatch == "contract":
        contract["schema_version"] = "2026-07-13.1"
    elif mismatch == "reward":
        reward = dict(provenance["reward_spec"])
        reward["fingerprint"] = reward["fingerprint"] + "|archived-reward"
        reward["fingerprint_sha256"] = hashlib.sha256(reward["fingerprint"].encode("utf-8")).hexdigest()
        provenance["reward_spec"] = reward
    elif mismatch == "dependency":
        locks = [dict(item) for item in provenance["dependency_locks"]]
        locks[0]["sha256"] = "0" * 64
        provenance["dependency_locks"] = locks
    else:  # pragma: no cover - test helper guard
        raise AssertionError(mismatch)

    manifest["contract"] = contract
    manifest["provenance"] = provenance
    metadata["contract"] = contract
    metadata["provenance"] = provenance
    metadata_bytes = (json.dumps(metadata, indent=2, sort_keys=True) + "\n").encode()
    metadata_path.write_bytes(metadata_bytes)
    for entry in manifest["files"]:
        if entry["path"] == "metadata.json":
            entry["size_bytes"] = len(metadata_bytes)
            entry["sha256"] = hashlib.sha256(metadata_bytes).hexdigest()
            break
    else:  # pragma: no cover - publisher always lists metadata
        raise AssertionError("metadata entry missing")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {"contract": contract, "provenance": provenance}


def _event_loop_config(*, durable_window: int = 4) -> TrainingConfig:
    base = _config(total_steps=20)
    return replace(
        base,
        environment=replace(
            base.environment,
            scenario="full-run",
            max_episode_steps=20,
        ),
        curriculum=replace(base.curriculum, reward_objective="run"),
        diagnostics=DiagnosticsConfig(
            deadlock_window=128,
            deadlock_repeat_threshold=8,
            combat_net_progress_window=256,
            noncombat_durable_progress_window=durable_window,
            combat_min_net_hp_fraction=0.05,
            journal_policy_topk=5,
        ),
    )


def test_collector_emits_contiguous_recurrent_unroll() -> None:
    resources = build_training_resources(_config(), backend=FakeCombatBackend())
    try:
        episode = resources.collector.collect_episode(
            epsilon=0.2,
            record=True,
            policy_version=3,
        )
        assert episode.metrics.steps == 2
        assert episode.metrics.combat_won
        assert not episode.metrics.deadlocked
        assert len(episode.unrolls) == 1
        unroll = episode.unrolls[0]
        assert unroll.policy_version == 3
        assert len(unroll.steps) == 2
        assert unroll.bootstrap_snapshot is None
        assert unroll.steps[-1].discount == 0.0
        assert unroll.initial_recurrent_state.shape == (32,)
    finally:
        resources.close()


def test_collector_shadow_emits_separate_formal_failure_credit_records() -> None:
    base = _config()
    config = replace(
        base,
        failure_credit=FailureCreditConfig(
            mode="shadow",
            burn_in_steps=1,
            maximum_context_steps=8,
        ),
    )
    resources = build_training_resources(config, backend=FakeCombatBackend())
    try:
        resources.collector.bind_failure_credit_run_id("collector-shadow-test")
        episode = resources.collector.collect_episode(
            epsilon=0.0,
            record=True,
            policy_version=3,
        )
        assert episode.failure_credit_records
        assert episode.failure_credit_shadow_metrics is not None
        assert episode.failure_credit_shadow_metrics.decisions == 2
        assert all(
            record.incident.provenance.run_id == "collector-shadow-test" for record in episode.failure_credit_records
        )
        assert all(record.plan.actor_label_count == 0 for record in episode.failure_credit_records)
        # Legacy transaction-v3 remains an independent compatibility plane.
        assert episode.transaction_traces == ()
    finally:
        resources.close()


def test_failure_credit_factory_modes_do_not_depend_on_transaction_v3() -> None:
    base = replace(
        _config(),
        transaction_learning=TransactionLearningConfig(enabled=False),
    )
    for mode in ("disabled", "shadow", "learning"):
        config = replace(
            base,
            failure_credit=FailureCreditConfig(
                mode=mode,
                burn_in_steps=1,
                maximum_context_steps=8,
            ),
        )
        resources = build_training_resources(
            config,
            backend=FakeCombatBackend(),
        )
        try:
            assert resources.transaction_replay is None
            assert not resources.model.transaction_heads_enabled
            assert not resources.collector_model.transaction_heads_enabled
            assert resources.collector.failure_credit_shadow_enabled is (mode == "shadow")
            assert resources.collector.failure_credit_learning_enabled is (mode == "learning")
            assert resources.model.liveness_head_enabled is (mode == "learning")
            assert resources.collector_model.liveness_head_enabled is (mode == "learning")
            assert (resources.failure_credit_replay is not None) is (mode == "learning")
            assert resources.learner.failure_credit_config.mode == mode
        finally:
            resources.close()


def test_runtime_failure_credit_quotas_cover_every_learnable_stratum() -> None:
    config = replace(
        _config(),
        failure_credit=FailureCreditConfig(
            mode="learning",
            sample_records=6,
            direct_witness_quota=1,
            multi_edge_cycle_quota=1,
            risk_sequence_quota=1,
            unresolved_stall_quota=1,
            completion_control_quota=1,
            matched_outcome_pair_quota=1,
            liveness_risk_actor_start_update=512,
        ),
    )

    quotas = runtime_module._failure_credit_quotas(config)

    assert {item.stratum: item.minimum for item in quotas} == {
        EvidenceStratum.DIRECT_WITNESS: 1,
        EvidenceStratum.MULTI_EDGE_CYCLE: 1,
        EvidenceStratum.RISK_SEQUENCE: 1,
        EvidenceStratum.UNRESOLVED_STALL: 1,
        EvidenceStratum.COMPLETION_CONTROL: 1,
        EvidenceStratum.MATCHED_OUTCOME_PAIR: 1,
    }
    assert EvidenceStratum.CENSORED not in {item.stratum for item in quotas}

    calibration_quotas = runtime_module._failure_credit_quotas(
        config,
        learner_updates=511,
    )
    assert EvidenceStratum.RISK_SEQUENCE not in {item.stratum for item in calibration_quotas}
    mature_quotas = runtime_module._failure_credit_quotas(
        config,
        learner_updates=512,
    )
    assert {item.stratum: item.minimum for item in mature_quotas} == {item.stratum: item.minimum for item in quotas}


def test_failure_credit_retention_covers_every_configured_stall_window() -> None:
    base = _config()
    config = replace(
        base,
        diagnostics=replace(
            base.diagnostics,
            deadlock_window=16,
            combat_net_progress_window=32,
            noncombat_durable_progress_window=64,
            combat_net_progress_room_windows={"wide_room": 512},
            combat_net_progress_encounter_windows={"wide_encounter": 384},
        ),
        failure_credit=FailureCreditConfig(
            mode="shadow",
            burn_in_steps=1,
            maximum_context_steps=8,
        ),
    )
    resources = build_training_resources(
        config,
        backend=FakeCombatBackend(),
    )
    try:
        assert resources.collector.failure_credit_pipeline_config.detector_window_steps == 512
    finally:
        resources.close()


def test_collector_learns_group_index_but_dispatches_raw_representative(
    tmp_path: Path,
) -> None:
    backend = StrictGroupedSelectionBackend()
    base = _config(total_steps=1)
    config = replace(
        base,
        transaction_learning=replace(
            base.transaction_learning,
            enabled=True,
            burn_in_steps=0,
        ),
    )
    resources = build_training_resources(config, backend=backend)
    journal_path = tmp_path / "grouped-collector.jsonl"
    try:
        initial_decision = resources.encoder.encode(
            backend._observation(terminal=False),
            backend._actions(),
            device="cpu",
        )
        assert initial_decision.snapshot.candidate_count == 2
        selected_reference = initial_decision.action(1)
        assert selected_reference.position == 3
        assert selected_reference.multiplicity == 3
        assert selected_reference.equivalence_fingerprint is not None

        with TrajectoryJournal(journal_path) as journal:
            episode = resources.collector.collect_episode(
                epsilon=0.0,
                deterministic=True,
                record=True,
                trajectory_journal=journal,
            )

        assert len(backend.step_requests) == 1
        request = backend.step_requests[0]
        assert request.action_id == "select-CARD.BURN-0"
        assert request.action_index is None

        rollout_step = episode.unrolls[0].steps[0]
        assert rollout_step.snapshot.candidate_count == 2
        assert rollout_step.action_index == 1
        assert rollout_step.behavior_log_probability == pytest.approx(0.0)
        assert episode.metrics.maximum_observed_candidates == 6
        assert episode.metrics.maximum_observed_semantic_candidates == 2
        assert episode.metrics.maximum_equivalence_class_size == 3

        assert len(episode.transaction_traces) == 1
        transaction_step = episode.transaction_traces[0].steps[-1]
        assert transaction_step.action_index == 1
        assert transaction_step.action_fingerprint == selected_reference.equivalence_fingerprint

        records = [json.loads(line) for line in journal_path.read_text(encoding="utf-8").splitlines()]
        summary = next(item for item in records if item.get("record_kind") == "summary")
        assert summary["selected_candidate_index"] == 1
        assert summary["selected_dispatch_index"] == 3
        assert summary["selected_action_multiplicity"] == 3
        assert summary["semantic_candidate_count"] == 2
        assert summary["raw_legal_action_count"] == 6
        assert summary["maximum_candidate_multiplicity"] == 3
        assert summary["selected_action"]["card"]["id"] == "CARD.BURN"
        selected_topk = next(item for item in summary["policy_topk"] if item["candidate_index"] == 1)
        assert selected_topk["dispatch_index"] == 3
        assert selected_topk["multiplicity"] == 3
        assert selected_topk["action"]["card"]["id"] == "CARD.BURN"
    finally:
        resources.close()


def test_epsilon_exploration_balances_semantic_branches_after_strict_grouping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class UnequalGroupedSelectionBackend(StrictGroupedSelectionBackend):
        @staticmethod
        def _actions() -> tuple[dict[str, Any], ...]:
            actions: list[dict[str, Any]] = []
            for card_id, copies in (("CARD.WOUND", 1), ("CARD.BURN", 3)):
                for copy_index in range(copies):
                    actions.append(
                        {
                            "action_handle": f"select-{card_id}-{copy_index}",
                            "action": "combat_select_card",
                            "kind": "combat_select_card",
                            "model_action_kind": "card_selection",
                            "model_action_variant": "select",
                            "selection_operation": "select",
                            "enabled": True,
                            "card_index": len(actions),
                            "card": {
                                "id": card_id,
                                "pile": "Discard",
                                "cost": -2,
                                "is_selected": False,
                            },
                        }
                    )
            return tuple(actions)

        def _observation(self, *, terminal: bool) -> dict[str, Any]:
            observation = super()._observation(terminal=terminal)
            selection = observation["card_selection"]
            assert isinstance(selection, dict)
            selection["cards"] = [
                {
                    "card_instance_id": f"instance-{index}",
                    "id": "CARD.WOUND" if index == 0 else "CARD.BURN",
                    "pile": "Discard",
                    "cost": -2,
                    "is_selected": False,
                }
                for index in range(4)
            ]
            return observation

    class FixedCandidateRng:
        def __init__(self) -> None:
            self.probabilities: np.ndarray[Any, np.dtype[np.float64]] | None = None

        def choice(
            self,
            candidate_count: int,
            *,
            p: np.ndarray[Any, np.dtype[np.float64]],
        ) -> int:
            assert candidate_count == 2
            self.probabilities = np.asarray(p, dtype=np.float64)
            return 0

    backend = UnequalGroupedSelectionBackend()
    resources = build_training_resources(_config(total_steps=1), backend=backend)
    fixed_rng = FixedCandidateRng()
    monkeypatch.setattr(resources.collector, "_rng", fixed_rng)
    try:
        encoded = resources.encoder.encode(
            backend._observation(terminal=False),
            backend._actions(),
            device="cpu",
        )
        assert encoded.snapshot.candidate_count == 2
        assert [reference.multiplicity for reference in encoded.actions] == [1, 3]

        episode = resources.collector.collect_episode(
            epsilon=1.0,
            deterministic=False,
            record=True,
            policy_version=0,
        )

        # The raw 1:3 multiplicity is collapsed before exploration. Both
        # semantic candidates are in one branch, so each receives half of that
        # branch rather than probabilities 1/4 and 3/4. The exact selected
        # behavior probability is what enters V-trace.
        assert fixed_rng.probabilities is not None
        np.testing.assert_allclose(
            fixed_rng.probabilities,
            np.asarray([0.5, 0.5]),
            rtol=0.0,
            atol=1e-15,
        )
        step = episode.unrolls[0].steps[0]
        assert step.action_index == 0
        assert step.behavior_log_probability == pytest.approx(math.log(0.5))
        assert episode.metrics.maximum_observed_candidates == 4
        assert episode.metrics.maximum_observed_semantic_candidates == 2
        assert episode.metrics.maximum_equivalence_class_size == 3
    finally:
        resources.close()


def test_baseline_inspection_exposes_active_shapes_separately_from_capacities() -> None:
    config = _config()
    report = inspect_baseline(config)
    assert report["pipeline"] == ("bounded-fifo-async-vtrace-failure-credit-v5-v8")
    assert report["active_shape_batching"] is True
    assert report["deterministic_probe_environment_steps"] == []
    assert report["encoding_capacities"]["candidates"] == 6
    assert report["candidate_shape"][1] < 6
    assert report["episodic_learning"]["macro_sample_fraction"] == 0.0
    assert report["failure_credit"]["sampling_quotas"][EvidenceStratum.UNRESOLVED_STALL.value] == 1
    assert report["failure_credit"]["censored_quota"] == 0


def test_cpu_resource_build_does_not_seed_or_initialize_unused_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    accelerator_seed_calls: list[int] = []

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    monkeypatch.setattr(
        torch.cuda,
        "manual_seed_all",
        lambda seed: accelerator_seed_calls.append(int(seed)),
    )

    def unexpected_cuda_rng_read() -> list[torch.Tensor]:
        raise AssertionError("CPU resource construction initialized CUDA RNG state")

    monkeypatch.setattr(torch.cuda, "get_rng_state_all", unexpected_cuda_rng_read)
    resources = build_training_resources(_config(), backend=FakeCombatBackend())
    try:
        assert resources.device.type == "cpu"
        assert accelerator_seed_calls == []
    finally:
        resources.close()


def test_resource_composition_failure_closes_only_factory_owned_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    internally_created = FakeCombatBackend()
    externally_owned = FakeCombatBackend()
    monkeypatch.setattr(
        factory_module,
        "build_backend",
        lambda _config: internally_created,
    )

    def fail_learner_construction(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("injected learner construction failure")

    monkeypatch.setattr(
        factory_module,
        "VTraceLearner",
        fail_learner_construction,
    )

    with pytest.raises(RuntimeError, match="injected learner construction failure"):
        build_training_resources(_config())
    assert internally_created.closed

    with pytest.raises(RuntimeError, match="injected learner construction failure"):
        build_training_resources(_config(), backend=externally_owned)
    assert not externally_owned.closed


def test_in_process_frozen_cuda_evaluation_fails_before_untracked_rng_init(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)

    with pytest.raises(RuntimeError, match="requires CUDA to be initialized"):
        evaluate_checkpoint_policy(
            tmp_path / "checkpoint-not-read",
            output_directory=tmp_path / "evaluation-not-created",
            episodes=1,
            device="cuda",
            collector_device="cpu",
            backend=FakeCombatBackend(),
        )

    assert not (tmp_path / "evaluation-not-created").exists()


def test_collector_can_adopt_new_policy_only_between_complete_unrolls() -> None:
    resources = build_training_resources(
        _config(total_steps=4),
        backend=FakeCombatBackend(terminal_step=4),
    )
    emitted = []
    progress = []

    def sink(unroll: Any) -> int:
        emitted.append(unroll)
        return 7 if len(emitted) == 1 else 9

    try:
        episode = resources.collector.collect_episode(
            record=True,
            policy_version=0,
            unroll_sink=sink,
            progress_sink=progress.append,
        )
        assert [unroll.policy_version for unroll in emitted] == [0, 7]
        assert [item.steps for item in progress] == [2, 4]
        assert [item.behavior_policy_version for item in progress] == [0, 7]
        assert all(item.max_act == 1 and item.max_floor == 1 for item in progress)
        assert all(item.maximum_observed_candidates == 2 for item in progress)
        assert episode.metrics.maximum_observed_candidates == 2
        assert episode.metrics.stall_evidence is None
        assert episode.behavior_policy_version == 7
        assert episode.actor_policy_version == 9
    finally:
        resources.close()


def test_runtime_budget_cut_bootstraps_instead_of_fabricating_preheat_loss() -> None:
    base = _config(total_steps=1)
    config = replace(
        base,
        optimization=replace(base.optimization, discount=1.0),
        curriculum=CurriculumConfig(
            mode="native-revival-preheat",
            reward_objective="combat",
            revival_mechanism=ENGINE_REVIVAL_MECHANISM,
            revival_budget=64,
            epsilon_start=0.2,
            epsilon_end=0.1,
            epsilon_decay_steps=10,
        ),
    )
    resources = build_training_resources(
        config,
        backend=FakeCombatBackend(terminal_step=10),
    )
    try:
        assert resources.collector.additional_relics == ()
        assert resources.collector.training_revival_budget == 64
        episode = resources.collector.collect_episode(
            record=True,
            maximum_steps=1,
        )
        assert episode.metrics.terminal_reason == "collection_budget"
        assert not episode.metrics.combat_won
        assert episode.metrics.reward_total > -0.1
        assert len(episode.unrolls) == 1
        assert episode.unrolls[0].steps[-1].discount == 1.0
        assert episode.unrolls[0].bootstrap_snapshot is not None
    finally:
        resources.close()

    report = inspect_baseline(config)
    assert report["revival_mechanism"] == ENGINE_REVIVAL_MECHANISM
    assert report["revival_contract"]["model_visible_game_entity"] is None


@pytest.mark.parametrize(
    ("backend", "budget", "durable_window", "repeat_threshold"),
    (
        (DynamicPreviewLoopBackend(), 2, 2, 8),
        (StaticEventLoopBackend(), 4, 100, 4),
    ),
    ids=("durable-window-at-budget", "exact-cycle-at-budget"),
)
def test_collection_budget_precedes_noncombat_liveness_boundaries(
    backend: DynamicPreviewLoopBackend,
    budget: int,
    durable_window: int,
    repeat_threshold: int,
) -> None:
    base = _event_loop_config(durable_window=durable_window)
    config = replace(
        base,
        environment=replace(base.environment, max_episode_steps=10),
        diagnostics=replace(
            base.diagnostics,
            deadlock_window=8,
            deadlock_repeat_threshold=repeat_threshold,
            noncombat_durable_progress_window=durable_window,
        ),
        episodic_learning=replace(base.episodic_learning, enabled=True),
    )
    resources = build_training_resources(config, backend=backend)
    try:
        episode = resources.collector.collect_episode(
            record=True,
            maximum_steps=budget,
        )

        assert episode.metrics.steps == budget
        assert episode.metrics.terminal_reason == "collection_budget"
        assert episode.metrics.truncated
        assert not episode.metrics.deadlocked
        assert not episode.metrics.noncombat_progress_stalled
        assert not episode.metrics.trusted_policy_failure
        assert episode.unrolls[-1].steps[-1].discount > 0.0
        assert episode.unrolls[-1].bootstrap_snapshot is not None
        assert episode.completed_episode is not None
        assert not episode.completed_episode.completion.authoritative
        assert episode.completed_episode.completion.won is None
    finally:
        resources.close()


@pytest.mark.parametrize("marker_key", ["wave_index", "stage_id"])
def test_combat_progress_tracker_preserves_phase_and_wave_resets(
    marker_key: str,
) -> None:
    tracker = _CombatNetProgressTracker(window=3, minimum_hp_fraction=0.10)

    def observation(marker: int) -> dict[str, object]:
        return {
            "combat": {
                "in_progress": True,
                marker_key: marker,
                "enemies": [{"id": "enemy", "hp": 100, "max_hp": 100}],
            }
        }

    assert tracker.observe(step=0, observation=observation(1)).age_steps == 0
    assert tracker.observe(step=1, observation=observation(1)).age_steps == 1
    reset = tracker.observe(step=2, observation=observation(2))
    assert reset.progress_kind == "phase_or_wave_advanced"
    assert reset.age_steps == 0
    after_reset = tracker.observe(step=3, observation=observation(2))
    assert after_reset.age_steps == 1
    assert not after_reset.stalled


def test_combat_progress_tracker_preserves_meaningful_net_hp_reset() -> None:
    tracker = _CombatNetProgressTracker(window=3, minimum_hp_fraction=0.10)

    def observation(hp: int) -> dict[str, object]:
        return {
            "combat": {
                "in_progress": True,
                "enemies": [{"id": "enemy", "hp": hp, "max_hp": 100}],
            }
        }

    assert tracker.observe(step=0, observation=observation(100)).age_steps == 0
    assert tracker.observe(step=1, observation=observation(100)).age_steps == 1
    reset = tracker.observe(step=2, observation=observation(89))
    assert reset.progress_kind == "meaningful_net_hp_reduction"
    assert reset.age_steps == 0
    after_reset = tracker.observe(step=3, observation=observation(89))
    assert after_reset.age_steps == 1
    assert not after_reset.stalled


def test_combat_stall_records_one_bounded_exhausted_hand_evidence(
    tmp_path: Path,
) -> None:
    base = _config(total_steps=8)
    config = replace(
        base,
        model=replace(base.model, max_world_tokens=64),
        environment=replace(base.environment, max_episode_steps=8),
        diagnostics=DiagnosticsConfig(
            deadlock_window=128,
            deadlock_repeat_threshold=8,
            combat_net_progress_window=3,
            combat_min_net_hp_fraction=0.10,
            journal_policy_topk=5,
        ),
    )
    resources = build_training_resources(
        config,
        backend=ExhaustedOnlyEndTurnBackend(terminal_step=100),
    )
    journal_path = tmp_path / "combat-stall.jsonl"
    try:
        with TrajectoryJournal(
            journal_path,
            snapshot_interval=1_000,
            anomaly_context_steps=0,
        ) as journal:
            episode = resources.collector.collect_episode(
                record=True,
                trajectory_journal=journal,
            )

        assert episode.metrics.terminal_reason == "combat_progress_stall"
        evidence = episode.metrics.stall_evidence
        assert evidence is not None
        assert evidence["anchor_enemy_hp_total"] == 100.0
        assert evidence["current_enemy_hp_total"] == 100.0
        assert evidence["net_enemy_hp_progress"] == 0.0
        assert evidence["required_net_enemy_hp_progress"] == 10.0
        assert evidence["hand_cards"] == 10
        assert evidence["draw_cards"] == 0
        assert evidence["discard_cards"] == 0
        assert evidence["exhaust_cards"] == 1
        assert evidence["legal_action_kinds"] == {"end_turn": 1}
        preview = evidence["hand_card_preview"]
        assert isinstance(preview, tuple)
        assert len(preview) == 8
        assert all(card["is_playable"] is False for card in preview)
        assert evidence["hand_card_preview_truncated"] is True
        assert len(json.dumps(evidence, sort_keys=True)) < 4_096

        records = [json.loads(line) for line in journal_path.read_text(encoding="utf-8").splitlines()]
        terminal_stalls = [
            record
            for record in records
            if record.get("record_kind") == "summary"
            and isinstance(record.get("deadlock"), dict)
            and record["deadlock"].get("kind") == "combat_no_net_progress"
        ]
        assert len(terminal_stalls) == 1
        assert terminal_stalls[0]["deadlock"]["draw_cards"] == 0
        assert terminal_stalls[0]["deadlock"]["discard_cards"] == 0
        assert terminal_stalls[0]["deadlock"]["legal_action_kinds"] == {"end_turn": 1}
    finally:
        resources.close()


def test_combat_without_net_enemy_hp_progress_ends_as_diagnosed_deadlock() -> None:
    base = _config(total_steps=12)
    config = replace(
        base,
        environment=replace(base.environment, max_episode_steps=12),
        diagnostics=DiagnosticsConfig(
            deadlock_window=128,
            deadlock_repeat_threshold=8,
            combat_net_progress_window=3,
            combat_min_net_hp_fraction=0.05,
            journal_policy_topk=5,
        ),
    )
    resources = build_training_resources(
        config,
        backend=FakeCombatBackend(terminal_step=100),
    )
    progress = []
    try:
        episode = resources.collector.collect_episode(
            record=True,
            progress_sink=progress.append,
        )
        assert episode.metrics.steps == 9
        assert episode.metrics.deadlocked
        assert episode.metrics.combat_progress_stalled
        assert episode.metrics.terminal_reason == "combat_progress_stall"
        assert episode.metrics.maximum_combat_no_net_progress_steps == 3
        assert progress[-1].combat_in_progress
        assert progress[-1].combat_no_net_progress_steps == 3
        assert progress[-1].combat_anchor_enemy_hp_total == 1.0
        assert progress[-1].enemy_hp_total == 1.0
        assert progress[-1].legal_action_kinds == {
            "end_turn": 1,
            "play_card": 1,
        }
        assert sum(progress[-1].selected_action_kinds.values()) == 9
        assert progress[-1].last_selected_action_kind in {
            "end_turn",
            "play_card",
        }
    finally:
        resources.close()


def test_transient_damage_followed_by_healing_does_not_fake_combat_progress() -> None:
    base = _config(total_steps=12)
    config = replace(
        base,
        environment=replace(base.environment, max_episode_steps=12),
        diagnostics=DiagnosticsConfig(
            deadlock_window=128,
            deadlock_repeat_threshold=8,
            combat_net_progress_window=4,
            combat_min_net_hp_fraction=0.10,
            journal_policy_topk=5,
        ),
    )
    resources = build_training_resources(
        config,
        backend=OscillatingDamageBackend(terminal_step=100),
    )
    try:
        episode = resources.collector.collect_episode(record=True)
        assert episode.metrics.steps == 4
        assert episode.metrics.combat_progress_stalled
        assert episode.metrics.terminal_reason == "combat_progress_stall"
        assert episode.metrics.maximum_combat_no_net_progress_steps == 4
    finally:
        resources.close()


def test_noncombat_dynamic_preview_and_changing_labels_cannot_evade_stall(
    tmp_path: Path,
) -> None:
    resources = build_training_resources(
        _event_loop_config(),
        backend=DynamicPreviewLoopBackend(),
    )
    progress = []
    journal_path = tmp_path / "event-loop.jsonl"
    try:
        with TrajectoryJournal(
            journal_path,
            snapshot_interval=1_000,
            anomaly_context_steps=1,
        ) as journal:
            episode = resources.collector.collect_episode(
                record=True,
                progress_sink=progress.append,
                trajectory_journal=journal,
            )
        assert episode.metrics.steps == 4
        assert episode.metrics.deadlocked
        assert episode.metrics.noncombat_progress_stalled
        assert not episode.metrics.combat_progress_stalled
        assert episode.metrics.terminal_reason == "noncombat_progress_stall"
        assert episode.metrics.maximum_noncombat_no_durable_progress_steps == 4
        assert progress[-1].noncombat_no_durable_progress_steps == 4

        records = [json.loads(line) for line in journal_path.read_text(encoding="utf-8").splitlines()]
        evidence = [
            item["deadlock"]
            for item in records
            if isinstance(item.get("deadlock"), dict)
            and item["deadlock"].get("kind") == "noncombat_no_durable_progress"
        ]
        assert evidence
        assert evidence[-1]["steps_without_durable_progress"] == 4
        assert len(evidence[-1]["durable_state_fingerprint"]) == 64
        assert len(evidence[-1]["locus_fingerprint"]) == 64
        assert len(evidence[-1]["resource_fingerprint"]) == 64
        assert len(evidence[-1]["action_fingerprint"]) == 64
        assert evidence[-1]["detected_step"] == 4
        assert evidence[-1]["context"]["room_model_id"] == "EVENT.LOOP"
        assert "observation" not in evidence[-1]
        anomaly = next(
            item
            for item in records
            if item.get("record_kind") == "rich_snapshot"
            and isinstance(item.get("deadlock"), dict)
            and item["deadlock"].get("kind") == "noncombat_no_durable_progress"
        )
        assert anomaly["step_index"] == 3
        assert anomaly["result_step_index"] == 4
        assert anomaly["result_observation"]["event"]["dynamic_vars"][0]["int_value"] == 4
        assert anomaly["observation"]["event"]["dynamic_vars"][0]["int_value"] == 3
        assert "action_handle" not in anomaly["result_legal_actions"][0]
    finally:
        resources.close()


def test_event_page_cycle_ignores_vitality_revival_churn_and_credits_action() -> None:
    base = _event_loop_config(durable_window=20)
    config = replace(
        base,
        environment=replace(base.environment, max_episode_steps=40),
        diagnostics=replace(
            base.diagnostics,
            deadlock_window=8,
            deadlock_repeat_threshold=3,
            noncombat_durable_progress_window=20,
        ),
        transaction_learning=TransactionLearningConfig(
            enabled=True,
            replay_capacity=16,
            replay_byte_capacity=20_000_000,
            sample_traces=4,
            burn_in_steps=0,
        ),
        episodic_learning=replace(base.episodic_learning, enabled=True),
    )
    resources = build_training_resources(
        config,
        backend=VitalityEventCycleBackend(),
    )
    try:
        # Uniform logits make deterministic argmax choose option zero while a
        # second legal option remains available throughout the factual cycle.
        with torch.no_grad():
            for parameter in resources.model.parameters():
                parameter.zero_()
        episode = resources.collector.collect_episode(
            record=True,
            deterministic=True,
        )

        assert episode.metrics.steps == 5
        assert episode.metrics.deadlocked
        assert episode.metrics.noncombat_progress_stalled
        assert episode.metrics.noncombat_event_cycle
        assert episode.metrics.trusted_policy_failure
        assert not episode.metrics.combat_policy_failed
        assert episode.metrics.terminal_reason == "noncombat_event_action_cycle"
        assert episode.completed_episode is not None
        assert episode.completed_episode.completion.authoritative
        assert episode.completed_episode.won is False
        evidence = episode.metrics.stall_evidence
        assert evidence is not None
        assert evidence["kind"] == "noncombat_event_action_cycle"
        assert evidence["repeat_threshold"] == 3
        assert evidence["cycle_span"] == 2

        assert len(episode.transaction_traces) == 1
        trace = episode.transaction_traces[0]
        assert all(step.policy_node_key is not None for step in trace.steps)
        assert all(step.policy_action_fingerprint is not None for step in trace.steps)
        labels = factual_transaction_policy_targets(trace)
        assert labels
        assert all(label.target is TransactionPolicyTarget.AVOID for label in labels)
        assert {trace.steps[label.step_index].action_index for label in labels} == {trace.steps[0].action_index}
        assert all(np.count_nonzero(step.snapshot.action_mask) == 2 for step in trace.steps)
    finally:
        resources.close()


def test_event_cycle_requires_the_same_factual_legal_action_surface() -> None:
    base = _event_loop_config(durable_window=8)
    config = replace(
        base,
        environment=replace(base.environment, max_episode_steps=20),
        diagnostics=replace(
            base.diagnostics,
            deadlock_window=8,
            deadlock_repeat_threshold=3,
            noncombat_durable_progress_window=8,
        ),
        transaction_learning=TransactionLearningConfig(
            enabled=True,
            replay_capacity=16,
            replay_byte_capacity=20_000_000,
            sample_traces=4,
            burn_in_steps=0,
        ),
        episodic_learning=replace(base.episodic_learning, enabled=True),
    )
    resources = build_training_resources(
        config,
        backend=ChangingAlternativeEventBackend(),
    )
    try:
        with torch.no_grad():
            for parameter in resources.model.parameters():
                parameter.zero_()
        episode = resources.collector.collect_episode(record=True, deterministic=True)

        assert episode.metrics.steps == 8
        assert episode.metrics.terminal_reason == "noncombat_progress_stall"
        assert episode.metrics.noncombat_progress_stalled
        assert not episode.metrics.noncombat_event_cycle
        assert not episode.metrics.trusted_policy_failure
        assert episode.completed_episode is not None
        assert not episode.completed_episode.completion.authoritative
        assert all(trace.outcome is TransactionOutcome.CENSORED for trace in episode.transaction_traces)
        assert all(not step.q_observed for trace in episode.transaction_traces for step in trace.steps)
    finally:
        resources.close()


def test_event_cycle_policy_node_preserves_strict_group_multiplicity() -> None:
    base = _event_loop_config(durable_window=4)
    config = replace(
        base,
        environment=replace(base.environment, max_episode_steps=12),
        diagnostics=replace(
            base.diagnostics,
            deadlock_window=8,
            deadlock_repeat_threshold=3,
            noncombat_durable_progress_window=4,
        ),
        transaction_learning=TransactionLearningConfig(
            enabled=True,
            replay_capacity=16,
            replay_byte_capacity=20_000_000,
            sample_traces=4,
            burn_in_steps=0,
        ),
        episodic_learning=replace(base.episodic_learning, enabled=True),
    )
    resources = build_training_resources(
        config,
        backend=ChangingMultiplicityEventBackend(),
    )
    try:
        with torch.no_grad():
            for parameter in resources.model.parameters():
                parameter.zero_()
        episode = resources.collector.collect_episode(record=True, deterministic=True)

        assert episode.metrics.steps == 4
        assert episode.metrics.terminal_reason == "noncombat_progress_stall"
        assert episode.metrics.noncombat_progress_stalled
        assert not episode.metrics.noncombat_event_cycle
        assert not episode.metrics.trusted_policy_failure
        assert episode.completed_episode is not None
        assert not episode.completed_episode.completion.authoritative
        assert all(trace.outcome is TransactionOutcome.CENSORED for trace in episode.transaction_traces)
        assert not any(factual_transaction_policy_targets(trace) for trace in episode.transaction_traces)
    finally:
        resources.close()


def test_event_cycle_does_not_blame_one_enabled_choice_beside_locked_option() -> None:
    base = _event_loop_config(durable_window=4)
    config = replace(
        base,
        environment=replace(base.environment, max_episode_steps=12),
        diagnostics=replace(
            base.diagnostics,
            deadlock_window=8,
            deadlock_repeat_threshold=3,
            noncombat_durable_progress_window=4,
        ),
        transaction_learning=TransactionLearningConfig(
            enabled=True,
            replay_capacity=16,
            replay_byte_capacity=20_000_000,
            sample_traces=4,
            burn_in_steps=0,
        ),
        episodic_learning=replace(base.episodic_learning, enabled=True),
    )
    resources = build_training_resources(
        config,
        backend=LockedAlternativeEventBackend(),
    )
    try:
        episode = resources.collector.collect_episode(record=True, deterministic=True)

        assert episode.metrics.steps == 4
        assert episode.metrics.terminal_reason == "noncombat_progress_stall"
        assert episode.metrics.noncombat_progress_stalled
        assert not episode.metrics.noncombat_event_cycle
        assert not episode.metrics.trusted_policy_failure
        assert episode.completed_episode is not None
        assert not episode.completed_episode.completion.authoritative
        assert all(trace.outcome is TransactionOutcome.CENSORED for trace in episode.transaction_traces)
        assert not any(factual_transaction_policy_targets(trace) for trace in episode.transaction_traces)
    finally:
        resources.close()


def test_event_cycle_recurrence_window_uses_environment_steps_across_forced_gap() -> None:
    tracker = _NonCombatEventCycleTracker(window=32, repeat_threshold=2)

    def event_observation(page: str) -> dict[str, Any]:
        return {
            "phase": "event",
            "decision_domain": "event",
            "state_type": "event",
            "run": {
                "act": 1,
                "floor": 9,
                "room_type": "event",
                "room_model_id": "EVENT.FORCED_GAP",
            },
            "player": {"gold": 0, "deck_cards": [], "relics": [], "potions": []},
            "event": {
                "event_id": "EVENT.FORCED_GAP",
                "description_key": page,
                "is_finished": False,
            },
        }

    before = event_observation("EVENT.FORCED_GAP.pages.CHOICE")
    after = event_observation("EVENT.FORCED_GAP.pages.FORCED_START")
    assert (
        tracker.observe(
            step=1,
            before_observation=before,
            after_observation=after,
            action_fingerprint="choose:linger",
            policy_node_key="choice-node",
            durable_progress_kind="recurring_durable_resource_state",
        )
        is None
    )
    # The collector calls observe on forced pages too, but they carry no policy
    # node and must not preserve an actionable event choice beyond the 32-step
    # learner tail.
    for step in range(2, 35):
        assert (
            tracker.observe(
                step=step,
                before_observation=event_observation(f"EVENT.FORCED_GAP.pages.FORCED_{step}"),
                after_observation=event_observation(f"EVENT.FORCED_GAP.pages.FORCED_{step + 1}"),
                action_fingerprint=f"forced:{step}",
                policy_node_key=None,
                durable_progress_kind="recurring_durable_resource_state",
            )
            is None
        )
    assert (
        tracker.observe(
            step=35,
            before_observation=before,
            after_observation=after,
            action_fingerprint="choose:linger",
            policy_node_key="choice-node",
            durable_progress_kind="recurring_durable_resource_state",
        )
        is None
    )
    # A new recurrence wholly inside the tail is actionable again.
    evidence = tracker.observe(
        step=36,
        before_observation=before,
        after_observation=after,
        action_fingerprint="choose:linger",
        policy_node_key="choice-node",
        durable_progress_kind="recurring_durable_resource_state",
    )
    assert evidence is not None
    assert evidence.occurrences == 2
    assert evidence.cycle_span == 1


def test_sparse_event_repeat_outside_learnable_tail_is_not_authoritative() -> None:
    base = _event_loop_config(durable_window=40)
    config = replace(
        base,
        environment=replace(base.environment, max_episode_steps=50),
        diagnostics=replace(
            base.diagnostics,
            deadlock_window=128,
            deadlock_repeat_threshold=3,
            noncombat_durable_progress_window=40,
        ),
        transaction_learning=TransactionLearningConfig(
            enabled=True,
            replay_capacity=16,
            replay_byte_capacity=20_000_000,
            sample_traces=4,
            burn_in_steps=2,
        ),
        episodic_learning=replace(base.episodic_learning, enabled=True),
    )
    resources = build_training_resources(config, backend=SparseRepeatEventBackend())
    try:
        with torch.no_grad():
            for parameter in resources.model.parameters():
                parameter.zero_()
        episode = resources.collector.collect_episode(record=True, deterministic=True)

        assert episode.metrics.steps == 40
        assert episode.metrics.terminal_reason == "noncombat_progress_stall"
        assert not episode.metrics.noncombat_event_cycle
        assert not episode.metrics.trusted_policy_failure
        assert episode.completed_episode is not None
        assert not episode.completed_episode.completion.authoritative
        assert all(trace.outcome is TransactionOutcome.CENSORED for trace in episode.transaction_traces)
        assert not any(factual_transaction_policy_targets(trace) for trace in episode.transaction_traces)
    finally:
        resources.close()


def test_alternating_select_cancel_and_ui_state_do_not_reset_durable_window() -> None:
    resources = build_training_resources(
        _event_loop_config(),
        backend=DynamicPreviewLoopBackend(alternate_ui_actions=True),
    )
    try:
        episode = resources.collector.collect_episode(record=True)
        assert episode.metrics.steps == 4
        assert episode.metrics.noncombat_progress_stalled
        assert episode.metrics.terminal_reason == "noncombat_progress_stall"
        assert episode.metrics.maximum_noncombat_no_durable_progress_steps == 4
    finally:
        resources.close()


def test_three_state_durable_resource_cycle_cannot_reset_window_forever() -> None:
    resources = build_training_resources(
        _event_loop_config(),
        backend=DynamicPreviewLoopBackend(
            durable_resource_cycle=(99, 100, 101),
        ),
    )
    try:
        episode = resources.collector.collect_episode(record=True)
        assert episode.metrics.steps == 6
        assert episode.metrics.noncombat_progress_stalled
        assert episode.metrics.terminal_reason == "noncombat_progress_stall"
        assert episode.metrics.maximum_noncombat_no_durable_progress_steps == 4
    finally:
        resources.close()


@pytest.mark.parametrize("change_kind", ["gold", "deck", "room"])
def test_true_noncombat_durable_progress_resets_stall_window(
    change_kind: str,
) -> None:
    resources = build_training_resources(
        _event_loop_config(),
        backend=DynamicPreviewLoopBackend(
            durable_change_step=3,
            durable_change_kind=change_kind,
        ),
    )
    try:
        episode = resources.collector.collect_episode(record=True)
        assert episode.metrics.steps == 7
        assert episode.metrics.noncombat_progress_stalled
        assert episode.metrics.terminal_reason == "noncombat_progress_stall"
        assert episode.metrics.maximum_noncombat_no_durable_progress_steps == 4
    finally:
        resources.close()


def test_live_integer_deck_uses_deck_cards_for_durable_progress() -> None:
    resources = build_training_resources(
        _event_loop_config(),
        backend=DynamicPreviewLoopBackend(
            durable_change_step=3,
            durable_change_kind="deck",
            live_deck_shape=True,
        ),
    )
    try:
        episode = resources.collector.collect_episode(record=True)
        assert episode.metrics.steps == 7
        assert episode.metrics.noncombat_progress_stalled
        assert episode.metrics.maximum_noncombat_no_durable_progress_steps == 4
    finally:
        resources.close()


def test_environment_result_terminal_flags_prevent_deadlock_override() -> None:
    resources = build_training_resources(
        _event_loop_config(),
        backend=TerminalWithoutObservationFlagsBackend(terminal_step=4),
    )
    try:
        episode = resources.collector.collect_episode(record=True)
        assert episode.metrics.steps == 4
        assert episode.metrics.run_won
        assert not episode.metrics.combat_won
        assert not episode.metrics.deadlocked
        assert not episode.metrics.combat_progress_stalled
        assert not episode.metrics.noncombat_progress_stalled
        assert episode.metrics.terminal_reason == "run_victory"
        assert episode.unrolls[-1].steps[-1].reward > 0.0
        assert episode.unrolls[-1].steps[-1].discount == 0.0
    finally:
        resources.close()


def test_collector_records_typed_run_defeat() -> None:
    resources = build_training_resources(
        _event_loop_config(),
        backend=RunDefeatBackend(terminal_step=2),
    )
    try:
        episode = resources.collector.collect_episode(record=True)
        assert not episode.metrics.run_won
        assert not episode.metrics.combat_won
        assert episode.metrics.terminal_reason == "run_defeat"
        assert episode.unrolls[-1].steps[-1].reward < 0.0
        assert episode.unrolls[-1].steps[-1].discount == 0.0
    finally:
        resources.close()


def test_act1_collector_rejects_transition_reason_contradiction() -> None:
    base = _event_loop_config()
    config = replace(
        base,
        curriculum=replace(base.curriculum, reward_objective="act1"),
    )
    resources = build_training_resources(
        config,
        backend=Act1ContradictoryTerminalReasonBackend(terminal_step=2),
    )
    try:
        with pytest.raises(CollectionProtocolError, match="facts/result reason"):
            resources.collector.collect_episode(record=True)
    finally:
        resources.close()


def test_collector_rejects_run_terminal_without_typed_run_result() -> None:
    resources = build_training_resources(
        _event_loop_config(),
        backend=RunTerminalWithoutRunResultBackend(terminal_step=2),
    )
    try:
        with pytest.raises(CollectionProtocolError, match="typed run_result"):
            resources.collector.collect_episode(record=True)
    finally:
        resources.close()


def test_generic_stall_evidence_precedes_a_forced_semantic_event_cycle(
    tmp_path: Path,
) -> None:
    base = _event_loop_config(durable_window=4)
    config = replace(
        base,
        diagnostics=replace(
            base.diagnostics,
            deadlock_window=8,
            deadlock_repeat_threshold=4,
        ),
    )
    resources = build_training_resources(config, backend=StaticEventLoopBackend())
    journal_path = tmp_path / "simultaneous-deadlock.jsonl"
    try:
        with TrajectoryJournal(journal_path) as journal:
            episode = resources.collector.collect_episode(
                record=True,
                trajectory_journal=journal,
            )
        assert episode.metrics.noncombat_progress_stalled
        assert not episode.metrics.noncombat_event_cycle
        assert not episode.metrics.trusted_policy_failure
        assert episode.metrics.terminal_reason == "noncombat_progress_stall"
        records = [json.loads(line) for line in journal_path.read_text(encoding="utf-8").splitlines()]
        terminal_summary = next(
            item for item in records if item.get("record_kind") == "summary" and item.get("outcome") == "deadlock"
        )
        assert terminal_summary["deadlock"]["kind"] == "noncombat_no_durable_progress"
    finally:
        resources.close()


def test_collector_fails_closed_on_stale_transition_revision() -> None:
    resources = build_training_resources(_config(), backend=StaleRevisionBackend())
    try:
        with pytest.raises(CollectionProtocolError, match="stale"):
            resources.collector.collect_episode(record=True)
    finally:
        resources.close()


def test_vtrace_learner_updates_policy_value_and_recurrent_parameters() -> None:
    resources = build_training_resources(_config(), backend=FakeCombatBackend())
    try:
        unroll = resources.collector.collect_episode(record=True).unrolls[0]
        progress: list[tuple[str, dict[str, int | float]]] = []
        before = {name: value.detach().clone() for name, value in resources.model.state_dict().items()}
        metrics = resources.learner.update(
            (unroll,),
            current_policy_version=0,
            current_learner_update=0,
            schedule_policy_version=(
                resources.learner.config.entropy_decay_updates
            ),
            schedule_learner_update=0,
            progress=lambda stage, payload: progress.append((stage, payload)),
        )
        assert metrics.environment_steps == 2
        assert metrics.to_mapping()["batch_environment_steps"] == 2
        assert "environment_steps" not in metrics.to_mapping()
        assert metrics.unrolls == 1
        assert metrics.entropy_weight == pytest.approx(
            resources.learner.config.entropy_weight_end
        )
        assert torch.isfinite(torch.tensor(metrics.loss))
        assert metrics.importance_ratio_mean > 0.0
        assert progress[0][0] == "validation_complete"
        stages = [stage for stage, _ in progress]
        assert stages[:7] == [
            "validation_complete",
            "recurrent_batch_setup_start",
            "recurrent_batch_setup_complete",
            "recurrent_step_collate_start",
            "recurrent_step_collate_complete",
            "recurrent_step_forward_start",
            "recurrent_step_forward_complete",
        ]
        assert stages.index("recurrent_step_forward_complete") < stages.index(
            "recurrent_forward_progress"
        )
        assert progress[-1][0] == "optimizer_complete"
        assert any(stage == "backward_complete" for stage, _ in progress)
        assert all(payload["elapsed_ms"] >= 0.0 for _, payload in progress)
        assert any(not torch.equal(before[name], value) for name, value in resources.model.state_dict().items())
    finally:
        resources.close()


def test_async_pipeline_streams_fifo_data_and_finishes_exact_horizon() -> None:
    config = _config(total_steps=4)
    resources = build_training_resources(config, backend=FakeCombatBackend())
    pipeline = ActorLearnerPipeline(
        resources,
        total_environment_steps=4,
        starting_environment_steps=0,
        starting_policy_version=0,
        epsilon=lambda _: 0.1,
    )
    try:
        pipeline.start()
        first = resources.rollout_queue.get_batch(1, minimum=1, timeout=5.0)
        metrics = resources.learner.update(
            first,
            current_policy_version=0,
            current_learner_update=0,
        )
        assert metrics.environment_steps == 2
        pipeline.request_policy_publication(1)
        episodes = []
        while sum(item.metrics.steps for item in episodes) < 4:
            episode = pipeline.next_episode(timeout=5.0)
            assert episode is not None
            episodes.append(episode)
            pipeline.release_episode_boundary()
        pipeline.join(timeout=10.0)
        assert pipeline.environment_steps == 4
        assert sum(item.metrics.steps for item in episodes) == 4
        assert [item.behavior_policy_version for item in episodes] == [0, 1]
        assert [item.actor_policy_version for item in episodes] == [0, 1]
        assert [item.liveness_probe for item in episodes] == [False, False]
    finally:
        resources.close()


def test_async_pipeline_schedules_training_only_greedy_probe_every_n_episodes() -> None:
    resources = build_training_resources(
        _config(total_steps=4),
        backend=FakeCombatBackend(),
    )
    pipeline = ActorLearnerPipeline(
        resources,
        total_environment_steps=4,
        starting_environment_steps=0,
        starting_policy_version=0,
        starting_episode_count=0,
        deterministic_probe_interval_episodes=2,
        epsilon=lambda _: 0.75,
    )
    try:
        pipeline.start()
        episodes = []
        while sum(item.metrics.steps for item in episodes) < 4:
            episode = pipeline.next_episode(timeout=5.0)
            assert episode is not None
            episodes.append(episode)
            pipeline.release_episode_boundary()
        pipeline.join(timeout=10.0)

        assert [item.liveness_probe for item in episodes] == [False, True]
        assert resources.rollout_queue.put_count == 2
        # Both episodes use the normal even training-seed partition.  The
        # second is a deterministic epsilon-zero probe, not held-out
        # validation data.
        assert all(int(item.metrics.reset_seed) % 2 == 0 for item in episodes)
    finally:
        pipeline.stop()
        if pipeline.alive:
            pipeline.join(timeout=10.0)
        resources.close()


def test_async_pipeline_schedules_one_probe_for_crossed_step_milestones() -> None:
    resources = build_training_resources(
        _config(total_steps=6),
        backend=FakeCombatBackend(),
    )
    pipeline = ActorLearnerPipeline(
        resources,
        total_environment_steps=6,
        starting_environment_steps=0,
        starting_policy_version=0,
        deterministic_probe_environment_steps=(1, 2),
        epsilon=lambda _: 0.75,
    )
    try:
        pipeline.start()
        episodes = []
        while sum(item.metrics.steps for item in episodes) < 6:
            episode = pipeline.next_episode(timeout=5.0)
            assert episode is not None
            episodes.append(episode)
            pipeline.release_episode_boundary()
        pipeline.join(timeout=10.0)

        # The first two-step episode crosses both overdue milestones. They are
        # coalesced into the next single greedy training episode rather than a
        # burst of duplicate probes.
        assert [item.liveness_probe for item in episodes] == [False, True, False]
        assert all(int(item.metrics.reset_seed) % 2 == 0 for item in episodes)
    finally:
        pipeline.stop()
        if pipeline.alive:
            pipeline.join(timeout=10.0)
        resources.close()


def test_async_pipeline_does_not_replay_step_probes_before_resume_position() -> None:
    resources = build_training_resources(
        _config(total_steps=6),
        backend=FakeCombatBackend(),
    )
    pipeline = ActorLearnerPipeline(
        resources,
        total_environment_steps=6,
        starting_environment_steps=2,
        starting_policy_version=0,
        deterministic_probe_environment_steps=(1, 2, 3),
        epsilon=lambda _: 0.75,
    )
    try:
        pipeline.start()
        episodes = []
        while pipeline.environment_steps < 6:
            episode = pipeline.next_episode(timeout=5.0)
            assert episode is not None
            episodes.append(episode)
            pipeline.release_episode_boundary()
        pipeline.join(timeout=10.0)

        # Milestones in the checkpointed prefix (1 and 2) are not replayed.
        # The first new episode crosses 3, so only its successor is a probe.
        assert [item.liveness_probe for item in episodes] == [False, True]
    finally:
        pipeline.stop()
        if pipeline.alive:
            pipeline.join(timeout=10.0)
        resources.close()


def test_actor_waits_for_main_thread_at_episode_boundary() -> None:
    backend = FakeCombatBackend()
    resources = build_training_resources(_config(total_steps=4), backend=backend)
    pipeline = ActorLearnerPipeline(
        resources,
        total_environment_steps=4,
        starting_environment_steps=0,
        starting_policy_version=0,
        epsilon=lambda _: 0.1,
    )
    try:
        pipeline.start()
        episode = pipeline.next_episode(timeout=5.0)
        assert episode is not None
        deadline = time.monotonic() + 5.0
        while not pipeline.at_episode_boundary and time.monotonic() < deadline:
            time.sleep(0.01)
        assert pipeline.at_episode_boundary
        assert pipeline.environment_steps == 2
        assert len(backend.reset_seeds) == 1
        first_seed = backend.reset_seeds[0]

        pipeline.request_pause()
        pipeline.release_episode_boundary()
        deadline = time.monotonic() + 5.0
        while not pipeline.paused and time.monotonic() < deadline:
            time.sleep(0.01)
        assert pipeline.paused
        assert backend.reset_seeds == [first_seed]

        pipeline.resume()
        second = pipeline.next_episode(timeout=5.0)
        assert second is not None
        pipeline.release_episode_boundary()
        pipeline.join(timeout=10.0)
        assert len(backend.reset_seeds) == 2
        assert backend.reset_seeds[0] == first_seed
        assert backend.reset_seeds[1] != first_seed
    finally:
        pipeline.stop()
        if pipeline.alive:
            pipeline.join(timeout=10.0)
        resources.close()


def test_paused_actor_can_rewind_private_checkpoint_counters() -> None:
    backend = FakeCombatBackend()
    resources = build_training_resources(_config(total_steps=4), backend=backend)
    pipeline = ActorLearnerPipeline(
        resources,
        total_environment_steps=4,
        starting_environment_steps=0,
        starting_policy_version=3,
        epsilon=lambda _: 0.1,
    )
    restored_supervisor = ActorSupervisorState(
        episode_attempts=11,
        consecutive_incidents=2,
        incident_fingerprints=(("transport:reset", 3),),
        recent_incident_attempts=(4, 8, 11),
    )
    restored_state = TrainingState(
        environment_steps=1,
        episodes=5,
        policy_version=2,
        actor_policy_version=2,
    )
    try:
        pipeline.start()
        episode = pipeline.next_episode(timeout=5.0)
        assert episode is not None
        queued = len(resources.rollout_queue)
        assert queued > 0
        resources.rollout_queue.get_batch(queued, minimum=queued, timeout=1.0)

        pipeline.request_pause()
        pipeline.release_episode_boundary()
        deadline = time.monotonic() + 5.0
        while not pipeline.paused and time.monotonic() < deadline:
            time.sleep(0.01)
        assert pipeline.paused
        assert not pipeline.at_episode_boundary
        assert not pipeline.at_incident_boundary

        pipeline.restore_paused_checkpoint_state(
            restored_state,
            supervisor_state=restored_supervisor,
        )
        assert pipeline.environment_steps == 1
        assert pipeline.actor_policy_version == 2
        assert pipeline.supervisor_state == restored_supervisor
        assert pipeline.actor_progress is None
    finally:
        pipeline.stop()
        if pipeline.alive:
            pipeline.join(timeout=10.0)
        resources.close()


def test_actor_incident_preserves_emitted_prefix_and_replaces_backend() -> None:
    first_backend = RecoverableIncidentBackend(fail_step=4)
    replacement = FakeCombatBackend(terminal_step=2)
    resources = build_training_resources(_config(total_steps=4), backend=first_backend)
    pipeline = ActorLearnerPipeline(
        resources,
        total_environment_steps=4,
        starting_environment_steps=0,
        starting_policy_version=0,
        epsilon=lambda _: 0.1,
    )
    try:
        pipeline.start()
        incident = pipeline.next_episode(timeout=5.0)
        assert isinstance(incident, RecoverableActorIncident)
        assert incident.emitted_environment_steps == 2
        assert incident.validated_environment_steps == 3
        assert incident.lost_valid_prefix_steps == 1
        assert not incident.circuit_breaker_open
        old_session, new_session = pipeline.replace_backend(replacement)
        assert old_session == "fake-session"
        assert new_session == "fake-session"
        assert first_backend.closed
        assert resources.backend is replacement
        assert resources.collector.backend is replacement
        pipeline.release_incident_boundary()

        episode = pipeline.next_episode(timeout=5.0)
        assert episode is not None
        assert not isinstance(episode, RecoverableActorIncident)
        assert episode.metrics.steps == 2
        pipeline.release_episode_boundary()
        pipeline.join(timeout=10.0)
        assert pipeline.environment_steps == 4
    finally:
        pipeline.stop()
        if pipeline.alive:
            pipeline.join(timeout=10.0)
        resources.close()


def test_actor_incident_circuit_breaks_on_second_same_fingerprint() -> None:
    first_backend = RecoverableIncidentBackend(fail_step=1)
    second_backend = RecoverableIncidentBackend(fail_step=1)
    resources = build_training_resources(_config(total_steps=2), backend=first_backend)
    pipeline = ActorLearnerPipeline(
        resources,
        total_environment_steps=2,
        starting_environment_steps=0,
        starting_policy_version=0,
        epsilon=lambda _: 0.1,
    )
    try:
        pipeline.start()
        first = pipeline.next_episode(timeout=5.0)
        assert isinstance(first, RecoverableActorIncident)
        assert not first.circuit_breaker_open
        pipeline.replace_backend(second_backend)
        pipeline.release_incident_boundary()

        second = pipeline.next_episode(timeout=5.0)
        assert isinstance(second, RecoverableActorIncident)
        assert second.fingerprint_occurrences == 2
        assert second.consecutive_incidents == 2
        assert second.circuit_breaker_open
    finally:
        pipeline.stop()
        if pipeline.alive:
            pipeline.join(timeout=10.0)
        resources.close()


def test_actor_incident_keeps_unflushed_tail_candidate_maximum() -> None:
    base = _config(total_steps=4)
    config = replace(base, model=replace(base.model, max_candidates=256))
    resources = build_training_resources(
        config,
        backend=WideRecoverableIncidentBackend(fail_step=2),
    )
    pipeline = ActorLearnerPipeline(
        resources,
        total_environment_steps=4,
        starting_environment_steps=0,
        starting_policy_version=0,
        epsilon=lambda _: 0.1,
    )
    try:
        pipeline.start()
        incident = pipeline.next_episode(timeout=5.0)
        assert isinstance(incident, RecoverableActorIncident)
        # The one accepted step is shorter than the two-step unroll, so there
        # is no EpisodeProgress snapshot to carry this diagnostic.
        assert incident.emitted_environment_steps == 0
        assert incident.validated_environment_steps == 1
        assert incident.lost_valid_prefix_steps == 1
        assert pipeline.actor_progress is None
        assert incident.maximum_observed_candidates == 111
    finally:
        pipeline.stop()
        if pipeline.alive:
            pipeline.join(timeout=10.0)
        resources.close()


def test_runtime_checkpoints_each_crossed_episode_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("STS2_ARTIFACT_ROOT", str(tmp_path))
    base = _config(total_steps=4)
    config = replace(
        base,
        runtime=replace(
            base.runtime,
            log_dir="runs/boundary-checkpoint",
            checkpoint_dir="checkpoints/boundary-checkpoint",
            checkpoint_interval_steps=2,
        ),
    )

    state = run_training(config, backend=FakeCombatBackend())

    assert state.environment_steps == 4
    assert state.episodes == 2
    assert state.maximum_observed_candidates == 2
    checkpoint_root = tmp_path / "checkpoints" / "boundary-checkpoint"
    periodic = sorted(checkpoint_root.glob("run-*/periodic-*"))
    assert len(periodic) == 2
    assert all((path / "checkpoint.manifest.json").is_file() for path in periodic)
    assert all((path / "metadata.json").is_file() for path in periodic)
    first_metadata, second_metadata = [
        json.loads((path / "metadata.json").read_text(encoding="utf-8")) for path in periodic
    ]
    assert first_metadata["provenance"]["checkpoint_load_mode"] == "fresh"
    assert first_metadata["provenance"]["parent_checkpoint"] is None
    assert second_metadata["provenance"]["checkpoint_load_mode"] == "in_process_successor"
    assert second_metadata["provenance"]["parent_checkpoint"]["relation"] == "in_process_successor"
    assert second_metadata["provenance"]["parent_checkpoint"]["checkpoint_id"] == first_metadata["checkpoint_id"]
    assert second_metadata["training_state"]["maximum_observed_candidates"] == 2
    final = next(checkpoint_root.glob("run-*/final-*"))
    final_metadata = json.loads((final / "metadata.json").read_text(encoding="utf-8"))
    assert final_metadata["provenance"]["checkpoint_load_mode"] == "in_process_successor"
    assert final_metadata["provenance"]["parent_checkpoint"]["checkpoint_id"] == second_metadata["checkpoint_id"]
    assert final_metadata["provenance"]["parent_checkpoint"]["relation"] == "in_process_successor"

    metrics_path = next((tmp_path / "runs" / "boundary-checkpoint").glob("run-*/metrics.jsonl"))
    metrics_text = metrics_path.read_text(encoding="utf-8")
    assert metrics_text.count('"event": "train_episode"') == 2
    assert metrics_text.count('"event": "checkpoint"') == 2
    assert '"behavior_policy_version": 0' in metrics_text
    assert '"actor_progress": {' in metrics_text
    assert '"maximum_observed_candidates": 2' in metrics_text
    assert '"run_maximum_observed_candidates": 2' in metrics_text


@pytest.mark.parametrize("mode", ["shadow", "learning"])
def test_runtime_failure_credit_modes_preserve_shadow_isolation_and_episode_order(
    mode: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A one-batch episode publishes evidence before its only learner sample.

    This is also the shadow-mode isolation guard: shadow compiles and reports
    the same episode evidence but owns neither liveness heads nor replay and
    supplies no failure-credit plans to the learner.
    """

    monkeypatch.setenv("STS2_ARTIFACT_ROOT", str(tmp_path))
    base = _config(total_steps=2)
    config = replace(
        base,
        transaction_learning=TransactionLearningConfig(enabled=False),
        failure_credit=FailureCreditConfig(
            mode=mode,
            replay_capacity=8,
            replay_byte_capacity=10_000_000,
            sample_records=1,
            burn_in_steps=1,
            maximum_context_steps=8,
            maximum_episode_completion_bytes=1_000_000,
            direct_witness_quota=0,
            multi_edge_cycle_quota=0,
            risk_sequence_quota=0,
            unresolved_stall_quota=0,
            completion_control_quota=1,
            matched_outcome_pair_quota=0,
        ),
        runtime=replace(
            base.runtime,
            log_dir=f"runs/failure-credit-{mode}",
            checkpoint_dir=f"checkpoints/failure-credit-{mode}",
            checkpoint_interval_steps=100,
        ),
    )

    state = run_training(config, backend=FakeCombatBackend())

    assert state.environment_steps == 2
    assert state.episodes == 1
    assert state.learner_updates == state.consumed_unrolls == 1
    metrics_path = next((tmp_path / "runs" / f"failure-credit-{mode}").glob("run-*/metrics.jsonl"))
    events = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines()]
    run_start = next(item for item in events if item["event"] == "run_start")
    assert run_start["pipeline"] == ("bounded-fifo-async-vtrace-failure-credit-v5-v8")
    train_episode = next(item for item in events if item["event"] == "train_episode")
    assert train_episode["failure_credit_mode"] == mode
    assert train_episode["failure_credit_records_emitted"] >= 1
    assert train_episode["failure_credit_shadow_funnel"]["records"] >= 1
    learner_start = next(item for item in events if item["event"] == "learner_update_start")
    learner_update = next(item for item in events if item["event"] == "learner_update")
    checkpoint = next((tmp_path / "checkpoints" / f"failure-credit-{mode}").glob("run-*/final-*"))
    metadata = json.loads((checkpoint / "metadata.json").read_text(encoding="utf-8"))

    if mode == "learning":
        assert train_episode["failure_credit_records_stored"] >= 1
        assert train_episode["failure_credit_replay"]["put_batch_count"] == 1
        assert learner_start["failure_credit_records"] >= 1
        assert learner_start["failure_credit_replay"]["sample_request_count"] == 1
        assert events.index(train_episode) < events.index(learner_start)
        assert learner_start["failure_credit_quota"]["satisfied"] is True
        assert learner_update["liveness_credit_plans"] >= 1
        assert learner_update["liveness_value_labels"] >= 1
        assert learner_update["liveness_q_labels"] >= 1
        # Generic completion is a zero-cost critic control, not PREFER.
        assert learner_update["liveness_completion_policy_labels"] == 0
        assert metadata["liveness_cost_heads_enabled"] is True
        assert metadata["failure_credit_replay_enabled"] is True
        assert metadata["failure_credit_replay_spec"]["size"] >= 1
        assert metadata["failure_credit_replay_spec"]["put_batch_count"] == 1
        assert metadata["failure_credit_replay_spec"]["sample_request_count"] == 1
        assert (checkpoint / "failure_credit_replay.pkl").is_file()
    else:
        assert train_episode["failure_credit_records_stored"] == 0
        assert learner_start["failure_credit_records"] == 0
        assert learner_start["failure_credit_quota"] is None
        assert learner_update["liveness_credit_plans"] == 0
        assert learner_update["liveness_cost_labels"] == 0
        assert learner_update["liveness_credit_loss"] == 0.0
        assert metadata["liveness_cost_heads_enabled"] is False
        assert metadata["failure_credit_replay_enabled"] is False
        assert metadata["failure_credit_replay_spec"] is None
        assert not (checkpoint / "failure_credit_replay.pkl").exists()


def test_terminal_native_exit_cannot_flush_pending_before_evidence_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reproduce a boundary message becoming visible just after native exit."""

    monkeypatch.setenv("STS2_ARTIFACT_ROOT", str(tmp_path))
    real_pipeline = ActorLearnerPipeline

    class DelayedTerminalBoundaryPipeline:
        def __init__(self, *args: object, **kwargs: object) -> None:
            self._inner = real_pipeline(*args, **kwargs)
            self._held_result: object | None = None
            self._delayed_once = False

        @property
        def alive(self) -> bool:
            # Simulate the native worker having exited immediately after it
            # published the boundary into its transport, before the runtime's
            # first non-blocking poll observed that message.
            if self._held_result is not None:
                return False
            return self._inner.alive

        def next_episode(self, *, timeout: float | None = None) -> object | None:
            if self._held_result is not None:
                result = self._held_result
                self._held_result = None
                return result
            result = self._inner.next_episode(timeout=timeout)
            if result is not None and not self._delayed_once:
                self._delayed_once = True
                self._held_result = result
                self._inner.resources.rollout_queue.close()
                return None
            return result

        def __getattr__(self, name: str) -> object:
            return getattr(self._inner, name)

    monkeypatch.setattr(
        runtime_module,
        "ActorLearnerPipeline",
        DelayedTerminalBoundaryPipeline,
    )
    base = _config(total_steps=2)
    config = replace(
        base,
        failure_credit=FailureCreditConfig(
            mode="learning",
            replay_capacity=8,
            replay_byte_capacity=10_000_000,
            maximum_episode_completion_bytes=10_000_000,
            sample_records=1,
            burn_in_steps=1,
            maximum_context_steps=8,
            direct_witness_quota=0,
            multi_edge_cycle_quota=0,
            risk_sequence_quota=0,
            unresolved_stall_quota=0,
            completion_control_quota=1,
            matched_outcome_pair_quota=0,
        ),
        runtime=replace(
            base.runtime,
            log_dir="runs/failure-credit-terminal-race",
            checkpoint_dir="checkpoints/failure-credit-terminal-race",
            checkpoint_interval_steps=100,
        ),
    )

    state = run_training(config, backend=FakeCombatBackend())

    assert state.learner_updates == state.consumed_unrolls == 1
    metrics_path = next((tmp_path / "runs" / "failure-credit-terminal-race").glob("run-*/metrics.jsonl"))
    events = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines()]
    train_index = next(index for index, item in enumerate(events) if item["event"] == "train_episode")
    learner_index, learner = next(
        (index, item) for index, item in enumerate(events) if item["event"] == "learner_update_start"
    )
    assert train_index < learner_index
    assert learner["failure_credit_records"] >= 1
    assert learner["failure_credit_quota"]["satisfied"] is True


def test_boundary_result_visible_before_rollout_tail_is_drained_before_release(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep separate rollout/result transports ordered at a budget boundary."""

    monkeypatch.setenv("STS2_ARTIFACT_ROOT", str(tmp_path))
    real_pipeline = ActorLearnerPipeline

    class BoundaryFirstRolloutQueue:
        def __init__(self, inner: object) -> None:
            self._inner = inner
            self.pipeline: ActorLearnerPipeline | None = None
            self._hid_first_read = False

        def get_batch(self, *args: object, **kwargs: object) -> object:
            if not self._hid_first_read:
                if self.pipeline is None:  # pragma: no cover - construction invariant
                    raise RuntimeError("test rollout proxy has no actor pipeline")
                deadline = time.monotonic() + 5.0
                while not self.pipeline.at_episode_boundary:
                    if time.monotonic() >= deadline:
                        raise TimeoutError("actor did not reach the test boundary")
                    time.sleep(0.001)
                self._hid_first_read = True
                return ()
            return self._inner.get_batch(*args, **kwargs)  # type: ignore[attr-defined,no-any-return]

        def __len__(self) -> int:
            return len(self._inner)  # type: ignore[arg-type]

        def __getattr__(self, name: str) -> object:
            return getattr(self._inner, name)

    class BoundaryFirstPipeline:
        def __init__(self, *args: object, **kwargs: object) -> None:
            resources = args[0]
            original_queue = resources.rollout_queue  # type: ignore[attr-defined]
            proxy = BoundaryFirstRolloutQueue(original_queue)
            resources.rollout_queue = proxy  # type: ignore[attr-defined,assignment]
            self._inner = real_pipeline(*args, **kwargs)
            proxy.pipeline = self._inner

        def __getattr__(self, name: str) -> object:
            return getattr(self._inner, name)

    monkeypatch.setattr(runtime_module, "ActorLearnerPipeline", BoundaryFirstPipeline)
    # Reproduce the production v29 terminal tail: seventeen unrolls, learner
    # batches of four, and one non-divisible remainder, all owned by the same
    # collection-budget boundary.
    base = _config(total_steps=34)
    config = replace(
        base,
        optimization=replace(base.optimization, batch_unrolls=4),
        rollout=replace(base.rollout, queue_capacity=24),
        environment=replace(base.environment, max_episode_steps=1_000),
        diagnostics=replace(
            base.diagnostics,
            deadlock_window=1_000,
            deadlock_repeat_threshold=999,
            combat_net_progress_window=1_000,
            noncombat_durable_progress_window=1_000,
        ),
        transaction_learning=TransactionLearningConfig(enabled=False),
        failure_credit=FailureCreditConfig(
            mode="learning",
            replay_capacity=8,
            replay_byte_capacity=10_000_000,
            maximum_episode_completion_bytes=10_000_000,
            sample_records=1,
            burn_in_steps=1,
            maximum_context_steps=8,
            direct_witness_quota=0,
            multi_edge_cycle_quota=0,
            risk_sequence_quota=0,
            unresolved_stall_quota=0,
            completion_control_quota=1,
            matched_outcome_pair_quota=0,
        ),
        runtime=replace(
            base.runtime,
            log_dir="runs/boundary-first-tail",
            checkpoint_dir="checkpoints/boundary-first-tail",
            checkpoint_interval_steps=100,
        ),
    )

    state = run_training(config, backend=FakeCombatBackend(terminal_step=10_000))

    assert state.environment_steps == 34
    assert state.episodes == 1
    assert state.learner_updates == 5
    assert state.consumed_unrolls == 17
    metrics_path = next((tmp_path / "runs" / "boundary-first-tail").glob("run-*/metrics.jsonl"))
    events = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines()]
    train_index = next(index for index, item in enumerate(events) if item["event"] == "train_episode")
    learner_index = next(index for index, item in enumerate(events) if item["event"] == "learner_update_start")
    drain = next(item for item in events if item["event"] == "rollout_boundary_drain")
    assert train_index < learner_index
    assert drain["boundary_kind"] == "episode"
    assert drain["unrolls"] == 17
    assert drain["rollout_steps"] == 34
    assert any(item["event"] == "run_complete" for item in events)


def test_runtime_final_audit_matches_final_policy_after_all_learning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("STS2_ARTIFACT_ROOT", str(tmp_path))
    base = _config(total_steps=4)
    config = replace(
        base,
        model=replace(
            base.model,
            max_world_tokens=512,
            max_candidates=8,
            max_candidate_local_tokens=32,
        ),
        runtime=replace(
            base.runtime,
            log_dir="runs/final-audit-policy",
            checkpoint_dir="checkpoints/final-audit-policy",
            checkpoint_interval_steps=100,
            # Final-audit steps are eligibility thresholds, but the audit is
            # terminal: it must cover the fully drained policy saved below.
            final_audit_steps=(2,),
            final_audit_episodes=1,
        ),
    )

    state = run_training(config, backend=FakeCombatBackend())
    assert state.learner_updates == state.consumed_unrolls == 2
    assert state.evaluation_episodes == 1

    metrics_path = next((tmp_path / "runs" / "final-audit-policy").glob("run-*/metrics.jsonl"))
    events = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines()]
    final_audits = [
        (index, event)
        for index, event in enumerate(events)
        if event.get("event") == "evaluation" and event.get("gate_kind") == "final_audit"
    ]
    assert len(final_audits) == 1
    final_audit_index, final_audit = final_audits[0]
    run_complete_index = next(index for index, event in enumerate(events) if event.get("event") == "run_complete")
    assert final_audit_index < run_complete_index
    assert not [
        event
        for event in events[final_audit_index + 1 :]
        if event.get("event") in {"learner_update_start", "learner_update"}
    ]

    final_checkpoint = next((tmp_path / "checkpoints" / "final-audit-policy").glob("run-*/final-*"))
    final_metadata = json.loads((final_checkpoint / "metadata.json").read_text(encoding="utf-8"))
    final_state = final_metadata["training_state"]
    evaluation_context = final_audit["evaluation_context"]
    assert evaluation_context["actual_environment_steps"] == state.environment_steps
    assert evaluation_context["policy_version"] == state.policy_version
    assert evaluation_context["actor_policy_version"] == state.policy_version
    assert final_state["policy_version"] == state.policy_version
    assert final_state["actor_policy_version"] == state.policy_version
    assert final_state["evaluation_episodes"] == state.evaluation_episodes == 1
    assert final_metadata["evaluation_state"]["completed_final_audit_steps"] == [2]

    def checkpoint_model_sha256(filename: str) -> str:
        payload = torch.load(
            final_checkpoint / filename,
            map_location="cpu",
            weights_only=True,
        )
        assert isinstance(payload, dict)
        digest = hashlib.sha256()
        for name, tensor in sorted(payload.items()):
            value = tensor.detach().cpu().contiguous()
            digest.update(name.encode("utf-8"))
            digest.update(str(value.dtype).encode("ascii"))
            digest.update(
                json.dumps(
                    list(value.shape),
                    separators=(",", ":"),
                ).encode("ascii")
            )
            digest.update(value.numpy().tobytes(order="C"))
        return digest.hexdigest()

    audited_digest = evaluation_context["policy_model_state_sha256"]
    assert checkpoint_model_sha256("network.pt") == audited_digest
    assert checkpoint_model_sha256("actor_network.pt") == audited_digest


def test_crash_before_final_audit_resumes_before_gate_and_does_not_skip_audit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("STS2_ARTIFACT_ROOT", str(tmp_path))
    base = _config(total_steps=4)
    config = replace(
        base,
        model=replace(
            base.model,
            max_world_tokens=512,
            max_candidates=8,
            max_candidate_local_tokens=32,
        ),
        runtime=replace(
            base.runtime,
            log_dir="runs/final-audit-resume",
            checkpoint_dir="checkpoints/final-audit-resume",
            checkpoint_interval_steps=2,
            final_audit_steps=(4,),
            final_audit_episodes=1,
        ),
    )
    real_evaluate_training_gate = runtime_module._evaluate_training_gate

    def crash_before_audit(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("injected final-audit crash")

    monkeypatch.setattr(
        runtime_module,
        "_evaluate_training_gate",
        crash_before_audit,
    )
    with pytest.raises(RuntimeError, match="injected final-audit crash"):
        run_training(config, backend=FakeCombatBackend())

    checkpoint_root = tmp_path / "checkpoints" / "final-audit-resume"
    first_run_root = next(checkpoint_root.glob("run-*"))
    periodic = sorted(first_run_root.glob("periodic-*"))
    assert len(periodic) == 2
    resume_checkpoint = periodic[-1]
    periodic_metadata = json.loads((resume_checkpoint / "metadata.json").read_text(encoding="utf-8"))
    assert periodic_metadata["training_state"]["environment_steps"] == 4
    assert periodic_metadata["evaluation_state"] == {
        "version": "sts2-evaluation-gate-state-v1",
        "completed_validation_steps": [],
        "completed_early_validation_steps": [],
        "completed_final_audit_steps": [],
    }
    first_metrics_path = next((tmp_path / "runs" / "final-audit-resume").glob("run-*/metrics.jsonl"))
    first_events = [json.loads(line) for line in first_metrics_path.read_text(encoding="utf-8").splitlines()]
    assert not [event for event in first_events if event.get("event") == "evaluation"]

    monkeypatch.setattr(
        runtime_module,
        "_evaluate_training_gate",
        real_evaluate_training_gate,
    )
    resumed = run_training(
        config,
        backend=FakeCombatBackend(),
        resume_from=resume_checkpoint,
    )
    assert resumed.environment_steps == 4
    assert resumed.evaluation_episodes == 1
    resumed_metrics_path = next(
        path
        for path in (tmp_path / "runs" / "final-audit-resume").glob("run-*/metrics.jsonl")
        if path != first_metrics_path
    )
    resumed_events = [json.loads(line) for line in resumed_metrics_path.read_text(encoding="utf-8").splitlines()]
    resumed_audits = [
        event
        for event in resumed_events
        if event.get("event") == "evaluation" and event.get("gate_kind") == "final_audit"
    ]
    assert len(resumed_audits) == 1
    assert resumed_audits[0]["evaluation_context"]["policy_version"] == resumed.policy_version
    resumed_final = next((tmp_path / "checkpoints" / "final-audit-resume").glob("run-*/final-*"))
    resumed_final_metadata = json.loads((resumed_final / "metadata.json").read_text(encoding="utf-8"))
    assert resumed_final_metadata["evaluation_state"]["completed_final_audit_steps"] == [4]

    prior_metrics_paths = {first_metrics_path, resumed_metrics_path}
    resumed_again = run_training(
        config,
        backend=FakeCombatBackend(),
        resume_from=resumed_final,
    )
    assert resumed_again.environment_steps == 4
    assert resumed_again.evaluation_episodes == 1
    final_resume_metrics_path = next(
        path
        for path in (tmp_path / "runs" / "final-audit-resume").glob("run-*/metrics.jsonl")
        if path not in prior_metrics_paths
    )
    final_resume_events = [
        json.loads(line) for line in final_resume_metrics_path.read_text(encoding="utf-8").splitlines()
    ]
    assert not [event for event in final_resume_events if event.get("event") == "evaluation"]


def test_extended_exact_resume_reaudits_the_new_terminal_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("STS2_ARTIFACT_ROOT", str(tmp_path))
    base = _config(total_steps=2)
    first_config = replace(
        base,
        model=replace(
            base.model,
            max_world_tokens=512,
            max_candidates=8,
            max_candidate_local_tokens=32,
        ),
        runtime=replace(
            base.runtime,
            log_dir="r/fae1",
            checkpoint_dir="c/fae1",
            final_audit_steps=(2,),
            final_audit_episodes=1,
        ),
    )
    first_state = run_training(first_config, backend=FakeCombatBackend())
    assert first_state.environment_steps == 2
    assert first_state.evaluation_episodes == 1
    first_final = next((tmp_path / "c" / "fae1").glob("run-*/final-*"))

    extended_config = replace(
        first_config,
        runtime=replace(
            first_config.runtime,
            total_environment_steps=4,
            log_dir="r/fae2",
            checkpoint_dir="c/fae2",
        ),
    )
    extended_state = run_training(
        extended_config,
        backend=FakeCombatBackend(),
        resume_from=first_final,
    )

    assert extended_state.environment_steps == 4
    assert extended_state.evaluation_episodes == 2
    second_metrics = next((tmp_path / "r" / "fae2").glob("run-*/metrics.jsonl"))
    second_events = [json.loads(line) for line in second_metrics.read_text(encoding="utf-8").splitlines()]
    audits = [
        event
        for event in second_events
        if event.get("event") == "evaluation" and event.get("gate_kind") == "final_audit"
    ]
    assert len(audits) == 1
    assert audits[0]["evaluation_context"]["policy_version"] == (extended_state.policy_version)
    audit_index = second_events.index(audits[0])
    assert not [
        event
        for event in second_events[audit_index + 1 :]
        if event.get("event") in {"learner_update_start", "learner_update"}
    ]
    second_final = next((tmp_path / "c" / "fae2").glob("run-*/final-*"))
    second_metadata = json.loads((second_final / "metadata.json").read_text(encoding="utf-8"))
    assert second_metadata["training_state"]["policy_version"] == (extended_state.policy_version)
    assert second_metadata["evaluation_state"]["completed_final_audit_steps"] == [2]


def test_runtime_recovers_one_infrastructure_abort_without_fabricating_episode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("STS2_ARTIFACT_ROOT", str(tmp_path))
    first = RecoverableIncidentBackend(fail_step=4)
    replacement = FakeCombatBackend(terminal_step=2)
    backends = iter((first, replacement))

    def next_backend(_config: TrainingConfig) -> FakeCombatBackend:
        return next(backends)

    monkeypatch.setattr(factory_module, "build_backend", next_backend)
    monkeypatch.setattr(runtime_module, "build_backend", next_backend)
    base = _config(total_steps=4)
    config = replace(
        base,
        runtime=replace(
            base.runtime,
            log_dir="runs/recoverable-incident",
            checkpoint_dir="checkpoints/recoverable-incident",
        ),
    )

    state = run_training(config)

    assert state.environment_steps == 4
    assert state.episodes == 1
    assert first.closed
    metrics_path = next((tmp_path / "runs" / "recoverable-incident").glob("run-*/metrics.jsonl"))
    events = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines()]
    assert sum(item["event"] == "backend_protocol_incident" for item in events) == 1
    assert sum(item["event"] == "episode_aborted" for item in events) == 1
    assert sum(item["event"] == "backend_restart" for item in events) == 1
    assert sum(item["event"] == "train_episode" for item in events) == 1
    aborted = next(item for item in events if item["event"] == "episode_aborted")
    assert aborted["emitted_environment_steps"] == 2
    assert aborted["lost_valid_prefix_steps"] == 1
    assert aborted["task_terminal_or_reward_fabricated"] is False


def test_exact_resume_restores_actor_incident_circuit_breaker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("STS2_ARTIFACT_ROOT", str(tmp_path))
    base = _config(total_steps=2)
    first_config = replace(
        base,
        runtime=replace(
            base.runtime,
            log_dir="runs/supervisor-source",
            checkpoint_dir="checkpoints/supervisor-source",
        ),
    )
    first_backends = iter(
        (
            RecoverableIncidentBackend(fail_step=1),
            FakeCombatBackend(terminal_step=2),
        )
    )

    def first_next_backend(_config: TrainingConfig) -> FakeCombatBackend:
        return next(first_backends)

    monkeypatch.setattr(factory_module, "build_backend", first_next_backend)
    monkeypatch.setattr(runtime_module, "build_backend", first_next_backend)
    first_state = run_training(first_config)
    assert first_state.environment_steps == 2
    assert first_state.episodes == 1

    checkpoint = next((tmp_path / "checkpoints" / "supervisor-source").glob("run-*/final-*"))
    metadata = json.loads((checkpoint / "metadata.json").read_text(encoding="utf-8"))
    restored_supervisor = checkpointing_module.actor_supervisor_state_from_metadata(metadata)
    assert restored_supervisor == ActorSupervisorState(
        episode_attempts=2,
        consecutive_incidents=0,
        incident_fingerprints=((RecoverableTestProtocolError.fingerprint, 1),),
        recent_incident_attempts=(1,),
    )

    second_backend = RecoverableIncidentBackend(fail_step=1)

    def second_next_backend(_config: TrainingConfig) -> FakeCombatBackend:
        return second_backend

    monkeypatch.setattr(factory_module, "build_backend", second_next_backend)
    monkeypatch.setattr(runtime_module, "build_backend", second_next_backend)
    second_config = replace(
        first_config,
        runtime=replace(
            first_config.runtime,
            total_environment_steps=4,
            log_dir="runs/supervisor-resume",
            checkpoint_dir="checkpoints/supervisor-resume",
        ),
    )
    with pytest.raises(RuntimeError, match="circuit breaker opened"):
        run_training(second_config, resume_from=checkpoint)

    metrics_path = next((tmp_path / "runs" / "supervisor-resume").glob("run-*/metrics.jsonl"))
    events = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines()]
    breaker = next(item for item in events if item["event"] == "circuit_breaker_open")
    assert breaker["fingerprint"] == RecoverableTestProtocolError.fingerprint
    assert breaker["fingerprint_occurrences"] == 2
    assert breaker["consecutive_incidents"] == 1


def test_runtime_marks_only_the_first_post_resume_checkpoint_as_exact_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("STS2_ARTIFACT_ROOT", str(tmp_path))
    base = _config(total_steps=4)
    config = replace(
        base,
        runtime=replace(
            base.runtime,
            log_dir="runs/exact-resume-lineage",
            checkpoint_dir="checkpoints/exact-resume-lineage",
            checkpoint_interval_steps=2,
        ),
    )
    source = build_training_resources(config, backend=FakeCombatBackend())
    try:
        unroll = source.collector.collect_episode(record=True).unrolls[0]
        source.rollout_queue.put(unroll)
        source_checkpoint = save_training_checkpoint(
            tmp_path / "resume-source",
            config=config,
            resources=source,
            state=TrainingState(
                environment_steps=2,
                episodes=1,
                maximum_observed_candidates=2,
            ),
            run_id="resume-source-run",
            checkpoint_load_mode="fresh",
        )
    finally:
        source.close()

    checkpoint_reference = runtime_module._checkpoint_reference(source_checkpoint)
    assert checkpoint_reference is not None
    source_metadata = json.loads((source_checkpoint / "metadata.json").read_text(encoding="utf-8"))
    assert checkpoint_reference["checkpoint_id"] == source_metadata["checkpoint_id"]
    assert (
        checkpoint_reference["manifest_sha256"]
        == hashlib.sha256((source_checkpoint / "checkpoint.manifest.json").read_bytes()).hexdigest()
    )

    state = run_training(
        config,
        backend=FakeCombatBackend(),
        resume_from=source_checkpoint,
    )

    assert state.environment_steps == 4
    resumed_metrics = next((tmp_path / "runs" / "exact-resume-lineage").glob("run-*/metrics.jsonl"))
    resumed_events = [json.loads(line) for line in resumed_metrics.read_text(encoding="utf-8").splitlines()]
    restored_drain = next(event for event in resumed_events if event["event"] == "restored_rollout_queue_drain")
    assert restored_drain["unrolls"] == 1
    assert restored_drain["rollout_steps"] == 2
    assert restored_drain["evidence_boundary"] == "checkpoint_committed"
    checkpoint_root = tmp_path / "checkpoints" / "exact-resume-lineage"
    periodic = next(checkpoint_root.glob("run-*/periodic-*"))
    periodic_metadata = json.loads((periodic / "metadata.json").read_text(encoding="utf-8"))
    assert periodic_metadata["provenance"]["checkpoint_load_mode"] == "exact_resume"
    assert periodic_metadata["provenance"]["parent_checkpoint"]["relation"] == "loaded_parent"
    final = next(checkpoint_root.glob("run-*/final-*"))
    final_metadata = json.loads((final / "metadata.json").read_text(encoding="utf-8"))
    assert final_metadata["provenance"]["checkpoint_load_mode"] == "in_process_successor"
    assert final_metadata["provenance"]["parent_checkpoint"]["checkpoint_id"] == periodic_metadata["checkpoint_id"]


def test_exact_resume_does_not_repeat_evaluation_gate_at_checkpoint_step(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("STS2_ARTIFACT_ROOT", str(tmp_path))
    base = _config(total_steps=4)
    config = replace(
        base,
        runtime=replace(
            base.runtime,
            log_dir="runs/evaluation-resume",
            checkpoint_dir="checkpoints/evaluation-resume",
            checkpoint_interval_steps=2,
            evaluation_steps=(2,),
            evaluation_episodes=1,
        ),
    )
    source = build_training_resources(config, backend=FakeCombatBackend())
    try:
        source_checkpoint = save_training_checkpoint(
            tmp_path / "evaluation-resume-source",
            config=config,
            resources=source,
            state=TrainingState(
                environment_steps=2,
                episodes=1,
                evaluation_episodes=1,
                maximum_observed_candidates=2,
            ),
            run_id="evaluation-resume-source-run",
            checkpoint_load_mode="fresh",
        )
    finally:
        source.close()

    state = run_training(
        config,
        backend=FakeCombatBackend(),
        resume_from=source_checkpoint,
    )

    assert state.environment_steps == 4
    assert state.evaluation_episodes == 1
    metrics_path = next((tmp_path / "runs" / "evaluation-resume").glob("run-*/metrics.jsonl"))
    events = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines()]
    assert not [event for event in events if event["event"] == "evaluation"]


def test_explicit_gate_state_treats_new_past_validation_as_historical(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("STS2_ARTIFACT_ROOT", str(tmp_path))
    source_config = _config(total_steps=4)
    resume_config = replace(
        source_config,
        runtime=replace(
            source_config.runtime,
            log_dir="runs/new-past-evaluation",
            checkpoint_dir="checkpoints/new-past-evaluation",
            evaluation_steps=(2,),
            evaluation_episodes=1,
        ),
    )
    source = build_training_resources(
        source_config,
        backend=FakeCombatBackend(),
    )
    try:
        source_checkpoint = save_training_checkpoint(
            tmp_path / "new-past-evaluation-source",
            config=source_config,
            resources=source,
            state=TrainingState(
                environment_steps=2,
                episodes=1,
                maximum_observed_candidates=2,
            ),
            run_id="new-past-evaluation-source-run",
            checkpoint_load_mode="fresh",
            evaluation_state=EvaluationGateState(),
        )
    finally:
        source.close()

    state = run_training(
        resume_config,
        backend=FakeCombatBackend(),
        resume_from=source_checkpoint,
    )

    assert state.environment_steps == 4
    assert state.evaluation_episodes == 0
    metrics_path = next((tmp_path / "runs" / "new-past-evaluation").glob("run-*/metrics.jsonl"))
    events = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines()]
    assert not [event for event in events if event["event"] == "evaluation"]
    final_checkpoint = next((tmp_path / "checkpoints" / "new-past-evaluation").glob("run-*/final-*"))
    final_metadata = json.loads((final_checkpoint / "metadata.json").read_text(encoding="utf-8"))
    assert final_metadata["evaluation_state"]["completed_validation_steps"] == [2]


def test_legacy_exact_resume_conservatively_repeats_final_audit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("STS2_ARTIFACT_ROOT", str(tmp_path))
    base = _config(total_steps=2)
    config = replace(
        base,
        model=replace(
            base.model,
            max_world_tokens=512,
            max_candidates=8,
            max_candidate_local_tokens=32,
        ),
        runtime=replace(
            base.runtime,
            log_dir="runs/legacy-final-audit-resume",
            checkpoint_dir="checkpoints/legacy-final-audit-resume",
            checkpoint_interval_steps=100,
            final_audit_steps=(2,),
            final_audit_episodes=1,
        ),
    )
    source = build_training_resources(config, backend=FakeCombatBackend())
    try:
        legacy_checkpoint = save_training_checkpoint(
            tmp_path / "legacy-final-audit-source",
            config=config,
            resources=source,
            state=TrainingState(
                environment_steps=2,
                episodes=1,
                evaluation_episodes=1,
                maximum_observed_candidates=2,
            ),
            run_id="legacy-final-audit-source-run",
            checkpoint_load_mode="fresh",
        )
    finally:
        source.close()

    legacy_metadata = json.loads((legacy_checkpoint / "metadata.json").read_text(encoding="utf-8"))
    assert "evaluation_state" not in legacy_metadata

    resumed = run_training(
        config,
        backend=FakeCombatBackend(),
        resume_from=legacy_checkpoint,
    )

    assert resumed.environment_steps == 2
    # Legacy evaluation_episodes already includes the possibly stale audit;
    # the conservative terminal-policy re-audit is deliberately additive.
    assert resumed.evaluation_episodes == 2
    metrics_path = next((tmp_path / "runs" / "legacy-final-audit-resume").glob("run-*/metrics.jsonl"))
    events = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines()]
    audits = [
        event for event in events if event.get("event") == "evaluation" and event.get("gate_kind") == "final_audit"
    ]
    assert len(audits) == 1
    assert audits[0]["evaluation_context"]["policy_version"] == resumed.policy_version

    final_checkpoint = next((tmp_path / "checkpoints" / "legacy-final-audit-resume").glob("run-*/final-*"))
    final_metadata = json.loads((final_checkpoint / "metadata.json").read_text(encoding="utf-8"))
    assert final_metadata["evaluation_state"]["completed_final_audit_steps"] == [2]


def test_evaluation_uses_odd_heldout_seeds_and_records_no_unrolls(
    tmp_path: Path,
) -> None:
    backend = FakeCombatBackend()
    resources = build_training_resources(_config(), backend=backend)
    journal_path = tmp_path / "trajectory.jsonl"
    context = {
        "schema_version": "sts2-training-evaluation-context-v1",
        "policy_version": 7,
        "policy_model_state_sha256": "test-policy-digest",
    }
    try:
        episodes, summary = evaluate_policy(
            resources,
            episodes=2,
            base_seed=6,
            journal_path=journal_path,
            evaluation_context=context,
        )
        assert len(episodes) == 2
        assert summary["evaluation_objective"] == "combat"
        assert summary["combat_win_rate_applicable"] is True
        assert summary["combat_win_rate"] == 1.0
        assert summary["act1_clear_count"] == 0
        assert summary["act3_reach_count"] == 0
        assert summary["act3_reach_rate"] == 0.0
        assert summary["maximum_observed_candidates"] == 2
        assert all(item.maximum_observed_candidates == 2 for item in episodes)
        assert all(int(seed) % 2 == 1 for seed in backend.reset_seeds)
        assert summary["data_partition"] == "validation"
        assert summary["evaluation_seeds"] == backend.reset_seeds
        assert summary["epsilon"] == 0.0
        assert summary["deterministic"] is True
        assert len(resources.rollout_queue) == 0
        journal_events = [json.loads(line) for line in journal_path.read_text(encoding="utf-8").splitlines()]
        start = journal_events[0]
        assert start["event"] == "evaluation_started"
        assert start["data_partition"] == "validation"
        assert start["evaluation_seeds"] == backend.reset_seeds
        assert start["epsilon"] == 0.0
        assert start["deterministic"] is True
        assert start["policy_version"] == 7
        assert start["policy_model_state_sha256"] == "test-policy-digest"
    finally:
        resources.close()


def test_evaluation_context_distinguishes_initialization_from_updated_successor() -> None:
    config = _config()
    resources = build_training_resources(config, backend=FakeCombatBackend())
    try:
        initial = runtime_module._evaluation_context(
            resources,
            config=config,
            state=TrainingState(),
            evaluation_gate=0,
            gate_kind="validation",
            parent_checkpoint=Path("source-checkpoint"),
            load_mode="model_initialization",
            runtime_provenance={"backend": "fake"},
        )
        assert initial["checkpoint_association"]["relation"] == "model_parameter_initialization"

        updated = runtime_module._evaluation_context(
            resources,
            config=config,
            state=TrainingState(
                environment_steps=5_000,
                episodes=2,
                policy_version=17,
                actor_policy_version=17,
            ),
            evaluation_gate=5_000,
            gate_kind="early_validation",
            parent_checkpoint=Path("source-checkpoint"),
            load_mode="model_initialization",
            runtime_provenance={"backend": "fake"},
        )
        assert updated["checkpoint_association"]["relation"] == "in_memory_successor"
        assert updated["checkpoint_association"]["load_mode"] == "model_initialization"
        assert updated["actual_environment_steps"] == 5_000
        assert updated["policy_version"] == 17
    finally:
        resources.close()


def test_final_audit_uses_never_reused_odd_namespace_and_no_replay(
    tmp_path: Path,
) -> None:
    backend = FakeCombatBackend()
    resources = build_training_resources(_config(), backend=backend)
    journal_path = tmp_path / "final-audit.jsonl"
    try:
        episodes, summary = evaluate_policy(
            resources,
            episodes=1,
            base_seed=6,
            journal_path=journal_path,
            data_partition="final_audit",
        )

        assert len(episodes) == 1
        assert summary["data_partition"] == "final_audit"
        assert summary["evaluation_seeds"] == backend.reset_seeds
        assert backend.reset_seeds[0] % 2 == 1
        assert backend.reset_seeds[0] > 100_000_000
        assert len(resources.rollout_queue) == 0
        records = [json.loads(line) for line in journal_path.read_text(encoding="utf-8").splitlines()]
        assert records[0]["event"] == "evaluation_started"
        assert records[0]["data_partition"] == "final_audit"
        decision_episode_ids = {
            str(item["episode_id"]) for item in records if item.get("event") in {"decision", "decision_snapshot"}
        }
        assert decision_episode_ids
        assert all(episode_id.startswith("final-audit-seed-") for episode_id in decision_episode_ids)
    finally:
        resources.close()


def test_training_gate_adds_diagnostic_only_macro_metrics_without_learning(
    tmp_path: Path,
) -> None:
    base = _config()
    config = replace(
        base,
        model=replace(
            base.model,
            max_world_tokens=512,
            max_candidates=8,
            max_candidate_local_tokens=32,
        ),
    )
    resources = build_training_resources(config, backend=FakeCombatBackend())
    learner_before = {key: value.detach().clone() for key, value in resources.model.state_dict().items()}
    collector_before = {key: value.detach().clone() for key, value in resources.collector_model.state_dict().items()}
    collector_state_before = deepcopy(resources.collector.state_dict())
    learner_mode_before = resources.model.training
    collector_mode_before = resources.collector_model.training
    try:
        episodes, summary = runtime_module._evaluate_training_gate(
            resources,
            episodes=1,
            base_seed=6,
            journal_path=tmp_path / "training-gate.jsonl",
            backend_factory=None,
        )

        assert len(episodes) == 1
        assert summary["combat_win_rate"] == 1.0
        telemetry = summary["macro_surface_telemetry"]
        assert telemetry["diagnostic_only"] is True
        assert telemetry["training_samples_emitted"] == 0
        assert telemetry["total_macro_decisions"] == 0
        sensitivity = summary["macro_policy_sensitivity"]
        assert sensitivity["diagnostic_only"] is True
        assert sensitivity["training_samples_emitted"] == 0
        assert sensitivity["expected_action_labels"] is False
        assert sensitivity["case_count"] == 7
        assert sensitivity["recurrent_state_contract"] == "fresh_zero_state_per_pair"

        assert len(resources.rollout_queue) == 0
        assert resources.collector.state_dict() == collector_state_before
        assert resources.model.training is learner_mode_before
        assert resources.collector_model.training is collector_mode_before
        assert set(resources.model.state_dict()) == set(learner_before)
        assert set(resources.collector_model.state_dict()) == set(collector_before)
        assert all(torch.equal(resources.model.state_dict()[key], value) for key, value in learner_before.items())
        assert all(
            torch.equal(resources.collector_model.state_dict()[key], value) for key, value in collector_before.items()
        )
    finally:
        resources.close()


def test_training_gate_does_not_terminate_on_optional_macro_telemetry_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = _config()
    config = replace(
        base,
        model=replace(
            base.model,
            max_world_tokens=512,
            max_candidates=8,
            max_candidate_local_tokens=32,
        ),
    )
    resources = build_training_resources(config, backend=FakeCombatBackend())

    def fail_macro_telemetry(*args: object, **kwargs: object) -> dict[str, object]:
        del args, kwargs
        raise ValueError("synthetic optional diagnostic failure")

    monkeypatch.setattr(
        macro_evaluation_module,
        "read_macro_journal",
        fail_macro_telemetry,
    )
    try:
        episodes, summary = runtime_module._evaluate_training_gate(
            resources,
            episodes=1,
            base_seed=6,
            journal_path=tmp_path / "training-gate-invalid-macro.jsonl",
            backend_factory=None,
        )

        assert len(episodes) == 1
        telemetry = summary["macro_surface_telemetry"]
        assert telemetry["diagnostic_only"] is True
        assert telemetry["training_samples_emitted"] == 0
        assert telemetry["valid"] is False
        assert telemetry["read_error"] == {
            "type": "ValueError",
            "message": "synthetic optional diagnostic failure",
        }
        # The actual liveness guard input remains present and independently
        # fail-closed; only optional macro telemetry is isolated.
        assert "greedy_liveness" in summary
    finally:
        resources.close()


def test_evaluation_retries_same_seed_once_on_fresh_backend(tmp_path: Path) -> None:
    first = RecoverableIncidentBackend(fail_step=2)
    replacement = FakeCombatBackend(terminal_step=2)
    resources = build_training_resources(_config(), backend=first)
    try:
        episodes, summary = evaluate_policy(
            resources,
            episodes=2,
            base_seed=6,
            journal_path=tmp_path / "retry-trajectory.jsonl",
            backend_factory=lambda: replacement,
        )
        assert len(episodes) == 2
        assert summary["infrastructure_retries"] == 1
        assert first.closed
        assert replacement.reset_seeds[0] == first.reset_seeds[0]
        assert replacement.reset_seeds == [13, 15]
        journal_events = [
            json.loads(line) for line in (tmp_path / "retry-trajectory.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        boundaries = [item for item in journal_events if item.get("event", "").startswith("evaluation_attempt_")]
        assert [item["event"] for item in boundaries[:4]] == [
            "evaluation_attempt_started",
            "evaluation_attempt_aborted",
            "evaluation_attempt_started",
            "evaluation_attempt_completed",
        ]
        first_seed_episode_ids = {
            str(item["episode_id"])
            for item in journal_events
            if item.get("event") in {"decision", "decision_snapshot"} and item.get("reset_seed") == 13
        }
        assert any("attempt-1:" in item for item in first_seed_episode_ids)
        assert any("attempt-2:" in item for item in first_seed_episode_ids)
        assert not any(item == "combat-episode-1" for item in first_seed_episode_ids)
        first_abort_index = next(
            index
            for index, item in enumerate(journal_events)
            if item.get("event") == "evaluation_attempt_aborted" and item.get("evaluation_seed") == 13
        )
        second_start_index = next(
            index
            for index, item in enumerate(journal_events)
            if item.get("event") == "evaluation_attempt_started"
            and item.get("evaluation_seed") == 13
            and item.get("attempt") == 2
        )
        first_attempt_records = [
            index
            for index, item in enumerate(journal_events)
            if item.get("event") in {"decision", "decision_snapshot"}
            and "heldout-seed-13-attempt-1:" in str(item.get("episode_id", ""))
        ]
        assert first_attempt_records
        assert max(first_attempt_records) < first_abort_index < second_start_index
    finally:
        resources.close()


def test_evaluation_marks_gate_invalid_after_repeated_infrastructure_fault() -> None:
    first = RecoverableIncidentBackend(fail_step=1)
    replacement = RecoverableIncidentBackend(fail_step=1)
    resources = build_training_resources(_config(), backend=first)
    try:
        with pytest.raises(EvaluationInfrastructureError, match="infrastructure-invalid"):
            evaluate_policy(
                resources,
                episodes=1,
                base_seed=6,
                backend_factory=lambda: replacement,
            )
    finally:
        resources.close()


def test_frozen_checkpoint_evaluation_never_loads_or_consumes_pending_queue(
    tmp_path: Path,
) -> None:
    config = _config(total_steps=4)
    source = build_training_resources(config, backend=FakeCombatBackend())
    try:
        pending = source.collector.collect_episode(record=True).unrolls[0]
        source.rollout_queue.put(pending)
        checkpoint = save_training_checkpoint(
            tmp_path / "frozen-eval-source",
            config=config,
            resources=source,
            state=TrainingState(
                environment_steps=2,
                episodes=1,
                policy_version=7,
                actor_policy_version=7,
            ),
            run_id="frozen-eval-source-run",
            checkpoint_load_mode="fresh",
        )
    finally:
        source.close()

    queue_path = checkpoint / "rollout_queue.pkl"
    queue_before = queue_path.read_bytes()
    random.seed(119)
    np.random.seed(223)
    torch.manual_seed(337)
    python_rng_before = random.getstate()
    numpy_rng_before = np.random.get_state()
    torch_rng_before = torch.get_rng_state().clone()
    cuda_rng_before = [state.clone() for state in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else None
    result = evaluate_checkpoint_policy(
        checkpoint,
        output_directory=tmp_path / "frozen-evaluation",
        episodes=2,
        base_seed=6,
        device="cpu",
        collector_device="cpu",
        backend=FakeCombatBackend(),
    )

    assert result.source_environment_steps == 2
    assert result.source_policy_version == 7
    assert result.summary["episodes"] == 2
    assert queue_path.read_bytes() == queue_before
    audit = json.loads(result.audit_path.read_text(encoding="utf-8"))
    assert audit["policy_source"] == "network.pt"
    assert audit["learner_updates_performed"] == 0
    assert audit["optimizer_loaded"] is False
    assert audit["rollout_queue_loaded"] is False
    assert audit["training_rng_loaded"] is False
    assert audit["training_checkpoint_published"] is False
    assert audit["evaluation_of"]["training_state"]["policy_version"] == 7
    assert audit["evaluation"]["macro_surface_telemetry"]["diagnostic_only"] is True
    assert audit["evaluation"]["macro_surface_telemetry"]["training_samples_emitted"] == 0
    assert random.getstate() == python_rng_before
    numpy_rng_after = np.random.get_state()
    assert numpy_rng_after[0] == numpy_rng_before[0]
    np.testing.assert_array_equal(numpy_rng_after[1], numpy_rng_before[1])
    assert numpy_rng_after[2:] == numpy_rng_before[2:]
    assert torch.equal(torch.get_rng_state(), torch_rng_before)
    if cuda_rng_before is not None:
        assert all(
            torch.equal(after, before)
            for after, before in zip(
                torch.cuda.get_rng_state_all(),
                cuda_rng_before,
                strict=True,
            )
        )


def test_failed_frozen_evaluation_cleans_staging_and_allows_same_output_retry(
    tmp_path: Path,
) -> None:
    config = _config(total_steps=4)
    source = build_training_resources(config, backend=FakeCombatBackend())
    try:
        checkpoint = save_training_checkpoint(
            tmp_path / "failed-frozen-eval-source",
            config=config,
            resources=source,
            state=TrainingState(environment_steps=2, episodes=1),
            run_id="failed-frozen-eval-source-run",
            checkpoint_load_mode="fresh",
        )
    finally:
        source.close()

    output = tmp_path / "atomic-frozen-evaluation"
    random.seed(421)
    np.random.seed(431)
    torch.manual_seed(433)
    python_rng_before = random.getstate()
    numpy_rng_before = np.random.get_state()
    torch_rng_before = torch.get_rng_state().clone()
    cuda_rng_before = [state.clone() for state in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else None
    with pytest.raises(EvaluationInfrastructureError, match="infrastructure-invalid"):
        evaluate_checkpoint_policy(
            checkpoint,
            output_directory=output,
            episodes=1,
            base_seed=6,
            device="cpu",
            collector_device="cpu",
            backend=RecoverableIncidentBackend(fail_step=1),
        )
    assert not output.exists()
    assert not tuple(tmp_path.glob(".atomic-frozen-evaluation.staging-*"))
    assert random.getstate() == python_rng_before
    numpy_rng_after = np.random.get_state()
    assert numpy_rng_after[0] == numpy_rng_before[0]
    np.testing.assert_array_equal(numpy_rng_after[1], numpy_rng_before[1])
    assert numpy_rng_after[2:] == numpy_rng_before[2:]
    assert torch.equal(torch.get_rng_state(), torch_rng_before)
    if cuda_rng_before is not None:
        assert all(
            torch.equal(after, before)
            for after, before in zip(
                torch.cuda.get_rng_state_all(),
                cuda_rng_before,
                strict=True,
            )
        )

    result = evaluate_checkpoint_policy(
        checkpoint,
        output_directory=output,
        episodes=1,
        base_seed=6,
        device="cpu",
        collector_device="cpu",
        backend=FakeCombatBackend(),
    )
    assert result.audit_path.is_file()


def test_v2_checkpoint_roundtrip_restores_models_optimizer_queue_and_rng(
    tmp_path: Path,
) -> None:
    config = _config()
    source = build_training_resources(config, backend=FakeCombatBackend())
    try:
        unroll = source.collector.collect_episode(record=True).unrolls[0]
        source.rollout_queue.put(unroll)
        state = TrainingState(
            environment_steps=2,
            learner_updates=0,
            episodes=1,
            policy_version=0,
            actor_policy_version=0,
            maximum_observed_candidates=2,
        )
        checkpoint = save_training_checkpoint(
            tmp_path / "checkpoint",
            config=config,
            resources=source,
            state=state,
            run_id="test-run",
            checkpoint_load_mode="fresh",
        )
    finally:
        source.close()

    restored = build_training_resources(config, backend=FakeCombatBackend())
    try:
        loaded = load_training_checkpoint(
            checkpoint,
            config=config,
            resources=restored,
        )
        assert loaded == state
        assert len(restored.rollout_queue) == 1
        assert restored.rollout_queue.snapshot()[0].episode_id == unroll.episode_id
        assert (checkpoint / "actor_network.pt").is_file()
        assert (checkpoint / "rollout_queue.pkl").is_file()
        assert not (checkpoint / "replay_buffer.pkl").exists()
    finally:
        restored.close()


def test_exact_resume_accepts_legacy_checkpoint_without_candidate_diagnostic() -> None:
    legacy_state = asdict(
        TrainingState(
            environment_steps=123,
            learner_updates=7,
            episodes=3,
        )
    )
    legacy_state.pop("maximum_observed_candidates")

    migrated = checkpointing_module.training_state_from_metadata({"training_state": legacy_state})

    assert migrated.environment_steps == 123
    assert migrated.learner_updates == 7
    assert migrated.episodes == 3
    assert migrated.maximum_observed_candidates == 0


def test_training_schedule_state_inherits_effective_clocks_without_counter_resume() -> None:
    source_state = TrainingState(
        environment_steps=80_000,
        learner_updates=1_250,
        policy_version=1_250,
        actor_policy_version=1_250,
    )
    source_schedule = checkpointing_module.TrainingScheduleState(
        environment_steps_offset=250_000,
        learner_updates_offset=3_900,
        policy_version_offset=3_900,
    )

    inherited = source_schedule.inherited_after(source_state)
    fresh_lineage = TrainingState()

    assert inherited.effective_environment_steps(fresh_lineage) == 330_000
    assert inherited.effective_learner_updates(fresh_lineage) == 5_150
    assert inherited.effective_policy_version(fresh_lineage) == 5_150
    assert checkpointing_module.training_schedule_state_from_metadata({}) == (
        checkpointing_module.TrainingScheduleState()
    )
    assert checkpointing_module.training_schedule_state_from_metadata(
        {"training_schedule_state": inherited.to_mapping()}
    ) == inherited


def test_training_schedule_state_rejects_partial_metadata() -> None:
    with pytest.raises(ValueError, match="training_schedule_state keys mismatch"):
        checkpointing_module.training_schedule_state_from_metadata(
            {
                "training_schedule_state": {
                    "version": "sts2-training-schedule-state-v1",
                    "environment_steps_offset": 1,
                }
            }
        )


def test_actor_supervisor_metadata_has_strict_legacy_compatibility() -> None:
    # Absence is the sole legacy migration: old exact-resume checkpoints start
    # with a clean supervisor rather than being rejected.
    assert checkpointing_module.actor_supervisor_state_from_metadata({}) == (ActorSupervisorState())

    valid = ActorSupervisorState(
        episode_attempts=3,
        consecutive_incidents=1,
        incident_fingerprints=(("test:fingerprint", 2),),
        recent_incident_attempts=(1, 3),
    )
    assert (
        checkpointing_module.actor_supervisor_state_from_metadata({"actor_supervisor_state": valid.to_mapping()})
        == valid
    )

    malformed = valid.to_mapping()
    malformed["unexpected"] = True
    with pytest.raises(ValueError, match="keys mismatch"):
        checkpointing_module.actor_supervisor_state_from_metadata({"actor_supervisor_state": malformed})
    with pytest.raises(TypeError, match="must be an object"):
        checkpointing_module.actor_supervisor_state_from_metadata({"actor_supervisor_state": None})


def test_capacity_change_uses_explicit_model_parameter_initialization_lineage(
    tmp_path: Path,
) -> None:
    source_config = _config()
    source = build_training_resources(source_config, backend=FakeCombatBackend())
    try:
        source.rollout_queue.put(source.collector.collect_episode(record=True).unrolls[0])
        source.optimizer.zero_grad(set_to_none=True)
        objective = sum(parameter.square().mean() for parameter in source.model.parameters())
        objective.backward()
        source.optimizer.step()
        source_state = {key: value.detach().clone() for key, value in source.model.state_dict().items()}
        source_training_state = TrainingState(
            environment_steps=40_737,
            learner_updates=638,
            episodes=14,
            evaluation_episodes=1,
            policy_version=638,
            actor_policy_version=638,
            consumed_unrolls=2_552,
        )
        checkpoint = save_training_checkpoint(
            tmp_path / "policy-638",
            config=source_config,
            resources=source,
            state=source_training_state,
            run_id="source-policy-638",
            checkpoint_load_mode="fresh",
        )
    finally:
        source.close()

    target_config = replace(
        source_config,
        model=replace(source_config.model, max_candidates=256),
    )
    with pytest.raises(ValueError, match=r"identical immutable.*lineage"):
        preflight_training_checkpoint(
            checkpoint,
            config=target_config,
            resolved_device="cpu",
            resolved_collector_device="cpu",
        )
    validated = preflight_model_initialization(
        checkpoint,
        config=target_config,
    )
    assert validated.metadata["training_state"]["policy_version"] == 638

    target = build_training_resources(target_config, backend=FakeCombatBackend())
    try:
        initial_collector_state = target.collector.state_dict()
        parent = initialize_model_from_checkpoint(
            checkpoint,
            config=target_config,
            resources=target,
        )
        assert parent == checkpoint.resolve()
        assert len(target.optimizer.state) == 0
        assert len(target.rollout_queue) == 0
        assert target.collector.state_dict() == initial_collector_state
        for key, expected in source_state.items():
            assert torch.equal(target.model.state_dict()[key], expected), key
            assert torch.equal(target.collector_model.state_dict()[key], expected), key

        migrated = save_training_checkpoint(
            tmp_path / "new-lineage-step-zero",
            config=target_config,
            resources=target,
            state=TrainingState(),
            parent_checkpoint=checkpoint,
            run_id="candidate256-lineage",
            checkpoint_load_mode="model_initialization",
            parent_relation="model_parameter_initialization",
        )
        metadata = json.loads((migrated / "metadata.json").read_text(encoding="utf-8"))
        assert metadata["training_state"] == asdict(TrainingState())
        assert metadata["provenance"]["checkpoint_load_mode"] == "model_initialization"
        parent_metadata = metadata["provenance"]["parent_checkpoint"]
        assert parent_metadata["relation"] == "model_parameter_initialization"
        assert parent_metadata["training_state"]["policy_version"] == 638
    finally:
        target.close()


@pytest.mark.parametrize("mismatch", ("contract", "reward", "dependency"))
def test_model_initialization_migrates_only_parameters_across_runtime_identities(
    tmp_path: Path,
    mismatch: str,
) -> None:
    config = _config()
    source = build_training_resources(config, backend=FakeCombatBackend())
    try:
        source.optimizer.zero_grad(set_to_none=True)
        objective = sum(parameter.square().mean() for parameter in source.model.parameters())
        objective.backward()
        source.optimizer.step()
        source_state = {key: value.detach().clone() for key, value in source.model.state_dict().items()}
        checkpoint = save_training_checkpoint(
            tmp_path / f"archived-{mismatch}",
            config=config,
            resources=source,
            state=TrainingState(environment_steps=21_023, policy_version=268),
            checkpoint_load_mode="fresh",
        )
    finally:
        source.close()

    archived = _rewrite_checkpoint_runtime_identity(checkpoint, mismatch=mismatch)
    with pytest.raises(CheckpointIntegrityError, match=f"{mismatch}.*identity"):
        preflight_training_checkpoint(
            checkpoint,
            config=config,
            resolved_device="cpu",
            resolved_collector_device="cpu",
        )
    validated = preflight_model_initialization(checkpoint, config=config)
    assert validated.root == checkpoint.resolve()

    target = build_training_resources(config, backend=FakeCombatBackend())
    try:
        initial_collector_state = target.collector.state_dict()
        parent = initialize_model_from_checkpoint(
            checkpoint,
            config=config,
            resources=target,
        )
        assert parent == checkpoint.resolve()
        assert len(target.optimizer.state) == 0
        assert len(target.rollout_queue) == 0
        assert target.collector.state_dict() == initial_collector_state
        for key, expected in source_state.items():
            assert torch.equal(target.model.state_dict()[key], expected), key
            assert torch.equal(target.collector_model.state_dict()[key], expected), key

        migrated = save_training_checkpoint(
            tmp_path / f"current-{mismatch}",
            config=config,
            resources=target,
            state=TrainingState(),
            parent_checkpoint=checkpoint,
            checkpoint_load_mode="model_initialization",
            parent_relation="model_parameter_initialization",
        )
        migrated_metadata = json.loads((migrated / "metadata.json").read_text(encoding="utf-8"))
        assert migrated_metadata["training_state"] == asdict(TrainingState())
        provenance = migrated_metadata["provenance"]
        assert provenance["checkpoint_load_mode"] == "model_initialization"
        parent_metadata = provenance["parent_checkpoint"]
        assert parent_metadata["relation"] == "model_parameter_initialization"
        source_identity = parent_metadata["source_runtime_identity"]
        assert source_identity["contract"] == archived["contract"]
        assert (
            source_identity["reward_spec_fingerprint_sha256"]
            == (archived["provenance"]["reward_spec"]["fingerprint_sha256"])
        )
        assert source_identity["dependency_locks"] == (archived["provenance"]["dependency_locks"])
    finally:
        target.close()


def test_prevalidated_exact_resume_skips_only_the_duplicate_directory_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()
    source = build_training_resources(config, backend=FakeCombatBackend())
    try:
        source.rollout_queue.put(source.collector.collect_episode(record=True).unrolls[0])
        checkpoint = save_training_checkpoint(
            tmp_path / "prevalidated-source",
            config=config,
            resources=source,
            state=TrainingState(
                environment_steps=2,
                episodes=1,
                maximum_observed_candidates=2,
            ),
            run_id="prevalidated-source-run",
            checkpoint_load_mode="fresh",
        )
    finally:
        source.close()

    validated = preflight_training_checkpoint(
        checkpoint,
        config=config,
        resolved_device="cpu",
        resolved_collector_device="cpu",
    )

    from sts2_rl.checkpoints import resume as resume_module

    hash_passes: list[Path] = []
    real_verify = resume_module.verify_checkpoint_directory

    def counting_verify(root: Path, **kwargs: Any) -> Any:
        hash_passes.append(root)
        return real_verify(root, **kwargs)

    monkeypatch.setattr(resume_module, "verify_checkpoint_directory", counting_verify)

    target = build_training_resources(config, backend=FakeCombatBackend())
    try:
        # A same-process handle must not repeat the whole-directory hash pass.
        state = load_training_checkpoint(
            checkpoint,
            config=config,
            resources=target,
            prevalidated=validated,
        )
        assert state.environment_steps == 2
        assert hash_passes == []

        # The handle only stands in for exactly its own validated path.
        with pytest.raises(ValueError, match="does not match the requested checkpoint path"):
            load_training_checkpoint(
                tmp_path / "some-other-checkpoint",
                config=config,
                resources=target,
                prevalidated=validated,
            )
    finally:
        target.close()

    # An independent call without the handle still performs the complete
    # byte validation of the atomic directory.
    independent = build_training_resources(config, backend=FakeCombatBackend())
    try:
        load_training_checkpoint(checkpoint, config=config, resources=independent)
        assert hash_passes == [checkpoint.resolve()]
    finally:
        independent.close()


def test_prevalidated_handle_cannot_weaken_exact_resume_identity(
    tmp_path: Path,
) -> None:
    config = _config()
    source = build_training_resources(config, backend=FakeCombatBackend())
    try:
        checkpoint = save_training_checkpoint(
            tmp_path / "archived-runtime-identity",
            config=config,
            resources=source,
            state=TrainingState(environment_steps=2, policy_version=1, actor_policy_version=1),
            checkpoint_load_mode="fresh",
        )
    finally:
        source.close()

    _rewrite_checkpoint_runtime_identity(checkpoint, mismatch="contract")
    # The weaker model-initialization validation accepts the archived identity.
    weak = preflight_model_initialization(checkpoint, config=config)
    # Reusing that weaker handle for exact resume must still fail closed on
    # the strict current-runtime identity, which is re-checked in full.
    with pytest.raises(CheckpointIntegrityError, match="contract identity"):
        preflight_training_checkpoint(
            checkpoint,
            config=config,
            resolved_device="cpu",
            resolved_collector_device="cpu",
            prevalidated=weak,
        )


def test_previous_selection_abi_checkpoint_is_rejected_before_tensor_load(
    tmp_path: Path,
) -> None:
    config = _config()
    old_encoding = dict(grounding_encoding_identity())
    old_encoding.update(
        {
            "version": "grounded-selection-semantics-encoding-v5",
            "min_token_feature_dim": 128,
            "fingerprint_sha256": "0" * 64,
        }
    )
    validated = ValidatedResumeCheckpoint(
        root=tmp_path,
        manifest={},
        metadata={
            "format": "sts2-recurrent-vtrace-checkpoint-v3",
            "model_config": asdict(config.model.to_model_config()),
            "encoding_contract": old_encoding,
            "model_state_spec": {},
        },
    )

    with pytest.raises(ValueError, match="encoding contract does not match"):
        checkpointing_module._validate_metadata(
            validated,
            config=config,
            resolved_device=None,
            resolved_collector_device=None,
            model_only=True,
        )
