from __future__ import annotations

import hashlib
import json
import random
import time
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

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
    CurriculumConfig,
    DiagnosticsConfig,
    EnvironmentConfig,
    ModelConfig,
    OptimizationConfig,
    RolloutConfig,
    RuntimeConfig,
    TrainingConfig,
    build_training_resources,
    evaluate_policy,
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
from sts2_rl.training.collector import CollectionProtocolError, _CombatNetProgressTracker
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
        reward["fingerprint_sha256"] = hashlib.sha256(
            reward["fingerprint"].encode("utf-8")
        ).hexdigest()
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


def test_baseline_inspection_exposes_active_shapes_separately_from_capacities() -> None:
    config = _config()
    report = inspect_baseline(config)
    assert report["active_shape_batching"] is True
    assert report["encoding_capacities"]["candidates"] == 6
    assert report["candidate_shape"][1] < 6
    assert report["episodic_learning"]["macro_sample_fraction"] == 0.0


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
            revival_relic_id="RELIC.LIZARD_TAIL",
            revival_budget=-1,
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

        records = [
            json.loads(line)
            for line in journal_path.read_text(encoding="utf-8").splitlines()
        ]
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
        assert terminal_stalls[0]["deadlock"]["legal_action_kinds"] == {
            "end_turn": 1
        }
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


def test_noncombat_stall_evidence_precedes_simultaneous_semantic_evidence(
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
        assert episode.metrics.terminal_reason == "noncombat_progress_stall"
        records = [json.loads(line) for line in journal_path.read_text(encoding="utf-8").splitlines()]
        terminal_summary = next(
            item for item in records if item.get("record_kind") == "summary" and item.get("outcome") == "deadlock"
        )
        assert terminal_summary["deadlock"]["kind"] == ("noncombat_no_durable_progress")
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
            progress=lambda stage, payload: progress.append((stage, payload)),
        )
        assert metrics.environment_steps == 2
        assert metrics.to_mapping()["batch_environment_steps"] == 2
        assert "environment_steps" not in metrics.to_mapping()
        assert metrics.unrolls == 1
        assert torch.isfinite(torch.tensor(metrics.loss))
        assert metrics.importance_ratio_mean > 0.0
        assert progress[0][0] == "validation_complete"
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
        metrics = resources.learner.update(first, current_policy_version=0)
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
    finally:
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
    metrics_path = next(
        (tmp_path / "runs" / "recoverable-incident").glob("run-*/metrics.jsonl")
    )
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

    checkpoint = next(
        (tmp_path / "checkpoints" / "supervisor-source").glob("run-*/final-*")
    )
    metadata = json.loads((checkpoint / "metadata.json").read_text(encoding="utf-8"))
    restored_supervisor = checkpointing_module.actor_supervisor_state_from_metadata(
        metadata
    )
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

    metrics_path = next(
        (tmp_path / "runs" / "supervisor-resume").glob("run-*/metrics.jsonl")
    )
    events = [
        json.loads(line)
        for line in metrics_path.read_text(encoding="utf-8").splitlines()
    ]
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

    state = run_training(
        config,
        backend=FakeCombatBackend(),
        resume_from=source_checkpoint,
    )

    assert state.environment_steps == 4
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
    metrics_path = next(
        (tmp_path / "runs" / "evaluation-resume").glob("run-*/metrics.jsonl")
    )
    events = [
        json.loads(line)
        for line in metrics_path.read_text(encoding="utf-8").splitlines()
    ]
    assert not [event for event in events if event["event"] == "evaluation"]


def test_evaluation_uses_odd_heldout_seeds_and_records_no_unrolls(
    tmp_path: Path,
) -> None:
    backend = FakeCombatBackend()
    resources = build_training_resources(_config(), backend=backend)
    try:
        episodes, summary = evaluate_policy(
            resources,
            episodes=2,
            base_seed=6,
            journal_path=tmp_path / "trajectory.jsonl",
        )
        assert len(episodes) == 2
        assert summary["combat_win_rate"] == 1.0
        assert summary["act1_clear_count"] == 0
        assert summary["act3_reach_count"] == 0
        assert summary["act3_reach_rate"] == 0.0
        assert summary["maximum_observed_candidates"] == 2
        assert all(item.maximum_observed_candidates == 2 for item in episodes)
        assert all(int(seed) % 2 == 1 for seed in backend.reset_seeds)
        assert (tmp_path / "trajectory.jsonl").read_text(encoding="utf-8")
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
    learner_before = {
        key: value.detach().clone()
        for key, value in resources.model.state_dict().items()
    }
    collector_before = {
        key: value.detach().clone()
        for key, value in resources.collector_model.state_dict().items()
    }
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
        assert (
            sensitivity["recurrent_state_contract"]
            == "fresh_zero_state_per_pair"
        )

        assert len(resources.rollout_queue) == 0
        assert resources.collector.state_dict() == collector_state_before
        assert resources.model.training is learner_mode_before
        assert resources.collector_model.training is collector_mode_before
        assert set(resources.model.state_dict()) == set(learner_before)
        assert set(resources.collector_model.state_dict()) == set(collector_before)
        assert all(
            torch.equal(resources.model.state_dict()[key], value)
            for key, value in learner_before.items()
        )
        assert all(
            torch.equal(resources.collector_model.state_dict()[key], value)
            for key, value in collector_before.items()
        )
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
            json.loads(line)
            for line in (tmp_path / "retry-trajectory.jsonl").read_text(encoding="utf-8").splitlines()
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
            if item.get("event") in {"decision", "decision_snapshot"}
            and item.get("reset_seed") == 13
        }
        assert any("attempt-1:" in item for item in first_seed_episode_ids)
        assert any("attempt-2:" in item for item in first_seed_episode_ids)
        assert not any(item == "combat-episode-1" for item in first_seed_episode_ids)
        first_abort_index = next(
            index
            for index, item in enumerate(journal_events)
            if item.get("event") == "evaluation_attempt_aborted"
            and item.get("evaluation_seed") == 13
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
    cuda_rng_before = (
        [state.clone() for state in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_available()
        else None
    )
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
    assert (
        audit["evaluation"]["macro_surface_telemetry"]["training_samples_emitted"]
        == 0
    )
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
    cuda_rng_before = (
        [state.clone() for state in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_available()
        else None
    )
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


def test_actor_supervisor_metadata_has_strict_legacy_compatibility() -> None:
    # Absence is the sole legacy migration: old exact-resume checkpoints start
    # with a clean supervisor rather than being rejected.
    assert checkpointing_module.actor_supervisor_state_from_metadata({}) == (
        ActorSupervisorState()
    )

    valid = ActorSupervisorState(
        episode_attempts=3,
        consecutive_incidents=1,
        incident_fingerprints=(("test:fingerprint", 2),),
        recent_incident_attempts=(1, 3),
    )
    assert checkpointing_module.actor_supervisor_state_from_metadata(
        {"actor_supervisor_state": valid.to_mapping()}
    ) == valid

    malformed = valid.to_mapping()
    malformed["unexpected"] = True
    with pytest.raises(ValueError, match="keys mismatch"):
        checkpointing_module.actor_supervisor_state_from_metadata(
            {"actor_supervisor_state": malformed}
        )
    with pytest.raises(TypeError, match="must be an object"):
        checkpointing_module.actor_supervisor_state_from_metadata(
            {"actor_supervisor_state": None}
        )


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
    with pytest.raises(ValueError, match="identical immutable.*lineage"):
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
        source_state = {
            key: value.detach().clone() for key, value in source.model.state_dict().items()
        }
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
        migrated_metadata = json.loads(
            (migrated / "metadata.json").read_text(encoding="utf-8")
        )
        assert migrated_metadata["training_state"] == asdict(TrainingState())
        provenance = migrated_metadata["provenance"]
        assert provenance["checkpoint_load_mode"] == "model_initialization"
        parent_metadata = provenance["parent_checkpoint"]
        assert parent_metadata["relation"] == "model_parameter_initialization"
        source_identity = parent_metadata["source_runtime_identity"]
        assert source_identity["contract"] == archived["contract"]
        assert source_identity["reward_spec_fingerprint_sha256"] == (
            archived["provenance"]["reward_spec"]["fingerprint_sha256"]
        )
        assert source_identity["dependency_locks"] == (
            archived["provenance"]["dependency_locks"]
        )
    finally:
        target.close()


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
