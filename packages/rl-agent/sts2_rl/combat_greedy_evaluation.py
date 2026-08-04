"""Paired combat-reset evaluation of frozen RL and one-step greedy policies.

This is a diagnostic, not a training policy.  It deliberately lives outside
the collector/learner composition root and never contributes trajectories to
rollout, episodic, transaction, or failure-credit replay.

The full-run simulator save API intentionally restores mid-combat snapshots to
the *start* of the encounter, so it is not a valid branch primitive.  This
diagnostic instead uses ``combat_reset`` with a fixed encounter/seed/build and
replays the already committed action prefix before every candidate probe.
Every candidate therefore starts at the same factual root without copying any
game rule into Python.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
import time
from collections import Counter, deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch

from sts2_env.headless_sim_bridge_client import HeadlessSimBridgeClient
from sts2_rl.models import RecurrentCandidateModel

from .checkpoints import validate_resume_checkpoint
from .encoding import GroundedObservationEncoder
from .training.checkpoint_evaluation import checkpoint_training_config
from .training.checkpointing import preflight_model_initialization
from .training.seeding import held_out_evaluation_seeds

_JOURNAL_SCHEMA = "sts2-combat-greedy-paired-journal-v1"
_SUMMARY_SCHEMA = "sts2-combat-greedy-paired-summary-v1"
_COMBAT_DOMAINS = frozenset({"combat"})


class CombatEvaluationError(RuntimeError):
    """The paired combat contract could not be preserved."""


class _RawHeadlessClient(Protocol):
    @property
    def session_id(self) -> str: ...

    def _rpc(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        *,
        timeout_s: float | None = None,
    ) -> dict[str, Any]: ...

    def reset(self, **kwargs: Any) -> dict[str, Any]: ...

    def combat_reset(self, **kwargs: Any) -> dict[str, Any]: ...

    def step(
        self,
        episode_id: str,
        action_index: int | None = None,
        action_id: str | None = None,
        timeout_ms: int = 20_000,
    ) -> dict[str, Any]: ...

    def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class CombatFeatures:
    player_hp: float
    player_max_hp: float
    player_block: float
    enemy_hp: float
    enemy_max_hp: float
    alive_enemies: int
    incoming_damage: float
    selected_count: int
    remaining_select: int


@dataclass(frozen=True, slots=True)
class StateEvaluation:
    score: float
    components: dict[str, float]


@dataclass(frozen=True, slots=True)
class CandidateProbe:
    position: int
    action: dict[str, Any]
    evaluation: StateEvaluation
    resulting_fingerprint: str
    resulting_state_type: str
    terminal_reason: str | None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class GreedyChoice:
    position: int
    probes: tuple[CandidateProbe, ...]


@dataclass(frozen=True, slots=True)
class PolicyStep:
    position: int
    recurrent_state: torch.Tensor | None
    diagnostics: dict[str, Any] = field(default_factory=dict)
    episode_id: str | None = None


@dataclass(frozen=True, slots=True)
class CombatOutcome:
    agent: str
    won: bool
    lost: bool
    truncated: bool
    deadlocked: bool
    no_progress_stall: bool
    terminal_reason: str | None
    start_hp: float
    end_hp: float
    hp_loss: float
    exact_hp_loss: float
    revivals: int
    turns: int
    actions: int
    elapsed_s: float
    final_state_type: str
    action_trace: tuple[dict[str, Any], ...]
    recurrent_state: torch.Tensor | None = field(repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class PairedCombatResult:
    pair_index: int
    evaluation_seed: int
    act: int
    floor: int
    encounter_id: str
    room_type: str
    initial_fingerprint: str
    revival_budget: int
    rl: CombatOutcome
    greedy: CombatOutcome
    comparison_rl: tuple[tuple[str, CombatOutcome], ...] = ()


@dataclass(frozen=True, slots=True)
class CombatResetSpec:
    character: str
    encounter_id: str
    seed: int
    revival_budget: int = 0

    def kwargs(self) -> dict[str, Any]:
        return {
            "character": self.character,
            "encounter_id": self.encounter_id,
            "seed": self.seed,
            "training_revival_budget": self.revival_budget,
        }


def _prepare_combat_reset(
    client: _RawHeadlessClient,
    spec: CombatResetSpec,
) -> dict[str, Any]:
    """Clear a prior combat-victory boundary, then create the fixed combat."""

    client.reset(
        character=spec.character,
        seed=spec.seed,
        force_fresh=True,
        training_revival_budget=spec.revival_budget,
    )
    return client.combat_reset(**spec.kwargs())


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: object) -> Sequence[Any]:
    return value if isinstance(value, Sequence) and not isinstance(value, str | bytes) else ()


def _number(value: object, default: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return float(default)
    converted = float(value)
    return converted if math.isfinite(converted) else float(default)


def _integer(value: object, default: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return int(default)
    converted = float(value)
    return int(converted) if math.isfinite(converted) else int(default)


def _power_amount(entity: Mapping[str, Any], fragment: str) -> float:
    needle = fragment.casefold()
    amount = 0.0
    for power_value in _sequence(entity.get("powers")):
        power = _mapping(power_value)
        power_id = str(power.get("id") or power.get("power_id") or "").casefold()
        if needle in power_id:
            amount += _number(power.get("amount", power.get("stacks", 0.0)))
    return amount


def combat_features(observation: Mapping[str, Any]) -> CombatFeatures:
    """Project only generic factual combat scalars used by the heuristic."""

    player = _mapping(observation.get("player"))
    combat = _mapping(observation.get("combat"))
    enemies = [_mapping(item) for item in _sequence(combat.get("enemies"))]
    player_hp = _number(player.get("hp", player.get("current_hp", 0.0)))
    player_max_hp = max(_number(player.get("max_hp", player_hp)), 1.0)
    player_block = _number(player.get("block", combat.get("player_block", 0.0)))
    enemy_hp = 0.0
    enemy_max_hp = 0.0
    alive_enemies = 0
    incoming_damage = 0.0
    for enemy in enemies:
        hp = max(_number(enemy.get("hp", enemy.get("current_hp", 0.0))), 0.0)
        max_hp = max(_number(enemy.get("max_hp", hp)), 1.0)
        enemy_hp += hp
        enemy_max_hp += max_hp
        if hp > 0.0 and not bool(enemy.get("dead", False)):
            alive_enemies += 1
        for intent_value in _sequence(enemy.get("intents")):
            intent = _mapping(intent_value)
            total = intent.get("total_damage")
            if total is None:
                total = _number(intent.get("damage")) * max(
                    _number(intent.get("repeats"), 1.0),
                    1.0,
                )
            incoming_damage += max(_number(total), 0.0)
    selection = _mapping(observation.get("card_selection"))
    selected_count = _integer(
        selection.get("selected_count"),
        len(_sequence(selection.get("selected_cards"))),
    )
    remaining_select = _integer(
        selection.get("remaining_select", selection.get("remaining_picks", 0)),
        0,
    )
    return CombatFeatures(
        player_hp=player_hp,
        player_max_hp=player_max_hp,
        player_block=max(player_block, 0.0),
        enemy_hp=enemy_hp,
        enemy_max_hp=max(enemy_max_hp, 1.0),
        alive_enemies=alive_enemies,
        incoming_damage=max(incoming_damage, 0.0),
        selected_count=selected_count,
        remaining_select=max(remaining_select, 0),
    )


def is_combat_state(observation: Mapping[str, Any]) -> bool:
    domain = str(observation.get("decision_domain") or "").strip().lower()
    combat = _mapping(observation.get("combat"))
    return domain in _COMBAT_DOMAINS or bool(combat.get("in_progress", False))


def _terminal_reason(result: Mapping[str, Any]) -> str | None:
    value = result.get("terminal_reason")
    if value is None:
        value = _mapping(result.get("info")).get("terminal_reason")
    return str(value) if value is not None and str(value).strip() else None


def _is_terminal(result: Mapping[str, Any]) -> bool:
    return bool(result.get("done", result.get("terminated", False)))


def _is_truncated(result: Mapping[str, Any]) -> bool:
    return bool(result.get("truncated", False))


def _result_observation(result: Mapping[str, Any]) -> Mapping[str, Any]:
    value = result.get("obs", result.get("observation"))
    if not isinstance(value, Mapping):
        raise CombatEvaluationError("environment result has no observation object")
    return value


def _result_actions(result: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    values = result.get("legal_actions")
    if not isinstance(values, list | tuple):
        raise CombatEvaluationError("environment result has no legal action sequence")
    actions = tuple(dict(item) for item in values if isinstance(item, Mapping))
    if len(actions) != len(values):
        raise CombatEvaluationError("environment result contains a non-object legal action")
    return actions


def _combat_outcome_kind(result: Mapping[str, Any]) -> tuple[bool, bool]:
    observation = _result_observation(result)
    reason = (_terminal_reason(result) or "").casefold()
    player_hp = combat_features(observation).player_hp
    if not is_combat_state(observation) and not _is_terminal(result):
        return True, False
    if any(fragment in reason for fragment in ("victory", "won", "run_win", "combat_win")):
        return True, False
    if _is_terminal(result) and any(
        fragment in reason for fragment in ("defeat", "death", "dead", "loss", "failed")
    ):
        return False, True
    if _is_terminal(result) and player_hp <= 0.0:
        return False, True
    return False, False


def evaluate_one_step(
    before_result: Mapping[str, Any],
    after_result: Mapping[str, Any],
) -> StateEvaluation:
    """Evaluate one simulator transition with transparent generic terms.

    Coefficients intentionally make survival more important than immediate
    damage while still preferring kills and lower exposed next-turn damage.
    They are diagnostic constants and never enter the training reward.
    """

    before_observation = _result_observation(before_result)
    after_observation = _result_observation(after_result)
    before = combat_features(before_observation)
    after = combat_features(after_observation)
    won, lost = _combat_outcome_kind(after_result)

    hp_fraction = after.player_hp / after.player_max_hp
    enemy_fraction = after.enemy_hp / after.enemy_max_hp
    dead_enemies = max(before.alive_enemies - after.alive_enemies, 0)
    uncovered_incoming = max(after.incoming_damage - after.player_block, 0.0)
    uncovered_fraction = uncovered_incoming / after.player_max_hp
    block_fraction = min(after.player_block, after.incoming_damage) / after.player_max_hp
    selection_progress = float(before.remaining_select - after.remaining_select)
    selection_progress += 0.25 * float(after.selected_count - before.selected_count)
    factual_progress = (
        after.enemy_hp < before.enemy_hp
        or after.player_hp != before.player_hp
        or after.alive_enemies != before.alive_enemies
        or not is_combat_state(after_observation)
        or selection_progress != 0.0
    )

    components = {
        "terminal_win": 1_000_000.0 if won else 0.0,
        "terminal_loss": -1_000_000.0 if lost else 0.0,
        "hp_survival": 200.0 * hp_fraction,
        "enemy_remaining": -100.0 * enemy_fraction,
        "enemy_kills": 25.0 * dead_enemies,
        "covered_intent": 8.0 * block_fraction,
        "uncovered_intent": -20.0 * uncovered_fraction,
        "selection_progress": 2.0 * selection_progress,
        "no_factual_progress": -0.05 if not factual_progress else 0.0,
    }
    return StateEvaluation(score=sum(components.values()), components=components)


def _semantic_action(action: Mapping[str, Any]) -> dict[str, Any]:
    card = _mapping(action.get("card"))
    target = _mapping(action.get("target"))
    potion = _mapping(action.get("potion"))
    selection = _mapping(action.get("selection"))
    return {
        "kind": str(action.get("kind") or action.get("model_action_kind") or "unknown"),
        "variant": action.get("model_action_variant"),
        "card_id": card.get("id"),
        "card_instance_id": card.get("instance_id"),
        "target_model_id": target.get("model_id", target.get("id")),
        "target_instance_id": target.get("instance_id"),
        "potion_id": potion.get("id"),
        "selection_operation": selection.get("operation_type", action.get("selection_operation")),
    }


def _without_runtime_identity(value: object) -> object:
    """Remove per-reset object identities while preserving factual ordering."""

    if isinstance(value, Mapping):
        return {
            str(key): _without_runtime_identity(item)
            for key, item in value.items()
            if str(key).casefold()
            not in {
                "instance_id",
                "runtime_id",
                "transition_token",
                "request_id",
            }
        }
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return [_without_runtime_identity(item) for item in value]
    return value


def _combat_run_projection(observation: Mapping[str, Any]) -> object:
    """Project run facts while excluding the transport full-run reset seed."""

    run = dict(_mapping(observation.get("run")))
    run.pop("seed", None)
    return run


def combat_state_fingerprint(result: Mapping[str, Any]) -> str:
    """Hash the factual paired combat state while excluding runtime journals.

    ``_training`` contains cumulative diagnostic counters whose serialization
    can be refreshed by a save/load operation even when the authoritative run,
    combat, player, selection, and legal action surface are identical.  Those
    counters are audited separately and must not invalidate a factual branch.
    """

    observation = _result_observation(result)
    actions = _result_actions(result)
    payload = _without_runtime_identity({
        "state_type": observation.get("state_type"),
        "decision_domain": observation.get("decision_domain"),
        "player": observation.get("player"),
        "combat": observation.get("combat"),
        "run": _combat_run_projection(observation),
        "card_selection": observation.get("card_selection"),
        "actions": [_semantic_action(action) for action in actions],
    })
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def choose_greedy_candidate(
    before_result: Mapping[str, Any],
    probe: Callable[[int], Mapping[str, Any]],
) -> GreedyChoice:
    """Probe every enabled legal candidate and return the stable best index."""

    actions = _result_actions(before_result)
    if not is_combat_state(_result_observation(before_result)):
        raise CombatEvaluationError("one-step greedy refuses a non-combat decision")
    candidates: list[CandidateProbe] = []
    for position, action in enumerate(actions):
        if action.get("is_enabled", True) is not True:
            continue
        try:
            after = probe(position)
            evaluation = evaluate_one_step(before_result, after)
            candidates.append(
                CandidateProbe(
                    position=position,
                    action=_semantic_action(action),
                    evaluation=evaluation,
                    resulting_fingerprint=combat_state_fingerprint(after),
                    resulting_state_type=str(_result_observation(after).get("state_type") or ""),
                    terminal_reason=_terminal_reason(after),
                )
            )
        except Exception as exc:
            candidates.append(
                CandidateProbe(
                    position=position,
                    action=_semantic_action(action),
                    evaluation=StateEvaluation(score=-math.inf, components={"probe_error": -math.inf}),
                    resulting_fingerprint="",
                    resulting_state_type="",
                    terminal_reason=None,
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
    viable = [item for item in candidates if math.isfinite(item.evaluation.score)]
    if not viable:
        failures = "; ".join(
            f"position={item.position}: {item.error or 'non-finite score'}"
            for item in candidates[:8]
        )
        raise CombatEvaluationError(
            "every one-step legal-candidate probe failed: " + failures
        )
    # max() is stable for ties because viable retains authoritative action order.
    selected = max(viable, key=lambda item: item.evaluation.score)
    return GreedyChoice(position=selected.position, probes=tuple(candidates))


class DeterministicCombatReplay:
    """Restore a combat root by deterministic reset plus authoritative replay."""

    def __init__(self, client: _RawHeadlessClient, spec: CombatResetSpec) -> None:
        self.client = client
        self.spec = spec
        self.committed_positions: list[int] = []
        self.initial_fingerprint: str | None = None

    def reset(self) -> dict[str, Any]:
        result = _prepare_combat_reset(self.client, self.spec)
        fingerprint = combat_state_fingerprint(result)
        if self.initial_fingerprint is None:
            self.initial_fingerprint = fingerprint
        elif fingerprint != self.initial_fingerprint:
            raise CombatEvaluationError(
                "combat_reset is non-deterministic for the fixed scenario: "
                f"expected={self.initial_fingerprint} actual={fingerprint}"
            )
        return result

    def restore_prefix(self, *, expected_fingerprint: str) -> dict[str, Any]:
        result = self.reset()
        for position in self.committed_positions:
            actions = _result_actions(result)
            if not 0 <= position < len(actions):
                raise CombatEvaluationError(
                    "recorded action position is no longer legal during deterministic replay"
                )
            result = self.client.step(
                str(result.get("episode_id") or ""),
                action_index=position,
            )
        actual = combat_state_fingerprint(result)
        if actual != expected_fingerprint:
            raise CombatEvaluationError(
                "deterministic action-prefix replay restored a different root: "
                f"expected={expected_fingerprint} actual={actual}"
            )
        return result

    def record(self, position: int) -> None:
        self.committed_positions.append(int(position))


class FrozenRlCombatPolicy:
    """Frozen deterministic recurrent policy loaded from ``network.pt`` only."""

    def __init__(self, checkpoint: str | Path, *, device: str = "cpu") -> None:
        self.checkpoint = Path(checkpoint).expanduser().resolve()
        self.config = checkpoint_training_config(self.checkpoint)
        validated = preflight_model_initialization(self.checkpoint, config=self.config)
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise CombatEvaluationError("CUDA/ROCm requested for frozen combat policy but unavailable")
        self.model = RecurrentCandidateModel(
            self.config.model.to_model_config(),
            enable_transaction_heads=self.config.transaction_learning.enabled,
            enable_liveness_head=self.config.failure_credit.learning_enabled,
        ).to(self.device)
        state = torch.load(
            validated.root / "network.pt",
            map_location=self.device,
            weights_only=True,
        )
        if not isinstance(state, dict):
            raise CombatEvaluationError("checkpoint network.pt is not a tensor mapping")
        self.model.load_state_dict(state, strict=True)
        self.model.eval()
        self.encoder = GroundedObservationEncoder(self.config.model.to_encoding_config())
        checkpoint_id = validated.manifest.get("checkpoint_id")
        self.checkpoint_id = str(checkpoint_id or "")

    def initial_state(self) -> torch.Tensor:
        return self.model.initial_state(1, device=self.device)

    def select(
        self,
        result: Mapping[str, Any],
        recurrent_state: torch.Tensor,
    ) -> PolicyStep:
        observation = _result_observation(result)
        actions = _result_actions(result)
        encoded = self.encoder.encode(observation, actions, device=self.device)
        with torch.inference_mode():
            output = self.model(encoded.batch, recurrent_state, validate=False)
            selected = int(output.greedy_action_indices()[0].item())
            probabilities = output.policy_probabilities()[0].float().cpu()
            value = float(output.value[0].item())
        if selected < 0 or selected >= len(encoded.actions):
            raise CombatEvaluationError(f"frozen RL policy selected invalid candidate {selected}")
        reference = encoded.action(selected)
        if not 0 <= reference.position < len(actions):
            raise CombatEvaluationError("encoder dispatch position escaped legal action surface")
        top_count = min(5, probabilities.numel())
        top_values, top_indices = torch.topk(probabilities, k=top_count)
        return PolicyStep(
            position=reference.position,
            recurrent_state=output.recurrent_state.detach(),
            diagnostics={
                "candidate_index": selected,
                "value": value,
                "selected_probability": float(probabilities[selected].item()),
                "policy_top": [
                    {
                        "candidate_index": int(index),
                        "probability": float(probability),
                    }
                    for probability, index in zip(top_values.tolist(), top_indices.tolist(), strict=True)
                ],
            },
        )


class OneStepGreedyCombatPolicy:
    def __init__(self, replay: DeterministicCombatReplay, *, candidate_topk: int = 8) -> None:
        self.replay = replay
        self.candidate_topk = max(int(candidate_topk), 0)

    def select(self, result: Mapping[str, Any]) -> PolicyStep:
        root_fingerprint = combat_state_fingerprint(result)
        def probe(position: int) -> Mapping[str, Any]:
            restored = self.replay.restore_prefix(expected_fingerprint=root_fingerprint)
            return self.replay.client.step(
                str(restored.get("episode_id") or ""),
                action_index=position,
            )

        choice = choose_greedy_candidate(result, probe)
        restored_root = self.replay.restore_prefix(expected_fingerprint=root_fingerprint)
        ordered = sorted(
            choice.probes,
            key=lambda item: item.evaluation.score,
            reverse=True,
        )
        top = ordered[: self.candidate_topk] if self.candidate_topk else []
        diagnostics = {
            "replayed_prefix_length": len(self.replay.committed_positions),
            "candidate_count": len(choice.probes),
            "probe_failures": sum(item.error is not None for item in choice.probes),
            "selected_score": next(
                item.evaluation.score for item in choice.probes if item.position == choice.position
            ),
            "candidate_top": [
                {
                    "position": item.position,
                    "action": item.action,
                    "score": item.evaluation.score,
                    "components": item.evaluation.components,
                    "resulting_state_type": item.resulting_state_type,
                    "terminal_reason": item.terminal_reason,
                    "error": item.error,
                }
                for item in top
            ],
        }
        return PolicyStep(
            position=choice.position,
            recurrent_state=None,
            diagnostics=diagnostics,
            episode_id=str(restored_root.get("episode_id") or ""),
        )

    def record_committed(self, position: int) -> None:
        self.replay.record(position)


def _action_cycle_key(result: Mapping[str, Any], position: int) -> str:
    action = _result_actions(result)[position]
    payload = {
        "state": combat_state_fingerprint(result),
        "action": _semantic_action(action),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def _combat_progress_marker(result: Mapping[str, Any]) -> tuple[float, int, int]:
    observation = _result_observation(result)
    features = combat_features(observation)
    combat = _mapping(observation.get("combat"))
    return (features.enemy_hp, features.alive_enemies, _integer(combat.get("turn", 0)))


def _run_combat(
    client: _RawHeadlessClient,
    start_result: Mapping[str, Any],
    *,
    agent: str,
    rl_policy: FrozenRlCombatPolicy | None,
    rl_recurrent_state: torch.Tensor | None,
    greedy_policy: OneStepGreedyCombatPolicy | None,
    maximum_steps: int,
    stall_window: int,
    repeat_threshold: int,
) -> tuple[dict[str, Any], CombatOutcome]:
    if not is_combat_state(_result_observation(start_result)):
        raise CombatEvaluationError("paired combat runner received a non-combat root")
    uses_rl = rl_policy is not None
    if uses_rl and rl_recurrent_state is None:
        raise ValueError("RL combat requires a recurrent state")
    if not uses_rl and greedy_policy is None:
        raise ValueError("greedy combat requires one-step policy")
    current = dict(start_result)
    current_recurrent = rl_recurrent_state
    start_features = combat_features(_result_observation(current))
    training_start = _mapping(_result_observation(current).get("_training"))
    start_exact_hp_loss = _number(training_start.get("player_hp_lost"))
    start_revivals = _integer(training_start.get("revivals_used"))
    trace: list[dict[str, Any]] = []
    cycle_counts: Counter[str] = Counter()
    progress_history: deque[tuple[float, int, int]] = deque(maxlen=max(stall_window, 2))
    deadlocked = False
    no_progress_stall = False
    started = time.perf_counter()
    max_turn = 0

    for decision_index in range(maximum_steps):
        observation = _result_observation(current)
        if _is_terminal(current) or _is_truncated(current) or not is_combat_state(observation):
            break
        actions = _result_actions(current)
        if not actions:
            no_progress_stall = True
            break
        if uses_rl:
            assert rl_policy is not None and current_recurrent is not None
            selected = rl_policy.select(current, current_recurrent)
            current_recurrent = selected.recurrent_state
        else:
            assert greedy_policy is not None
            selected = greedy_policy.select(current)
        if not 0 <= selected.position < len(actions):
            raise CombatEvaluationError(f"{agent} selected action outside legal surface")
        cycle_key = _action_cycle_key(current, selected.position)
        cycle_counts[cycle_key] += 1
        if cycle_counts[cycle_key] >= repeat_threshold:
            deadlocked = True
            break
        before_fingerprint = combat_state_fingerprint(current)
        action_projection = _semantic_action(actions[selected.position])
        next_result = client.step(
            selected.episode_id or str(current.get("episode_id") or ""),
            action_index=selected.position,
        )
        if not uses_rl:
            assert greedy_policy is not None
            greedy_policy.record_committed(selected.position)
        after_fingerprint = combat_state_fingerprint(next_result)
        after_observation = _result_observation(next_result)
        turn = _integer(_mapping(after_observation.get("combat")).get("turn", 0))
        max_turn = max(max_turn, turn)
        trace.append(
            {
                "decision": decision_index,
                "before_fingerprint": before_fingerprint,
                "after_fingerprint": after_fingerprint,
                "action_position": selected.position,
                "action": action_projection,
                "diagnostics": selected.diagnostics,
            }
        )
        current = dict(next_result)
        marker = _combat_progress_marker(current)
        progress_history.append(marker)
        if len(progress_history) == progress_history.maxlen:
            first_enemy_hp, first_alive, first_turn = progress_history[0]
            last_enemy_hp, last_alive, last_turn = progress_history[-1]
            # Merely advancing turns is not net combat progress.  The window
            # must reduce total enemy HP or alive-enemy count.
            _ = (first_turn, last_turn)
            if last_enemy_hp >= first_enemy_hp and last_alive >= first_alive:
                no_progress_stall = True
                break
    else:
        no_progress_stall = True

    final_observation = _result_observation(current)
    final_features = combat_features(final_observation)
    final_training = _mapping(final_observation.get("_training"))
    exact_hp_loss = max(
        _number(final_training.get("player_hp_lost")) - start_exact_hp_loss,
        0.0,
    )
    revivals = max(_integer(final_training.get("revivals_used")) - start_revivals, 0)
    won, lost = _combat_outcome_kind(current)
    if deadlocked or no_progress_stall:
        won = False
        lost = True
    outcome = CombatOutcome(
        agent=agent,
        won=won,
        lost=lost,
        truncated=_is_truncated(current),
        deadlocked=deadlocked,
        no_progress_stall=no_progress_stall,
        terminal_reason=_terminal_reason(current),
        start_hp=start_features.player_hp,
        end_hp=final_features.player_hp,
        hp_loss=max(start_features.player_hp - final_features.player_hp, 0.0),
        exact_hp_loss=exact_hp_loss,
        revivals=revivals,
        turns=max_turn,
        actions=len(trace),
        elapsed_s=time.perf_counter() - started,
        final_state_type=str(final_observation.get("state_type") or ""),
        action_trace=tuple(trace),
        recurrent_state=current_recurrent,
    )
    return current, outcome


def _encounter_metadata(observation: Mapping[str, Any]) -> tuple[int, int, str, str]:
    run = _mapping(observation.get("run"))
    combat = _mapping(observation.get("combat"))
    act = _integer(run.get("act", run.get("act_index", 0)))
    floor = _integer(run.get("floor", run.get("total_floor", 0)))
    encounter_id = str(
        combat.get("encounter_id")
        or combat.get("room_id")
        or run.get("encounter_id")
        or "unknown"
    )
    room_type = str(combat.get("room_type") or run.get("room_type") or observation.get("state_type") or "")
    return act, floor, encounter_id, room_type


def _training_counts(result: Mapping[str, Any]) -> tuple[int, float]:
    training = _mapping(_result_observation(result).get("_training"))
    return _integer(training.get("revivals_used")), _number(training.get("player_hp_lost"))


def _outcome_payload(outcome: CombatOutcome) -> dict[str, Any]:
    # ``asdict`` deep-copies values.  Strip the (potentially device-resident)
    # recurrent tensor first rather than asking deepcopy to clone it.
    return asdict(replace(outcome, recurrent_state=None))


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(dict(payload), sort_keys=True, default=str) + "\n")


def _wilson_interval(successes: int, total: int, *, z: float = 1.959963984540054) -> list[float]:
    if total <= 0:
        return [0.0, 1.0]
    probability = successes / total
    denominator = 1.0 + z * z / total
    center = (probability + z * z / (2.0 * total)) / denominator
    radius = (
        z
        * math.sqrt(probability * (1.0 - probability) / total + z * z / (4.0 * total * total))
        / denominator
    )
    return [max(0.0, center - radius), min(1.0, center + radius)]


def _bootstrap_mean_interval(
    values: Sequence[float],
    *,
    seed: int = 0,
    samples: int = 10_000,
) -> list[float]:
    if not values:
        return [0.0, 0.0]
    array = np.asarray(values, dtype=np.float64)
    if len(array) == 1:
        return [float(array[0]), float(array[0])]
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    for start in range(0, samples, 1000):
        count = min(1000, samples - start)
        indices = rng.integers(0, len(array), size=(count, len(array)))
        means[start : start + count] = array[indices].mean(axis=1)
    lower, upper = np.quantile(means, [0.025, 0.975])
    return [float(lower), float(upper)]


def _mcnemar_exact_pvalue(rl_only: int, greedy_only: int) -> float:
    discordant = rl_only + greedy_only
    if discordant == 0:
        return 1.0
    tail = min(rl_only, greedy_only)
    probability = sum(math.comb(discordant, index) for index in range(tail + 1)) / (2**discordant)
    return float(min(1.0, 2.0 * probability))


def summarize_pairs(pairs: Sequence[PairedCombatResult]) -> dict[str, Any]:
    count = len(pairs)
    rl_wins = sum(item.rl.won for item in pairs)
    greedy_wins = sum(item.greedy.won for item in pairs)
    rl_only = sum(item.rl.won and not item.greedy.won for item in pairs)
    greedy_only = sum(item.greedy.won and not item.rl.won for item in pairs)
    both_win = sum(item.rl.won and item.greedy.won for item in pairs)
    both_lose = count - rl_only - greedy_only - both_win
    win_differences = [float(item.rl.won) - float(item.greedy.won) for item in pairs]
    hp_loss_differences = [item.rl.exact_hp_loss - item.greedy.exact_hp_loss for item in pairs]
    win_difference_ci = _bootstrap_mean_interval(win_differences, seed=11)
    if count < 8:
        diagnosis = "insufficient_data"
    elif win_difference_ci[1] < 0.0:
        diagnosis = "combat_micro_underperformance"
    elif win_difference_ci[0] >= 0.0:
        diagnosis = "macro_bottleneck_supported"
    else:
        diagnosis = "inconclusive_more_pairs_required"
    summary: dict[str, Any] = {
        "schema_version": _SUMMARY_SCHEMA,
        "paired_combats": count,
        "rl": {
            "wins": rl_wins,
            "win_rate": rl_wins / count if count else 0.0,
            "win_rate_wilson95": _wilson_interval(rl_wins, count),
            "mean_exact_hp_loss": statistics.fmean(item.rl.exact_hp_loss for item in pairs) if pairs else 0.0,
            "mean_turns": statistics.fmean(item.rl.turns for item in pairs) if pairs else 0.0,
            "mean_actions": statistics.fmean(item.rl.actions for item in pairs) if pairs else 0.0,
            "deadlocks": sum(item.rl.deadlocked for item in pairs),
            "no_progress_stalls": sum(item.rl.no_progress_stall for item in pairs),
            "revivals": sum(item.rl.revivals for item in pairs),
        },
        "greedy": {
            "wins": greedy_wins,
            "win_rate": greedy_wins / count if count else 0.0,
            "win_rate_wilson95": _wilson_interval(greedy_wins, count),
            "mean_exact_hp_loss": statistics.fmean(item.greedy.exact_hp_loss for item in pairs) if pairs else 0.0,
            "mean_turns": statistics.fmean(item.greedy.turns for item in pairs) if pairs else 0.0,
            "mean_actions": statistics.fmean(item.greedy.actions for item in pairs) if pairs else 0.0,
            "deadlocks": sum(item.greedy.deadlocked for item in pairs),
            "no_progress_stalls": sum(item.greedy.no_progress_stall for item in pairs),
            "revivals": sum(item.greedy.revivals for item in pairs),
        },
        "paired": {
            "both_win": both_win,
            "rl_only_win": rl_only,
            "greedy_only_win": greedy_only,
            "both_lose": both_lose,
            "rl_minus_greedy_win_rate": statistics.fmean(win_differences) if win_differences else 0.0,
            "rl_minus_greedy_win_rate_bootstrap95": win_difference_ci,
            "rl_minus_greedy_exact_hp_loss": statistics.fmean(hp_loss_differences) if hp_loss_differences else 0.0,
            "rl_minus_greedy_exact_hp_loss_bootstrap95": _bootstrap_mean_interval(
                hp_loss_differences,
                seed=17,
            ),
            "mcnemar_exact_two_sided_p": _mcnemar_exact_pvalue(rl_only, greedy_only),
        },
        "diagnosis": diagnosis,
        "diagnosis_contract": {
            "combat_micro_underperformance": "upper 95% paired bootstrap bound for RL-greedy win rate is below zero",
            "macro_bottleneck_supported": "lower 95% paired bootstrap bound for RL-greedy win rate is non-negative",
            "inconclusive_more_pairs_required": "paired win-rate interval crosses zero",
        },
    }
    comparison_labels = sorted(
        {label for pair in pairs for label, _outcome in pair.comparison_rl}
    )
    if comparison_labels:
        comparison_summaries: dict[str, Any] = {}
        for label in comparison_labels:
            projected: list[PairedCombatResult] = []
            for pair in pairs:
                outcomes = dict(pair.comparison_rl)
                if label not in outcomes:
                    continue
                projected.append(
                    replace(
                        pair,
                        rl=outcomes[label],
                        comparison_rl=(),
                    )
                )
            projected_summary = summarize_pairs(projected)
            comparison_summaries[label] = {
                "paired_combats": projected_summary["paired_combats"],
                "rl": projected_summary["rl"],
                "paired_vs_greedy": projected_summary["paired"],
                "diagnosis": projected_summary["diagnosis"],
            }
        summary["comparison_checkpoints"] = comparison_summaries
    return summary


_DEFAULT_COMBAT_ENCOUNTERS = (
    "BOWLBUGS_WEAK",
    "CULTISTS_NORMAL",
    "SLIMES_NORMAL",
    "BYGONE_EFFIGY_ELITE",
)


def _catalog_encounter_ids(client: HeadlessSimBridgeClient) -> set[str]:
    payload = client.combat_catalog()
    values = payload.get("encounters")
    if not isinstance(values, list):
        raise CombatEvaluationError("combat catalog has no encounter list")
    return {
        str(item.get("encounter_id"))
        for item in values
        if isinstance(item, Mapping) and item.get("encounter_id")
    }


def run_paired_combat_evaluation(
    checkpoint: str | Path,
    *,
    comparison_checkpoints: Sequence[str | Path] = (),
    output_directory: str | Path,
    pairs: int,
    encounter_ids: Sequence[str] = (),
    base_seed: int = 0,
    device: str = "cpu",
    sim_exe_path: str | Path | None = None,
    revival_budget: int = 0,
    maximum_combat_steps: int = 512,
    stall_window: int = 128,
    repeat_threshold: int = 8,
    candidate_topk: int = 8,
) -> dict[str, Any]:
    """Evaluate all agents on identical deterministic combat-reset scenarios.

    A scenario is the tuple ``(character, encounter_id, seed, build)``.  The
    current diagnostic deliberately uses the simulator's default starter build
    so the variable under test is combat action selection, not deck building.
    Every frozen policy gets a fresh recurrent state at the encounter boundary.
    The greedy policy restores each decision root by resetting the exact same
    scenario and replaying its committed authoritative action positions.
    """

    if pairs <= 0:
        raise ValueError("pairs must be positive")
    if revival_budget < 0:
        raise ValueError("combat diagnosis requires a finite non-negative revival budget")
    checkpoint_path = Path(checkpoint).expanduser().resolve()
    validated = validate_resume_checkpoint(checkpoint_path)
    output_root = Path(output_directory).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=False)
    journal_path = output_root / "paired-combat-journal.jsonl"
    summary_path = output_root / "summary.json"

    primary_policy = FrozenRlCombatPolicy(checkpoint_path, device=device)
    comparison_policies: list[tuple[str, FrozenRlCombatPolicy]] = []
    seen_labels: set[str] = set()
    for value in comparison_checkpoints:
        comparison_path = Path(value).expanduser().resolve()
        comparison_policy = FrozenRlCombatPolicy(comparison_path, device=device)
        if comparison_policy.config.model != primary_policy.config.model:
            raise CombatEvaluationError(
                "comparison checkpoint model ABI differs from primary: "
                f"{comparison_path}"
            )
        label_base = comparison_path.name
        label = label_base
        suffix = 2
        while label in seen_labels:
            label = f"{label_base}-{suffix}"
            suffix += 1
        seen_labels.add(label)
        comparison_policies.append((label, comparison_policy))

    client = HeadlessSimBridgeClient(exe_path=sim_exe_path)
    results: list[PairedCombatResult] = []
    requested_encounters = tuple(encounter_ids) or _DEFAULT_COMBAT_ENCOUNTERS
    seeds = held_out_evaluation_seeds(base_seed, pairs)
    try:
        catalog = _catalog_encounter_ids(client)
        unknown = sorted(set(requested_encounters) - catalog)
        if unknown:
            raise CombatEvaluationError(f"unknown combat-reset encounters: {unknown}")
        _append_jsonl(
            journal_path,
            {
                "schema_version": _JOURNAL_SCHEMA,
                "event": "evaluation_started",
                "checkpoint": str(checkpoint_path),
                "checkpoint_id": validated.manifest.get("checkpoint_id"),
                "comparison_checkpoints": [
                    {
                        "label": label,
                        "checkpoint": str(item.checkpoint),
                        "checkpoint_id": item.checkpoint_id,
                    }
                    for label, item in comparison_policies
                ],
                "requested_pairs": pairs,
                "held_out_seeds": seeds,
                "encounter_cycle": requested_encounters,
                "revival_budget": revival_budget,
                "combat_reset_build": "simulator_default_starter_build",
                "recurrent_state": "fresh_at_combat_boundary",
                "branching": "deterministic_reset_plus_action_prefix_replay",
                "combat_only_statistics": True,
                "one_step_evaluator": {
                    "terminal_win": 1_000_000.0,
                    "terminal_loss": -1_000_000.0,
                    "hp_survival": 200.0,
                    "enemy_remaining": -100.0,
                    "enemy_kills": 25.0,
                    "covered_intent": 8.0,
                    "uncovered_intent": -20.0,
                    "selection_progress": 2.0,
                    "no_factual_progress": -0.05,
                },
            },
        )

        for pair_index, evaluation_seed in enumerate(seeds):
            encounter_id = requested_encounters[pair_index % len(requested_encounters)]
            spec = CombatResetSpec(
                character=primary_policy.config.environment.character or "ironclad",
                encounter_id=encounter_id,
                seed=evaluation_seed,
                revival_budget=revival_budget,
            )
            root = _prepare_combat_reset(client, spec)
            root_fingerprint = combat_state_fingerprint(root)
            observation = _result_observation(root)
            act, floor, observed_encounter, room_type = _encounter_metadata(observation)
            if observed_encounter != "unknown" and encounter_id not in observed_encounter:
                raise CombatEvaluationError(
                    "combat_reset returned an unexpected encounter: "
                    f"requested={encounter_id} observed={observed_encounter}"
                )

            primary_root = _prepare_combat_reset(client, spec)
            if combat_state_fingerprint(primary_root) != root_fingerprint:
                raise CombatEvaluationError("primary reset did not reproduce the scenario root")
            _primary_final, primary_outcome = _run_combat(
                client,
                primary_root,
                agent="rl",
                rl_policy=primary_policy,
                rl_recurrent_state=primary_policy.initial_state(),
                greedy_policy=None,
                maximum_steps=maximum_combat_steps,
                stall_window=stall_window,
                repeat_threshold=repeat_threshold,
            )

            comparison_outcomes: list[tuple[str, CombatOutcome]] = []
            for label, comparison_policy in comparison_policies:
                comparison_root = _prepare_combat_reset(client, spec)
                if combat_state_fingerprint(comparison_root) != root_fingerprint:
                    raise CombatEvaluationError(
                        f"comparison policy {label} did not receive the same scenario root"
                    )
                _comparison_final, comparison_outcome = _run_combat(
                    client,
                    comparison_root,
                    agent=label,
                    rl_policy=comparison_policy,
                    rl_recurrent_state=comparison_policy.initial_state(),
                    greedy_policy=None,
                    maximum_steps=maximum_combat_steps,
                    stall_window=stall_window,
                    repeat_threshold=repeat_threshold,
                )
                comparison_outcomes.append((label, comparison_outcome))

            replay = DeterministicCombatReplay(client, spec)
            greedy_root = replay.reset()
            if combat_state_fingerprint(greedy_root) != root_fingerprint:
                raise CombatEvaluationError("greedy policy did not receive the same scenario root")
            greedy_policy = OneStepGreedyCombatPolicy(replay, candidate_topk=candidate_topk)
            _greedy_final, greedy_outcome = _run_combat(
                client,
                greedy_root,
                agent="greedy",
                rl_policy=None,
                rl_recurrent_state=None,
                greedy_policy=greedy_policy,
                maximum_steps=maximum_combat_steps,
                stall_window=stall_window,
                repeat_threshold=repeat_threshold,
            )

            pair = PairedCombatResult(
                pair_index=pair_index,
                evaluation_seed=evaluation_seed,
                act=act,
                floor=floor,
                encounter_id=encounter_id,
                room_type=room_type,
                initial_fingerprint=root_fingerprint,
                revival_budget=revival_budget,
                rl=primary_outcome,
                greedy=greedy_outcome,
                comparison_rl=tuple(comparison_outcomes),
            )
            results.append(pair)
            _append_jsonl(
                journal_path,
                {
                    "schema_version": _JOURNAL_SCHEMA,
                    "event": "paired_combat",
                    "pair_index": pair_index,
                    "evaluation_seed": evaluation_seed,
                    "encounter_id": encounter_id,
                    "room_type": room_type,
                    "initial_fingerprint": root_fingerprint,
                    "revival_budget": revival_budget,
                    "rl": _outcome_payload(primary_outcome),
                    "comparison_rl": {
                        label: _outcome_payload(outcome)
                        for label, outcome in comparison_outcomes
                    },
                    "greedy": _outcome_payload(greedy_outcome),
                },
            )
    finally:
        client.close()

    summary = summarize_pairs(results)
    summary.update(
        {
            "created_unix_s": time.time(),
            "checkpoint": str(checkpoint_path),
            "checkpoint_id": primary_policy.checkpoint_id,
            "comparison_checkpoint_provenance": [
                {
                    "label": label,
                    "checkpoint": str(item.checkpoint),
                    "checkpoint_id": item.checkpoint_id,
                }
                for label, item in comparison_policies
            ],
            "journal": str(journal_path),
            "requested_pairs": pairs,
            "completed_pairs": len(results),
            "encounter_cycle": list(requested_encounters),
            "revival_budget": revival_budget,
            "held_out_seed_contract": "even-training/odd-held-out-v1",
            "combat_reset_build": "simulator_default_starter_build",
            "recurrent_state": "fresh_at_combat_boundary",
            "branching": "deterministic_reset_plus_action_prefix_replay",
            "combat_only_statistics": True,
        }
    )
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    _append_jsonl(
        journal_path,
        {
            "schema_version": _JOURNAL_SCHEMA,
            "event": "evaluation_finished",
            "summary": summary,
        },
    )
    return summary

__all__ = [
    "CandidateProbe",
    "CombatEvaluationError",
    "CombatFeatures",
    "CombatOutcome",
    "CombatResetSpec",
    "DeterministicCombatReplay",
    "FrozenRlCombatPolicy",
    "GreedyChoice",
    "OneStepGreedyCombatPolicy",
    "PairedCombatResult",
    "StateEvaluation",
    "choose_greedy_candidate",
    "combat_features",
    "combat_state_fingerprint",
    "evaluate_one_step",
    "is_combat_state",
    "run_paired_combat_evaluation",
    "summarize_pairs",
]



