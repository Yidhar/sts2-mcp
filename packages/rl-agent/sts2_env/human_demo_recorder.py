"""Append-only human demonstration recorder for combat decisions.

This module records the exact bridge-shaped decision surface that the policy
sees: raw observation, legal actions, selected action id/index, and lightweight
outcome metadata.  The decision JSONL intentionally matches
``muzero.demo_dataset``'s schema, so it can be validated/loaded directly for
imitation learning.

The recorder is inert unless explicitly instantiated or enabled via
``STS2_HUMAN_DEMO_RECORD=1`` in :class:`sts2_env.combat_env.CombatSandboxEnv`.
"""

from __future__ import annotations

import json
import os
import uuid
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
DEFAULT_DEMO_DIR = Path(__file__).resolve().parents[1] / "human_demos"


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def local_session_id(prefix: str = "human") -> str:
    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"


def _jsonable(value: Any) -> Any:
    """Return a JSON-serializable clone.

    Bridge payloads are normally plain dict/list/scalar values, but this keeps
    the recorder robust around numpy scalars, arrays, pathlib values, or small
    custom objects accidentally inserted into info/diagnostics.
    """

    try:
        json.dumps(value, ensure_ascii=False)
        return deepcopy(value)
    except Exception:
        pass

    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "tolist"):
        try:
            return _jsonable(value.tolist())
        except Exception:
            return str(value)
    if hasattr(value, "item"):
        try:
            return _jsonable(value.item())
        except Exception:
            return str(value)
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _safe_dict(value: Any) -> dict[str, Any]:
    return _jsonable(value) if isinstance(value, dict) else {}


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if out != out:  # NaN
        return None
    return out


def _safe_action_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    out: list[dict[str, Any]] = []
    for item in value:
        if isinstance(item, dict):
            out.append(_jsonable(item))
    return out


def _action_id(action: Any) -> str:
    if not isinstance(action, dict):
        return ""
    return str(action.get("action_id") or "")


def _infer_tier(encounter_id: str | None, obs: dict[str, Any] | None = None) -> str:
    enc = str(encounter_id or "").lower()
    if "boss" in enc:
        return "boss"
    if "elite" in enc:
        return "elite"
    run = obs.get("run") if isinstance(obs, dict) else None
    if isinstance(run, dict):
        room_type = str(run.get("room_type") or "").lower()
        if "boss" in room_type:
            return "boss"
        if "elite" in room_type:
            return "elite"
    return "normal"


def _extract_turn(obs: dict[str, Any] | None) -> int:
    combat = obs.get("combat") if isinstance(obs, dict) else None
    if not isinstance(combat, dict):
        return 0
    for key in ("round", "turn", "turn_number"):
        try:
            return int(float(combat.get(key) or 0))
        except (TypeError, ValueError):
            continue
    return 0


def _extract_player_hp(obs: dict[str, Any] | None) -> float | None:
    player = obs.get("player") if isinstance(obs, dict) else None
    if not isinstance(player, dict):
        return None
    return _safe_float(player.get("hp", player.get("current_hp")))


def _extract_player_max_hp(obs: dict[str, Any] | None) -> float | None:
    player = obs.get("player") if isinstance(obs, dict) else None
    if not isinstance(player, dict):
        return None
    return _safe_float(player.get("max_hp", player.get("maximum_hp")))


def _extract_state_blocks(
    *,
    info: dict[str, Any] | None,
    next_obs: dict[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return best-effort ``(player, combat)`` blocks for a post-step state."""

    info = info or {}
    transition_state = info.get("transition_state") if isinstance(info, dict) else None
    if isinstance(transition_state, dict):
        player = transition_state.get("player")
        combat = transition_state.get("combat")
        return (
            player if isinstance(player, dict) else {},
            combat if isinstance(combat, dict) else {},
        )
    if isinstance(next_obs, dict):
        player = next_obs.get("player")
        combat = next_obs.get("combat")
        return (
            player if isinstance(player, dict) else {},
            combat if isinstance(combat, dict) else {},
        )
    return {}, {}


def _terminal_outcome(
    *,
    reward: float | None,
    done: bool,
    truncated: bool,
    info: dict[str, Any] | None,
    next_obs: dict[str, Any] | None,
    start_hp: float | None = None,
    start_max_hp: float | None = None,
    start_turn: int | None = None,
    fallback_turn: int | None = None,
    decision_count: int | None = None,
) -> dict[str, Any]:
    player, combat = _extract_state_blocks(info=info, next_obs=next_obs)
    hp = _safe_float(player.get("hp", player.get("current_hp")))
    max_hp = _safe_float(player.get("max_hp", player.get("maximum_hp")))
    if max_hp is None:
        max_hp = start_max_hp
    enemies = combat.get("enemies") if isinstance(combat, dict) else None
    enemy_hp_total = None
    if isinstance(enemies, list):
        total = 0.0
        for enemy in enemies:
            if not isinstance(enemy, dict):
                continue
            try:
                total += float(enemy.get("hp", enemy.get("current_hp")) or 0.0)
            except (TypeError, ValueError):
                continue
        enemy_hp_total = total
    hp_loss = None
    if start_hp is not None and hp is not None:
        hp_loss = max(float(start_hp) - float(hp), 0.0)
    hp_loss_ratio = None
    denom = max_hp if max_hp and max_hp > 0 else start_max_hp
    if hp_loss is not None and denom and denom > 0:
        hp_loss_ratio = float(hp_loss) / float(denom)
    end_turn = _extract_turn(next_obs) if isinstance(next_obs, dict) else 0
    if end_turn <= 0:
        end_turn = _safe_int(combat.get("round") or combat.get("turn") or combat.get("turn_number")) or 0
    if end_turn <= 0 and fallback_turn is not None:
        end_turn = int(fallback_turn)
    turns = None
    if start_turn is not None and end_turn > 0:
        turns = max(int(end_turn) - int(start_turn) + 1, 1)
    win: bool | None = None
    if done and enemy_hp_total is not None:
        win = enemy_hp_total <= 0.0
    out = {
        "done": bool(done),
        "truncated": bool(truncated),
        "reward": float(reward or 0.0),
        "combat_win": win,
        "player_hp": hp,
        "player_max_hp": max_hp,
        "enemy_hp_total": enemy_hp_total,
        "decision_count": decision_count,
    }
    if hp_loss is not None:
        out["hp_loss"] = hp_loss
    if hp_loss_ratio is not None:
        out["hp_loss_ratio"] = hp_loss_ratio
    if turns is not None:
        out["turns"] = turns
    return out


def _safe_int(value: Any) -> int | None:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


@dataclass
class HumanDemoPaths:
    session_dir: Path
    decisions_path: Path
    episodes_path: Path
    manifest_path: Path


class HumanDemoRecorder:
    """Record combat decisions to JSONL in a MuZero-demo-compatible format."""

    def __init__(
        self,
        output_dir: str | Path | None = None,
        *,
        session_id: str | None = None,
        source: str = "human",
        flush_each: bool = True,
    ) -> None:
        self.session_id = session_id or local_session_id(source)
        self.source = source
        base = Path(output_dir or DEFAULT_DEMO_DIR)
        # If the caller points at a .jsonl file, use it directly for decisions
        # and put sidecar files next to it.  Otherwise create a session dir.
        if base.suffix.lower() == ".jsonl":
            session_dir = base.parent
            decisions_path = base
        else:
            session_dir = base / self.session_id
            decisions_path = session_dir / "decisions.jsonl"
        self.paths = HumanDemoPaths(
            session_dir=session_dir,
            decisions_path=decisions_path,
            episodes_path=session_dir / "episodes.jsonl",
            manifest_path=session_dir / "manifest.json",
        )
        self.flush_each = bool(flush_each)
        self.paths.session_dir.mkdir(parents=True, exist_ok=True)
        self._decision_fh = self.paths.decisions_path.open("a", encoding="utf-8")
        self._episode_fh = self.paths.episodes_path.open("a", encoding="utf-8")
        self._episode_id: str = ""
        self._encounter_id: str = ""
        self._step_in_episode: int = 0
        self._step_in_turn: int = 0
        self._last_turn: int = -1
        self._episode_start_hp: float | None = None
        self._episode_start_max_hp: float | None = None
        self._episode_start_turn: int | None = None
        self._write_manifest()

    @classmethod
    def from_env(cls, *, default_source: str = "human") -> "HumanDemoRecorder | None":
        flag = os.environ.get("STS2_HUMAN_DEMO_RECORD", "").strip().lower()
        if flag not in {"1", "true", "yes", "on"}:
            return None
        output_dir = os.environ.get("STS2_HUMAN_DEMO_DIR") or None
        session_id = os.environ.get("STS2_HUMAN_DEMO_SESSION") or None
        source = os.environ.get("STS2_HUMAN_DEMO_SOURCE") or default_source
        return cls(output_dir=output_dir, session_id=session_id, source=source)

    def close(self) -> None:
        for fh in (self._decision_fh, self._episode_fh):
            try:
                fh.flush()
                fh.close()
            except Exception:
                pass

    def __enter__(self) -> "HumanDemoRecorder":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
        self.close()

    def _write_manifest(self) -> None:
        manifest = {
            "version": SCHEMA_VERSION,
            "session_id": self.session_id,
            "source": self.source,
            "created_at": utc_timestamp(),
            "decisions_path": str(self.paths.decisions_path),
            "episodes_path": str(self.paths.episodes_path),
            "format": "sts2_muzero_human_demo_decision_jsonl",
        }
        self.paths.manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def start_episode(
        self,
        *,
        episode_id: str | None,
        encounter_id: str | None,
        reset_kwargs: dict[str, Any] | None = None,
        obs: dict[str, Any] | None = None,
        legal_actions: list[dict[str, Any]] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._episode_id = str(episode_id or "")
        self._encounter_id = str(encounter_id or "")
        self._step_in_episode = 0
        self._step_in_turn = 0
        self._last_turn = _extract_turn(obs)
        self._episode_start_hp = _extract_player_hp(obs)
        self._episode_start_max_hp = _extract_player_max_hp(obs)
        self._episode_start_turn = self._last_turn if self._last_turn > 0 else None
        row = {
            "version": SCHEMA_VERSION,
            "event": "episode_start",
            "source": self.source,
            "timestamp": utc_timestamp(),
            "session_id": self.session_id,
            "episode_id": self._episode_id,
            "encounter_id": self._encounter_id,
            "tier": _infer_tier(self._encounter_id, obs),
            "turn": self._last_turn,
            "legal_action_count": len(legal_actions or []),
            "reset_kwargs": _jsonable(reset_kwargs or {}),
            "metadata": _jsonable(metadata or {}),
        }
        self._episode_fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        if self.flush_each:
            self._episode_fh.flush()

    def record_decision(
        self,
        *,
        obs: dict[str, Any],
        legal_actions: list[dict[str, Any]],
        selected_action: dict[str, Any],
        selected_action_index: int,
        next_obs: dict[str, Any] | None = None,
        reward: float | None = None,
        done: bool = False,
        truncated: bool = False,
        info: dict[str, Any] | None = None,
        reason_tags: list[str] | None = None,
        comment: str = "",
    ) -> None:
        turn = _extract_turn(obs)
        if turn != self._last_turn:
            self._step_in_turn = 0
            self._last_turn = turn
        else:
            self._step_in_turn += 1
        encounter_id = self._encounter_id or str((info or {}).get("encounter_id") or "")
        selected_id = _action_id(selected_action)
        safe_legal_actions = _safe_action_list(legal_actions)
        prev_hp = _extract_player_hp(obs)
        next_hp = _extract_player_hp(next_obs)
        transition_hp_loss = None
        if prev_hp is not None and next_hp is not None:
            transition_hp_loss = max(float(prev_hp) - float(next_hp), 0.0)
        decision_count = self._step_in_episode + 1
        row = {
            "version": SCHEMA_VERSION,
            "event": "decision",
            "source": self.source,
            "timestamp": utc_timestamp(),
            "session_id": self.session_id,
            "episode_id": self._episode_id or str((info or {}).get("episode_id") or ""),
            "encounter_id": encounter_id,
            "tier": _infer_tier(encounter_id, obs),
            "turn": turn,
            "step_in_turn": self._step_in_turn,
            "step_in_episode": self._step_in_episode,
            "obs": _safe_dict(obs),
            "legal_actions": safe_legal_actions,
            "selected_action_id": selected_id,
            "selected_action_index": int(selected_action_index),
            "selected_action": _safe_dict(selected_action),
            "reason_tags": list(reason_tags or []),
            "comment": str(comment or ""),
            "reward": float(reward or 0.0),
            "done": bool(done),
            "truncated": bool(truncated),
            "transition": {
                "player_hp_before": prev_hp,
                "player_hp_after": next_hp,
                "hp_loss": transition_hp_loss,
            },
            "outcome": _terminal_outcome(
                reward=reward,
                done=done,
                truncated=truncated,
                info=info,
                next_obs=next_obs,
                start_hp=self._episode_start_hp,
                start_max_hp=self._episode_start_max_hp,
                start_turn=self._episode_start_turn,
                fallback_turn=turn,
                decision_count=decision_count,
            )
            if (done or truncated)
            else {},
        }
        self._decision_fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        if self.flush_each:
            self._decision_fh.flush()
        self._step_in_episode += 1
        if done or truncated:
            self.end_episode(
                final_obs=next_obs,
                info=info,
                reward=reward,
                done=done,
                truncated=truncated,
            )

    def end_episode(
        self,
        *,
        final_obs: dict[str, Any] | None = None,
        info: dict[str, Any] | None = None,
        reward: float | None = None,
        done: bool = True,
        truncated: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        row = {
            "version": SCHEMA_VERSION,
            "event": "episode_end",
            "source": self.source,
            "timestamp": utc_timestamp(),
            "session_id": self.session_id,
            "episode_id": self._episode_id,
            "encounter_id": self._encounter_id,
            "tier": _infer_tier(self._encounter_id, final_obs),
            "step_count": self._step_in_episode,
            "outcome": _terminal_outcome(
                reward=reward,
                done=done,
                truncated=truncated,
                info=info,
                next_obs=final_obs,
                start_hp=self._episode_start_hp,
                start_max_hp=self._episode_start_max_hp,
                start_turn=self._episode_start_turn,
                decision_count=self._step_in_episode,
            ),
            "metadata": _jsonable(metadata or {}),
        }
        self._episode_fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        if self.flush_each:
            self._episode_fh.flush()
