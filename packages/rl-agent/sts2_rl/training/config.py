"""Strict configuration ABI for the relational recurrent V-trace v3 baseline."""

from __future__ import annotations

import math
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Literal, TypeVar

from sts2_baseline import TASK_REWARD_SPEC
from sts2_rl.encoding import GroundedEncodingConfig
from sts2_rl.models import GroundedCandidateConfig

from .seeding import validate_seed_budget

CONFIG_VERSION = "sts2-relational-curriculum-config-v3"
PROFILE_DIR = Path(__file__).resolve().parents[2] / "config" / "profiles"
T = TypeVar("T")


def _require_int(value: object, *, label: str, minimum: int | None = None) -> int:
    """Validate a config integer without accepting ``bool`` or lossy coercion."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{label} must be >= {minimum}")
    return value


def _require_finite_number(
    value: object,
    *,
    label: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    """Validate a finite TOML number without treating booleans as numbers."""

    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{label} must be a number")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{label} must be finite")
    if minimum is not None and normalized < minimum:
        raise ValueError(f"{label} must be >= {minimum}")
    if maximum is not None and normalized > maximum:
        raise ValueError(f"{label} must be <= {maximum}")
    return normalized


def _require_optional_text(value: object, *, label: str) -> None:
    if value is None:
        return
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{label} must be a non-empty string or null")


@dataclass(frozen=True, slots=True)
class ModelConfig:
    architecture: str = "relational_candidate_v3"
    token_feature_dim: int = 224
    d_model: int = 128
    n_heads: int = 4
    ffn_dim: int = 384
    world_layers: int = 3
    latent_slots: int = 12
    latent_layers: int = 2
    local_layers: int = 1
    candidate_layers: int = 1
    recurrent_hidden_dim: int = 256
    dropout: float = 0.05
    type_vocab_size: int = 128
    role_vocab_size: int = 64
    owner_vocab_size: int = 128
    entity_vocab_size: int = 8192
    zone_vocab_size: int = 32
    order_vocab_size: int = 128
    domain_count: int = 8
    max_world_tokens: int = 512
    max_candidates: int = 96
    max_candidate_local_tokens: int = 64

    def __post_init__(self) -> None:
        for name in (
            "token_feature_dim",
            "d_model",
            "n_heads",
            "ffn_dim",
            "world_layers",
            "latent_slots",
            "latent_layers",
            "local_layers",
            "candidate_layers",
            "recurrent_hidden_dim",
            "type_vocab_size",
            "role_vocab_size",
            "owner_vocab_size",
            "entity_vocab_size",
            "zone_vocab_size",
            "order_vocab_size",
            "domain_count",
            "max_world_tokens",
            "max_candidates",
            "max_candidate_local_tokens",
        ):
            _require_int(getattr(self, name), label=f"model.{name}", minimum=1)
        _require_finite_number(
            self.dropout,
            label="model.dropout",
            minimum=0.0,
        )
        if self.architecture != "relational_candidate_v3":
            raise ValueError("only architecture='relational_candidate_v3' is supported")
        # Reuse the model's own shape validation as the single source of truth.
        self.to_model_config()
        if (
            min(
                self.max_world_tokens,
                self.max_candidates,
                self.max_candidate_local_tokens,
            )
            <= 0
        ):
            raise ValueError("model token capacities must be positive")

    def to_model_config(self) -> GroundedCandidateConfig:
        return GroundedCandidateConfig(
            token_feature_dim=self.token_feature_dim,
            d_model=self.d_model,
            n_heads=self.n_heads,
            ffn_dim=self.ffn_dim,
            world_layers=self.world_layers,
            latent_slots=self.latent_slots,
            latent_layers=self.latent_layers,
            local_layers=self.local_layers,
            candidate_layers=self.candidate_layers,
            recurrent_hidden_dim=self.recurrent_hidden_dim,
            dropout=self.dropout,
            domain_count=self.domain_count,
            type_vocab_size=self.type_vocab_size,
            role_vocab_size=self.role_vocab_size,
            owner_vocab_size=self.owner_vocab_size,
            entity_vocab_size=self.entity_vocab_size,
            zone_vocab_size=self.zone_vocab_size,
            order_vocab_size=self.order_vocab_size,
        )

    def to_encoding_config(self) -> GroundedEncodingConfig:
        return GroundedEncodingConfig.from_model_config(
            self.to_model_config(),
            max_world_tokens=self.max_world_tokens,
            max_candidates=self.max_candidates,
            max_candidate_local_tokens=self.max_candidate_local_tokens,
        )


@dataclass(frozen=True, slots=True)
class OptimizationConfig:
    learning_rate: float = 3.0e-4
    weight_decay: float = 1.0e-4
    batch_unrolls: int = 8
    discount: float = 0.997
    gradient_clip_norm: float = 1.0
    vtrace_rho_clip: float = 1.0
    vtrace_c_clip: float = 1.0
    policy_rho_clip: float = 1.0
    policy_weight: float = 1.0
    value_weight: float = 0.5
    entropy_weight: float = 0.01

    def __post_init__(self) -> None:
        learning_rate = _require_finite_number(
            self.learning_rate,
            label="optimization.learning_rate",
        )
        weight_decay = _require_finite_number(
            self.weight_decay,
            label="optimization.weight_decay",
        )
        _require_int(
            self.batch_unrolls,
            label="optimization.batch_unrolls",
            minimum=1,
        )
        discount = _require_finite_number(
            self.discount,
            label="optimization.discount",
        )
        gradient_clip_norm = _require_finite_number(
            self.gradient_clip_norm,
            label="optimization.gradient_clip_norm",
        )
        if learning_rate <= 0.0 or weight_decay < 0.0:
            raise ValueError("learning_rate must be positive and weight_decay non-negative")
        if not 0.0 < discount <= 1.0:
            raise ValueError("discount must be in (0, 1]")
        if gradient_clip_norm <= 0.0:
            raise ValueError("gradient_clip_norm must be positive")
        for name in ("vtrace_rho_clip", "vtrace_c_clip", "policy_rho_clip"):
            value = _require_finite_number(
                getattr(self, name),
                label=f"optimization.{name}",
            )
            if value <= 0.0:
                raise ValueError(f"{name} must be positive")
        for name in (
            "policy_weight",
            "value_weight",
            "entropy_weight",
        ):
            value = _require_finite_number(
                getattr(self, name),
                label=f"optimization.{name}",
            )
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative")


@dataclass(frozen=True, slots=True)
class RolloutConfig:
    """Short-lived FIFO data plane; every unroll is consumed at most once."""

    unroll_length: int = 64
    queue_capacity: int = 256
    minimum_unrolls: int = 8
    collector_workers: int = 1
    policy_sync_interval_unrolls: int = 8
    max_policy_lag: int = 1_024

    def __post_init__(self) -> None:
        for name in (
            "unroll_length",
            "queue_capacity",
            "minimum_unrolls",
            "collector_workers",
            "policy_sync_interval_unrolls",
            "max_policy_lag",
        ):
            _require_int(
                getattr(self, name),
                label=f"rollout.{name}",
                minimum=1,
            )
        if self.minimum_unrolls > self.queue_capacity:
            raise ValueError("rollout minimum_unrolls cannot exceed queue_capacity")
        if self.collector_workers != 1:
            raise ValueError("v2 currently requires one collector worker per typed backend session")


@dataclass(frozen=True, slots=True)
class EnvironmentConfig:
    backend: Literal["live", "headless"] = "headless"
    scenario: Literal["full-run", "combat"] = "full-run"
    session_path: str | None = None
    sim_exe_path: str | None = None
    character: str | None = None
    encounter_id: str | None = None
    max_episode_steps: int = 10_000

    def __post_init__(self) -> None:
        if self.backend not in {"live", "headless"}:
            raise ValueError("environment backend must be live or headless")
        if self.scenario not in {"full-run", "combat"}:
            raise ValueError("environment scenario must be full-run or combat")
        _require_int(
            self.max_episode_steps,
            label="environment.max_episode_steps",
            minimum=1,
        )
        for name in ("session_path", "sim_exe_path", "character", "encounter_id"):
            _require_optional_text(
                getattr(self, name),
                label=f"environment.{name}",
            )
        if self.backend == "headless" and self.session_path is not None:
            raise ValueError("environment.session_path is only valid for the live backend")
        if self.backend == "live" and self.sim_exe_path is not None:
            raise ValueError("environment.sim_exe_path is only valid for the headless backend")
        if self.scenario == "full-run" and self.encounter_id is not None:
            raise ValueError("environment.encounter_id is only valid for combat scenarios")


@dataclass(frozen=True, slots=True)
class CurriculumConfig:
    """Task horizon and exploration schedule without mechanics rules."""

    mode: Literal["standard", "native-revival-preheat"] = "standard"
    reward_objective: Literal["combat", "act1", "run"] = "run"
    revival_relic_id: str | None = None
    revival_budget: int | None = None
    epsilon_start: float = 0.30
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 250_000

    def __post_init__(self) -> None:
        if self.mode not in {"standard", "native-revival-preheat"}:
            raise ValueError("unsupported curriculum mode")
        if self.reward_objective not in {"combat", "act1", "run"}:
            raise ValueError("reward_objective must be combat, act1, or run")
        _require_optional_text(
            self.revival_relic_id,
            label="curriculum.revival_relic_id",
        )
        if self.mode == "standard" and self.revival_relic_id is not None:
            raise ValueError("standard curriculum cannot inject a revival relic")
        if self.mode == "standard" and self.revival_budget is not None:
            raise ValueError("standard curriculum cannot set a revival budget")
        if self.mode == "native-revival-preheat" and self.revival_relic_id is None:
            raise ValueError("native revival preheat requires revival_relic_id")
        if self.mode == "native-revival-preheat" and self.revival_budget is None:
            raise ValueError("native revival preheat requires revival_budget")
        if self.revival_budget is not None:
            _require_int(
                self.revival_budget,
                label="curriculum.revival_budget",
                minimum=-1,
            )
        epsilon_start = _require_finite_number(
            self.epsilon_start,
            label="curriculum.epsilon_start",
        )
        epsilon_end = _require_finite_number(
            self.epsilon_end,
            label="curriculum.epsilon_end",
        )
        if not 0.0 <= epsilon_end <= epsilon_start <= 1.0:
            raise ValueError("exploration epsilon must satisfy 0 <= end <= start <= 1")
        _require_int(
            self.epsilon_decay_steps,
            label="curriculum.epsilon_decay_steps",
            minimum=1,
        )


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    device: str = "auto"
    collector_device: str = "cpu"
    total_environment_steps: int = 1_000_000
    seed: int = 0
    log_dir: str = "runs/recurrent-vtrace"
    checkpoint_dir: str = "checkpoints/recurrent-vtrace"
    checkpoint_interval_steps: int = 25_000
    evaluation_steps: tuple[int, ...] = (0, 10_000, 25_000, 50_000)
    evaluation_episodes: int = 20

    def __post_init__(self) -> None:
        for name in (
            "total_environment_steps",
            "checkpoint_interval_steps",
        ):
            _require_int(
                getattr(self, name),
                label=f"runtime.{name}",
                minimum=1,
            )
        _require_int(
            self.seed,
            label="runtime.seed",
            minimum=0,
        )
        _require_int(
            self.evaluation_episodes,
            label="runtime.evaluation_episodes",
            minimum=0,
        )
        if not isinstance(self.evaluation_steps, tuple):
            object.__setattr__(self, "evaluation_steps", tuple(self.evaluation_steps))
        previous = -1
        for index, step in enumerate(self.evaluation_steps):
            _require_int(step, label=f"runtime.evaluation_steps[{index}]", minimum=0)
            if step <= previous:
                raise ValueError("runtime.evaluation_steps must be strictly increasing")
            previous = step
        if not isinstance(self.device, str) or not self.device.strip():
            raise TypeError("runtime.device must be a non-empty string")
        if not isinstance(self.collector_device, str) or not self.collector_device.strip():
            raise TypeError("runtime.collector_device must be a non-empty string")
        if (
            not isinstance(self.log_dir, str)
            or not isinstance(self.checkpoint_dir, str)
            or not self.log_dir.strip()
            or not self.checkpoint_dir.strip()
        ):
            raise ValueError("log_dir and checkpoint_dir must be non-empty")
        validate_seed_budget(
            self.seed,
            maximum_training_episodes=self.total_environment_steps,
            evaluation_episodes=self.evaluation_episodes,
        )


@dataclass(frozen=True, slots=True)
class DiagnosticsConfig:
    deadlock_window: int = 128
    deadlock_repeat_threshold: int = 8
    journal_policy_topk: int = 5

    def __post_init__(self) -> None:
        for name in (
            "deadlock_window",
            "deadlock_repeat_threshold",
            "journal_policy_topk",
        ):
            _require_int(
                getattr(self, name),
                label=f"diagnostics.{name}",
                minimum=1,
            )
        if self.deadlock_window < 2:
            raise ValueError("diagnostics.deadlock_window must be at least 2")
        if self.deadlock_repeat_threshold < 2:
            raise ValueError("diagnostics.deadlock_repeat_threshold must be at least 2")
        if self.deadlock_repeat_threshold > self.deadlock_window:
            raise ValueError("diagnostics.deadlock_repeat_threshold cannot exceed deadlock_window")


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    version: str = CONFIG_VERSION
    profile: str = "default"
    model: ModelConfig = field(default_factory=ModelConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    rollout: RolloutConfig = field(default_factory=RolloutConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    diagnostics: DiagnosticsConfig = field(default_factory=DiagnosticsConfig)

    def __post_init__(self) -> None:
        if not isinstance(self.version, str):
            raise TypeError("training config version must be a string")
        if self.version != CONFIG_VERSION:
            raise ValueError(f"unsupported training config version {self.version!r}; expected {CONFIG_VERSION!r}")
        if not isinstance(self.profile, str) or not self.profile.strip():
            raise ValueError("training profile must be non-empty")
        for name, expected_type in (
            ("model", ModelConfig),
            ("optimization", OptimizationConfig),
            ("rollout", RolloutConfig),
            ("environment", EnvironmentConfig),
            ("curriculum", CurriculumConfig),
            ("runtime", RuntimeConfig),
            ("diagnostics", DiagnosticsConfig),
        ):
            if not isinstance(getattr(self, name), expected_type):
                raise TypeError(f"training config {name} must be {expected_type.__name__}")
        if self.environment.scenario == "combat" and self.curriculum.reward_objective != "combat":
            raise ValueError(
                "reward objective must match the environment horizon: " "combat scenarios require objective='combat'"
            )
        if self.environment.scenario == "full-run" and self.curriculum.reward_objective == "combat":
            raise ValueError("full-run scenarios require objective='act1' or objective='run'")
        if self.curriculum.mode == "native-revival-preheat":
            if self.environment.backend != "headless":
                raise ValueError("native revival preheat requires the headless backend")
        expected_discount = 1.0 if self.curriculum.mode == "native-revival-preheat" else TASK_REWARD_SPEC.discount
        if self.optimization.discount != expected_discount:
            raise ValueError(
                "optimization.discount must equal the reward contract discount "
                f"{expected_discount} for curriculum mode {self.curriculum.mode!r}"
            )

    def to_mapping(self) -> dict[str, Any]:
        return asdict(self)

    def lineage_mapping(self) -> dict[str, Any]:
        """Return only immutable training semantics for exact-resume identity.

        Execution horizon, output locations, and observation-only schedules may
        change when continuing a checkpoint.  Model, task, rollout, optimizer,
        seed, and environment semantics remain part of exact lineage identity.
        """

        payload = self.to_mapping()
        runtime = payload["runtime"]
        if not isinstance(runtime, dict):  # pragma: no cover - asdict invariant
            raise TypeError("serialized runtime config must be an object")
        for key in (
            "total_environment_steps",
            "log_dir",
            "checkpoint_dir",
            "checkpoint_interval_steps",
            "evaluation_steps",
            "evaluation_episodes",
        ):
            runtime.pop(key)
        return payload


def _load_toml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"training config not found: {path}")
    payload = tomllib.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"training config must be a TOML table: {path}")
    return payload


def _deep_merge(base: dict[str, Any], override: Mapping[str, Any]) -> None:
    for raw_key, value in override.items():
        key = str(raw_key).replace("-", "_")
        if isinstance(value, Mapping) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = dict(value) if isinstance(value, Mapping) else value


def _override_value(raw: str) -> Any:
    try:
        return tomllib.loads(f"value = {raw}")["value"]
    except tomllib.TOMLDecodeError:
        return raw


def _apply_dotted_override(payload: dict[str, Any], item: str) -> None:
    if "=" not in item:
        raise ValueError(f"config override must be KEY=VALUE, got {item!r}")
    raw_path, raw_value = item.split("=", 1)
    parts = [part.strip().replace("-", "_") for part in raw_path.split(".") if part.strip()]
    if not parts:
        raise ValueError(f"config override has no key: {item!r}")
    cursor = payload
    for part in parts[:-1]:
        child = cursor.get(part)
        if not isinstance(child, dict):
            raise ValueError(f"unknown/non-table config override path: {raw_path!r}")
        cursor = child
    if parts[-1] not in cursor:
        raise ValueError(f"unknown config override key: {raw_path!r}")
    cursor[parts[-1]] = _override_value(raw_value.strip())


def _construct(cls: type[T], payload: Mapping[str, Any], *, label: str) -> T:
    raw_fields = getattr(cls, "__dataclass_fields__", None)
    if not isinstance(raw_fields, dict):
        raise TypeError(f"{cls!r} is not a dataclass type")
    allowed = set(raw_fields)
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(f"unknown {label} config keys: {', '.join(unknown)}")
    return cls(**dict(payload))


def training_config_from_mapping(payload: Mapping[str, Any]) -> TrainingConfig:
    allowed = {field.name for field in fields(TrainingConfig)}
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(f"unknown training config sections: {', '.join(unknown)}")
    version = payload.get("version", CONFIG_VERSION)
    profile = payload.get("profile", "default")
    if not isinstance(version, str):
        raise TypeError("training config version must be a string")
    if not isinstance(profile, str):
        raise TypeError("training config profile must be a string")
    return TrainingConfig(
        version=version,
        profile=profile,
        model=_construct(ModelConfig, _table(payload, "model"), label="model"),
        optimization=_construct(OptimizationConfig, _table(payload, "optimization"), label="optimization"),
        rollout=_construct(RolloutConfig, _table(payload, "rollout"), label="rollout"),
        environment=_construct(EnvironmentConfig, _table(payload, "environment"), label="environment"),
        curriculum=_construct(CurriculumConfig, _table(payload, "curriculum"), label="curriculum"),
        runtime=_construct(RuntimeConfig, _table(payload, "runtime"), label="runtime"),
        diagnostics=_construct(DiagnosticsConfig, _table(payload, "diagnostics"), label="diagnostics"),
    )


def _table(payload: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = payload.get(name, {})
    if not isinstance(value, Mapping):
        raise ValueError(f"training config section {name!r} must be a table")
    return value


def load_training_config(
    *,
    profile: str = "default",
    config_path: str | Path | None = None,
    overrides: Sequence[str] = (),
) -> TrainingConfig:
    profile_name = str(profile).strip()
    if not profile_name.replace("_", "").replace("-", "").isalnum():
        raise ValueError(f"invalid profile name: {profile_name!r}")
    payload = _load_toml(PROFILE_DIR / f"{profile_name}.toml")
    if config_path is not None:
        _deep_merge(payload, _load_toml(Path(config_path).expanduser()))
    for item in overrides:
        _apply_dotted_override(payload, item)
    payload["profile"] = profile_name
    return training_config_from_mapping(payload)


def replace_runtime(config: TrainingConfig, **changes: Any) -> TrainingConfig:
    """Typed CLI convenience without reintroducing a flat option bag."""

    return replace(config, runtime=replace(config.runtime, **changes))


__all__ = [
    "CONFIG_VERSION",
    "PROFILE_DIR",
    "CurriculumConfig",
    "DiagnosticsConfig",
    "EnvironmentConfig",
    "ModelConfig",
    "OptimizationConfig",
    "RolloutConfig",
    "RuntimeConfig",
    "TrainingConfig",
    "load_training_config",
    "replace_runtime",
    "training_config_from_mapping",
]
