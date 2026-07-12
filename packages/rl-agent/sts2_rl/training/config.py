"""Versioned TOML configuration for the active MuZero training path."""

from __future__ import annotations

import argparse
import sys
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

CONFIG_VERSION = "sts2-training-config-v1"
PROFILE_DIR = Path(__file__).resolve().parents[2] / "config" / "profiles"


def _flatten(mapping: Mapping[str, Any], output: dict[str, Any]) -> None:
    for key, value in mapping.items():
        if key in {"version", "profile"}:
            continue
        if isinstance(value, Mapping):
            _flatten(value, output)
        else:
            normalized = str(key).replace("-", "_")
            if normalized in output:
                raise ValueError(f"duplicate config key after flattening: {normalized}")
            output[normalized] = value


def _load_toml(path: Path) -> tuple[dict[str, Any], str | None]:
    if not path.is_file():
        raise FileNotFoundError(f"training config not found: {path}")
    payload = tomllib.loads(path.read_text(encoding="utf-8"))
    version = str(payload.get("version") or "")
    if version != CONFIG_VERSION:
        raise ValueError(
            f"unsupported training config version {version!r} in {path}; "
            f"expected {CONFIG_VERSION!r}"
        )
    flattened: dict[str, Any] = {}
    _flatten(payload, flattened)
    return flattened, str(payload.get("profile") or "") or None


def load_config_defaults(
    *,
    profile: str = "default",
    config_path: str | Path | None = None,
) -> tuple[dict[str, Any], str]:
    """Merge built-in profile then optional user config."""

    profile_name = str(profile or "default").strip()
    if not profile_name.replace("_", "").replace("-", "").isalnum():
        raise ValueError(f"invalid profile name: {profile_name!r}")
    defaults, declared_profile = _load_toml(PROFILE_DIR / f"{profile_name}.toml")
    effective_profile = declared_profile or profile_name
    if config_path:
        custom, custom_profile = _load_toml(Path(config_path).expanduser())
        defaults.update(custom)
        effective_profile = custom_profile or effective_profile
    return defaults, effective_profile


def _parse_override_value(raw: str) -> Any:
    try:
        return tomllib.loads(f"value = {raw}")["value"]
    except tomllib.TOMLDecodeError:
        return raw


def _apply_overrides(namespace: argparse.Namespace, overrides: Sequence[str]) -> None:
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"--set expects KEY=VALUE, got {item!r}")
        raw_key, raw_value = item.split("=", 1)
        key = raw_key.strip().split(".")[-1].replace("-", "_")
        if not hasattr(namespace, key):
            raise ValueError(f"unknown training config override key: {raw_key!r}")
        setattr(namespace, key, _parse_override_value(raw_value.strip()))


def parse_args_with_config(
    parser: argparse.ArgumentParser,
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    """Apply versioned profile/config defaults before ordinary CLI flags."""

    arguments = list(sys.argv[1:] if argv is None else argv)
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--profile", default="default")
    bootstrap.add_argument("--config")
    bootstrap.add_argument("--set", dest="config_overrides", action="append", default=[])
    known, _ = bootstrap.parse_known_args(arguments)
    defaults, effective_profile = load_config_defaults(
        profile=known.profile,
        config_path=known.config,
    )
    known_dests = {action.dest for action in parser._actions}
    unknown = sorted(key for key in defaults if key not in known_dests)
    if unknown:
        raise ValueError(f"unknown keys in training config: {', '.join(unknown)}")
    parser.set_defaults(**defaults)
    args = parser.parse_args(arguments)
    _apply_overrides(args, list(args.config_overrides or []))
    args.profile = effective_profile
    return args


@dataclass(frozen=True, slots=True)
class ModelConfig:
    obs_mode: str = "token_v3"
    model_arch: str = "token_memory_v1"

    def __post_init__(self) -> None:
        if (self.obs_mode, self.model_arch) != ("token_v3", "token_memory_v1"):
            raise ValueError(
                "active training requires obs_mode=token_v3 and model_arch=token_memory_v1"
            )


@dataclass(frozen=True, slots=True)
class OptimizationConfig:
    learning_rate: float
    batch_size: int
    discount: float
    n_step_return: int

    def __post_init__(self) -> None:
        if self.learning_rate <= 0 or self.batch_size <= 0:
            raise ValueError("learning_rate and batch_size must be positive")
        if not 0.0 < self.discount <= 1.0 or self.n_step_return <= 0:
            raise ValueError("invalid discount/n_step_return")


@dataclass(frozen=True, slots=True)
class EnvironmentConfig:
    backend: str = "live"
    sim_exe_path: str | None = None

    def __post_init__(self) -> None:
        if self.backend not in {"live", "headless"}:
            raise ValueError("environment backend must be live or headless")


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    device: str
    total_timesteps: int
    log_dir: str
    checkpoint_dir: str
    profile: str

    def __post_init__(self) -> None:
        if self.total_timesteps <= 0:
            raise ValueError("total_timesteps must be positive")


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Typed core plus all migration-period CLI options."""

    version: str
    model: ModelConfig
    optimization: OptimizationConfig
    environment: EnvironmentConfig
    runtime: RuntimeConfig
    options: Mapping[str, Any] = field(repr=False)

    @classmethod
    def from_namespace(cls, args: argparse.Namespace) -> TrainingConfig:
        values = dict(vars(args))
        return cls(
            version=CONFIG_VERSION,
            model=ModelConfig(
                obs_mode=str(values["obs_mode"]),
                model_arch=str(values["model_arch"]),
            ),
            optimization=OptimizationConfig(
                learning_rate=float(values["learning_rate"]),
                batch_size=int(values["batch_size"]),
                discount=float(values["discount"]),
                n_step_return=int(values["n_step_return"]),
            ),
            environment=EnvironmentConfig(
                backend=str(values.get("environment_backend") or "live"),
                sim_exe_path=(
                    str(values["sim_exe_path"])
                    if values.get("sim_exe_path") is not None
                    else None
                ),
            ),
            runtime=RuntimeConfig(
                device=str(values["device"]),
                total_timesteps=int(values["total_timesteps"]),
                log_dir=str(values["log_dir"]),
                checkpoint_dir=str(values["checkpoint_dir"]),
                profile=str(values.get("profile") or "default"),
            ),
            options=MappingProxyType(values),
        )

    def option(self, name: str, default: Any = None) -> Any:
        return self.options.get(name, default)
