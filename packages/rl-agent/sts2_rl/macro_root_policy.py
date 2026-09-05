"""Read-only frozen hybrid policy for same-root macro diagnostics.

This adapter owns inference only.  It does not build training resources, an
optimizer, a replay store or a simulator, and it never publishes a checkpoint.
An IQL parent is accepted only as an explicit, in-memory model initialization;
its historical reward identity is not confused with the diagnostic course.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

import torch
from torch import Tensor

from sts2_baseline import (
    absolute_death_price_reward_identity,
    act_frontloaded_death_reward_identity,
)
from sts2_rl.encoding.snapshot import EncodedDecisionSnapshot, collate_encoded_snapshots
from sts2_rl.models.hybrid_actor_critic import (
    HybridActorCriticConfig,
    HybridDomainActorCriticModel,
    initialize_actor_critic_from_iql,
)
from sts2_rl.models.hybrid_iql import HybridIQLConfig
from sts2_rl.training.config import TrainingConfig
from sts2_rl.training.sdpa import configure_rocm_sdpa_backend

HYBRID_VTRACE_PUBLICATION: Final = "sts2-hybrid-vtrace-publication-v1"
HYBRID_IQL_PUBLICATION: Final = "sts2-hybrid-publication-v3"
HYBRID_MACRO_CREDIT_PUBLICATION: Final = "sts2-hybrid-macro-credit-publication-v1"


def _model_file(path: Path) -> Path:
    if path.is_dir():
        for candidate in (path / "network.pt", path / "publication" / "network.pt"):
            if candidate.is_file():
                return candidate.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"frozen macro policy publication does not exist: {path}")
    return path.resolve()


def _online_tensors(payload: Mapping[str, Any]) -> dict[str, Tensor]:
    raw = payload.get("online")
    if not isinstance(raw, Mapping) or not raw:
        raise ValueError("frozen macro policy publication has no online model")
    result: dict[str, Tensor] = {}
    for name, value in raw.items():
        if not isinstance(name, str) or not torch.is_tensor(value):
            raise ValueError("frozen macro policy online state must map tensor names to tensors")
        if value.is_floating_point() and not bool(torch.isfinite(value).all().item()):
            raise ValueError(f"frozen macro policy source tensor is non-finite: {name}")
        result[name] = value
    return result


class FrozenMacroRootPolicy:
    """Load one frozen publication and expose semantic-snapshot inference.

    ``initialize_from_iql=True`` transfers the IQL actor, value and factual
    representation into the current actor-critic in memory.  All destination
    tensors must be inherited, so missing parameters cannot silently introduce
    a randomly initialized policy.  The original source policy version is
    retained as provenance, not reported as a resumed V-trace learner version.
    """

    def __init__(
        self,
        model_path: Path,
        config: TrainingConfig,
        device: str,
        initialize_from_iql: bool = False,
    ) -> None:
        source_path = _model_file(model_path)
        resolved_device = "cuda" if device == "auto" and torch.cuda.is_available() else device
        if resolved_device == "auto":
            resolved_device = "cpu"
        self.device = torch.device(resolved_device)
        self.encoding_config = config.model.to_encoding_config()
        model_config = HybridActorCriticConfig(config.model.to_model_config())
        revival_budget = config.curriculum.revival_budget
        if isinstance(revival_budget, bool) or not isinstance(revival_budget, int) or revival_budget <= 0:
            raise ValueError("frozen macro policy diagnostic requires a positive revival budget")
        reward_identity = act_frontloaded_death_reward_identity(revival_budget=revival_budget)
        payload = torch.load(source_path, map_location="cpu", weights_only=True)
        if not isinstance(payload, Mapping):
            raise ValueError("frozen macro policy publication must be a mapping")
        allowed_formats = (
            (HYBRID_IQL_PUBLICATION,)
            if initialize_from_iql
            else (HYBRID_VTRACE_PUBLICATION, HYBRID_MACRO_CREDIT_PUBLICATION)
        )
        expected_format = payload.get("format")
        if expected_format not in allowed_formats:
            raise ValueError(
                f"frozen macro policy expected {allowed_formats}; "
                "IQL parents require explicit initialize_from_iql=True"
            )
        source_model_config = (
            HybridIQLConfig(config.model.to_model_config()).as_mapping()
            if initialize_from_iql
            else model_config.as_mapping()
        )
        if payload.get("model_config") != source_model_config:
            raise ValueError("frozen macro policy publication model_config differs")
        source_reward_identity = absolute_death_price_reward_identity() if initialize_from_iql else reward_identity
        if payload.get("reward_identity") != source_reward_identity:
            raise ValueError("frozen macro policy publication reward_identity differs")
        if (initialize_from_iql or expected_format == HYBRID_MACRO_CREDIT_PUBLICATION) and payload.get(
            "control_scores"
        ) != "policy_logits":
            raise ValueError("frozen IQL/macro-credit publication must declare policy_logits control scores")
        policy_version = payload.get("policy_version")
        if isinstance(policy_version, bool) or not isinstance(policy_version, int) or policy_version < 0:
            raise ValueError("frozen macro policy publication policy_version must be non-negative")
        online = _online_tensors(payload)
        configure_rocm_sdpa_backend(config.runtime.rocm_sdpa_backend, devices=(self.device,))
        self.model = HybridDomainActorCriticModel(model_config)
        if initialize_from_iql:
            inherited = initialize_actor_critic_from_iql(self.model, online)
            if inherited["fresh"]:
                raise ValueError(
                    "frozen IQL parent leaves actor-critic tensors uninitialized: " + ", ".join(inherited["fresh"])
                )
            loaded_names = list(inherited["inherited"])
        else:
            self.model.load_state_dict(online, strict=True)
            loaded_names = sorted(online)
        self.model.to(self.device)
        self.model.eval()
        self.model.requires_grad_(False)
        self.provenance: dict[str, Any] = {
            "source_kind": "hybrid_iql_publication"
            if initialize_from_iql
            else (
                "hybrid_macro_credit_publication"
                if expected_format == HYBRID_MACRO_CREDIT_PUBLICATION
                else "hybrid_vtrace_publication"
            ),
            "source_path": str(source_path),
            "source_format": expected_format,
            "policy_version": policy_version,
            "initialization_mode": "in_memory_model_initialization" if initialize_from_iql else "frozen_publication",
            "exact_resume": False,
            "source_model_config": source_model_config,
            "model_config": model_config.as_mapping(),
            "source_reward_identity": source_reward_identity,
            "reward_identity": reward_identity,
            "loaded_tensor_count": len(loaded_names),
            "loaded_tensor_names": loaded_names,
            "ignored_source_tensor_names": sorted(set(online) - set(loaded_names)),
            "fresh_tensor_names": [],
            "training_state_loaded": False,
            "device": str(self.device),
        }

    def initial_state(self) -> Tensor:
        """Return a fresh run/combat memory without any training state."""
        with torch.inference_mode():
            return self.model.initial_state(1, device=self.device).detach()

    def forward(
        self,
        snapshot: EncodedDecisionSnapshot,
        hidden: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        """Score the encoded semantic candidates, not native dispatch indices."""
        encoded = collate_encoded_snapshots(
            (snapshot,),
            expected_config=self.encoding_config,
            expected_fingerprint=snapshot.encoding_fingerprint,
            device=self.device,
        )
        if hidden is None:
            hidden = self.initial_state()
        with torch.inference_mode():
            output = self.model(encoded, hidden, validate=False)
            logits = output.policy_logits[0].detach()
            next_hidden = output.recurrent_state.detach()
        return logits, next_hidden


__all__ = ["FrozenMacroRootPolicy"]
