"""Shared token-memory schema, layout, and checkpoint-safe helpers."""

from __future__ import annotations

import os
import time
from typing import NamedTuple

import torch
from torch.utils.checkpoint import checkpoint as activation_checkpoint

from sts2_env.attention_blocks import CrossAttentionBlock, TransformerEncoderBlock
from sts2_env.objective_heads import HEAD_HP_PRESERVATION, HEAD_SURVIVAL
from sts2_env.observation_v2 import NUM_DOMAINS
from sts2_env.observation_v3 import (
    ENTITY_HASH_BUCKETS,
    MAX_ORDER_ID,
    MAX_OWNER_ID,
    MAX_ROLE_ID,
    MAX_ZONE_ID,
    NUM_TOKEN_TYPES,
    POWER_ID_BUCKETS,
    TOKEN_ROLE_TO_ID,
    TOKEN_TYPE_TO_ID,
    TOKEN_ZONE_TO_ID,
)

_POWER_SLOT_ROLE_ID = TOKEN_ROLE_TO_ID.get("POWER_SLOT", 0)
_HISTORY_ROLE_ID = TOKEN_ROLE_TO_ID.get("HISTORY", 0)
_RELATION_BIAS_KW = {
    "num_token_types": NUM_TOKEN_TYPES,
    "max_owner_id": MAX_OWNER_ID,
    "max_role_id": MAX_ROLE_ID,
    "max_zone_id": MAX_ZONE_ID,
    "max_order_id": MAX_ORDER_ID,
    "power_bucket_count": POWER_ID_BUCKETS,
    "power_slot_role_id": _POWER_SLOT_ROLE_ID,
    "history_card_bucket_count": ENTITY_HASH_BUCKETS,
    "history_role_id": _HISTORY_ROLE_ID,
}

WORLD_BANK_NAMES = ("runtime", "support", "enemy", "build", "route", "powers", "history")
_WORLD_BANK_ROLE_NAMES = {
    "runtime": (
        "WORLD",
        "PLAYER_STATE",
        "RESOURCE",
        "THREAT",
        "OBJECTIVE",
        "RUN_MEMORY",
        "HAND_CARD",
        "DRAW_PILE",
        "DISCARD_PILE",
        "EXHAUST_PILE",
        "PLAY_PILE",
        "PILE_LINK",
        "CYCLE_PLAN",
        "ENERGY_BUDGET",
    ),
    "support": ("RELIC_SUPPORT", "POTION_SUPPORT", "SUPPORT_GRAPH"),
    "enemy": ("ENEMY_CORE", "ENEMY_INTENT", "ENEMY_POWER", "ENEMY_TRAIT", "ENEMY_REACTION"),
    "build": ("DECK_CARD", "BUILD_STATE", "DECK_SYNERGY", "REWARD_OPTION", "SHOP_OPTION", "UPGRADE_OPTION"),
    "route": ("ROUTE_SUMMARY", "ROUTE_NODE", "ROUTE_RISK", "ROUTE_VALUE"),
    "powers": ("POWER_SLOT", "CARD_KEYWORD"),
    "history": ("HISTORY",),
}
_WORLD_BANK_ZONE_NAMES = {
    "runtime": ("WORLD", "PLAYER", "HAND", "DRAW", "DISCARD", "EXHAUST", "PLAY"),
    "support": ("RELIC", "POTION"),
    "enemy": ("ENEMY",),
    "build": ("DECK", "REWARD", "SHOP", "UPGRADE"),
    "route": ("ROUTE",),
    "powers": (),
    "history": (),
}
WORLD_BANK_ROLE_IDS = {
    bank_name: tuple(
        TOKEN_ROLE_TO_ID[role_name]
        for role_name in role_names
        if role_name in TOKEN_ROLE_TO_ID
    )
    for bank_name, role_names in _WORLD_BANK_ROLE_NAMES.items()
}
WORLD_BANK_ZONE_IDS = {
    bank_name: tuple(
        TOKEN_ZONE_TO_ID[zone_name]
        for zone_name in zone_names
        if zone_name in TOKEN_ZONE_TO_ID
    )
    for bank_name, zone_names in _WORLD_BANK_ZONE_NAMES.items()
}
GLOBAL_MEMORY_BANK_NAME = "global"
MEMORY_BANK_NAMES = (*WORLD_BANK_NAMES, GLOBAL_MEMORY_BANK_NAME)
GLOBAL_MEMORY_BANK_INDEX = len(WORLD_BANK_NAMES)
MEMORY_SLOT_LAYOUT_LEGACY = "legacy"
MEMORY_SLOT_LAYOUT_QUOTA_V1 = "quota_v1"
MEMORY_SLOT_LAYOUT_PASS_LARGE_V1 = "pass_large_v1"
VALID_MEMORY_SLOT_LAYOUTS = (
    MEMORY_SLOT_LAYOUT_LEGACY,
    MEMORY_SLOT_LAYOUT_QUOTA_V1,
    MEMORY_SLOT_LAYOUT_PASS_LARGE_V1,
)
RISK_OBJECTIVE_HEAD_INDICES = (HEAD_SURVIVAL, HEAD_HP_PRESERVATION)


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return value.strip().lower() not in {"0", "false", "no", "off", ""}


def _parse_length_buckets(raw: str | None, default: tuple[int, ...]) -> tuple[int, ...]:
    if raw is None or not raw.strip():
        return default
    buckets: list[int] = []
    for part in raw.replace(";", ",").split(","):
        item = part.strip()
        if not item:
            continue
        try:
            value = int(item)
        except ValueError:
            continue
        if value > 0:
            buckets.append(value)
    if not buckets:
        return default
    return tuple(sorted(set(buckets)))


def _bucket_effective_len(
    active_len: int,
    *,
    full_len: int,
    env_name: str,
    default_buckets: tuple[int, ...],
) -> int:
    """Round dynamic token packing to stable bucket sizes.

    Exact suffix trimming saves memory but creates many attention shapes during
    live combat.  On ROCm that can trigger repeated backend/kernel
    specialization and rare hard stalls.  Bucketing keeps the memory win while
    constraining attention shapes to a small, predictable set.
    """

    active_len = max(int(active_len), 1)
    full_len = max(int(full_len), active_len)
    if not _env_flag("STS2_TOKEN_LENGTH_BUCKETING", True):
        return min(active_len, full_len)
    for bucket in _parse_length_buckets(os.environ.get(env_name), default_buckets):
        if active_len <= bucket:
            return min(max(bucket, active_len), full_len)
    return full_len


def _token_encoder_trace_enabled() -> bool:
    return _env_flag("STS2_TOKEN_ENCODER_TRACE", False)


def _token_encoder_trace(label: str) -> None:
    if not _token_encoder_trace_enabled():
        return
    if torch.cuda.is_available() and _env_flag("STS2_TOKEN_ENCODER_TRACE_SYNC", True):
        # Synchronize before printing so a missing "after <stage>" line points
        # directly at the preceding queued kernel/stage.
        torch.cuda.synchronize()
    print(f"[token_encoder] {time.strftime('%H:%M:%S')} {label}", flush=True)


def _should_activation_checkpoint(enabled: bool, *args: object) -> bool:
    return bool(enabled) and torch.is_grad_enabled() and any(
        isinstance(arg, torch.Tensor) and bool(arg.requires_grad)
        for arg in args
    )


def _run_encoder_block(
    enabled: bool,
    block: TransformerEncoderBlock,
    x: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
    attn_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    if _should_activation_checkpoint(enabled, x):
        def _forward(x_: torch.Tensor) -> torch.Tensor:
            return block(x_, mask=mask, attn_bias=attn_bias)

        return activation_checkpoint(_forward, x, use_reentrant=False)
    return block(x, mask=mask, attn_bias=attn_bias)


def _run_cross_block(
    enabled: bool,
    block: CrossAttentionBlock,
    query: torch.Tensor,
    memory: torch.Tensor,
    *,
    query_mask: torch.Tensor | None = None,
    memory_mask: torch.Tensor | None = None,
    attn_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    if _should_activation_checkpoint(enabled, query, memory):
        def _forward(query_: torch.Tensor, memory_: torch.Tensor) -> torch.Tensor:
            return block(
                query_,
                memory_,
                query_mask=query_mask,
                memory_mask=memory_mask,
                attn_bias=attn_bias,
            )

        return activation_checkpoint(_forward, query, memory, use_reentrant=False)
    return block(query, memory, query_mask=query_mask, memory_mask=memory_mask, attn_bias=attn_bias)


def _run_function_with_checkpoint(enabled: bool, fn, *args):
    if _should_activation_checkpoint(enabled, *args):
        return activation_checkpoint(fn, *args, use_reentrant=False)
    return fn(*args)


_QUOTA_V1_SLOT_BANK_NAMES = (
    # One pass over every explicit bank plus a global slot keeps the 8-slot
    # behavior readable and backward-like.
    "runtime",
    "support",
    "enemy",
    "build",
    "route",
    "powers",
    "history",
    "global",
    # Extra capacity is deliberately spent on the long-horizon banks that were
    # previously compressed into one 128-d vector each.
    "build",
    "route",
    "history",
    "enemy",
    "build",
    "route",
    "build",
    "global",
)

_PASS_LARGE_V1_SLOT_BANK_NAMES = (
    # STS2-Pass-Large-v1 24-slot target:
    # runtime=4, enemy=3, build=4, route=3, support=2,
    # powers=2, history=2, global=4.
    #
    # The ordering keeps the first two passes balanced across all banks, then
    # spends the final slots on runtime/build/route/enemy/global capacity.
    "runtime",
    "support",
    "enemy",
    "build",
    "route",
    "powers",
    "history",
    "global",
    "runtime",
    "support",
    "enemy",
    "build",
    "route",
    "powers",
    "history",
    "global",
    "runtime",
    "build",
    "route",
    "global",
    "runtime",
    "enemy",
    "build",
    "global",
)


def normalize_memory_slot_layout(layout: str | None) -> str:
    normalized = str(layout or MEMORY_SLOT_LAYOUT_LEGACY).strip().lower()
    if normalized not in VALID_MEMORY_SLOT_LAYOUTS:
        valid = ", ".join(VALID_MEMORY_SLOT_LAYOUTS)
        raise ValueError(f"Unknown token memory slot layout {layout!r}; expected one of: {valid}.")
    return normalized


def build_memory_slot_bank_ids(num_memory_slots: int, layout: str = MEMORY_SLOT_LAYOUT_LEGACY) -> list[int]:
    """Assign each latent slot to a persistent bank identity.

    ``legacy`` preserves the original checkpoint-compatible behavior:
      - one reserved slot per explicit world bank
      - any extra slots become global/planning slots

    When the caller requests fewer slots than explicit banks, we compress the
    coverage by evenly subsampling bank identities. The default configuration
    should keep at least one slot per bank.

    ``quota_v1`` is the first medium-capacity long-horizon layout.  It keeps the same
    first 8 slots as the legacy/default profile, then spends additional slots on
    the high-entropy STS2 banks:

    - build: deck curve, card valuation, archetype, reward/shop/smith state
    - route: future rest/shop/elite/event timing and risk
    - history: run trajectory, recent losses, choices already made
    - enemy: multi-enemy/boss mechanics context

    At 16 slots the intended quota is:
      runtime=1, support=1, enemy=2, build=4, route=3,
      powers=1, history=2, global=2.

    ``pass_large_v1`` is the STS2 pass-target layout.  At 24 slots it gives:
      runtime=4, enemy=3, build=4, route=3, support=2,
      powers=2, history=2, global=4.
    """
    slot_count = max(int(num_memory_slots), 1)
    normalized_layout = normalize_memory_slot_layout(layout)
    if normalized_layout in {MEMORY_SLOT_LAYOUT_QUOTA_V1, MEMORY_SLOT_LAYOUT_PASS_LARGE_V1}:
        name_to_index = {
            **{bank_name: bank_index for bank_index, bank_name in enumerate(WORLD_BANK_NAMES)},
            GLOBAL_MEMORY_BANK_NAME: GLOBAL_MEMORY_BANK_INDEX,
        }
        template = (
            _PASS_LARGE_V1_SLOT_BANK_NAMES
            if normalized_layout == MEMORY_SLOT_LAYOUT_PASS_LARGE_V1
            else _QUOTA_V1_SLOT_BANK_NAMES
        )
        if slot_count <= len(template):
            slot_names = template[:slot_count]
        else:
            slot_names = template + (GLOBAL_MEMORY_BANK_NAME,) * (slot_count - len(template))
        return [name_to_index[slot_name] for slot_name in slot_names]

    bank_count = len(WORLD_BANK_NAMES)
    if slot_count >= bank_count:
        return list(range(bank_count)) + [GLOBAL_MEMORY_BANK_INDEX] * (slot_count - bank_count)
    if slot_count == 1:
        return [GLOBAL_MEMORY_BANK_INDEX]
    positions = torch.linspace(0, bank_count - 1, steps=slot_count).round().long().tolist()
    return [int(position) for position in positions]


def build_zone_transport_prior() -> torch.Tensor:
    """Construct a zone-to-zone transport prior for copy / migration logits."""

    num_zones = MAX_ZONE_ID + 1
    prior = torch.zeros((num_zones, num_zones), dtype=torch.float32)
    prior.fill_diagonal_(0.2)

    def add(target_zone: str, source_zone: str, value: float) -> None:
        target_id = TOKEN_ZONE_TO_ID.get(target_zone)
        source_id = TOKEN_ZONE_TO_ID.get(source_zone)
        if target_id is None or source_id is None:
            return
        prior[target_id, source_id] += float(value)

    for stable_zone in (
        "WORLD",
        "PLAYER",
        "ENEMY",
        "RELIC",
        "POTION",
        "DECK",
        "ROUTE",
        "SHOP",
        "REWARD",
        "UPGRADE",
        "HISTORY",
    ):
        add(stable_zone, stable_zone, 0.55)

    add("HAND", "DRAW", 1.25)
    add("HAND", "HAND", 0.35)
    add("DISCARD", "HAND", 1.10)
    add("EXHAUST", "HAND", 1.00)
    add("PLAY", "HAND", 0.85)
    add("DISCARD", "PLAY", 0.75)
    add("EXHAUST", "PLAY", 0.65)
    add("PLAY", "PLAY", 0.35)
    add("DRAW", "DISCARD", 0.60)
    add("DISCARD", "DISCARD", 0.40)
    add("DRAW", "DRAW", 0.35)
    add("EXHAUST", "EXHAUST", 0.65)
    add("HAND", "DISCARD", 0.20)
    add("DRAW", "HAND", 0.15)
    return prior


class TokenMemoryEncoderOutput(NamedTuple):
    hidden_state: torch.Tensor
    action_embeddings: torch.Tensor
    world_pool: torch.Tensor
    world_bank_summaries: torch.Tensor | None
    world_bank_occupancy: torch.Tensor | None
    world_bank_token_presence: torch.Tensor | None
    world_bank_token_distribution: torch.Tensor | None
    world_bank_token_slot_states: torch.Tensor | None
    world_bank_token_slot_mask: torch.Tensor | None
    world_bank_token_slot_type_ids: torch.Tensor | None
    world_bank_token_slot_zone_ids: torch.Tensor | None
    world_bank_token_slot_entity_ids: torch.Tensor | None
    world_bank_token_slot_order_ids: torch.Tensor | None
    decision_domain: torch.Tensor | None


def infer_token_decision_domain(
    obs: dict[str, torch.Tensor],
    *,
    device: torch.device | None = None,
) -> torch.Tensor | None:
    """Infer dense-v2 style decision_domain one-hot from token query roles."""
    query_roles = obs.get("candidate_query_role_ids")
    if query_roles is None:
        return None
    role_ids = torch.as_tensor(query_roles, dtype=torch.long, device=device)
    if role_ids.ndim == 1:
        role_ids = role_ids.unsqueeze(0)

    action_mask = obs.get("action_mask")
    if action_mask is not None:
        action_mask = torch.as_tensor(action_mask, dtype=torch.bool, device=role_ids.device)
        if action_mask.ndim == 1:
            action_mask = action_mask.unsqueeze(0)
    else:
        action_mask = torch.ones_like(role_ids, dtype=torch.bool)

    inferred = torch.zeros((role_ids.shape[0], NUM_DOMAINS), dtype=torch.float32, device=role_ids.device)
    role_to_domain = {
        TOKEN_ROLE_TO_ID.get("QUERY_COMBAT", -1): 0,
        TOKEN_ROLE_TO_ID.get("QUERY_BUILD", -1): 1,
        TOKEN_ROLE_TO_ID.get("QUERY_SELECTION", -1): 1,
        TOKEN_ROLE_TO_ID.get("QUERY_ROUTE", -1): 2,
    }
    for role_id, domain_idx in role_to_domain.items():
        if role_id < 0:
            continue
        inferred[:, domain_idx] = ((role_ids == role_id) & action_mask).any(dim=1).float()

    unresolved = inferred.sum(dim=1, keepdim=True) <= 0.0
    if unresolved.any():
        world_type_ids = obs.get("world_token_type_ids")
        world_token_mask = obs.get("world_token_mask")
        if world_type_ids is not None:
            world_type_ids = torch.as_tensor(world_type_ids, dtype=torch.long, device=role_ids.device)
            if world_type_ids.ndim == 1:
                world_type_ids = world_type_ids.unsqueeze(0)
            if world_token_mask is not None:
                world_token_mask = torch.as_tensor(world_token_mask, dtype=torch.bool, device=role_ids.device)
                if world_token_mask.ndim == 1:
                    world_token_mask = world_token_mask.unsqueeze(0)
            else:
                world_token_mask = torch.ones_like(world_type_ids, dtype=torch.bool)
            cls_type_to_domain = {
                TOKEN_TYPE_TO_ID.get("CLS_COMBAT", -1): 0,
                TOKEN_TYPE_TO_ID.get("CLS_BUILD", -1): 1,
                TOKEN_TYPE_TO_ID.get("CLS_ROUTE", -1): 2,
            }
            fallback = torch.zeros_like(inferred)
            for type_id, domain_idx in cls_type_to_domain.items():
                if type_id < 0:
                    continue
                fallback[:, domain_idx] = ((world_type_ids == type_id) & world_token_mask).any(dim=1).float()
            inferred = torch.where(unresolved, fallback, inferred)
            unresolved = inferred.sum(dim=1, keepdim=True) <= 0.0
    if unresolved.any():
        inferred = inferred.clone()
        inferred[unresolved.squeeze(-1), 1] = 1.0
    return inferred
