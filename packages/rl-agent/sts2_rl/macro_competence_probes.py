"""Domain-agnostic paired macro competence probes for Durable Afterstate v2.

These read-only probes never enter learner losses, reward functions, or action
selection rules. They measure structural model competence across three axes:
1. Campfire HP sensitivity: paired 20% HP vs 85% HP on identical builds.
2. Smith vs Remove target divergence: within-branch distribution separation over
   the identical upgradable/removable deck cards.
3. Rolling episode macro discipline: factual Skip rate, Rest rate, Remove rate,
   and final deck size across recent committed cohort episodes.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, Final

import numpy as np
import torch

from sts2_rl.artifacts import artifact_root, resolve_external_input_path
from sts2_rl.encoding import GroundedObservationEncoder
from sts2_rl.encoding.snapshot import collate_encoded_snapshots, collate_encoded_worlds
from sts2_rl.joint.afterstate import synthesize_deterministic_afterstates
from sts2_rl.macro_root_policy import FrozenMacroRootPolicy
from sts2_rl.models.hybrid_actor_critic import HybridDomainActorCriticModel
from sts2_rl.semantics.forward import forward_decision
from sts2_rl.training.config import load_training_config

MACRO_COMPETENCE_PROBE_VERSION: Final = "sts2-macro-competence-probes-v1"


def default_probe_decks() -> tuple[list[dict[str, Any]], ...]:
    """Generic structural deck profiles varying only in cost/delta geometry."""
    deck_alpha = [
        {
            "id": f"probe_card_basic_a_{idx}",
            "type": "Attack",
            "rarity": "Basic",
            "cost": 1,
            "is_upgradable": True,
            "is_removable": True,
            "dynamic_vars": {"Damage": 6},
            "upgrade_preview": {
                "id": f"probe_card_basic_a_{idx}",
                "type": "Attack",
                "rarity": "Basic",
                "cost": 1,
                "upgraded": True,
                "dynamic_vars": {"Damage": 9},
            },
        }
        for idx in range(4)
    ] + [
        {
            "id": f"probe_card_basic_d_{idx}",
            "type": "Skill",
            "rarity": "Basic",
            "cost": 1,
            "is_upgradable": True,
            "is_removable": True,
            "dynamic_vars": {"Block": 5},
            "upgrade_preview": {
                "id": f"probe_card_basic_d_{idx}",
                "type": "Skill",
                "rarity": "Basic",
                "cost": 1,
                "upgraded": True,
                "dynamic_vars": {"Block": 8},
            },
        }
        for idx in range(4)
    ] + [
        {
            "id": "probe_card_cost_drop",
            "type": "Power",
            "rarity": "Uncommon",
            "cost": 2,
            "is_upgradable": True,
            "is_removable": True,
            "dynamic_vars": {"Magic": 2},
            "keywords": ["retain"],
            "upgrade_preview": {
                "id": "probe_card_cost_drop",
                "type": "Power",
                "rarity": "Uncommon",
                "cost": 1,
                "upgraded": True,
                "dynamic_vars": {"Magic": 3},
                "keywords": ["retain", "innate"],
            },
        },
        {
            "id": "probe_card_high_scale",
            "type": "Attack",
            "rarity": "Rare",
            "cost": 2,
            "is_upgradable": True,
            "is_removable": True,
            "dynamic_vars": {"Damage": 14, "Vulnerable": 2},
            "upgrade_preview": {
                "id": "probe_card_high_scale",
                "type": "Attack",
                "rarity": "Rare",
                "cost": 2,
                "upgraded": True,
                "dynamic_vars": {"Damage": 22, "Vulnerable": 3},
            },
        },
    ]

    deck_beta = [
        *copy.deepcopy(deck_alpha),
        {
            "id": "probe_card_utility_draw",
            "type": "Skill",
            "rarity": "Uncommon",
            "cost": 1,
            "is_upgradable": True,
            "is_removable": True,
            "dynamic_vars": {"Cards": 2},
            "upgrade_preview": {
                "id": "probe_card_utility_draw",
                "type": "Skill",
                "rarity": "Uncommon",
                "cost": 0,
                "upgraded": True,
                "dynamic_vars": {"Cards": 3},
            },
        },
    ]
    return (deck_alpha, deck_beta)


def _build_rest_probe_surface(
    deck: Sequence[Mapping[str, Any]],
    *,
    hp: int,
    max_hp: int = 80,
    act: int = 1,
    floor: int = 6,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    heal_amount = round(0.30 * max_hp)
    observation: dict[str, Any] = {
        "phase": "rest_site",
        "decision_domain": "rest_site",
        "run": {"active": True, "act": act, "floor": floor, "ascension": 0},
        "player": {
            "character_id": "IRONCLAD",
            "hp": hp,
            "max_hp": max_hp,
            "gold": 120,
            "open_potion_slots": 2,
            "deck": [copy.deepcopy(dict(card)) for card in deck],
            "relics": [{"id": "BURNING_BLOOD", "has_upon_pickup_effect": False}],
            "potions": [],
        },
        "rest_site": {
            "visible": True,
            "options": [
                {"index": 0, "id": "rest", "type": "heal", "heal_amount": heal_amount, "is_enabled": True},
                {"index": 1, "id": "smith", "type": "smith", "is_enabled": True},
            ],
        },
    }
    legal_actions: list[dict[str, Any]] = [
        {
            "action_id": "rest:0",
            "kind": "choose_rest_option",
            "model_action_kind": "rest_site",
            "is_enabled": True,
            "option": {"index": 0, "id": "rest", "type": "heal", "heal_amount": heal_amount},
        },
        {
            "action_id": "rest:1",
            "kind": "choose_rest_option",
            "model_action_kind": "rest_site",
            "is_enabled": True,
            "option": {"index": 1, "id": "smith", "type": "smith"},
        },
    ]
    return observation, legal_actions


def _build_shop_remove_probe_surface(
    deck: Sequence[Mapping[str, Any]],
    *,
    hp: int = 60,
    max_hp: int = 80,
    act: int = 1,
    floor: int = 6,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    observation: dict[str, Any] = {
        "phase": "shop",
        "decision_domain": "shop",
        "run": {"active": True, "act": act, "floor": floor, "ascension": 0},
        "player": {
            "character_id": "IRONCLAD",
            "hp": hp,
            "max_hp": max_hp,
            "gold": 120,
            "open_potion_slots": 2,
            "deck": [copy.deepcopy(dict(card)) for card in deck],
            "relics": [{"id": "BURNING_BLOOD", "has_upon_pickup_effect": False}],
            "potions": [],
        },
        "shop": {
            "visible": True,
            "is_open": True,
            "gold": 120,
            "items": [
                {
                    "index": 0,
                    "slot_index": 0,
                    "category": "card_removal",
                    "type": "card_removal",
                    "cost": 75,
                    "price": 75,
                    "can_afford": True,
                }
            ],
        },
    }
    legal_actions: list[dict[str, Any]] = [
        {
            "action_id": "shop:remove:0",
            "kind": "shop_purchase",
            "model_action_kind": "shop",
            "is_enabled": True,
            "item": {
                "index": 0,
                "slot_index": 0,
                "category": "card_removal",
                "type": "card_removal",
                "cost": 75,
                "price": 75,
            },
        },
        {
            "action_id": "shop:leave",
            "kind": "shop_leave",
            "model_action_kind": "shop",
            "is_enabled": True,
        },
    ]
    return observation, legal_actions


def _spearman_correlation(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0

    def average_ranks(values: Sequence[float]) -> np.ndarray:
        arr = np.asarray(values, dtype=np.float64)
        order = np.argsort(arr, kind="mergesort")
        ranks = np.empty_like(arr)
        idx = 0
        while idx < len(arr):
            end = idx + 1
            while end < len(arr) and arr[order[end]] == arr[order[idx]]:
                end += 1
            mean_rank = 0.5 * float(idx + end - 1)
            ranks[order[idx:end]] = mean_rank
            idx = end
        return ranks

    rx = average_ranks(xs)
    ry = average_ranks(ys)
    sx = float(np.std(rx))
    sy = float(np.std(ry))
    if sx <= 1e-12 or sy <= 1e-12:
        return 0.0
    corr = float(np.mean((rx - np.mean(rx)) * (ry - np.mean(ry))) / (sx * sy))
    return max(-1.0, min(1.0, corr))


def _jensen_shannon_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p_safe = np.asarray(p, dtype=np.float64)
    q_safe = np.asarray(q, dtype=np.float64)
    p_safe = p_safe / max(float(p_safe.sum()), 1e-12)
    q_safe = q_safe / max(float(q_safe.sum()), 1e-12)
    m = 0.5 * (p_safe + q_safe)
    term_p = np.where(p_safe > 0.0, p_safe * np.log(np.maximum(p_safe, 1e-30) / np.maximum(m, 1e-30)), 0.0)
    term_q = np.where(q_safe > 0.0, q_safe * np.log(np.maximum(q_safe, 1e-30) / np.maximum(m, 1e-30)), 0.0)
    return max(0.0, float(0.5 * (term_p.sum() + term_q.sum())))


@torch.no_grad()
def probe_campfire_hp_sensitivity(
    model: HybridDomainActorCriticModel,
    encoder: GroundedObservationEncoder,
    *,
    decks: Sequence[Sequence[Mapping[str, Any]]] | None = None,
    low_hp_ratio: float = 0.20,
    high_hp_ratio: float = 0.85,
    max_hp: int = 80,
) -> dict[str, Any]:
    """Compare p(Rest) and post-state V_durable advantage at low vs high HP."""
    model.eval()
    device = next(model.parameters()).device
    active_decks = tuple(default_probe_decks() if decks is None else decks)
    low_hp = max(1, round(low_hp_ratio * max_hp))
    high_hp = max(low_hp + 1, min(max_hp, round(high_hp_ratio * max_hp)))

    low_rest_probs: list[float] = []
    high_rest_probs: list[float] = []
    low_rest_advs: list[float] = []
    high_rest_advs: list[float] = []
    durable_adv_shifts: list[float] = []

    for deck in active_decks:
        pair_results: list[tuple[float, float | None]] = []
        for hp in (low_hp, high_hp):
            obs, legal = _build_rest_probe_surface(deck, hp=hp, max_hp=max_hp)
            decision = forward_decision(obs, legal, control_domain="macro")
            if decision is None:
                raise RuntimeError("campfire competence probe failed to compile rest surface")
            encoded = encoder.encode(obs, [cand.semantic_action for cand in decision.candidates])
            batch = collate_encoded_snapshots(
                (encoded.snapshot,),
                expected_config=encoded.snapshot.config,
                expected_fingerprint=encoded.snapshot.encoding_fingerprint,
                device=device,
            )
            out = model(batch, validate=False)
            mask = torch.tensor(encoded.snapshot.action_mask.tolist(), device=device, dtype=torch.bool)
            probs = torch.softmax(out.policy_logits[0, : mask.numel()].double().masked_fill(~mask, -torch.inf), dim=0)

            # Map grouped candidates back to semantic branches via representative index.
            p_rest = sum(
                float(probs[group_idx].item())
                for group_idx, action_ref in enumerate(encoded.actions)
                if decision.candidates[action_ref.representative_position].branch == "rest"
            )
            grouped_candidates = tuple(
                decision.candidates[action_ref.representative_position]
                for action_ref in encoded.actions
            )
            durable_rest_adv: float | None = None
            if model.actor_critic_config.enable_durable_value_head:
                post_worlds = synthesize_deterministic_afterstates(
                    encoder=encoder,
                    observation=obs,
                    decision=decision,
                    candidates=grouped_candidates,
                )
                if post_worlds is not None:
                    world_batch, domain_ids = collate_encoded_worlds(
                        post_worlds,
                        expected_config=post_worlds[0].config,
                        expected_fingerprint=post_worlds[0].encoding_fingerprint,
                        device=device,
                    )
                    expanded_hidden = out.recurrent_state.expand(len(post_worlds), -1).contiguous()
                    q_after, _ = model.forward_durable_value(
                        world_batch,
                        domain_ids,
                        expanded_hidden,
                        validate=False,
                    )
                    rest_vals = [
                        float(q_after[idx].item())
                        for idx, cand in enumerate(grouped_candidates)
                        if cand.branch == "rest"
                    ]
                    smith_vals = [
                        float(q_after[idx].item())
                        for idx, cand in enumerate(grouped_candidates)
                        if cand.branch == "smith"
                    ]
                    if rest_vals and smith_vals:
                        durable_rest_adv = max(rest_vals) - max(smith_vals)
            pair_results.append((p_rest, durable_rest_adv))

        (p_low, adv_low), (p_high, adv_high) = pair_results
        low_rest_probs.append(p_low)
        high_rest_probs.append(p_high)
        if adv_low is not None and adv_high is not None:
            low_rest_advs.append(adv_low)
            high_rest_advs.append(adv_high)
            durable_adv_shifts.append(adv_low - adv_high)

    mean_low = float(statistics.fmean(low_rest_probs))
    mean_high = float(statistics.fmean(high_rest_probs))
    return {
        "low_hp": low_hp,
        "high_hp": high_hp,
        "max_hp": max_hp,
        "deck_count": len(active_decks),
        "mean_p_rest_low_hp": mean_low,
        "mean_p_rest_high_hp": mean_high,
        "p_rest_low_minus_high": mean_low - mean_high,
        "mean_durable_rest_adv_low_hp": (
            float(statistics.fmean(low_rest_advs)) if low_rest_advs else None
        ),
        "mean_durable_rest_adv_high_hp": (
            float(statistics.fmean(high_rest_advs)) if high_rest_advs else None
        ),
        "durable_value_rest_advantage_shift": (
            float(statistics.fmean(durable_adv_shifts)) if durable_adv_shifts else None
        ),
    }


@torch.no_grad()
def probe_smith_vs_remove_divergence(
    model: HybridDomainActorCriticModel,
    encoder: GroundedObservationEncoder,
    *,
    decks: Sequence[Sequence[Mapping[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Compare within-branch card target distributions under Smith vs Remove."""
    model.eval()
    device = next(model.parameters()).device
    active_decks = tuple(default_probe_decks() if decks is None else decks)

    jsd_values: list[float] = []
    spearman_values: list[float] = []
    top1_mismatch_flags: list[float] = []

    for deck in active_decks:
        rest_obs, rest_legal = _build_rest_probe_surface(deck, hp=60, max_hp=80)
        shop_obs, shop_legal = _build_shop_remove_probe_surface(deck, hp=60, max_hp=80)
        rest_dec = forward_decision(rest_obs, rest_legal, control_domain="macro")
        shop_dec = forward_decision(shop_obs, shop_legal, control_domain="macro")
        if rest_dec is None or shop_dec is None:
            raise RuntimeError("smith/remove probe failed to compile surface")

        rest_enc = encoder.encode(rest_obs, [cand.semantic_action for cand in rest_dec.candidates])
        shop_enc = encoder.encode(shop_obs, [cand.semantic_action for cand in shop_dec.candidates])

        rest_batch = collate_encoded_snapshots(
            (rest_enc.snapshot,),
            expected_config=rest_enc.snapshot.config,
            expected_fingerprint=rest_enc.snapshot.encoding_fingerprint,
            device=device,
        )
        shop_batch = collate_encoded_snapshots(
            (shop_enc.snapshot,),
            expected_config=shop_enc.snapshot.config,
            expected_fingerprint=shop_enc.snapshot.encoding_fingerprint,
            device=device,
        )
        rest_out = model(rest_batch, validate=False)
        shop_out = model(shop_batch, validate=False)

        rest_probs = torch.softmax(rest_out.policy_logits[0, : rest_enc.snapshot.candidate_count].double(), dim=0)
        shop_probs = torch.softmax(shop_out.policy_logits[0, : shop_enc.snapshot.candidate_count].double(), dim=0)

        smith_scores_by_card: dict[str, float] = {}
        for group_idx, action_ref in enumerate(rest_enc.actions):
            cand = rest_dec.candidates[action_ref.representative_position]
            if cand.branch == "smith" and isinstance(cand.target, Mapping):
                card_id = str(cand.target.get("id") or "")
                if card_id:
                    smith_scores_by_card[card_id] = smith_scores_by_card.get(card_id, 0.0) + float(
                        rest_probs[group_idx].item()
                    )

        remove_scores_by_card: dict[str, float] = {}
        for group_idx, action_ref in enumerate(shop_enc.actions):
            cand = shop_dec.candidates[action_ref.representative_position]
            if cand.branch == "remove" and isinstance(cand.target, Mapping):
                card_id = str(cand.target.get("id") or "")
                if card_id:
                    remove_scores_by_card[card_id] = remove_scores_by_card.get(card_id, 0.0) + float(
                        shop_probs[group_idx].item()
                    )

        shared_cards = sorted(smith_scores_by_card.keys() & remove_scores_by_card.keys())
        if len(shared_cards) < 2:
            continue
        p_smith = np.asarray([smith_scores_by_card[cid] for cid in shared_cards], dtype=np.float64)
        p_remove = np.asarray([remove_scores_by_card[cid] for cid in shared_cards], dtype=np.float64)
        p_smith = p_smith / max(float(p_smith.sum()), 1e-12)
        p_remove = p_remove / max(float(p_remove.sum()), 1e-12)

        jsd_values.append(_jensen_shannon_divergence(p_smith, p_remove))
        spearman_values.append(_spearman_correlation(p_smith.tolist(), p_remove.tolist()))
        top_smith = shared_cards[int(np.argmax(p_smith))]
        top_remove = shared_cards[int(np.argmax(p_remove))]
        top1_mismatch_flags.append(1.0 if top_smith != top_remove else 0.0)

    return {
        "evaluated_decks": len(jsd_values),
        "mean_smith_remove_jsd": float(statistics.fmean(jsd_values)) if jsd_values else 0.0,
        "mean_smith_remove_spearman": float(statistics.fmean(spearman_values)) if spearman_values else 0.0,
        "top1_target_divergence_rate": float(statistics.fmean(top1_mismatch_flags)) if top1_mismatch_flags else 0.0,
    }


def summarize_episode_macro_discipline(episodes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize factual Skip/Rest/Remove rates and final deck sizes from episode logs."""
    deck_sizes: list[float] = []
    revivals: list[float] = []
    max_floors: list[float] = []
    afterstate_counts: list[float] = []
    victories = 0
    total_episodes = 0
    branch_totals: dict[str, int] = {}

    for ep in episodes:
        if not isinstance(ep, Mapping):
            continue
        total_episodes += 1
        if ep.get("run_result") == "victory":
            victories += 1
        if isinstance(ep.get("final_deck_size"), int) and not isinstance(ep.get("final_deck_size"), bool):
            deck_sizes.append(float(ep["final_deck_size"]))
        if isinstance(ep.get("revivals_used"), int | float) and not isinstance(ep.get("revivals_used"), bool):
            revivals.append(float(ep["revivals_used"]))
        if isinstance(ep.get("max_floor"), int | float) and not isinstance(ep.get("max_floor"), bool):
            max_floors.append(float(ep["max_floor"]))
        if isinstance(ep.get("afterstate_step_count"), int | float) and not isinstance(
            ep.get("afterstate_step_count"), bool
        ):
            afterstate_counts.append(float(ep["afterstate_step_count"]))
        raw_counts = ep.get("macro_surface_branch_counts")
        if isinstance(raw_counts, Mapping):
            for key, count in raw_counts.items():
                if isinstance(count, int) and not isinstance(count, bool):
                    branch_totals[str(key)] = branch_totals.get(str(key), 0) + count

    reward_take = branch_totals.get("reward:take", 0)
    reward_skip = branch_totals.get("reward:skip", 0)
    rest_heal = branch_totals.get("rest:rest", 0)
    rest_smith = branch_totals.get("rest:smith", 0)
    shop_remove = branch_totals.get("shop:remove", 0)
    shop_buy_card = branch_totals.get("shop:buy_card", 0)
    shop_leave = branch_totals.get("shop:leave", 0)

    return {
        "episode_count": total_episodes,
        "victory_rate": victories / max(1, total_episodes),
        "card_reward_skip_rate": reward_skip / max(1, reward_take + reward_skip),
        "campfire_rest_rate": rest_heal / max(1, rest_heal + rest_smith),
        "shop_remove_per_visit_rate": shop_remove / max(1, shop_leave),
        "shop_buy_card_per_visit_rate": shop_buy_card / max(1, shop_leave),
        "mean_final_deck_size": float(statistics.fmean(deck_sizes)) if deck_sizes else None,
        "median_final_deck_size": float(statistics.median(deck_sizes)) if deck_sizes else None,
        "mean_revivals_used": float(statistics.fmean(revivals)) if revivals else None,
        "mean_max_floor": float(statistics.fmean(max_floors)) if max_floors else None,
        "mean_afterstate_steps": float(statistics.fmean(afterstate_counts)) if afterstate_counts else None,
        "branch_totals": dict(sorted(branch_totals.items())),
    }


def run_all_competence_probes(
    model: HybridDomainActorCriticModel,
    encoder: GroundedObservationEncoder,
    *,
    recent_episodes: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    report = {
        "version": MACRO_COMPETENCE_PROBE_VERSION,
        "campfire_hp_sensitivity": probe_campfire_hp_sensitivity(model, encoder),
        "smith_vs_remove_divergence": probe_smith_vs_remove_divergence(model, encoder),
        "episode_macro_discipline": summarize_episode_macro_discipline(recent_episodes),
    }
    for key, value in report.items():
        if isinstance(value, float) and not math.isfinite(value):
            raise FloatingPointError(f"non-finite competence probe metric: {key}")
    return report


def _collect_recent_episodes(experiment_dir: Path, *, max_cohorts: int = 4) -> list[dict[str, Any]]:
    cohorts_dir = experiment_dir / "cohorts"
    if not cohorts_dir.is_dir():
        return []
    update_files = sorted(
        cohorts_dir.glob("update-*/update.json"),
        key=lambda item: item.parent.name,
    )
    episodes: list[dict[str, Any]] = []
    for update_path in update_files[-max_cohorts:]:
        raw = json.loads(update_path.read_text(encoding="utf-8"))
        for ep in raw.get("training_episodes", ()):
            if isinstance(ep, dict):
                episodes.append(ep)
    return episodes


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publication", type=Path, required=True)
    parser.add_argument("--experiment-dir", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-cohorts", type=int, default=4)
    args = parser.parse_args(argv)

    root = artifact_root()
    pub_path = resolve_external_input_path(args.publication, root=root)
    raw_pub = torch.load(pub_path, map_location="cpu", weights_only=True)
    pub_budget = 64
    if isinstance(raw_pub, Mapping):
        raw_rew = raw_pub.get("reward_identity")
        if isinstance(raw_rew, Mapping):
            raw_b = raw_rew.get("revival_budget")
            if isinstance(raw_b, int) and not isinstance(raw_b, bool) and raw_b > 0:
                pub_budget = int(raw_b)
    base_training = load_training_config(profile="preheat")
    training = replace(
        base_training,
        curriculum=replace(
            base_training.curriculum,
            mode="native-revival-preheat",
            reward_objective="run",
            revival_mechanism="engine-bailout-v1",
            revival_budget=pub_budget,
        ),
    )
    loaded = FrozenMacroRootPolicy(
        pub_path,
        training,
        args.device,
        initialize_from_macro_publication=True,
        evaluation_only_allow_zero=True,
        actor_critic_config=None,
    )
    encoder = GroundedObservationEncoder(training.model.to_encoding_config())
    recent_episodes = (
        _collect_recent_episodes(resolve_external_input_path(args.experiment_dir, root=root), max_cohorts=args.max_cohorts)
        if args.experiment_dir is not None
        else []
    )
    report = run_all_competence_probes(loaded.model, encoder, recent_episodes=recent_episodes)
    report["policy_version"] = loaded.provenance.get("policy_version")
    print(json.dumps(report, ensure_ascii=False, allow_nan=False))
    return 0


__all__ = [
    "MACRO_COMPETENCE_PROBE_VERSION",
    "default_probe_decks",
    "probe_campfire_hp_sensitivity",
    "probe_smith_vs_remove_divergence",
    "run_all_competence_probes",
    "summarize_episode_macro_discipline",
]


if __name__ == "__main__":
    raise SystemExit(main())
