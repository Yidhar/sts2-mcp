"""Route heuristic telemetry for self-play episodes.

The route heuristic has two separate concerns:

* ``muzero.route_heuristics`` owns pure scoring / root-prior helpers.
* this training mixin owns dry-run record capture and TensorBoard emission.

Keeping both out of ``self_play.py`` prevents the rollout loop from becoming a
route-planning monolith again.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from muzero.route_heuristics.root_bias import compute_route_heuristic_bias_vector


class RouteHeuristicTelemetryMixin:
    """Helpers used by ``SelfPlayMixin`` for route-prior experiments."""

    def _compute_route_heuristic_bias_vector(
        self,
        *,
        decision_domain: str,
        legal_actions: list[dict[str, Any]] | list[Any] | None,
    ) -> np.ndarray | None:
        """Build the optional Phase-3 route root-bias vector for MCTS."""

        if str(decision_domain or "").strip().lower() != "route":
            return None
        if float(getattr(self, "route_heuristic_bias_weight", 0.0) or 0.0) <= 0.0:
            return None
        env_unwrapped = getattr(self.env, "unwrapped", self.env)
        raw_obs = getattr(env_unwrapped, "_last_obs_raw", None)
        full_legal_actions = getattr(env_unwrapped, "_legal_actions", None)
        return compute_route_heuristic_bias_vector(
            raw_obs=raw_obs if isinstance(raw_obs, dict) else None,
            legal_actions=full_legal_actions if isinstance(full_legal_actions, list) else None,
            fallback_legal_actions=legal_actions if isinstance(legal_actions, list) else None,
            bias_weight=float(getattr(self, "route_heuristic_bias_weight", 0.0) or 0.0),
        )

    def _record_route_heuristic_dry_run(
        self,
        *,
        records: list[dict[str, Any]],
        error_count: int,
        action_idx: int,
        legal_actions: list[dict[str, Any]] | list[Any] | None,
    ) -> int:
        """Append one route dry-run record and return the updated error count."""

        try:
            from sts2_env.deck_quality import deck_quality_v2_from_obs
            from sts2_env.route_heuristic import (
                rank_legal_route_actions,
                count_non_empty_potions,
            )

            env_unwrap = getattr(self.env, "unwrapped", self.env)
            raw_for_heuristic = getattr(env_unwrap, "_last_obs_raw", None)
            # ``legal_actions`` from info["legal_actions_compact"] is stripped
            # of route_summary by compact_action_signature.  The route
            # heuristic needs the full bridge-side actions (which DO carry
            # route_summary and route_nodes), so read them from the env wrapper
            # directly.
            full_legal_actions = getattr(env_unwrap, "_legal_actions", None) or legal_actions
            deck_quality_now = deck_quality_v2_from_obs(raw_for_heuristic)
            player = (
                raw_for_heuristic.get("player")
                if isinstance(raw_for_heuristic, dict) and isinstance(raw_for_heuristic.get("player"), dict)
                else {}
            )
            run = (
                raw_for_heuristic.get("run")
                if isinstance(raw_for_heuristic, dict) and isinstance(raw_for_heuristic.get("run"), dict)
                else {}
            )
            cur_hp = float(player.get("hp") or 0.0)
            cur_max_hp = float(player.get("max_hp") or 0.0)
            cur_gold = float(player.get("gold") or 0.0)
            cur_potions = count_non_empty_potions(player.get("potions"))
            cur_floor = int(run.get("floor") or 0)
            hp_ratio_now = (cur_hp / cur_max_hp) if cur_max_hp > 0 else 1.0

            ranked = rank_legal_route_actions(
                legal_actions=full_legal_actions,
                deck_quality=deck_quality_now,
                hp=cur_hp,
                max_hp=cur_max_hp,
                gold=cur_gold,
                potion_count=cur_potions,
                floor=cur_floor,
            )
            map_indices = [i for i, b in enumerate(ranked) if b is not None]
            map_count = len(map_indices)

            # Always record the decision, even when there are zero map
            # candidates — that lets us compute available_rate honestly.
            sel_action_full = (
                full_legal_actions[action_idx]
                if isinstance(full_legal_actions, list) and 0 <= action_idx < len(full_legal_actions)
                else None
            )
            record: dict[str, Any] = {
                "route_decision": True,
                "map_candidate_count": map_count,
                "available": map_count > 0,
                "multi_candidate": map_count >= 2,
                "summary_used_candidate_count": 0,
                "selected_idx": int(action_idx),
                "selected_is_map": (
                    isinstance(sel_action_full, dict)
                    and str(sel_action_full.get("kind") or "").lower() == "map"
                ),
                "hp_ratio": float(hp_ratio_now),
                "gold": float(cur_gold),
            }

            if map_count > 0:
                # Candidate-level "available" flags (any-of across all legal
                # map actions).  Available rate is not selected rate.
                any_rest_before_elite = False
                any_shop_with_gold = False
                any_safe_route = False  # any candidate with elite_risk <= 1.0
                summary_used = 0
                for i in map_indices:
                    b = ranked[i]
                    if b.get("summary_used"):
                        summary_used += 1
                    if b.get("rest_before_elite_available", 0.0) > 0.5:
                        any_rest_before_elite = True
                    sub_summary = full_legal_actions[i].get("route_summary") if isinstance(full_legal_actions[i], dict) else {}
                    if isinstance(sub_summary, dict):
                        if float(sub_summary.get("count_shop") or 0) > 0 and cur_gold >= 75:
                            any_shop_with_gold = True
                    if float(b.get("elite_risk_factor", 0.0)) <= 1.0:
                        any_safe_route = True
                record["summary_used_candidate_count"] = summary_used
                record["candidate_rest_before_elite_available"] = any_rest_before_elite
                record["candidate_shop_with_gold_available"] = any_shop_with_gold
                record["candidate_safe_route_available"] = any_safe_route

                # Ranking.
                scores = [ranked[i]["score"] for i in map_indices]
                order = sorted(range(len(scores)), key=lambda k: -scores[k])
                top1_idx = map_indices[order[0]]
                top2_idx = map_indices[order[1]] if len(order) >= 2 else None
                best_score = scores[order[0]]
                record["top1_idx"] = int(top1_idx)
                record["top2_idx"] = int(top2_idx) if top2_idx is not None else None
                record["best_score"] = float(best_score)

                # Selected-level breakdown.
                if action_idx in map_indices:
                    sel_b = ranked[action_idx]
                    sel_summary = (
                        full_legal_actions[action_idx].get("route_summary")
                        if isinstance(full_legal_actions[action_idx], dict) else None
                    ) or {}
                    sel_score = float(sel_b["score"])
                    record["selected_score"] = sel_score
                    record["best_minus_selected"] = float(best_score - sel_score)
                    record["selected_summary_used"] = bool(sel_b.get("summary_used"))
                    record["selected_unsafe_elite_penalty"] = float(sel_b["unsafe_elite_penalty"])
                    record["selected_forced_elite_penalty"] = float(sel_b["forced_elite_penalty"])
                    record["selected_no_rest_before_elite_penalty"] = float(sel_b["no_rest_before_elite_penalty"])
                    record["selected_branch_value"] = float(sel_b["branch_value"])
                    record["selected_forced_elite_count"] = float(sel_b.get("forced_elite_count", 0.0))
                    record["selected_immediate_elite_count"] = float(sel_b.get("immediate_elite_count", 0.0))
                    record["selected_optional_elite_count"] = float(sel_b.get("optional_elite_count", 0.0))
                    best_b = ranked[top1_idx]
                    record["best_unsafe_elite_penalty"] = float(best_b["unsafe_elite_penalty"])
                    record["best_branch_value"] = float(best_b["branch_value"])
                    record["best_forced_elite_count"] = float(best_b.get("forced_elite_count", 0.0))
                    record["best_summary_used"] = bool(best_b.get("summary_used"))
                    record["selected_low_hp_flag"] = float(sel_b["low_hp_flag"])
                    record["selected_low_hp_elite_flag"] = float(sel_b["low_hp_elite_flag"])
                    record["selected_rest_before_elite_chosen"] = bool(sel_b.get("rest_before_elite_available", 0.0) > 0.5)
                    sel_has_elite = float(sel_summary.get("count_elite") or 0) > 0
                    sel_has_rest = float(sel_summary.get("count_rest_site") or 0) > 0
                    sel_has_shop = float(sel_summary.get("count_shop") or 0) > 0
                    record["selected_has_elite"] = bool(sel_has_elite)
                    record["selected_has_rest"] = bool(sel_has_rest)
                    record["selected_has_shop"] = bool(sel_has_shop)
                    record["selected_unsafe_elite"] = bool(sel_b["unsafe_elite_penalty"] > 1.0)
                    record["selected_forced_elite"] = bool(sel_b["forced_elite_penalty"] > 0.0)
                    record["selected_no_rest_before_elite"] = bool(sel_b["no_rest_before_elite_penalty"] > 0.0)
                    record["selected_shop_with_gold"] = bool(sel_has_shop and cur_gold >= 75)
                    record["selected_top1"] = bool(action_idx == top1_idx)
                    record["selected_top2"] = bool(top2_idx is not None and action_idx in (top1_idx, top2_idx))

            records.append(record)
            return int(error_count)
        except Exception as exc:
            error_count = int(error_count) + 1
            if error_count <= 5:
                print(
                    f"[route_heuristic] dry-run record exception "
                    f"#{error_count}: {type(exc).__name__}: {exc}",
                    flush=True,
                )
            return error_count

    def _emit_route_heuristic_dry_run_metrics(
        self,
        *,
        records: list[dict[str, Any]],
        error_count: int,
    ) -> None:
        """Emit route-heuristic dry-run metrics for the completed episode."""

        try:
            from sts2_env.route_heuristic import REST_URGENCY_HP_RATIO

            n_records = len(records)
            self.writer.add_scalar(
                "route_heuristic/route_decision_count",
                float(n_records),
                self.episode_count,
            )
            self.writer.add_scalar(
                "route_heuristic/error_count",
                float(error_count),
                self.episode_count,
            )
            if n_records <= 0:
                return

            # All-decision rates.
            available = [r for r in records if r.get("available")]
            n_available = len(available)
            multi = [r for r in records if r.get("multi_candidate")]
            n_multi = len(multi)
            self.writer.add_scalar(
                "route_heuristic/available_rate",
                n_available / n_records,
                self.episode_count,
            )
            self.writer.add_scalar(
                "route_heuristic/multi_candidate_rate",
                n_multi / n_records,
                self.episode_count,
            )
            cand_count_mean = sum(r["map_candidate_count"] for r in records) / n_records
            self.writer.add_scalar(
                "route_heuristic/candidate_count_mean",
                float(cand_count_mean),
                self.episode_count,
            )
            if n_available > 0:
                summary_used_total = sum(r["map_candidate_count"] for r in available)
                summary_used_used = sum(r["summary_used_candidate_count"] for r in available)
                self.writer.add_scalar(
                    "route_heuristic/summary_used_rate",
                    (summary_used_used / summary_used_total) if summary_used_total else 0.0,
                    self.episode_count,
                )
                rest_before_elite_avail_n = sum(
                    1 for r in available if r.get("candidate_rest_before_elite_available")
                )
                shop_with_gold_avail_n = sum(
                    1 for r in available if r.get("candidate_shop_with_gold_available")
                )
                self.writer.add_scalar(
                    "route_heuristic/rest_before_elite_available_rate",
                    rest_before_elite_avail_n / n_available,
                    self.episode_count,
                )
                self.writer.add_scalar(
                    "route_heuristic/shop_with_gold_available_rate",
                    shop_with_gold_avail_n / n_available,
                    self.episode_count,
                )

                # Selected-level rates (use available denominator since we
                # want to know "of decisions where we had a choice").
                selected_recs = [r for r in available if r.get("selected_is_map") and "selected_score" in r]
                n_sel = len(selected_recs)
                if n_sel > 0:
                    self.writer.add_scalar(
                        "route_heuristic/selected_score_mean",
                        sum(r["selected_score"] for r in selected_recs) / n_sel,
                        self.episode_count,
                    )
                    self.writer.add_scalar(
                        "route_heuristic/best_score_mean",
                        sum(r["best_score"] for r in selected_recs) / n_sel,
                        self.episode_count,
                    )
                    self.writer.add_scalar(
                        "route_heuristic/best_minus_selected_mean",
                        sum(r["best_minus_selected"] for r in selected_recs) / n_sel,
                        self.episode_count,
                    )
                    self.writer.add_scalar(
                        "route_heuristic/top1_selected_rate",
                        sum(1 for r in selected_recs if r.get("selected_top1")) / n_sel,
                        self.episode_count,
                    )
                    self.writer.add_scalar(
                        "route_heuristic/top2_selected_rate",
                        sum(1 for r in selected_recs if r.get("selected_top2")) / n_sel,
                        self.episode_count,
                    )
                    self.writer.add_scalar(
                        "route_heuristic/unsafe_elite_selected_rate",
                        sum(1 for r in selected_recs if r.get("selected_unsafe_elite")) / n_sel,
                        self.episode_count,
                    )
                    self.writer.add_scalar(
                        "route_heuristic/selected_unsafe_elite_penalty_mean",
                        sum(r.get("selected_unsafe_elite_penalty", 0.0) for r in selected_recs) / n_sel,
                        self.episode_count,
                    )
                    self.writer.add_scalar(
                        "route_heuristic/best_unsafe_elite_penalty_mean",
                        sum(r.get("best_unsafe_elite_penalty", 0.0) for r in selected_recs) / n_sel,
                        self.episode_count,
                    )
                    self.writer.add_scalar(
                        "route_heuristic/selected_branch_value_mean",
                        sum(r.get("selected_branch_value", 0.0) for r in selected_recs) / n_sel,
                        self.episode_count,
                    )
                    self.writer.add_scalar(
                        "route_heuristic/best_branch_value_mean",
                        sum(r.get("best_branch_value", 0.0) for r in selected_recs) / n_sel,
                        self.episode_count,
                    )
                    self.writer.add_scalar(
                        "route_heuristic/selected_forced_elite_count_mean",
                        sum(r.get("selected_forced_elite_count", 0.0) for r in selected_recs) / n_sel,
                        self.episode_count,
                    )
                    self.writer.add_scalar(
                        "route_heuristic/best_forced_elite_count_mean",
                        sum(r.get("best_forced_elite_count", 0.0) for r in selected_recs) / n_sel,
                        self.episode_count,
                    )
                    self.writer.add_scalar(
                        "route_heuristic/selected_optional_elite_count_mean",
                        sum(r.get("selected_optional_elite_count", 0.0) for r in selected_recs) / n_sel,
                        self.episode_count,
                    )
                    self.writer.add_scalar(
                        "route_heuristic/forced_elite_selected_rate",
                        sum(1 for r in selected_recs if r.get("selected_forced_elite")) / n_sel,
                        self.episode_count,
                    )
                    self.writer.add_scalar(
                        "route_heuristic/no_rest_before_elite_selected_rate",
                        sum(1 for r in selected_recs if r.get("selected_no_rest_before_elite")) / n_sel,
                        self.episode_count,
                    )
                    self.writer.add_scalar(
                        "route_heuristic/low_hp_elite_selected_rate",
                        sum(1 for r in selected_recs if r.get("selected_low_hp_elite_flag", 0.0) > 0.5) / n_sel,
                        self.episode_count,
                    )
                    self.writer.add_scalar(
                        "route_heuristic/rest_before_elite_selected_rate",
                        sum(1 for r in selected_recs if r.get("selected_rest_before_elite_chosen")) / n_sel,
                        self.episode_count,
                    )
                    self.writer.add_scalar(
                        "route_heuristic/shop_selected_high_gold_rate",
                        sum(1 for r in selected_recs if r.get("selected_shop_with_gold")) / n_sel,
                        self.episode_count,
                    )
                    self.writer.add_scalar(
                        "route_heuristic/selected_summary_used_rate",
                        sum(1 for r in selected_recs if r.get("selected_summary_used")) / n_sel,
                        self.episode_count,
                    )
                    self.writer.add_scalar(
                        "route_heuristic/best_summary_used_rate",
                        sum(1 for r in selected_recs if r.get("best_summary_used")) / n_sel,
                        self.episode_count,
                    )

                    # Rest-when-low-HP (denominator: low-HP decisions).
                    low_hp_decisions = [
                        r for r in selected_recs if r["hp_ratio"] < REST_URGENCY_HP_RATIO
                    ]
                    if low_hp_decisions:
                        rest_when_low = sum(
                            1 for r in low_hp_decisions if r.get("selected_has_rest")
                        )
                        for tag in (
                            "route_heuristic/rest_selected_low_hp_rate",
                            "route_heuristic/rest_selected_when_low_hp_rate",
                        ):
                            self.writer.add_scalar(
                                tag,
                                rest_when_low / len(low_hp_decisions),
                                self.episode_count,
                            )

            # Multi-candidate variants: single-candidate forks always tie
            # top1/top2/best_minus_selected, so this is the actual route
            # mistake measure.
            if n_multi > 0:
                multi_sel = [r for r in multi if r.get("selected_is_map") and "selected_score" in r]
                if multi_sel:
                    n_multi_sel = len(multi_sel)
                    self.writer.add_scalar(
                        "route_heuristic/top1_selected_rate_multi",
                        sum(1 for r in multi_sel if r.get("selected_top1")) / n_multi_sel,
                        self.episode_count,
                    )
                    self.writer.add_scalar(
                        "route_heuristic/top2_selected_rate_multi",
                        sum(1 for r in multi_sel if r.get("selected_top2")) / n_multi_sel,
                        self.episode_count,
                    )
                    self.writer.add_scalar(
                        "route_heuristic/best_minus_selected_mean_multi",
                        sum(r["best_minus_selected"] for r in multi_sel) / n_multi_sel,
                        self.episode_count,
                    )
        except Exception as exc:
            print(
                f"[route_heuristic] dry-run emit exception: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
