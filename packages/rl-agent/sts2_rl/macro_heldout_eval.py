"""Odd-seed zero-exploration held-out evaluation for Durable Afterstate Macro Credit v2.

Evaluates a published ``publication.pt`` strictly on the held-out odd seed
namespace (``held_out_evaluation_seeds``, ``seed % 2 == 1``) with greedy
hierarchical action selection (``epsilon = 0.0``), emitting a canonical JSON
summary suitable for ``heldout_evaluations.jsonl`` and live dashboard sync.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Final, Literal
from uuid import uuid4

import numpy as np
import torch

from sts2_baseline import ActFrontloadedDeathRewardCalculator
from sts2_env.headless_sim_bridge_client import resolve_headless_sim_exe
from sts2_rl.artifacts import artifact_root, resolve_external_input_path
from sts2_rl.backends import HeadlessBackend
from sts2_rl.contracts import EnvironmentResult, ResetRequest
from sts2_rl.encoding import EncodedDecisionSnapshot, GroundedObservationEncoder
from sts2_rl.joint.authority import HybridCollectionAuthority
from sts2_rl.joint.parallel_collection import (
    BatchedPolicyInferenceServer,
    RemoteBatchedFrozenPolicy,
)
from sts2_rl.macro_root_diagnostic import (
    FrozenPolicy,
    dispatch_request,
    validate_course,
)
from sts2_rl.macro_root_policy import load_frozen_macro_teacher
from sts2_rl.models.hybrid_actor_critic import hierarchical_greedy_action_index
from sts2_rl.semantics.forward import ForwardDecision, SemanticCandidate
from sts2_rl.simulator_identity import verify_headless_simulator
from sts2_rl.train_macro_credit import explicit_device, load_macro_training_config
from sts2_rl.training.config import TrainingConfig
from sts2_rl.training.sdpa import configure_rocm_sdpa_backend
from sts2_rl.training.seeding import EVALUATION_SEED_PARITY, held_out_evaluation_seeds

MACRO_HELDOUT_EVAL_VERSION: Final = "sts2-macro-heldout-eval-v1"


class _HeldOutGreedyAuthority(HybridCollectionAuthority):
    """Zero-exploration greedy authority tracking macro discipline on held-out runs."""

    def __init__(self, policy: FrozenPolicy, seed: int) -> None:
        super().__init__(
            forward_q=policy.forward,
            initial_state=policy.initial_state,
            epsilon=0.0,
            seed=seed,
            evaluation_ownership=True,
        )
        self._policy_ref = policy
        self.reward_decisions = 0
        self.reward_skips = 0
        self.rest_decisions = 0
        self.rest_heals = 0
        self.shop_decisions = 0
        self.shop_removes = 0

    def _semantic_choice(
        self,
        *,
        candidates: Sequence[SemanticCandidate],
        snapshot: EncodedDecisionSnapshot,
    ) -> int:
        selected = super()._semantic_choice(candidates=candidates, snapshot=snapshot)
        if self.forward_retries:
            raise FloatingPointError("held-out evaluation refuses non-finite policy forwards")
        assert self.last_q_values is not None
        if getattr(self._policy_ref, "hierarchical_branch_greedy", False):
            selected = hierarchical_greedy_action_index(
                torch.tensor(self.last_q_values.tolist(), dtype=torch.float32),
                torch.tensor(snapshot.candidates.ids[:, 13].tolist(), dtype=torch.int64),
                torch.tensor(snapshot.action_mask.tolist(), dtype=torch.bool),
                domain_id=snapshot.domain_id,
            )
        return selected

    def _record_semantic_choice(
        self,
        *,
        observation: Mapping[str, Any],
        decision: ForwardDecision,
        candidates: Sequence[SemanticCandidate],
        selected_index: int,
        control_domain: Literal["macro", "combat"],
    ) -> None:
        del observation
        if control_domain != "macro" or not (0 <= selected_index < len(candidates)):
            return
        branches = {cand.branch for cand in candidates}
        chosen_branch = candidates[selected_index].branch
        if decision.surface == "reward" and {"take", "skip"}.issubset(branches):
            self.reward_decisions += 1
            if chosen_branch == "skip":
                self.reward_skips += 1
        elif decision.surface == "rest" and len(branches) >= 2:
            self.rest_decisions += 1
            if chosen_branch == "rest":
                self.rest_heals += 1
        elif decision.surface == "shop" and len(branches) >= 2:
            self.shop_decisions += 1
            if chosen_branch == "remove_card":
                self.shop_removes += 1

    def _open_transition(self, **kwargs: Any) -> None:
        del kwargs

    def _close_open(self, **kwargs: Any) -> None:
        del kwargs
        self._open = None


def _extract_floor(state: EnvironmentResult) -> int:
    run = state.observation.get("run")
    raw_floor = run.get("floor") if isinstance(run, Mapping) else None
    if isinstance(raw_floor, int) and not isinstance(raw_floor, bool) and raw_floor >= 0:
        return raw_floor
    return 0


def _extract_deck_size(state: EnvironmentResult) -> int | None:
    player = state.observation.get("player")
    if not isinstance(player, Mapping):
        return None
    deck = player.get("deck")
    if isinstance(deck, list):
        return sum(int(c.get("quantity", 1)) if isinstance(c, Mapping) else 1 for c in deck)
    return None


def _is_victory(state: EnvironmentResult) -> bool:
    if state.transition is not None:
        facts = getattr(state.transition, "facts", None)
        if isinstance(facts, Mapping):
            run_res = str(facts.get("run_result") or "").strip().lower()
            if run_res in {"victory", "run_won", "win", "act3_victory"}:
                return True
        if bool(getattr(state.transition, "run_won", False)):
            return True
    reason = str(state.terminal_reason or "").strip().lower()
    if reason in {"victory", "run_won", "win", "act3_victory"}:
        return True
    run = state.observation.get("run")
    if isinstance(run, Mapping) and bool(run.get("victory") or run.get("is_victory")):
        return True
    return False


def _run_one_heldout_seed(
    *,
    seed: int,
    policy: FrozenPolicy,
    training_config: TrainingConfig,
    sim_exe: Path,
    max_native_steps: int,
    timeout_seconds: float,
    shared_backend: HeadlessBackend | None = None,
) -> dict[str, Any]:
    if seed % 2 != EVALUATION_SEED_PARITY:
        raise ValueError(f"held-out evaluation refuses even training seed {seed}")
    revival_budget = validate_course(training_config)
    started = time.monotonic()

    def remaining_ms(cap_ms: int) -> int:
        rem = timeout_seconds - (time.monotonic() - started)
        if rem <= 0.0:
            raise TimeoutError(f"held-out seed {seed} exceeded {timeout_seconds}s")
        return max(1, min(cap_ms, int(rem * 1000)))

    owns_backend = shared_backend is None
    backend = shared_backend if shared_backend is not None else HeadlessBackend(exe_path=str(sim_exe))
    try:
        authority = _HeldOutGreedyAuthority(policy, seed)
        encoder = GroundedObservationEncoder(training_config.model.to_encoding_config())
        reward_calc = ActFrontloadedDeathRewardCalculator(
            objective=training_config.curriculum.reward_objective,
            discount=training_config.optimization.discount,
            maximum_episode_steps=training_config.environment.max_episode_steps,
            revival_budget=revival_budget,
        )
        reward_calc.begin_episode()
        revision = backend.get_state()["state_version"]
        state = backend.reset(
            ResetRequest(
                request_id=str(uuid4()),
                session_id=backend.session_id,
                scenario="full-run",
                expected_state_version=revision,
                character=training_config.environment.character,
                seed=seed,
                force_fresh=True,
                defensive_buffs=False,
                additional_relics=None,
                training_revival_budget=revival_budget,
                timeout_ms=remaining_ms(45_000),
            )
        )
        authority.begin_episode(state.episode_id)
        max_floor = _extract_floor(state)
        last_deck_size = _extract_deck_size(state)
        total_return = 0.0
        revivals_used = 0
        player_hp_lost = 0.0
        native_steps = 0
        stalled = False

        while not (state.terminated or state.truncated):
            if native_steps >= max_native_steps:
                stalled = True
                break
            cur_deck = _extract_deck_size(state)
            if cur_deck is not None:
                last_deck_size = cur_deck
            encoded = encoder.encode(state.observation, state.legal_actions)
            valid = encoded.snapshot.action_mask.astype(bool, copy=False)
            semantic_actions = tuple(
                {
                    "surface_kind": "strict_action_group",
                    "prototype": group.prototype,
                    "multiplicity": group.reference.multiplicity,
                    "equivalence_fingerprint": group.reference.equivalence_fingerprint,
                    "enabled": group.reference.enabled,
                }
                for group in encoded.semantic_groups
            )
            selected = authority.choose(
                observation=state.observation,
                semantic_actions=semantic_actions,
                snapshot=encoded.snapshot,
                valid=valid,
            )
            if selected is None:
                indices = np.flatnonzero(valid)
                if len(indices) != 1:
                    raise RuntimeError("held-out authority declined a non-forced surface")
                selected = int(indices[0])
            req = dispatch_request(
                backend,
                state,
                encoded,
                selected,
                timeout_ms=remaining_ms(20_000),
            )
            prev_state = state
            state = backend.step(req)
            native_steps += 1
            max_floor = max(max_floor, _extract_floor(state))
            post_deck = _extract_deck_size(state)
            if post_deck is not None:
                last_deck_size = post_deck
            tr_obs = state.observation.get("_training")
            if isinstance(tr_obs, Mapping):
                raw_rev = tr_obs.get("revivals_used")
                if isinstance(raw_rev, int | float) and not isinstance(raw_rev, bool):
                    revivals_used = max(revivals_used, int(raw_rev))
                raw_hp = tr_obs.get("player_hp_lost")
                if isinstance(raw_hp, int | float) and not isinstance(raw_hp, bool):
                    player_hp_lost = max(player_hp_lost, float(raw_hp))
            elif state.transition is not None and not state.truncated:
                revivals_used += int(getattr(state.transition, "revivals_used_delta", 0))
                player_hp_lost += float(getattr(state.transition, "player_hp_lost", 0.0))
            if state.transition is not None and not state.truncated:
                step_r = reward_calc.evaluate(
                    before=prev_state,
                    after=state,
                    deadlock=False,
                    horizon_exhausted=False,
                )
                total_return += float(step_r.reward)
                if step_r.task_terminal:
                    break

        won = (not stalled) and (not state.truncated) and _is_victory(state)
        return {
            "seed": seed,
            "won": won,
            "act1_cleared": bool(won or max_floor >= 17),
            "act3_reached": bool(won or max_floor >= 34),
            "zero_revival_victory": bool(won and revivals_used == 0),
            "max_floor": max_floor,
            "revivals_used": revivals_used,
            "player_hp_lost": player_hp_lost,
            "observed_return": total_return,
            "final_deck_size": last_deck_size,
            "native_steps": native_steps,
            "stalled": stalled or bool(state.truncated and not state.terminated),
            "reward_decisions": authority.reward_decisions,
            "reward_skips": authority.reward_skips,
            "rest_decisions": authority.rest_decisions,
            "rest_heals": authority.rest_heals,
            "shop_decisions": authority.shop_decisions,
            "shop_removes": authority.shop_removes,
        }
    finally:
        if owns_backend:
            backend.close()


def _heldout_worker_entry(
    worker_id: int,
    task_queue: Any,
    result_queue: Any,
    infer_request_queue: Any,
    infer_response_conn: Any,
    sim_exe_str: str,
    artifact_root_str: str,
    training_config: TrainingConfig,
    max_native_steps: int,
    timeout_seconds: float,
    recurrent_hidden_dim: int,
    hierarchical_branch_greedy: bool,
) -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["STS2_ARTIFACT_ROOT"] = artifact_root_str
    if "tcmalloc" in os.environ.get("LD_PRELOAD", ""):
        os.environ.pop("LD_PRELOAD", None)
    torch.set_num_threads(1)

    policy = RemoteBatchedFrozenPolicy(
        worker_id=worker_id,
        request_queue=infer_request_queue,
        response_conn=infer_response_conn,
        recurrent_hidden_dim=recurrent_hidden_dim,
        hierarchical_branch_greedy=hierarchical_branch_greedy,
    )
    sim_path = Path(sim_exe_str)
    backend: HeadlessBackend | None = None
    try:
        while True:
            seed = task_queue.get()
            if seed is None:
                break
            if backend is None:
                backend = HeadlessBackend(exe_path=str(sim_path))
            try:
                row = _run_one_heldout_seed(
                    seed=int(seed),
                    policy=policy,
                    training_config=training_config,
                    sim_exe=sim_path,
                    max_native_steps=max_native_steps,
                    timeout_seconds=timeout_seconds,
                    shared_backend=backend,
                )
                result_queue.put(("ok", row))
            except BaseException as exc:
                try:
                    backend.close()
                except BaseException:
                    pass
                backend = None
                result_queue.put(("err", f"{type(exc).__name__}: {exc}"))
    finally:
        if backend is not None:
            try:
                backend.close()
            except BaseException:
                pass


def evaluate_heldout_publication(
    *,
    publication_path: Path,
    config_path: Path,
    policy_version: int,
    episodes: int = 48,
    workers: int = 32,
    device_spec: str = "cuda",
    sim_exe_override: Path | None = None,
    overrides: Sequence[str] = (),
) -> dict[str, Any]:
    if episodes < 1:
        raise ValueError("held-out episodes must be >= 1")
    if workers < 1:
        raise ValueError("held-out workers must be >= 1")
    macro_cfg = load_macro_training_config(config_path, overrides)
    revival_budget = validate_course(macro_cfg.training)
    root = artifact_root()
    pub_resolved = resolve_external_input_path(publication_path, root=root)
    pub_sha256 = hashlib.sha256(pub_resolved.read_bytes()).hexdigest()
    device = explicit_device(device_spec)
    configure_rocm_sdpa_backend(macro_cfg.training.runtime.rocm_sdpa_backend, devices=(device,))
    simulator = verify_headless_simulator(
        resolve_headless_sim_exe(sim_exe_override or macro_cfg.training.environment.sim_exe_path)
    )
    loaded = load_frozen_macro_teacher(
        pub_resolved,
        expected_encoding=macro_cfg.training.model.to_encoding_config(),
        device=device,
        rocm_sdpa_backend=macro_cfg.training.runtime.rocm_sdpa_backend,
        expected_sha256=pub_sha256,
    )
    seeds = held_out_evaluation_seeds(macro_cfg.training.runtime.seed, episodes)
    eff_workers = min(episodes, workers)
    started = time.monotonic()
    rows: list[dict[str, Any]] = []
    if eff_workers == 1:
        with BatchedPolicyInferenceServer(
            loaded.model,
            macro_cfg.training.model.to_encoding_config(),
            device,
            max_batch_size=8,
            allow_tf32=macro_cfg.learner.allow_tf32,
            enable_bf16_amp=macro_cfg.learner.enable_bf16_amp,
        ) as server:
            shared = HeadlessBackend(exe_path=str(simulator.executable))
            try:
                for s in seeds:
                    rows.append(
                        _run_one_heldout_seed(
                            seed=s,
                            policy=server,
                            training_config=macro_cfg.training,
                            sim_exe=simulator.executable,
                            max_native_steps=macro_cfg.branches.max_native_steps,
                            timeout_seconds=macro_cfg.branches.timeout_seconds,
                            shared_backend=shared,
                        )
                    )
            finally:
                shared.close()
    else:
        mp_ctx = get_context("spawn")
        task_queue = mp_ctx.Queue()
        result_queue = mp_ctx.Queue()
        infer_request_queue = mp_ctx.Queue()
        recv_conns: list[Any] = []
        send_conns: list[Any] = []
        for _ in range(eff_workers):
            r_conn, s_conn = mp_ctx.Pipe(duplex=False)
            recv_conns.append(r_conn)
            send_conns.append(s_conn)
        for s in seeds:
            task_queue.put(int(s))
        for _ in range(eff_workers):
            task_queue.put(None)

        procs: list[Any] = []
        with BatchedPolicyInferenceServer(
            loaded.model,
            macro_cfg.training.model.to_encoding_config(),
            device,
            worker_send_conns=send_conns,
            mp_request_queue=infer_request_queue,
            max_batch_size=max(8, eff_workers),
            allow_tf32=macro_cfg.learner.allow_tf32,
            enable_bf16_amp=macro_cfg.learner.enable_bf16_amp,
        ) as server:
            try:
                for wid in range(eff_workers):
                    p = mp_ctx.Process(
                        target=_heldout_worker_entry,
                        args=(
                            wid,
                            task_queue,
                            result_queue,
                            infer_request_queue,
                            recv_conns[wid],
                            str(simulator.executable),
                            str(root),
                            macro_cfg.training,
                            macro_cfg.branches.max_native_steps,
                            macro_cfg.branches.timeout_seconds,
                            server.recurrent_hidden_dim,
                            server.hierarchical_branch_greedy,
                        ),
                        daemon=True,
                    )
                    p.start()
                    recv_conns[wid].close()
                    procs.append(p)
                for _ in range(len(seeds)):
                    status_tag, payload = result_queue.get(
                        timeout=max(120.0, float(macro_cfg.branches.timeout_seconds) + 60.0)
                    )
                    if status_tag != "ok":
                        raise RuntimeError(f"held-out worker failed: {payload}")
                    rows.append(payload)
            finally:
                for p in procs:
                    p.join(timeout=5.0)
                    if p.is_alive():
                        p.terminate()
                        p.join(timeout=2.0)
                for s_conn in send_conns:
                    try:
                        s_conn.close()
                    except OSError:
                        pass

    rows.sort(key=lambda item: int(item["seed"]))
    n = len(rows)
    wins = sum(1 for r in rows if r["won"])
    a1 = sum(1 for r in rows if r["act1_cleared"])
    a3 = sum(1 for r in rows if r["act3_reached"])
    z_wins = sum(1 for r in rows if r["zero_revival_victory"])
    stalls = sum(1 for r in rows if r["stalled"])
    deck_vals = [int(r["final_deck_size"]) for r in rows if r["final_deck_size"] is not None]
    tot_rew_dec = sum(int(r["reward_decisions"]) for r in rows)
    tot_rew_skp = sum(int(r["reward_skips"]) for r in rows)
    tot_rst_dec = sum(int(r["rest_decisions"]) for r in rows)
    tot_rst_hl = sum(int(r["rest_heals"]) for r in rows)
    tot_shp_dec = sum(int(r["shop_decisions"]) for r in rows)
    tot_shp_rm = sum(int(r["shop_removes"]) for r in rows)
    now_ts = time.time()

    return {
        "version": MACRO_HELDOUT_EVAL_VERSION,
        "evaluated_at_utc": datetime.now(UTC).isoformat(),
        "unix_s": now_ts,
        "evaluation_gate": int(policy_version),
        "policy_version": int(policy_version),
        "data_partition": "validation",
        "seed_parity": "odd_held_out",
        "revival_budget": int(revival_budget),
        "publication_sha256": pub_sha256,
        "episodes": n,
        "successful_run_count": wins,
        "run_win_rate": wins / n,
        "act1_clear_count": a1,
        "act1_clear_rate": a1 / n,
        "act3_reach_count": a3,
        "act3_reach_rate": a3 / n,
        "zero_revival_victories": z_wins,
        "zero_revival_act1_successes": z_wins,
        "revival_free_act1_clear_rate": z_wins / n,
        "mean_revivals_used": sum(float(r["revivals_used"]) for r in rows) / n,
        "mean_player_hp_lost": sum(float(r["player_hp_lost"]) for r in rows) / n,
        "mean_max_floor": sum(float(r["max_floor"]) for r in rows) / n,
        "maximum_floor": max((int(r["max_floor"]) for r in rows), default=0),
        "mean_observed_return": sum(float(r["observed_return"]) for r in rows) / n,
        "mean_final_deck_size": (sum(deck_vals) / len(deck_vals)) if deck_vals else None,
        "skip_rate": (tot_rew_skp / tot_rew_dec) if tot_rew_dec > 0 else None,
        "rest_rate": (tot_rst_hl / tot_rst_dec) if tot_rst_dec > 0 else None,
        "shop_remove_rate": (tot_shp_rm / tot_shp_dec) if tot_shp_dec > 0 else None,
        "deadlock_count": stalls,
        "deadlock_rate": stalls / n,
        "elapsed_seconds": time.monotonic() - started,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publication", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--policy-version", type=int, required=True)
    parser.add_argument("--episodes", type=int, default=48)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sim-exe", type=Path)
    parser.add_argument("--set", action="append", default=[], dest="overrides", metavar="KEY=TOML_VALUE")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    summary = evaluate_heldout_publication(
        publication_path=args.publication,
        config_path=args.config,
        policy_version=args.policy_version,
        episodes=args.episodes,
        workers=args.workers,
        device_spec=args.device,
        sim_exe_override=args.sim_exe,
        overrides=args.overrides,
    )
    print(json.dumps(summary, sort_keys=True, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
