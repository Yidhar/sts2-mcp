"""Argument parser construction for the MuZero training CLI.

Keep parser growth out of ``muzero.training.cli_main``.  This module should
remain a declarative list of CLI flags plus defaults; post-parse normalization
and environment construction stay in ``cli_main`` or smaller setup modules.
"""

from __future__ import annotations

import argparse

# Parser defaults come from the trainer core.  Do not import via
# ``muzero.train``; that file must remain a thin compatibility entrypoint.
# Do not add strategy/search logic here.
from muzero.training.trainer import (
    DEFAULT_COMBAT_ENCOUNTER_PRIORITY_WEIGHTS,
    DEFAULT_COMBAT_ENCOUNTER_TIER_WEIGHTS,
    DEFAULT_COMBAT_SNAPSHOT_SAMPLE_MODE,
    DEFAULT_CURATED_COMBINED_SUBSET,
    DEFAULT_RECENT_TAIL_TRACKED_ENCOUNTERS,
    DEFAULT_REPLAY_ENCOUNTER_PRIORITY_WEIGHTS,
    DEFAULT_REPLAY_ENCOUNTER_TIER_WEIGHTS,
    VALID_CURATED_COMBINED_SUBSETS,
)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for ``python -m muzero.train``."""
    parser = argparse.ArgumentParser(description="MuZero Training for STS2")

    # Core training
    parser.add_argument("--total-timesteps", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--unroll-steps", type=int, default=3)
    parser.add_argument("--num-simulations", type=int, default=50)
    parser.add_argument("--max-sampled-actions", type=int, default=32)
    parser.add_argument("--min-expanded-actions", type=int, default=4)
    parser.add_argument("--root-max-sampled-actions", type=int, default=None)
    parser.add_argument("--child-max-sampled-actions", type=int, default=8)
    parser.add_argument("--root-min-expanded-actions", type=int, default=None)
    parser.add_argument("--child-min-expanded-actions", type=int, default=4)
    parser.add_argument("--combat-num-simulations", type=int, default=16,
                        help="Per-decision MCTS simulations for combat domain self-play.")
    parser.add_argument("--build-num-simulations", type=int, default=12,
                        help="Per-decision MCTS simulations for build domain self-play (rewards/shop/rest/events).")
    parser.add_argument("--route-num-simulations", type=int, default=8,
                        help="Per-decision MCTS simulations for route domain self-play.")
    parser.add_argument("--route-heuristic-bias", type=float, default=0.0,
                        help=(
                            "Phase 3 (recovery 2026-05-09): magnitude of the route-heuristic "
                            "prior bias added to MCTS root logits on map decisions. "
                            "0.0 (default) disables — keep at 0 until Phase 2 dry-run "
                            "validates correlation. Each map action's bias = weight * "
                            "score_normalized in ``[-weight, +weight]``. Recommended ramp: "
                            "0.0 -> 0.1 -> 0.2 -> 0.3."
                        ))
    parser.add_argument("--route-safety-guard", action="store_true", default=False,
                        help=(
                            "Act1 recovery (2026-05-10): enable a narrow hard guard for route "
                            "decisions. If the selected map action is high-risk (forced/immediate/"
                            "no-rest elite or low-HP chain) and a scored low-risk alternative is "
                            "legal and positionally aligned, override to that safe alternative. "
                            "Default off; intended for recovery runs with route_heuristic_bias=0."
                        ))
    parser.add_argument("--root-progressive-widening-init", type=int, default=2,
                        help="Initially selectable root children before visit-based widening grows the frontier.")
    parser.add_argument("--child-progressive-widening-init", type=int, default=1,
                        help="Initially selectable non-root children before visit-based widening grows the frontier.")
    parser.add_argument("--root-progressive-widening-growth", type=float, default=1.0,
                        help="sqrt(visit)-scaled widening growth coefficient for the root.")
    parser.add_argument("--child-progressive-widening-growth", type=float, default=1.0,
                        help="sqrt(visit)-scaled widening growth coefficient for non-root nodes.")
    parser.add_argument("--prior-floor", type=float, default=0.0)
    parser.add_argument("--relative-prior-floor", type=float, default=0.0)
    parser.add_argument("--surface-mask-blend", type=float, default=0.15,
                        help="Blend predicted latent legal-mask into child expansion ranking.")
    parser.add_argument("--end-turn-prior-bias", type=float, default=-1.0,
                        help="Root-only logit bias for end_turn when energy remains and a playable card exists.")
    parser.add_argument("--objective-value-blend", type=float, default=0.70,
                        help="Blend planner-objective value heads into MCTS backup/search value.")
    parser.add_argument("--objective-prior-blend", type=float, default=0.30,
                        help="Scale objective-aware semantic prior bonuses at the MCTS root.")
    parser.add_argument("--semantic-switch-depth", type=int, default=2,
                        help="Depth after which MCTS switches from concrete rollout to semantic rollout.")
    parser.add_argument("--disable-semantic-rollout", action="store_true",
                        help="Ablation: keep MCTS in pure concrete rollout mode.")
    parser.add_argument("--semantic-rollout-chain-steps", type=int, default=1,
                        help="Extra semantic plies to auto-drill inside the same simulation after crossing the semantic switch.")
    parser.add_argument("--semantic-revisit-bonus", type=float, default=0.15,
                        help="Soft UCB bonus for semantic nodes so search revisits abstract branches instead of spending all budget breadth-first.")
    parser.add_argument("--combat-search-mode", type=str, default="grounded_root",
                        choices=["full", "grounded_root"],
                        help="Combat-domain planner mode. grounded_root keeps combat search on root-level real actions and blocks deeper latent rollout.")
    parser.add_argument("--disable-combat-full-root-frontier", action="store_true",
                        help="When combat grounded_root mode is active, keep root progressive widening instead of exposing the whole root frontier.")
    parser.add_argument("--disable-semantic-training", action="store_true",
                        help="Ablation: zero out semantic rollout losses during optimization.")
    parser.add_argument("--root-bias-min-scale", type=float, default=0.33,
                        help="Final scale of root prior bias after linear decay.")
    parser.add_argument("--root-bias-decay-steps", type=int, default=200000,
                        help="Training steps over which root prior bias linearly decays.")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--buffer-size", type=int, default=100_000)
    parser.add_argument("--min-buffer-size", type=int, default=500)
    parser.add_argument("--train-every", type=int, default=50,
                        help="Train every N environment steps")
    parser.add_argument("--updates-per-train", type=int, default=3,
                        help="Gradient updates per training call")
    parser.add_argument("--checkpoint-freq", type=int, default=2048)
    parser.add_argument("--checkpoint-keep-last", type=int, default=3,
                        help="Auto-prune old muzero_step_* checkpoints in the current run directory after each save; 0 disables pruning.")
    parser.add_argument("--resume-from", type=str, default=None,
                        help="Resume MuZero training from a checkpoint directory")
    parser.add_argument("--resume-without-buffer", action="store_true", default=False,
                        help="Resume weights/optimizer from checkpoint but start with an empty replay buffer.")
    parser.add_argument("--resume-without-optimizer", action="store_true", default=False,
                        help="Resume model weights only; re-initialize the optimizer (Adam moments). "
                             "Use this when the aux-target schema changed mid-run and the old momentum "
                             "is steering away from the new objective.")
    parser.add_argument("--latent-policy-distill-weight", type=float, default=0.25,
                        help="Distill latent search policy toward observation-conditioned policy.")
    parser.add_argument("--latent-policy-target-weight", type=float, default=0.5,
                        help="Direct supervised target loss on latent-policy logits.")
    parser.add_argument("--planner-q-loss-weight", type=float, default=0.75,
                        help="Loss weight for action-conditioned planner Q / lookahead value supervision.")
    parser.add_argument("--planner-objective-q-loss-weight", type=float, default=0.75,
                        help="Loss weight for multi-head planner Q decomposition (survival/hp/build/resource).")
    parser.add_argument("--objective-value-weight", type=float, default=1.0,
                        help="Loss weight for multi-head objective value prediction.")
    parser.add_argument("--objective-reward-weight", type=float, default=0.75,
                        help="Loss weight for multi-head objective reward prediction.")
    parser.add_argument("--semantic-policy-weight", type=float, default=1.0,
                        help="Loss weight for semantic rollout policy supervision.")
    parser.add_argument("--semantic-value-weight", type=float, default=1.0,
                        help="Loss weight for semantic rollout value supervision.")
    parser.add_argument("--semantic-reward-weight", type=float, default=0.75,
                        help="Loss weight for semantic rollout reward supervision.")
    parser.add_argument("--semantic-state-consistency-weight", type=float, default=2.0,
                        help="Consistency weight between semantic rollout latent and projected teacher latent.")
    parser.add_argument("--objective-diversity-weight", type=float, default=0.05,
                        help="Regularize objective heads against collapse into the same signal.")
    parser.add_argument("--semantic-policy-label-smoothing", type=float, default=0.02,
                        help="Label smoothing applied to semantic rollout policy targets.")
    parser.add_argument("--state-consistency-weight", type=float, default=2.0,
                        help="Weight for raw hidden-state consistency between dynamics rollout and representation(next_obs).")
    parser.add_argument("--jepa-next-hidden-weight", type=float, default=None,
                        help="Alias/override for --state-consistency-weight; names the JEPA next-hidden prediction loss explicitly.")
    parser.add_argument("--future-world-aux-weight", type=float, default=1.0,
                        help="Global multiplier applied to the full future token-world auxiliary supervision stack.")
    parser.add_argument("--future-bank-state-weight", type=float, default=1.0,
                        help="Weight for token-mode future world-bank consistency between latent rollout and next observation bank summaries.")
    parser.add_argument("--future-bank-delta-weight", type=float, default=0.5,
                        help="Weight for future world-bank delta prediction relative to the current latent bank reference.")
    parser.add_argument("--future-bank-occupancy-weight", type=float, default=0.25,
                        help="Weight for future world-bank occupancy prediction.")
    parser.add_argument("--future-bank-token-presence-weight", type=float, default=0.25,
                        help="Weight for future world-bank token-family presence prediction.")
    parser.add_argument("--future-bank-token-distribution-weight", type=float, default=0.2,
                        help="Weight for future world-bank token-family distribution prediction.")
    parser.add_argument("--future-bank-token-slot-state-weight", type=float, default=0.5,
                        help="Weight for future world-bank top-k token-slot state reconstruction.")
    parser.add_argument("--future-bank-token-slot-mask-weight", type=float, default=0.15,
                        help="Weight for future world-bank top-k token-slot occupancy prediction.")
    parser.add_argument("--future-bank-token-slot-type-weight", type=float, default=0.2,
                        help="Weight for future world-bank top-k token-slot type prediction.")
    parser.add_argument("--future-bank-token-slot-zone-weight", type=float, default=0.15,
                        help="Weight for future world-bank top-k token-slot zone / pile prediction.")
    parser.add_argument("--future-bank-token-slot-source-weight", type=float, default=0.2,
                        help="Weight for future world-bank token migration/copy source prediction.")
    parser.add_argument("--future-world-rollout-weight", type=float, default=0.35,
                        help="Extra multiplier for multi-step future-world rollout supervision beyond the one-step target.")
    parser.add_argument("--future-world-rollout-steps", type=int, default=2,
                        help="Number of additional future rollout horizons (beyond t+1) supervised inside training.")
    parser.add_argument("--future-world-rollout-decay", type=float, default=0.7,
                        help="Geometric decay applied across extra future rollout horizons.")
    parser.add_argument("--token-teacher-ema-decay", type=float, default=0.995,
                        help="EMA decay for the token-world target encoder used for future-world teacher targets.")
    parser.add_argument("--latent-gaussian-reg-weight", type=float, default=0.005,
                        help="Global SIGReg/JEPA-style Gaussian latent regularization weight.")
    parser.add_argument("--latent-gaussian-reg-projections", type=int, default=64,
                        help="Random projection count for Gaussian latent regularization.")
    parser.add_argument("--latent-gaussian-reg-slot-weight", type=float, default=0.25,
                        help="Extra slot-level latent Gaussian regularization weight in token_memory_v1.")
    parser.add_argument("--latent-gaussian-reg-dynamics-weight", type=float, default=0.5,
                        help="Multiplier for latent Gaussian regularization on dynamics-produced next hidden states.")
    parser.add_argument("--latent-gaussian-reg-cov-weight", type=float, default=0.05,
                        help="Off-diagonal covariance penalty weight inside latent Gaussian regularization.")
    parser.add_argument("--surprise-loss-weight", type=float, default=0.25,
                        help="Loss weight for trainable dynamics surprise / latent prediction error head.")
    parser.add_argument("--surprise-hidden-scale", type=float, default=100.0,
                        help="Scale applied to normalized JEPA hidden prediction error before surprise supervision.")
    parser.add_argument("--surprise-surface-scale", type=float, default=5.0,
                        help="Batch-level surface prediction error scale added to surprise supervision target.")
    parser.add_argument("--surprise-future-aux-scale", type=float, default=1.0,
                        help="Batch-level future-world auxiliary error scale added to surprise supervision target.")
    parser.add_argument("--surprise-target-cap", type=float, default=50.0,
                        help="Optional cap on the final surprise target after hidden/surface/future-world terms; 0 disables.")
    parser.add_argument("--surface-mask-weight", type=float, default=1.5,
                        help="Weight for next-step legal-mask prediction loss.")
    parser.add_argument("--surface-count-weight", type=float, default=0.25,
                        help="Weight for predicted legal-count calibration loss.")
    parser.add_argument("--surface-domain-weight", type=float, default=0.15,
                        help="Weight for next-step decision-domain prediction loss.")
    parser.add_argument("--surface-phase-weight", type=float, default=0.1,
                        help="Weight for next-step phase prediction loss.")
    parser.add_argument("--combat-direct-policy", action="store_true", default=False,
                        help="Search-free combat self-play: bypass combat MCTS and sample from policy logits blended with latent rollout Q/uncertainty.")
    parser.add_argument("--combat-policy-mode", type=str, default="auto", choices=["auto", "direct", "mcts"],
                        help="Combat policy selector. auto enables search-free direct policy for token combat sandbox, direct always bypasses combat MCTS, mcts forces legacy combat MCTS.")
    parser.add_argument("--planner-memory-profile", type=str, default="custom", choices=["custom", "train", "eval", "max"],
                        help=("Preset for search-free planner memory use. custom preserves explicit CLI values; "
                              "train clamps rollout to low-memory 1x1 + small buckets; "
                              "eval/max restore deeper rollout while keeping planner row chunking on by default."))
    parser.add_argument("--combat-rollout-q-blend", type=float, default=0.75,
                        help="Extra direct-policy blend weight for explicit one-step rollout Q bias in combat.")
    parser.add_argument("--combat-rollout-objective-q-blend", type=float, default=0.5,
                        help="Extra direct-policy blend weight for explicit one-step rollout objective-Q bias in combat.")
    parser.add_argument("--combat-rollout-risk-blend", type=float, default=0.35,
                        help="Extra direct-policy blend weight for explicit one-step rollout survival/HP risk bias in combat.")
    parser.add_argument("--combat-rollout-steps", type=int, default=2,
                        help="Search-free rollout depth for combat direct-policy. 1 = one-step Q, 2+ adds latent continuation.")
    parser.add_argument("--combat-rollout-beam-width", type=int, default=2,
                        help="Per-root latent beam width used by multi-step combat direct-policy rollout.")
    parser.add_argument("--combat-rollout-legal-logit-scale", type=float, default=0.75,
                        help="Scale applied to predicted next-step legal-mask logits when ranking latent continuation actions.")
    parser.add_argument("--combat-rollout-uncertainty-blend", type=float, default=0.35,
                        help="Direct-policy penalty weight for rollout surprise/uncertainty bias.")
    parser.add_argument("--combat-rollout-uncertainty-surprise-weight", type=float, default=1.0,
                        help="Internal planner uncertainty weight for trainable dynamics surprise.")
    parser.add_argument("--combat-rollout-uncertainty-surface-weight", type=float, default=0.10,
                        help="Internal planner uncertainty weight for next-surface entropy.")
    parser.add_argument("--combat-rollout-uncertainty-latent-weight", type=float, default=0.05,
                        help="Internal planner uncertainty weight for latent norm drift.")
    parser.add_argument("--combat-rollout-uncertainty-disagreement-weight", type=float, default=0.25,
                        help="Internal planner uncertainty weight for branch Q disagreement.")
    parser.add_argument("--combat-rollout-continuation-uncertainty-penalty", type=float, default=0.25,
                        help="Penalty applied while pruning latent continuation beams.")
    parser.add_argument("--settlement-weight", type=float, default=0.12,
                        help="Light episode-settlement weight, only backfilled to recent build/route decisions.")
    parser.add_argument("--settlement-decay", type=float, default=0.99,
                        help="Decay factor for backward episode settlement over recent build/route steps.")
    parser.add_argument("--settlement-max-steps", type=int, default=12,
                        help="Maximum number of recent build/route steps that receive settlement credit.")
    parser.add_argument("--boundary-replay-build-bonus", type=float, default=1.0,
                        help="Extra replay sampling bonus for build-domain starting positions.")
    parser.add_argument("--boundary-replay-route-bonus", type=float, default=1.5,
                        help="Extra replay sampling bonus for route-domain starting positions.")
    parser.add_argument("--boundary-replay-family-bonus", type=float, default=0.75,
                        help="Extra replay bonus for map/reward/shop/rest/event-style boundary actions.")
    parser.add_argument("--boundary-replay-quality-bonus", type=float, default=0.25,
                        help="Episode-level replay priority bonus for deeper / stronger full-run trajectories.")
    parser.add_argument("--wasteful-end-turn-replay-scale", type=float, default=0.35,
                        help="Replay downweight for wasteful end_turn samples that ignore playable progress.")
    parser.add_argument("--wasteful-proceed-replay-scale", type=float, default=0.55,
                        help="Replay downweight for wasteful proceed/skip samples on reward-like surfaces.")
    parser.add_argument("--replay-encounter-tier-weights", type=str, default=DEFAULT_REPLAY_ENCOUNTER_TIER_WEIGHTS,
                        help="Trajectory-level replay oversampling weights by encounter tier, e.g. weak=0.8,normal=1.1,elite=1.8,boss=1.35")
    parser.add_argument("--replay-encounter-weights", type=str, default=DEFAULT_REPLAY_ENCOUNTER_PRIORITY_WEIGHTS,
                        help="Trajectory-level replay oversampling weights for hard encounters, e.g. ENCOUNTER.X=2.5")
    # P0-2 (recovery 2026-05-06): batch-level hard tier quota — guarantees a
    # minimum normal/elite share and a maximum boss share per training batch.
    parser.add_argument("--replay-tier-quota", action="store_true", default=False,
                        help="Enable batch-level hard tier quota for replay sampling (P0-2 recovery).")
    parser.add_argument("--replay-tier-quota-targets", type=str,
                        default="boss=0.60,elite=0.25,normal=0.15,weak=0.0",
                        help="Per-tier target share (sums to 1.0); used to allocate batch slots.")
    parser.add_argument("--replay-tier-quota-min", type=str,
                        default="elite=0.18,normal=0.10",
                        help="Per-tier minimum share — sampler refuses to drop below when pool is non-empty.")
    parser.add_argument("--replay-tier-quota-max", type=str,
                        default="boss=0.65",
                        help="Per-tier maximum share — sampler refuses to exceed even if priorities dominate.")
    parser.add_argument("--recent-tail-windows", type=str, default="64,256",
                        help="Comma-separated rolling episode windows to monitor, e.g. 64,256")
    parser.add_argument("--recent-tail-tracked-encounters", type=str, default=DEFAULT_RECENT_TAIL_TRACKED_ENCOUNTERS,
                        help="Comma-separated encounter ids to expose in recent-tail monitoring.")
    parser.add_argument("--recent-tail-min-samples", type=int, default=4,
                        help="Minimum recent samples required before per-encounter recent-tail win-rate scalars are emitted.")
    parser.add_argument("--disable-trivial-build-fast-path", action="store_true",
                        help="Disable shortcutting obvious build-domain decisions like taking gold, safe potion claims, and proceed-only screens.")
    parser.add_argument("--disable-potion-reward-fast-path", action="store_true",
                        help="Disable only automatic potion reward claims in the build fast-path; gold and proceed-only shortcuts remain enabled.")
    parser.add_argument("--obs-mode", type=str, default="dense_v2",
                        choices=["dense_v2", "token_v3"],
                        help="Observation encoder mode. token_v3 enables token-world MuZero inputs.")
    parser.add_argument("--model-arch", type=str, default="dense_v1",
                        choices=["dense_v1", "token_memory_v1"],
                        help="MuZero model path. token_memory_v1 uses token-world encoder + latent memory slots.")
    parser.add_argument("--token-d-model", type=int, default=128,
                        help="Token-world backbone width for token_memory_v1.")
    parser.add_argument("--token-n-heads", type=int, default=4,
                        help="Attention heads for token_memory_v1.")
    parser.add_argument("--token-ffn-dim", type=int, default=512,
                        help="FFN width for token_memory_v1.")
    parser.add_argument("--token-world-layers", type=int, default=4,
                        help="World self-attention layers for token_memory_v1.")
    parser.add_argument("--token-local-layers", type=int, default=1,
                        help="Candidate-local self-attention layers for token_memory_v1.")
    parser.add_argument("--token-decoder-layers", type=int, default=2,
                        help="Banked candidate-to-world decoder layers for token_memory_v1.")
    parser.add_argument("--token-candidate-set-layers", type=int, default=1,
                        help="Post-bank candidate-set self-attention layers for token_memory_v1.")
    parser.add_argument("--token-memory-slots", type=int, default=8,
                        help="Latent memory slots for token_memory_v1; hidden_dim becomes token_d_model * token_memory_slots.")
    parser.add_argument("--token-memory-slot-layout", type=str, default="legacy",
                        choices=["legacy", "quota_v1", "pass_large_v1"],
                        help=(
                            "Persistent bank layout for token memory slots. legacy preserves old checkpoint behavior; "
                            "quota_v1 gives extra slots to build/route/history/enemy when token_memory_slots > 8; "
                            "pass_large_v1 is the 24-slot STS2-Pass-Large layout "
                            "(runtime=4, enemy=3, build=4, route=3, support=2, powers=2, history=2, global=4)."
                        ))
    parser.add_argument("--action-embed-dim", type=int, default=64,
                        help="Action embedding dimension used by MuZero prediction/dynamics; Pass-Large target uses 128.")
    parser.add_argument("--support-size", type=int, default=25,
                        help="Categorical scalar support half-width; number of bins is 2*support_size+1. Pass-Large target uses 31.")
    parser.add_argument("--dynamics-res-blocks", type=int, default=4,
                        help="Residual transition refinement depth for MuZero dynamics; Pass-Large target uses 6.")
    parser.add_argument("--token-bank-token-slots", type=int, default=4,
                        help="Top-k per-bank world token slots used for token_memory_v1 future-world reconstruction.")
    parser.add_argument("--token-world-bank-top-k", type=int, default=3,
                        help="Top-k world banks each candidate may attend in token_memory_v1.")
    parser.add_argument("--token-slot-source-same-bank-bias", type=float, default=0.35,
                        help="Transport prior logit bias for copying from the same world bank.")
    parser.add_argument("--token-slot-source-same-slot-bias", type=float, default=0.2,
                        help="Transport prior logit bias for copying from the same slot position inside a bank.")
    parser.add_argument("--token-slot-source-type-match-scale", type=float, default=0.5,
                        help="Transport prior scale for preferring source slots whose current type matches the predicted future type.")
    parser.add_argument("--token-slot-source-zone-transport-scale", type=float, default=0.35,
                        help="Transport prior scale for zone / pile-aware copy transitions such as draw->hand and hand->discard.")
    parser.add_argument("--token-internal-planner-blend", type=float, default=0.7,
                        help="Blend weight for the token-mode internal action-imagination planner added on top of candidate policy logits.")
    parser.add_argument("--token-internal-planner-q-blend", type=float, default=0.5,
                        help="Blend weight for planner Q / lookahead value bias injected into token-mode policy logits.")
    parser.add_argument("--token-internal-planner-objective-q-blend", type=float, default=0.35,
                        help="Blend weight for multi-objective planner Q scalarization injected into token-mode policy logits.")
    parser.add_argument("--token-internal-planner-risk-blend", type=float, default=0.25,
                        help="Blend weight for planner survival/HP risk bias injected into token-mode policy logits.")
    parser.add_argument("--token-dropout", type=float, default=0.0,
                        help="Dropout for token_memory_v1 attention blocks.")
    parser.add_argument("--action-rollout-buckets", type=str, default="8,16,32,64,96",
                        help="Comma-separated bucket sizes for search-free action_rollout_planner padding; larger counts round to multiples of the largest bucket.")
    parser.add_argument("--action-rollout-chunk-size", type=int, default=0,
                        help=("Optional max row chunk for search-free planner dynamics/value passes; "
                              "0 disables chunking in custom mode, but train/eval/max profiles default it to 16."))

    # Directories
    parser.add_argument("--log-dir", type=str, default="runs")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")

    # Environment setup
    parser.add_argument("--combat-sandbox", action="store_true", default=False)
    parser.add_argument("--encounter-pool", type=str, default=None)
    # Full-run mode seed pinning (recovery 2026-05-08): when set, each
    # SlayTheSpire2EnvV2.reset() picks a seed from the pool and passes it
    # to /env/reset so the run RNG is deterministic. Use a single-seed
    # pool to lock the agent to one map for early-curriculum convergence.
    # Requires bridge mod env_api_version >= bridge-env-v2-seed.
    parser.add_argument("--seed-pool", type=str, default=None,
                        help="Comma-separated STS2 run seeds (10 chars each, 0-9A-Z minus O/I) for full-run mode.")
    parser.add_argument("--seed-pool-file", type=str, default=None,
                        help="Path to a newline-delimited seed list. Takes precedence over --seed-pool.")
    parser.add_argument("--seed-strategy", type=str, default="round_robin",
                        choices=("round_robin", "random_per_episode"),
                        help="How to pick from --seed-pool per env.reset().")
    parser.add_argument("--combat-encounter-tiers", type=str, default=None,
                        help="Comma-separated tier filter for combat sandbox snapshots / encounters: weak,normal,elite,boss")
    parser.add_argument("--combat-snapshot-dataset", type=str, default=None)
    parser.add_argument(
        "--combat-curated-subset",
        type=str,
        default=DEFAULT_CURATED_COMBINED_SUBSET,
        choices=sorted(VALID_CURATED_COMBINED_SUBSETS),
        help=(
            "When --combat-snapshot-dataset points at a curated combat root/combined dir, "
            "resolve this subset. Default keeps all human rows plus only local runs that cleared Act 1, "
            "then drops explicit losing-room combat snapshots."
        ),
    )
    parser.add_argument("--combat-snapshot-split", type=str, default="train")
    parser.add_argument(
        "--combat-snapshot-sample-mode",
        type=str,
        default=DEFAULT_COMBAT_SNAPSHOT_SAMPLE_MODE,
        choices=["row_uniform", "encounter_balanced", "tier_weighted_encounter_balanced"],
        help="Sampling mode for combat snapshot curriculum.",
    )
    parser.add_argument(
        "--combat-tier-weights",
        type=str,
        default=DEFAULT_COMBAT_ENCOUNTER_TIER_WEIGHTS,
        help="Optional tier sampling weights, e.g. weak=0.6,normal=0.4",
    )
    parser.add_argument(
        "--combat-encounter-weights",
        type=str,
        default=DEFAULT_COMBAT_ENCOUNTER_PRIORITY_WEIGHTS,
        help="Optional encounter weights, e.g. ENCOUNTER.BOWLBUGS_WEAK=3,ENCOUNTER.TUNNELER_WEAK=3",
    )
    parser.add_argument("--character", type=str, default=None,
                        help="Character to use for training")
    parser.add_argument("--defensive-buffs", action="store_true", default=False,
                        help="Apply defensive buffs on env/reset (mainly for debugging).")
    parser.add_argument("--combat-snapshot-character", type=str, default=None)
    parser.add_argument(
        "--combat-sandbox-potions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Enable potion state/use in combat sandbox resets when snapshot or override potion data is available. "
            "Use --no-combat-sandbox-potions for ablations."
        ),
    )
    parser.add_argument("--session-file", type=str, default=None)
    parser.add_argument("--session-files", type=str, default=None)
    parser.add_argument("--n-envs", type=int, default=1)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--mixed-precision",
        type=str,
        default="auto",
        choices=["auto", "off", "fp32", "bf16", "fp16"],
        help=(
            "Training/inference autocast mode. auto uses CUDA bf16 when supported, "
            "otherwise CUDA fp16+GradScaler; CPU auto stays fp32/off."
        ),
    )
    parser.add_argument(
        "--amp-init-scale",
        type=float,
        default=65536.0,
        help="Initial GradScaler scale used for fp16 mixed precision.",
    )
    parser.add_argument(
        "--activation-checkpointing",
        type=str,
        default="auto",
        choices=["auto", "on", "off"],
        help=(
            "Activation checkpointing for token-memory MuZero. "
            "auto enables it for token_v3/token_memory_v1 training and keeps dense models off."
        ),
    )

    # Discount factors
    parser.add_argument("--discount", type=float, default=0.997)
    parser.add_argument("--n-step-return", type=int, default=10)
    return parser
