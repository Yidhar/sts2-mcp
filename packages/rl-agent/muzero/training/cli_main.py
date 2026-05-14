"""Command-line training loop for ``python -m muzero.train``.

The legacy entrypoint delegates here so parser/orchestration changes do not
continue growing ``muzero.train``.  Runtime symbols are imported from the
trainer core directly; ``muzero.train`` is now only a compatibility wrapper.
"""

from __future__ import annotations

# Import the trainer namespace explicitly from the core module; do not route
# through ``muzero.train`` or the compatibility wrapper can become monolithic
# again.
from muzero.training.trainer import *  # noqa: F401,F403
from muzero.training.trainer import _get_live_supported_encounter_ids  # noqa: F401
from muzero.training.cli_args import build_arg_parser
from muzero.training.cli_parsing import resolve_combat_sandbox_encounter_pool
from muzero.training.async_telemetry import log_async_episode_scalars
from muzero.sts2_env.planner_memory_profile import apply_planner_memory_profile_to_args


def main():
    """Main training loop."""
    parser = build_arg_parser()
    args = parser.parse_args()
    activation_checkpointing_mode = str(args.activation_checkpointing or "auto").strip().lower()
    args.activation_checkpointing_enabled = bool(
        activation_checkpointing_mode == "on"
        or (
            activation_checkpointing_mode == "auto"
            and (
                str(args.obs_mode).strip().lower() == "token_v3"
                or str(args.model_arch).strip().lower() == "token_memory_v1"
            )
        )
    )
    if args.jepa_next_hidden_weight is not None:
        args.state_consistency_weight = float(args.jepa_next_hidden_weight)
    if args.combat_policy_mode == "direct":
        args.combat_direct_policy = True
    elif args.combat_policy_mode == "mcts":
        args.combat_direct_policy = False
    elif not args.combat_direct_policy and args.combat_sandbox and args.model_arch == "token_memory_v1":
        args.combat_direct_policy = True
    apply_planner_memory_profile_to_args(args)
    args.device = resolve_torch_device(args.device)
    args.log_dir = normalize_path_str(args.log_dir) or args.log_dir
    args.checkpoint_dir = normalize_path_str(args.checkpoint_dir) or args.checkpoint_dir
    args.resume_from = normalize_path_str(args.resume_from)
    args.combat_snapshot_dataset = normalize_path_str(args.combat_snapshot_dataset)
    args.session_file = normalize_path_str(args.session_file)
    run_paths = RunPaths.from_args(args)
    run_paths.ensure_dirs()
    PolicyModulePaths.from_package_root(run_paths.package_root).ensure_package_dirs()
    args.log_dir = str(run_paths.log_dir)
    args.checkpoint_dir = str(run_paths.checkpoint_dir)
    args.resume_from = str(run_paths.resume_from) if run_paths.resume_from is not None else None
    if args.obs_mode == "token_v3" and args.model_arch != "token_memory_v1":
        raise ValueError("--obs-mode token_v3 requires --model-arch token_memory_v1.")
    if args.model_arch == "token_memory_v1" and args.obs_mode != "token_v3":
        raise ValueError("--model-arch token_memory_v1 requires --obs-mode token_v3.")

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "Requested GPU device ('cuda' / ROCm) but torch.cuda.is_available() is False. "
            "Inside WSL, make sure ROCm is installed and the WSL relay base URL is configured."
        )

    # Setup directories were created through RunPaths above.

    if args.device == "cuda":
        device_name = torch.cuda.get_device_name(0)
        print(f"[setup] Using GPU device: {device_name}")
    else:
        print(f"[setup] Using device: {args.device}")

    if running_in_wsl():
        print(f"[setup] WSL detected: distro={os.environ.get('WSL_DISTRO_NAME', 'unknown')}")
        print(f"[setup] STS2_BRIDGE_SESSION_FILE={os.environ.get('STS2_BRIDGE_SESSION_FILE', '<default>')}")
        print(f"[setup] STS2_BRIDGE_BASE_URL={os.environ.get('STS2_BRIDGE_BASE_URL', '<session.json base_url>')}")

    # Resolve session files
    session_files = resolve_training_session_files(
        n_envs=args.n_envs,
        session_file=args.session_file,
        session_files=parse_session_files(args.session_files),
    )

    combat_multi_env = bool(args.combat_sandbox and args.n_envs > 1)
    if not args.combat_sandbox and args.n_envs != 1:
        print(
            "[warning] Only single environment (--n-envs 1) is supported for full-run MuZero; "
            "extra session files will be ignored outside --combat-sandbox."
        )

    train_session_files = session_files if args.combat_sandbox else [session_files[0] if session_files else None]
    session_file = train_session_files[0] if train_session_files else None
    supported_encounter_ids: set[str] | None = None

    encounter_tiers = parse_encounter_tiers(args.combat_encounter_tiers)
    tier_weights = parse_tier_weights(args.combat_tier_weights)
    encounter_weights = parse_encounter_weights(args.combat_encounter_weights)
    replay_tier_weights = parse_tier_weights(args.replay_encounter_tier_weights)
    replay_encounter_weights = parse_encounter_weights(args.replay_encounter_weights)
    recent_tail_windows = parse_int_list(args.recent_tail_windows) or DEFAULT_RECENT_TAIL_WINDOWS
    recent_tail_tracked_encounters = parse_encounter_pool(args.recent_tail_tracked_encounters)
    # Full-run seed pool (recovery 2026-05-08): pin run RNG when --seed-pool
    # or --seed-pool-file is set so the agent trains on a fixed map curriculum.
    resolved_seed_pool: list[str] = []
    if args.seed_pool_file:
        try:
            with open(args.seed_pool_file, "r", encoding="utf-8") as fh:
                for raw_line in fh:
                    line = raw_line.strip()
                    if not line or line.startswith("#"):
                        continue
                    resolved_seed_pool.append(line)
        except Exception as exc:
            print(f"[setup] Failed to read --seed-pool-file {args.seed_pool_file}: {exc}")
    elif args.seed_pool:
        resolved_seed_pool = [
            entry.strip() for entry in args.seed_pool.split(",") if entry.strip()
        ]
    if resolved_seed_pool:
        print(
            f"[setup] Full-run seed pool: count={len(resolved_seed_pool)} "
            f"strategy={args.seed_strategy} first={resolved_seed_pool[0]}"
        )

    snapshot_pool = None
    encounter_pool: list[str] = []
    if args.combat_sandbox:
        encounter_pool = resolve_combat_sandbox_encounter_pool(
            explicit_pool=args.encounter_pool,
            combat_snapshot_dataset=args.combat_snapshot_dataset,
            encounter_tiers=encounter_tiers,
            default_pool=DEFAULT_COMBAT_SANDBOX_TRAIN_POOL,
        )

    # Setup snapshot pool if provided
    if args.combat_sandbox and args.combat_snapshot_dataset:
        print(f"[setup] Loading combat snapshot dataset from {args.combat_snapshot_dataset}...")
        supported_encounter_ids = _get_live_supported_encounter_ids(session_file)
        print(f"[setup] Live combat catalog supports {len(supported_encounter_ids)} encounters")
        snapshot_pool = CombatSnapshotPool.from_path(
            args.combat_snapshot_dataset,
            curated_subset=args.combat_curated_subset,
            split=args.combat_snapshot_split or None,
            character=args.combat_snapshot_character,
            encounter_ids=encounter_pool or None,
            encounter_tiers=encounter_tiers or None,
            sample_mode=args.combat_snapshot_sample_mode,
            tier_weights=tier_weights or None,
            encounter_weights=encounter_weights or None,
            supported_encounter_ids=supported_encounter_ids,
        )
        print(f"[setup] Loaded {len(snapshot_pool)} snapshots")
        print(f"[setup] Snapshot summary: {snapshot_pool.summary()}")

    def build_env_for_index(env_index: int) -> gym.Env:
        return build_train_env(
            env_index=env_index,
            session_file=train_session_files[env_index],
            combat_sandbox=args.combat_sandbox,
            combat_sandbox_potions=args.combat_sandbox_potions,
            character=args.character,
            defensive_buffs=args.defensive_buffs,
            encounter_id=None,
            encounter_pool=encounter_pool,
            snapshot_pool=snapshot_pool,
            reset_timeout_ms=15000 if args.combat_sandbox else 60000,
            step_timeout_ms=20000,
            obs_mode=args.obs_mode,
            seed_pool=resolved_seed_pool,
            seed_strategy=args.seed_strategy,
        )

    def close_env_safely(train_env: gym.Env | None, env_index: int) -> None:
        if train_env is None:
            return
        try:
            train_env.close()
        except Exception as exc:
            print(f"[cleanup] Failed to close env[{env_index}]: {exc}")

    print(f"[setup] Creating training environment...")
    envs = [build_env_for_index(env_index) for env_index in range(len(train_session_files))]
    env = envs[0]

    # Create network, MCTS, buffer
    print(f"[setup] Creating MuZero network...")
    network_kwargs = {
        "obs_mode": args.obs_mode,
        "model_arch": args.model_arch,
        "action_embed_dim": args.action_embed_dim,
        "support_size": args.support_size,
        "dynamics_res_blocks": args.dynamics_res_blocks,
        "token_d_model": args.token_d_model,
        "token_n_heads": args.token_n_heads,
        "token_ffn_dim": args.token_ffn_dim,
        "token_world_layers": args.token_world_layers,
        "token_local_layers": args.token_local_layers,
        "token_decoder_layers": args.token_decoder_layers,
        "token_candidate_set_layers": args.token_candidate_set_layers,
        "token_memory_slots": args.token_memory_slots,
        "token_memory_slot_layout": args.token_memory_slot_layout,
        "token_bank_token_slots": args.token_bank_token_slots,
        "token_world_bank_top_k": args.token_world_bank_top_k,
        "token_slot_source_same_bank_bias": args.token_slot_source_same_bank_bias,
        "token_slot_source_same_slot_bias": args.token_slot_source_same_slot_bias,
        "token_slot_source_type_match_scale": args.token_slot_source_type_match_scale,
        "token_slot_source_zone_transport_scale": args.token_slot_source_zone_transport_scale,
        "token_internal_planner_blend": args.token_internal_planner_blend,
        "token_internal_planner_q_blend": args.token_internal_planner_q_blend,
        "token_internal_planner_objective_q_blend": args.token_internal_planner_objective_q_blend,
        "token_internal_planner_risk_blend": args.token_internal_planner_risk_blend,
        "token_dropout": args.token_dropout,
        "action_rollout_buckets": args.action_rollout_buckets,
        "action_rollout_chunk_size": args.action_rollout_chunk_size,
        "activation_checkpointing": args.activation_checkpointing_enabled,
    }
    network = MuZeroNetwork(**network_kwargs)
    network.to(args.device)

    def build_mcts_instance() -> MCTS:
        return MCTS(
            num_simulations=args.num_simulations,
            discount=args.discount,
            max_sampled_actions=args.max_sampled_actions,
            min_expanded_actions=args.min_expanded_actions,
            root_max_sampled_actions=args.root_max_sampled_actions,
            child_max_sampled_actions=args.child_max_sampled_actions,
            root_min_expanded_actions=args.root_min_expanded_actions,
            child_min_expanded_actions=args.child_min_expanded_actions,
            prior_floor=args.prior_floor,
            relative_prior_floor=args.relative_prior_floor,
            surface_mask_blend=args.surface_mask_blend,
            end_turn_prior_bias=args.end_turn_prior_bias,
            objective_value_blend=args.objective_value_blend,
            objective_prior_blend=args.objective_prior_blend,
            semantic_switch_depth=args.semantic_switch_depth,
            enable_semantic_rollout=not args.disable_semantic_rollout,
            semantic_rollout_chain_steps=args.semantic_rollout_chain_steps,
            semantic_revisit_bonus=args.semantic_revisit_bonus,
            root_progressive_widening_init=args.root_progressive_widening_init,
            child_progressive_widening_init=args.child_progressive_widening_init,
            root_progressive_widening_growth=args.root_progressive_widening_growth,
            child_progressive_widening_growth=args.child_progressive_widening_growth,
            root_bias_min_scale=args.root_bias_min_scale,
            root_bias_decay_steps=args.root_bias_decay_steps,
            combat_search_mode=args.combat_search_mode,
            combat_full_root_frontier=not args.disable_combat_full_root_frontier,
        )

    mcts = build_mcts_instance()

    # P0-2 (recovery 2026-05-06): build a TierQuotaConfig from the
    # ``--replay-tier-quota-*`` flags when ``--replay-tier-quota`` is set.
    tier_quota_config = None
    if args.replay_tier_quota:
        from muzero.replay_scheduler import TierQuotaConfig

        def _parse_share_dict(raw: str | None) -> dict[str, float]:
            out: dict[str, float] = {}
            if not raw:
                return out
            for chunk in raw.split(","):
                entry = chunk.strip()
                if not entry or "=" not in entry:
                    continue
                key, value = entry.split("=", 1)
                try:
                    out[key.strip().lower()] = float(value)
                except ValueError:
                    continue
            return out

        tier_quota_config = TierQuotaConfig(
            targets=_parse_share_dict(args.replay_tier_quota_targets) or {
                "boss": 0.60, "elite": 0.25, "normal": 0.15, "weak": 0.0,
            },
            min_caps=_parse_share_dict(args.replay_tier_quota_min) or {
                "elite": 0.18, "normal": 0.10,
            },
            max_caps=_parse_share_dict(args.replay_tier_quota_max) or {"boss": 0.65},
        )
        print(
            f"[setup] Replay tier quota: targets={tier_quota_config.targets} "
            f"min_caps={tier_quota_config.min_caps} max_caps={tier_quota_config.max_caps}"
        )

    buffer = MuZeroReplayBuffer(
        capacity=args.buffer_size,
        boundary_build_bonus=args.boundary_replay_build_bonus,
        boundary_route_bonus=args.boundary_replay_route_bonus,
        boundary_family_bonus=args.boundary_replay_family_bonus,
        trajectory_quality_bonus=args.boundary_replay_quality_bonus,
        wasteful_end_turn_scale=args.wasteful_end_turn_replay_scale,
        wasteful_proceed_scale=args.wasteful_proceed_replay_scale,
        encounter_tier_weights=replay_tier_weights,
        encounter_priority_weights=replay_encounter_weights,
        tier_quota_config=tier_quota_config,
    )

    optimizer = optim.Adam(
        network.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    trainer = MuZeroTrainer(
        network=network,
        mcts=mcts,
        buffer=buffer,
        env=env,
        optimizer=optimizer,
        device=args.device,
        discount=args.discount,
        n_step_return=args.n_step_return,
        latent_policy_distill_weight=args.latent_policy_distill_weight,
        latent_policy_target_weight=args.latent_policy_target_weight,
        planner_q_loss_weight=args.planner_q_loss_weight,
        planner_objective_q_loss_weight=args.planner_objective_q_loss_weight,
        objective_value_weight=args.objective_value_weight,
        objective_reward_weight=args.objective_reward_weight,
        semantic_policy_weight=0.0 if args.disable_semantic_training else args.semantic_policy_weight,
        semantic_value_weight=0.0 if args.disable_semantic_training else args.semantic_value_weight,
        semantic_reward_weight=0.0 if args.disable_semantic_training else args.semantic_reward_weight,
        semantic_state_consistency_weight=0.0 if args.disable_semantic_training else args.semantic_state_consistency_weight,
        objective_diversity_weight=0.0 if args.disable_semantic_training else args.objective_diversity_weight,
        semantic_policy_label_smoothing=args.semantic_policy_label_smoothing,
        state_consistency_weight=args.state_consistency_weight,
        future_world_aux_weight=args.future_world_aux_weight,
        future_bank_state_weight=args.future_bank_state_weight,
        future_bank_delta_weight=args.future_bank_delta_weight,
        future_bank_occupancy_weight=args.future_bank_occupancy_weight,
        future_bank_token_presence_weight=args.future_bank_token_presence_weight,
        future_bank_token_distribution_weight=args.future_bank_token_distribution_weight,
        future_bank_token_slot_state_weight=args.future_bank_token_slot_state_weight,
        future_bank_token_slot_mask_weight=args.future_bank_token_slot_mask_weight,
        future_bank_token_slot_type_weight=args.future_bank_token_slot_type_weight,
        future_bank_token_slot_zone_weight=args.future_bank_token_slot_zone_weight,
        future_bank_token_slot_source_weight=args.future_bank_token_slot_source_weight,
        token_teacher_ema_decay=args.token_teacher_ema_decay,
        future_world_rollout_weight=args.future_world_rollout_weight,
        future_world_rollout_steps=args.future_world_rollout_steps,
        future_world_rollout_decay=args.future_world_rollout_decay,
        latent_gaussian_reg_weight=args.latent_gaussian_reg_weight,
        latent_gaussian_reg_projections=args.latent_gaussian_reg_projections,
        latent_gaussian_reg_slot_weight=args.latent_gaussian_reg_slot_weight,
        latent_gaussian_reg_dynamics_weight=args.latent_gaussian_reg_dynamics_weight,
        latent_gaussian_reg_cov_weight=args.latent_gaussian_reg_cov_weight,
        surprise_loss_weight=args.surprise_loss_weight,
        surprise_hidden_scale=args.surprise_hidden_scale,
        surprise_surface_scale=args.surprise_surface_scale,
        surprise_future_aux_scale=args.surprise_future_aux_scale,
        surprise_target_cap=args.surprise_target_cap,
        surface_mask_weight=args.surface_mask_weight,
        surface_count_weight=args.surface_count_weight,
        surface_domain_weight=args.surface_domain_weight,
        surface_phase_weight=args.surface_phase_weight,
        combat_direct_policy=args.combat_direct_policy,
        combat_rollout_q_blend=args.combat_rollout_q_blend,
        combat_rollout_objective_q_blend=args.combat_rollout_objective_q_blend,
        combat_rollout_risk_blend=args.combat_rollout_risk_blend,
        combat_rollout_steps=args.combat_rollout_steps,
        combat_rollout_beam_width=args.combat_rollout_beam_width,
        combat_rollout_legal_logit_scale=args.combat_rollout_legal_logit_scale,
        combat_rollout_uncertainty_blend=args.combat_rollout_uncertainty_blend,
        combat_rollout_uncertainty_surprise_weight=args.combat_rollout_uncertainty_surprise_weight,
        combat_rollout_uncertainty_surface_weight=args.combat_rollout_uncertainty_surface_weight,
        combat_rollout_uncertainty_latent_weight=args.combat_rollout_uncertainty_latent_weight,
        combat_rollout_uncertainty_disagreement_weight=args.combat_rollout_uncertainty_disagreement_weight,
        combat_rollout_continuation_uncertainty_penalty=args.combat_rollout_continuation_uncertainty_penalty,
        combat_num_simulations=args.combat_num_simulations,
        build_num_simulations=args.build_num_simulations,
        route_num_simulations=args.route_num_simulations,
        settlement_weight=args.settlement_weight,
        settlement_decay=args.settlement_decay,
        settlement_max_steps=args.settlement_max_steps,
        trivial_build_fast_path=not args.disable_trivial_build_fast_path,
        potion_reward_fast_path=not args.disable_potion_reward_fast_path,
        recent_tail_windows=recent_tail_windows,
        recent_tail_tracked_encounters=recent_tail_tracked_encounters,
        recent_tail_min_samples=args.recent_tail_min_samples,
        route_heuristic_bias=args.route_heuristic_bias,
        route_safety_guard=args.route_safety_guard,
        log_dir=args.log_dir,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_keep_last=args.checkpoint_keep_last,
        mixed_precision=args.mixed_precision,
        amp_init_scale=args.amp_init_scale,
    )

    if args.resume_from:
        print(f"[resume] Loading checkpoint from {args.resume_from}...")
        resume_metadata = load_resume_checkpoint(
            args.resume_from,
            network=network,
            optimizer=optimizer,
            buffer=buffer,
            device=args.device,
            load_buffer=not args.resume_without_buffer,
            load_optimizer=not args.resume_without_optimizer,
            token_target_encoder=trainer.token_target_encoder,
            amp_grad_scaler=trainer.amp_grad_scaler,
        )
        trainer.total_steps = int(resume_metadata.get("total_steps", 0))
        trainer.episode_count = int(resume_metadata.get("episode_count", 0))
        print(
            f"[resume] Loaded total_steps={trainer.total_steps} "
            f"episode_count={trainer.episode_count} buffer={len(buffer)}"
        )
        if args.total_timesteps <= trainer.total_steps:
            raise ValueError(
                f"--total-timesteps ({args.total_timesteps}) must be greater than resumed total_steps "
                f"({trainer.total_steps})."
            )

    mode_name = "combat_sandbox" if args.combat_sandbox else "full_run"
    async_combat_actor_learner = bool(args.combat_sandbox and len(envs) > 1)
    semantic_training_enabled = not args.disable_semantic_training and (
        args.semantic_policy_weight > 0.0
        or args.semantic_value_weight > 0.0
        or args.semantic_reward_weight > 0.0
        or args.semantic_state_consistency_weight > 0.0
        or args.objective_diversity_weight > 0.0
    )
    root_bias_enabled = (args.objective_prior_blend > 0.0) or (args.end_turn_prior_bias != 0.0)
    print(f"[setup] Starting training with {args.total_timesteps} timesteps")
    print(f"[setup] Mode: {mode_name}")
    print(
        "[setup] Semantic rollout: "
        f"search={'on' if not args.disable_semantic_rollout else 'off'} "
        f"(switch_depth={args.semantic_switch_depth}, "
        f"chain_steps={args.semantic_rollout_chain_steps}, "
        f"revisit_bonus={args.semantic_revisit_bonus:.2f}) | "
        f"training={'on' if semantic_training_enabled else 'off'}"
    )
    print(
        "[setup] Combat planner: "
        f"mode={args.combat_search_mode} "
        f"(full_root_frontier={'off' if args.disable_combat_full_root_frontier else 'on'})"
    )
    print(
        "[setup] Root prior bias: "
        f"{'on' if root_bias_enabled else 'off'} "
        f"(objective_prior_blend={args.objective_prior_blend:.2f}, "
        f"end_turn_bias={args.end_turn_prior_bias:.2f}, "
        f"min_scale={args.root_bias_min_scale:.2f}, "
        f"decay_steps={args.root_bias_decay_steps})"
    )
    print(
        "[setup] Search budgets: "
        f"combat={args.combat_num_simulations}, "
        f"build={args.build_num_simulations}, "
        f"route={args.route_num_simulations}, "
        f"fallback={args.num_simulations}"
    )
    print(
        "[setup] Network: "
        f"obs_mode={args.obs_mode} "
        f"model_arch={args.model_arch} "
        f"hidden_dim={network.hidden_dim} "
        f"action_embed_dim={network.action_embed_dim} "
        f"support_size={network.support_size} "
        f"dynamics_res_blocks={args.dynamics_res_blocks}"
    )
    if args.model_arch == "token_memory_v1":
        print(
            "[setup] Token-memory: "
            f"d_model={args.token_d_model} "
            f"slots={args.token_memory_slots} "
            f"slot_layout={args.token_memory_slot_layout} "
            f"bank_token_slots={args.token_bank_token_slots} "
            f"world_layers={args.token_world_layers} "
            f"decoder_layers={args.token_decoder_layers} "
            f"bank_top_k={args.token_world_bank_top_k} "
            f"planner_blend={args.token_internal_planner_blend:.2f} "
            f"planner_q_blend={args.token_internal_planner_q_blend:.2f} "
            f"planner_obj_q_blend={args.token_internal_planner_objective_q_blend:.2f} "
            f"planner_risk_blend={args.token_internal_planner_risk_blend:.2f} "
            f"rollout_buckets={network.action_rollout_buckets} "
            f"source_prior(bank={args.token_slot_source_same_bank_bias:.2f},"
            f"slot={args.token_slot_source_same_slot_bias:.2f},"
            f"type={args.token_slot_source_type_match_scale:.2f},"
            f"zone={args.token_slot_source_zone_transport_scale:.2f})"
        )
        print(
            "[setup] Future-world rollout: "
            f"weight={args.future_world_rollout_weight:.2f} "
            f"steps={args.future_world_rollout_steps} "
            f"decay={args.future_world_rollout_decay:.2f}"
        )
        print(
            "[setup] JEPA latent regularization: "
            f"weight={args.latent_gaussian_reg_weight:.4f} "
            f"proj={args.latent_gaussian_reg_projections} "
            f"slot={args.latent_gaussian_reg_slot_weight:.2f} "
            f"dyn={args.latent_gaussian_reg_dynamics_weight:.2f} "
            f"cov={args.latent_gaussian_reg_cov_weight:.2f} | "
            f"surprise_weight={args.surprise_loss_weight:.2f} "
            f"surprise_hidden_scale={args.surprise_hidden_scale:.1f} "
            f"surprise_surface_scale={args.surprise_surface_scale:.1f} "
            f"surprise_future_aux_scale={args.surprise_future_aux_scale:.1f} "
            f"surprise_cap={args.surprise_target_cap:.1f}"
        )
    print(
        "[setup] Replay shaping: "
        f"build_bonus={args.boundary_replay_build_bonus:.2f}, "
        f"route_bonus={args.boundary_replay_route_bonus:.2f}, "
        f"family_bonus={args.boundary_replay_family_bonus:.2f}, "
        f"quality_bonus={args.boundary_replay_quality_bonus:.2f}, "
        f"wasteful_end_turn_scale={args.wasteful_end_turn_replay_scale:.2f}, "
        f"wasteful_proceed_scale={args.wasteful_proceed_replay_scale:.2f}"
    )
    print(
        "[setup] Replay encounter oversampling: "
        f"tier_weights={dict(sorted(replay_tier_weights.items()))} | "
        f"hard_encounters={dict(list(sorted(replay_encounter_weights.items()))[:10])}"
    )
    print(
        "[setup] Recent-tail monitor: "
        f"windows={list(recent_tail_windows)} "
        f"tracked={recent_tail_tracked_encounters} "
        f"min_samples={args.recent_tail_min_samples}"
    )
    print(
        "[setup] Episode settlement: "
        f"weight={args.settlement_weight:.3f}, "
        f"decay={args.settlement_decay:.3f}, "
        f"max_steps={args.settlement_max_steps}"
    )
    print(
        "[setup] Build fast-path: "
        f"{'on' if not args.disable_trivial_build_fast_path else 'off'} "
        f"(gold=yes, safe_potion={'no' if args.disable_potion_reward_fast_path else 'yes'}, proceed_only=yes)"
    )
    print(
        "[setup] Mixed precision: "
        f"mode={trainer.mixed_precision}, "
        f"enabled={trainer.amp_enabled}, "
        f"device_type={trainer.amp_device_type}, "
        f"dtype={trainer.amp_dtype}, "
        f"scaler={trainer.amp_scaler_enabled}"
    )
    print(
        "[setup] Activation checkpointing: "
        f"cli={args.activation_checkpointing} "
        f"enabled={args.activation_checkpointing_enabled}"
    )
    print(
        "[setup] Planner memory profile: "
        f"{args.planner_memory_profile} "
        f"(steps={args.combat_rollout_steps}, "
        f"beam={args.combat_rollout_beam_width}, "
        f"buckets={args.action_rollout_buckets}, "
        f"chunk={args.action_rollout_chunk_size})"
    )
    print(
        "[setup] Allocator config: "
        f"PYTORCH_ALLOC_CONF={os.environ.get('PYTORCH_ALLOC_CONF')!r}, "
        f"PYTORCH_HIP_ALLOC_CONF={os.environ.get('PYTORCH_HIP_ALLOC_CONF')!r}"
    )
    print(
        "[setup] Combat direct policy: "
        f"{'on' if args.combat_direct_policy else 'off'} "
        f"(mode={args.combat_policy_mode}, "
        f"rollout_q={args.combat_rollout_q_blend:.2f}, "
        f"rollout_obj_q={args.combat_rollout_objective_q_blend:.2f}, "
        f"rollout_risk={args.combat_rollout_risk_blend:.2f}, "
        f"uncertainty={args.combat_rollout_uncertainty_blend:.2f}, "
        f"steps={args.combat_rollout_steps}, "
        f"beam={args.combat_rollout_beam_width}, "
        f"legal_scale={args.combat_rollout_legal_logit_scale:.2f}, "
        f"u_weights=(surprise:{args.combat_rollout_uncertainty_surprise_weight:.2f},"
        f"surface:{args.combat_rollout_uncertainty_surface_weight:.2f},"
        f"latent:{args.combat_rollout_uncertainty_latent_weight:.2f},"
        f"disagree:{args.combat_rollout_uncertainty_disagreement_weight:.2f}), "
        f"beam_uncertainty_penalty={args.combat_rollout_continuation_uncertainty_penalty:.2f})"
    )
    print(
        "[setup] Progressive widening: "
        f"root_init={args.root_progressive_widening_init}, "
        f"child_init={args.child_progressive_widening_init}, "
        f"root_growth={args.root_progressive_widening_growth:.2f}, "
        f"child_growth={args.child_progressive_widening_growth:.2f}"
    )
    print(
        "[setup] Checkpoint retention: "
        + (
            "disabled"
            if args.checkpoint_keep_last <= 0
            else f"keep last {args.checkpoint_keep_last} muzero_step_* checkpoint(s) per run"
        )
    )
    if encounter_tiers:
        print(f"[setup] Encounter tiers: {encounter_tiers}")
    if tier_weights:
        print(f"[setup] Encounter tier weights: {tier_weights}")
    if encounter_weights:
        print(f"[setup] Encounter weights: {encounter_weights}")
    if snapshot_pool is not None:
        print(f"[setup] Snapshot sample mode: {args.combat_snapshot_sample_mode}")
    if encounter_pool:
        print(f"[setup] Encounter pool: {encounter_pool}")
    if args.combat_sandbox:
        print(
            "[setup] Combat sandbox envs: "
            f"n_envs={len(envs)} "
            f"(multi_env={'on' if combat_multi_env else 'off'})"
        )
        print(
            "[setup] Combat sandbox potions: "
            f"{'on' if args.combat_sandbox_potions else 'off'}"
        )
        print(
            "[setup] Combat collector: "
            f"{'async_actor_learner' if async_combat_actor_learner else 'single_thread'}"
        )
        if len(train_session_files) > 1:
            print(f"[setup] Session files: {train_session_files}")

    # ``--train-every`` is documented as environment-step cadence, not episode cadence.
    # Use a step-based trigger so variable-length combats do not delay updates by dozens
    # of episodes and so resumed runs keep a stable optimizer rhythm.
    last_train_step = int(trainer.total_steps)
    actor_threads: list[threading.Thread] = []
    actor_stop_event = threading.Event()
    async_episode_queue: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=max(len(envs) * 2, 8))
    actor_failure_counts: list[int] = [0 for _ in envs]
    actor_completed_episodes: list[int] = [0 for _ in envs]
    latest_weight_lock = threading.Lock()
    latest_weight_state: dict[str, Any] = {
        "version": 0,
        "total_steps": int(trainer.total_steps),
        "state_dict": {},
    }

    def snapshot_network_state() -> dict[str, torch.Tensor]:
        return {
            key: value.detach().cpu().clone()
            for key, value in network.state_dict().items()
        }

    def publish_latest_weights() -> None:
        with latest_weight_lock:
            latest_weight_state["version"] = int(latest_weight_state.get("version", 0)) + 1
            latest_weight_state["total_steps"] = int(trainer.total_steps)
            latest_weight_state["state_dict"] = snapshot_network_state()

    def emit_async_env_failure(
        *,
        actor_index: int,
        failure_count: int,
        cooldown_s: float,
        message: str,
    ) -> None:
        packet = {
            "kind": "actor_recover",
            "actor_index": int(actor_index),
            "failure_count": int(failure_count),
            "cooldown_s": float(cooldown_s),
            "message": str(message),
        }
        try:
            async_episode_queue.put(packet, timeout=0.1)
        except queue.Full:
            pass

    def actor_thread_main(actor_index: int) -> None:
        actor_network = MuZeroNetwork(**network_kwargs)
        actor_network.to(args.device)
        actor_optimizer = optim.SGD(actor_network.parameters(), lr=0.0)
        actor_buffer = EpisodeCaptureBuffer()
        actor_log_dir = str(run_paths.async_actor_log_dir(actor_index))
        actor_ckpt_dir = str(run_paths.async_actor_checkpoint_dir(actor_index))
        actor_trainer = MuZeroTrainer(
            network=actor_network,
            mcts=build_mcts_instance(),
            buffer=actor_buffer,  # type: ignore[arg-type]
            env=envs[actor_index],
            optimizer=actor_optimizer,
            device=args.device,
            discount=args.discount,
            n_step_return=args.n_step_return,
            latent_policy_distill_weight=args.latent_policy_distill_weight,
            latent_policy_target_weight=args.latent_policy_target_weight,
            planner_q_loss_weight=args.planner_q_loss_weight,
            planner_objective_q_loss_weight=args.planner_objective_q_loss_weight,
            objective_value_weight=args.objective_value_weight,
            objective_reward_weight=args.objective_reward_weight,
            semantic_policy_weight=0.0 if args.disable_semantic_training else args.semantic_policy_weight,
            semantic_value_weight=0.0 if args.disable_semantic_training else args.semantic_value_weight,
            semantic_reward_weight=0.0 if args.disable_semantic_training else args.semantic_reward_weight,
            semantic_state_consistency_weight=0.0 if args.disable_semantic_training else args.semantic_state_consistency_weight,
            objective_diversity_weight=0.0 if args.disable_semantic_training else args.objective_diversity_weight,
            semantic_policy_label_smoothing=args.semantic_policy_label_smoothing,
            state_consistency_weight=args.state_consistency_weight,
            future_world_aux_weight=args.future_world_aux_weight,
            future_bank_state_weight=args.future_bank_state_weight,
            future_bank_delta_weight=args.future_bank_delta_weight,
            future_bank_occupancy_weight=args.future_bank_occupancy_weight,
            future_bank_token_presence_weight=args.future_bank_token_presence_weight,
            future_bank_token_distribution_weight=args.future_bank_token_distribution_weight,
            future_bank_token_slot_state_weight=args.future_bank_token_slot_state_weight,
            future_bank_token_slot_mask_weight=args.future_bank_token_slot_mask_weight,
            future_bank_token_slot_type_weight=args.future_bank_token_slot_type_weight,
            future_bank_token_slot_zone_weight=args.future_bank_token_slot_zone_weight,
            future_bank_token_slot_source_weight=args.future_bank_token_slot_source_weight,
            token_teacher_ema_decay=args.token_teacher_ema_decay,
            future_world_rollout_weight=args.future_world_rollout_weight,
            future_world_rollout_steps=args.future_world_rollout_steps,
            future_world_rollout_decay=args.future_world_rollout_decay,
            latent_gaussian_reg_weight=args.latent_gaussian_reg_weight,
            latent_gaussian_reg_projections=args.latent_gaussian_reg_projections,
            latent_gaussian_reg_slot_weight=args.latent_gaussian_reg_slot_weight,
            latent_gaussian_reg_dynamics_weight=args.latent_gaussian_reg_dynamics_weight,
            latent_gaussian_reg_cov_weight=args.latent_gaussian_reg_cov_weight,
            surprise_loss_weight=args.surprise_loss_weight,
            surprise_hidden_scale=args.surprise_hidden_scale,
            surprise_surface_scale=args.surprise_surface_scale,
            surprise_future_aux_scale=args.surprise_future_aux_scale,
            surprise_target_cap=args.surprise_target_cap,
            surface_mask_weight=args.surface_mask_weight,
            surface_count_weight=args.surface_count_weight,
            surface_domain_weight=args.surface_domain_weight,
            surface_phase_weight=args.surface_phase_weight,
            combat_direct_policy=args.combat_direct_policy,
            combat_rollout_q_blend=args.combat_rollout_q_blend,
            combat_rollout_objective_q_blend=args.combat_rollout_objective_q_blend,
            combat_rollout_risk_blend=args.combat_rollout_risk_blend,
            combat_rollout_steps=args.combat_rollout_steps,
            combat_rollout_beam_width=args.combat_rollout_beam_width,
            combat_rollout_legal_logit_scale=args.combat_rollout_legal_logit_scale,
            combat_rollout_uncertainty_blend=args.combat_rollout_uncertainty_blend,
            combat_rollout_uncertainty_surprise_weight=args.combat_rollout_uncertainty_surprise_weight,
            combat_rollout_uncertainty_surface_weight=args.combat_rollout_uncertainty_surface_weight,
            combat_rollout_uncertainty_latent_weight=args.combat_rollout_uncertainty_latent_weight,
            combat_rollout_uncertainty_disagreement_weight=args.combat_rollout_uncertainty_disagreement_weight,
            combat_rollout_continuation_uncertainty_penalty=args.combat_rollout_continuation_uncertainty_penalty,
            combat_num_simulations=args.combat_num_simulations,
            build_num_simulations=args.build_num_simulations,
            route_num_simulations=args.route_num_simulations,
            settlement_weight=args.settlement_weight,
            settlement_decay=args.settlement_decay,
            settlement_max_steps=args.settlement_max_steps,
            trivial_build_fast_path=not args.disable_trivial_build_fast_path,
            potion_reward_fast_path=not args.disable_potion_reward_fast_path,
            recent_tail_windows=recent_tail_windows,
            recent_tail_tracked_encounters=recent_tail_tracked_encounters,
            recent_tail_min_samples=args.recent_tail_min_samples,
            route_heuristic_bias=args.route_heuristic_bias,
            route_safety_guard=args.route_safety_guard,
            log_dir=actor_log_dir,
            checkpoint_dir=actor_ckpt_dir,
            checkpoint_keep_last=0,
            mixed_precision=args.mixed_precision,
            amp_init_scale=args.amp_init_scale,
        )
        try:
            actor_trainer.writer.close()
        except Exception:
            pass
        actor_trainer.writer = NullSummaryWriter()

        loaded_weight_version = -1
        while not actor_stop_event.is_set():
            try:
                latest_total_steps = int(trainer.total_steps)
                state_snapshot: dict[str, torch.Tensor] | None = None
                with latest_weight_lock:
                    weight_version = int(latest_weight_state.get("version", 0))
                    latest_total_steps = int(latest_weight_state.get("total_steps", trainer.total_steps))
                    if weight_version != loaded_weight_version:
                        state_snapshot = {
                            key: value.clone()
                            for key, value in latest_weight_state.get("state_dict", {}).items()
                        }

                if state_snapshot is not None:
                    actor_network.load_state_dict(
                        {
                            key: value.to(args.device)
                            for key, value in state_snapshot.items()
                        }
                    )
                    loaded_weight_version = weight_version

                actor_trainer.total_steps = int(latest_total_steps)
                temperature = actor_trainer.compute_temperature(latest_total_steps, args.total_timesteps)
                ep_reward, ep_length = actor_trainer.self_play_episode(temperature=temperature)
                trajectory = actor_buffer.pop_latest()
                if trajectory is None:
                    raise RuntimeError(f"Async combat actor {actor_index} produced no trajectory")

                actor_failure_counts[actor_index] = 0
                actor_completed_episodes[actor_index] += 1
                packet = {
                    "kind": "episode",
                    "actor_index": int(actor_index),
                    "episode_reward": float(ep_reward),
                    "episode_length": int(ep_length),
                    "temperature": float(temperature),
                    "trajectory": trajectory,
                    "episode_metrics": dict(actor_trainer.last_episode_metrics or {}),
                    "weight_version": int(loaded_weight_version),
                    "actor_episode_count": int(actor_completed_episodes[actor_index]),
                }
                while not actor_stop_event.is_set():
                    try:
                        async_episode_queue.put(packet, timeout=0.5)
                        break
                    except queue.Full:
                        continue
            except BridgeError as exc:
                actor_failure_counts[actor_index] += 1
                failure_count = int(actor_failure_counts[actor_index])
                cooldown_s = min(60.0, 5.0 * float(failure_count))
                emit_async_env_failure(
                    actor_index=actor_index,
                    failure_count=failure_count,
                    cooldown_s=cooldown_s,
                    message=(
                        f"[warn] Async combat actor[{actor_index}] bridge failure: {exc} "
                        f"| cooldown={cooldown_s:.1f}s | failures={failure_count}"
                    ),
                )
                close_env_safely(envs[actor_index], actor_index)
                deadline = time.time() + cooldown_s
                while not actor_stop_event.is_set() and time.time() < deadline:
                    time.sleep(0.25)
                if actor_stop_event.is_set():
                    break
                try:
                    envs[actor_index] = build_env_for_index(actor_index)
                    actor_trainer.env = envs[actor_index]
                    emit_async_env_failure(
                        actor_index=actor_index,
                        failure_count=failure_count,
                        cooldown_s=0.0,
                        message=f"[recover] Rebuilt async combat env[{actor_index}] bound to {train_session_files[actor_index]}",
                    )
                except Exception as rebuild_exc:
                    emit_async_env_failure(
                        actor_index=actor_index,
                        failure_count=failure_count,
                        cooldown_s=cooldown_s,
                        message=f"[recover] Failed to rebuild async combat env[{actor_index}]: {rebuild_exc}",
                    )
            except Exception as exc:
                try:
                    async_episode_queue.put(
                        {
                            "kind": "actor_fatal",
                            "actor_index": int(actor_index),
                            "message": f"Async combat actor[{actor_index}] crashed: {type(exc).__name__}: {exc}",
                        },
                        timeout=0.1,
                    )
                except queue.Full:
                    pass
                actor_stop_event.set()
                break


    # Training loop
    try:
        if async_combat_actor_learner:
            publish_latest_weights()
            for actor_index in range(len(envs)):
                actor_thread = threading.Thread(
                    target=actor_thread_main,
                    args=(actor_index,),
                    name=f"combat-actor-{actor_index}",
                    daemon=True,
                )
                actor_thread.start()
                actor_threads.append(actor_thread)

        while trainer.total_steps < args.total_timesteps:
            active_env_index = 0
            temperature = trainer.compute_temperature(trainer.total_steps, args.total_timesteps)
            if async_combat_actor_learner:
                try:
                    packet = async_episode_queue.get(timeout=5.0)
                except queue.Empty:
                    print("[warn] Async combat collector idle for 5s; actors are still gathering trajectories")
                    continue

                packet_kind = str(packet.get("kind") or "episode")
                if packet_kind == "actor_recover":
                    active_env_index = int(packet.get("actor_index", 0) or 0)
                    print(str(packet.get("message") or ""))
                    trainer.writer.add_scalar("env/failure_total", float(sum(actor_failure_counts)), trainer.total_steps)
                    trainer.writer.add_scalar(
                        f"env/{active_env_index}_failure_count",
                        float(packet.get("failure_count", 0.0) or 0.0),
                        trainer.total_steps,
                    )
                    trainer.writer.add_scalar(
                        f"env/{active_env_index}_cooldown_seconds",
                        float(packet.get("cooldown_s", 0.0) or 0.0),
                        trainer.total_steps,
                    )
                    continue
                if packet_kind == "actor_fatal":
                    raise RuntimeError(str(packet.get("message") or "Async combat actor crashed"))

                active_env_index = int(packet.get("actor_index", 0) or 0)
                ep_reward = float(packet.get("episode_reward", 0.0) or 0.0)
                ep_length = int(packet.get("episode_length", 0) or 0)
                temperature = float(packet.get("temperature", temperature) or temperature)
                trajectory = packet.get("trajectory")
                if not isinstance(trajectory, GameTrajectory):
                    raise RuntimeError(f"Async combat actor[{active_env_index}] returned invalid trajectory payload")

                buffer.save_episode(
                    trajectory,
                    discount=trainer.discount,
                    n_steps=trainer.n_step_return,
                )
                trainer.total_steps += ep_length
                trainer.episode_count += 1
                trainer.last_episode_metrics = (
                    packet.get("episode_metrics")
                    if isinstance(packet.get("episode_metrics"), dict)
                    else {}
                )
                recent_tail_snapshot = trainer.record_recent_combat_episode(getattr(trajectory, "metadata", None))
                if recent_tail_snapshot:
                    trainer.last_episode_metrics["recent_tail"] = recent_tail_snapshot

                trainer.writer.add_scalar("episode/reward", ep_reward, trainer.episode_count)
                trainer.writer.add_scalar("episode/length", ep_length, trainer.episode_count)
                trainer.writer.add_scalar("schedule/temperature", temperature, trainer.total_steps)
                trainer.writer.add_scalar("schedule/active_env_index", float(active_env_index), trainer.total_steps)
                trainer.writer.add_scalar("schedule/weight_version", float(packet.get("weight_version", 0) or 0), trainer.total_steps)
                log_async_episode_scalars(
                    trainer=trainer,
                    actor_index=active_env_index,
                    episode_metrics=trainer.last_episode_metrics or {},
                    actor_completed_episodes=actor_completed_episodes,
                )
            else:
                ep_reward, ep_length = trainer.self_play_episode(temperature=temperature)
                trainer.total_steps += ep_length
                trainer.writer.add_scalar("episode/reward", ep_reward, trainer.episode_count)
                trainer.writer.add_scalar("episode/length", ep_length, trainer.episode_count)
                trainer.writer.add_scalar("schedule/temperature", temperature, trainer.total_steps)

            # Train on buffer
            train_updates_triggered = 0
            while len(buffer) >= args.min_buffer_size and (trainer.total_steps - last_train_step) >= args.train_every:
                for _ in range(args.updates_per_train):
                    losses = trainer.train_step(
                        batch_size=args.batch_size,
                        unroll_steps=args.unroll_steps,
                    )
                    for key, value in losses.items():
                        trainer.writer.add_scalar(key, value, trainer.total_steps)
                    total_loss_value = float(losses.get("loss/total", 0.0) or 0.0)
                    if total_loss_value != 0.0:
                        aux_keys = (
                            "loss/planner_q",
                            "loss/planner_objective_q",
                            "loss/future_world_aux",
                            "loss/state_consistency",
                            "loss/semantic_state_consistency",
                            "loss/latent_policy_distill",
                            "loss/surface_mask",
                            "loss/surface_count",
                            "loss/surface_domain",
                            "loss/surface_phase",
                        )
                        aux_total = float(sum(float(losses.get(key, 0.0) or 0.0) for key in aux_keys))
                        for key in ("loss/policy", "loss/value", "loss/reward", *aux_keys):
                            safe_suffix = key.split("/", 1)[-1]
                            trainer.writer.add_scalar(
                                f"loss_ratio/{safe_suffix}",
                                float(losses.get(key, 0.0) or 0.0) / total_loss_value,
                                trainer.total_steps,
                            )
                        trainer.writer.add_scalar("loss_ratio/aux_total", aux_total / total_loss_value, trainer.total_steps)
                last_train_step += int(args.train_every)
                train_updates_triggered += int(args.updates_per_train)
            if train_updates_triggered > 0:
                if async_combat_actor_learner:
                    publish_latest_weights()
                print(
                    f"[train] Update | Episode {trainer.episode_count} | "
                    f"Steps {trainer.total_steps} | Buffer {len(buffer)} | "
                    f"OptimizerSteps {train_updates_triggered}"
                )
                trainer.writer.flush()

            # Logging
            if (trainer.episode_count % 10) == 0:
                print(
                    f"[train] Episode {trainer.episode_count} | "
                    f"Steps {trainer.total_steps}/{args.total_timesteps} | "
                    f"Reward {ep_reward:+.4f} | "
                    f"Length {ep_length} | "
                    f"Buffer {len(buffer)} | "
                    f"T {temperature:.3f}"
                )
                episode_metrics = trainer.last_episode_metrics or {}
                decision_counts = episode_metrics.get("decision_counts") if isinstance(episode_metrics, dict) else {}
                fast_path_counts = episode_metrics.get("fast_path_counts") if isinstance(episode_metrics, dict) else {}
                fast_path_total = int(episode_metrics.get("fast_path_total", 0)) if isinstance(episode_metrics, dict) else 0
                fast_path_reason_counts = episode_metrics.get("fast_path_reason_counts") if isinstance(episode_metrics, dict) else {}
                domain_search_means = episode_metrics.get("domain_search_means") if isinstance(episode_metrics, dict) else {}
                domain_family_rates = episode_metrics.get("domain_family_rates") if isinstance(episode_metrics, dict) else {}
                if isinstance(decision_counts, dict) and decision_counts:
                    print(
                        "[train] Decisions | "
                        + " ".join(
                            f"{domain}={int(decision_counts.get(domain, 0))}"
                            for domain in DECISION_DOMAINS
                        )
                    )
                if fast_path_total > 0 and isinstance(fast_path_counts, dict):
                    reason_summary = ""
                    if isinstance(fast_path_reason_counts, dict) and fast_path_reason_counts:
                        ranked_reasons = sorted(
                            fast_path_reason_counts.items(),
                            key=lambda item: (-int(item[1]), str(item[0])),
                        )
                        reason_summary = " | reasons=" + ",".join(
                            f"{reason}:{int(count)}"
                            for reason, count in ranked_reasons[:3]
                        )
                    print(
                        "[train] FastPath | "
                        f"total={fast_path_total} "
                        + " ".join(
                            f"{domain}={int(fast_path_counts.get(domain, 0))}"
                            for domain in DECISION_DOMAINS
                        )
                        + reason_summary
                    )
                if mode_name == "full_run" and isinstance(domain_search_means, dict):
                    planner_parts: list[str] = []
                    for domain in ("build", "route"):
                        metrics = domain_search_means.get(domain) if isinstance(domain_search_means.get(domain), dict) else None
                        if not metrics:
                            continue
                        segment = (
                            f"{domain}[root={float(metrics.get('root_candidates', 0.0)):.2f} "
                            f"expand={float(metrics.get('mean_expanded_children', 0.0)):.2f} "
                            f"leaf={float(metrics.get('mean_leaf_depth', 0.0)):.2f} "
                            f"max={float(metrics.get('max_search_depth', 0.0)):.2f}"
                        )
                        family_rates = domain_family_rates.get(domain) if isinstance(domain_family_rates, dict) else None
                        if isinstance(family_rates, dict) and family_rates:
                            top_family, top_rate = max(family_rates.items(), key=lambda item: item[1])
                            segment += f" top={top_family}:{float(top_rate):.2f}"
                        segment += "]"
                        planner_parts.append(segment)
                    if planner_parts:
                        print("[train] PlannerSearch | " + " | ".join(planner_parts))
                recent_tail = episode_metrics.get("recent_tail") if isinstance(episode_metrics, dict) else None
                if isinstance(recent_tail, dict) and recent_tail:
                    tail_parts: list[str] = []
                    for window in sorted(recent_tail.keys()):
                        window_stats = recent_tail.get(window) if isinstance(recent_tail.get(window), dict) else None
                        if not window_stats:
                            continue
                        groups = window_stats.get("groups") if isinstance(window_stats.get("groups"), dict) else {}
                        tiers = window_stats.get("tiers") if isinstance(window_stats.get("tiers"), dict) else {}
                        elite_stats = tiers.get("elite") if isinstance(tiers.get("elite"), dict) else {}
                        normal_stats = tiers.get("normal") if isinstance(tiers.get("normal"), dict) else {}
                        hard_normal_stats = groups.get("hard_normal") if isinstance(groups.get("hard_normal"), dict) else {}
                        hard_elite_stats = groups.get("hard_elite") if isinstance(groups.get("hard_elite"), dict) else {}
                        tail_parts.append(
                            f"{window}[win={float(window_stats.get('win_rate', 0.0)):.3f} "
                            f"reward={float(window_stats.get('reward_mean', 0.0)):+.3f} "
                            f"normal={float(normal_stats.get('win_rate', 0.0)):.3f} "
                            f"elite={float(elite_stats.get('win_rate', 0.0)):.3f} "
                            f"hardN={float(hard_normal_stats.get('win_rate', 0.0)):.3f} "
                            f"hardE={float(hard_elite_stats.get('win_rate', 0.0)):.3f}]"
                        )
                    if tail_parts:
                        print("[train] RecentTail | " + " | ".join(tail_parts))

            # Checkpointing
            if (trainer.total_steps % args.checkpoint_freq) < ep_length:
                trainer.save_checkpoint()

            trainer.writer.add_scalar("buffer/size", len(buffer), trainer.total_steps)
            if (trainer.episode_count % 5) == 0:
                trainer.writer.flush()

    except KeyboardInterrupt:
        print("[interrupt] Caught KeyboardInterrupt, saving emergency checkpoint...")
        trainer.save_checkpoint(tag="crash")
        raise
    else:
        print(f"[train] Finished training after {trainer.total_steps} steps")
        trainer.save_checkpoint(tag="final")
    finally:
        actor_stop_event.set()
        for actor_thread in actor_threads:
            actor_thread.join(timeout=5.0)
        for env_index, train_env in enumerate(envs):
            close_env_safely(train_env, env_index)
        trainer.writer.close()
