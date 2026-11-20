"""Tests for Durable Afterstate v7.4 Joint Macro+Combat B=0 curriculum and dashboard sync resume cleanup."""

from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from sts2_baseline import act_frontloaded_death_reward_identity  # noqa: E402
from sts2_rl.joint.macro_checkpoint import (  # noqa: E402
    MacroCheckpointIdentity,
    publish_macro_checkpoint,
    save_macro_checkpoint,
)
from sts2_rl.joint.macro_credit_learner import MACRO_CREDIT_LEARNER_VERSION  # noqa: E402
from sts2_rl.macro_root_policy import FrozenMacroRootPolicy  # noqa: E402
from sts2_rl.models.hybrid_actor_critic import (  # noqa: E402
    HybridActorCriticConfig,
    HybridDomainActorCriticModel,
)
from sts2_rl.train_macro_credit import _actor_critic_config, load_macro_training_config  # noqa: E402
from sts2_rl.training.config import ModelConfig  # noqa: E402


def _load_script_module(name: str, rel_path: str) -> Any:
    script_path = Path(__file__).resolve().parents[1] / rel_path
    spec = importlib.util.spec_from_file_location(name, script_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_v27_joint_b0_config_and_lossless_v73_hotstart(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.setenv("STS2_ARTIFACT_ROOT", str(tmp_path))
    cfg_path = (
        Path(__file__).resolve().parents[1]
        / "config"
        / "experiments"
        / "hybrid_macro_credit_v27_durable_afterstate_v74_joint_b0_act1_g4.toml"
    )
    mcfg = load_macro_training_config(cfg_path)
    assert mcfg.course.seed_index_offset == 765_000
    assert mcfg.training.curriculum.revival_budget == 0
    assert mcfg.training.curriculum.reward_objective == "act1"
    assert mcfg.learner.enable_direct_combat_vitals_context is True
    assert mcfg.learner.durable_warmup_updates == 5
    assert mcfg.learner.combat_actor_warmup_updates == 5
    assert mcfg.learner.macro_actor_warmup_updates == 5
    assert mcfg.learner.local_combat_win_reward == pytest.approx(1.5)
    assert mcfg.learner.local_combat_loss_reward == pytest.approx(-1.5)
    assert mcfg.learner.local_combat_hp_weight == pytest.approx(1.2)
    assert mcfg.learner.combat_entropy_weight == pytest.approx(0.012)
    assert mcfg.learner.macro_entropy_weight == pytest.approx(0.002)
    assert mcfg.learner.likelihood_tolerance == pytest.approx(0.15)
    assert mcfg.learner.target_macro_step_kl == pytest.approx(0.015)
    assert mcfg.learner.target_combat_step_kl == pytest.approx(0.025)

    tiny = ModelConfig(
        d_model=32,
        n_heads=4,
        ffn_dim=64,
        world_layers=1,
        latent_slots=4,
        latent_layers=1,
        recurrent_hidden_dim=64,
        dropout=0.0,
        max_world_tokens=256,
        max_candidates=32,
        max_candidate_local_tokens=16,
    )
    tiny_macro = replace(
        mcfg,
        training=replace(mcfg.training, model=tiny),
        branches=replace(mcfg.branches, max_candidates=32),
    )
    ac_cfg = _actor_critic_config(tiny_macro)
    assert ac_cfg.enable_direct_combat_vitals_context is True
    assert ac_cfg.enable_local_combat_value_head is True

    parent_model = HybridDomainActorCriticModel(ac_cfg)
    opt = torch.optim.AdamW(parent_model.parameters(), lr=1e-4)
    parent_id = MacroCheckpointIdentity(
        course_config={"version": "test"},
        model_config=ac_cfg.as_mapping(),
        reward_identity=act_frontloaded_death_reward_identity(revival_budget=0),
        lineage_id="v73-u80-parent",
        learner_version=MACRO_CREDIT_LEARNER_VERSION,
    )
    ckpt = tmp_path / "v73" / "checkpoint.pt"
    pub = tmp_path / "v73" / "publication.pt"
    save_macro_checkpoint(
        ckpt,
        identity=parent_id,
        model=parent_model,
        optimizer=opt,
        update=80,
        next_seed_index=740_000,
        initialization={"mode": "test"},
        artifact_root=tmp_path,
    )
    publish_macro_checkpoint(ckpt, pub, identity=parent_id, artifact_root=tmp_path)

    loaded = FrozenMacroRootPolicy(
        pub,
        tiny_macro.training,
        "cpu",
        initialize_from_macro_publication=True,
        actor_critic_config=ac_cfg,
    )
    assert loaded.provenance["fresh_tensor_names"] == []
    assert loaded.provenance.get("dropped_tensor_names", []) == []


def test_ratchet_start_stage_and_v74_joint_overrides() -> None:
    mod = _load_script_module("ratchet_v74_mod", "scripts/run_colab_g4_ratchet_curriculum.py")

    parser = mod._build_parser()
    parsed = parser.parse_args(
        [
            "--initial-publication",
            "dummy.pt",
            "--drive-root-dir",
            "dummy_drive",
            "--curriculum-mode",
            "act-b0",
            "--start-stage",
            "b0-act1",
        ]
    )
    assert parsed.max_updates_per_stage == 250
    assert parsed.start_stage == "b0-act1"

    sliced = mod._select_ratchet_stages(curriculum_mode="act-b0", start_stage="b0-act1")
    assert [s.stage_key for s in sliced] == ["b0-act1", "b0-act2", "b0-act3"]
    assert sliced[0].seed_index_offset == 765_000
    for expected in (
        "learner.macro_actor_warmup_updates=5",
        "learner.local_combat_win_reward=1.5",
        "learner.local_combat_loss_reward=-1.5",
        "learner.local_combat_hp_weight=1.2",
        "learner.combat_entropy_weight=0.012",
        "learner.likelihood_tolerance=0.15",
    ):
        assert expected in sliced[0].extra_overrides

    with pytest.raises(ValueError, match="cannot be combined"):
        mod._select_ratchet_stages(curriculum_mode="act-b0", stages="b0-act1", start_stage="b0-act1")

    with pytest.raises(ValueError, match="unknown --start-stage"):
        mod._select_ratchet_stages(curriculum_mode="act-b0", start_stage="b0-nope")


def test_dashboard_sync_clears_stale_paused_cap_terminal_event_on_resume(tmp_path: Path) -> None:
    sync_mod = _load_script_module("sync_v74_mod", "scripts/sync_colab_g4_to_dashboard.py")

    root = tmp_path / "runtime"
    lineage = "durable-afterstate-v7-4-b0-act1-g4"
    run_uuid = "11111111-2222-3333-4444-555555555555"
    fake_id = tmp_path / "sim_id.json"
    fake_id.write_text("{}", encoding="utf-8")
    fake_sim = SimpleNamespace(executable=tmp_path / "HeadlessSim", identity_path=fake_id)

    # 1. Stage reaches cap and pauses -> sync emits run_complete (paused_stage_cap_reached).
    paused_pull = tmp_path / "paused.json"
    paused_pull.write_text(
        json.dumps(
            {
                "alive": False,
                "status": {"state": "bounded_training_complete"},
                "revival_budget": 0,
                "reward_objective": "act1",
                "updates": [
                    {
                        "committed_update": 10,
                        "_mtime": 2_000.0,
                        "rollout_policy": "learned_model",
                        "learner": {"durable_value_loss": 0.05, "macro_mpo_policy_loss": 0.02},
                        "training_episodes": [
                            {
                                "seed": 765_000,
                                "native_steps": 400,
                                "run_result": "victory",
                                "max_floor": 17,
                                "revivals_used": 0,
                                "player_hp_lost": 30.0,
                                "observed_discounted_return": 1.2,
                            }
                        ],
                    }
                ],
                "heldout_evaluations": [],
                "cohort_replays": {},
            }
        ),
        encoding="utf-8",
    )
    sync_mod.sync_once(
        root=root,
        simulator=fake_sim,
        session="unused",
        lineage=lineage,
        run_uuid=run_uuid,
        pid_path="/unused",
        remote_local_dir="/unused",
        target_updates=250,
        revival_budget=0,
        seed_pull_file=paused_pull,
        offline_only=True,
        ratchet_state_label="paused_stage_cap_reached",
    )

    metrics_path = root / "runs" / lineage / f"run-{run_uuid}" / "metrics.jsonl"
    events_paused = [json.loads(ln) for ln in metrics_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert events_paused[0]["config"]["version"] == "sts2-durable-afterstate-v7.4-joint-b0"
    assert any(
        ev.get("event") == "run_complete" and ev.get("completion_status") == "paused_stage_cap_reached"
        for ev in events_paused
    )

    # 2. Stage resumes live (alive=True) -> stale paused_stage_cap_reached line MUST be stripped!
    resumed_pull = tmp_path / "resumed.json"
    resumed_pull.write_text(
        json.dumps(
            {
                "alive": True,
                "status": {"state": "collecting"},
                "revival_budget": 0,
                "reward_objective": "act1",
                "updates": [],
                "heldout_evaluations": [],
                "cohort_replays": {},
            }
        ),
        encoding="utf-8",
    )
    sync_mod.sync_once(
        root=root,
        simulator=fake_sim,
        session="unused",
        lineage=lineage,
        run_uuid=run_uuid,
        pid_path="/unused",
        remote_local_dir="/unused",
        target_updates=250,
        revival_budget=0,
        seed_pull_file=resumed_pull,
        offline_only=True,
        ratchet_state_label="running_stage",
    )

    events_resumed = [json.loads(ln) for ln in metrics_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert not any(ev.get("event") == "run_complete" for ev in events_resumed)
    assert any(ev.get("event") == "run_start" for ev in events_resumed)
    assert any(ev.get("event") == "learner_update" for ev in events_resumed)
    assert events_resumed[-1]["event"] == "learner_progress"
