"""Portable Colab GPU + Google Drive campaign runner for Durable Afterstate Macro Credit v2.

Runs ``sts2_rl.train_macro_credit`` in bounded update chunks on fast local storage
(``STS2_ARTIFACT_ROOT``), evaluates ``sts2_rl.macro_competence_probes`` after each
chunk, prunes bulky old raw ``*.jsonl`` rollout traces while retaining all cohort
metadata, and atomically mirrors resumable checkpoints, publications, metrics,
and milestone snapshots to a Google Drive directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sts2_rl.artifacts import (
    artifact_root,
    resolve_artifact_path,
    resolve_external_input_path,
    validate_artifact_component,
)

CAMPAIGN_RUNNER_VERSION = "sts2-colab-durable-campaign-v1"
_CORE_SYNC_FILES: tuple[str, ...] = (
    "checkpoint.pt",
    "publication.pt",
    "course.json",
    "source_manifest.json",
    "status.json",
    "probe_metrics.jsonl",
    "campaign_history.jsonl",
)


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with tmp.open("wb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = (json.dumps(payload, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    with path.open("ab") as handle:
        handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())


def _atomic_copy_file(src: Path, dst: Path, *, verify_sha256: bool = False) -> tuple[int, str | None]:
    dst.parent.mkdir(parents=True, exist_ok=True)
    expected_bytes = src.stat().st_size
    src_hash = _sha256_file(src) if verify_sha256 else None
    tmp = dst.with_name(f".{dst.name}.tmp-{os.getpid()}")
    with src.open("rb") as reader, tmp.open("wb") as writer:
        shutil.copyfileobj(reader, writer, length=4 * 1024 * 1024)
        writer.flush()
        os.fsync(writer.fileno())
    copied_bytes = tmp.stat().st_size
    if copied_bytes != expected_bytes:
        tmp.unlink(missing_ok=True)
        raise OSError(f"incomplete copy to {dst}: expected {expected_bytes} bytes, got {copied_bytes}")
    if verify_sha256:
        dst_hash = _sha256_file(tmp)
        if dst_hash != src_hash:
            tmp.unlink(missing_ok=True)
            raise OSError(f"sha256 mismatch copying {src} -> {dst}: {src_hash} != {dst_hash}")
    os.replace(tmp, dst)
    return copied_bytes, src_hash


def prune_old_raw_cohort_traces(output_dir: Path, *, keep_recent_cohorts: int) -> int:
    """Remove bulky raw ``*.jsonl`` files from older cohorts while keeping JSON summaries."""
    if keep_recent_cohorts < 0:
        raise ValueError("keep_recent_cohorts must be non-negative")
    cohorts_dir = output_dir / "cohorts"
    if not cohorts_dir.is_dir():
        return 0
    cohort_dirs = sorted(
        (entry for entry in cohorts_dir.iterdir() if entry.is_dir() and entry.name.startswith("update-")),
        key=lambda item: item.name,
    )
    cutoff = len(cohort_dirs) - keep_recent_cohorts if keep_recent_cohorts > 0 else len(cohort_dirs)
    removed_bytes = 0
    for old_dir in cohort_dirs[: max(0, cutoff)]:
        for trace_file in old_dir.rglob("*.jsonl"):
            removed_bytes += trace_file.stat().st_size
            trace_file.unlink()
    return removed_bytes


def sync_campaign_to_drive(
    output_dir: Path,
    drive_dir: Path,
    *,
    lineage_id: str,
    snapshot_interval: int = 5,
    latest_probe: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Atomically mirror resumable campaign state from local NVMe to Google Drive."""
    checkpoint_src = output_dir / "checkpoint.pt"
    publication_src = output_dir / "publication.pt"
    if not checkpoint_src.is_file() or not publication_src.is_file():
        raise FileNotFoundError(f"cannot sync incomplete campaign directory {output_dir}")

    drive_dir.mkdir(parents=True, exist_ok=True)
    drive_ckpt = drive_dir / "checkpoint.pt"
    if drive_ckpt.is_file():
        _atomic_copy_file(drive_ckpt, drive_dir / "checkpoint.prev.pt", verify_sha256=False)

    ckpt_bytes, ckpt_sha256 = _atomic_copy_file(checkpoint_src, drive_ckpt, verify_sha256=True)
    pub_bytes, pub_sha256 = _atomic_copy_file(publication_src, drive_dir / "publication.pt", verify_sha256=True)

    for rel_name in _CORE_SYNC_FILES:
        if rel_name in {"checkpoint.pt", "publication.pt"}:
            continue
        src_file = output_dir / rel_name
        if src_file.is_file():
            _atomic_copy_file(src_file, drive_dir / rel_name, verify_sha256=False)

    cohorts_src = output_dir / "cohorts"
    latest_update_payload: dict[str, Any] | None = None
    if cohorts_src.is_dir():
        for cohort_entry in sorted(cohorts_src.iterdir(), key=lambda item: item.name):
            if not cohort_entry.is_dir():
                continue
            dst_cohort = drive_dir / "cohorts" / cohort_entry.name
            for summary_name in ("collected.json", "update.json"):
                src_summary = cohort_entry / summary_name
                dst_summary = dst_cohort / summary_name
                if src_summary.is_file() and (
                    not dst_summary.is_file() or dst_summary.stat().st_size != src_summary.stat().st_size
                ):
                    _atomic_copy_file(src_summary, dst_summary, verify_sha256=False)
            update_json = cohort_entry / "update.json"
            if update_json.is_file():
                latest_update_payload = json.loads(update_json.read_text(encoding="utf-8"))

    status_payload: dict[str, Any] = {}
    status_path = output_dir / "status.json"
    if status_path.is_file():
        status_payload = json.loads(status_path.read_text(encoding="utf-8"))
    committed_update = int(status_payload.get("committed_update", 0))
    next_seed_index = int(status_payload.get("next_seed_index", 0))

    snapshot_saved = False
    if committed_update > 0 and (
        committed_update == 1 or (snapshot_interval > 0 and committed_update % snapshot_interval == 0)
    ):
        snap_dir = drive_dir / "snapshots" / f"update-{committed_update:06d}"
        if not (snap_dir / "checkpoint.pt").is_file():
            _atomic_copy_file(checkpoint_src, snap_dir / "checkpoint.pt", verify_sha256=True)
            _atomic_copy_file(publication_src, snap_dir / "publication.pt", verify_sha256=True)
            if latest_probe is not None:
                _atomic_write_json(snap_dir / "probe.json", latest_probe)
            snapshot_saved = True

    manifest = {
        "version": CAMPAIGN_RUNNER_VERSION,
        "lineage_id": lineage_id,
        "committed_update": committed_update,
        "next_seed_index": next_seed_index,
        "synced_at_utc": _utc_now_iso(),
        "checkpoint": {"bytes": ckpt_bytes, "sha256": ckpt_sha256},
        "publication": {"bytes": pub_bytes, "sha256": pub_sha256},
        "snapshot_saved": snapshot_saved,
        "latest_learner_metrics": (
            (latest_update_payload.get("learner") or latest_update_payload.get("metrics"))
            if isinstance(latest_update_payload, dict)
            else None
        ),
        "latest_probe": latest_probe,
    }
    _atomic_write_json(drive_dir / "sync_manifest.json", manifest)
    return manifest


def restore_campaign_from_drive(drive_dir: Path, output_dir: Path) -> dict[str, Any]:
    """Restore a resumable campaign directory from Google Drive onto local NVMe."""
    drive_ckpt = drive_dir / "checkpoint.pt"
    if not drive_ckpt.is_file():
        raise FileNotFoundError(f"no checkpoint.pt found in drive directory {drive_dir}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty local campaign directory {output_dir}")

    manifest_path = drive_dir / "sync_manifest.json"
    expected_ckpt_sha: str | None = None
    if manifest_path.is_file():
        manifest_raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        ckpt_meta = manifest_raw.get("checkpoint")
        if isinstance(ckpt_meta, dict) and isinstance(ckpt_meta.get("sha256"), str):
            expected_ckpt_sha = ckpt_meta["sha256"]

    output_dir.mkdir(parents=True, exist_ok=True)
    _, restored_sha = _atomic_copy_file(drive_ckpt, output_dir / "checkpoint.pt", verify_sha256=True)
    if expected_ckpt_sha is not None and restored_sha != expected_ckpt_sha:
        raise ValueError(
            f"restored checkpoint.pt sha256 {restored_sha} does not match sync_manifest.json {expected_ckpt_sha}"
        )

    for rel_name in _CORE_SYNC_FILES:
        if rel_name == "checkpoint.pt":
            continue
        src_file = drive_dir / rel_name
        if src_file.is_file():
            _atomic_copy_file(src_file, output_dir / rel_name, verify_sha256=rel_name == "publication.pt")

    drive_cohorts = drive_dir / "cohorts"
    if drive_cohorts.is_dir():
        for cohort_entry in sorted(drive_cohorts.iterdir(), key=lambda item: item.name):
            if not cohort_entry.is_dir():
                continue
            for summary_name in ("collected.json", "update.json"):
                src_summary = cohort_entry / summary_name
                if src_summary.is_file():
                    _atomic_copy_file(
                        src_summary,
                        output_dir / "cohorts" / cohort_entry.name / summary_name,
                        verify_sha256=False,
                    )

    status_path = output_dir / "status.json"
    status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.is_file() else {}
    return {
        "restored_from": str(drive_dir),
        "local_output_dir": str(output_dir),
        "checkpoint_sha256": restored_sha,
        "committed_update": int(status.get("committed_update", 0)),
    }


def _read_committed_update(output_dir: Path) -> int:
    status_path = output_dir / "status.json"
    if not status_path.is_file():
        return 0
    raw = json.loads(status_path.read_text(encoding="utf-8"))
    return int(raw.get("committed_update", 0))


def _run_competence_probes(
    *,
    python_exe: str,
    output_dir: Path,
    device: str,
    env: dict[str, str],
) -> dict[str, Any]:
    cmd = [
        python_exe,
        "-m",
        "sts2_rl.macro_competence_probes",
        "--publication",
        str(output_dir / "publication.pt"),
        "--experiment-dir",
        str(output_dir),
        "--device",
        device,
    ]
    completed = subprocess.run(cmd, env=env, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f"macro_competence_probes failed (exit {completed.returncode}): {completed.stderr.strip() or completed.stdout.strip()}"
        )
    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("macro_competence_probes produced empty stdout")
    payload: dict[str, Any] = json.loads(lines[-1])
    return payload


def _probe_prune_and_sync(
    *,
    output_dir: Path,
    drive_dir: Path,
    lineage_id: str,
    committed_after: int,
    train_summary: dict[str, Any],
    device: str,
    env: dict[str, str],
    snapshot_interval: int,
    keep_recent_cohorts: int,
) -> dict[str, Any]:
    probe_report = _run_competence_probes(
        python_exe=sys.executable,
        output_dir=output_dir,
        device=device,
        env=env,
    )
    probe_entry = {
        "committed_update": committed_after,
        "evaluated_at_utc": _utc_now_iso(),
        "probe": probe_report,
    }
    _append_jsonl(output_dir / "probe_metrics.jsonl", probe_entry)

    pruned_bytes = prune_old_raw_cohort_traces(
        output_dir,
        keep_recent_cohorts=keep_recent_cohorts,
    )
    split_inv = probe_report.get("multiset_split_invariance") or {}
    history_entry = {
        "committed_update": committed_after,
        "completed_at_utc": _utc_now_iso(),
        "train_summary": train_summary,
        "pruned_raw_trace_bytes": pruned_bytes,
        "campfire_low_minus_high_rest_prob": probe_report["campfire_hp_sensitivity"]["p_rest_low_minus_high"],
        "campfire_low_minus_high_durable_adv": probe_report["campfire_hp_sensitivity"][
            "durable_value_rest_advantage_shift"
        ],
        "smith_vs_remove_jsd": probe_report["smith_vs_remove_divergence"]["mean_smith_remove_jsd"],
        "smith_vs_remove_spearman": probe_report["smith_vs_remove_divergence"]["mean_smith_remove_spearman"],
        "grouped_vs_split_abs_diff": split_inv.get("grouped_vs_split_abs_diff"),
        "smith_high_delta_minus_basic_adv": split_inv.get("smith_high_delta_minus_basic_adv"),
        "episode_macro_discipline": probe_report["episode_macro_discipline"],
    }
    _append_jsonl(output_dir / "campaign_history.jsonl", history_entry)

    sync_manifest = sync_campaign_to_drive(
        output_dir,
        drive_dir,
        lineage_id=lineage_id,
        snapshot_interval=snapshot_interval,
        latest_probe=probe_entry,
    )
    print(
        json.dumps(
            {
                "event": "chunk_committed_and_synced",
                "committed_update": committed_after,
                "pruned_raw_trace_bytes": pruned_bytes,
                "drive_checkpoint_sha256": sync_manifest["checkpoint"]["sha256"],
                "snapshot_saved": sync_manifest["snapshot_saved"],
                "campfire_low_minus_high_rest_prob": history_entry["campfire_low_minus_high_rest_prob"],
                "smith_vs_remove_jsd": history_entry["smith_vs_remove_jsd"],
                "smith_vs_remove_spearman": history_entry["smith_vs_remove_spearman"],
                "grouped_vs_split_abs_diff": history_entry["grouped_vs_split_abs_diff"],
                "smith_high_delta_minus_basic_adv": history_entry["smith_high_delta_minus_basic_adv"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return sync_manifest


def _copy_migration_history(source_dir: Path, output_dir: Path) -> int:
    """Preserve prior cohort summaries and probe history when migrating runtime settings."""
    copied = 0
    for name in ("probe_metrics.jsonl", "campaign_history.jsonl"):
        src_file = source_dir / name
        dst_file = output_dir / name
        if src_file.is_file() and not dst_file.exists():
            _atomic_copy_file(src_file, dst_file, verify_sha256=False)
            copied += 1
    src_cohorts = source_dir / "cohorts"
    if src_cohorts.is_dir():
        for entry in sorted(src_cohorts.iterdir(), key=lambda item: item.name):
            if not entry.is_dir() or not entry.name.startswith("update-"):
                continue
            for summary_name in ("collected.json", "update.json"):
                src_summary = entry / summary_name
                dst_summary = output_dir / "cohorts" / entry.name / summary_name
                if src_summary.is_file() and not dst_summary.exists():
                    _atomic_copy_file(src_summary, dst_summary, verify_sha256=False)
                    copied += 1
    return copied


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("packages/rl-agent/config/experiments/hybrid_macro_credit_v6_durable_afterstate_b64.toml"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--drive-campaign-dir", type=Path, required=True)
    parser.add_argument("--lineage-id", required=True)
    parser.add_argument("--initialize-from-iql", type=Path)
    parser.add_argument("--initialize-from-macro-publication", type=Path)
    parser.add_argument("--distill-from-macro-publication", type=Path)
    parser.add_argument("--migrate-runtime-from-checkpoint", type=Path)
    parser.add_argument("--resume-from-drive", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sim-exe", type=Path)
    parser.add_argument("--target-updates", type=int, required=True)
    parser.add_argument("--updates-per-chunk", type=int, default=1)
    parser.add_argument("--snapshot-interval", type=int, default=5)
    parser.add_argument("--keep-raw-trace-cohorts", type=int, default=2)
    parser.add_argument("--set", action="append", default=[], dest="overrides", metavar="KEY=TOML_VALUE")
    return parser


def run_campaign(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.target_updates <= 0:
        raise ValueError("--target-updates must be positive")
    if args.updates_per_chunk <= 0:
        raise ValueError("--updates-per-chunk must be positive")
    lineage_id = validate_artifact_component(args.lineage_id, label="lineage_id")

    root = artifact_root()
    output_dir = resolve_artifact_path(args.output_dir, root=root)
    drive_dir = resolve_external_input_path(args.drive_campaign_dir, root=root)
    if drive_dir == output_dir or drive_dir.is_relative_to(output_dir) or output_dir.is_relative_to(drive_dir):
        raise ValueError("local output_dir and drive_campaign_dir must be disjoint directories")

    config_path = args.config if args.config.is_absolute() else Path.cwd() / args.config
    if not config_path.is_file():
        raise FileNotFoundError(f"config file not found: {config_path}")

    iql_source = (
        resolve_external_input_path(args.initialize_from_iql, root=root)
        if args.initialize_from_iql is not None
        else None
    )
    macro_source = (
        resolve_external_input_path(args.initialize_from_macro_publication, root=root)
        if args.initialize_from_macro_publication is not None
        else None
    )
    distill_source = (
        resolve_external_input_path(args.distill_from_macro_publication, root=root)
        if args.distill_from_macro_publication is not None
        else None
    )
    migrate_source = (
        resolve_external_input_path(args.migrate_runtime_from_checkpoint, root=root)
        if args.migrate_runtime_from_checkpoint is not None
        else None
    )
    init_count = sum(
        src is not None for src in (iql_source, macro_source, distill_source, migrate_source)
    )
    if init_count > 1:
        raise ValueError("specify at most one initialization/migration source")

    local_ckpt = output_dir / "checkpoint.pt"
    drive_ckpt = drive_dir / "checkpoint.pt"
    local_teacher_cache = root / "teachers" / f"{lineage_id}_teacher_publication.pt"
    drive_teacher_backup = drive_dir / "teacher_publication.pt"
    if (
        not local_ckpt.is_file()
        and migrate_source is None
        and args.resume_from_drive
        and drive_ckpt.is_file()
    ):
        restored = restore_campaign_from_drive(drive_dir, output_dir)
        print(json.dumps({"event": "restored_from_drive", **restored}, sort_keys=True), flush=True)

    if not local_ckpt.is_file() and init_count == 0:
        raise ValueError(
            "no local or restored checkpoint.pt found; supply --initialize-from-iql, "
            "--initialize-from-macro-publication, --distill-from-macro-publication, "
            "or --migrate-runtime-from-checkpoint"
        )

    if distill_source is not None:
        if not distill_source.is_file():
            raise FileNotFoundError(f"distillation teacher publication not found: {distill_source}")
        if not local_teacher_cache.is_file() or local_teacher_cache.resolve() != distill_source.resolve():
            _atomic_copy_file(distill_source, local_teacher_cache, verify_sha256=True)
        if not drive_teacher_backup.is_file():
            _atomic_copy_file(local_teacher_cache, drive_teacher_backup, verify_sha256=True)
    elif drive_teacher_backup.is_file() and not local_teacher_cache.is_file():
        _atomic_copy_file(drive_teacher_backup, local_teacher_cache, verify_sha256=True)

    if not local_ckpt.is_file() and migrate_source is not None:
        copied_files = _copy_migration_history(migrate_source.parent, output_dir)
        print(
            json.dumps(
                {
                    "event": "prepared_runtime_migration",
                    "source_checkpoint": str(migrate_source),
                    "copied_history_files": copied_files,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    env = dict(os.environ)
    env["STS2_ARTIFACT_ROOT"] = str(root)
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    # Colab runtimes preload libtcmalloc for TensorFlow, which aborts .NET 9 CoreCLR binaries.
    if "tcmalloc" in env.get("LD_PRELOAD", ""):
        env.pop("LD_PRELOAD", None)
    mountinfo_shim = Path("/root/libfix_mountinfo.so")
    if mountinfo_shim.is_file() and str(mountinfo_shim) not in env.get("LD_PRELOAD", ""):
        existing_preload = env.get("LD_PRELOAD", "").strip()
        env["LD_PRELOAD"] = f"{mountinfo_shim}:{existing_preload}" if existing_preload else str(mountinfo_shim)
    if not env.get("DOTNET_ROOT"):
        for candidate_dotnet in (root / "dotnet", Path.home() / ".dotnet"):
            if (candidate_dotnet / "dotnet").is_file():
                env["DOTNET_ROOT"] = str(candidate_dotnet)
                env["PATH"] = f"{candidate_dotnet}{os.pathsep}{env.get('PATH', '')}"
                break

    drive_synced_update = (
        int(json.loads((drive_dir / "sync_manifest.json").read_text(encoding="utf-8")).get("committed_update", 0))
        if (drive_dir / "sync_manifest.json").is_file()
        else 0
    )
    local_initial_update = _read_committed_update(output_dir) if local_ckpt.is_file() else 0
    if local_ckpt.is_file() and local_initial_update > drive_synced_update:
        _probe_prune_and_sync(
            output_dir=output_dir,
            drive_dir=drive_dir,
            lineage_id=lineage_id,
            committed_after=local_initial_update,
            train_summary={"recovered_unsynced_local_commit": True},
            device=args.device,
            env=env,
            snapshot_interval=args.snapshot_interval,
            keep_recent_cohorts=args.keep_raw_trace_cohorts,
        )

    while True:
        if local_ckpt.is_file():
            current_update = _read_committed_update(output_dir)
        elif migrate_source is not None:
            src_status = migrate_source.parent / "status.json"
            current_update = (
                int(json.loads(src_status.read_text(encoding="utf-8")).get("committed_update", 0))
                if src_status.is_file()
                else 0
            )
        else:
            current_update = 0
        if local_ckpt.is_file() and current_update >= args.target_updates:
            print(
                json.dumps(
                    {
                        "event": "campaign_target_reached",
                        "lineage_id": lineage_id,
                        "committed_update": current_update,
                        "target_updates": args.target_updates,
                        "drive_campaign_dir": str(drive_dir),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            return 0

        chunk_updates = min(args.updates_per_chunk, max(1, args.target_updates - current_update))
        if args.updates_per_chunk > 1 and args.snapshot_interval > 1:
            next_milestone = ((current_update // args.snapshot_interval) + 1) * args.snapshot_interval
            if current_update == 0:
                next_milestone = 1
            chunk_updates = min(chunk_updates, max(1, next_milestone - current_update))
        cmd = [
            sys.executable,
            "-m",
            "sts2_rl.train_macro_credit",
            "--config",
            str(config_path),
            "--output-dir",
            str(output_dir),
            "--lineage-id",
            lineage_id,
            "--device",
            args.device,
            "--max-updates",
            str(chunk_updates),
        ]
        if args.sim_exe is not None:
            cmd.extend(["--sim-exe", str(args.sim_exe)])
        for item in args.overrides:
            cmd.extend(["--set", item])
        if local_ckpt.is_file():
            cmd.extend(["--resume", str(local_ckpt)])
            if local_teacher_cache.is_file():
                cmd.extend(["--teacher-publication", str(local_teacher_cache)])
        elif migrate_source is not None:
            cmd.extend(["--migrate-runtime-from-checkpoint", str(migrate_source)])
            if local_teacher_cache.is_file():
                cmd.extend(["--teacher-publication", str(local_teacher_cache)])
        elif iql_source is not None:
            cmd.extend(["--initialize-from-iql", str(iql_source)])
        elif distill_source is not None:
            cmd.extend(["--distill-from-macro-publication", str(local_teacher_cache)])
        else:
            assert macro_source is not None
            cmd.extend(["--initialize-from-macro-publication", str(macro_source)])

        print(
            json.dumps(
                {
                    "event": "chunk_start",
                    "lineage_id": lineage_id,
                    "starting_update": current_update,
                    "chunk_updates": chunk_updates,
                    "started_at_utc": _utc_now_iso(),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        completed = subprocess.run(cmd, env=env, capture_output=True, text=True, check=False)
        if completed.returncode != 0:
            if local_ckpt.is_file():
                sync_campaign_to_drive(
                    output_dir,
                    drive_dir,
                    lineage_id=lineage_id,
                    snapshot_interval=args.snapshot_interval,
                )
            print(
                json.dumps(
                    {
                        "event": "chunk_failed",
                        "returncode": completed.returncode,
                        "stdout": completed.stdout[-4000:],
                        "stderr": completed.stderr[-4000:],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            return completed.returncode

        train_lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
        train_summary = json.loads(train_lines[-1]) if train_lines else {}
        committed_after = _read_committed_update(output_dir)

        _probe_prune_and_sync(
            output_dir=output_dir,
            drive_dir=drive_dir,
            lineage_id=lineage_id,
            committed_after=committed_after,
            train_summary=train_summary,
            device=args.device,
            env=env,
            snapshot_interval=args.snapshot_interval,
            keep_recent_cohorts=args.keep_raw_trace_cohorts,
        )


def main(argv: Sequence[str] | None = None) -> int:
    try:
        return run_campaign(argv)
    except BaseException as error:
        print(
            json.dumps(
                {
                    "event": "campaign_aborted",
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return 130 if isinstance(error, KeyboardInterrupt) else 1


if __name__ == "__main__":
    raise SystemExit(main())
