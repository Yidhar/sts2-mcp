from __future__ import annotations

import json
import io
import os
import shutil
import stat
import subprocess
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tools.artifacts.inventory import (
    INVENTORY_SCHEMA_VERSION,
    MOVE_MAP_PATH,
    _entry_kind,
    _safe_relative,
    _validate_inventory_output,
    _write_json,
    build_inventory,
    compare_inventories,
    load_move_map,
    main as inventory_main,
    move_map_paths,
    remove_overlapping_paths,
)
from tools.artifacts.path_policy import artifact_root as canonical_artifact_root
from tools.artifacts.path_policy import resolve_artifact_path, resolve_external_input_path
from tools.third_party.restore_sts2_ai import ROOT, artifact_root, resolve_destination


class MoveMapTests(unittest.TestCase):
    def test_move_map_is_unique_canonical_and_covers_migration_residue(self) -> None:
        move_map = load_move_map()
        sources = move_map_paths("source")
        destinations = move_map_paths("artifact")
        self.assertEqual(len(sources), len(destinations))
        self.assertEqual(len(sources), len({value.casefold() for value in sources}))
        self.assertEqual(len(destinations), len({value.casefold() for value in destinations}))
        self.assertIn("packages/rl-agent/checkpoints_muzero", sources)
        self.assertIn("checkpoints/muzero", destinations)
        self.assertIn("dependencies/sts2-ai", destinations)
        self.assertNotIn("packages/rl-agent/checkpoints_muzero", destinations)
        self.assertIn("packages/mcp-server/autoslay-runner-20260325-210624.log", sources)
        self.assertIn("packages/rl-agent/training_v2.log", sources)
        self.assertIn("legacy/environment-snapshots/windows-python", destinations)
        self.assertIn("legacy/environment-snapshots/wsl-rocm", destinations)
        self.assertGreater(len(move_map["forbidden_residue_patterns"]), 20)

        payload = json.loads(MOVE_MAP_PATH.read_text(encoding="utf-8"))
        self.assertTrue(all(entry.get("category") for entry in payload["entries"]))

    def test_compatibility_root_list_does_not_double_count_nested_destinations(self) -> None:
        selected = remove_overlapping_paths(move_map_paths("artifact"))
        self.assertIn("runs", selected)
        self.assertNotIn("runs/legacy-policy-runs", selected)
        self.assertIn("datasets", selected)
        self.assertNotIn("datasets/skada_clean", selected)

    def test_cross_platform_relative_validation_rejects_all_anchored_spellings(self) -> None:
        for invalid in (
            "/etc/passwd",
            r"\rooted-on-current-drive",
            r"\\server\share\path",
            r"C:\absolute",
            "C:drive-relative",
            "../escape",
            "nested/../escape",
            ".",
            "nested//ambiguous",
            "trailing/",
        ):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                _safe_relative(invalid, label="test path")
        self.assertEqual(_safe_relative(r"nested\portable\file", label="test path"), "nested/portable/file")


class InventoryTests(unittest.TestCase):
    def _make_comparable_trees(self, base: Path) -> tuple[Path, Path]:
        source = base / "source"
        artifact = base / "artifact"
        (source / "datasets").mkdir(parents=True)
        (source / "datasets" / "base.txt").write_text("base", encoding="utf-8")
        (source / "packages" / "rl-agent" / "logs_attention").mkdir(parents=True)
        (source / "packages" / "rl-agent" / "logs_attention" / "child.txt").write_text(
            "child", encoding="utf-8"
        )
        (artifact / "datasets").mkdir(parents=True)
        (artifact / "datasets" / "base.txt").write_text("base", encoding="utf-8")
        (artifact / "runs" / "attention-ppo-archive").mkdir(parents=True)
        (artifact / "runs" / "attention-ppo-archive" / "child.txt").write_text(
            "child", encoding="utf-8"
        )
        try:
            os.symlink("base.txt", source / "datasets" / "base-link")
            os.symlink("base.txt", artifact / "datasets" / "base-link")
        except OSError:
            # Windows CI may not grant CreateSymbolicLink. Reparse handling is
            # still covered below using a synthetic lstat record.
            pass
        return source, artifact

    def test_source_and_artifact_layouts_are_directly_comparable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source_root, artifact_root = self._make_comparable_trees(Path(temporary))
            before = build_inventory(root=source_root, layout="source", hash_limit=1024)
            after = build_inventory(root=artifact_root, layout="artifact", hash_limit=1024)
            comparison = compare_inventories(before, after)

        self.assertEqual(before["schema_version"], INVENTORY_SCHEMA_VERSION)
        self.assertTrue(before["complete"])
        self.assertTrue(after["complete"])
        self.assertEqual(comparison["status"], "ok", comparison["problems"])
        # The canonical runs parent exists only as scaffolding around another
        # mapping; its reserved child namespace must not be double counted.
        runs_parent = next(
            entry for entry in after["entries"] if entry["destination_relative_path"] == "runs"
        )
        self.assertTrue(runs_parent["exists"])
        self.assertEqual(runs_parent["files"], 0)

    def test_content_hash_mismatch_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source_root, artifact_root = self._make_comparable_trees(Path(temporary))
            before = build_inventory(root=source_root, layout="source", hash_limit=1024)
            (artifact_root / "datasets" / "base.txt").write_text("changed", encoding="utf-8")
            after = build_inventory(root=artifact_root, layout="artifact", hash_limit=1024)
            comparison = compare_inventories(before, after)

        self.assertEqual(comparison["status"], "mismatch")
        self.assertTrue(any("datasets -> datasets" in problem for problem in comparison["problems"]))

    def test_compare_cli_writes_machine_readable_success_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            source_root, artifact_root = self._make_comparable_trees(base)
            before = build_inventory(root=source_root, layout="source", hash_limit=1024)
            after = build_inventory(root=artifact_root, layout="artifact", hash_limit=1024)
            before_path = base / "before.json"
            after_path = base / "after.json"
            comparison_path = base / "comparison.json"
            before_path.write_text(json.dumps(before), encoding="utf-8")
            after_path.write_text(json.dumps(after), encoding="utf-8")
            with redirect_stdout(io.StringIO()):
                exit_code = inventory_main(
                    ["--compare", str(before_path), str(after_path), "--output", str(comparison_path)]
                )
                with self.assertRaises(ValueError):
                    inventory_main(
                        ["--compare", str(before_path), str(after_path), "--output", str(before_path)]
                    )
            comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
        self.assertEqual(exit_code, 0)
        self.assertEqual(comparison["status"], "ok")

    def test_scandir_errors_are_recorded_and_mark_inventory_incomplete(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "source"
            (root / "datasets").mkdir(parents=True)
            with patch("tools.artifacts.inventory.os.scandir", side_effect=PermissionError("denied")):
                result = build_inventory(root=root, layout="source", hash_limit=0)

        self.assertFalse(result["complete"])
        self.assertGreater(result["totals"]["errors"], 0)
        dataset = next(entry for entry in result["entries"] if entry["source_relative_path"] == "datasets")
        self.assertTrue(any("PermissionError" in error for error in dataset["errors"]))

    def test_windows_reparse_attribute_wins_over_regular_file_mode(self) -> None:
        metadata = SimpleNamespace(
            st_mode=stat.S_IFREG | 0o644,
            st_file_attributes=getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400),
        )
        self.assertEqual(_entry_kind(metadata), "reparse")

    def test_inventory_root_symlink_is_rejected_instead_of_followed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            physical = base / "physical"
            physical.mkdir()
            linked = base / "linked"
            try:
                os.symlink(physical, linked, target_is_directory=True)
            except OSError as exc:
                self.skipTest(f"cannot create directory symlink: {exc}")
            with self.assertRaises(ValueError):
                build_inventory(root=linked, layout="source", hash_limit=0)

    def test_mapped_parent_symlink_is_an_inventory_error(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "source"
            external = Path(temporary) / "external"
            root.mkdir()
            external.mkdir()
            try:
                os.symlink(external, root / "packages", target_is_directory=True)
            except OSError as exc:
                self.skipTest(f"cannot create directory symlink: {exc}")
            result = build_inventory(root=root, layout="source", hash_limit=0)
        self.assertFalse(result["complete"])
        self.assertTrue(
            any(
                "parent contains a link or reparse point" in error
                for entry in result["entries"]
                for error in entry["errors"]
            )
        )

    def test_inventory_output_cannot_be_written_into_or_above_checkout(self) -> None:
        for invalid in (ROOT / "inventory.json", ROOT.parent):
            with self.subTest(path=invalid), self.assertRaises(ValueError):
                _write_json(invalid, {"status": "must-not-write"})

    def test_inventory_evidence_cannot_pollute_the_tree_being_measured(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            with self.assertRaises(ValueError):
                _validate_inventory_output(
                    root / "datasets" / "inventory.json",
                    inventory_root=root,
                    layout="artifact",
                )
            with self.assertRaises(ValueError):
                _validate_inventory_output(root / "inside.json", inventory_root=root, layout="source")
            _validate_inventory_output(root / ".sts2-post.json", inventory_root=root, layout="artifact")

    def test_move_script_uses_exact_move_apis_and_post_residue_gate(self) -> None:
        script = MOVE_MAP_PATH.with_name("move_to_artifact_root.ps1").read_text(encoding="utf-8-sig")
        self.assertIn("[IO.Directory]::Move", script)
        self.assertIn("[IO.File]::Move", script)
        self.assertIn("Exact destination appeared before move", script)
        self.assertIn("Post-move forbidden runtime residue remains", script)
        self.assertNotIn("Move-Item -LiteralPath $move.Source", script)


class PowerShellDryRunTests(unittest.TestCase):
    def _fixture(self, base: Path, entries: list[dict[str, str]]) -> tuple[Path, Path]:
        repository = base / "repo"
        tool_dir = repository / "tools" / "artifacts"
        tool_dir.mkdir(parents=True)
        shutil.copy2(MOVE_MAP_PATH.with_name("move_to_artifact_root.ps1"), tool_dir)
        shutil.copy2(MOVE_MAP_PATH.with_name("inventory.py"), tool_dir)
        shutil.copy2(MOVE_MAP_PATH.with_name("path_policy.py"), tool_dir)
        (tool_dir / "move-map.json").write_text(
            json.dumps(
                {
                    "schema_version": "1.0.0",
                    "entries": entries,
                    "forbidden_residue_patterns": [],
                }
            ),
            encoding="utf-8",
        )
        return repository, tool_dir / "move_to_artifact_root.ps1"

    def _pwsh(self) -> str:
        executable = shutil.which("pwsh")
        if executable is None:
            self.skipTest("PowerShell Core is unavailable")
        return executable

    def test_parent_child_destination_namespace_collision_fails_before_move(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            entries = [
                {"source": "parent", "destination": "runs", "category": "logs"},
                {"source": "child", "destination": "runs/reserved", "category": "logs"},
            ]
            repository, script = self._fixture(base, entries)
            (repository / "parent" / "reserved").mkdir(parents=True)
            (repository / "child").mkdir()
            result = subprocess.run(
                [
                    self._pwsh(),
                    "-NoProfile",
                    "-File",
                    str(script),
                    "-ArtifactRoot",
                    str(base / "artifact"),
                    "-Mode",
                    "DryRun",
                ],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
            )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("reserved child destination namespace", result.stdout + result.stderr)

    def test_reparse_component_in_target_path_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            _repository, script = self._fixture(base, [])
            physical = base / "physical"
            physical.mkdir()
            link = base / "linked-target"
            try:
                os.symlink(physical, link, target_is_directory=True)
            except OSError as exc:
                self.skipTest(f"cannot create directory symlink: {exc}")
            result = subprocess.run(
                [
                    self._pwsh(),
                    "-NoProfile",
                    "-File",
                    str(script),
                    "-ArtifactRoot",
                    str(link / "runtime"),
                    "-Mode",
                    "DryRun",
                ],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
            )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("reparse component", result.stdout + result.stderr)

    def test_inventory_python_cannot_be_moved_out_from_under_itself(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            entries = [
                {"source": "venv", "destination": "legacy/venv", "category": "environment"}
            ]
            repository, script = self._fixture(base, entries)
            fake_python = repository / "venv" / "python.exe"
            fake_python.parent.mkdir()
            fake_python.write_bytes(b"not executed")
            result = subprocess.run(
                [
                    self._pwsh(),
                    "-NoProfile",
                    "-File",
                    str(script),
                    "-ArtifactRoot",
                    str(base / "artifact"),
                    "-Mode",
                    "DryRun",
                    "-PythonExe",
                    str(fake_python),
                ],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
            )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Inventory Python is inside a source", result.stdout + result.stderr)

    def test_resume_accepts_only_unambiguous_journal_state_and_reverifies(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            entries = [{"source": "payload", "destination": "runs/payload", "category": "logs"}]
            repository, script = self._fixture(base, entries)
            payload = repository / "payload"
            payload.mkdir()
            (payload / "event.json").write_text('{"event":"ok"}\n', encoding="utf-8")
            artifact = base / "artifact"
            command = [
                self._pwsh(),
                "-NoProfile",
                "-File",
                str(script),
                "-ArtifactRoot",
                str(artifact),
                "-SkipUserEnvironmentUpdate",
            ]
            execute = subprocess.run(
                [*command, "-Mode", "Execute"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
            )
            self.assertEqual(execute.returncode, 0, execute.stdout + execute.stderr)

            journal_path = artifact / ".sts2-source-externalization.json"
            journal = json.loads(journal_path.read_text(encoding="utf-8"))
            journal["status"] = "moving"
            journal["completed_at_utc"] = None
            journal["entries"][0]["status"] = "planned"
            journal["entries"][0]["destination_snapshot"] = None
            journal_path.write_text(json.dumps(journal), encoding="utf-8")
            (artifact / ".sts2-post-move-inventory.json").unlink()
            (artifact / ".sts2-inventory-comparison.json").unlink()

            resume = subprocess.run(
                [*command, "-Mode", "Resume"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
            )
            self.assertEqual(resume.returncode, 0, resume.stdout + resume.stderr)
            completed = json.loads(journal_path.read_text(encoding="utf-8"))
            self.assertEqual(completed["status"], "complete")
            self.assertEqual(completed["entries"][0]["status"], "verified")
            comparison = json.loads(
                (artifact / ".sts2-inventory-comparison.json").read_text(encoding="utf-8")
            )
            self.assertEqual(comparison["status"], "ok")


class PathPolicyTests(unittest.TestCase):
    def test_artifact_path_rejects_posix_unc_drive_and_traversal_spellings(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary).resolve()
            source = base / "checkout"
            artifacts = base / "artifacts"
            self.assertEqual(
                resolve_artifact_path("runs/one", root=artifacts, source_root=source),
                artifacts / "runs" / "one",
            )
            for invalid in (
                "/tmp/out",
                r"\\server\share\out",
                r"C:\out",
                "C:out",
                "../out",
                ".",
                "nested//out",
            ):
                with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                    resolve_artifact_path(invalid, root=artifacts, source_root=source)

    def test_root_source_descendant_ancestor_and_filesystem_root_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary).resolve()
            source = base / "checkout"
            for invalid in (source, source / "runtime", base, Path(base.anchor)):
                with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                    canonical_artifact_root(
                        environ={"STS2_ARTIFACT_ROOT": str(invalid)},
                        source_root=source,
                    )

    def test_absolute_output_cannot_bypass_artifact_root_but_external_input_can(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary).resolve()
            source = base / "checkout"
            artifacts = base / "artifacts"
            elsewhere = base / "read-only-input"
            with self.assertRaises(ValueError):
                resolve_artifact_path(elsewhere, root=artifacts, source_root=source)
            self.assertEqual(
                resolve_external_input_path(elsewhere, root=artifacts, source_root=source),
                elsewhere,
            )


class ThirdPartyRestorePathTests(unittest.TestCase):
    def test_default_is_external_and_write_destination_is_confined(self) -> None:
        lock = {"destination": "dependencies/sts2-ai"}
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary).resolve()
            external = base / "artifacts"
            outside = base / "outside"
            with patch.dict(os.environ, {"STS2_ARTIFACT_ROOT": str(external)}, clear=False):
                self.assertEqual(artifact_root(), external)
                self.assertEqual(
                    resolve_destination(None, lock, verify_only=False),
                    external / "dependencies" / "sts2-ai",
                )
                with self.assertRaises(SystemExit):
                    resolve_destination(str(outside), lock, verify_only=False)
                with self.assertRaises(SystemExit):
                    resolve_destination(str(ROOT / "third_party" / "sts2-ai"), lock, verify_only=False)

    def test_verify_only_requires_explicit_absolute_path_and_allows_old_checkout(self) -> None:
        lock = {"destination": "dependencies/sts2-ai"}
        with tempfile.TemporaryDirectory() as temporary:
            external = Path(temporary).resolve() / "artifacts"
            with patch.dict(os.environ, {"STS2_ARTIFACT_ROOT": str(external)}, clear=False):
                with self.assertRaises(SystemExit):
                    resolve_destination("third_party/sts2-ai", lock, verify_only=True)
                self.assertEqual(
                    resolve_destination(str(ROOT / "third_party" / "sts2-ai"), lock, verify_only=True),
                    (ROOT / "third_party" / "sts2-ai").resolve(strict=False),
                )

    def test_lock_destination_cannot_escape_artifact_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            external = Path(temporary).resolve() / "artifacts"
            with patch.dict(os.environ, {"STS2_ARTIFACT_ROOT": str(external)}, clear=False):
                with self.assertRaises(SystemExit):
                    resolve_destination(None, {"destination": "../escape"}, verify_only=False)

    def test_relative_artifact_root_is_rejected(self) -> None:
        with patch.dict(os.environ, {"STS2_ARTIFACT_ROOT": "relative-artifacts"}, clear=False):
            with self.assertRaises(SystemExit):
                artifact_root()


if __name__ == "__main__":
    unittest.main()
