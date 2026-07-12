from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).with_name("export_catalog.py")
SPEC = importlib.util.spec_from_file_location("sts2_catalog_export", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class CatalogExportTests(unittest.TestCase):
    def test_exports_verified_canonical_package(self) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as temporary:
            artifact_root = Path(temporary) / "artifacts"
            output = artifact_root / "catalog"
            result = MODULE.export_catalog(
                repo_root / "game-data",
                output,
                artifact_root_path=artifact_root,
            )
            self.assertGreater(len(result["files"]), 0)
            exported = json.loads((output / "catalog-export.manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(exported["source_manifest_sha256"], MODULE.sha256_file(repo_root / "game-data" / "manifest.json"))
            for entry in exported["files"]:
                self.assertEqual(MODULE.sha256_file(output / entry["path"]), entry["sha256"])

    def test_export_manifest_is_deterministic(self) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as temporary:
            artifact_root = Path(temporary) / "artifacts"
            first = artifact_root / "first"
            second = artifact_root / "second"
            MODULE.export_catalog(
                repo_root / "game-data",
                first,
                artifact_root_path=artifact_root,
            )
            MODULE.export_catalog(
                repo_root / "game-data",
                second,
                artifact_root_path=artifact_root,
            )
            self.assertEqual(
                (first / "catalog-export.manifest.json").read_bytes(),
                (second / "catalog-export.manifest.json").read_bytes(),
            )

    def test_rejects_destination_inside_source(self) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        with self.assertRaises(ValueError):
            MODULE.export_catalog(
                repo_root / "game-data",
                repo_root / "game-data" / "export",
                artifact_root_path=repo_root / "game-data",
            )


if __name__ == "__main__":
    unittest.main()
