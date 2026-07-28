"""Tests for the dependency-free deterministic CycloneDX generator."""

from __future__ import annotations

import base64
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from tools.release import build_sbom


ROOT = Path(__file__).resolve().parents[2]
STS2_AI_RUNTIME_COMMIT = "252bf989feb598c61c20267d3979b00bf316edb5"
STS2_AI_RUNTIME_PURL = f"pkg:github/frankqwang/sts2-ai@{STS2_AI_RUNTIME_COMMIT}"
STS2_AI_GAME_DATA_SNAPSHOT_COMMIT = "459f30873eac374c160f0b409d2d8167e7eead7c"


def _properties(component: dict[str, object]) -> dict[str, str]:
    return {
        str(item["name"]): str(item["value"])
        for item in component.get("properties", [])  # type: ignore[union-attr]
    }


class RepositorySbomTests(unittest.TestCase):
    def test_current_repository_is_deterministic_and_has_core_ecosystems(self) -> None:
        first = build_sbom.render_sbom(ROOT)
        second = build_sbom.render_sbom(ROOT)
        self.assertEqual(first, second)
        self.assertTrue(first.endswith(b"\n"))

        bom = json.loads(first)
        self.assertEqual(bom["bomFormat"], "CycloneDX")
        self.assertEqual(bom["specVersion"], "1.6")
        components = {component["purl"]: component for component in bom["components"]}
        self.assertEqual(list(components), sorted(components))

        release_manifest = json.loads(
            (ROOT / "release-manifest.json").read_text(encoding="utf-8")
        )
        bridge_version = release_manifest["components"]["bridge"]
        expected = {
            f"pkg:generic/sts2/bridge@{bridge_version}",
            "pkg:npm/%40modelcontextprotocol/sdk@1.29.0",
            "pkg:pypi/numpy@2.4.3?download_profile=windows-cp313",
            "pkg:pypi/torch@2.9.1%2Brocm7.2.1.lw.gitff65f5bc?download_profile=wsl-cp312-rocm-7.2.1",
            STS2_AI_RUNTIME_PURL,
        }
        self.assertTrue(expected.issubset(components))

        sdk = components["pkg:npm/%40modelcontextprotocol/sdk@1.29.0"]
        self.assertIn({"license": {"name": "MIT"}}, sdk["licenses"])
        self.assertEqual(sdk["hashes"][0]["alg"], "SHA-512")
        self.assertEqual(len(sdk["hashes"][0]["content"]), 128)

        rocm = components[
            "pkg:pypi/torch@2.9.1%2Brocm7.2.1.lw.gitff65f5bc?download_profile=wsl-cp312-rocm-7.2.1"
        ]
        self.assertEqual(_properties(rocm)["sts2:artifact-kind"], "rocm-wheel")
        self.assertEqual(rocm["externalReferences"][0]["type"], "distribution")
        self.assertEqual(rocm["hashes"][0]["alg"], "SHA-256")

        numpy = components["pkg:pypi/numpy@2.4.3?download_profile=windows-cp313"]
        self.assertEqual(numpy["hashes"][0]["alg"], "SHA-256")
        self.assertEqual(len(numpy["hashes"][0]["content"]), 64)

        third_party = components[STS2_AI_RUNTIME_PURL]
        third_party_properties = _properties(third_party)
        self.assertEqual(
            third_party_properties["sts2:license-status"], "review-required"
        )
        self.assertEqual(third_party_properties["sts2:distribution-allowed"], "false")
        self.assertEqual(
            third_party["hashes"],
            [{"alg": "SHA-1", "content": STS2_AI_RUNTIME_COMMIT}],
        )

        def all_keys(value: object) -> set[str]:
            if isinstance(value, dict):
                return set(value) | set().union(
                    *(all_keys(item) for item in value.values())
                )
            if isinstance(value, list):
                return (
                    set().union(*(all_keys(item) for item in value)) if value else set()
                )
            return set()

        self.assertNotIn("timestamp", all_keys(bom))
        self.assertNotIn(str(ROOT), first.decode("utf-8"))

    def test_game_data_snapshot_provenance_is_independent_of_runtime_pin(self) -> None:
        """A simulator re-pin must not silently relabel existing generated game data."""

        game_data_manifest = json.loads(
            (ROOT / "game-data" / "manifest.json").read_text(encoding="utf-8")
        )
        runtime_lock = json.loads(
            (ROOT / "third_party" / "sts2-ai.lock.json").read_text(encoding="utf-8")
        )

        self.assertEqual(runtime_lock["commit"], STS2_AI_RUNTIME_COMMIT)
        self.assertEqual(
            game_data_manifest["upstream_sts2_ai_commit"],
            STS2_AI_GAME_DATA_SNAPSHOT_COMMIT,
        )
        self.assertNotEqual(
            game_data_manifest["upstream_sts2_ai_commit"], runtime_lock["commit"]
        )

    def test_output_option_writes_identical_bytes(self) -> None:
        with (
            tempfile.TemporaryDirectory() as temporary_directory,
            redirect_stdout(StringIO()),
        ):
            first_path = Path(temporary_directory) / "first.json"
            second_path = Path(temporary_directory) / "second.json"
            self.assertEqual(build_sbom.main(["--output", str(first_path)]), 0)
            self.assertEqual(build_sbom.main(["--output", str(second_path)]), 0)
            self.assertEqual(first_path.read_bytes(), second_path.read_bytes())


class FailClosedTests(unittest.TestCase):
    def _fixture(self, root: Path) -> None:
        (root / "packages" / "mcp-server").mkdir(parents=True)
        (root / "packages" / "rl-agent").mkdir(parents=True)
        (root / "third_party").mkdir()
        (root / "release-manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": "1.0.0",
                    "release_channel": "test",
                    "components": {"bridge": "1.0.0"},
                }
            ),
            encoding="utf-8",
        )
        integrity = "sha512-" + base64.b64encode(b"x" * 64).decode("ascii")
        package_lock = {
            "name": "fixture-mcp",
            "version": "1.0.0",
            "lockfileVersion": 3,
            "packages": {
                "": {"name": "fixture-mcp", "version": "1.0.0", "license": "MIT"},
                "node_modules/example": {
                    "version": "2.0.0",
                    "integrity": integrity,
                    "license": "MIT",
                    "resolved": "https://registry.example/example-2.0.0.tgz",
                },
            },
        }
        (root / "packages" / "mcp-server" / "package-lock.json").write_text(
            json.dumps(package_lock), encoding="utf-8"
        )
        (root / "packages" / "rl-agent" / "requirements.lock").write_text(
            "# artifact-profile: windows-cp313\n"
            "--require-hashes\n"
            f"example-python==3.0.0 --hash=sha256:{'1' * 64}\n",
            encoding="utf-8",
        )
        (root / "packages" / "rl-agent" / "requirements-wsl-rocm.txt").write_text(
            "# External ROCm wheel artifact lock:\n"
            f"# torch==2.0.0+rocm :: https://repo.example/torch-2.0.0%2Brocm-cp312-cp312-linux_x86_64.whl :: sha256={'2' * 64}\n"
            "# artifact-profile: wsl-cp312-rocm-test\n"
            "--require-hashes\n"
            f"numpy==2.0.0 --hash=sha256:{'3' * 64}\n",
            encoding="utf-8",
        )
        (root / "third_party" / "vendor.lock.json").write_text(
            json.dumps(
                {
                    "name": "vendor",
                    "url": "https://github.com/example/vendor.git",
                    "commit": "a" * 40,
                    "tree": "b" * 40,
                    "destination": "third_party/vendor",
                    "license_status": "review-required",
                    "distribution_allowed": False,
                }
            ),
            encoding="utf-8",
        )

    def test_duplicate_python_requirement_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            self._fixture(root)
            (root / "packages" / "rl-agent" / "requirements.lock").write_text(
                "# artifact-profile: windows-cp313\n"
                "--require-hashes\n"
                f"Example_Python==3.0.0 --hash=sha256:{'1' * 64}\n"
                f"example-python==3.0.0 --hash=sha256:{'4' * 64}\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                build_sbom.SbomError, "duplicate Python requirement"
            ):
                build_sbom.build_sbom(root)

    def test_missing_python_or_rocm_artifact_hash_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            self._fixture(root)
            (root / "packages" / "rl-agent" / "requirements.lock").write_text(
                "# artifact-profile: windows-cp313\n"
                "--require-hashes\n"
                "example-python==3.0.0\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                build_sbom.SbomError, "exact name==version pin"
            ):
                build_sbom.build_sbom(root)

            (root / "packages" / "rl-agent" / "requirements.lock").write_text(
                "# artifact-profile: windows-cp313\n"
                "--require-hashes\n"
                f"example-python==3.0.0 --hash=sha256:{'1' * 64}\n",
                encoding="utf-8",
            )
            (root / "packages" / "rl-agent" / "requirements-wsl-rocm.txt").write_text(
                "# External ROCm wheel artifact lock:\n"
                "# torch==2.0.0+rocm :: https://repo.example/torch-2.0.0%2Brocm-cp312-cp312-linux_x86_64.whl\n"
                "# artifact-profile: wsl-cp312-rocm-test\n"
                "--require-hashes\n"
                f"numpy==2.0.0 --hash=sha256:{'3' * 64}\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(build_sbom.SbomError, "requires sha256"):
                build_sbom.build_sbom(root)

    def test_conflicting_duplicate_component_hash_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            self._fixture(root)
            path = root / "packages" / "mcp-server" / "package-lock.json"
            lock = json.loads(path.read_text(encoding="utf-8"))
            lock["packages"]["node_modules/parent/node_modules/example"] = {
                "version": "2.0.0",
                "integrity": "sha512-" + base64.b64encode(b"y" * 64).decode("ascii"),
                "license": "MIT",
                "resolved": "https://registry.example/example-2.0.0.tgz",
            }
            path.write_text(json.dumps(lock), encoding="utf-8")
            with self.assertRaisesRegex(
                build_sbom.SbomError, "conflicting SHA-512 hashes"
            ):
                build_sbom.build_sbom(root)

    def test_missing_third_party_tree_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            self._fixture(root)
            path = root / "third_party" / "vendor.lock.json"
            lock = json.loads(path.read_text(encoding="utf-8"))
            del lock["tree"]
            path.write_text(json.dumps(lock), encoding="utf-8")
            with self.assertRaisesRegex(
                build_sbom.SbomError, "tree must be a non-empty"
            ):
                build_sbom.build_sbom(root)


if __name__ == "__main__":
    unittest.main()
