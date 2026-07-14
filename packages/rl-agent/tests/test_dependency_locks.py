from __future__ import annotations

import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _pins(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith(("#", "-")):
            continue
        if "==" not in line:
            continue
        match = re.fullmatch(
            r"([A-Za-z0-9_.-]+)==([^\s]+) --hash=sha256:([0-9a-f]{64})",
            line,
        )
        assert match is not None, f"unhashed or malformed lock line: {path.name}: {line}"
        name, version, _digest = match.groups()
        result[name.casefold().replace("_", "-")] = version
    return result


def _exact_requirement(requirement: str) -> tuple[str, str]:
    assert "==" in requirement
    assert not re.search(r"(^|[^=])[<>~!]=?", requirement)
    name, version = requirement.split("==", 1)
    assert name.strip()
    assert version.strip()
    return name.casefold().replace("_", "-"), version


def test_package_version_matches_project_metadata() -> None:
    import sts2_rl

    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert sts2_rl.__version__ == pyproject["project"]["version"] == "0.4.0"


def test_project_direct_dependencies_and_build_tools_are_exact() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    for requirement in pyproject["build-system"]["requires"]:
        _exact_requirement(requirement)
    for requirement in pyproject["project"]["dependencies"]:
        _exact_requirement(requirement)
    for requirements in pyproject["project"]["optional-dependencies"].values():
        for requirement in requirements:
            _exact_requirement(requirement)


def test_cpu_and_wsl_locks_match_project_direct_metadata() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    direct = dict(_exact_requirement(item) for item in pyproject["project"]["dependencies"])
    train = dict(
        _exact_requirement(item)
        for item in pyproject["project"]["optional-dependencies"]["train"]
    )
    runtime = _pins(ROOT / "requirements.lock")
    wsl = _pins(ROOT / "requirements-wsl-rocm.txt")

    for name, version in direct.items():
        assert runtime[name] == version
        assert wsl[name] == version
    for name, version in train.items():
        assert runtime[name] == version

    assert wsl["numpy"] == "2.4.3"
    assert not (ROOT / "requirements-lock.txt").exists()


def test_locks_have_no_deleted_sb3_stack_and_rocm_artifact_is_explicit() -> None:
    all_lock_text = "\n".join(
        (ROOT / name).read_text(encoding="utf-8")
        for name in (
            "requirements-bootstrap.lock",
            "requirements.lock",
            "requirements-dev.lock",
            "requirements-wsl-rocm.txt",
        )
    ).casefold()
    assert "stable-baselines" not in all_lock_text
    assert "sb3-contrib" not in all_lock_text

    wsl = (ROOT / "requirements-wsl-rocm.txt").read_text(encoding="utf-8")
    bootstrap = (ROOT / "scripts" / "bootstrap_wsl_rocm.sh").read_text(encoding="utf-8")
    verifier = (ROOT / "scripts" / "verify_rocm_stack.py").read_text(encoding="utf-8")
    expected_distribution = "2.9.1+rocm7.2.1.lw.gitff65f5bc"
    expected_runtime = "2.9.1+rocm7.2.1.gitff65f5bc"
    assert f"torch=={expected_distribution}" in wsl
    assert f"# torch-runtime-version: {expected_runtime}" in wsl
    assert f"EXPECTED_TORCH_DISTRIBUTION_VERSION:-{expected_distribution}" in bootstrap
    assert f"EXPECTED_TORCH_RUNTIME_VERSION:-{expected_runtime}" in bootstrap
    assert "verify_rocm_stack.py" in bootstrap
    assert 'importlib.metadata.version("torch")' in verifier
    assert "download_verified" in bootstrap
    assert 'WHEEL_CACHE_DIR="${ROCM_WHEEL_CACHE_DIR:-$ARTIFACT_ROOT/downloads/rocm-$ROCM_VERSION}"' in bootstrap
    assert "Reusing verified $label" in bootstrap
    assert "python -m pip install --no-deps" in bootstrap
    assert "python -m pip install -r '$REQ_FILE'" in bootstrap
    assert "python -m pip install -e '$REPO_ROOT' --no-deps --no-build-isolation" in bootstrap
    assert "python -m pip check" in bootstrap
    assert wsl.count(" :: sha256=") == 4
    for variable in (
        "TORCH_WHL_SHA256",
        "TORCHVISION_WHL_SHA256",
        "TORCHAUDIO_WHL_SHA256",
        "TRITON_WHL_SHA256",
        "AMDGPU_INSTALL_DEB_SHA256",
    ):
        assert re.search(rf'^{variable}="\$\{{[^:]+:-[0-9a-f]{{64}}\}}"$', bootstrap, re.MULTILINE)


def test_every_python_lock_enforces_sha256_hashes() -> None:
    profiles = {
        "requirements-bootstrap.lock": "any-py3",
        "requirements.lock": "windows-cp313",
        "requirements-dev.lock": "windows-cp313",
        "requirements-wsl-rocm.txt": "wsl-cp312-rocm-7.2.1",
    }
    for filename, profile in profiles.items():
        text = (ROOT / filename).read_text(encoding="utf-8")
        assert text.count("--require-hashes") == 1
        assert text.count(f"# artifact-profile: {profile}") == 1
        pins = _pins(ROOT / filename)
        assert pins
