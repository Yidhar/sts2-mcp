"""Check that direct runtime/tool dependencies are exact and represented in locks."""

from __future__ import annotations

import json
import re
import sys
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EXACT_PYTHON = re.compile(r"^([A-Za-z0-9_.-]+)(?:\[[^]]+\])?==([^;\s]+)$")
HASHED_PYTHON = re.compile(
    r"^([A-Za-z0-9_.-]+)(?:\[[^]]+\])?==([^;\s]+)\s+--hash=sha256:([0-9a-f]{64})$"
)
EXACT_NODE = re.compile(r"^\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.-]+)?$")
ROCM_ARTIFACT = re.compile(
    r"^# ([A-Za-z0-9_.-]+)==([^ ]+) :: (https://\S+) :: sha256=([0-9a-f]{64})$",
    re.MULTILINE,
)


def normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def python_requirements(path: Path, seen: set[Path] | None = None) -> dict[str, str]:
    seen = seen or set()
    path = path.resolve()
    if path in seen:
        return {}
    seen.add(path)
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line.startswith(("#", "--")):
            continue
        if line.startswith(("-r ", "--requirement ")):
            values.update(python_requirements(path.parent / line.split(maxsplit=1)[1], seen))
            continue
        match = HASHED_PYTHON.fullmatch(line) or EXACT_PYTHON.fullmatch(line)
        if match:
            values[normalize(match.group(1))] = match.group(2)
    return values


def validate_python_hash_lock(
    path: Path,
    expected_profile: str,
    errors: list[str],
) -> None:
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    markers = [line.removeprefix("# artifact-profile:").strip() for line in lines if line.startswith("# artifact-profile:")]
    if markers != [expected_profile]:
        errors.append(f"{path.name} must declare exactly '# artifact-profile: {expected_profile}'")
    if sum(line.strip() == "--require-hashes" for line in lines) != 1:
        errors.append(f"{path.name} must enable --require-hashes exactly once")
    requirement_count = 0
    hashes: set[str] = set()
    for line_number, raw in enumerate(lines, 1):
        line = raw.strip()
        if not line or line.startswith(("#", "-r ", "--requirement ", "--")):
            continue
        match = HASHED_PYTHON.fullmatch(line)
        if match is None:
            errors.append(
                f"{path.name}:{line_number} must be an exact name==version pin with one sha256 hash"
            )
            continue
        requirement_count += 1
        hashes.add(match.group(3))
    if requirement_count == 0:
        errors.append(f"{path.name} contains no hashed Python requirements")
    if len(hashes) != requirement_count:
        errors.append(f"{path.name} reuses a wheel SHA-256 across distinct requirements")


def exact_python_specs(specs: list[str], label: str, errors: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for spec in specs:
        match = EXACT_PYTHON.fullmatch(spec)
        if not match:
            errors.append(f"{label} dependency is not exact: {spec}")
            continue
        result[normalize(match.group(1))] = match.group(2)
    return result


def require_subset(
    expected: dict[str, str], actual: dict[str, str], label: str, errors: list[str]
) -> None:
    for name, version in expected.items():
        if actual.get(name) != version:
            errors.append(f"{label} missing {name}=={version} (found {actual.get(name)!r})")


def main() -> int:
    errors: list[str] = []
    rl_root = ROOT / "packages" / "rl-agent"
    with (rl_root / "pyproject.toml").open("rb") as stream:
        project = tomllib.load(stream)
    base = exact_python_specs(project["project"]["dependencies"], "Python runtime", errors)
    build = exact_python_specs(project["build-system"]["requires"], "Python build", errors)
    extras = project["project"].get("optional-dependencies", {})
    train = exact_python_specs(extras.get("train", []), "Python train", errors)
    test = exact_python_specs(extras.get("test", []), "Python test", errors)
    dev = exact_python_specs(extras.get("dev", []), "Python dev", errors)
    text = exact_python_specs(extras.get("text", []), "Python text", errors)

    runtime_lock = python_requirements(rl_root / "requirements.lock")
    dev_lock = python_requirements(rl_root / "requirements-dev.lock")
    text_lock = python_requirements(rl_root / "requirements-text.lock")
    wsl_lock_path = rl_root / "requirements-wsl-rocm.txt"
    for filename, profile in (
        ("requirements-bootstrap.lock", "any-py3"),
        ("requirements.lock", "windows-cp313"),
        ("requirements-dev.lock", "windows-cp313"),
        ("requirements-text.lock", "windows-cp313"),
        ("requirements-wsl-rocm.txt", "wsl-cp312-rocm-7.2.1"),
    ):
        validate_python_hash_lock(rl_root / filename, profile, errors)
    wsl_lock = python_requirements(wsl_lock_path)
    require_subset({**base, **build, **train}, runtime_lock, "requirements.lock", errors)
    require_subset({**base, **build, **train, **test, **dev}, dev_lock, "requirements-dev.lock", errors)
    require_subset({**base, **text}, text_lock, "requirements-text.lock", errors)
    wsl_train = {name: version for name, version in train.items() if name != "torch"}
    require_subset(
        {**base, **build, **wsl_train, **text},
        wsl_lock,
        "requirements-wsl-rocm.txt",
        errors,
    )

    wsl_text = wsl_lock_path.read_text(encoding="utf-8-sig")
    rocm_artifacts = {
        normalize(name): (version, url, digest)
        for name, version, url, digest in ROCM_ARTIFACT.findall(wsl_text)
    }
    expected_rocm_artifacts = {"torch", "torchvision", "torchaudio", "triton"}
    if set(rocm_artifacts) != expected_rocm_artifacts:
        errors.append(
            "requirements-wsl-rocm.txt must lock exactly torch, torchvision, torchaudio, and triton wheel artifacts"
        )
    bootstrap = (rl_root / "scripts" / "bootstrap_wsl_rocm.sh").read_text(encoding="utf-8-sig")
    shell_urls = dict(
        re.findall(r'^(TORCH(?:VISION|AUDIO)?|TRITON)_WHL_URL="\$\{[^:]+:-([^}]+)\}"$', bootstrap, re.MULTILINE)
    )
    shell_names = {
        "TORCH": "torch",
        "TORCHVISION": "torchvision",
        "TORCHAUDIO": "torchaudio",
        "TRITON": "triton",
    }
    for shell_name, package_name in shell_names.items():
        artifact = rocm_artifacts.get(package_name)
        if artifact is not None and shell_urls.get(shell_name) != artifact[1]:
            errors.append(f"bootstrap_wsl_rocm.sh {shell_name}_WHL_URL differs from the ROCm artifact lock")
        expected_hash = artifact[2] if artifact is not None else None
        hash_match = re.search(
            rf'^{shell_name}_WHL_SHA256="\$\{{[^:]+:-([0-9a-f]{{64}})\}}"$',
            bootstrap,
            re.MULTILINE,
        )
        if expected_hash is not None and (hash_match is None or hash_match.group(1) != expected_hash):
            errors.append(
                f"bootstrap_wsl_rocm.sh {shell_name}_WHL_SHA256 differs from the ROCm artifact lock"
            )
    expected_torch_distribution = re.search(
        r'^EXPECTED_TORCH_DISTRIBUTION_VERSION="\$\{[^:]+:-([^}]+)\}"$',
        bootstrap,
        re.MULTILINE,
    )
    locked_torch = rocm_artifacts.get("torch")
    if locked_torch is not None and (
        expected_torch_distribution is None
        or expected_torch_distribution.group(1) != locked_torch[0]
    ):
        errors.append(
            "bootstrap_wsl_rocm.sh expected torch distribution version differs from the ROCm artifact lock"
        )
    runtime_lock = re.search(r"^# torch-runtime-version: (\S+)$", wsl_text, re.MULTILINE)
    expected_torch_runtime = re.search(
        r'^EXPECTED_TORCH_RUNTIME_VERSION="\$\{[^:]+:-([^}]+)\}"$',
        bootstrap,
        re.MULTILINE,
    )
    if runtime_lock is None or expected_torch_runtime is None or (
        runtime_lock.group(1) != expected_torch_runtime.group(1)
    ):
        errors.append(
            "bootstrap_wsl_rocm.sh expected torch runtime version differs from the ROCm runtime lock"
        )

    mcp_root = ROOT / "packages" / "mcp-server"
    package = json.loads((mcp_root / "package.json").read_text(encoding="utf-8"))
    lock = json.loads((mcp_root / "package-lock.json").read_text(encoding="utf-8"))
    lock_root = lock["packages"][""]
    if lock_root.get("version") != package.get("version"):
        errors.append("MCP package-lock root version does not match package.json")
    for section in ("dependencies", "devDependencies"):
        declared = package.get(section, {})
        locked_declared = lock_root.get(section, {})
        for name, version in declared.items():
            if not EXACT_NODE.fullmatch(version):
                errors.append(f"MCP {section} dependency is not exact: {name}@{version}")
            if locked_declared.get(name) != version:
                errors.append(f"MCP package-lock declaration differs: {name}@{version}")
    if not re.fullmatch(r"npm@\d+\.\d+\.\d+", str(package.get("packageManager") or "")):
        errors.append("MCP packageManager must pin an exact npm version")
    if not re.fullmatch(r"\d+\.\d+\.\d+", (ROOT / ".nvmrc").read_text().strip()):
        errors.append(".nvmrc must pin an exact Node patch version")
    if not re.fullmatch(r"\d+\.\d+\.\d+", (ROOT / ".python-version").read_text().strip()):
        errors.append(".python-version must pin an exact Python patch version")

    global_json = json.loads((ROOT / "global.json").read_text(encoding="utf-8"))
    if global_json.get("sdk", {}).get("rollForward") != "disable":
        errors.append("global.json must disable SDK roll-forward")

    workflow_use = re.compile(r"^\s*-\s+uses:\s+([^\s#]+)", re.MULTILINE)
    for workflow in sorted((ROOT / ".github" / "workflows").glob("*.yml")):
        for reference in workflow_use.findall(workflow.read_text(encoding="utf-8-sig")):
            if not re.fullmatch(r"actions/[A-Za-z0-9_.-]+@[0-9a-f]{40}", reference):
                errors.append(
                    f"{workflow.relative_to(ROOT).as_posix()} action is not pinned to a full commit SHA: {reference}"
                )

    if errors:
        print("dependency lock validation failed:", file=sys.stderr)
        print("\n".join(f"- {error}" for error in errors), file=sys.stderr)
        return 1
    print("dependency locks match exact direct declarations")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
