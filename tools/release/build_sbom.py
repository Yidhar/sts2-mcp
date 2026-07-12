"""Build a deterministic CycloneDX JSON SBOM from repository lock files.

The generator deliberately uses only the Python standard library.  It models
the *locked closure*, not the packages installed on the machine running it, so
the result contains no wall-clock timestamps or host-specific paths.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Sequence
from urllib.parse import quote, unquote, urlparse


ROOT = Path(__file__).resolve().parents[2]
CYCLONEDX_SCHEMA = "https://cyclonedx.org/schema/bom-1.6.schema.json"
GENERATOR_VERSION = "1.0.0"

_VERSION_RE = re.compile(r"^[^\s;#]+$")
_PYTHON_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_REQUIREMENT_RE = re.compile(
    r"^([A-Za-z0-9][A-Za-z0-9._-]*)==([^\s;#]+)\s+--hash=sha256:([0-9a-f]{64})$"
)
_HEX40_RE = re.compile(r"^[0-9a-f]{40}$")
_ROCM_ARTIFACT_RE = re.compile(
    r"^#\s*([A-Za-z0-9][A-Za-z0-9._-]*)==([^\s]+)\s*::\s*"
    r"(https://\S+?)(?:\s+::\s*(sha(?:256|384|512))=([0-9A-Fa-f]+))?\s*$",
    re.IGNORECASE,
)
_HASH_LENGTHS = {"SHA-256": 64, "SHA-384": 96, "SHA-512": 128, "SHA-1": 40}
_SRI_ALGORITHMS = {"sha256": "SHA-256", "sha384": "SHA-384", "sha512": "SHA-512"}


class SbomError(ValueError):
    """Raised when a lock is incomplete, ambiguous, or internally inconsistent."""


def _nonempty_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip() or any(ord(char) < 32 for char in value):
        raise SbomError(f"{field_name} must be a non-empty printable string")
    return value.strip()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SbomError(f"required input is missing: {path.name}") from exc
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SbomError(f"invalid UTF-8 JSON input {path.name}: {exc}") from exc
    if not isinstance(value, dict):
        raise SbomError(f"JSON input must be an object: {path.name}")
    return value


def _logical_path(root: Path, path: Path) -> str:
    try:
        logical = path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise SbomError(f"input escapes repository root: {path}") from exc
    if logical.startswith("../") or logical == "..":
        raise SbomError(f"input escapes repository root: {logical}")
    return logical


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except FileNotFoundError as exc:
        raise SbomError(f"required input is missing: {path.name}") from exc
    return digest.hexdigest()


def _purl_segment(value: str) -> str:
    return quote(value, safe="._~-")


def _python_name(value: str) -> str:
    if not _PYTHON_NAME_RE.fullmatch(value):
        raise SbomError(f"invalid Python distribution name: {value!r}")
    return re.sub(r"[-_.]+", "-", value).lower()


def _python_purl(name: str, version: str, artifact_profile: str | None = None) -> str:
    value = f"pkg:pypi/{_purl_segment(_python_name(name))}@{_purl_segment(version)}"
    if artifact_profile:
        value += f"?download_profile={_purl_segment(artifact_profile)}"
    return value


def _npm_purl(name: str, version: str) -> str:
    if name.startswith("@"):
        parts = name[1:].split("/")
        if len(parts) != 2 or not all(parts):
            raise SbomError(f"invalid scoped npm package name: {name!r}")
        namespace, package_name = parts
        path = f"{_purl_segment('@' + namespace)}/{_purl_segment(package_name)}"
    else:
        if "/" in name or not name:
            raise SbomError(f"invalid npm package name: {name!r}")
        path = _purl_segment(name)
    return f"pkg:npm/{path}@{_purl_segment(version)}"


def _generic_purl(namespace: str, name: str, version: str) -> str:
    return f"pkg:generic/{_purl_segment(namespace)}/{_purl_segment(name)}@{_purl_segment(version)}"


def _github_purl(url: str, commit: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.netloc.lower() != "github.com" or parsed.query or parsed.fragment:
        raise SbomError(f"third-party VCS URL must be a canonical HTTPS GitHub URL: {url!r}")
    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) != 2:
        raise SbomError(f"third-party GitHub URL must identify one repository: {url!r}")
    owner, repository = parts
    if repository.endswith(".git"):
        repository = repository[:-4]
    if not owner or not repository:
        raise SbomError(f"invalid third-party GitHub repository URL: {url!r}")
    return f"pkg:github/{_purl_segment(owner)}/{_purl_segment(repository)}@{commit}"


def _validate_https_url(value: str, field_name: str) -> str:
    url = _nonempty_string(value, field_name)
    parsed = urlparse(url)
    if parsed.scheme != "https" or not parsed.netloc:
        raise SbomError(f"{field_name} must be an absolute HTTPS URL: {url!r}")
    if parsed.username or parsed.password:
        raise SbomError(f"{field_name} must not contain credentials")
    return url


def _integrity_hashes(integrity: Any, context: str) -> dict[str, str]:
    if integrity is None:
        return {}
    value = _nonempty_string(integrity, f"{context}.integrity")
    result: dict[str, str] = {}
    for token in value.split():
        if "-" not in token:
            raise SbomError(f"malformed Subresource Integrity value for {context}")
        raw_algorithm, encoded = token.split("-", 1)
        algorithm = _SRI_ALGORITHMS.get(raw_algorithm.lower())
        if algorithm is None:
            raise SbomError(f"unsupported Subresource Integrity algorithm for {context}: {raw_algorithm}")
        try:
            digest = base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise SbomError(f"malformed base64 integrity digest for {context}") from exc
        content = digest.hex()
        if len(content) != _HASH_LENGTHS[algorithm]:
            raise SbomError(f"incorrect {algorithm} integrity length for {context}")
        previous = result.get(algorithm)
        if previous is not None and previous != content:
            raise SbomError(f"ambiguous duplicate {algorithm} integrity digests for {context}")
        result[algorithm] = content
    return result


@dataclass
class ComponentRecord:
    component_type: str
    name: str
    version: str
    purl: str
    hashes: dict[str, str] = field(default_factory=dict)
    licenses: set[str] = field(default_factory=set)
    references: set[tuple[str, str]] = field(default_factory=set)
    properties: dict[str, set[str]] = field(default_factory=dict)

    def add_property(self, name: str, value: str) -> None:
        self.properties.setdefault(name, set()).add(value)

    def merge(self, other: "ComponentRecord") -> None:
        if (self.component_type, self.name, self.version, self.purl) != (
            other.component_type,
            other.name,
            other.version,
            other.purl,
        ):
            raise SbomError(f"duplicate component identity has conflicting core fields: {self.purl}")
        for algorithm, content in other.hashes.items():
            previous = self.hashes.get(algorithm)
            if previous is not None and previous != content:
                raise SbomError(f"duplicate component has conflicting {algorithm} hashes: {self.purl}")
            self.hashes[algorithm] = content
        if self.licenses and other.licenses and self.licenses != other.licenses:
            raise SbomError(f"duplicate component has conflicting licenses: {self.purl}")
        self.licenses.update(other.licenses)
        self.references.update(other.references)
        for name, values in other.properties.items():
            self.properties.setdefault(name, set()).update(values)

    def as_cyclonedx(self) -> dict[str, Any]:
        component: dict[str, Any] = {
            "bom-ref": self.purl,
            "name": self.name,
            "purl": self.purl,
            "type": self.component_type,
            "version": self.version,
        }
        if self.hashes:
            component["hashes"] = [
                {"alg": algorithm, "content": content}
                for algorithm, content in sorted(self.hashes.items())
            ]
        if self.licenses:
            component["licenses"] = [
                {"license": {"name": license_name}} for license_name in sorted(self.licenses)
            ]
        if self.references:
            component["externalReferences"] = [
                {"type": reference_type, "url": url}
                for reference_type, url in sorted(self.references)
            ]
        if self.properties:
            component["properties"] = [
                {"name": name, "value": ",".join(sorted(values))}
                for name, values in sorted(self.properties.items())
            ]
        return component


class ComponentRegistry:
    def __init__(self) -> None:
        self._records: dict[str, ComponentRecord] = {}

    def add(self, record: ComponentRecord) -> None:
        if not _VERSION_RE.fullmatch(record.version):
            raise SbomError(f"component version is not an exact printable version: {record.version!r}")
        existing = self._records.get(record.purl)
        if existing is None:
            self._records[record.purl] = record
        else:
            existing.merge(record)

    def components(self) -> list[dict[str, Any]]:
        return [self._records[purl].as_cyclonedx() for purl in sorted(self._records)]


def _add_release_components(
    root: Path, registry: ComponentRegistry, input_paths: set[Path]
) -> tuple[str, str]:
    path = root / "release-manifest.json"
    manifest = _read_json(path)
    input_paths.add(path)
    schema_version = _nonempty_string(manifest.get("schema_version"), "release schema_version")
    if not re.fullmatch(r"\d+\.\d+\.\d+", schema_version):
        raise SbomError("release schema_version must be an exact semantic version")
    channel = _nonempty_string(manifest.get("release_channel"), "release release_channel")
    components = manifest.get("components")
    if not isinstance(components, dict) or not components:
        raise SbomError("release manifest components must be a non-empty object")
    source = _logical_path(root, path)
    type_by_name = {"contracts": "library", "game_data": "data"}
    for raw_name, raw_version in sorted(components.items()):
        name = _nonempty_string(raw_name, "release component name")
        version = _nonempty_string(raw_version, f"release component {name} version")
        if not re.fullmatch(r"\d+\.\d+\.\d+", version):
            raise SbomError(f"release component {name} must use an exact semantic version")
        purl = _generic_purl("sts2", name, version)
        record = ComponentRecord(type_by_name.get(name, "application"), name, version, purl)
        record.add_property("sts2:lock-sources", source)
        record.add_property("sts2:release-channel", channel)
        registry.add(record)
    return schema_version, channel


def _npm_name_from_install_path(install_path: str, root_name: str) -> str:
    if install_path == "":
        return root_name
    marker = "node_modules/"
    if marker not in install_path:
        raise SbomError(f"unsupported package-lock install path: {install_path!r}")
    name = install_path.rsplit(marker, 1)[1]
    if not name or (name.startswith("@") and name.count("/") != 1) or (not name.startswith("@") and "/" in name):
        raise SbomError(f"cannot derive npm package name from install path: {install_path!r}")
    return name


def _add_npm_components(root: Path, registry: ComponentRegistry, input_paths: set[Path]) -> None:
    path = root / "packages" / "mcp-server" / "package-lock.json"
    lock = _read_json(path)
    input_paths.add(path)
    if lock.get("lockfileVersion") not in (2, 3):
        raise SbomError("package-lock.json must use lockfileVersion 2 or 3")
    root_name = _nonempty_string(lock.get("name"), "package-lock name")
    root_version = _nonempty_string(lock.get("version"), "package-lock version")
    packages = lock.get("packages")
    if not isinstance(packages, dict) or "" not in packages:
        raise SbomError("package-lock.json must contain a packages object and root entry")
    source = _logical_path(root, path)
    for install_path, raw_entry in sorted(packages.items()):
        if not isinstance(install_path, str) or not isinstance(raw_entry, dict):
            raise SbomError("package-lock package entries must map paths to objects")
        name = _npm_name_from_install_path(install_path, root_name)
        version = _nonempty_string(raw_entry.get("version"), f"npm {name} version")
        if install_path == "" and version != root_version:
            raise SbomError("package-lock root entry version disagrees with top-level version")
        purl = _npm_purl(name, version)
        record = ComponentRecord("application" if install_path == "" else "library", name, version, purl)
        record.hashes.update(_integrity_hashes(raw_entry.get("integrity"), f"npm {name}@{version}"))
        license_value = raw_entry.get("license")
        if license_value is not None:
            record.licenses.add(_nonempty_string(license_value, f"npm {name} license"))
        resolved = raw_entry.get("resolved")
        if resolved is not None:
            record.references.add(("distribution", _validate_https_url(resolved, f"npm {name} resolved")))
        logical_install_path = install_path or "."
        if PurePosixPath(logical_install_path).is_absolute() or ".." in PurePosixPath(logical_install_path).parts:
            raise SbomError(f"npm install path must be repository-relative: {logical_install_path!r}")
        record.add_property("sts2:lock-sources", source)
        record.add_property("sts2:npm-install-paths", logical_install_path)
        if raw_entry.get("dev") is True:
            record.add_property("sts2:npm-dev-install-paths", logical_install_path)
        if raw_entry.get("optional") is True:
            record.add_property("sts2:npm-optional-install-paths", logical_install_path)
        registry.add(record)


def _requirement_profile(path: Path) -> str:
    return {
        "requirements-bootstrap.lock": "packaging-bootstrap",
        "requirements.lock": "runtime-cpu",
        "requirements-dev.lock": "development",
        "requirements-wsl-rocm.txt": "wsl-rocm",
    }.get(path.name, path.stem.removeprefix("requirements-") or "runtime")


def _artifact_profile(lines: list[str], path: Path) -> str:
    prefix = "# artifact-profile:"
    values = [line.removeprefix(prefix).strip() for line in lines if line.startswith(prefix)]
    if len(values) != 1 or not re.fullmatch(r"[a-z0-9][a-z0-9.-]{1,63}", values[0]):
        raise SbomError(f"{path.name} must declare exactly one valid artifact-profile")
    if sum(line.strip() == "--require-hashes" for line in lines) != 1:
        raise SbomError(f"{path.name} must enable --require-hashes exactly once")
    return values[0]


def _parse_requirements(path: Path, root: Path) -> tuple[str, list[tuple[str, str, str]]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as exc:
        raise SbomError(f"required Python lock is missing: {path.name}") from exc
    except UnicodeDecodeError as exc:
        raise SbomError(f"Python lock is not UTF-8: {path.name}") from exc
    artifact_profile = _artifact_profile(lines, path)
    requirements: list[tuple[str, str, str]] = []
    seen_names: set[str] = set()
    for line_number, raw_line in enumerate(lines, 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("-r ") or line.startswith("--requirement "):
            target_value = line.split(maxsplit=1)[1]
            target = (path.parent / target_value).resolve()
            try:
                target.relative_to(path.parent.resolve())
            except ValueError as exc:
                raise SbomError(f"{path.name}:{line_number}: requirement include escapes lock directory") from exc
            if not target.is_file():
                raise SbomError(f"{path.name}:{line_number}: included lock does not exist: {target_value}")
            continue
        if line.startswith("--extra-index-url "):
            _validate_https_url(line.split(maxsplit=1)[1], f"{path.name}:{line_number} index")
            continue
        if line == "--require-hashes":
            continue
        match = _REQUIREMENT_RE.fullmatch(line)
        if match is None:
            raise SbomError(f"{path.name}:{line_number}: requirement is not an exact name==version pin")
        raw_name, version, sha256 = match.groups()
        normalized_name = _python_name(raw_name)
        if normalized_name in seen_names:
            raise SbomError(f"{path.name}:{line_number}: duplicate Python requirement: {normalized_name}")
        seen_names.add(normalized_name)
        requirements.append((normalized_name, version, sha256))
    if not requirements:
        raise SbomError(f"Python lock contains no exact requirements: {path.name}")
    return artifact_profile, requirements


def _add_python_components(root: Path, registry: ComponentRegistry, input_paths: set[Path]) -> None:
    directory = root / "packages" / "rl-agent"
    lock_paths = sorted(directory.glob("requirements*.lock"), key=lambda path: path.name)
    if not lock_paths:
        raise SbomError("no Python requirements*.lock inputs were found")
    for path in lock_paths:
        input_paths.add(path)
        source = _logical_path(root, path)
        profile = _requirement_profile(path)
        artifact_profile, requirements = _parse_requirements(path, root)
        for name, version, sha256 in requirements:
            purl = _python_purl(name, version, artifact_profile)
            record = ComponentRecord("library", name, version, purl)
            record.hashes["SHA-256"] = sha256
            record.add_property("sts2:artifact-profile", artifact_profile)
            record.add_property("sts2:lock-sources", source)
            record.add_property("sts2:python-profiles", profile)
            registry.add(record)


def _normalise_wheel_text(value: str) -> str:
    return re.sub(r"[-_]+", "-", value).lower()


def _add_rocm_components(root: Path, registry: ComponentRegistry, input_paths: set[Path]) -> None:
    path = root / "packages" / "rl-agent" / "requirements-wsl-rocm.txt"
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise SbomError("required ROCm artifact lock is missing: requirements-wsl-rocm.txt") from exc
    except UnicodeDecodeError as exc:
        raise SbomError("ROCm artifact lock is not UTF-8") from exc
    input_paths.add(path)
    if "External ROCm wheel artifact lock" not in text:
        raise SbomError("ROCm lock is missing its external artifact lock marker")
    source = _logical_path(root, path)
    artifact_profile, requirements = _parse_requirements(path, root)
    seen_names: set[str] = set()
    artifact_count = 0
    for line_number, raw_line in enumerate(text.splitlines(), 1):
        match = _ROCM_ARTIFACT_RE.fullmatch(raw_line.strip())
        if match is None:
            continue
        raw_name, version, raw_url, raw_algorithm, raw_hash = match.groups()
        name = _python_name(raw_name)
        if name in seen_names:
            raise SbomError(f"{path.name}:{line_number}: duplicate ROCm wheel artifact: {name}")
        seen_names.add(name)
        url = _validate_https_url(raw_url, f"{path.name}:{line_number} artifact URL")
        filename = unquote(PurePosixPath(urlparse(url).path).name)
        expected_prefix = _normalise_wheel_text(f"{name}-{version}")
        if not filename.endswith(".whl") or not _normalise_wheel_text(filename).startswith(expected_prefix):
            raise SbomError(
                f"{path.name}:{line_number}: wheel filename does not match locked name/version: {filename}"
            )
        purl = _python_purl(name, version, artifact_profile)
        record = ComponentRecord("library", name, version, purl)
        record.references.add(("distribution", url))
        record.add_property("sts2:artifact-kind", "rocm-wheel")
        record.add_property("sts2:artifact-filenames", filename)
        record.add_property("sts2:lock-sources", source)
        record.add_property("sts2:python-profiles", "wsl-rocm")
        if raw_algorithm is None or raw_hash is None:
            raise SbomError(f"{path.name}:{line_number}: ROCm wheel artifact requires sha256")
        algorithm = raw_algorithm.upper().replace("SHA", "SHA-")
        content = raw_hash.lower()
        if algorithm != "SHA-256" or len(content) != 64 or not re.fullmatch(r"[0-9a-f]+", content):
            raise SbomError(f"{path.name}:{line_number}: invalid SHA-256 artifact hash")
        record.hashes[algorithm] = content
        record.add_property("sts2:artifact-profile", artifact_profile)
        registry.add(record)
        artifact_count += 1
    if artifact_count == 0:
        raise SbomError("ROCm artifact lock contains no locked wheel URLs")

    # The same file is also a complete, pinned ROCm Python userspace profile.
    for name, version, sha256 in requirements:
        purl = _python_purl(name, version, artifact_profile)
        record = ComponentRecord("library", name, version, purl)
        record.hashes["SHA-256"] = sha256
        record.add_property("sts2:artifact-profile", artifact_profile)
        record.add_property("sts2:lock-sources", source)
        record.add_property("sts2:python-profiles", "wsl-rocm")
        registry.add(record)


def _relative_destination(value: Any, context: str) -> str:
    destination = _nonempty_string(value, context).replace("\\", "/")
    parsed = PurePosixPath(destination)
    if parsed.is_absolute() or ".." in parsed.parts or re.match(r"^[A-Za-z]:", destination):
        raise SbomError(f"{context} must be a repository-relative path")
    return destination


def _add_third_party_components(root: Path, registry: ComponentRegistry, input_paths: set[Path]) -> None:
    directory = root / "third_party"
    paths = sorted(directory.glob("*.lock.json"), key=lambda path: path.name)
    if not paths:
        raise SbomError("no third-party lock files were found")
    seen_purls: set[str] = set()
    for path in paths:
        lock = _read_json(path)
        input_paths.add(path)
        name = _nonempty_string(lock.get("name"), f"{path.name} name")
        url = _validate_https_url(lock.get("url"), f"{path.name} url")
        commit = _nonempty_string(lock.get("commit"), f"{path.name} commit").lower()
        tree = _nonempty_string(lock.get("tree"), f"{path.name} tree").lower()
        if not _HEX40_RE.fullmatch(commit) or not _HEX40_RE.fullmatch(tree):
            raise SbomError(f"{path.name} commit and tree must be full lowercase Git SHA-1 object IDs")
        license_status = _nonempty_string(lock.get("license_status"), f"{path.name} license_status")
        distribution_allowed = lock.get("distribution_allowed")
        if not isinstance(distribution_allowed, bool):
            raise SbomError(f"{path.name} distribution_allowed must be boolean")
        destination = _relative_destination(lock.get("destination"), f"{path.name} destination")
        purl = _github_purl(url, commit)
        if purl in seen_purls:
            raise SbomError(f"duplicate third-party component lock: {purl}")
        seen_purls.add(purl)
        record = ComponentRecord("application", name, commit, purl)
        record.hashes["SHA-1"] = commit
        declared_license = lock.get("license") or lock.get("license_expression")
        if declared_license is None:
            record.licenses.add("NOASSERTION")
            if license_status == "approved":
                raise SbomError(f"{path.name} is approved but does not identify its license")
        else:
            record.licenses.add(_nonempty_string(declared_license, f"{path.name} license"))
        record.references.add(("vcs", url))
        record.add_property("sts2:distribution-allowed", str(distribution_allowed).lower())
        record.add_property("sts2:git-commit", commit)
        record.add_property("sts2:git-tree", tree)
        record.add_property("sts2:license-status", license_status)
        record.add_property("sts2:lock-sources", _logical_path(root, path))
        record.add_property("sts2:restore-destinations", destination)
        registry.add(record)


def _all_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from _all_strings(key)
            yield from _all_strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _all_strings(item)


def _validate_bom(bom: dict[str, Any]) -> None:
    if bom.get("bomFormat") != "CycloneDX" or bom.get("specVersion") != "1.6" or bom.get("version") != 1:
        raise SbomError("invalid CycloneDX document header")
    components = bom.get("components")
    if not isinstance(components, list) or not components:
        raise SbomError("SBOM contains no components")
    refs: set[str] = set()
    purls: set[str] = set()
    previous_ref: str | None = None
    for index, component in enumerate(components):
        if not isinstance(component, dict):
            raise SbomError(f"component {index} is not an object")
        for field_name in ("bom-ref", "name", "purl", "type", "version"):
            _nonempty_string(component.get(field_name), f"component {index} {field_name}")
        bom_ref = component["bom-ref"]
        purl = component["purl"]
        if bom_ref != purl or not purl.startswith("pkg:"):
            raise SbomError(f"component has a non-canonical identity: {bom_ref}")
        if bom_ref in refs or purl in purls:
            raise SbomError(f"duplicate component in generated SBOM: {bom_ref}")
        if previous_ref is not None and previous_ref >= bom_ref:
            raise SbomError("generated components are not in strict stable order")
        previous_ref = bom_ref
        refs.add(bom_ref)
        purls.add(purl)
        property_names: set[str] = set()
        for prop in component.get("properties", []):
            if not isinstance(prop, dict):
                raise SbomError(f"component property is not an object: {bom_ref}")
            property_name = _nonempty_string(prop.get("name"), f"{bom_ref} property name")
            _nonempty_string(prop.get("value"), f"{bom_ref} property value")
            if property_name in property_names:
                raise SbomError(f"duplicate property on component {bom_ref}: {property_name}")
            property_names.add(property_name)
        for hash_entry in component.get("hashes", []):
            if not isinstance(hash_entry, dict):
                raise SbomError(f"component hash is not an object: {bom_ref}")
            algorithm = hash_entry.get("alg")
            content = hash_entry.get("content")
            if algorithm not in _HASH_LENGTHS or not isinstance(content, str):
                raise SbomError(f"component has unsupported hash metadata: {bom_ref}")
            if len(content) != _HASH_LENGTHS[algorithm] or not re.fullmatch(r"[0-9a-f]+", content):
                raise SbomError(f"component has malformed {algorithm} hash: {bom_ref}")
    for value in _all_strings(bom):
        if value.startswith("file:") or value.startswith("\\\\") or re.match(r"^[A-Za-z]:[\\/]", value):
            raise SbomError("generated SBOM contains a machine-specific absolute path")
        if value.startswith("/"):
            raise SbomError("generated SBOM contains an absolute filesystem path")


def build_sbom(root: Path = ROOT) -> dict[str, Any]:
    """Return a validated, deterministic CycloneDX object for ``root``."""

    root = root.resolve()
    registry = ComponentRegistry()
    input_paths: set[Path] = set()
    release_schema_version, release_channel = _add_release_components(root, registry, input_paths)
    _add_npm_components(root, registry, input_paths)
    _add_python_components(root, registry, input_paths)
    _add_rocm_components(root, registry, input_paths)
    _add_third_party_components(root, registry, input_paths)

    input_hashes = {
        _logical_path(root, path): _sha256(path)
        for path in sorted(input_paths, key=lambda item: _logical_path(root, item))
    }
    bom: dict[str, Any] = {
        "$schema": CYCLONEDX_SCHEMA,
        "bomFormat": "CycloneDX",
        "components": registry.components(),
        "metadata": {
            "properties": [
                {
                    "name": "sts2:input-sha256",
                    "value": json.dumps(input_hashes, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
                },
                {"name": "sts2:release-channel", "value": release_channel},
                {"name": "sts2:release-manifest-schema", "value": release_schema_version},
            ],
            "tools": {
                "components": [
                    {
                        "bom-ref": _generic_purl("sts2", "build-sbom", GENERATOR_VERSION),
                        "name": "build_sbom.py",
                        "purl": _generic_purl("sts2", "build-sbom", GENERATOR_VERSION),
                        "type": "application",
                        "version": GENERATOR_VERSION,
                    }
                ]
            },
        },
        "specVersion": "1.6",
        "version": 1,
    }
    _validate_bom(bom)
    return bom


def render_sbom(root: Path = ROOT) -> bytes:
    """Serialize the SBOM as canonical repository UTF-8 JSON bytes."""

    return (json.dumps(build_sbom(root), ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="destination CycloneDX JSON file")
    args = parser.parse_args(argv)
    try:
        payload = render_sbom(ROOT)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(payload)
    except (OSError, SbomError) as exc:
        print(f"SBOM generation failed: {exc}", file=sys.stderr)
        return 1
    print(f"wrote deterministic CycloneDX SBOM: {args.output} ({len(payload)} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
