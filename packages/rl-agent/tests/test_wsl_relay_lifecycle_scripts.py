from __future__ import annotations

from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
START = (SCRIPTS / "start_wsl_bridge_relay.ps1").read_text(encoding="utf-8")
STOP = (SCRIPTS / "stop_wsl_bridge_relay.ps1").read_text(encoding="utf-8")
TRAIN = (SCRIPTS / "train_grounded_wsl_rocm.sh").read_text(encoding="utf-8")


def test_start_defaults_to_loopback_and_rejects_wildcard() -> None:
    assert '[string]$ListenHost = "127.0.0.1"' in START
    assert 'return "127.0.0.1"' in START
    assert 'Wildcard relay binding is forbidden' in START
    assert "-ListenHost '0.0.0.0'" not in TRAIN
    assert 'relay_host="${STS2_BRIDGE_RELAY_HOST:-' in TRAIN


def test_lifecycle_never_enumerates_or_kills_by_command_substring() -> None:
    forbidden = (
        'Where-Object { $_.CommandLine -like "*bridge_wsl_relay.py*" }',
        "taskkill /IM",
        "Get-Process -Name",
    )
    for text in (START, STOP):
        for pattern in forbidden:
            assert pattern not in text
        assert 'Get-CimInstance Win32_Process -Filter "ProcessId = $ProcessId"' in text


def test_stop_requires_complete_process_identity_before_termination() -> None:
    for field in (
        "process_creation_time_utc",
        "executable_path",
        "script_path",
        "instance_id",
        "command_line_sha256",
    ):
        assert field in STOP
    identity_guard = STOP.index("if (-not $identity.Matches)")
    stop_call = STOP.index("Stop-Process -Id ([int]$state.pid) -Force")
    assert identity_guard < stop_call
    assert 'reason = "identity_mismatch"' in STOP


def test_start_protects_state_acl_and_passes_security_limits() -> None:
    assert "SetAccessRuleProtection($true, $false)" in START
    assert "Protect-RelayStatePath -Path $stateFile" in START
    for flag in (
        "--auth-session-file",
        "--allow-client-cidr",
        "--max-body-bytes",
        "--max-response-bytes",
        "--max-concurrency",
        "--target-timeout-s",
        "--instance-id",
    ):
        assert flag in START


def test_retained_wsl_launcher_has_no_personal_absolute_path() -> None:
    assert "yidhar" not in TRAIN.lower()
    assert "/mnt/c/Users/" not in TRAIN
    assert "GetFolderPath" in TRAIN


def test_training_uses_environment_selected_relay_authority() -> None:
    assert 'export STS2_BRIDGE_RELAY_ALLOWLIST="$STS2_BRIDGE_BASE_URL"' in TRAIN
    assert '--session-path "$STS2_BRIDGE_SESSION_FILE"' not in TRAIN
    assert "export STS2_BRIDGE_SESSION_FILE=" in TRAIN


def test_launchers_keep_virtual_environments_outside_checkout() -> None:
    bootstrap = (SCRIPTS / "bootstrap_wsl_rocm.sh").read_text(encoding="utf-8")
    for text in (bootstrap, TRAIN):
        assert '$ARTIFACT_ROOT/environments/wsl-rocm' in text
        assert '$REPO_ROOT/.venv-wsl-rocm' not in text
        assert "artifact root must be disjoint from the source checkout" in text
    assert 'environments\\windows-python\\Scripts\\python.exe' in START
    assert '"venv\\Scripts\\python.exe"' not in START
    assert "Artifact root must be disjoint from the source checkout" in START
