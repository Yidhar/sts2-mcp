from __future__ import annotations

from pathlib import Path

import pytest

from sts2_rl.game_data import resolve_game_data_file, resolve_generated_game_data_output


def test_repository_game_data_is_canonical() -> None:
    path = resolve_game_data_file("cards.generated.json", required=True)
    assert path.parts[-3:] == ("game-data", "generated", "cards.generated.json")


def test_environment_override_has_highest_precedence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    generated = tmp_path / "generated"
    generated.mkdir()
    target = generated / "custom.json"
    target.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("STS2_GAME_DATA_ROOT", str(tmp_path))
    assert resolve_game_data_file("custom.json", required=True) == target


def test_locator_rejects_path_escape() -> None:
    with pytest.raises(ValueError):
        resolve_game_data_file("../secret.json")


def test_generated_output_is_confined_to_configured_generated_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    game_data = tmp_path / "shared-game-data"
    monkeypatch.setenv("STS2_GAME_DATA_ROOT", str(game_data))

    assert resolve_generated_game_data_output("cards.json") == game_data / "generated" / "cards.json"
    assert (
        resolve_generated_game_data_output(game_data / "generated" / "nested" / "cards.json")
        == game_data / "generated" / "nested" / "cards.json"
    )

    with pytest.raises(ValueError, match="must stay below"):
        resolve_generated_game_data_output(game_data / "raw" / "cards.json")
    with pytest.raises(ValueError, match="must stay below"):
        resolve_generated_game_data_output("../outside.json")
