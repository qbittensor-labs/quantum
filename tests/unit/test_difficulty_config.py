from __future__ import annotations

import json
from pathlib import Path

import pytest
from qbittensor.validator.config.difficulty_config import DifficultyConfig

def _stub_max_solved(val: float):
    """Return a fake sql_utils.max_solved_difficulty implementation."""
    return lambda *_a, **_kw: val


CFG_PATH_STR = "qbittensor.validator.config.sql_utils.max_solved_difficulty"


# Tests
def test_peaked_clamping(monkeypatch, tmp_path: Path):
    """
    For peaked circuits `lamp=True, values above 0.7 clamp until the
    miner has solved harder problems.
    Downward moves are always allowed.
    """
    cfg_path = tmp_path / "peaked.json"
    db_path = tmp_path / "db.sqlite"

    # Pretend the miner once solved up to 0.50
    monkeypatch.setattr(CFG_PATH_STR, _stub_max_solved(0.50))

    dc = DifficultyConfig(
        path=cfg_path,
        uids=[0],
        default=0.0,
        db_path=db_path,
        hotkey_lookup=lambda uid: "hk-0",
        clamp=True, # PEAKED behaviour
    )

    # Initial default
    assert dc.get(0) == 0.0

    # Ask for 0.90 → should clamp to 0.70
    assert dc.set(0, 0.90) is True
    assert dc.get(0) == pytest.approx(0.70)

    # Ask for a *mall increase within +0.40 window (0.70 → 1.05 allowed max)
    assert dc.set(0, 1.00) is True
    assert dc.get(0) == pytest.approx(1.00)

    # Attempt to overshoot the +0.40 window (1.00 current -> 1.45 asked)
    # Allowed max is 1.40, so result should clamp to 1.40
    assert dc.set(0, 1.45) is True
    assert dc.get(0) == pytest.approx(1.40)

    # Downward move always accepted
    assert dc.set(0, 0.20) is True
    assert dc.get(0) == pytest.approx(0.20)


def test_hstab_unclamped(monkeypatch, tmp_path: Path):

    cfg_path = tmp_path / "hstab.json"

    monkeypatch.setattr(CFG_PATH_STR, _stub_max_solved(0.0))

    dc = DifficultyConfig(
        path=cfg_path,
        uids=[1],
        default=26.0, # default for hstab
        clamp=False, # H‑STAB behaviour
    )

    # Can jump far beyond 0.7 and +0.40 limits
    assert dc.set(1, 50.0) is True
    assert dc.get(1) == pytest.approx(50.0)


def test_negative_difficulty_rejected(tmp_path: Path):
    cfg_path = tmp_path / "neg.json"
    dc = DifficultyConfig(path=cfg_path, uids=[2])

    # Negative values are ignored
    assert dc.set(2, -1.0) is False
    assert dc.get(2) == pytest.approx(0.0)


def test_uid_backfill(tmp_path: Path):
    """
    New UIDs are back-filled with the default difficulty value.
    """
    cfg_path = tmp_path / "uids.json"
    dc = DifficultyConfig(path=cfg_path, uids=[1, 2], default=0.3)

    dc.update_uid_list([1, 2, 3])
    assert dc.get(3) == pytest.approx(0.3)

    # Verify persisted JSON also contains the new UID
    data = json.loads(cfg_path.read_text())
    assert data["3"] == 0.3
