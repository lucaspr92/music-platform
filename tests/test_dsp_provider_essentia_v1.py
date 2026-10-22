"""Essentia provider reports the installed library; the licence lives elsewhere."""

from __future__ import annotations

import importlib.util
import sys

from copilot.audio.physical_dsp_v2.providers import (
    essentia_status,
    provider_inventory,
    require,
)


def test_essentia_status_matches_whether_the_module_is_importable() -> None:
    installed = importlib.util.find_spec("essentia") is not None
    status = essentia_status()
    assert status.provider_id == "essentia"
    assert status.available is installed


def test_available_essentia_carries_no_limitation() -> None:
    """An available provider must not carry a limitation: both the inventory and
    tonal.analyze_tonal read a limitation as 'this provider is unusable'."""
    status = essentia_status()
    if not status.available:
        return
    assert status.limitation is None
    assert require(status) is None
    assert status.version not in ("", "missing")


def test_missing_essentia_reports_unavailable(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "essentia", None)  # makes `import essentia` raise
    status = essentia_status()
    assert status.available is False
    assert status.version == "missing"
    limitation = require(status)
    assert limitation is not None
    assert limitation.code == "ESSENTIA_UNAVAILABLE"


def test_inventory_never_pairs_available_with_a_limitation() -> None:
    for status in provider_inventory():
        if status.available:
            assert status.limitation is None, status.provider_id


def test_licence_blocked_code_is_no_longer_produced() -> None:
    """The old hardcoded 'license-blocked' posture is gone: an installed Essentia
    must report itself, so the inventory stops contradicting the venv."""
    codes = [row.provider_id for row in provider_inventory()]
    assert "essentia" in codes
    assert essentia_status().version != "license-blocked"
