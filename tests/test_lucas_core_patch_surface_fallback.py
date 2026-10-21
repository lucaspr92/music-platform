from __future__ import annotations

from pathlib import Path

from copilot.daw.mock import MockAbletonAdapter
from copilot.integration.lucas_core_v1 import execute_lucas_patch_contracts_through_core
import copilot.producer.soniq_surface as soniq_surface


def _seed_serum_limited_surface() -> tuple[MockAbletonAdapter, object]:
    daw = MockAbletonAdapter()
    daw.connect()
    daw.create_midi_track("Bass", 0)
    track = daw.tracks[0]
    track["devices"].append(
        {
            "name": "Serum",
            "class_name": "PluginDevice",
            "enabled": True,
            "parameters": [
                {"index": 0, "name": "Device On", "value": 1.0, "min": 0.0, "max": 1.0},
            ],
        }
    )
    return daw, daw.snapshot()


def test_serum2_alias_uses_auto_surface_when_limited(monkeypatch, tmp_path: Path) -> None:
    daw, session = _seed_serum_limited_surface()

    def fake_auto_mode(*_args, **_kwargs):
        return {"ok": True, "routing_mode": "soniq_full_surface"}

    monkeypatch.setattr(soniq_surface, "apply_patch_contract_auto_mode", fake_auto_mode)

    report = execute_lucas_patch_contracts_through_core(
        contracts=[
            {
                "operation": "set_device_parameter",
                "track": "Bass",
                "device": "Serum2",
                "writes": [{"index": 205, "value": 0.61}],
                "constraints": {"max_writes": 1},
            }
        ],
        session=session,
        daw=daw,
        persist_dir=tmp_path,
    )

    assert report["status"] == "SAFE_WRITE_COMPLETE"
    assert len(report["accepted"]) == 1
    assert report["accepted"][0]["surface_backend"] == "soniq_auto_mode"
    assert report["accepted"][0]["routing_mode"] == "soniq_full_surface"
    assert report["deferred"] == []
