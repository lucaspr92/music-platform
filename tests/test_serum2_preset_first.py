from __future__ import annotations

from pathlib import Path

from copilot.daw.mock import MockAbletonAdapter
from copilot.producer.serum2_catalog import build_local_preset_catalog
from copilot.producer.serum2_preset_first import identify_serum2_device, select_real_serum2_preset


def _write(path: Path, data: bytes = b'x') -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def test_identify_serum2_device_in_session() -> None:
    daw = MockAbletonAdapter()
    daw.connect()
    daw.create_midi_track('Synth', 0)
    track = daw.tracks[0]
    track['devices'].append(
        {
            'name': 'Serum 2',
            'class_name': 'Serum2',
            'enabled': True,
            'parameters': [],
        }
    )
    session = daw.snapshot()
    ident = identify_serum2_device(session=session)
    assert ident.found is True
    assert ident.track_name == 'Synth'
    assert ident.device_name == 'Serum 2'
    assert ident.vendor == 'Xfer Records'


def test_select_real_serum2_preset_from_temp_catalog(monkeypatch, tmp_path: Path) -> None:
    root = tmp_path / 'Serum 2 Presets'
    _write(root / 'Bass' / 'A Bass.SerumPreset')
    _write(root / 'Lead' / 'A Lead.SerumPreset')

    from copilot.producer import serum2_preset_first as spf

    monkeypatch.setattr(spf, 'discover_serum2_resources', lambda: type('R', (), {
        'serum2_installation': 'VERIFIED',
        'serum2_preset_roots': [str(root)],
        'preset_count': 2,
        'notes': [],
    })())
    monkeypatch.setattr(spf, 'build_local_preset_catalog', lambda include_sha1=False: build_local_preset_catalog(roots=[root], include_sha1=False))

    rep = select_real_serum2_preset(role='bass')
    assert rep.status == 'SELECTED'
    assert rep.selected_name == 'A Bass'
    assert rep.compatible_count >= 1
