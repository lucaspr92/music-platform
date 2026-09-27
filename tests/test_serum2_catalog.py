from __future__ import annotations

from pathlib import Path

from copilot.producer.serum2_catalog import (
    build_local_preset_catalog,
    discover_serum2_resources,
)


def _write(path: Path, data: bytes = b'x') -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def test_discovery_reports_missing_serum2_when_binary_absent(tmp_path: Path) -> None:
    root = tmp_path / 'Presets'
    _write(root / 'Bass' / 'My Bass.SerumPreset')

    rep = discover_serum2_resources(
        preset_roots=[root],
        plugin_candidates=[('Serum 2', 'VST3', tmp_path / 'missing' / 'Serum2.vst3')],
    )

    assert rep.serum2_installation == 'NOT_FOUND'
    assert rep.preset_count == 1
    assert 'SERUM2_BINARY_NOT_FOUND' in rep.notes


def test_build_local_catalog_collects_metadata_and_roles(tmp_path: Path) -> None:
    root = tmp_path / 'Serum 2 Presets'
    _write(root / 'Bass' / 'Deep Bass.SerumPreset', b'123')
    _write(root / 'Leads' / 'Acid Lead.SerumPreset', b'456')
    _write(root / 'Pads' / 'Wide Pad.SerumPreset', b'789')

    catalog = build_local_preset_catalog(roots=[root], include_sha1=False)

    assert len(catalog.presets) == 3
    by_name = {p.name: p for p in catalog.presets}
    assert by_name['Deep Bass'].plugin == 'Serum 2'
    assert 'bass' in by_name['Deep Bass'].roles
    assert 'lead' in by_name['Acid Lead'].roles
    assert 'pad' in by_name['Wide Pad'].roles
    assert by_name['Deep Bass'].provenance['source_root'] == str(root)
    assert by_name['Deep Bass'].provenance['extension'] == '.SerumPreset'


def test_old_fxp_root_classifies_as_serum_not_serum2(tmp_path: Path) -> None:
    root = tmp_path / 'Serum Presets'
    _write(root / 'Presets' / 'Misc' / 'Clean Bass.fxp')

    catalog = build_local_preset_catalog(roots=[root], include_sha1=False)
    assert len(catalog.presets) == 1
    preset = catalog.presets[0]
    assert preset.plugin == 'Serum'
    assert 'bass' in preset.roles
