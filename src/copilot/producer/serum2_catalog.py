"""Real local Serum/Serum2 discovery + preset catalog builder.

This module is read-only over local filesystem state. It does not mutate Ableton.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from copilot.producer.preset_catalog import PresetCandidate, PresetCatalog


class SerumPluginInstall(BaseModel):
    plugin: str
    format: str
    path: str
    exists: bool


class SerumDiscoveryReport(BaseModel):
    serum2_installation: str
    serum2_plugin_paths: list[SerumPluginInstall] = Field(default_factory=list)
    serum2_preset_roots: list[str] = Field(default_factory=list)
    preset_count: int = 0
    notes: list[str] = Field(default_factory=list)


def _sha1(path: Path) -> str:
    h = hashlib.sha1()
    with path.open('rb') as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def default_plugin_candidates() -> list[tuple[str, str, Path]]:
    """Best-effort local plugin paths (macOS + Windows conventions)."""
    home = Path.home()
    appdata = Path(os.environ.get('APPDATA') or home / 'AppData/Roaming')
    localappdata = Path(os.environ.get('LOCALAPPDATA') or home / 'AppData/Local')
    common = [
        ('Serum 2', 'VST3', Path('/Library/Audio/Plug-Ins/VST3/Serum2.vst3')),
        ('Serum 2', 'AU', Path('/Library/Audio/Plug-Ins/Components/Serum2.component')),
        ('Serum 2', 'VST', Path('/Library/Audio/Plug-Ins/VST/Serum.vst')),
        ('Serum 2', 'VST3', home / 'Library/Audio/Plug-Ins/VST3/Serum2.vst3'),
        ('Serum 2', 'AU', home / 'Library/Audio/Plug-Ins/Components/Serum2.component'),
        ('Serum 2', 'VST', home / 'Library/Audio/Plug-Ins/VST/Serum.vst'),
        ('Serum 2', 'VST3', Path(r'C:/Program Files/Common Files/VST3/Serum2.vst3')),
        ('Serum 2', 'VST3', Path(r'C:/Program Files/Common Files/VST3/Serum 2.vst3')),
        ('Serum 2', 'VST', Path(r'C:/Program Files/VstPlugins/Serum.dll')),
        ('Serum 2', 'VST', Path(r'C:/Program Files/Steinberg/VstPlugins/Serum.dll')),
        ('Serum 2', 'VST', localappdata / 'Programs/Xfer/Serum/Serum.dll'),
        ('Serum', 'VST3', Path('/Library/Audio/Plug-Ins/VST3/Serum.vst3')),
        ('Serum', 'AU', Path('/Library/Audio/Plug-Ins/Components/Serum.component')),
        ('Serum', 'VST', Path('/Library/Audio/Plug-Ins/VST/Serum.vst')),
    ]
    return common


def default_preset_roots() -> list[Path]:
    home = Path.home()
    appdata = Path(os.environ.get('APPDATA') or home / 'AppData/Roaming')
    roots = [
        Path('/Library/Audio/Presets/Xfer Records/Serum 2 Presets'),
        Path('/Library/Audio/Presets/Xfer Records/Serum Presets'),
        home / 'Library/Audio/Presets/Xfer Records/Serum 2 Presets',
        home / 'Library/Audio/Presets/Xfer Records/Serum Presets',
        home / 'Music/Ableton/User Library',
        home / 'Documents/Xfer/Serum Presets',
        home / 'Music/Xfer/Serum Presets',
        appdata / 'Xfer/Serum Presets',
    ]
    out: list[Path] = []
    seen: set[str] = set()
    for r in roots:
        key = str(r)
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def discover_serum2_resources(*, preset_roots: list[Path] | None = None, plugin_candidates: list[tuple[str, str, Path]] | None = None) -> SerumDiscoveryReport:
    installs: list[SerumPluginInstall] = []
    for plugin, fmt, path in (plugin_candidates or default_plugin_candidates()):
        installs.append(SerumPluginInstall(plugin=plugin, format=fmt, path=str(path), exists=path.exists()))

    roots = [p for p in (preset_roots or default_preset_roots()) if p.exists() and p.is_dir()]
    total = 0
    for root in roots:
        total += sum(1 for p in root.rglob('*') if p.is_file() and p.suffix.lower() in {'.serumpreset', '.fxp', '.fxb', '.vstpreset'})

    has_serum2 = any(i.exists and i.plugin.casefold() == 'serum 2' for i in installs)
    notes: list[str] = []
    if not has_serum2:
        notes.append('SERUM2_BINARY_NOT_FOUND')
    if not roots:
        notes.append('NO_PRESET_ROOT_FOUND')

    return SerumDiscoveryReport(
        serum2_installation='VERIFIED' if has_serum2 else 'NOT_FOUND',
        serum2_plugin_paths=installs,
        serum2_preset_roots=[str(p) for p in roots],
        preset_count=total,
        notes=notes,
    )


def _infer_roles(path: Path) -> list[str]:
    text = f"{path.parent.name} {path.name}".casefold()
    role_tokens = [
        ('bass', ('bass', ' bs ', 'sub', 'reese', '808')),
        ('lead', ('lead', ' ld ', 'scream')),
        ('pluck', ('pluck', ' pl ')),
        ('pad', ('pad', ' pd ', 'atmo')),
        ('keys', ('keys', 'key ', 'ky ', 'piano', 'organ')),
        ('fx', ('fx', 'impact', 'riser', 'downlifter', 'sweep')),
        ('arp', ('arp', 'arpeggio')),
    ]
    roles: list[str] = []
    for role, tokens in role_tokens:
        for tok in tokens:
            if tok.strip() and tok in text:
                roles.append(role)
                break
    return roles or ['unknown']


def _infer_plugin(path: Path, source_root: Path) -> str:
    t = f"{source_root} {path}".casefold()
    if 'serum 2' in t or path.suffix.casefold() == '.serumpreset':
        return 'Serum 2'
    return 'Serum'


def build_local_preset_catalog(*, roots: list[Path] | None = None, include_sha1: bool = True) -> PresetCatalog:
    found_roots = [p for p in (roots or default_preset_roots()) if p.exists() and p.is_dir()]
    presets: list[PresetCandidate] = []
    valid_ext = {'.serumpreset', '.fxp', '.fxb', '.vstpreset'}

    for root in sorted(found_roots, key=lambda p: str(p).casefold()):
        for path in sorted((p for p in root.rglob('*') if p.is_file() and p.suffix.lower() in valid_ext), key=lambda p: str(p).casefold()):
            stat = path.stat()
            plugin = _infer_plugin(path, root)
            rel = path.relative_to(root) if path.is_relative_to(root) else path
            uri = f"file://{path}"
            provenance: dict[str, Any] = {
                'source_root': str(root),
                'relative_path': str(rel),
                'file_size': int(stat.st_size),
                'mtime': int(stat.st_mtime),
                'extension': path.suffix,
            }
            if include_sha1:
                provenance['sha1'] = _sha1(path)
            presets.append(
                PresetCandidate(
                    uri=uri,
                    name=path.stem,
                    plugin=plugin,
                    roles=_infer_roles(path),
                    tags=[],
                    provenance=provenance,
                )
            )
    return PresetCatalog(presets=presets)
