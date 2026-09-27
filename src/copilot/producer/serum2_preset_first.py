"""Serum2 preset-first orchestration on top of real local catalog.

Selection only: no Ableton writes are performed here.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from copilot.producer.preset_catalog import PresetSelectionRequest
from copilot.producer.serum2_catalog import (
    build_local_preset_catalog,
    discover_serum2_resources,
)
from copilot.schemas.session import SessionState


class Serum2SelectionReport(BaseModel):
    status: str
    role: str
    desired_tags: list[str] = Field(default_factory=list)
    candidate_count: int = 0
    compatible_count: int = 0
    selected_preset_id: str | None = None
    selected_name: str | None = None
    ranking_evidence: list[str] = Field(default_factory=list)
    selection_provenance: dict[str, Any] = Field(default_factory=dict)


class Serum2DeviceIdentity(BaseModel):
    found: bool = False
    track_name: str | None = None
    device_name: str | None = None
    device_index: int | None = None
    class_name: str | None = None
    plugin_format: str | None = None
    vendor: str | None = None
    binary_path: str | None = None
    evidence: list[str] = Field(default_factory=list)


_FORMAT_BY_SUFFIX = {
    '.vst3': 'VST3',
    '.component': 'AU',
    '.vst': 'VST',
    '.dll': 'VST',
}


def select_real_serum2_preset(*, role: str = 'bass', desired_tags: list[str] | None = None, limit: int = 3) -> Serum2SelectionReport:
    desired_tags = list(desired_tags or [])
    discovery = discover_serum2_resources()
    if discovery.serum2_installation != 'VERIFIED':
        return Serum2SelectionReport(
            status='BLOCKED_LOCAL_RESOURCE',
            role=role,
            desired_tags=desired_tags,
            ranking_evidence=['SERUM2_BINARY_NOT_FOUND'],
            selection_provenance={
                'installation': discovery.serum2_installation,
                'notes': discovery.notes,
            },
        )

    catalog = build_local_preset_catalog(include_sha1=False)
    compatible_count = sum(
        1
        for p in catalog.presets
        if p.plugin.casefold().strip() == 'serum 2' and role.casefold().strip() in {r.casefold().strip() for r in p.roles}
    )
    res = catalog.select(
        PresetSelectionRequest(
            plugin='Serum 2',
            role=role,
            desired_tags=desired_tags,
            limit=limit,
        )
    )
    selected = res.candidates[0] if res.candidates else None
    return Serum2SelectionReport(
        status=res.status,
        role=role,
        desired_tags=desired_tags,
        candidate_count=len(catalog.presets),
        compatible_count=compatible_count,
        selected_preset_id=None if selected is None else selected.uri,
        selected_name=None if selected is None else selected.name,
        ranking_evidence=list(res.reasons),
        selection_provenance={
            'preset_roots': discovery.serum2_preset_roots,
            'preset_count': discovery.preset_count,
            'installation': discovery.serum2_installation,
        },
    )


def identify_serum2_device(*, session: SessionState, preferred_track: str | None = None) -> Serum2DeviceIdentity:
    tracks = session.tracks
    if preferred_track:
        t = session.track_by_name(preferred_track)
        tracks = [t] if t is not None else []

    for tr in tracks:
        for dev in tr.devices:
            n = (dev.name or '').casefold()
            c = (dev.class_name or '').casefold()
            if 'serum' not in n and 'serum' not in c:
                continue
            # strongest local evidence we can add without writing: installed plugin binary candidates
            discovery = discover_serum2_resources()
            existing = [x for x in discovery.serum2_plugin_paths if x.exists and x.plugin.casefold() == 'serum 2']
            chosen = existing[0] if existing else None
            fmt = None
            bin_path = None
            ev = [f"track={tr.name}", f"device_name={dev.name}", f"class_name={dev.class_name}"]
            if chosen is not None:
                bin_path = chosen.path
                fmt = chosen.format
                ev.append(f"binary={chosen.path}")
                ev.append(f"format={chosen.format}")
            else:
                # fallback from filename
                for suffix, f in _FORMAT_BY_SUFFIX.items():
                    if n.endswith(suffix) or c.endswith(suffix):
                        fmt = f
                        ev.append(f"format_inferred={f}")
                        break
            return Serum2DeviceIdentity(
                found=True,
                track_name=tr.name,
                device_name=dev.name,
                device_index=dev.index,
                class_name=dev.class_name,
                plugin_format=fmt,
                vendor='Xfer Records',
                binary_path=bin_path,
                evidence=ev,
            )

    return Serum2DeviceIdentity(found=False, evidence=['NO_SERUM_DEVICE_IN_SESSION'])
