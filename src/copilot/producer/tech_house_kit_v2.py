"""Deterministic 16-bar A/B candidate derived from the unchanged V1 kit."""

from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path

from copilot.producer import tech_house_kit_v1 as v1
from copilot.schemas.session import MidiNote


MILESTONE = "TECH_HOUSE_KIT_V2_GROOVE_AND_SOUND"
KIT = tuple(
    replace(sound, relative_path=sound.fallback_relative_path,
            browser_name=sound.fallback_browser_name, ableton_device="Operator")
    if sound.role == "Bass" else sound for sound in v1.KIT
)


def _note(pitch: int, beat: float, duration: float, velocity: int) -> MidiNote:
    return MidiNote(pitch=pitch, start_time=round(beat, 6),
                    duration=round(duration, 6), velocity=velocity)


def bass_notes() -> list[MidiNote]:
    notes = []
    motif = (0.5, 1.75, 2.5, 3.25)
    velocities = (104, 96, 108, 100)
    durations = (0.24, 0.19, 0.22, 0.18)
    departures = {3: (3, 46), 7: (2, 50), 11: (3, 41)}
    for bar in range(16):
        for cell, onset in enumerate(motif):
            pitch = departures.get(bar, (-1, 43))[1] if departures.get(bar, (-1, 43))[0] == cell else 43
            notes.append(_note(pitch, 4 * bar + onset, durations[cell],
                               velocities[cell] + (2 if 8 <= bar < 12 else 0)))
        if bar in (3, 7, 11, 15):
            notes.append(_note(43, 4 * bar + 3.77, 0.16, 72))
    return notes


def closed_hat_notes() -> list[MidiNote]:
    notes = []
    for bar in range(16):
        for beat in range(4):
            for subdivision, velocity in ((0.25, 58), (0.75, 76)):
                if (bar + beat) % 4 == 3 and subdivision == 0.25:
                    continue
                if bar == 14 and beat in (1, 3) and subdivision == 0.75:
                    continue
                offset = 0.024 if subdivision == 0.25 else 0.032
                accent = (4 if beat in (1, 3) else 0) + (2 if 8 <= bar < 12 else 0)
                notes.append(_note(60, bar * 4 + beat + subdivision + offset,
                                   0.10, velocity + accent))
        if bar in (3, 7, 11, 15):
            notes.append(_note(60, bar * 4 + 3.93, 0.06, 50))
    return notes


def open_hat_notes() -> list[MidiNote]:
    notes = []
    for bar in range(16):
        if bar == 14:
            continue
        for beat in range(4):
            if (bar, beat) in {(5, 2), (7, 0), (15, 1), (15, 3)}:
                continue
            notes.append(_note(60, bar * 4 + beat + 0.5 + (0.015 if beat % 2 else 0.01),
                               0.23, 88 + (6 if beat in (1, 3) else 0)))
    return notes


def perc_notes() -> list[MidiNote]:
    notes = []
    for bar in range(16):
        if bar == 15:
            cells = ((0.73, 62), (2.23, 72), (3.23, 60), (3.48, 67), (3.73, 74))
        elif bar == 14:
            cells = ((0.73, 68),)
        elif 4 <= bar < 8 or 12 <= bar < 14:
            cells = ((0.73, 72), (1.73, 50), (2.27, 76), (3.53, 64))
        else:
            cells = ((0.73, 72), (2.27, 76), (3.53, 62))
        for onset, velocity in cells:
            notes.append(_note(60, bar * 4 + onset, 0.10, velocity))
    return notes


def stab_notes() -> list[MidiNote]:
    notes = []
    for bar in range(8, 16):
        if bar == 14:
            continue
        cells = ((1.5, (58, 62, 65, 67), 78),) if bar % 2 == 0 else (
            (0.75, (58, 65), 72), (3.25, (62, 67), 68))
        for onset, pitches, velocity in cells:
            for pitch in pitches:
                notes.append(_note(pitch, bar * 4 + onset, 0.20, velocity))
    return notes


GENERATORS = {
    "Kick": v1.kick_notes,
    "Clap": v1.clap_notes,
    "Closed Hat": closed_hat_notes,
    "Open Hat": open_hat_notes,
    "Perc": perc_notes,
    "Bass": bass_notes,
    "Stab": stab_notes,
}


def groove_plan(tempo_bpm: float = v1.TEMPO_BPM, key: str = v1.KEY) -> dict:
    """Same contract as V1: defaults frozen, other MINOR keys transpose pitched roles only."""
    semitones = v1.transposition_to(key)
    roles = {}
    for sound in KIT:
        notes = GENERATORS[sound.role]()
        if not sound.is_one_shot:
            notes = v1.transpose_notes(notes, semitones)
        roles[sound.role] = {
            "track_name": f"KIT DIRECT V2 - {sound.role}",
            "clip_index": 0, "length_beats": v1.CLIP_BEATS,
            "arrangement_start_beat": 0.0, "note_count": len(notes),
            "source_name": sound.browser_name,
            "notes": [note.model_dump(mode="json") for note in notes],
        }
    return {
        "milestone": MILESTONE, "tempo_bpm": float(tempo_bpm),
        "meter": list(v1.METER), "bars": v1.BARS, "key": key,
        "key_transposition_semitones": semitones,
        "sections": list(v1.SECTIONS), "sidechain": "DEFERRED",
        "automation": "DEFERRED", "roles": roles,
    }


def kit_definition(core_library: Path) -> dict:
    rows = []
    for sound in KIT:
        path = v1.resolve_relative(core_library, sound.relative_path)
        if not path.is_file():
            raise FileNotFoundError(f"KIT_SOUND_MISSING: {sound.role}: {path}")
        rows.append({**asdict(sound), "path": str(path),
                     "sha256": v1.sha256_file(path), "bytes": path.stat().st_size})
    return {"milestone": MILESTONE, "core_library": str(core_library), "sounds": rows}
