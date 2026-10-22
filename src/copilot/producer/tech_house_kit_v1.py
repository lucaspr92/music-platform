"""TECH_HOUSE_PRODUCTION_KIT_V1 — seven real sounds and one 16-bar groove plan.

Pure data and deterministic note generation; nothing here talks to Live.
The orchestrator in ``copilot.integration.tech_house_kit_v1`` turns this into a
MusicPlan and executes it through ProductionCompiler -> SafeWrite.

Sound choice is fixed to the first candidate of each role. Alternatives are
technical fallbacks only (does not load, silent, wrong device, corrupt), never
a taste decision — taste belongs to human audition.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from pathlib import PureWindowsPath
from typing import Any

from copilot.schemas.session import MidiNote

MILESTONE = "TECH_HOUSE_PRODUCTION_KIT_V1"
TEMPO_BPM = 126.0
METER = (4, 4)
BARS = 16
BEATS_PER_BAR = 4
CLIP_BEATS = float(BARS * BEATS_PER_BAR)
# Simpler plays a one-shot at its recorded pitch when triggered at C3.
ONE_SHOT_PITCH = 60
TRACK_PREFIX = "KIT V1 - "

PROVENANCE_CORE_LIBRARY = (
    "Ableton Live Core Library factory content, licensed under the Ableton EULA. "
    "Installed edition is Live 12 Trial: internal audition is fine; commercial "
    "release requires a valid Live licence."
)

# Live's factory content forbids none of these; the list exists so a future
# edit cannot quietly swap a role to a placeholder.
FORBIDDEN_SOURCE_MARKERS = (
    "operator default",
    "basic sine",
    "placeholder",
    "test tone",
    "generated_",
    "stem_v2_",
    "abletunes",
)


@dataclass(frozen=True)
class KitSound:
    role: str
    source_type: str          # "one_shot_sample" | "instrument_preset"
    relative_path: str        # relative to the Core Library root
    browser_name: str         # exact Live Browser item name
    ableton_device: str       # device the sound ends up in
    fallback_relative_path: str
    fallback_browser_name: str
    provenance: str = PROVENANCE_CORE_LIBRARY
    processing: tuple[str, ...] = field(default_factory=tuple)

    @property
    def track_name(self) -> str:
        return TRACK_PREFIX + self.role

    @property
    def is_one_shot(self) -> bool:
        return self.source_type == "one_shot_sample"


_DRUMS = r"Samples\One Shots\Drums"

KIT: tuple[KitSound, ...] = (
    KitSound("Kick", "one_shot_sample", rf"{_DRUMS}\Kick\Kick 909 1.aif", "Kick 909 1.aif", "Simpler",
             rf"{_DRUMS}\Kick\Kick Short and Solid 1.wav", "Kick Short and Solid 1.wav"),
    KitSound("Clap", "one_shot_sample", rf"{_DRUMS}\Clap\Clap 909.aif", "Clap 909.aif", "Simpler",
             rf"{_DRUMS}\Clap\Clap Tight.wav", "Clap Tight.wav"),
    KitSound("Closed Hat", "one_shot_sample", rf"{_DRUMS}\Hihat\Hihat Closed 909.aif", "Hihat Closed 909.aif",
             "Simpler", rf"{_DRUMS}\Hihat\Hihat Closed 808.aif", "Hihat Closed 808.aif"),
    KitSound("Open Hat", "one_shot_sample", rf"{_DRUMS}\Hihat\Hihat Open 909.aif", "Hihat Open 909.aif",
             "Simpler", rf"{_DRUMS}\Hihat\Hihat Open 808.aif", "Hihat Open 808.aif"),
    KitSound("Perc", "one_shot_sample", rf"{_DRUMS}\Shaker\Shaker Short.aif", "Shaker Short.aif", "Simpler",
             rf"{_DRUMS}\Rim\Rim 909.aif", "Rim 909.aif"),
    KitSound("Bass", "instrument_preset", r"Racks\Instrument Racks\Bass\Basic FM House Bass.adg",
             "Basic FM House Bass.adg", "Instrument Rack",
             r"Devices\Instruments\Operator\Bass\House Bass.adv", "House Bass.adv"),
    KitSound("Stab", "instrument_preset", r"Racks\Instrument Racks\Plucked\House Stab.adg",
             "House Stab.adg", "Instrument Rack",
             r"Racks\Instrument Racks\Plucked\Chord Dub Pluck.adg", "Chord Dub Pluck.adg"),
)


def sound(role: str) -> KitSound:
    for item in KIT:
        if item.role == role:
            return item
    raise KeyError(role)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_relative(core_library: Path, relative_path: str) -> Path:
    """Resolve a bank-relative path on any host.

    Live documents its factory content with Windows separators, and on macOS a
    backslash is an ordinary filename character, so joining the raw value onto a
    root looks for one file with separators inside its name. PureWindowsPath
    accepts either separator, which is the same normalisation the sample library
    and the studio workbench already perform.
    """
    return Path(core_library).joinpath(*PureWindowsPath(relative_path).parts)


def kit_definition(core_library: Path) -> dict[str, Any]:
    """Resolve every sound on disk. Missing files fail closed: no substitution."""
    rows = []
    for item in KIT:
        path = resolve_relative(core_library, item.relative_path)
        if not path.is_file():
            raise FileNotFoundError(f"KIT_SOUND_MISSING: {item.role}: {path}")
        rows.append({**asdict(item), "track_name": item.track_name, "path": str(path),
                     "sha256": sha256_file(path), "bytes": path.stat().st_size})
    return {"milestone": MILESTONE, "core_library": str(core_library), "sounds": rows}


# --------------------------------------------------------------- groove ----
def _note(pitch: int, beat: float, duration: float, velocity: int) -> MidiNote:
    return MidiNote(pitch=int(pitch), start_time=round(float(beat), 6),
                    duration=round(float(duration), 6), velocity=int(velocity))


def _bars(first: int, last: int) -> range:
    """1-based inclusive bar numbers -> 0-based bar indexes."""
    return range(first - 1, last)


SECTIONS = (
    {"bars": "1-4", "label": "base groove"},
    {"bars": "5-8", "label": "percussion variation"},
    {"bars": "9-12", "label": "stab enters, bass variation"},
    {"bars": "13-16", "label": "small transition"},
)

# G minor. Ableton convention: C3 = 60, so G1 = 43 (~49 Hz).
KEY = "G minor"
G1, BB1, D2, F1, F2, G2 = 43, 46, 50, 41, 53, 55
GM7_VOICING = (58, 62, 65, 67)  # Bb2 D3 F3 G3

TONIC_SEMITONES = {"C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3, "E": 4, "F": 5,
                   "F#": 6, "Gb": 6, "G": 7, "G#": 8, "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11}


def transposition_to(key: str) -> int:
    """Semitone offset that renders this G-minor material in ``key``.

    Minor only. The kit's material is a minor-key groove, so a major mode fails
    closed instead of silently changing the music. The offset is the shortest
    motion (G -> D is -5, not +7) so the bass keeps its register.
    """
    parts = key.split()
    tonic = parts[0]
    mode = parts[1].lower() if len(parts) > 1 else "minor"
    if mode != "minor":
        raise ValueError(f"UNSUPPORTED_MODE: {key}: the kit material is minor")
    if tonic not in TONIC_SEMITONES:
        raise ValueError(f"UNSUPPORTED_TONIC: {key}")
    delta = (TONIC_SEMITONES[tonic] - TONIC_SEMITONES["G"]) % 12
    return delta - 12 if delta > 6 else delta


def transpose_notes(notes: list[MidiNote], semitones: int) -> list[MidiNote]:
    """Shift pitched notes. The caller decides who is pitched: one-shots are not."""
    if not semitones:
        return list(notes)
    return [note.model_copy(update={"pitch": note.pitch + semitones}) for note in notes]


def kick_notes() -> list[MidiNote]:
    return [_note(ONE_SHOT_PITCH, beat, 0.25, 118) for beat in range(int(CLIP_BEATS))]


def clap_notes() -> list[MidiNote]:
    notes = [_note(ONE_SHOT_PITCH, bar * 4 + offset, 0.25, 108)
             for bar in _bars(1, 16) for offset in (1, 3)]
    # One soft pickup into the last bar's backbeat, nothing more.
    notes.append(_note(ONE_SHOT_PITCH, 15 * 4 + 3.75, 0.2, 64))
    return notes


def closed_hat_notes() -> list[MidiNote]:
    # 16ths on the "e" and "a" of every beat; the offbeat belongs to the open hat.
    notes = []
    for bar in _bars(1, 16):
        for beat in range(4):
            base = bar * 4 + beat
            notes.append(_note(ONE_SHOT_PITCH, base + 0.25, 0.1, 62 if bar % 2 == 0 else 66))
            notes.append(_note(ONE_SHOT_PITCH, base + 0.75, 0.1, 78))
    return notes


def open_hat_notes() -> list[MidiNote]:
    notes = []
    for bar in _bars(1, 16):
        if bar == 14:  # bar 15: open hat drops out for the transition
            continue
        for beat in range(4):
            notes.append(_note(ONE_SHOT_PITCH, bar * 4 + beat + 0.5, 0.25, 96))
    return notes


_PERC_BASE = (0.75, 2.25, 3.75)
_PERC_VARIATION = (0.75, 1.5, 2.25, 3.25, 3.75)


def perc_notes() -> list[MidiNote]:
    notes = []
    for bar in _bars(1, 16):
        if bar == 15:  # bar 16: short 16th fill on beats 3-4
            for step in range(8):
                notes.append(_note(ONE_SHOT_PITCH, bar * 4 + 2 + step * 0.25, 0.1, 60 + step * 5))
            continue
        offsets = _PERC_VARIATION if bar in _bars(5, 8) or bar in _bars(13, 15) else _PERC_BASE
        notes.extend(_note(ONE_SHOT_PITCH, bar * 4 + offset, 0.1, 84) for offset in offsets)
    return notes


# Two-bar bass phrases. Every onset is off the beat so the bass never lands on
# a kick (sidechain is unavailable; the line is written around the kick).
_BASS_A = ((0.5, G1), (1.5, G1), (1.75, G2), (2.5, G1), (3.5, G1),
           (4.5, G1), (5.5, G1), (6.5, BB1), (7.5, F1), (7.75, F1))
_BASS_B = ((0.5, G1), (1.5, G1), (1.75, G2), (2.5, D2), (3.5, G1),
           (4.5, G1), (5.5, BB1), (6.5, G1), (7.5, F1), (7.75, F2))


def bass_notes() -> list[MidiNote]:
    notes = []
    for phrase in range(8):  # 8 two-bar phrases = 16 bars
        first_bar = phrase * 2
        cells = _BASS_B if first_bar in _bars(9, 12) else _BASS_A
        for offset, pitch in cells:
            notes.append(_note(pitch, first_bar * 4 + offset, 0.22, 104 if offset % 1 == 0.5 else 92))
    return notes


_STAB_CELL = (0.75, 2.25, 5.75)  # beats within a two-bar phrase


def stab_notes() -> list[MidiNote]:
    notes = []
    for phrase in range(4, 8):  # bars 9-16 only
        for offset in _STAB_CELL:
            for pitch in GM7_VOICING:
                notes.append(_note(pitch, phrase * 8 + offset, 0.2, 96))
    return notes


GENERATORS = {
    "Kick": kick_notes,
    "Clap": clap_notes,
    "Closed Hat": closed_hat_notes,
    "Open Hat": open_hat_notes,
    "Perc": perc_notes,
    "Bass": bass_notes,
    "Stab": stab_notes,
}


def groove_plan(tempo_bpm: float = TEMPO_BPM, key: str = KEY) -> dict[str, Any]:
    """Structured 16-bar plan; every role is one 64-beat clip placed at bar 1.

    ``tempo_bpm`` and ``key`` default to the frozen V1 material, so the default
    plan carries the same tempo, key and notes that shipped. Any other MINOR key
    transposes the PITCHED roles only: one-shot roles trigger at
    ``ONE_SHOT_PITCH``, and moving that pitch would select a different sample
    pitch rather than transpose the key.
    """
    semitones = transposition_to(key)
    roles = {}
    for item in KIT:
        notes = GENERATORS[item.role]()
        if not item.is_one_shot:
            notes = transpose_notes(notes, semitones)
        roles[item.role] = {
            "track_name": item.track_name,
            "clip_index": 0,
            "length_beats": CLIP_BEATS,
            "arrangement_start_beat": 0.0,
            "note_count": len(notes),
            "notes": [note.model_dump(mode="json") for note in notes],
        }
    return {
        "milestone": MILESTONE,
        "tempo_bpm": float(tempo_bpm),
        "meter": list(METER),
        "bars": BARS,
        "key": key,
        "key_transposition_semitones": semitones,
        "sections": list(SECTIONS),
        "sidechain": "BLOCKED_BY_CONTROL_SURFACE",
        "sidechain_note": "No kick->bass sidechain or emulated ducking. Bass onsets avoid every kick.",
        "roles": roles,
    }
