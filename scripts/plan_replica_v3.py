"""Plan v3: the arrangement plus the two synthesised parts, each measurement from the library
whose job it is.

  * section bounds and presence : demucs stems (ft for bass, 6s for the stab layer)
  * grid                        : Essentia RhythmExtractor2013 (128.008 BPM, 16th = 0.1172s)
  * rhythm                      : Essentia onsets per stem, measured, not thresholded by hand
  * harmony                     : Essentia KeyExtractor/ChordsDetection (D minor)

Region: the bass is a ROLLING 16th line (Essentia found onsets on 16 of 16 sixteenths), so it
is written as such; the stab layer is syncopated and only its measured accents are written.
Both are played on Serum 2, whose browser URIs were resolved by walking the browser by path.

Usage: python scripts/plan_replica_v3.py
"""
from __future__ import annotations
import json

from pathlib import Path

SCRATCH = Path(__file__).resolve().parents[1] / "logs"
V2 = SCRATCH / "tribal_replica_plan_v2.json"
BASS_LINE = SCRATCH / "reference_bass_line.json"
STAB_LINE = SCRATCH / "reference_guitar_line.json"
RHYTHM = SCRATCH / "reference_rhythm_essentia.json"
OUT = SCRATCH / "tribal_replica_plan_v3.json"

SERUM2_DEVICE_URI = "query:Plugins#VST3:Xfer%20Records:Serum%202"
BASS_PRESET_URI = "query:Plugins#VST3:Xfer%20Records:Serum%202:FileId_104823"
STAB_CHORD = [50, 53, 57]        # D3 F3 A3 = D minor triad
BASS_FOLD_ABOVE = 45             # fold the octave down: yin locks to the octave when the
                                 # fundamental is weak, and the bass register is the low one
ACCENT_FLOOR_RATIO = 0.25        # a 16th is accented when its measured count reaches this
                                 # fraction of the busiest 16th of that layer


def bounds(sections: list[dict], total_bars: int) -> dict[int, int]:
    """end_bar comes from the structure's own `bars` field, which is the one that exists."""
    out = {}
    for section in sections:
        start = int(section["start_bar"])
        length = section.get("bars")
        end = start + int(round(length)) if length else None
        out[start] = min(int(end), total_bars) if end else total_bars
    ordered = sorted(out)
    for i, start in enumerate(ordered[:-1]):
        if out[start] > ordered[i + 1]:
            out[start] = ordered[i + 1]
    return out


def notes_at_onsets(onset_times: list[float], beat_s: float, pitch: int,
                    bounds_by_start: dict[int, int], present_starts: list[int],
                    duration: float, velocity: int, chord: list[int] | None = None
                    ) -> list[dict]:
    """A note at every MEASURED onset, quantised to the 16th of the measured grid.

    Thresholding an aggregated histogram invents density: the stab layer carries 174 onsets
    over the whole track (about 1.4 per bar), so writing eight chord hits per bar by ratio
    would be fabrication. Onsets are the datum, so the notes follow them.
    """
    sixteenth = beat_s / 4.0
    pitches = chord if chord else [pitch]
    out: list[dict] = []
    for t_s in onset_times:
        bar_float = t_s / (beat_s * 4.0)
        bar = int(bar_float)
        if not any(start <= bar < bounds_by_start[start] for start in present_starts):
            continue                      # the stem is silent here: an onset is noise floor
        step = int(round((bar_float - bar) * 16)) % 16
        start = bar * 4.0 + step * 0.25
        for pitch_value in pitches:
            out.append({"pitch": int(pitch_value), "start": round(start, 4),
                        "duration": duration, "velocity": velocity})
    # quantising can land two onsets on the same 16th: keep one note per pitch and step
    seen: set[tuple[int, float]] = set()
    unique: list[dict] = []
    for note in out:
        key = (note["pitch"], note["start"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(note)
    return unique


def main() -> int:
    v2 = json.loads(V2.read_text(encoding="utf-8"))
    bass_line = json.loads(BASS_LINE.read_text(encoding="utf-8"))
    stab_line = json.loads(STAB_LINE.read_text(encoding="utf-8"))
    rhythm = json.loads(RHYTHM.read_text(encoding="utf-8"))
    total_bars = int(v2["total_bars"])

    bass_bounds = bounds(bass_line["sections"], total_bars)
    stab_bounds = bounds(stab_line["sections"], total_bars)
    stab_histogram = rhythm["guitar_bar_histogram"]

    beat_s = 60.0 / float(rhythm["bpm"])
    bass_present = [int(s["start_bar"]) for s in bass_line["sections"] if s.get("present")]
    stab_present = [int(s["start_bar"]) for s in stab_line["sections"] if s.get("present")]

    # one pitch per bar, from the section the bar falls in; the octave is folded because the
    # estimate is unreliable when the fundamental is weak and the bass register is the low one
    pitch_by_bar: dict[int, int] = {}
    for section in bass_line["sections"]:
        if not section.get("present") or not section.get("midi"):
            continue
        start = int(section["start_bar"])
        pitch = int(section["midi"])
        if pitch >= BASS_FOLD_ABOVE:
            pitch -= 12
        for bar in range(start, bass_bounds[start]):
            pitch_by_bar[bar] = pitch

    bass_notes: list[dict] = []
    for t_s in rhythm.get("bass_onset_times_s", []):
        bar = int(t_s / (beat_s * 4.0))
        if bar not in pitch_by_bar:
            continue
        step = int(round((t_s / (beat_s * 4.0) - bar) * 16)) % 16
        bass_notes.append({"pitch": pitch_by_bar[bar],
                           "start": round(bar * 4.0 + step * 0.25, 4),
                           "duration": 0.22, "velocity": 100})
    seen: set[tuple[int, float]] = set()
    bass_unique = []
    for note in bass_notes:
        key = (note["pitch"], note["start"])
        if key in seen:
            continue
        seen.add(key)
        bass_unique.append(note)
    bass_notes = bass_unique

    stab_notes = notes_at_onsets(rhythm.get("guitar_onset_times_s", []), beat_s, 0,
                                 stab_bounds, stab_present, 0.25, 92, chord=STAB_CHORD)

    plan = {
        "version": 3,
        "reference": v2["reference"], "bpm": v2["bpm"], "bar_s": v2["bar_s"],
        "total_bars": total_bars,
        "library": {"palette": v2["palette"], "placements": v2["placements"]},
        "synth": {
            "bass": {"track": "REPLICA - bass (Serum 2)", "device_uri": SERUM2_DEVICE_URI,
                     "preset_uri": BASS_PRESET_URI, "notes": bass_notes,
                     "shape": "rolling 16ths, pitch per section from the bass stem",
                     "source": "Essentia RhythmExtractor2013 grid + Essentia onsets on the "
                               "demucs ft bass stem"},
            "stabs": {"track": "REPLICA - stabs (Serum 2)", "device_uri": SERUM2_DEVICE_URI,
                      "preset_uri": None, "notes": stab_notes, "chord_midi": STAB_CHORD,
                      "shape": "measured accents only",
                      "source": "Essentia onsets on the htdemucs_6s guitar bucket"},
        },
        "rhythm": {"bpm": rhythm["bpm"], "sixteenth_s": rhythm["sixteenth_s"],
                   "bass_bar_histogram": rhythm["bass_bar_histogram"],
                   "guitar_bar_histogram": stab_histogram},
        "evidence": {"bass_median_db": bass_line.get("stem_median_db", bass_line.get("bass_median_db")),
                     "stab_median_db": stab_line.get("stem_median_db", stab_line.get("bass_median_db"))},
        "limits": [
            "no melodic line exists in the reference; the hook is harmonic",
            "stabs use the init patch: Serum 2 exposes one parameter, so no patch shaping",
            "master tune cannot be set to the measured -23 cents yet",
            "preset load is proven to answer, not yet independently verified",
        ],
    }
    OUT.write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")

    bars = [s for s in bass_line["sections"] if s.get("present")]
    onsets_bass = len(rhythm.get("bass_onset_times_s", []))
    onsets_guitar = len(rhythm.get("guitar_onset_times_s", []))
    stab_bars = [s for s in stab_line["sections"] if s.get("present")]
    print(f"plan v3 -> {OUT.name}   {plan['bpm']} BPM, {total_bars} compases")
    print(f"  bajo : {onsets_bass} onsets medidos -> {len(bass_notes)} notas "
          f"(una por onset, cuantizada al 16vo)")
    print(f"  stabs: {onsets_guitar} onsets medidos -> {len(stab_notes) // len(STAB_CHORD)} "
          f"golpes del acorde {STAB_CHORD} ({len(stab_notes)} notas)")
    print(f"  sanity: max notas por compas de bajo = "
          f"{max((sum(1 for n in bass_notes if b * 4 <= n['start'] < (b + 1) * 4) for b in range(total_bars)), default=0)}")
    print(f"  primeros compases de bajo: pitch(es) medidos/fold = "
          f"{sorted({n['pitch'] for n in bass_notes})}")
    print(f"  16vos del stab con acento: "
          f"{[i + 1 for i, c in enumerate(stab_histogram) if c >= max(stab_histogram) * ACCENT_FLOOR_RATIO]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
