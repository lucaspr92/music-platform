"""Bassline audio -> notes, with librosa.pyin.

Rejected first and why, so nobody repeats it: `torchcrepe` (CREPE, better on paper) loads its
weights under this runtime's torch but returns a constant 31.7 Hz on a pure 220 Hz sine, i.e. the
inference silently does not work here. Validated with that exact control before trusting any output.

pyin passes the same control: 220 Hz sine -> 220.1 Hz, voiced 1.0. So it is used, with
frame_length 4096 because 2048 cannot hold two periods of 30 Hz and librarosa warns exactly that.

Guards kept from the two failed attempts at this before:
  * silence gate against the stem's own median (no fixed floor: the bass stem sits at -70 dB until
    bar 33 and -15 dB afterwards)
  * note resolution, never coarse blocks (that produced 69 bars of D2, a drone)

Usage: python scripts/transcribe_bass_notes.py --audio bass.wav --out logs/reference_bass_notes.json
"""
from __future__ import annotations
import json

from pathlib import Path

import librosa
import numpy as np

SR = 44100
HOP = 512
FRAME = 4096
FMIN, FMAX = 30.0, 300.0
SILENCE_BELOW_MEDIAN_DB = 12.0
MIN_NOTE_S = 0.055
MAX_GAP_S = 0.035


def transcribe(path: Path, start: float = 0.0, duration: float | None = None) -> dict:
    audio, sr = librosa.load(str(path), sr=SR, mono=True, offset=start, duration=duration)
    f0, voiced_flag, voiced_prob = librosa.pyin(audio, fmin=FMIN, fmax=FMAX, sr=SR,
                                                frame_length=FRAME, hop_length=HOP)
    times = librosa.times_like(f0, sr=SR, hop_length=HOP)

    frame_energy = librosa.feature.rms(y=audio, frame_length=FRAME, hop_length=HOP)[0]
    energy_db = 20 * np.log10(frame_energy + 1e-9)
    energy_db = np.interp(times, librosa.times_like(frame_energy, sr=SR, hop_length=HOP), energy_db)
    audible = energy_db > (np.median(energy_db) - SILENCE_BELOW_MEDIAN_DB)
    voiced = np.logical_and(voiced_flag, audible)
    voiced = np.logical_and(voiced, np.nan_to_num(f0, nan=0.0) > 0)

    midi = np.full(f0.size, np.nan)
    midi[voiced] = 69.0 + 12.0 * np.log2(f0[voiced] / 440.0)

    # A bass note repeated at the same pitch is two notes, not one. Splitting only on pitch change
    # merged them and produced 4.4 notes/bar against a measured 16th grid that is fully occupied,
    # so onsets are a second boundary.
    onset_env = librosa.onset.onset_strength(y=audio, sr=SR, hop_length=HOP)
    onset_frames = set(int(f) for f in librosa.onset.onset_detect(
        onset_envelope=onset_env, sr=SR, hop_length=HOP, units="frames"))

    notes: list[dict] = []
    current: dict | None = None
    last_voiced_time: float | None = None
    for index in range(f0.size):
        time = float(times[index])
        if not voiced[index]:
            if current is not None and last_voiced_time is not None:
                if time - last_voiced_time > MAX_GAP_S:
                    current["end_s"] = last_voiced_time
                    notes.append(current)
                    current = None
            continue
        note = int(round(float(midi[index])))
        rearticulated = index in onset_frames and current is not None and (
            time - current["start_s"]) > MIN_NOTE_S
        if rearticulated:
            current["end_s"] = time
            notes.append(current)
            current = None
        if current is None:
            current = {"start_s": time, "end_s": None, "midi": note,
                       "confidence": [float(voiced_prob[index])]}
        elif note != current["midi"]:
            current["end_s"] = time
            notes.append(current)
            current = {"start_s": time, "end_s": None, "midi": note,
                       "confidence": [float(voiced_prob[index])]}
        else:
            current["confidence"].append(float(voiced_prob[index]))
        last_voiced_time = time
    if current is not None:
        current["end_s"] = last_voiced_time
        notes.append(current)

    cleaned = []
    for note in notes:
        length = float(note["end_s"] - note["start_s"])
        if length < MIN_NOTE_S:
            continue
        cleaned.append({"start_s": round(float(note["start_s"]) + start, 4),
                        "duration_s": round(length, 4), "midi": int(note["midi"]),
                        "confidence": round(float(np.mean(note["confidence"])), 3)})
    return {"audio": str(path), "notes": cleaned, "frames": int(f0.size),
            "voiced_frames": int(voiced.sum()),
            "voiced_ratio": round(float(voiced.mean()), 3),
            "offset_s": start}


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--duration", type=float, default=None)
    argv = parser.parse_args()
    result = transcribe(Path(argv.audio), argv.start, argv.duration)
    structure = json.loads((Path(__file__).resolve().parents[1] / "logs"
                            / "reference_structure.json").read_text(encoding="utf-8"))
    bar_s = float(structure["bar_s"])
    notes = result["notes"]
    histogram: dict[int, int] = {}
    sixteenth_hist: dict[int, int] = {}
    for note in notes:
        note["bar"] = int(note["start_s"] // bar_s)
        note["sixteenth"] = int(round((note["start_s"] % bar_s) / (bar_s / 16))) % 16 + 1
        histogram[note["midi"]] = histogram.get(note["midi"], 0) + 1
        sixteenth_hist[note["sixteenth"]] = sixteenth_hist.get(note["sixteenth"], 0) + 1
    bars_with_notes = len({note["bar"] for note in notes})
    result.update({"notes_count": len(notes),
                   "bars_with_notes": bars_with_notes,
                   "per_bar": round(len(notes) / (bars_with_notes or 1), 2),
                   "midi_histogram": dict(sorted(histogram.items(), key=lambda kv: -kv[1])),
                   "sixteenth_histogram": dict(sorted(sixteenth_hist.items())),
                   "median_duration_s": round(float(np.median(
                       [n["duration_s"] for n in notes])), 3) if notes else None,
                   "bar_s": bar_s})
    Path(argv.out).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"notas: {len(notes)} en {bars_with_notes} compases ({result['per_bar']}/compas) "
          f"| voiced {result['voiced_ratio']}")
    print("midi:", result["midi_histogram"])
    print("16vos:", result["sixteenth_histogram"])
    print("duracion mediana:", result["median_duration_s"], "s")
    print("->", argv.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
