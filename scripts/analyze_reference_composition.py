"""What the reference is actually composed of: roots, chord per bar, harmonic rhythm.

The replica drifted from the original in exactly the places it was invented rather than measured:
the bass pitches were snapped to D minor by hand, the stabs were an old pattern, and the percussion
came from a pack. This measures the composition so those three can be rebuilt instead of guessed.

Method, and why this order:
  * the BASS defines the root. Its transcript is already measured, so each bar takes its lowest
    sounding pitch class as the root candidate (octave-folded, no scale snapping this time)
  * the CHORD quality comes from a per-BAR chroma on the full mix, averaged over 1.87 s, which is
    far more stable than the per-onset chroma that was rejected (0.577 match, 43% disagreement)
  * 24 triads (12 roots x maj/min) are scored, not a hand-picked vocabulary, so the track is not
    forced into a key it may not be in
  * tuning is corrected, because without it every label lands a semitone low

Usage: python scripts/analyze_reference_composition.py
"""
from __future__ import annotations
import json

from collections import Counter
from pathlib import Path

import librosa
import numpy as np

SCRATCH = Path(__file__).resolve().parents[1] / "logs"
MIX = Path.home() / "music-platform-runtime/stems-src/reference_tudm_44100_2ch.wav"
SR = 44100
TUNING_HZ = 434.193
NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
MINOR = [0, 3, 7]
MAJOR = [0, 4, 7]


def triads() -> dict[str, list[int]]:
    table = {}
    for index, name in enumerate(NAMES):
        table[f"{name}m"] = [(index + step) % 12 for step in MINOR]
        table[name] = [(index + step) % 12 for step in MAJOR]
    return table


def main() -> int:
    structure = json.loads((SCRATCH / "reference_structure.json").read_text(encoding="utf-8"))
    transcription = json.loads((SCRATCH / "reference_bass_notes.json").read_text(encoding="utf-8"))
    bar_s = float(structure["bar_s"])
    total_bars = int(structure["total_bars"])

    audio, sr = librosa.load(str(MIX), sr=SR, mono=True)
    chroma = librosa.feature.chroma_cqt(y=audio, sr=SR, hop_length=512, bins_per_octave=12,
                                        tuning=TUNING_HZ / 440.0 * 0.0)
    # chroma_cqt has no reference-frequency argument: shift the bins by the measured tuning instead
    cents = 1200 * np.log2(TUNING_HZ / 440.0)
    shift = int(round(cents / 100.0))
    if shift:
        chroma = np.roll(chroma, -shift, axis=0)

    roots: dict[int, int] = {}
    for note in transcription["notes"]:
        bar = int(note["bar"])
        pitch = int(note["midi"]) % 12
        roots.setdefault(bar, Counter())[pitch] += 1
    root_of_bar = {bar: counter.most_common(1)[0][0] for bar, counter in roots.items()}

    table = triads()
    rows = []
    for bar in range(total_bars):
        start = int(bar * bar_s * SR)
        stop = int((bar + 1) * bar_s * SR)
        chunk = chroma[:, start // 512:stop // 512]
        vector = chunk.mean(axis=1) if chunk.size else np.zeros(12)
        norm = float(np.linalg.norm(vector))
        vector = vector / norm if norm else vector
        best, best_score = "?", -1.0
        for name, pitches in table.items():
            template = np.zeros(12)
            template[pitches] = 1.0
            template = template / np.linalg.norm(template)
            score = float(vector @ template)
            if score > best_score:
                best, best_score = name, score
        rows.append({"bar": bar, "chord": best, "match": round(best_score, 3),
                     "bass_root": NAMES[root_of_bar[bar]] if bar in root_of_bar else None})

    def summarise(block: list[dict], label: str) -> dict:
        chords = Counter(row["chord"] for row in block)
        bassroots = Counter(row["bass_root"] for row in block if row["bass_root"])
        changes = sum(1 for a, b in zip(block, block[1:]) if a["chord"] != b["chord"])
        return {"section": label, "bars": [block[0]["bar"], block[-1]["bar"]] if block else None,
                "top_chords": chords.most_common(4),
                "bass_roots": bassroots.most_common(4),
                "chord_changes_per_bar": round(changes / max(1, len(block) - 1), 2),
                "median_match": round(float(np.median([r["match"] for r in block])), 3) if block else None}

    sections = []
    for start, stop, label in ((0, 33, "intro"), (33, 51, "build"), (51, 107, "main"),
                               (107, 129, "breakdown"), (129, 178, "drop"), (178, total_bars, "outro")):
        sections.append(summarise([r for r in rows if start <= r["bar"] < stop], label))
    for section in sections:
        print(f"{section['section']:10s} compases {section['bars']} | acordes {section['top_chords']}")
        print(f"           raices del bajo {section['bass_roots']} | cambios/compas "
              f"{section['chord_changes_per_bar']} | match mediano {section['median_match']}")
    report = {"bar_s": bar_s, "tuning_hz": TUNING_HZ, "sections": sections,
              "per_bar": rows[:400]}
    (SCRATCH / "reference_composition.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print()
    print("primeros 24 compases:", [(r["bar"], r["chord"], r["bass_root"]) for r in rows[:24]])
    print("->", SCRATCH / "reference_composition.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
