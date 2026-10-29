"""The reference's bass: what rhythm it plays, on which note, per section.

We already know the pitch content (the melody detector traced a D2 drone at 73.42 Hz) and the
harmony (D minor, tuned -23 cents). What a synthesized bass needs is the RHYTHM, so this folds
the bass stem's onsets onto a 16th grid per section and reports the pattern.

Reported, not acted on: this writes evidence and decides nothing.

Usage: python scripts/measure_bass_line.py [--stems <dir>] [--json <path>]
"""
from __future__ import annotations
import argparse, json

from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

# Scratch artifacts stay in the gitignored logs/ dir, resolved from the repo root so the
# script runs from any working directory.
SCRATCH = Path(__file__).resolve().parents[1] / "logs"
DEFAULT_STEMS = Path.home() / "music-platform-runtime/stems/ft/htdemucs_ft"
# same relative rule the stem measurement uses, so presence means the same thing everywhere
PRESENCE_DROP_DB = 12.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stems", type=Path, default=DEFAULT_STEMS)
    parser.add_argument("--stem", default="bass", help="which stem file to measure")
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--structure", type=Path, default=SCRATCH / "reference_structure.json")
    args = parser.parse_args()
    if args.json is None:
        args.json = SCRATCH / f"reference_{args.stem}_line.json"

    data, sr = sf.read(str(args.stems / f"{args.stem}.wav"), always_2d=True, dtype="float32")
    mono = librosa.to_mono(data.T)
    structure = json.loads(args.structure.read_text(encoding="utf-8"))
    bpm, bar_s = float(structure["bpm"]), float(structure["bar_s"])
    beat_s = bar_s / 4.0

    # onsets at a fine hop, then folded to the nearest 16th within the bar
    onsets = librosa.onset.onset_detect(y=mono, sr=sr, hop_length=512, backtrack=True,
                                        units="time", delta=0.05)
    report: dict = {"stems": str(args.stems), "stem": args.stem, "bpm": bpm, "bar_s": bar_s,
                    "onsets_total": int(len(onsets)), "sections": []}

    # a silence guard is not optional: onset detectors fire on noise floor and yin returns
    # nonsense on a silent stem, which would fabricate a bass note where none is played
    rms_db = []
    for section in structure.get("sections", []):
        a, b = int(float(section["start_s"]) * sr), int(float(section["end_s"]) * sr)
        chunk = mono[a:b]
        rms_db.append(20 * np.log10(float(np.sqrt(np.mean(chunk ** 2)) + 1e-12)))
    median_db = float(np.median(rms_db))
    report["stem_median_db"] = round(median_db, 1)
    report["presence_rule"] = f"present above median - {PRESENCE_DROP_DB} dB"

    for index, section in enumerate(structure.get("sections", [])):
        a, b = float(section["start_s"]), float(section["end_s"])
        present = rms_db[index] > median_db - PRESENCE_DROP_DB
        if not present:
            report["sections"].append({
                "start_bar": section["start_bar"], "end_bar": section.get("end_bar"),
                "present": False, "rms_db": round(rms_db[index], 1),
                "onsets": 0, "onsets_per_bar": 0.0,
                "grid_16": [0.0] * 16, "grid_16_counts": [0] * 16,
                "pitch_hz": None, "midi": None,
            })
            continue
        inside = onsets[np.logical_and(onsets >= a, onsets < b)]
        grid = np.zeros(16)
        for t in inside:
            grid[int(round(((t - a) / beat_s) * 4)) % 16] += 1
        bars = max(1.0, (b - a) / bar_s)
        share = grid / grid.sum() if grid.sum() else grid
        report["sections"].append({
            "start_bar": section["start_bar"], "end_bar": section.get("end_bar"),
            "present": True, "rms_db": round(rms_db[index], 1),
            "onsets": int(len(inside)), "onsets_per_bar": round(len(inside) / bars, 2),
            "grid_16": [round(float(x), 3) for x in share],
            "grid_16_counts": [int(x) for x in grid],
        })

    # pitch: the fundamental the bass actually sits on, section by section
    for row in report["sections"]:
        section = next(s for s in structure["sections"] if s["start_bar"] == row["start_bar"])
        a, b = int(float(section["start_s"]) * sr), int(float(section["end_s"]) * sr)
        chunk = mono[a:b]
        if not row.get("present") or chunk.size < sr:
            row["pitch_hz"], row["midi"] = None, None
            continue
        # fmin 43 keeps at least two periods inside the 2048-sample frame, which is what
        # keeps yin's estimate trustworthy at the low end
        f0 = librosa.yin(chunk, fmin=43, fmax=200, sr=sr)
        f0 = f0[np.isfinite(f0)]
        hz = float(np.median(f0)) if f0.size else None
        row["pitch_hz"] = round(hz, 2) if hz else None
        row["midi"] = int(round(69 + 12 * np.log2(hz / 440.0))) if hz and hz > 20 else None

    args.json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"onsets del stem '{args.stem}': {report['onsets_total']} en toda la pista ({bpm} BPM)")
    print()
    print(f"mediana del stem: {report['stem_median_db']} dB | regla: {report['presence_rule']}")
    print()
    print(f"{'compas':>7} {'dB':>7} {'ons/comp':>8} {'nota':>6} {'Hz':>8}  grilla de 16vos")
    print("-" * 96)
    for row in report["sections"]:
        if not row.get("present"):
            print(f"{row['start_bar']:>7.0f} {row['rms_db']:>7.1f} {'-- sin bajo --':>25}")
            continue
        counts = row["grid_16_counts"]
        order = sorted(range(16), key=lambda i: -counts[i])[:5]
        marks = " ".join(f"{i + 1}vo={counts[i]}" for i in order if counts[i])
        note = f"M{row['midi']}" if row["midi"] else "-"
        print(f"{row['start_bar']:>7.0f} {row['rms_db']:>7.1f} {row['onsets_per_bar']:>8.2f} "
              f"{note:>6} {row['pitch_hz'] or 0:>8.2f}  {marks}")
    print()
    print("->", args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
