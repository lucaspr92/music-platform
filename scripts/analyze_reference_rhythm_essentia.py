"""Beats and onsets from Essentia, then onsets folded onto the beat grid.

The grid is not assumed here: RhythmExtractor2013 measures it, and every onset is quantised
to the nearest 16th of the measured beats, so the rhythm comes from the audio rather than
from a threshold someone picked.

Usage: python scripts/analyze_reference_rhythm_essentia.py
"""
from __future__ import annotations
import argparse, json

from pathlib import Path

import numpy as np

import essentia
import essentia.standard as es

essentia.log.warningActive = False

SCRATCH = Path(__file__).resolve().parents[1] / "logs"
AUDIO = Path.home() / "music-platform-runtime/stems-src/reference_tudm_44100_2ch.wav"
STEMS = Path.home() / "music-platform-runtime/stems/ft/htdemucs_ft"
# guitar and piano only exist in the 6-stem model, so each stem is looked up where it lives
STEM_DIRS = {"bass": STEMS, "drums": STEMS, "vocals": STEMS, "other": STEMS,
             "guitar": Path.home() / "music-platform-runtime/stems/6s/htdemucs_6s",
             "piano": Path.home() / "music-platform-runtime/stems/6s/htdemucs_6s"}


def onsets_of(path: Path) -> list[float]:
    audio = es.MonoLoader(filename=str(path), resampleQuality=4)()
    times, _rate = es.OnsetRate()(audio)
    return [round(float(t), 4) for t in times]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", type=Path, default=AUDIO)
    parser.add_argument("--json", type=Path, default=SCRATCH / "reference_rhythm_essentia.json")
    args = parser.parse_args()

    audio = es.MonoLoader(filename=str(args.audio), resampleQuality=4)()
    bpm, beats, confidence, _estimates, _intervals = es.RhythmExtractor2013(method="multifeature")(audio)
    report = {
        "bpm": round(float(bpm), 3),
        "beats_confidence": round(float(confidence), 4),
        "n_beats": len(beats),
        "first_beats_s": [round(float(b), 3) for b in beats[:8]],
        "last_beat_s": round(float(beats[-1]), 3) if len(beats) else None,
    }
    print(f"RhythmExtractor2013 multifeature: {report['bpm']} BPM "
          f"(conf {report['beats_confidence']}) | {report['n_beats']} beats")

    beat_s = 60.0 / float(bpm)
    sixteenth = beat_s / 4.0
    print(f"beat = {beat_s:.4f}s | 16vo = {sixteenth:.4f}s")

    for stem in ("bass", "guitar"):
        times = onsets_of(STEM_DIRS.get(stem, STEMS) / f"{stem}.wav")
        steps: dict[int, int] = {}
        for t in times:
            step = int(round(t / sixteenth))
            steps[step] = steps.get(step, 0) + 1
        report[f"{stem}_onsets"] = len(times)
        report[f"{stem}_onset_times_s"] = times
        # a 4-bar rolling picture: which 16th-of-the-bar carries the most onsets
        per_bar = np.zeros(16)
        for step, count in steps.items():
            per_bar[step % 16] += count
        report[f"{stem}_bar_histogram"] = [int(x) for x in per_bar]
        report[f"{stem}_active_sixteenths"] = int((per_bar > 0).sum())
        print(f"  {stem}: {len(times)} onsets | 16vos del compas con onsets: "
              f"{int((per_bar > 0).sum())}/16 | {[int(x) for x in per_bar]}")
    report["sixteenth_s"] = round(sixteenth, 4)
    args.json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("->", args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
