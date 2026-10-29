"""Characterise the reference's vocal layer, so R7 is decided on evidence.

The decision (instrumental replica / library chop / the reference's own stem) needs to know what
the layer actually is: a sustained sung phrase, a chopped word, or speech-like bleed. Measured
here: where it is present, how many events per bar, how long they last, and how voiced they are.

Sampling the record is not an option for a deliverable, whatever this returns; the point is to
know whether a library chop could stand in for it.

Usage: python scripts/measure_vocal_layer.py
"""
from __future__ import annotations
import json

from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

SCRATCH = Path(__file__).resolve().parents[1] / "logs"
VOCALS = Path.home() / "music-platform-runtime/stems/ft/htdemucs_ft/vocals.wav"
SR = 44100


def main() -> int:
    structure = json.loads((SCRATCH / "reference_structure.json").read_text(encoding="utf-8"))
    bar_s = float(structure["bar_s"])
    audio, sr = librosa.load(str(VOCALS), sr=SR, mono=True)
    frame = int(bar_s * sr)
    bars = len(audio) // frame
    levels = []
    for bar in range(bars):
        chunk = audio[bar * frame:(bar + 1) * frame]
        levels.append(20 * np.log10(float(np.sqrt(np.mean(chunk ** 2))) + 1e-9))
    levels = np.array(levels)
    audible = levels > (np.median(levels) + 3.0)
    active_bars = [int(i) for i in np.where(audible)[0]]

    hop = 512
    envelope = librosa.onset.onset_strength(y=audio, sr=SR, hop_length=hop)
    frames = librosa.onset.onset_detect(onset_envelope=envelope, sr=SR, hop_length=hop, units="frames")
    times = librosa.frames_to_time(frames, sr=SR, hop_length=hop)
    onsets_in_active = [float(t) for t in times
                        if len(audio) > 0 and levels[int(min(bars - 1, t // bar_s))] > np.median(levels) + 3.0]
    durations = np.diff(onsets_in_active) if len(onsets_in_active) > 1 else np.array([])

    f0, voiced, prob = librosa.pyin(audio, fmin=80, fmax=600, sr=SR,
                                    frame_length=2048, hop_length=hop)
    f0_voiced = f0[voiced]
    report = {
        "source": str(VOCALS),
        "bars_total": bars,
        "active_bars": len(active_bars),
        "first_active_bar": active_bars[0] if active_bars else None,
        "last_active_bar": active_bars[-1] if active_bars else None,
        "onsets": int(times.size),
        "onsets_in_active": len(onsets_in_active),
        "events_per_active_bar": round(len(onsets_in_active) / (len(active_bars) or 1), 2),
        "median_event_gap_s": round(float(np.median(durations)), 3) if durations.size else None,
        "voiced_ratio": round(float(np.mean(voiced)), 3),
        "pitch_hz_median": round(float(np.median(f0_voiced)), 1) if f0_voiced.size else None,
        "pitch_hz_range": [round(float(np.min(f0_voiced)), 1), round(float(np.max(f0_voiced)), 1)]
        if f0_voiced.size else None,
        "levels_db_sample": [round(float(levels[i]), 1) for i in range(0, min(bars, 200), 16)],
    }
    (SCRATCH / "vocal_layer.json").write_text(json.dumps(report, indent=2, ensure_ascii=False),
                                              encoding="utf-8")
    for key, value in report.items():
        if key != "source":
            print(f"  {key}: {value}")
    print("->", SCRATCH / "vocal_layer.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
