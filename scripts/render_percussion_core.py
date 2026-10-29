"""Render the percussive core: kick and hats at the reference's measured onsets.

What is measured and what is assumed, stated explicitly:
  * kick  : measured. 40-120 Hz band of the drums stem, 515 onsets after the silence guard.
  * hats  : measured. 6-16 kHz band, 2300 onsets.
  * clap  : NOT measurable from this stem. Every band of the drums stem yields ~11 onsets per
            bar because hi-hat transients dominate all of them, so a clap cannot be isolated.
            The backbeat below is an ASSUMPTION (beats 2 and 4), gated by the measured presence
            of the 500-2000 Hz band in the mix.

One-shots come from the Ableton core library kit that already ran on this machine, resolved the
same way the kit resolves them.

Usage: python scripts/render_percussion_core.py
"""
from __future__ import annotations
import json

from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from copilot.platform.runtime_paths import ableton_core_library
from copilot.producer import tech_house_kit_v1 as kit

SCRATCH = Path(__file__).resolve().parents[1] / "logs"
ONSETS = SCRATCH / "reference_band_onsets.json"
STRUCTURE = SCRATCH / "reference_structure.json"
SR = 44100
PRESENCE_DROP_DB = 12.0
GAINS = {"kick": 0.85, "hats": 0.42, "clap": 0.55}


def load_oneshot(relative_path: str) -> tuple[np.ndarray, int]:
    path = kit.resolve_relative(ableton_core_library(), relative_path)
    if not path.is_file():
        raise FileNotFoundError(f"CORE_ONESHOT_MISSING: {path}")
    audio, sr = sf.read(str(path), always_2d=True, dtype="float32")
    mono = audio.mean(axis=1)
    # core library one-shots ship at 96 kHz; resample instead of refusing the file
    if sr != SR:
        mono = librosa.resample(mono, orig_sr=sr, target_sr=SR)
    return mono.astype(np.float32), SR


def place(buffer: np.ndarray, sample: np.ndarray, start_s: float, gain: float) -> int:
    start = int(round(start_s * SR))
    stop = min(start + len(sample), len(buffer))
    if start < 0 or start >= len(buffer):
        return 0
    buffer[start:stop] += sample[: stop - start] * gain
    return 1


def main() -> int:
    onsets = json.loads(ONSETS.read_text(encoding="utf-8"))
    structure = json.loads(STRUCTURE.read_text(encoding="utf-8"))
    bar_s = float(structure["bar_s"])
    total_bars = int(structure["total_bars"])
    track_s = total_bars * 4.0 * bar_s / 4.0 + 2.0
    report = {"clap_pattern": "ASSUMED backbeat on beats 2 and 4",
              "clap_reason": "not isolable: every band of the drums stem gives ~11 onsets/bar",
              "elements": {}}

    sources = {"kick": "Samples/One Shots/Drums/Kick/Kick 909 1.aif",
               "hats": "Samples/One Shots/Drums/Hihat/Hihat Closed 909.aif",
               "clap": "Samples/One Shots/Drums/Clap/Clap 909.aif"}

    buffers = {}
    for element in ("kick", "hats", "clap"):
        sample, sr = load_oneshot(sources[element])
        if sr != SR:
            raise SystemExit(f"SR_MISMATCH for {element}: {sr}")
        buffer = np.zeros(int(track_s * SR), dtype=np.float32)
        count = 0
        if element == "clap":
            # gate the backbeat by the measured presence of the 500-2000 Hz band
            levels = []
            for section in structure["sections"]:
                band = section["perc_band"]["snare_clap"]
                levels.append(float(band))
            median = float(np.median(levels))
            for section in structure["sections"]:
                if float(section["perc_band"]["snare_clap"]) <= median - PRESENCE_DROP_DB:
                    continue
                start_bar = int(section["start_bar"])
                end_bar = start_bar + int(round(float(section["bars"])))
                for bar in range(start_bar, min(end_bar, total_bars)):
                    for beat in (1, 3):          # beats 2 and 4, zero-indexed
                        count += place(buffer, sample, (bar * 4 + beat) * bar_s / 4.0,
                                       GAINS[element])
        else:
            for t in onsets["families"][element]["times_s"]:
                count += place(buffer, sample, float(t), GAINS[element])
        peak = float(np.max(np.abs(buffer)) or 1.0)
        if peak > 0.99:
            buffer *= 0.99 / peak
        out = SCRATCH / f"core_{element}.wav"
        sf.write(str(out), buffer, SR)
        bars = len(buffer) / SR / bar_s
        report["elements"][element] = {"instances": count, "per_bar": round(count / bars, 2),
                                       "source": sources[element], "gain": GAINS[element],
                                       "wav": str(out), "measured": element != "clap"}
        print(f"  {element:5s} {count:>5} golpes  {count / bars:5.2f}/compas  "
              f"{'medido' if element != 'clap' else 'SUPUESTO'}  -> {out.name}")

    (SCRATCH / "percussion_core.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
