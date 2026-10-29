"""The reference's groove, per percussive family, from the isolated drums stem.

Measured, not assumed: each band of the drums stem gets its own onset times, and onsets are
dropped where that band is silent (a filtered band is mostly noise floor in a breakdown, and
an onset detector happily fires on it).

Usage: python scripts/measure_rhythm_by_band.py
"""
from __future__ import annotations
import json

from pathlib import Path

import numpy as np

import essentia
import essentia.standard as es
from scipy.signal import butter, sosfiltfilt

essentia.log.warningActive = False

SCRATCH = Path(__file__).resolve().parents[1] / "logs"
DRUMS = Path.home() / "music-platform-runtime/stems/ft/htdemucs_ft/drums.wav"
STRUCTURE = SCRATCH / "reference_structure.json"
OUT = SCRATCH / "reference_band_onsets.json"
SR = 44100
FAMILIES = {"kick": (40, 120), "clap": (500, 2000), "hats": (6000, 16000)}
PRESENCE_DROP_DB = 12.0


def band_filter(audio: np.ndarray, lo: float, hi: float, order: int = 4) -> np.ndarray:
    nyquist = SR / 2.0
    sos = butter(order, [lo / nyquist, min(hi / nyquist, 0.99)], btype="band", output="sos")
    return sosfiltfilt(sos, audio).astype(np.float32)


def main() -> int:
    audio = np.asarray(es.MonoLoader(filename=str(DRUMS), resampleQuality=4)(), dtype=np.float32)
    structure = json.loads(STRUCTURE.read_text(encoding="utf-8"))
    report: dict = {"drums": str(DRUMS), "families": {}}

    for family, (lo, hi) in FAMILIES.items():
        filtered = band_filter(audio, lo, hi)
        times, rate = es.OnsetRate()(filtered)
        times = [float(t) for t in times]

        # per-section level of that band, to drop onsets where the family is not playing
        levels = []
        for section in structure["sections"]:
            a, b = int(float(section["start_s"]) * SR), int(float(section["end_s"]) * SR)
            chunk = filtered[a:b]
            levels.append(20 * np.log10(float(np.sqrt(np.mean(chunk ** 2)) + 1e-12)))
        median_db = float(np.median(levels))
        keep_sections = [(s, lvl) for s, lvl in zip(structure["sections"], levels)
                         if lvl > median_db - PRESENCE_DROP_DB]

        kept = []
        for t in times:
            for section, _lvl in keep_sections:
                if float(section["start_s"]) <= t < float(section["end_s"]):
                    kept.append(t)
                    break

        bar_s = float(structure["bar_s"])
        report["families"][family] = {
            "band_hz": [lo, hi], "onsets_raw": len(times), "onsets_kept": len(kept),
            "onsets_per_bar": round(len(kept) / (len(audio) / SR / bar_s), 2),
            "median_band_db": round(median_db, 1),
            "sections_kept": len(keep_sections), "sections_total": len(structure["sections"]),
            "times_s": [round(t, 4) for t in kept],
        }
        print(f"  {family:6s} {lo}-{hi}Hz: {len(times)} onsets -> {len(kept)} tras guarda "
              f"({len(keep_sections)}/{len(structure['sections'])} secciones con la familia)  "
              f"{report['families'][family]['onsets_per_bar']}/compas")

    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("->", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
