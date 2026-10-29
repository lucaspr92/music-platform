"""Find the risers and impacts at the section boundaries, so the replica can have them.

The replica currently has none, and the reference uses them at every transition: a riser builds
before a boundary, an impact lands on it. Both are detectable without a model:

  riser  : high-band energy rising monotonically over the bars before a boundary
  impact : a broadband burst in the first bar after a boundary, above the surrounding bars

Detection only. Placement is a separate decision, and a detector that finds nothing is a valid
answer: reporting "no riser detected here" is better than inventing one.

Usage: python scripts/detect_transitions_fx.py
"""
from __future__ import annotations
import json

from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

SCRATCH = Path(__file__).resolve().parents[1] / "logs"
MIX = Path.home() / "music-platform-runtime/stems-src/reference_tudm_44100_2ch.wav"
SR = 44100
HIGH = (4000.0, 16000.0)
BROAD = (40.0, 16000.0)


def band_energy_per_bar(mono: np.ndarray, sr: int, bar_s: float, band: tuple[float, float]) -> np.ndarray:
    from scipy.signal import butter, sosfiltfilt
    nyquist = sr / 2.0
    sos = butter(4, [band[0] / nyquist, min(band[1] / nyquist, 0.99)], btype="band", output="sos")
    filtered = sosfiltfilt(sos, mono).astype(np.float32)
    frame = max(1, int(bar_s * sr))
    bars = filtered.size // frame
    levels = []
    for bar in range(bars):
        chunk = filtered[bar * frame:(bar + 1) * frame]
        levels.append(float(np.sqrt(np.mean(chunk ** 2))) if chunk.size else 0.0)
    return 20 * np.log10(np.array(levels) + 1e-9)


def main() -> int:
    mono, sr = librosa.load(str(MIX), sr=SR, mono=True)
    structure = json.loads((SCRATCH / "reference_structure.json").read_text(encoding="utf-8"))
    bar_s = float(structure["bar_s"])
    boundaries = sorted({int(s["start_bar"]) for s in structure.get("sections", [])
                         if int(s["start_bar"]) > 0})
    high = band_energy_per_bar(mono, sr, bar_s, HIGH)
    broad = band_energy_per_bar(mono, sr, bar_s, BROAD)
    print("fronteras:", boundaries[:12])

    events = []
    for boundary in boundaries:
        window = high[max(0, boundary - 8):boundary]
        riser = None
        if window.size >= 4:
            slope = np.polyfit(np.arange(window.size), window, 1)[0]
            span = float(window[-1] - window[0])
            if slope > 0.35 and span > 3.0:
                riser = {"type": "riser", "end_bar": boundary, "start_bar": boundary - window.size,
                         "gain_db": round(span, 1), "slope_db_per_bar": round(float(slope), 2)}
        # an impact is a short burst, not a loud bar: measure peak 50 ms RMS against the
        # neighbourhood, because per-bar RMS averages the burst away (that found zero of them).
        impact = None
        burst = 0.05
        step = max(1, int(burst * sr))
        envelope = np.sqrt(np.convolve(mono ** 2, np.ones(step) / step, mode="same"))
        start = int(boundary * bar_s * sr)
        stop = int((boundary + 2) * bar_s * sr)
        if start < envelope.size:
            peak = float(envelope[start:stop].max())
            before = envelope[max(0, int((boundary - 6) * bar_s * sr)):start]
            after = envelope[stop:int((boundary + 5) * bar_s * sr)]
            neighbourhood = np.concatenate([before, after]) if before.size and after.size else before
            if neighbourhood.size:
                reference = float(np.median(neighbourhood))
                delta = 20 * np.log10(peak / (reference + 1e-9))
                if delta > 3.0:
                    # A run of these every 8 bars is a crash or a bar accent, not a one-off
                    # impact: the flag keeps the two from being conflated when placing fx.
                    impact = {"type": "impact", "bar": boundary,
                              "peak_above_neighbourhood_db": round(float(delta), 1),
                              "strong": bool(delta >= 6.0)}
        for event in (riser, impact):
            if event:
                events.append(event)
                print(f"  {event}")
    report = {"source": str(MIX), "bar_s": bar_s, "boundaries": boundaries,
              "high_band_hz": HIGH, "events": events,
              "count": {"riser": sum(1 for e in events if e["type"] == "riser"),
                        "impact": sum(1 for e in events if e["type"] == "impact"),
                        "impact_strong": sum(1 for e in events
                                             if e["type"] == "impact" and e.get("strong"))},
              "note": "a series every 8 bars is a crash or bar accent, not a one-off impact"}
    (SCRATCH / "transitions_fx.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print()
    print("detectados:", report["count"])
    print("->", SCRATCH / "transitions_fx.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
