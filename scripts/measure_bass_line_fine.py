"""The bass line at note resolution: pitch, length and accent per note.

The first attempt took one pitch per 8-bar block, which produced 69 consecutive bars of D2 and
no dynamics: a machine-gun drone, not a bassline. This measures what a bassline actually is:

  * pitch      : yin per frame on the isolated bass stem, read at each onset
  * length     : the gap to the following onset (capped), so legato stays legato
  * accent     : velocity from the local level against the section's own median
  * tremolo    : notes closer than a 16th of each other are re-articulations of one note,
                 so they are merged instead of machine-gunned

Every number here comes from the audio; nothing is assumed.

Usage: python scripts/measure_bass_line_fine.py
"""
from __future__ import annotations
import argparse, json

from pathlib import Path

import numpy as np

import essentia
import essentia.standard as es

essentia.log.warningActive = False

SCRATCH = Path(__file__).resolve().parents[1] / "logs"
STEM = Path.home() / "music-platform-runtime/stems/ft/htdemucs_ft/bass.wav"
SR = 44100
FRAME, HOP = 2048, 256


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stem", type=Path, default=STEM)
    parser.add_argument("--rhythm", type=Path, default=SCRATCH / "reference_rhythm_essentia.json")
    parser.add_argument("--json", type=Path, default=SCRATCH / "reference_bass_fine.json")
    args = parser.parse_args()

    audio = es.MonoLoader(filename=str(args.stem), resampleQuality=4)()
    rhythm = json.loads(args.rhythm.read_text(encoding="utf-8"))
    beat_s = 60.0 / float(rhythm["bpm"])
    sixteenth = beat_s / 4.0
    onsets = [t for t in rhythm["bass_onset_times_s"]]

    # pitch and level per frame
    window = es.Windowing(type="hann")
    spectrum = es.Spectrum()
    yin = es.PitchYinFFT(frameSize=FRAME)
    pitches, confidences, levels, times = [], [], [], []
    for frame in es.FrameGenerator(audio, frameSize=FRAME, hopSize=HOP, startFromZero=True):
        spec = spectrum(window(frame))
        f0, conf = yin(spec)
        pitches.append(float(f0))
        confidences.append(float(conf))
        levels.append(float(np.sqrt(np.mean(frame ** 2))))
        times.append(len(times) * HOP / SR)
    pitches = np.asarray(pitches)
    confidences = np.asarray(confidences)
    levels = np.asarray(levels)
    times = np.asarray(times)

    def midi_of(hz: float) -> int | None:
        return int(round(69 + 12 * np.log2(hz / 440.0))) if 25.0 < hz < 400.0 else None

    median_level = float(np.median(levels)) or 1e-9
    events = []
    for index, onset in enumerate(onsets):
        nxt = onsets[index + 1] if index + 1 < len(onsets) else onset + sixteenth
        mask = (times >= onset) & (times < min(onset + 0.15, nxt))
        if not mask.any():
            continue
        voiced = mask & (confidences > 0.4)
        window_mask = voiced if voiced.any() else mask
        hz = float(np.median(pitches[window_mask]))
        midi = midi_of(hz)
        if midi is None:
            continue
        level = float(np.max(levels[window_mask]))
        velocity = int(np.clip(80 + 40 * (level / median_level - 1), 55, 122))
        gap = min(nxt - onset, beat_s)                 # cap at one beat
        events.append({"start_s": round(onset, 4), "pitch": midi,
                       "length_s": round(gap, 4), "velocity": velocity,
                       "level": round(level, 5)})

    # merge re-articulations closer than a 16th with the same pitch: that is one note
    merged: list[dict] = []
    for event in events:
        if merged and event["pitch"] == merged[-1]["pitch"] and \
                event["start_s"] - merged[-1]["start_s"] < sixteenth * 1.5:
            merged[-1]["length_s"] = round(event["start_s"] + event["length_s"]
                                           - merged[-1]["start_s"], 4)
            merged[-1]["velocity"] = max(merged[-1]["velocity"], event["velocity"])
            continue
        merged.append(event)

    report = {"stem": str(args.stem), "bpm": rhythm["bpm"], "sixteenth_s": sixteenth,
              "onsets": len(onsets), "notes": len(events), "notes_after_merge": len(merged),
              "events": merged}
    args.json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    uniq, counts = np.unique([e["pitch"] for e in merged], return_counts=True)
    print(f"onsets {len(onsets)} -> notas {len(events)} -> tras fusionar re-articulaciones "
          f"< 16vo: {len(merged)}")
    print("alturas (MIDI: notas):", {int(u): int(c) for u, c in zip(uniq, counts)})
    lengths = [e["length_s"] for e in merged]
    print(f"duraciones: min={min(lengths):.3f}s mediana={np.median(lengths):.3f}s "
          f"max={max(lengths):.3f}s (16vo={sixteenth:.3f}s)")
    vels = [e["velocity"] for e in merged]
    print(f"velocidades: min={min(vels)} mediana={int(np.median(vels))} max={max(vels)}")
    print()
    print("=== primeros 16 eventos: se mueve dentro del compas? ===")
    for event in merged[:16]:
        bar = event["start_s"] / (beat_s * 4.0)
        print(f"  t={event['start_s']:>7.2f}s (compas {bar:>6.2f}) pitch={event['pitch']:>3} "
              f"dur={event['length_s']:.3f}s vel={event['velocity']}")
    print()
    print("->", args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
