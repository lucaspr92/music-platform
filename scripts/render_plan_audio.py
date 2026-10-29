"""Render a replica plan to audio offline, so the similarity metric does not need Live.

The placement logic here mirrors build_replica_v3.py on purpose: same step computation, same
run bounds, same per-track volume. If the two ever diverge, the score stops describing what the
set actually contains, so keep them in step.

Synth tracks are skipped: their sound comes from Serum, which cannot be rendered offline. The
report says so explicitly instead of pretending the render is the whole mix.

Usage: python scripts/render_plan_audio.py [--plan logs/tribal_replica_plan_v3_2.json]
"""
from __future__ import annotations
import argparse, json

from pathlib import Path

import numpy as np
import soundfile as sf

SCRATCH = Path(__file__).resolve().parents[1] / "logs"
SR = 44100


def synth_part(notes: list[dict], beat_s: float, duration_s: float, sr: int,
               kind: str) -> np.ndarray:
    """A simple, honest stand-in for the plugin: enough to hear the PART.

    The plugin cannot be rendered offline, so the bass is a filtered saw plus a sub sine and the
    stabs are a short bandpassed saw cluster. This is not the reference's patch and does not claim
    to be: it exists so a listener can judge the rhythm and the arrangement instead of a number.
    """
    from scipy.signal import butter, sosfilt
    buffer = np.zeros(int(duration_s * sr) + sr, dtype=np.float32)
    for note in notes:
        start = int(round(float(note["start"]) * beat_s * sr))
        length_s = max(0.04, float(note.get("duration", 0.22)) * beat_s)
        count = max(1, int(length_s * sr))
        if start >= buffer.size:
            continue
        time_axis = np.arange(count) / sr
        freq = 440.0 * 2 ** ((int(note["pitch"]) - 69) / 12.0)
        wave = np.sin(2 * np.pi * freq * time_axis)
        if kind == "bass":
            saw = 2.0 * ((freq * time_axis) % 1.0) - 1.0
            wave = 0.6 * wave + 0.4 * saw
        else:
            wave = 0.5 * np.sign(np.sin(2 * np.pi * freq * time_axis))
        attack = max(1, int(0.004 * sr))
        envelope = np.ones(count, dtype=np.float32)
        envelope[:attack] = np.linspace(0, 1, attack)
        decay = np.exp(-time_axis * (6.0 if kind == "bass" else 14.0))
        gain = float(note.get("velocity", 100)) / 127.0 * float(note.get("volume", 0.85))
        buffer[start:start + count] += (wave * envelope * decay * gain).astype(np.float32)
    nyquist = sr / 2.0
    band = (30.0, 420.0) if kind == "bass" else (700.0, 3200.0)
    sos = butter(4, [band[0] / nyquist, band[1] / nyquist], btype="band", output="sos")
    return sosfilt(sos, buffer).astype(np.float32)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, default=SCRATCH / "tribal_replica_plan_v3_2.json")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--with-synth", action="store_true",
                        help="synthesise the plugin parts with simple oscillators so the part is audible")
    args = parser.parse_args()
    with_synth = bool(args.with_synth)

    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    bpm = float(plan["bpm"])
    beat_s = 60.0 / bpm
    total_bars = int(plan["total_bars"])
    track_s = total_bars * 4.0 * beat_s
    buffer = np.zeros(int(track_s * SR) + SR, dtype=np.float32)

    volumes = {"kick": .75, "perc": .52, "hats": .46, "shaker": .40, "conga": .46,
               "djembe": .46, "bongo": .40, "bass": .70, "hook": .50}
    palette = plan["library"]["palette"]
    placements = plan["library"]["placements"]
    placed = {}
    for element, runs in placements.items():
        choice = palette.get(element)
        if not choice:
            continue
        audio, sr = sf.read(str(choice["path"]), always_2d=True, dtype="float32")
        if sr != SR:
            raise SystemExit(f"SR_MISMATCH: {choice['path']} is {sr} Hz, expected {SR}")
        clip = audio.mean(axis=1)
        clip_beats = len(clip) / sr / beat_s
        step = min(max(4.0, round(clip_beats)), 32.0)
        gain = (plan.get("gains") or {}).get(element, volumes.get(element, .55))
        count = 0
        for run in runs:
            end = run["end_bar"] if run.get("end_bar") is not None else total_bars
            beat = float(run["start_bar"]) * 4.0
            while beat < float(end) * 4.0 - 1e-6:
                start_sample = int(round(beat * beat_s * SR))
                stop = min(start_sample + len(clip), len(buffer))
                if start_sample < len(buffer):
                    buffer[start_sample:stop] += clip[: stop - start_sample] * gain
                count += 1
                beat += step
        placed[element] = {"instances": count, "clip_beats": round(clip_beats, 3),
                           "source": Path(choice["path"]).name, "gain": gain}

    for element, info in (plan.get("core") or {}).items():
        audio, sr = sf.read(info["wav"], always_2d=True, dtype="float32")
        clip = audio.mean(axis=1)
        count = 1 if len(clip) <= len(buffer) else 1
        stop = min(len(clip), len(buffer))
        buffer[:stop] += clip[:stop] * (plan.get("gains") or {}).get(f"core_{element}", 1.0)
        placed[f"core_{element}"] = {"instances": count, "clip_beats": round(len(clip) / sr / beat_s, 3),
                                     "source": Path(info["wav"]).name, "gain": 1.0}

    if with_synth:
        for name, spec in (plan.get("synth") or {}).items():
            notes = spec.get("notes") or []
            default = 0.85 if name == "bass" else 0.45
            volume = spec.get("volume")
            volume = float(default if volume is None else volume)
            if not notes or volume <= 0.0:
                continue
            part = synth_part([{**note, "volume": volume} for note in notes], beat_s,
                              len(buffer) / SR, SR, "bass" if name == "bass" else "stabs")
            length = min(len(part), len(buffer))
            buffer[:length] += part[:length]
            placed[f"synth_{name}"] = {"instances": len(notes), "volume": volume,
                                       "source": "simple oscillator (patch unavailable, part only)"}

    peak = float(np.max(np.abs(buffer)) or 1.0)
    if peak > 0.99:
        buffer = buffer * (0.99 / peak)
    out = args.out or SCRATCH / f"render_{plan['version']}.wav"
    sf.write(str(out), buffer, SR)

    report = {"plan": str(args.plan), "version": plan["version"], "bpm": bpm,
              "duration_s": round(len(buffer) / SR, 2), "peak_before_norm": round(peak, 4),
              "placed": placed,
              "limitations": [] if with_synth else ["synth tracks (Serum) are not rendered: no offline sound"]}
    (SCRATCH / f"render_{plan['version']}.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"render -> {out.name}  {report['duration_s']}s  pico {peak:.3f}")
    for element, info in placed.items():
        if "clip_beats" not in info:
            print(f"  {element:18s} synth: {info['instances']:>4} notas  vol={info['volume']}")
            continue
        print(f"  {element:8s} {info['instances']:>4} instancias  clip={info['clip_beats']} beats  "
              f"gain={info['gain']}  {info['source'][:44]}")
    print("  (partes de plugin: sintesis simple, NO el patch real)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
