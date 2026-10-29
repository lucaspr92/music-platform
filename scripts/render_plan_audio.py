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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, default=SCRATCH / "tribal_replica_plan_v3_2.json")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

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

    peak = float(np.max(np.abs(buffer)) or 1.0)
    if peak > 0.99:
        buffer = buffer * (0.99 / peak)
    out = args.out or SCRATCH / f"render_{plan['version']}.wav"
    sf.write(str(out), buffer, SR)

    report = {"plan": str(args.plan), "version": plan["version"], "bpm": bpm,
              "duration_s": round(len(buffer) / SR, 2), "peak_before_norm": round(peak, 4),
              "placed": placed,
              "limitations": ["synth tracks (Serum) are not rendered: no offline sound"]}
    (SCRATCH / f"render_{plan['version']}.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"render -> {out.name}  {report['duration_s']}s  pico {peak:.3f}")
    for element, info in placed.items():
        print(f"  {element:8s} {info['instances']:>4} instancias  clip={info['clip_beats']} beats  "
              f"gain={info['gain']}  {info['source'][:44]}")
    print("  (los tracks de Serum no se renderizan: no hay sonido offline)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
