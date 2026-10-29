"""Split the reference's drums stem into per-instrument stems, then sanity-check the split.

Band filtering could never isolate a clap: every band of the drums stem shows ~11 onsets/bar from
hat transients, so the clap was an assumption. Per-instrument separation attacks the root cause.

The sanity check is not optional: a separation that does not reproduce the independently measured
counts (kick ~2.65/bar, hats ~11.9/bar) is not usable, whatever it looks like.

Usage: python scripts/separate_drums_per_instrument.py
"""
from __future__ import annotations
import json

from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

SCRATCH = Path(__file__).resolve().parents[1] / "logs"
DRUMS = Path.home() / "music-platform-runtime/stems/ft/htdemucs_ft/drums.wav"
OUT = Path.home() / "music-platform-runtime/stems/drums_per_instrument"
EXPECTED = {"kick": (2.0, 3.5), "hihat": (9.0, 14.0)}


def onsets_per_bar(path: Path, bar_s: float) -> tuple[int, float]:
    audio, sr = sf.read(str(path), always_2d=True, dtype="float32")
    mono = audio.mean(axis=1)
    if mono.size == 0 or float(np.max(np.abs(mono))) < 1e-5:
        return 0, 0.0
    frame = 1024
    hop = 256
    envelope = librosa.onset.onset_strength(y=mono, sr=sr, hop_length=hop)
    frames = librosa.onset.onset_detect(onset_envelope=envelope, sr=sr, hop_length=hop,
                                        units="frames")
    times = librosa.frames_to_time(frames, sr=sr, hop_length=hop)
    duration_s = mono.size / sr
    bars = duration_s / bar_s
    return int(times.size), round(times.size / bars, 2) if bars else 0.0


def main() -> int:
    structure = json.loads((SCRATCH / "reference_structure.json").read_text(encoding="utf-8"))
    bar_s = float(structure["bar_s"])
    from drumsep import separate
    OUT.mkdir(parents=True, exist_ok=True)
    result = separate(str(DRUMS), output_dir=str(OUT), bass_path=str(
        Path.home() / "music-platform-runtime/stems/ft/htdemucs_ft/bass.wav"))
    stems = getattr(result, "stems", result)
    report: dict = {"source": str(DRUMS), "bar_s": bar_s, "stems": {}, "expected": EXPECTED}
    print(f"{'stem':10s} {'file':38s} {'onsets':>7s} {'per bar':>8s}")
    for name, path in sorted(stems.items()):
        path = Path(path)
        count, per_bar = onsets_per_bar(path, bar_s)
        report["stems"][name] = {"path": str(path), "onsets": count, "per_bar": per_bar,
                                 "size": path.stat().st_size if path.exists() else 0}
        print(f"{name:10s} {path.name:38s} {count:7d} {per_bar:8.2f}")
    for name, (low, high) in EXPECTED.items():
        value = report["stems"].get(name, {}).get("per_bar", 0.0)
        report.setdefault("checks", {})[name] = {
            "per_bar": value, "expected": [low, high], "ok": low <= value <= high}
        print(f"check {name}: {value:.2f}/bar in [{low}, {high}] -> "
              f"{'OK' if low <= value <= high else 'FUERA DE RANGO'}")
    (SCRATCH / "drums_per_instrument.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("->", SCRATCH / "drums_per_instrument.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
