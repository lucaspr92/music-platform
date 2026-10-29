"""Measure the per-instrument drum stems, and check them against the band measurements.

The separation is produced by mdxnet-infer with the DrumSep 6-stem checkpoint, run in the torch
runtime venv because it needs torch:

  python -c "from mdxnet_infer import separate; separate(drums.wav, output_dir=OUT,
             model_name='drumsep-6stem')"

This script only measures. It runs in the repository venv (librosa is there, torch is not).

The check is the point: a split that does not reproduce the independently measured counts
(kick ~2.65/bar from the 40-120 Hz band, hats ~11.9/bar from 6-16 kHz) is not usable, however
convincing the stems sound.

Usage: python scripts/measure_drums_per_instrument.py
"""
from __future__ import annotations
import json

from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

SCRATCH = Path(__file__).resolve().parents[1] / "logs"
STEMS = Path.home() / "music-platform-runtime/stems/drums_per_instrument"
EXPECTED = {"kick": (2.0, 3.5), "hh": (9.0, 14.0)}
STEMS_OF_INTEREST = ["kick", "snare", "hh", "toms", "ride", "crash"]


def measure(path: Path, bar_s: float) -> dict:
    audio, sr = sf.read(str(path), always_2d=True, dtype="float32")
    mono = audio.mean(axis=1)
    peak = float(np.max(np.abs(mono))) if mono.size else 0.0
    if mono.size == 0 or peak < 1e-5:
        return {"onsets": 0, "per_bar": 0.0, "rms_db": None, "silent": True}
    hop = 256
    envelope = librosa.onset.onset_strength(y=mono, sr=sr, hop_length=hop)
    frames = librosa.onset.onset_detect(onset_envelope=envelope, sr=sr, hop_length=hop,
                                        units="frames")
    bars = (mono.size / sr) / bar_s
    rms_db = 20 * np.log10(float(np.sqrt(np.mean(mono ** 2))) + 1e-9)
    return {"onsets": int(frames.size), "per_bar": round(float(frames.size) / bars, 2),
            "rms_db": round(rms_db, 1), "silent": False}


def main() -> int:
    structure = json.loads((SCRATCH / "reference_structure.json").read_text(encoding="utf-8"))
    bar_s = float(structure["bar_s"])
    report: dict = {"source": str(STEMS), "bar_s": bar_s, "expected": EXPECTED, "stems": {}}
    print(f"{'stem':8s} {'onsets':>7s} {'per bar':>8s} {'rms dB':>8s}")
    for name in STEMS_OF_INTEREST:
        path = STEMS / f"drums_{name}.wav"
        if not path.is_file():
            print(f"{name:8s} FALTA {path.name}")
            continue
        row = measure(path, bar_s)
        report["stems"][name] = {"path": str(path), **row}
        print(f"{name:8s} {row['onsets']:7d} {row['per_bar']:8.2f} "
              f"{row['rms_db'] if row['rms_db'] is not None else float('nan'):8.1f}")
    report["checks"] = {}
    for name, (low, high) in EXPECTED.items():
        value = report["stems"].get(name, {}).get("per_bar", 0.0)
        ok = low <= value <= high
        report["checks"][name] = {"per_bar": value, "expected": [low, high], "ok": ok}
        print(f"check {name}: {value:.2f}/bar in [{low}, {high}] -> {'OK' if ok else 'FUERA DE RANGO'}")
    silent = [name for name, row in report["stems"].items() if row["silent"]]
    report["silent_stems"] = silent
    if silent:
        print("stems mudos:", silent)
    (SCRATCH / "drums_per_instrument.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("->", SCRATCH / "drums_per_instrument.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
