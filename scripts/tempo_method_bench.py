"""Which tempo detector is right on SHORT loops? Measured against pack labels.

The packs label their loops ('095bpm', '128 BPM', '124'), so the label is ground
truth for tempo. Each detector is scored: exact, octave error (x2 / /2), or wrong.
"""
from __future__ import annotations
import re, statistics
from pathlib import Path

# Scratch artifacts stay in the gitignored logs/ dir, resolved from the repo root so
# the script runs from any working directory.
SCRATCH = Path(__file__).resolve().parents[1] / "logs"


import numpy as np
import soundfile as sf

for _a in ("warningActive", "infoActive"):
    try:
        import essentia; setattr(essentia.log, _a, False)
    except Exception: pass
import essentia.standard as es
import librosa
import librosa.feature.rhythm as lfrhythm

from copilot.sample_library.config import index_path
from copilot.sample_library.library_v1 import load_index

MAX_PAIRS = 60


def label_bpm(name: str) -> float | None:
    for token in re.findall(r"\d{3}", name):
        value = float(token)
        if 60.0 <= value <= 190.0:
            return value
    return None


def method_multifeature(mono, sr):
    return float(es.RhythmExtractor2013(method="multifeature")(mono)[0])


def method_degara(mono, sr):
    return float(es.RhythmExtractor2013(method="degara")(mono)[0])


def method_percival(mono, sr):
    return float(es.PercivalBpmEstimator()(mono))


def method_librosa_tempogram(mono, sr):
    return float(lfrhythm.tempo(y=mono, sr=sr, aggregate=np.median)[0])


def method_librosa_beat_track(mono, sr):
    tempo, _ = librosa.beat.beat_track(y=mono, sr=sr)
    return float(np.atleast_1d(tempo)[0])


def method_autocorr_env(mono, sr):
    """The index's own cheap estimator: onset envelope autocorrelation, 0.01 s hops."""
    hop = int(0.01 * sr)
    env = np.array([np.sqrt(np.mean(mono[i:i + hop] ** 2) + 1e-12) for i in range(0, len(mono) - hop, hop)])
    env = np.diff(env)
    env = np.maximum(env, 0.0)
    if env.size < 8:
        return float("nan")
    env = env - env.mean()
    ac = np.correlate(env, env, mode="full")[env.size - 1:]
    ac = ac / (ac[0] + 1e-12)
    lo = max(2, int((60.0 / 190.0) / 0.01))
    hi = int((60.0 / 60.0) / 0.01)
    if hi <= lo + 2:
        return float("nan")
    peak = int(np.argmax(ac[lo:hi]) + lo)
    return 60.0 / (peak * 0.01)


METHODS = {
    "essentia multifeature": method_multifeature,
    "essentia degara": method_degara,
    "essentia percival": method_percival,
    "librosa tempogram": method_librosa_tempogram,
    "librosa beat_track": method_librosa_beat_track,
    "autocorr (el del indice)": method_autocorr_env,
}


def main() -> int:
    index = load_index(index_path())
    rows = []
    for asset in index.assets.values():
        if asset.status != "INDEXED" or asset.extension.lower() not in (".wav", ".aif", ".aiff"):
            continue
        label = label_bpm(asset.filename)
        if label is None:
            continue
        duration = (asset.descriptors.duration_s if asset.descriptors else None) or 0.0
        if not 1.0 <= duration <= 20.0:
            continue
        rows.append((asset.path, asset.filename, label, duration))
    rows = rows[:MAX_PAIRS]
    print(f"loops con etiqueta de BPM: {len(rows)} (duracion media "
          f"{statistics.mean(r[3] for r in rows):.1f}s)\n")

    results = {name: {"exact": 0, "octave": 0, "wrong": 0, "none": 0, "errs": []} for name in METHODS}
    for path, name, label, duration in rows:
        data, sr = sf.read(path, always_2d=True, dtype="float32")
        mono = data.mean(axis=1).astype(np.float32)
        for method_name, fn in METHODS.items():
            try:
                value = fn(mono, sr)
            except Exception:
                value = float("nan")
            bucket = results[method_name]
            if value is None or not np.isfinite(value) or value <= 0:
                bucket["none"] += 1
                continue
            err = abs(value - label)
            ratio = value / label
            if err <= 1.0:
                bucket["exact"] += 1
                bucket["errs"].append(err)
            elif 0.47 <= ratio <= 0.53 or 1.94 <= ratio <= 2.06:
                bucket["octave"] += 1
                bucket["errs"].append(err)
            else:
                bucket["wrong"] += 1
                bucket["errs"].append(err)

    total = len(rows)
    print(f"{'metodo':26s} {'exacto':>7s} {'octava':>7s} {'mal':>6s} {'sinValor':>9s} {'err med':>8s}")
    print("-" * 70)
    for method_name, b in sorted(results.items(), key=lambda kv: -kv[1]["exact"]):
        errs = b["errs"] or [0.0]
        print(f"{method_name:26s} {b['exact']/total*100:6.1f}% {b['octave']/total*100:6.1f}% "
              f"{b['wrong']/total*100:5.1f}% {b['none']/total*100:8.1f}% {statistics.median(errs):7.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
