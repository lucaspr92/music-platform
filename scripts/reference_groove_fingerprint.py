"""Groove fingerprint of a track or a loop, from Essentia + librosa.

Usage:
  python scripts/reference_groove_fingerprint.py <audio> [--json out.json] [--genre]

Answers "which elements, and where": how much of it is percussive, in which band
each part lives, where the hits fall inside the beat (16th grid), how dense they
are, and -- with --genre -- which Discogs styles the model hears.

Taste is still human. This is what can be counted.
"""
from __future__ import annotations
import json, sys
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

BANDS = ((60, 200), (200, 500), (500, 2000), (2000, 6000), (6000, 16000))
NAMES = ("kick_body", "drum_body", "snare_clap", "attack", "air")
MODELS = Path.home() / "music-platform-runtime" / "models" / "essentia"
INTERESTING = ("tribal", "afro", "latin", "house", "techno", "percussion", "world")


def band_share(signal: np.ndarray, sr: int) -> dict:
    spec = np.abs(np.fft.rfft(signal * np.hanning(len(signal))))
    freqs = np.fft.rfftfreq(len(signal), 1 / sr)
    power = spec ** 2
    total = float(power.sum()) or 1.0
    out = {}
    for name, (lo, hi) in zip(NAMES, BANDS):
        m = np.logical_and(freqs >= lo, freqs < hi)
        out[name] = round(float(10 * np.log10((float(power[m].sum()) or 1e-12) / total)), 2)
    return out


def grid_profile(onset_times: np.ndarray, beats: np.ndarray) -> dict | None:
    """Where hits land inside the beat: 4 sixteenths -> fraction per cell."""
    if onset_times.size < 4 or beats.size < 4:
        return None
    period = float(np.median(np.diff(beats)))
    if period <= 0:
        return None
    origin = float(beats[0])
    phase = ((onset_times - origin) / period) % 1.0
    cell = np.minimum((phase * 4).astype(int), 3)
    hist = np.bincount(cell, minlength=4).astype(float)
    hist = hist / hist.sum()
    return {"sixteenths": [round(float(v), 3) for v in hist],
            "on_beat": round(float(hist[0]), 3),
            "offbeat_8th": round(float(hist[2]), 3),
            "syncopated_16ths": round(float(hist[1] + hist[3]), 3),
            "beat_period_s": round(period, 4)}


def genre_top(path: str, k: int = 8) -> list | None:
    try:
        meta = json.loads((MODELS / "genre_discogs400-discogs-effnet-1.json").read_text())
        labels = meta.get("classes") or meta.get("labels") or []
        st = es.MonoLoader(filename=path, sampleRate=16000)
        audio = st()
        emb = es.TensorflowPredictEffnetDiscogs(graphFilename=str(MODELS / "discogs-effnet-bs64-1.pb"),
                                                output="PartitionedCall:1")(audio)
        preds = es.TensorflowPredict2D(graphFilename=str(MODELS / "genre_discogs400-discogs-effnet-1.pb"),
                                       input="serving_default_model_Placeholder",
                                       output="PartitionedCall:0")(emb)
        mean = np.mean(preds, axis=0)
        order = np.argsort(mean)[::-1]
        rows = [{"label": labels[i] if i < len(labels) else str(i), "p": round(float(mean[i]), 4)}
                for i in order[:k]]
        return rows
    except Exception as exc:
        return [{"error": f"{type(exc).__name__}: {exc}"}]


def main(path: str, want_genre: bool) -> dict:
    data, sr = sf.read(path, always_2d=True, dtype="float32")
    mono = data.mean(axis=1).astype(np.float32)
    y, ysr = librosa.load(path, sr=None, mono=True)
    bpm, beats, conf, _, _ = es.RhythmExtractor2013(method="multifeature")(mono)
    onsets, rate = es.OnsetRate()(mono)
    key, scale, strength = es.KeyExtractor()(mono)
    yh, yp = librosa.effects.hpss(y, margin=3.0)
    rms = lambda a: float(np.sqrt(np.mean(a ** 2)))          # noqa: E731
    rep = {
        "file": Path(path).name, "duration_s": round(len(data) / sr, 2),
        "bpm": round(float(bpm), 2), "beat_confidence": round(float(conf), 3),
        "onsets": len(onsets), "onsets_per_s": round(float(rate), 2),
        "key": f"{key} {scale}", "key_strength": round(float(strength), 3),
        "percussive_over_harmonic": round(rms(yp) / (rms(yh) + 1e-12), 3),
        "percussive_band_share_db": band_share(yp, sr),
        "grid": grid_profile(np.asarray(onsets, dtype=float), np.asarray(beats, dtype=float)),
        "clipping": bool(float(np.max(np.abs(data))) >= 1.0),
    }
    if want_genre:
        rep["genre_discogs400"] = genre_top(path)
    return rep


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__); raise SystemExit(2)
    rep = main(sys.argv[1], "--genre" in sys.argv)
    print(json.dumps(rep, indent=2, ensure_ascii=False))
    if "--json" in sys.argv:
        out = Path(sys.argv[sys.argv.index("--json") + 1])
        out.write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
        print("->", out)
