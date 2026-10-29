"""Timbre profile of the reference's elements, and a timbre ranking of library candidates.

Groove got the track from 0.61 to 0.73. What is left is largely timbre: a library conga is not
their conga. This measures it instead of guessing:

  * reference profile : MFCCs, spectral shape, noisiness, HPSS ratio, attack and decay, measured
                        on each element's own stem (or its isolated band of the drums stem)
  * candidate profile : the same features on every library candidate for that role
  * ranking           : z-scored features, cosine distance to the reference profile

Honest caveats, printed with the report: the clap band is contaminated (hat transients dominate
every band of the drums stem), and the profile describes a whole file, not one hit.

Usage: python scripts/analyze_timbre.py
"""
from __future__ import annotations
import json

from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

SCRATCH = Path(__file__).resolve().parents[1] / "logs"
INDEX = Path.home() / "Library/Application Support/MusicPlatform/sample-library-index.json"
FT = Path.home() / "music-platform-runtime/stems/ft/htdemucs_ft"
SIX = Path.home() / "music-platform-runtime/stems/6s/htdemucs_6s"
SR = 44100
MAX_PER_ROLE = 40   # bounded sample: the ranking does not need 1752 percussion loops
ROLES = {"kick": ("KICK",), "clap": ("CLAP", "SNARE"), "hats": ("TOP_LOOP", "RIDE"),
         "percussion": ("PERCUSSION",), "bass": ("BASS",), "synth": ("SYNTH",)}


def features(signal: np.ndarray, sr: int = SR) -> dict | None:
    if signal.size < sr // 4:
        return None
    mono = signal.astype(np.float32)
    mfcc = librosa.feature.mfcc(y=mono, sr=sr, n_mfcc=20)
    centroid = librosa.feature.spectral_centroid(y=mono, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(y=mono, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=mono, sr=sr)
    flatness = librosa.feature.spectral_flatness(y=mono)
    zcr = librosa.feature.zero_crossing_rate(mono)
    contrast = librosa.feature.spectral_contrast(y=mono, sr=sr)
    harmonic, percussive = librosa.effects.hpss(mono)
    env = np.abs(librosa.stft(mono)).sum(axis=0)
    env = env / (env.max() or 1.0)
    attack = float(np.argmax(env) / max(1, len(env)))
    peak_at = int(np.argmax(env))
    after = env[peak_at:]
    below = np.where(after < 0.1)[0]
    decay = float(below[0] / max(1, len(after))) if below.size else 1.0
    spectrum = np.abs(np.fft.rfft(mono[: min(len(mono), sr * 4)]))
    freqs = np.fft.rfftfreq(min(len(mono), sr * 4), 1 / sr)
    slope = float(np.polyfit(np.log10(freqs[1:] + 1), np.log10(spectrum[1:] + 1e-9), 1)[0])
    return {
        "mfcc_mean": np.mean(mfcc, axis=1), "mfcc_std": np.std(mfcc, axis=1),
        "centroid": float(np.mean(centroid)), "bandwidth": float(np.mean(bandwidth)),
        "rolloff": float(np.mean(rolloff)), "flatness": float(np.mean(flatness)),
        "zcr": float(np.mean(zcr)), "contrast": np.mean(contrast, axis=1),
        "harmonic_ratio": float(np.sum(harmonic ** 2) / (np.sum(percussive ** 2) + 1e-12)),
        "attack": attack, "decay": decay, "tilt": slope,
    }


def to_vector(feat: dict) -> np.ndarray:
    parts = [feat["mfcc_mean"], feat["mfcc_std"], feat["contrast"],
             np.array([feat["centroid"], feat["bandwidth"], feat["rolloff"], feat["flatness"],
                       feat["zcr"], np.log10(feat["harmonic_ratio"] + 1e-9),
                       feat["attack"], feat["decay"], feat["tilt"]])]
    return np.concatenate([np.asarray(part, dtype=float).ravel() for part in parts])


def band(signal: np.ndarray, lo: float, hi: float) -> np.ndarray:
    from scipy.signal import butter, sosfiltfilt
    nyquist = SR / 2.0
    sos = butter(4, [lo / nyquist, min(hi / nyquist, 0.99)], btype="band", output="sos")
    return sosfiltfilt(sos, signal).astype(np.float32)


def load(path: Path) -> np.ndarray:
    audio, sr = sf.read(str(path), always_2d=True, dtype="float32")
    mono = audio.mean(axis=1)
    if sr != SR:
        mono = librosa.resample(mono, orig_sr=sr, target_sr=SR)
    return mono


def main() -> int:
    drums = load(FT / "drums.wav")
    reference = {
        "kick": features(band(drums, 40, 120)),
        "clap": features(band(drums, 500, 2000)),
        "hats": features(band(drums, 6000, 16000)),
        "bass": features(load(FT / "bass.wav")),
        "synth": features(load(SIX / "guitar.wav")),
    }
    reference = {k: v for k, v in reference.items() if v}
    print("perfil de la referencia medido para:", list(reference))

    assets = list(json.loads(INDEX.read_text(encoding="utf-8"))["assets"].values())
    candidates: dict[str, list] = {}
    for element, roles in ROLES.items():
        if element not in reference:
            continue
        rows = []
        for asset in assets:
            if str(asset.get("semantic_role")).upper() not in roles:
                continue
            groove = asset.get("groove") or {}
            if (groove.get("bpm") or 0) <= 0:
                continue
            try:
                feat = features(load(Path(asset["path"])))
            except Exception:
                continue
            if feat and len(rows) < MAX_PER_ROLE:
                rows.append({"file": Path(asset["relative_path"]).name, "path": asset["path"],
                             "bpm": round(float(groove["bpm"]), 2), "features": feat})
        candidates[element] = rows
        print(f"  {element:10s} {len(rows)} candidatos con features")

    report: dict = {"reference": {k: {kk: (vv.tolist() if isinstance(vv, np.ndarray) else vv)
                                      for kk, vv in v.items()} for k, v in reference.items()},
                    "caveats": ["clap band contaminated by hat transients",
                                "profile describes a whole file, not a single hit"],
                    "rankings": {}}
    for element, rows in candidates.items():
        if not rows:
            continue
        matrix = np.array([to_vector(row["features"]) for row in rows])
        ref_vector = to_vector(reference[element])
        pooled = np.vstack([matrix, ref_vector[None, :]])
        mean, std = pooled.mean(axis=0), pooled.std(axis=0) + 1e-9
        z = (matrix - mean) / std
        z_ref = (ref_vector - mean) / std
        distances = 1.0 - (z @ z_ref) / (np.linalg.norm(z, axis=1) * np.linalg.norm(z_ref) + 1e-12)
        order = np.argsort(distances)
        report["rankings"][element] = [
            {"file": rows[i]["file"], "path": rows[i]["path"], "bpm": rows[i]["bpm"],
             "timbre_distance": round(float(distances[i]), 4)} for i in order[:12]]
        print()
        print(f"=== {element}: mas cercano en TIMBRE a la referencia ===")
        for i in order[:8]:
            print(f"  {distances[i]:.3f}  bpm={rows[i]['bpm']:>6.2f}  {rows[i]['file'][:56]}")

    (SCRATCH / "timbre_profile.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n->", SCRATCH / "timbre_profile.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
