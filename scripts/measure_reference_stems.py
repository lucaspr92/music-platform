"""Per-section presence of each percussive FAMILY, measured on separated stems.

Why stems: on the full mix, kick, claps, hats and hand drums all arrive together, so the
arrangement cannot tell which family enters when. Separating the reference into
drums / bass / vocals / other and measuring each section per band gives that map.

Honest limit: four stems separate FAMILIES (kick body, hand-drum body, clap/snare, air),
not instruments within a family. conga vs djembe vs bongo cannot be told apart, so those
tracks are layered by the measured DENSITY of the body family, not by transcription.

Usage: python scripts/measure_reference_stems.py <stems_dir> [--structure f.json] [--json out]
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

FAMILIES = {"kick_body": (60, 200), "drum_body": (200, 500),
            "snare_clap": (500, 2000), "air": (6000, 16000)}
# A fixed dB floor is wrong: measured medians are about -17 dB for a bass stem and -32 dB
# for a vocals stem, so one threshold either flags everything or nothing. Presence is
# therefore relative to each stem's own median.
PRESENCE_DROP_DB = 12.0     # a stem counts as present within this many dB of its median


def _family_energy(signal: np.ndarray, sr: int) -> dict:
    if signal.size < 2048:
        return {name: -96.0 for name in FAMILIES}
    n = int(min(len(signal), 1 << 18))
    seg = signal[(len(signal) - n) // 2:(len(signal) - n) // 2 + n]
    spec = np.abs(np.fft.rfft(seg * np.hanning(n)))
    freqs = np.fft.rfftfreq(n, 1 / sr)
    power = spec ** 2
    total = float(power.sum()) or 1e-12
    out = {}
    for name, (lo, hi) in FAMILIES.items():
        mask = np.logical_and(freqs >= lo, freqs < hi)
        energy = float(power[mask].sum()) or 1e-12
        out[name] = round(float(10 * np.log10((energy / total) + 1e-12)), 1)
    return out


def _onsets_in_band(signal: np.ndarray, sr: int, lo: int, hi: int) -> int:
    """Onset count inside one band: how BUSY a family is, not just whether it exists."""
    if signal.size < 8192:
        return 0
    try:
        y = librosa.resample(signal, orig_sr=sr, target_sr=22050)
        spec = librosa.stft(y, n_fft=1024, hop_length=256)
        freqs = librosa.fft_frequencies(sr=22050, n_fft=1024)
        mask = np.logical_and(freqs >= lo, freqs < hi)
        filtered = np.abs(spec[mask]).sum(axis=0)
        env = librosa.onset.onset_strength(S=np.atleast_2d(filtered), sr=22050, hop_length=256)
        return int(len(librosa.onset.onset_detect(onset_envelope=env, sr=22050, hop_length=256)))
    except Exception:
        return 0


def main(stems_dir: str, structure_path: Path | None = None) -> dict:
    stems = Path(stems_dir)
    structure = json.loads((structure_path or (SCRATCH / "reference_structure.json")).read_text(encoding="utf-8"))
    bar_s = structure["bar_s"]
    report: dict = {"stems_dir": str(stems), "files": {}, "sections": []}

    loaded: dict[str, tuple] = {}
    for name in ("drums", "bass", "vocals", "other", "piano", "guitar"):
        candidates = list(stems.glob(f"{name}.*"))
        if not candidates:
            continue
        data, sr = sf.read(str(candidates[0]), always_2d=True, dtype="float32")
        loaded[name] = (data, sr)
        report["files"][name] = candidates[0].name

    for section in structure["sections"]:
        start_bar = round(section["start_s"] / bar_s, 1)
        row: dict = {"start_bar": start_bar, "bars": section["bars"], "lufs_mix": section["lufs"],
                     "families": {}, "stem_present": {}}
        if "drums" in loaded:
            data, sr = loaded["drums"]
            a, b = int(section["start_s"] * sr), int(section["end_s"] * sr)
            mono = data[a:b].mean(axis=1)
            fam = _family_energy(mono, sr)
            row["families"] = fam
            rms = float(np.sqrt(np.mean(mono ** 2)) + 1e-12)
            row["drums_rms_db"] = round(20 * np.log10(rms), 1)
            for name, (lo, hi) in FAMILIES.items():
                row[f"onsets_{name}"] = _onsets_in_band(mono, sr, lo, hi)
        for stem, (data, sr) in loaded.items():
            a, b = int(section["start_s"] * sr), int(section["end_s"] * sr)
            chunk = data[a:b]
            if chunk.size == 0:
                row[f"{stem}_rms_db"] = -96.0
                continue
            rms = float(np.sqrt(np.mean(chunk ** 2)) + 1e-12)
            row[f"{stem}_rms_db"] = round(20 * np.log10(rms), 1)
        report["sections"].append(row)

    # presence relative to each stem's own median, now that every section is measured
    for stem in loaded:
        values = [s.get(f"{stem}_rms_db", -96.0) for s in report["sections"]]
        median = float(np.median(values))
        report.setdefault("stem_median_db", {})[stem] = round(median, 1)
        for section in report["sections"]:
            value = section.get(f"{stem}_rms_db", -96.0)
            section.setdefault("stem_present", {})[stem] = bool(value > median - PRESENCE_DROP_DB)

    # per-family presence map and a layering suggestion for the hand-percussion body
    body = [s.get("onsets_drum_body", 0) for s in report["sections"]]
    if body:
        lo, hi = min(body), max(body)
        span = (hi - lo) or 1
        for section in report["sections"]:
            density = (section.get("onsets_drum_body", 0) - lo) / span
            section["body_density_0_1"] = round(density, 3)
            # 1..4 hand-percussion layers, driven by how busy the body family is
            section["suggested_body_layers"] = int(1 + round(density * 3))
    return report


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__); raise SystemExit(2)
    structure = Path(sys.argv[sys.argv.index("--structure") + 1]) if "--structure" in sys.argv else None
    rep = main(sys.argv[1], structure)
    print(json.dumps(rep, indent=2, ensure_ascii=False))
    if "--json" in sys.argv:
        out = Path(sys.argv[sys.argv.index("--json") + 1])
        out.write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
        print("->", out)
