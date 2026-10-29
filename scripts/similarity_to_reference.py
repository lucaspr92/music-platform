"""Score how close a candidate render is to the reference, per dimension.

Reported separately on purpose: a single total hides which gap to fix. Two rules make the
number mean something:

  * the reference scored against itself must be 1.0 on every dimension
  * a dimension that cannot be measured is reported as null, never as a pass

Dimensions here are all audio-based and computable offline. Timbre (clip embeddings) needs the
analysis venv that holds torch, so it is left null by this script rather than faked.

Usage:
  python scripts/similarity_to_reference.py --candidate logs/render_3.2.wav
  python scripts/similarity_to_reference.py --candidate <ref.wav>     # sanity: must be 1.0
"""
from __future__ import annotations
import argparse, json, math

from pathlib import Path

import numpy as np
import soundfile as sf

import essentia
import essentia.standard as es
from scipy.signal import butter, sosfiltfilt

essentia.log.warningActive = False

SCRATCH = Path(__file__).resolve().parents[1] / "logs"
REFERENCE = Path.home() / "music-platform-runtime/stems-src/reference_tudm_44100_2ch.wav"
SR = 44100
SECTION_BARS = 8
BANDS = {"kick_body": (40, 120), "drum_body": (200, 500), "snare_clap": (500, 2000),
         "attack": (2000, 6000), "air": (6000, 16000)}


def load_mono(path: Path) -> np.ndarray:
    audio = es.MonoLoader(filename=str(path), resampleQuality=4)()
    return np.asarray(audio, dtype=np.float32)


def band_filter(audio: np.ndarray, lo: float, hi: float, order: int = 4) -> np.ndarray:
    """Butterworth band-pass. scipy rather than Essentia's BandPass: its parameter set is
    narrower than the filter this needs, and guessing parameters is how time gets lost."""
    nyquist = SR / 2.0
    sos = butter(order, [lo / nyquist, min(hi / nyquist, 0.99)], btype="band", output="sos")
    return sosfiltfilt(sos, audio).astype(np.float32)


def band_onset_histograms(audio: np.ndarray, beat_s: float, bar_s: float) -> dict[str, list]:
    """16th-of-bar onset histograms per percussive family, using Essentia's onset detector."""
    out: dict[str, list] = {}
    for name, (lo, hi) in (("kick", (40, 120)), ("body", (200, 500)), ("clap", (500, 2000)),
                           ("air", (6000, 16000))):
        filtered = band_filter(audio, lo, hi)
        times, _rate = es.OnsetRate()(filtered)
        per_bar = np.zeros(16)
        for t in times:
            bar = int(t / bar_s)
            step = int(round((t - bar * bar_s) / (beat_s / 4.0))) % 16
            per_bar[step] += 1
        total = per_bar.sum()
        out[name] = {"hist": list(per_bar / total) if total else [0.0] * 16,
                     "onset_rate": round(float(len(times)) / (len(audio) / SR / bar_s), 2)}
    return out


def band_shares(audio: np.ndarray, bar_s: float, total_bars: int) -> list[dict]:
    """Per 8-bar section, the share of energy in each band."""
    sections = []
    window = 8192
    for start_bar in range(0, total_bars, SECTION_BARS):
        a = int(start_bar * bar_s * SR)
        b = min(int(min(start_bar + SECTION_BARS, total_bars) * bar_s * SR), len(audio))
        chunk = audio[a:b]
        if chunk.size < window:
            continue
        spectrum = np.abs(np.fft.rfft(chunk[: (chunk.size // window) * window] *
                                      np.hanning((chunk.size // window) * window))) ** 2
        freqs = np.fft.rfftfreq((chunk.size // window) * window, 1 / SR)
        total = spectrum.sum() or 1.0
        shares = {}
        for name, (lo, hi) in BANDS.items():
            shares[name] = round(float(spectrum[(freqs >= lo) & (freqs < hi)].sum() / total), 4)
        sections.append({"start_bar": start_bar, "shares": shares,
                        "rms_db": round(float(20 * np.log10(np.sqrt(np.mean(chunk ** 2)) + 1e-12)), 1)})
    return sections


def l1_to_score(a: dict[str, float] | list[float], b: dict[str, float] | list[float]) -> float:
    """Total-variation similarity for distributions that each sum to 1.

    Using the MEAN absolute difference was a bug: over 16 normalised bins every pair looked
    alike (white noise scored 0.94 against the reference). Total variation, sum(|a-b|)/2, is
    the right distance for distributions and separates them properly.
    """
    if isinstance(a, dict):
        diffs = [abs(float(a[k]) - float(b[k])) for k in a if k in b]
    else:
        diffs = [abs(float(x) - float(y)) for x, y in zip(a, b)]
    if not diffs:
        return 0.0
    return max(0.0, 1.0 - float(np.sum(diffs)) / 2.0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--reference", type=Path, default=REFERENCE)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    ref = load_mono(args.reference)
    cand = load_mono(args.candidate)

    bpm, beats, confidence, _e, _i = es.RhythmExtractor2013(method="multifeature")(ref)
    beat_s = 60.0 / float(bpm)
    bar_s = beat_s * 4.0
    total_bars = int(len(ref) / SR / bar_s)
    print(f"grilla de la referencia: {bpm:.3f} BPM  beat {beat_s:.4f}s  bar {bar_s:.4f}s  "
          f"{total_bars} compases")

    ref_hist = band_onset_histograms(ref, beat_s, bar_s)
    cand_hist = band_onset_histograms(cand, beat_s, bar_s)
    # distribution agreement alone lets a detector that fires on everything look close: white
    # noise scored 0.94 on the raw histograms. Density is part of the groove, so a candidate
    # whose onset rate is far from the reference's is penalised in proportion.
    groove_scores = {}
    for family in ref_hist:
        shape = l1_to_score(ref_hist[family]["hist"], cand_hist[family]["hist"])
        ref_rate = ref_hist[family]["onset_rate"]
        cand_rate = cand_hist[family]["onset_rate"]
        density = min(ref_rate, cand_rate) / max(ref_rate, cand_rate) if max(ref_rate, cand_rate) else 0.0
        groove_scores[family] = round(shape * density, 4)

    ref_bands = band_shares(ref, bar_s, total_bars)
    cand_bands = band_shares(cand, bar_s, total_bars)
    n = min(len(ref_bands), len(cand_bands))
    # a section that is silent on either side says nothing about balance or dynamics: the
    # reference file simply ends, so comparing its tail against the render's tail is noise
    usable = [i for i in range(n)
              if ref_bands[i]["rms_db"] > -60.0 and cand_bands[i]["rms_db"] > -60.0]
    band_score = round(float(np.mean([
        l1_to_score(ref_bands[i]["shares"], cand_bands[i]["shares"])
        for i in usable])), 4) if usable else None

    ref_energy = np.array([ref_bands[i]["rms_db"] for i in usable], dtype=float)
    cand_energy = np.array([cand_bands[i]["rms_db"] for i in usable], dtype=float)
    n = len(usable)
    if n > 2 and ref_energy.std() > 0 and cand_energy.std() > 0:
        correlation = float(np.corrcoef(ref_energy, cand_energy)[0, 1])
    else:
        correlation = 0.0
    energy_score = round((correlation + 1.0) / 2.0, 4)

    dimensions = {
        "groove_kick": groove_scores["kick"], "groove_body": groove_scores["body"],
        "groove_clap": groove_scores["clap"], "groove_air": groove_scores["air"],
        "band_balance": band_score, "energy_curve": energy_score,
        "timbre": None,
    }
    weights = {"groove_kick": .2, "groove_body": .1, "groove_clap": .1, "groove_air": .1,
               "band_balance": .25, "energy_curve": .25}
    total = sum(dimensions[k] * w for k, w in weights.items()) / sum(weights.values())

    result = {"candidate": str(args.candidate), "reference": str(args.reference),
              "onset_rates": {"reference": {k: v["onset_rate"] for k, v in ref_hist.items()},
                              "candidate": {k: v["onset_rate"] for k, v in cand_hist.items()}},
              "bpm": round(float(bpm), 3), "sections_compared": n,
              "dimensions": dimensions, "weights": weights,
              "total": round(float(total), 4),
              "notes": ["timbre is null: clip embeddings need the analysis venv",
                        "synth tracks are absent from an offline render"]}

    print()
    print(f"{'dimension':14s} {'score':>7}")
    print("-" * 24)
    for name, value in dimensions.items():
        print(f"{name:14s} {'n/a' if value is None else f'{value:.3f}':>7}")
    print("-" * 24)
    print(f"{'TOTAL':14s} {total:>7.3f}")

    out = args.json or SCRATCH / f"similarity_{Path(args.candidate).stem}.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
