"""Structure map of a reference track: sections, what plays in each, and the bar grid.

Answers the composition question, not just the tempo: where sections change, and for
every section which elements are present (kick, hand percussion, hats/air, low
harmonic, mid harmonic/vocal) and how loud it is.

Usage: python scripts/reference_structure.py <audio> [--json out.json]
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


def band_share(signal, sr):
    """Band shares from the CENTRED window of a section (not its first N samples).

    A section can be minutes long; a window from its middle represents it far better
    than its opening, and 2**18 samples is plenty for a stable band split.
    """
    if signal.size < 2048:
        return {n: None for n in NAMES}
    n = int(min(len(signal), 1 << 18))
    start = (len(signal) - n) // 2
    seg = signal[start:start + n]
    spec = np.abs(np.fft.rfft(seg * np.hanning(n)))
    freqs = np.fft.rfftfreq(n, 1 / sr)
    power = spec ** 2
    total = float(power.sum()) or 1.0
    out = {}
    for name, (lo, hi) in zip(NAMES, BANDS):
        m = np.logical_and(freqs >= lo, freqs < hi)
        out[name] = round(float(10 * np.log10((float(power[m].sum()) or 1e-12) / total)), 1)
    return out


def main(path: str) -> dict:
    data, sr = sf.read(path, always_2d=True, dtype="float32")
    y, ysr = librosa.load(path, sr=None, mono=True)
    bpm, beats, conf, _, _ = es.RhythmExtractor2013(method="multifeature")(data.mean(axis=1).astype(np.float32))
    beat_period = 60.0 / bpm
    bar = beat_period * 4

    # section boundaries: novelty over beat-synchronous MFCC, clustered then merged
    hop = 512
    onset_env = librosa.onset.onset_strength(y=y, sr=ysr, hop_length=hop)
    mfcc = librosa.feature.mfcc(y=y, sr=ysr, hop_length=hop, n_mfcc=13)
    times = librosa.times_like(mfcc[0], sr=ysr, hop_length=hop)
    novelty = np.abs(np.diff(mfcc, axis=1)).sum(axis=0)
    novelty = np.convolve(novelty, np.ones(31) / 31, mode="same")
    # local maxima of novelty at least 8 bars apart
    min_gap = max(int(8 * bar / (hop / ysr)), 4)
    peaks = []
    for i in range(2, len(novelty) - 2):
        if novelty[i] == max(novelty[max(0, i - 15):i + 15]) and novelty[i] > np.percentile(novelty, 70):
            if not peaks or i - peaks[-1] >= min_gap:
                peaks.append(i)
    bounds = [0.0] + [float(times[i]) for i in peaks] + [len(data) / sr]
    bounds = sorted(set(round(b, 3) for b in bounds))

    harmonic, percussive = librosa.effects.hpss(y, margin=3.0)
    sections = []
    for i in range(len(bounds) - 1):
        start, end = bounds[i], bounds[i + 1]
        s0, s1 = int(start * ysr), int(end * ysr)
        if s1 - s0 < ysr:            # ignore slivers
            continue
        seg_p, seg_h = percussive[s0:s1], harmonic[s0:s1]
        seg = data[int(start * sr):int(end * sr)]
        rms = float(np.sqrt(np.mean(seg ** 2)) + 1e-12)
        loud = es.LoudnessEBUR128(sampleRate=sr)(seg)[2]
        pb = band_share(seg_p, ysr)
        hb = band_share(seg_h, ysr)
        bars = (end - start) / bar
        sections.append({
            "start_s": round(start, 2), "end_s": round(end, 2), "bars": round(bars, 1),
            "start_bar": round(start / bar, 1),
            "rms_db": round(20 * np.log10(rms), 2), "lufs": round(float(loud), 2),
            "perc_band": pb, "harm_band": hb,
            "kick": bool(pb["kick_body"] is not None and pb["kick_body"] > -14),
            "hand_perc": bool(pb["drum_body"] is not None and pb["drum_body"] > -12),
            "hats_air": bool(pb["air"] is not None and pb["air"] > -16),
            "low_harm": bool(hb["kick_body"] is not None and hb["kick_body"] > -14),
            "mid_harm": bool(hb["snare_clap"] is not None and hb["snare_clap"] > -26),
        })
    return {"file": Path(path).name, "duration_s": round(len(data) / sr, 2), "bpm": round(float(bpm), 2),
            "bar_s": round(bar, 4), "total_bars": round(len(data) / sr / bar, 1), "sections": sections}


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__); raise SystemExit(2)
    rep = main(sys.argv[1])
    print(json.dumps(rep, indent=2, ensure_ascii=False))
    if "--json" in sys.argv:
        out = Path(sys.argv[sys.argv.index("--json") + 1])
        out.write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
        print("->", out)
