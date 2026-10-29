"""Harmony, melody and structural-change analysis of the reference, measured not guessed.

Band shares tell you how much energy sits in a range, never WHICH CHORD sounds or WHAT LINE
is played. Essentia is already installed, so the only cost is the measurement.

  * KeyExtractor + HPCP -> ChordsDetection  -> the chord progression, per section
  * PredominantPitchMelodia                 -> the melodic line (a hook is a line, not a loop)
  * SBic over MFCC frames                   -> an independent opinion on where sections start
  * BeatsLoudness                           -> energy at beat resolution

Reported, not acted on: this writes evidence and decides nothing.

Usage: python scripts/analyze_reference_harmony.py [--audio <path>] [--json <path>]
"""
from __future__ import annotations
import argparse, json, time

from pathlib import Path

import numpy as np

import essentia
import essentia.standard as es

essentia.log.warningActive = False

DEFAULT_AUDIO = Path.home() / "music-platform-runtime/stems-src/reference_tudm_44100_2ch.wav"
# Scratch artifacts stay in the gitignored logs/ dir, resolved from the repo root so the
# script runs from any working directory.
SCRATCH = Path(__file__).resolve().parents[1] / "logs"

SR = 44100
CHORD_HOP = 2048          # hop the HPCP frames are computed with; ChordsDetection needs it
SBIC_FRAME, SBIC_HOP = 2048, 1024


def load_mono(path: Path) -> np.ndarray:
    return es.MonoLoader(filename=str(path), resampleQuality=4)()


def chord_progression(audio: np.ndarray) -> dict:
    """Spectral peaks once, then tuning, then the best major/minor triad per frame window.

    The tuning estimate is not optional: TuningFrequency takes spectral peaks (not audio),
    and without correcting for them every pitch class lands about a semitone low on a track
    that is not tuned to A440, so the whole progression comes out a semitone flat of the key
    KeyExtractor reports.
    """
    window = es.Windowing(type="blackmanharris62")
    spectrum = es.Spectrum()
    peaks = es.SpectralPeaks(orderBy="magnitude", magnitudeThreshold=1e-5,
                             minFrequency=40, maxFrequency=5000, maxPeaks=60)
    pairs = [peaks(spectrum(window(frame)))
             for frame in es.FrameGenerator(audio, frameSize=CHORD_HOP * 2,
                                            hopSize=CHORD_HOP, startFromZero=True)]
    freqs = np.concatenate([f for f, _ in pairs]) if pairs else np.zeros(0, dtype=np.float32)
    mags = np.concatenate([m for _, m in pairs]) if pairs else np.zeros(0, dtype=np.float32)
    if freqs.size:
        tuning_hz, tuning_cents = es.TuningFrequency()(freqs, mags)
    else:
        tuning_hz, tuning_cents = 440.0, 0.0

    hpcp = es.HPCP(size=36, referenceFrequency=float(tuning_hz), harmonics=8, bandPreset=False,
                   minFrequency=40.0, maxFrequency=5000.0, weightType="cosine",
                   windowSize=1.0, sampleRate=SR)
    frames = [hpcp(f, m) for f, m in pairs]
    names, strengths = es.ChordsDetection(hopSize=CHORD_HOP, sampleRate=float(SR),
                                          windowSize=2.0)(np.asarray(frames, dtype=np.float32))
    return {"hop_s": CHORD_HOP / SR, "names": [str(n) for n in names],
            "strengths": [round(float(s), 4) for s in strengths],
            "tuning_hz": round(float(tuning_hz), 3),
            "tuning_cents": round(float(tuning_cents), 1),
            "frames_with_peaks": len(pairs)}


def chord_runs(names: list[str], strengths: list[float], hop_s: float,
               min_frames: int = 6) -> list[dict]:
    """Collapse per-frame chords into runs, dropping low-confidence flicker."""
    runs: list[dict] = []
    for index, (name, strength) in enumerate(zip(names, strengths)):
        if strength < 0.30:
            continue
        if runs and runs[-1]["chord"] == name:
            runs[-1]["end_s"] = round(index * hop_s, 2)
            runs[-1]["frames"] += 1
        else:
            runs.append({"chord": name, "start_s": round(index * hop_s, 2),
                         "end_s": round(index * hop_s, 2), "frames": 1,
                         "strength": round(float(strength), 3)})
    return [r for r in runs if r["frames"] >= min_frames]


def melody(audio: np.ndarray) -> dict:
    """Monophonic pitch contour: how much it sounds, where the notes sit."""
    pitch, confidence = es.PredominantPitchMelodia(
        frameSize=2048, hopSize=128, minFrequency=55.0, maxFrequency=1760.0,
        voicingTolerance=0.2, filterIterations=3)(audio)
    pitch = np.asarray(pitch, dtype=np.float32)
    voiced = np.asarray(confidence) > 0.5
    # keep contour and mask the SAME length, otherwise per_section cannot line them up
    length = min(len(pitch), len(voiced))
    pitch, voiced = pitch[:length], voiced[:length]
    sounding = pitch[voiced]
    hist: dict[str, int] = {}
    if sounding.size:
        midi = np.round(69 + 12 * np.log2(np.maximum(sounding, 1e-6) / 440.0)).astype(int)
        for value, count in zip(*np.unique(midi, return_counts=True)):
            hist[str(int(value))] = int(count)
    return {"hop_s": 128 / SR, "voiced_ratio": round(float(voiced.mean()), 4),
            "median_hz": round(float(np.median(sounding)), 2) if sounding.size else None,
            "range_hz": [round(float(sounding.min()), 1), round(float(sounding.max()), 1)]
            if sounding.size else None,
            "midi_histogram": dict(sorted(hist.items(), key=lambda kv: -kv[1])[:12]),
            "_contour": pitch, "_voiced": voiced}


def structural_changes(audio: np.ndarray) -> dict:
    """SBic wants a feature matrix, not audio: MFCC frames, then frame indices out."""
    window = es.Windowing(type="hann")
    spectrum = es.Spectrum()
    mfcc = es.MFCC(numberCoefficients=13)
    features = np.asarray([mfcc(spectrum(window(frame)))[1]
                           for frame in es.FrameGenerator(audio, frameSize=SBIC_FRAME,
                                                          hopSize=SBIC_HOP, startFromZero=True)],
                          dtype=np.float32).T
    indices = es.SBic(cpw=1.5, minLength=10, size1=300, size2=200,
                      inc1=60, inc2=20)(features)
    return {"features": "MFCC-13", "frame_s": SBIC_HOP / SR,
            "changes_s": [round(float(i) * SBIC_HOP / SR, 2) for i in indices]}


def per_section(sections: list[dict], chords: dict, mel: dict) -> list[dict]:
    """Which chords and which pitches actually sound inside each measured section."""
    runs = chord_runs(chords["names"], chords["strengths"], chords["hop_s"])
    contour, voiced, hop = mel["_contour"], mel["_voiced"], mel["hop_s"]
    out = []
    for section in sections:
        start, end = float(section["start_s"]), float(section["end_s"])
        window = [r for r in runs if r["start_s"] < end and r["end_s"] >= start]
        counts: dict[str, float] = {}
        for run in window:
            counts[run["chord"]] = counts.get(run["chord"], 0.0) + run["frames"]
        a, b = int(start / hop), min(int(end / hop), len(voiced))
        pitched = contour[a:b][voiced[a:b]] if b > a else np.array([])
        midi = (np.round(69 + 12 * np.log2(np.maximum(pitched, 1e-6) / 440.0)).astype(int)
                if pitched.size else np.array([]))
        out.append({
            "start_bar": section["start_bar"], "end_bar": section.get("end_bar"),
            "chords": sorted(counts, key=counts.get, reverse=True)[:4],
            "chord_weights": {k: round(v, 1) for k, v in
                              sorted(counts.items(), key=lambda kv: -kv[1])[:4]},
            "melody_note_ratio": round(float(len(pitched)) / max(1, (b - a)), 3),
            "melody_midi_mode": int(np.bincount(midi - midi.min()).argmax() + midi.min())
            if midi.size else None,
        })
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", type=Path, default=DEFAULT_AUDIO)
    parser.add_argument("--json", type=Path, default=SCRATCH / "reference_harmony.json")
    parser.add_argument("--sections", type=Path, default=SCRATCH / "reference_structure.json")
    args = parser.parse_args()

    started = time.perf_counter()
    audio = load_mono(args.audio)
    report: dict = {"audio": str(args.audio), "seconds": round(len(audio) / SR, 2)}

    key, scale, key_strength = es.KeyExtractor()(audio)
    report["key"] = {"key": str(key), "scale": str(scale), "strength": round(float(key_strength), 3)}
    print(f"tonalidad (KeyExtractor): {key} {scale} (fuerza {key_strength:.3f})")

    chords = chord_progression(audio)
    report["tuning"] = {"hz": chords["tuning_hz"], "cents": chords["tuning_cents"]}
    print(f"tuning: {chords['tuning_hz']} Hz ({chords['tuning_cents']:+.1f} cents) "
          f"sobre {chords['frames_with_peaks']} ventanas")
    report["chord_runs"] = chord_runs(chords["names"], chords["strengths"], chords["hop_s"])
    # signature is (chords, key, scale): key and scale are INPUTS (the reference tonic used
    # to build the circle-of-fifths histogram), while the returned key/scale are the
    # algorithm's own reading of the progression.
    descriptors = es.ChordsDescriptors()(chords["names"], str(key), str(scale))
    # Observed return order, which does NOT match the order declared in the docstring:
    # [0] histogram(24) [1] numberRate [2] changesRate [3] key [4] scale. Verified against a
    # synthetic Dm triad, where the histogram put 100% on Dm and numberRate came out 1/43.
    report["chord_descriptors"] = {"key": str(descriptors[3]), "scale": str(descriptors[4]),
                                  "chord_histogram": [round(float(x), 4) for x in descriptors[0]],
                                  "chord_number_rate": round(float(descriptors[1]), 4),
                                  "changes_rate": round(float(descriptors[2]), 4)}
    print(f"acordes: {len(report['chord_runs'])} tramos | cambios/s "
          f"{report['chord_descriptors']['changes_rate']}")
    top = [f"{r['chord']}({r['frames']})" for r in
           sorted(report["chord_runs"], key=lambda r: -r["frames"])[:8]]
    print("  acordes con mas peso:", " ".join(top))

    mel = melody(audio)
    report["melody"] = {k: v for k, v in mel.items() if not k.startswith("_")}
    print(f"melodia: suena {mel['voiced_ratio'] * 100:.1f}% | mediana {mel['median_hz']} Hz | "
          f"rango {mel['range_hz']}")
    print("  notas MIDI mas frecuentes:", list(report["melody"]["midi_histogram"].items())[:6])

    report["structural_changes"] = structural_changes(audio)
    print("cambios estructurales (SBic sobre MFCC):",
          report["structural_changes"]["changes_s"][:16])

    if args.sections.is_file():
        sections = json.loads(args.sections.read_text(encoding="utf-8")).get("sections", [])
        report["sections"] = per_section(sections, chords, mel)
        changes = report["structural_changes"]["changes_s"]
        matched = 0
        for row in report["sections"]:
            section = next(s for s in sections if s["start_bar"] == row["start_bar"])
            nearest = min(changes, key=lambda c: abs(c - float(section["start_s"]))) if changes else None
            row["sbic_s"] = nearest
            row["sbic_delta_s"] = round(abs(nearest - float(section["start_s"])), 2) if nearest else None
            matched += int(row["sbic_delta_s"] is not None and row["sbic_delta_s"] <= 2.0)
        print(f"\nSBic vs nuestros cortes: {matched}/{len(report['sections'])} dentro de 2 s")
        print(f"{'compas':>7} {'nuestro':>8} {'SBic':>8} {'delta':>6}  acordes")
        for row in report["sections"]:
            print(f"{row['start_bar']:>7.0f} {next(s['start_s'] for s in sections if s['start_bar'] == row['start_bar']):>8.1f} "
                  f"{row['sbic_s'] or 0:>8.1f} {row['sbic_delta_s'] or 0:>6.1f}  "
                  f"{' '.join(row['chords'])}")

    report["seconds_measured"] = round(time.perf_counter() - started, 1)
    args.json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n-> {args.json}  ({report['seconds_measured']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
