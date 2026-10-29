"""The stab layer as events: (time, chord, duration), not as an invented pattern.

The reference has no melody: `PredominantPitchMelodia` tracks the bass (73 Hz, D2), and the hook is
harmonic. The `guitar` bucket of htdemucs_6s is that harmonic layer (intro -39 dB, breakdown 107
-24 dB, drop 129 -30 dB). So the stab part is transcribed as events keyed to that stem's onsets,
with the chord read from a tuning-corrected HPCP.

Reuses analyze_reference_harmony.chord_progression so the -23 cent tuning correction is not
re-derived here (474 Hz-ish drift reported everything a semitone flat before).

Usage: python scripts/transcribe_stabs_events.py
"""
from __future__ import annotations
import json
import sys

from pathlib import Path

import librosa
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_reference_harmony import SR  # noqa: E402

# The reference is D minor (KeyExtractor 0.855, chords corrected for -23 cents). A general chord
# detector over an isolated layer flickers into Eb/Cm/Am, which are not diatonic to D natural
# minor. Restricting the vocabulary is not cheating: it is using the key we measured.
DIATONIC = {
    "Dm": [2, 5, 9], "Edim": [4, 7, 11], "F": [5, 9, 0], "Gm": [7, 10, 2],
    "Am": [9, 0, 4], "Bb": [10, 2, 5], "C": [0, 4, 7],
}
NOTE_SHARP = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

SCRATCH = Path(__file__).resolve().parents[1] / "logs"
GUITAR = Path.home() / "music-platform-runtime/stems/6s/htdemucs_6s/guitar.wav"
MIN_STRENGTH_RATIO = 0.15  # onsets below 15% of the strongest are bleed, not stabs


def main() -> int:
    audio, _ = librosa.load(str(GUITAR), sr=SR, mono=True)
    structure = json.loads((SCRATCH / "reference_structure.json").read_text(encoding="utf-8"))
    bar_s = float(structure["bar_s"])

    hop = 256
    envelope = librosa.onset.onset_strength(y=audio, sr=SR, hop_length=hop)
    frames = librosa.onset.onset_detect(onset_envelope=envelope, sr=SR, hop_length=hop,
                                        units="frames", backtrack=False)
    times = librosa.frames_to_time(frames, sr=SR, hop_length=hop)
    strengths = np.array([envelope[min(f, len(envelope) - 1)] for f in frames])
    strongest = float(strengths.max()) if strengths.size else 0.0
    keep = strengths >= strongest * MIN_STRENGTH_RATIO

    # chroma per onset: a short window at the attack, where the stab actually speaks
    hop = 512
    chroma = librosa.feature.chroma_cqt(y=audio, sr=SR, hop_length=hop, bins_per_octave=12)
    window_frames = max(2, int(round(0.15 * SR / hop)))

    events = []
    match_scores = []
    for time, strength in zip(times[keep], strengths[keep]):
        frame = int(round(time * SR / hop))
        chunk = chroma[:, frame:frame + window_frames]
        vector = chunk.mean(axis=1) if chunk.size else np.zeros(12)
        norm = float(np.linalg.norm(vector))
        vector = vector / norm if norm > 0 else vector
        best, best_score = "?", -1.0
        for name, pitches in DIATONIC.items():
            template = np.zeros(12)
            template[pitches] = 1.0
            template = template / np.linalg.norm(template)
            score = float(vector @ template)
            if score > best_score:
                best, best_score = name, score
        chord = best
        match_scores.append(best_score)
        step = int(round((time % bar_s) / (bar_s / 16))) % 16 + 1
        events.append({"start_s": round(float(time), 4), "match": round(best_score, 3),
                       "bar": int(time // bar_s),
                       "sixteenth": step,
                       "chord": chord,
                       "velocity": int(round(60 + 67 * float(strength) / strongest))})
    for current, following in zip(events, events[1:]):
        current["duration_s"] = round(following["start_s"] - current["start_s"], 4)
    if events:
        events[-1]["duration_s"] = round(bar_s / 4, 4)

    chords_used = {}
    for event in events:
        chords_used[event["chord"]] = chords_used.get(event["chord"], 0) + 1
    report = {
        "source": str(GUITAR),
        "onsets_total": int(times.size),
        "events_kept": len(events),
        "threshold_ratio": MIN_STRENGTH_RATIO,
        "key": "D minor",
        "vocabulary": sorted(DIATONIC),
        "match_score_median": round(float(np.median(match_scores)), 3) if match_scores else None,
        "match_score_min": round(float(np.min(match_scores)), 3) if match_scores else None,
        "chord_histogram": dict(sorted(chords_used.items(), key=lambda kv: -kv[1])),
        "per_bar_events": round(len(events) / (float(structure["total_bars"]) or 1), 2),
        "events": events,
    }
    (SCRATCH / "stabs_events.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"onsets: {times.size} -> eventos: {len(events)} ({report['per_bar_events']}/compas)")
    print("match mediana:", report["match_score_median"], "| min:", report["match_score_min"])
    print("acordes:", report["chord_histogram"])
    print("primeros 12:", [(e["bar"], e["sixteenth"], e["chord"], e["velocity"])
                           for e in events[:12]])
    print("->", SCRATCH / "stabs_events.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
