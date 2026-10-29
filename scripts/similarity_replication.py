"""Replication score: per element, level-invariant, with no mix terms.

The older scorer averaged in `band_balance` and `energy_curve`, which are mix: mastering moved it
+0.037 without changing a note. This one cannot be moved that way. It compares element against
element, using sources that are isolated on both sides:

  reference : the drums stem split per instrument (kick/snare/hihat/cymbals/toms), the bass stem,
              the guitar stem (the stabs layer). See separate_drums_per_instrument.py.
  replica   : the plan's own isolated sources. The core is rendered per instrument already; loops
              are rendered one at a time by zeroing every other gain.

Four sub-metrics per element, each in [0, 1]:
  presence  per-bar audibility, Jaccard against the reference
  pattern   where hits fall inside the bar: 16th histogram, total-variation agreement
  sequence  the same agreement computed per section, so a pattern that drifts is penalised
  timbre    MFCC profile cosine with c0 dropped, which makes it level-invariant

Deliberately NOT here: loudness, band balance, stereo width. Those belong to the mix stage and
live in similarity_to_reference.py.

Usage: python scripts/similarity_replication.py [--plan logs/tribal_replica_plan_v8.json]
"""
from __future__ import annotations
import json
import subprocess
import sys
import tempfile

from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
SCRATCH = ROOT / "logs"
PER_DRUM = Path.home() / "music-platform-runtime/stems/drums_per_instrument"
FT = Path.home() / "music-platform-runtime/stems/ft/htdemucs_ft"
SIX = Path.home() / "music-platform-runtime/stems/6s/htdemucs_6s"
BEATS_PER_BAR = 4
SIXTEENTHS = 16

ELEMENTS = {
    # element      reference stem          replica source
    "kick":        ("kick",                 ("core", "kick")),
    "clap":        ("snare",                ("core", "clap")),
    "hihat":       ("hh",                   ("core", "hats")),
    "percussion":  ("toms",                 ("loops", ("conga", "djembe", "bongo", "shaker"))),
    "bass":        ("bass",                 ("loop", "bass")),
}


def load(path: Path) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(str(path), always_2d=True, dtype="float32")
    return audio.mean(axis=1), sr


def bar_and_section_grid(plan: dict) -> tuple[float, int, int, list[tuple[int, int]]]:
    bar_s = float(plan["bar_s"])
    total_bars = int(plan["total_bars"])
    sections = [(int(s["start_bar"]), int(s["start_bar"]) + int(s.get("bars") or 1))
                for s in (plan.get("structure") or {}).get("sections", [])]
    if not sections:
        sections = [(0, total_bars)]
    return bar_s, total_bars, SIXTEENTHS, sections


def presence_vector(mono: np.ndarray, sr: int, bar_s: float, total_bars: int) -> np.ndarray:
    """Per-bar audibility, decided against the signal's own median so no fixed dB floor is needed."""
    frame = max(1, int(bar_s * sr))
    levels = []
    for bar in range(total_bars):
        chunk = mono[bar * frame:(bar + 1) * frame]
        levels.append(float(np.sqrt(np.mean(chunk ** 2))) if chunk.size else 0.0)
    levels_db = 20 * np.log10(np.array(levels) + 1e-9)
    audible = levels_db > (np.median(levels_db) - 6.0)
    return audible


def sixteenth_grid(mono: np.ndarray, sr: int, bar_s: float, total_bars: int) -> np.ndarray:
    """Which 16th of the bar carries an onset, counted over the whole track."""
    hop = 256
    envelope = librosa.onset.onset_strength(y=mono, sr=sr, hop_length=hop)
    frames = librosa.onset.onset_detect(onset_envelope=envelope, sr=sr, hop_length=hop,
                                        units="frames")
    times = librosa.frames_to_time(frames, sr=sr, hop_length=hop)
    six_s = bar_s / SIXTEENTHS
    counts = np.zeros(SIXTEENTHS)
    for time in times:
        step = int(round((time % bar_s) / six_s)) % SIXTEENTHS
        counts[step] += 1
    return counts


def per_section_grid(mono: np.ndarray, sr: int, bar_s: float,
                     sections: list[tuple[int, int]]) -> list[np.ndarray]:
    grids = []
    for start, end in sections:
        start_s = start * bar_s
        stop_s = end * bar_s
        chunk = mono[int(start_s * sr):int(stop_s * sr)]
        if chunk.size < sr:
            continue
        grids.append(sixteenth_grid(chunk, sr, bar_s, end - start))
    return grids


def timbre_vector(mono: np.ndarray, sr: int) -> np.ndarray | None:
    """MFCC plus the descriptors that separate a transient from noise.

    MFCC alone scored white noise at 0.53 against a kick, which is useless: the flatness and the
    centroid are what tell those two apart, so they belong in the vector.
    """
    if mono.size < sr // 2:
        return None
    audio = mono.astype(np.float32)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)[1:]  # drop c0: level-invariant
    centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
    flatness = librosa.feature.spectral_flatness(y=audio)
    rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(audio)
    shape = np.array([
        np.log10(np.mean(centroid) + 1.0) / 4.0,
        np.mean(flatness),
        np.log10(np.mean(rolloff) + 1.0) / 4.0,
        np.mean(zcr),
    ])
    return np.concatenate([mfcc.mean(axis=1) / 20.0, mfcc.std(axis=1) / 20.0, shape])


def tv_agreement(a: np.ndarray, b: np.ndarray) -> float:
    a_sum, b_sum = a.sum(), b.sum()
    if a_sum <= 0 or b_sum <= 0:
        return 1.0 if a_sum == b_sum else 0.0
    return float(1.0 - 0.5 * np.abs(a / a_sum - b / b_sum).sum())


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(a, b).sum() / union)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    if denominator <= 0:
        return 0.0
    return float(np.clip(a @ b / denominator, 0.0, 1.0))


def bounded_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Distance mapped to [0, 1] against the vector's own scale, so it saturates honestly."""
    scale = 0.5 * (np.linalg.norm(a) + np.linalg.norm(b))
    if scale <= 0:
        return 1.0
    return float(np.clip(1.0 - np.linalg.norm(a - b) / (2.0 * scale), 0.0, 1.0))


def score_element(reference: np.ndarray, replica: np.ndarray, sr: int, bar_s: float,
                  total_bars: int, sections: list[tuple[int, int]]) -> dict:
    presence = jaccard(presence_vector(reference, sr, bar_s, total_bars),
                       presence_vector(replica, sr, bar_s, total_bars))
    pattern = tv_agreement(sixteenth_grid(reference, sr, bar_s, total_bars),
                           sixteenth_grid(replica, sr, bar_s, total_bars))
    ref_sections = per_section_grid(reference, sr, bar_s, sections)
    rep_sections = per_section_grid(replica, sr, bar_s, sections)
    pairs = list(zip(ref_sections, rep_sections))
    sequence = float(np.mean([tv_agreement(a, b) for a, b in pairs])) if pairs else pattern
    ref_timbre, rep_timbre = timbre_vector(reference, sr), timbre_vector(replica, sr)
    timbre = (bounded_similarity(ref_timbre, rep_timbre)
              if ref_timbre is not None and rep_timbre is not None else None)
    # presence alone is weak evidence: a constant noise floor is "audible" in every bar, so it is
    # weighted down instead of being allowed a free 1.0.
    weights = {"presence": 0.15, "pattern": 0.35, "sequence": 0.25, "timbre": 0.25}
    parts = {"presence": presence, "pattern": pattern, "sequence": sequence}
    if timbre is not None:
        parts["timbre"] = timbre
    total_weight = sum(weights[name] for name in parts)
    score = sum(parts[name] * weights[name] for name in parts) / total_weight
    return {"presence": round(presence, 4), "pattern": round(pattern, 4),
            "sequence": round(sequence, 4),
            "timbre": round(timbre, 4) if timbre is not None else None,
            "score": round(float(score), 4)}


def render_element(plan: dict, element: str, out: Path) -> Path:
    """Render one loop element alone by zeroing every other gain."""
    variant = json.loads(json.dumps(plan))
    gains = variant.setdefault("gains", {})
    for key in list(gains):
        gains[key] = 0.0
    gains[element] = 1.0
    for name in (variant.get("synth") or {}):
        variant["synth"][name]["volume"] = 0.0
    plan_path = out / f"only_{element}.json"
    plan_path.write_text(json.dumps(variant, indent=2), encoding="utf-8")
    rendered = out / f"only_{element}.wav"
    subprocess.run([sys.executable, str(ROOT / "scripts/render_plan_audio.py"),
                    "--plan", str(plan_path), "--out", str(rendered)],
                   cwd=ROOT, check=True, capture_output=True, text=True)
    return rendered


def replica_source(element: str, spec: tuple, plan: dict, tmp: Path) -> Path:
    kind, value = spec
    if kind == "core":
        return Path(plan["core"][value]["wav"])
    if kind == "loop":
        return render_element(plan, value, tmp)
    if kind == "loops":
        # the four percussion loops together: render them as a group
        variant = json.loads(json.dumps(plan))
        gains = variant.setdefault("gains", {})
        for key in list(gains):
            gains[key] = 0.0
        for name in value:
            gains[name] = 1.0
        for name in (variant.get("synth") or {}):
            variant["synth"][name]["volume"] = 0.0
        plan_path = tmp / "only_percussion.json"
        plan_path.write_text(json.dumps(variant, indent=2), encoding="utf-8")
        rendered = tmp / "only_percussion.wav"
        subprocess.run([sys.executable, str(ROOT / "scripts/render_plan_audio.py"),
                        "--plan", str(plan_path), "--out", str(rendered)],
                       cwd=ROOT, check=True, capture_output=True, text=True)
        return rendered
    raise ValueError(kind)


def reference_source(stem: str) -> Path:
    # DrumSep names its outputs drums_<klass>; the 6-stem model targets a drum KIT, so hand
    # percussion lands in the snare/toms buckets. Named honestly here:
    #   kick  -> drums_kick   validated against the band count (2.99 vs 2.65 per bar)
    #   clap  -> drums_snare  CONTAMINATED: this bucket also carries the tribal percussion
    #   hihat -> drums_hh     validated against the band count (11.44 vs 11.92 per bar)
    if stem == "bass":
        return FT / "bass.wav"
    if stem == "toms+cymbals":
        raise ValueError("aggregate")
    return PER_DRUM / f"drums_{stem}.wav"


def reference_aggregate(stems: list[str]) -> np.ndarray:
    signals = []
    sr_ref = None
    for name in stems:
        mono, sr = load(PER_DRUM / f"{name}.wav")
        signals.append(mono)
        sr_ref = sr
    length = min(len(s) for s in signals)
    return np.sum([s[:length] for s in signals], axis=0), sr_ref


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", default=str(SCRATCH / "tribal_replica_plan_v8.json"))
    parser.add_argument("--json", default=str(SCRATCH / "replication_score.json"))
    args = parser.parse_args()
    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    bar_s, total_bars, _, sections = bar_and_section_grid(plan)

    report: dict = {"plan": args.plan, "bar_s": bar_s, "total_bars": total_bars,
                    "elements": {}, "method": "per element, level-invariant, no mix terms"}
    with tempfile.TemporaryDirectory() as tmp_name:
        tmp = Path(tmp_name)
        placements = (plan.get("library") or {}).get("placements") or {}
        for element, (stem, spec) in ELEMENTS.items():
            kind, value = spec
            if kind == "loop" and value not in placements:
                # v9 replaced the bass loop with a transcribed part. The PART is exact by
                # construction, but the audio cannot be rendered offline (no Serum), so scoring
                # it here would measure silence and lie. Reported, not scored.
                report["elements"][element] = {
                    "score": None,
                    "reason": "part transcribed onto Serum 2; audio not renderable offline",
                    "part": {"notes": (plan.get("synth", {}).get(value) or {}).get("measured_notes")}}
                print(f"{element:12s} PART-ONLY (transcribed, audio unmeasurable offline)")
                continue
            if stem == "toms":
                reference, sr = load(PER_DRUM / "drums_toms.wav")
            else:
                reference, sr = load(reference_source(stem))
            replica, sr_rep = load(replica_source(element, spec, plan, tmp))
            if sr != sr_rep:
                raise SystemExit(f"SR mismatch for {element}: {sr} vs {sr_rep}")
            report["elements"][element] = score_element(
                reference, replica, sr, bar_s, total_bars, sections)
            print(f"{element:12s} {report['elements'][element]}")

    scores = [row["score"] for row in report["elements"].values() if row.get("score") is not None]
    excluded = [name for name, row in report["elements"].items() if row.get("score") is None]
    report["replication"] = round(float(np.mean(scores)), 4)
    report["scored_elements"] = len(scores)
    report["excluded_elements"] = excluded
    print()
    print(f"REPLICATION = {report['replication']:.4f} over {report['scored_elements']} elements"
          f" | excluidos: {excluded}")
    Path(args.json).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("->", args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
