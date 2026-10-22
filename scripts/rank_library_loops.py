"""Rank library loops against the reference's measured groove, per role.

Fingerprints come from the STORED index block (sample_library.groove_v1): measured
once, reused forever. This script fills the block for its candidates and then reads
it back, so a second run costs no analysis at all.

Scoring is evidence: the band a role must occupy, how evenly the loop fills the 16th
grid (the reference spreads ~25% per sixteenth), tempo proximity, and for tonal roles
the key named in the file. Taste stays human: this is a shortlist, not a verdict.
"""
from __future__ import annotations
import json, re, sys, time
from pathlib import Path

# Scratch artifacts stay in the gitignored logs/ dir, resolved from the repo root so
# the script runs from any working directory.
SCRATCH = Path(__file__).resolve().parents[1] / "logs"


import numpy as np

from copilot.sample_library.config import index_path
from copilot.sample_library.groove_v1 import attach_groove
from copilot.sample_library.library_v1 import load_index

TARGET_BPM = 128.0
TARGET_GRID = [0.257, 0.252, 0.246, 0.245]
TARGET_BANDS = {"kick_body": -9.58, "drum_body": -4.79, "snare_clap": -6.33, "attack": -8.25, "air": -7.69}

ROLE_KEYWORDS = {
    "kick": ("kick", "kik"),
    "clap": ("clap", "rim", "snare"),
    "hat_top": ("hat loop", "top loop", "hats", "hihat"),
    "shaker": ("shaker", "shak", "cabasa", "guiro"),
    "conga": ("conga",),
    "djembe": ("djembe", "jembe"),
    "bongo": ("bongo", "timba", "timbale"),
    "talking": ("talking", "tama"),
    "tribal_perc": ("percussion loop", "perc loop", "tribal", "afro perc"),
    "bass": ("bass loop",),
    "hook": ("synth loop", "organ", "stab", "melod", "vox", "vocal"),
}
ROLE_BAND = {"kick": "kick_body", "clap": "snare_clap", "hat_top": "air", "shaker": "air",
             "conga": "drum_body", "djembe": "drum_body", "bongo": "drum_body",
             "talking": "drum_body", "tribal_perc": "drum_body", "bass": "kick_body", "hook": None}
MAX_PER_ROLE = 16
KEY_HINT = (" dm", "d min", " d.", " d ", "_d_", "d.wav")


def score(role: str, g: dict) -> tuple[float, dict]:
    detail: dict = {}
    bpm = g.get("bpm") or 0.0
    bpm_err = abs(bpm - TARGET_BPM) if bpm else 99.0
    detail.update(bpm=bpm, bpm_err=round(bpm_err, 2))

    band = ROLE_BAND[role]
    if band:
        share = (g.get("percussive_band_share_db") or {}).get(band)
        band_err = abs((share if share is not None else -99.0) - TARGET_BANDS[band])
        detail.update(band=band, band_db=share, band_err=round(band_err, 2))
    else:
        perc = g.get("percussive_over_harmonic") or 1.0
        band_err = max(0.0, perc - 0.5) * 10
        detail.update(perc_over_harm=perc, band_err=round(band_err, 2))

    six = (g.get("grid") or {}).get("sixteenths")
    grid_err = float(np.mean(np.abs(np.array(six) - np.array(TARGET_GRID)))) * 100 if six else 25.0
    detail.update(grid_sixteenths=six, grid_err=round(grid_err, 2))

    total = 3.0 * bpm_err + 1.5 * band_err + 1.0 * grid_err
    if role in ("bass", "hook"):
        name = str(g.get("_filename", "")).lower()
        key_penalty = 0.0 if any(h in name for h in KEY_HINT) else 6.0
        detail["key_penalty"] = key_penalty
        total += key_penalty
    detail["score"] = round(total, 2)
    return total, detail


def main() -> int:
    started = time.perf_counter()
    path = index_path()
    index = load_index(path)
    if index is None:
        print("NO_INDEX"); return 2
    def label_bpm(name: str) -> float | None:
        """The packs label their loops; that label beats any short-loop estimate."""
        for token in re.findall(r"\d{2,3}", name):
            value = float(token)
            if 60.0 <= value <= 190.0:
                return value
        return None

    def pre_rank(asset) -> tuple:
        label = label_bpm(asset.filename)
        if label is None:
            return (2, 0.0, asset.filename)                 # unlabeled: measured, no free evidence
        if abs(label - TARGET_BPM) <= 6:
            return (0, abs(label - TARGET_BPM), asset.filename)   # labeled and on target
        return (1, abs(label - TARGET_BPM), asset.filename)       # labeled, off target

    role_cands: dict[str, list] = {}
    for role, keywords in ROLE_KEYWORDS.items():
        rows = [(sha, asset) for sha, asset in index.assets.items()
                if asset.status == "INDEXED" and asset.extension.lower() in (".wav", ".aif", ".aiff")
                and any(k in asset.filename.lower() for k in keywords)]
        rows.sort(key=lambda pair: pre_rank(pair[1]))   # quality first, not index order
        role_cands[role] = rows[:MAX_PER_ROLE]
    wanted = {sha for rows in role_cands.values() for sha, _ in rows}
    print(f"candidatos: {len(wanted)} shas en {len(role_cands)} roles")
    report = attach_groove(path, only_shas=wanted, progress=lambda d, t: print(f"  groove {d}/{t}", end="\r"))
    print(f"\ngroove: {json.dumps(report)}")
    print(f"tiempo total: {time.perf_counter() - started:.1f}s")

    index = load_index(path)   # re-read: now carrying the stored fingerprints
    out = {}
    for role, rows in role_cands.items():
        scored = []
        for sha, _stale in rows:
            asset = index.assets.get(sha)      # re-read: the fresh objects carry the stored groove
            if asset is None or asset.groove is None:
                continue
            g = asset.groove.model_dump()
            g["_filename"] = asset.filename
            total, detail = score(role, g)
            scored.append({"path": asset.path, "filename": asset.filename,
                           "sha256": sha, "score": total, "detail": detail})
        scored.sort(key=lambda r: r["score"])
        out[role] = scored
        print(f"\n=== {role}: {len(scored)} medidos ===")
        for r in scored[:4]:
            d = r["detail"]
            print(f"  {r['filename'][:46]:46s} score={r['score']:7.2f} bpm={d.get('bpm')} "
                  f"{d.get('band')}={d.get('band_db')} grid={d.get('grid_sixteenths')}")
    (SCRATCH / "tribal_fingerprint_ranking.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n-> logs/tribal_fingerprint_ranking.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
