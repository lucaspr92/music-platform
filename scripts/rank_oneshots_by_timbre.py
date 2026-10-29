"""Rank ONE-SHOT candidates by timbre distance to the reference's measured element.

Why one-shots and not loops: the percussive core is built from one-shots placed at the reference's
own measured onsets. A loop would take back the grid we already matched. But the ranking must
still be filtered musically first, or timbre alone picks a bass at 123 BPM in the wrong key.

This ranks only ONE_SHOT assets, which carry no tempo or key, so timbre is the only discriminating
axis and the ranking is trustworthy here.

Usage: python scripts/rank_oneshots_by_timbre.py
"""
from __future__ import annotations
import json
import sys

from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_timbre import features, load, to_vector  # noqa: E402

SCRATCH = Path(__file__).resolve().parents[1] / "logs"
INDEX = Path.home() / "Library/Application Support/MusicPlatform/sample-library-index.json"
ROLES = {"kick": ("KICK",), "clap": ("CLAP",), "hats": ("TOP_LOOP", "RIDE", "HIHAT")}
MAX_SECONDS = 3.0


def main() -> int:
    profile = json.loads((SCRATCH / "timbre_profile.json").read_text(encoding="utf-8"))
    assets = list(json.loads(INDEX.read_text(encoding="utf-8"))["assets"].values())
    out: dict = {"method": "one-shots only, z-scored features, cosine distance",
                 "rankings": {}}
    for role, semantic in ROLES.items():
        if role not in profile["reference"]:
            continue
        ref_vector = to_vector(profile["reference"][role])
        rows = []
        for asset in assets:
            if str(asset.get("sample_type")) != "ONE_SHOT":
                continue
            if str(asset.get("semantic_role")).upper() not in semantic:
                continue
            if float(asset.get("duration_s") or 0) > MAX_SECONDS:
                continue
            try:
                feat = features(load(Path(asset["path"])))
            except Exception:
                continue
            if feat:
                rows.append({"file": Path(asset["relative_path"]).name,
                             "path": asset["path"], "features": feat})
        if not rows:
            continue
        matrix = np.array([to_vector(r["features"]) for r in rows])
        pooled = np.vstack([matrix, ref_vector[None, :]])
        mean, std = pooled.mean(axis=0), pooled.std(axis=0) + 1e-9
        z = (matrix - mean) / std
        z_ref = (ref_vector - mean) / std
        distance = 1.0 - (z @ z_ref) / (
            np.linalg.norm(z, axis=1) * np.linalg.norm(z_ref) + 1e-12)
        order = np.argsort(distance)
        out["rankings"][role] = [
            {"file": rows[i]["file"], "path": rows[i]["path"],
             "timbre_distance": round(float(distance[i]), 4)} for i in order[:15]]
        print(f"=== {role}: {len(rows)} one-shots, mas cercano en TIMBRE ===")
        for i in order[:8]:
            print(f"  {distance[i]:.3f}  {rows[i]['file'][:64]}")
        print()
    (SCRATCH / "oneshot_timbre_ranking.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("->", SCRATCH / "oneshot_timbre_ranking.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
