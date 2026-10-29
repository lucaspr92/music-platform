"""Rank library candidates for a role by measured fit, with an optional key filter.

Usage: python scripts/rank_role_candidates.py --role BASS --key "D"
       python scripts/rank_role_candidates.py --role SYNTH
"""
from __future__ import annotations
import argparse, json

from pathlib import Path

SCRATCH = Path(__file__).resolve().parents[1] / "logs"
INDEX = Path.home() / "Library/Application Support/MusicPlatform/sample-library-index.json"
TARGET_BPM = 128.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", required=True)
    parser.add_argument("--key", default=None, help="prefer loops whose key contains this")
    parser.add_argument("--top", type=int, default=10)
    args = parser.parse_args()

    assets = list(json.loads(INDEX.read_text(encoding="utf-8"))["assets"].values())
    rows = []
    for asset in assets:
        if str(asset.get("semantic_role")).upper() != args.role.upper():
            continue
        groove = asset.get("groove") or {}
        bpm = float(groove.get("bpm") or 0.0)
        if bpm <= 0 or not groove.get("grid"):
            continue
        key = str(groove.get("key") or "")
        key_match = bool(args.key) and args.key.lower() in key.lower()
        tempo_penalty = min(abs(bpm - TARGET_BPM) / 40.0, 1.0)
        bands = groove.get("percussive_band_share_db") or {}
        rows.append({"path": asset["path"], "file": Path(asset["relative_path"]).name,
                     "bpm": round(bpm, 2), "key": key,
                     "key_strength": round(float(groove.get("key_strength") or 0), 3),
                     "key_match": key_match, "tempo_penalty": round(tempo_penalty, 3),
                     "bands": {k: round(float(v), 1) for k, v in bands.items()},
                     "score": round((1.0 - tempo_penalty) * (1.25 if key_match else 1.0), 4)})
    rows.sort(key=lambda item: (-item["score"], -item["key_strength"]))

    print(f"{args.role}: {len(rows)} candidatos con fingerprint valido"
          + (f" | filtro de tonalidad: {args.key}" if args.key else ""))
    print()
    print(f"{'score':>6} {'bpm':>7} {'tonalidad':>12} {'fuerza':>6}  archivo")
    print("-" * 88)
    for row in rows[: args.top]:
        print(f"{row['score']:>6.3f} {row['bpm']:>7.2f} {row['key']:>12} {row['key_strength']:>6.2f}  "
              f"{row['file'][:48]}")

    out = SCRATCH / f"ranking_{args.role.lower()}.json"
    out.write_text(json.dumps({"role": args.role, "key_filter": args.key, "rows": rows},
                              indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
