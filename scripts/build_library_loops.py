"""Three 8-bar grooves built from the indexed loop library, not from synthesised one-shots.

"Use the library" is the right instinct: a Full Drum Loop at 128 from a current tech house pack
already carries the kick, the hats and the percussion in a way that fits together, which is what a
producer starts from. The bass is the one thing the pack cannot supply in the right key, so it is a
library bass loop in D minor.

Each variant changes the loop combination, not a hand-written pattern, so the comparison is between
real grooves instead of between my ideas.

Usage: python scripts/build_library_loops.py
"""
from __future__ import annotations
import json
import re

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRATCH = ROOT / "logs"
OUT = SCRATCH / "library_loops"
INDEX = Path.home() / "Library/Application Support/MusicPlatform/sample-library-index.json"
BAR_S = 1.87486
BARS = 8

VARIANTS = {
    "A_full_drums": {
        "drums": "Phantom - Fiesta 4 Full Drum Loop 1 - 128 BPM.wav",
        "bass": "Phantom - Fiesta 4 Bass Loop 23 - 128 BPM Dm.wav",
    },
    "B_tribal_layer": {
        "drums": "Phantom - Fiesta 4 Full Drum Loop 7 - 128 BPM.wav",
        "bass": "Phantom - Fiesta 4 Bass Loop 23 - 128 BPM Dm.wav",
        "percussion": "Phantom - Fiesta 4 Percussion Loop 12 - 128 BPM.wav",
    },
    "C_hats_on_top": {
        "drums": "Phantom - Fiesta 4 Full Drum Loop 14 - 128 BPM.wav",
        "bass": "Phantom - Fiesta 4 Bass Loop 25 - 128 BPM Dm.wav",
        "hats": "Phantom - Fiesta 4 Hat Loop 9 - 128 BPM.wav",
    },
}
GAINS = {"drums": 0.8, "bass": 0.62, "percussion": 0.32, "hats": 0.26}


def main() -> int:
    assets = list(json.loads(INDEX.read_text(encoding="utf-8"))["assets"].values())
    by_name = {}
    for asset in assets:
        by_name.setdefault(Path(asset["relative_path"]).name, asset)
    OUT.mkdir(parents=True, exist_ok=True)
    base = json.loads((SCRATCH / "tribal_replica_plan_v12.json").read_text(encoding="utf-8"))
    master = {"version": "library_loops", "bpm": 128.0, "bar_s": BAR_S, "total_bars": BARS,
              "core": {}, "synth": {}, "gains": {}, "library": {"palette": {}, "placements": {}}}

    for name, parts in VARIANTS.items():
        plan = json.loads(json.dumps(master))
        missing = []
        for element, filename in parts.items():
            asset = by_name.get(filename)
            if asset is None:
                missing.append(filename)
                continue
            plan["library"]["palette"][element] = {
                "path": asset["path"], "filename": filename,
                "bpm_label": 128.0, "tonic": "Dm" if element == "bass" else None}
            plan["library"]["placements"][element] = [{"start_bar": 0, "end_bar": BARS}]
            plan["gains"][element] = GAINS[element]
        if missing:
            print(f"  {name}: FALTAN {missing}")
            continue
        path = OUT / f"plan_{name}.json"
        path.write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  {name:16s} " + " + ".join(Path(v).name[:34] for v in parts.values()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
