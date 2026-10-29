"""v4: the rendered percussive core replaces the kick and hat loops.

The kick and hat loops brought a groove of their own that was not the reference's. The core is
rendered from the reference's own measured onsets, so the groove now comes from the reference
itself. Hand percussion and bass stay as loops, gated by what the stems measured.

Usage: python scripts/plan_replica_v4.py
"""
from __future__ import annotations
import json

from pathlib import Path

SCRATCH = Path(__file__).resolve().parents[1] / "logs"
V3_2 = SCRATCH / "tribal_replica_plan_v3_2.json"
CORE = SCRATCH / "percussion_core.json"
OUT = SCRATCH / "tribal_replica_plan_v4.json"
REPLACED = ("kick", "hats")


def main() -> int:
    plan = json.loads(V3_2.read_text(encoding="utf-8"))
    core = json.loads(CORE.read_text(encoding="utf-8"))
    plan["version"] = "4"

    removed = {}
    for element in REPLACED:
        if element in plan["library"]["placements"]:
            plan["library"]["placements"].pop(element)
            plan["library"]["palette"].pop(element, None)
            removed[element] = "replaced by the rendered core, measured from the reference"
    plan["core"] = {
        element: {"wav": info["wav"], "instances": info["instances"],
                  "per_bar": info["per_bar"], "measured": info["measured"]}
        for element, info in core["elements"].items()
    }
    plan["removed_elements"] = removed
    plan["revisions"] = {"from": "v3.2", "changes": [
        "kick and hat loops replaced by the rendered percussive core",
        "core kick and hats come from measured onsets on the drums stem",
        "core clap is an assumed backbeat (not isolable from that stem)",
    ]}
    OUT.write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"plan v4 -> {OUT.name}")
    for element, info in plan["core"].items():
        print(f"  core    {element:8s} {info['instances']:>5} golpes  {info['per_bar']:>5.2f}/compas  "
              f"{'medido' if info['measured'] else 'SUPUESTO'}")
    for element, runs in plan["library"]["placements"].items():
        span = sum(r["end_bar"] - r["start_bar"] for r in runs)
        print(f"  libreria {element:8s} {span:6.1f} compases")
    for name, spec in plan["synth"].items():
        print(f"  synth   {name:8s} volumen={spec.get('volume', 'default')} notas={len(spec['notes'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
