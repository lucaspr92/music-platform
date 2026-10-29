"""v3.2: the library bass comes back (gated by the measured map), the synth bass is muted.

v3.1 removed the loop bass because it doubled with the Serum 2 bass. The synth bass then came
out as a machine-gun drone: one pitch per bar for 69 bars, no dynamics, and a patch whose
applied state could not be verified. So the loop returns - governed by the windows the bass
stem actually measured - and the synth track stays at zero volume for an A/B in Live.

Usage: python scripts/plan_replica_v3_2.py
"""
from __future__ import annotations
import json

from pathlib import Path

SCRATCH = Path(__file__).resolve().parents[1] / "logs"
V3_1 = SCRATCH / "tribal_replica_plan_v3_1.json"
V2 = SCRATCH / "tribal_replica_plan_v2.json"
OUT = SCRATCH / "tribal_replica_plan_v3_2.json"


def main() -> int:
    plan = json.loads(V3_1.read_text(encoding="utf-8"))
    v2 = json.loads(V2.read_text(encoding="utf-8"))
    plan["version"] = "3.2"

    # the bass windows are the ones measured on the bass stem in v2, not a guess
    plan["library"]["palette"]["bass"] = v2["palette"]["bass"]
    plan["library"]["placements"]["bass"] = v2["placements"]["bass"]
    plan["synth"]["bass"]["volume"] = 0.0
    plan["synth"]["bass"]["muted_reason"] = (
        "one pitch per bar for 69 bars and no dynamics: not a bassline. Kept at zero volume "
        "so it can be A/B compared in Live against the loop bass.")
    plan["revisions"] = {"from": "v3.1", "changes": [
        "library bass restored, gated by the measured stem windows",
        "synth bass volume 0.0 for A/B",
    ]}
    OUT.write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"plan v3.2 -> {OUT.name}")
    for element, runs in plan["library"]["placements"].items():
        span = sum(r["end_bar"] - r["start_bar"] for r in runs)
        print(f"  libreria {element:8s} {span:6.1f} compases  tramos={[(r['start_bar'], r['end_bar']) for r in runs][:4]}")
    for name, spec in plan["synth"].items():
        print(f"  synth    {name:8s} volumen={spec.get('volume', 'default')}  notas={len(spec['notes'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
