"""v3.1: v3 minus the library elements that duplicate or smear.

What the first v3 build got wrong, and why this exists:

  * a library loop bass AND a Serum 2 bass were both playing -> removed, Serum 2 covers it
  * a library loop hook AND the Serum 2 stabs -> removed, the stabs cover it
  * a generic percussion loop ran for 191 of 193 bars on top of conga, djembe, bongo and
    shaker -> removed; the specific hand-percussion tracks are already gated by the measured
    body density, so the generic one only added smear

What stays: the kick and hat loops, the four gated hand-percussion loops, the Serum 2 bass and
stabs. Re-picking the loops themselves needs instrument identity (PANNs), which is downloading.

Usage: python scripts/plan_replica_v3_1.py
"""
from __future__ import annotations
import json

from pathlib import Path

SCRATCH = Path(__file__).resolve().parents[1] / "logs"
V3 = SCRATCH / "tribal_replica_plan_v3.json"
OUT = SCRATCH / "tribal_replica_plan_v3_1.json"
DROPPED = ("bass", "hook", "perc")
REASON = {
    "bass": "replaced by the Serum 2 bass (was playing simultaneously)",
    "hook": "replaced by the Serum 2 stabs (was playing simultaneously)",
    "perc": "generic loop spanning 191 of 193 bars on top of the specific hand percussion",
}


def main() -> int:
    plan = json.loads(V3.read_text(encoding="utf-8"))
    plan["version"] = "3.1"
    removed = {}
    for element in DROPPED:
        if element in plan["library"]["placements"]:
            plan["library"]["placements"].pop(element)
            plan["library"]["palette"].pop(element, None)
            removed[element] = REASON[element]
    plan["revised_from"] = "v3"
    plan["removed_elements"] = removed
    OUT.write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"plan v3.1 -> {OUT.name}")
    print("  quitados:")
    for element, why in removed.items():
        print(f"    {element:8s} {why}")
    print("  quedan en libreria:")
    for element, runs in plan["library"]["placements"].items():
        span = sum(r["end_bar"] - r["start_bar"] for r in runs)
        print(f"    {element:8s} {span:6.1f} compases")
    print("  sintetizados (Serum 2):")
    for name, spec in plan["synth"].items():
        print(f"    {name:8s} {len(spec['notes']):>5} notas  preset={spec.get('preset_uri')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
