"""Plan v2: the arrangement map, upgraded with what the stems revealed.

v1 could only see the mix, so bass and vocals looked present everywhere and all four hand
percussion tracks entered and left together. The stems showed:
  * the bass is silent until bar 25 (a 50 dB jump, not a threshold artefact)
  * the vocal stem runs from bar 25 to about bar 153, silent before and after
  * the hand-percussion body has a measured density per section, from 1 to 4 layers

v2 therefore narrows bass and hook to the measured windows and LAYERS the hand percussion
by density instead of switching all four at once. What it deliberately does not do: it does
not distinguish conga from djembe from bongo, because four stems separate families, not
instruments inside a family.

Usage: python scripts/plan_replica_v2_from_stems.py
"""
from __future__ import annotations
import json

from pathlib import Path

# Scratch artifacts stay in the gitignored logs/ dir, resolved from the repo root so
# the script runs from any working directory.
SCRATCH = Path(__file__).resolve().parents[1] / "logs"

PLAN_V1 = SCRATCH / "tribal_replica_plan.json"
STEMS = SCRATCH / "reference_stems_report.json"
OUT = SCRATCH / "tribal_replica_plan_v2.json"

# the pool that gets layered, in musical order: base, answer, spice, texture
LAYER_POOL = ("conga", "djembe", "bongo", "shaker")


def runs(predicate, parts) -> list[dict]:
    """Merge consecutive parts where the predicate holds into bar runs."""
    out: list[list[float]] = []
    for part in parts:
        if not predicate(part):
            continue
        if out and abs(out[-1][1] - part["start_bar"]) < 0.5:
            out[-1][1] = part["end_bar"]
        else:
            out.append([part["start_bar"], part["end_bar"]])
    return [{"start_bar": a, "end_bar": b} for a, b in out]


def main() -> int:
    v1 = json.loads(PLAN_V1.read_text(encoding="utf-8"))
    stems = json.loads(STEMS.read_text(encoding="utf-8"))
    parts = v1["parts"]

    # sections keyed by start bar, carrying the stem evidence
    by_bar = {round(s["start_bar"]): s for s in stems["sections"]}
    evidence = {}
    for part in parts:
        key = min(by_bar, key=lambda b: abs(b - part["start_bar"])) if by_bar else None
        section = by_bar.get(key) if key is not None else None
        evidence[part["start_bar"]] = {
            "bass_present": bool(section and section["stem_present"].get("bass")),
            "vocals_present": bool(section and section["stem_present"].get("vocals")),
            "body_layers": int(section.get("suggested_body_layers", 1)) if section else 1,
        }

    placements: dict[str, list[dict]] = {}
    placements["kick"] = runs(lambda p: p["elements"].get("kick"), parts)
    placements["hats"] = runs(lambda p: p["elements"].get("hats"), parts)
    placements["perc"] = runs(lambda p: p["elements"].get("perc"), parts)
    placements["bass"] = runs(lambda p: evidence[p["start_bar"]]["bass_present"], parts)
    placements["hook"] = runs(lambda p: evidence[p["start_bar"]]["vocals_present"], parts)
    for index, element in enumerate(LAYER_POOL):
        placements[element] = runs(
            lambda p, i=index: (p["elements"].get("perc")
                                and evidence[p["start_bar"]]["body_layers"] > i), parts)

    plan = {
        "version": 2,
        "reference": v1["reference"], "bpm": v1["bpm"], "bar_s": v1["bar_s"],
        "total_bars": v1["total_bars"], "palette": v1["palette"],
        "method": "clip presence per section; hand percussion layered by measured body density",
        "evidence": {str(k): v for k, v in evidence.items()},
        "placements": placements,
    }
    OUT.write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"v2 sobre {plan['reference'][:42]}  {plan['bpm']} BPM  {plan['total_bars']} compases\n")
    elements = list(placements)
    header = f"{'desde':>7} " + " ".join(f"{e:>6s}" for e in elements) + "   capas"
    print(header)
    print("-" * len(header))
    for part in parts:
        key = part["start_bar"]
        cells = []
        for element in elements:
            on = any(r["start_bar"] <= key < r["end_bar"] for r in placements[element])
            cells.append(f"{('ON' if on else '.'):>6s}")
        mark = "  <= denso" if evidence[key]["body_layers"] >= 3 else ""
        print(f"{key:>7.0f} " + " ".join(cells) + f"   {evidence[key]['body_layers']}{mark}")
    print("\ncompases cubiertos por elemento en v2:")
    for element in elements:
        span = sum(r["end_bar"] - r["start_bar"] for r in placements[element])
        print(f"  {element:8s} {span:6.1f} compases en {len(placements[element])} tramos")
    print("\n->", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
