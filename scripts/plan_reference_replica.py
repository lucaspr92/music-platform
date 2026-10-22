"""Turn the measured structure into an arrangement plan, and assign the palette.

The plan says, per element, in which bar ranges it plays. That is an arrangement
expressed as CLIP PRESENCE (no automation), which is exactly what the bridge can
write: duplicate the loop into the arrangement where it belongs.
"""
from __future__ import annotations
import json
from pathlib import Path

# Scratch artifacts stay in the gitignored logs/ dir, resolved from the repo root so
# the script runs from any working directory.
SCRATCH = Path(__file__).resolve().parents[1] / "logs"


STRUCT = (SCRATCH / "reference_structure.json")
PALETTE = (SCRATCH / "tribal_palette_v1.json")
OUT = (SCRATCH / "tribal_replica_plan.json")

# element -> the role whose measured candidate fills it
ELEMENT_ROLE = {
    "kick": "kick", "perc": "tribal_perc", "hats": "hat_top",
    "shaker": "shaker", "conga": "conga", "djembe": "djembe", "bongo": "bongo",
    "bass": "bass", "hook": "hook",
}
# how a measured section flag maps onto an element
FLAG_FOR = {"kick": "kick", "perc": "hand_perc", "hats": "hats_air",
            "bass": "low_harm", "hook": "mid_harm"}
# elements the measurement cannot separate, so they follow the same flag as perc
FOLLOW_PERC = ("shaker", "conga", "djembe", "bongo")


def main() -> int:
    struct = json.loads(STRUCT.read_text(encoding="utf-8"))
    palette = json.loads(PALETTE.read_text(encoding="utf-8"))
    bar_s = struct["bar_s"]
    parts = []
    for section in struct["sections"]:
        start_bar = round(section["start_s"] / bar_s, 1)
        end_bar = round(section["end_s"] / bar_s, 1)
        if end_bar - start_bar < 3:      # ignore slivers the novelty pass produced
            continue
        on = {}
        for element, flag in FLAG_FOR.items():
            on[element] = bool(section[flag])
        for element in FOLLOW_PERC:
            on[element] = bool(section["hand_perc"])
        parts.append({"start_bar": start_bar, "end_bar": round(start_bar + round(end_bar - start_bar)),
                      "bars": round(end_bar - start_bar, 1), "lufs": section["lufs"], "elements": on})

    # one loop per element, taken from the measured shortlist (first = best score)
    palette_choice = {}
    for element, role in ELEMENT_ROLE.items():
        options = palette.get(role) or []
        if not options:
            continue
        pick = options[0]
        palette_choice[element] = {"filename": pick["filename"], "path": pick["path"],
                                   "bpm_label": pick.get("bpm_file"), "grid_err": pick.get("grid_err"),
                                   "tonic": pick.get("tonic")}
    plan = {"reference": struct["file"], "bpm": struct["bpm"], "bar_s": bar_s,
            "total_bars": round(struct["total_bars"]), "parts": parts,
            "palette": palette_choice,
            "method": "clip presence per section (no automation): a track holds clips only where it plays"}
    OUT.write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"referencia: {plan['reference']}  {plan['bpm']} BPM  {plan['total_bars']} compases\n")
    header = "compases   " + " ".join(f"{e:>7s}" for e in ELEMENT_ROLE)
    print(header)
    print("-" * len(header))
    for part in parts:
        cells = " ".join(f"{('ON' if part['elements'][e] else '.'):>7s}" for e in ELEMENT_ROLE)
        print(f"{part['start_bar']:>4.0f}-{part['end_bar']:<4.0f} {cells}   ({part['lufs']} LUFS)")
    print("\npaleta elegida (por evidencia medida):")
    for element, pick in palette_choice.items():
        print(f"  {element:8s} {pick['filename'][:56]:56s} bpm={pick['bpm_label']} grid_err={pick['grid_err']}")
    print("\n->", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
