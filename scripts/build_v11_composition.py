"""v11: the composition, measured instead of invented.

Three things were wrong in the previous versions, all of them composition:

  1. The octave folding. The reference's bass really does drop to G1 (49 Hz) and F1 below its D2
     root; "correcting" those into G2/F2 replaced their line with a different one.
  2. The scale snapping. Notes were pushed into D minor by hand, so the line stopped being theirs.
  3. The invented stab pattern. 480 notes that were never in the reference.

Now: the 16th pattern keeps only the steps with solid evidence (a step needs >= 9 measured notes,
otherwise it is a rest rather than a guess), and the drop is transposed to G because the measured
root per section is D in the main and G in the drop. Stabs follow the same root timeline.

Usage: python scripts/build_v11_composition.py
"""
from __future__ import annotations
import json

from pathlib import Path

SCRATCH = Path(__file__).resolve().parents[1] / "logs"

# measured on bars 33-56: sixteenth -> midi, keeping only steps with >= 9 occurrences
PATTERN = {1: 38, 2: 31, 3: 31, 4: 31, 7: 38, 10: 29, 12: 33, 14: 33, 15: 38, 16: 38}
DROPPED_STEPS = {8: 4, 11: 5, 13: 3}   # occurrence counts, too thin to trust
CHORDS = {"D": [62, 65, 69], "G": [67, 70, 74]}   # Dm and Gm, root from the measured timeline


def main() -> int:
    plan = json.loads((SCRATCH / "tribal_replica_plan_v9.json").read_text(encoding="utf-8"))
    structure = json.loads((SCRATCH / "reference_structure.json").read_text(encoding="utf-8"))
    stabs = json.loads((SCRATCH / "stabs_events.json").read_text(encoding="utf-8"))
    total_bars = int(plan["total_bars"])
    bar_s = float(plan["bar_s"])

    def root_of(bar: int) -> str:
        return "G" if bar >= 129 else "D"   # measured: main sits on D, the drop lifts to G

    bass = []
    for bar in range(33, total_bars):
        shift = 5 if root_of(bar) == "G" else 0
        for step, pitch in PATTERN.items():
            bass.append({"pitch": pitch + shift, "start": round(bar * 4 + (step - 1) * 0.25, 3),
                         "duration": 0.22, "velocity": 100 if step in (1, 7, 10, 12, 15) else 88})

    stab_notes = []
    for event in stabs["events"]:
        bar = int(event["bar"])
        if bar < 33:
            continue
        chord = CHORDS[root_of(bar)]
        start = bar * 4 + (int(event["sixteenth"]) - 1) * 0.25
        for pitch in chord:
            stab_notes.append({"pitch": pitch, "start": round(start, 3), "duration": 0.5,
                               "velocity": int(event.get("velocity") or 90)})

    plan["version"] = "v11"
    plan["synth"]["bass"].update({
        "notes": bass, "volume": 0.42, "measured_notes": len(bass),
        "source": "measured 16th pattern, octaves kept as recorded, root timeline D (main) / G (drop)",
        "pattern_per_sixteenth": {str(k): v for k, v in PATTERN.items()},
        "dropped_steps": {str(k): v for k, v in DROPPED_STEPS.items()},
        "limits": ["patch unavailable: Serum 2 exposes one parameter"],
    })
    plan["synth"]["stabs"].update({
        "notes": stab_notes, "volume": 0.45, "measured_notes": len(stab_notes),
        "source": "measured stab rhythm + chord from the measured root (Dm / Gm)",
        "chord_quality_note": "D minor confirmed by KeyExtractor 0.855; bar-level quality scored a tie "
                              "(Dm 0.532 vs D 0.530), so the key decides, not the noisy chroma",
    })
    plan["gains"] = {"conga": 0.5, "djembe": 0.5, "bongo": 0.42, "shaker": 0.42,
                     "core_kick": 0.55, "core_hats": 0.55, "core_clap": 0.6}
    note = "v11: measured composition (no octave folding, no scale snapping, root timeline)"
    if isinstance(plan.get("revisions"), dict):
        plan["revisions"]["v11"] = note
    else:
        plan.setdefault("revisions", []).append(note)
    path = SCRATCH / "tribal_replica_plan_v11.json"
    path.write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")
    print("bajo:", len(bass), "notas | stabs:", len(stab_notes), "notas")
    print("patron:", PATTERN, "| pasos descartados por evidencia fina:", DROPPED_STEPS)
    print("raiz por seccion: main D -> drop G (transposicion +5)")
    print("->", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
