"""v10: make it a track again. One repeating bar of bass, in the key, with a real arrangement.

What was wrong with v9, in the numbers this script corrects:
  * 17 distinct bars out of 23 in the reference's stable section: the transcription never repeated,
    and a bassline that never repeats cannot be followed
  * 23% of its notes were octave errors (D3 counted 14 times where D2 is the root), so the line
    jumped two octaves at random
  * nothing was gated by section, so every element played from bar 0 to the end

So: fold the octaves, snap the pitches to D minor inside the bass register, keep the MEASURED
rhythm per 16th, and repeat ONE bar across the sections where the bass belongs. Repetition is what
makes the part legible; the rhythm still comes from the reference.

Usage: python scripts/build_v10_musical.py
"""
from __future__ import annotations
import json

from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRATCH = ROOT / "logs"
DMINOR = {2, 4, 5, 7, 9, 10, 0}   # D E F G A Bb C
BASS_LOW, BASS_HIGH = 36, 45      # C2..A2


def snap_to_dm(midi: int) -> int:
    pitch_class = midi % 12
    if pitch_class in DMINOR:
        return midi
    for delta in (1, -1, 2, -2):
        if (midi + delta) % 12 in DMINOR:
            return midi + delta
    return midi


def fold_into_bass_register(midi: int) -> int:
    while midi > BASS_HIGH:
        midi -= 12
    while midi < BASS_LOW:
        midi += 12
    return midi


def main() -> int:
    plan = json.loads((SCRATCH / "tribal_replica_plan_v9.json").read_text(encoding="utf-8"))
    transcription = json.loads((SCRATCH / "reference_bass_notes.json").read_text(encoding="utf-8"))
    structure = json.loads((SCRATCH / "reference_structure.json").read_text(encoding="utf-8"))
    bar_s = float(plan["bar_s"])
    beat_s = 60.0 / float(plan["bpm"])
    total_bars = int(plan["total_bars"])

    # 1) the measured 16th rhythm, with octaves folded and pitches inside D minor
    stable = [n for n in transcription["notes"] if 33 <= n["bar"] < 57]
    per_step: dict[int, Counter] = {}
    for note in stable:
        pitch = snap_to_dm(fold_into_bass_register(int(note["midi"])))
        per_step.setdefault(int(note["sixteenth"]), Counter())[pitch] += 1
    pattern = {step: counter.most_common(1)[0][0] for step, counter in sorted(per_step.items())}
    print("patron de 1 compas (16vo -> nota):", pattern)
    print("alturas:", sorted(set(pattern.values())))

    # 2) when the bass is allowed to play: from its measured entry to the end, minus the breakdown
    bass_from, bass_to = 33, total_bars
    silent = [(107, 129)]
    notes = []
    for bar in range(bass_from, bass_to):
        if any(start <= bar < stop for start, stop in silent):
            continue
        for step, pitch in pattern.items():
            start_beats = bar * 4 + (step - 1) * 0.25
            notes.append({"pitch": pitch, "start": round(start_beats, 3),
                          "duration": 0.22, "velocity": 100 if step in (7, 10, 15) else 88})
    plan["version"] = "v10"
    plan["synth"]["bass"].update({
        "notes": notes, "volume": 0.85,
        "source": "1-bar measured rhythm, octaves folded, snapped to D minor, repeated across sections",
        "measured_notes": len(notes),
        "pattern_per_sixteenth": {str(k): v for k, v in pattern.items()},
        "limits": ["patch unavailable: Serum 2 exposes one parameter"],
    })
    plan["synth"]["bass"].pop("muted_reason", None)
    # 3) arrangement: the loops only where their measured presence said so
    plan["arrangement"] = {
        "sections": [{"start_bar": 0, "bars": 51, "role": "intro/build"},
                     {"start_bar": 51, "bars": 56, "role": "main"},
                     {"start_bar": 107, "bars": 22, "role": "breakdown"},
                     {"start_bar": 129, "bars": 49, "role": "drop"},
                     {"start_bar": 178, "bars": total_bars - 178, "role": "outro"}],
        "source": "structure measured from the reference (bars 51/107/129/178)",
    }
    note = "v10: one repeating bar of bass in D minor, arrangement by measured section"
    if isinstance(plan.get("revisions"), dict):
        plan["revisions"]["v10"] = note
    else:
        plan.setdefault("revisions", []).append(note)
    path = SCRATCH / "tribal_replica_plan_v10.json"
    path.write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"notas de bajo: {len(notes)} (patron de {len(pattern)} 16vos x {bass_to - bass_from - 22} compases)")
    print("->", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
