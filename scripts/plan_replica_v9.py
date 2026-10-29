"""v9: the bass becomes the transcribed part, not a library loop.

Measured reason: the bass was the weakest element of the replica (0.5258, pattern 0.404) because a
126 BPM library loop warped to 128 is somebody else's line. The reference's bass stem transcribed
to 885 notes (6.02 per bar, every 16th occupied), and those notes now drive Serum 2 directly.

Honest limits, recorded in the plan itself:
  * the PART is exact by transcription; the PATCH is not available (the plugin exposes one
    parameter through the bridge), so the audio will still differ in timbre
  * pyin can jump an octave; the transcription is kept raw rather than "corrected" by taste

Usage: python scripts/plan_replica_v9.py
"""
from __future__ import annotations
import json

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRATCH = ROOT / "logs"


def main() -> int:
    plan = json.loads((SCRATCH / "tribal_replica_plan_v8.json").read_text(encoding="utf-8"))
    transcription = json.loads((SCRATCH / "reference_bass_notes.json").read_text(encoding="utf-8"))
    bpm = float(plan["bpm"])
    beat_s = 60.0 / bpm
    notes = []
    for note in transcription["notes"]:
        confidence = float(note.get("confidence") or 0.5)
        notes.append({
            "pitch": int(note["midi"]),
            "start": round(float(note["start_s"]) / beat_s, 4),
            "duration": round(max(0.05, float(note["duration_s"])) / beat_s, 4),
            "velocity": int(round(min(127, max(55, 60 + 70 * confidence)))),
        })
    plan["version"] = "v9"
    plan["synth"]["bass"].update({
        "notes": notes,
        "volume": 0.9,
        "source": "transcribed from the reference bass stem (pyin, guards: silence + onset split)",
        "measured_notes": len(notes),
        "limits": ["part exact by transcription", "patch unavailable: Serum 2 exposes one parameter",
                   "pyin octave jumps kept, not corrected by taste"],
    })
    plan["synth"]["bass"].pop("muted_reason", None)
    # the loop must go, or it doubles the part
    removed = {}
    for container, key in ((plan["library"]["placements"], "bass"),
                           (plan["library"]["palette"], "bass"),
                           (plan.get("gains") or {}, "bass")):
        if key in container:
            removed[key] = container.pop(key)
    plan["removed_elements"] = {"bass_loop": "replaced by the transcribed part in v9"}
    note = "v9: bass loop removed, transcribed part drives Serum 2"
    if isinstance(plan.get("revisions"), dict):
        plan["revisions"]["v9"] = note
    else:
        plan.setdefault("revisions", []).append(note)
    path = SCRATCH / "tribal_replica_plan_v9.json"
    path.write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"notas de bajo: {len(notes)} | volumen {plan['synth']['bass']['volume']}")
    print("rango midi:", min(n["pitch"] for n in notes), "-", max(n["pitch"] for n in notes))
    print("duracion mediana (beats):", round(float(sorted(n['duration'] for n in notes)[len(notes)//2]), 3))
    print("loop de bajo eliminado:", bool(removed))
    print("->", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
