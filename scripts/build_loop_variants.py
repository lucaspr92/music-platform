"""Three 8-bar loops: same drums, three bass ideas. One variable at a time.

The point is a decision, not a deliverable: Lukas hears three grooves and picks one, so the next
stage builds on a loop he approved instead of on a six-minute render he never endorsed.

Bar 33 is used because that is where the reference's bass enters (measured), so all three variants
are heard in the part of the track where the full groove exists.

  A  offbeat 8ths        the bass marks every "and" - the most common tech house figure
  B  rolling 16ths       continuous sixteenths, driving, more energy and less space
  C  syncopated figure   five notes placed off the grid, the "dirty" classic

Usage: python scripts/build_loop_variants.py
"""
from __future__ import annotations
import json

from pathlib import Path

import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
SCRATCH = ROOT / "logs"
OUT = SCRATCH / "loops"
BAR_S = 1.87486
START_BAR = 33
BARS = 8
SR = 44100

VARIANTS = {
    "A_offbeat8ths": [(3, 38), (7, 38), (11, 38), (15, 38)],
    "B_rolling16ths": [(step, 38) for step in range(1, 17)],
    "C_syncopated": [(1, 38), (3, 38), (6, 41), (11, 38), (14, 33)],
}


def main() -> int:
    plan = json.loads((SCRATCH / "tribal_replica_plan_v12.json").read_text(encoding="utf-8"))
    OUT.mkdir(parents=True, exist_ok=True)
    start = int(START_BAR * BAR_S * SR)
    length = int(BARS * BAR_S * SR)
    cores = {}
    for element, info in plan["core"].items():
        audio, sr = sf.read(info["wav"], always_2d=True, dtype="float32")
        mono = audio.mean(axis=1)[start:start + length]
        path = OUT / f"slice_{element}.wav"
        sf.write(str(path), mono, SR)
        cores[element] = {**info, "wav": str(path)}
    print("cores de 8 compases:", list(cores))

    for name, pattern in VARIANTS.items():
        notes = []
        for bar in range(BARS):
            for step, pitch in pattern:
                value = pitch
                if step == 15 and bar % 4 == 3:
                    value = 31          # the octave drop on the last "&" of the phrase
                notes.append({"pitch": value, "start": round(bar * 4 + (step - 1) * 0.25, 3),
                              "duration": 0.3 if name == "A_offbeat8ths" else 0.22,
                              "velocity": 100 if step in (3, 7, 11, 15) else 88})
        stabs = [{"pitch": pitch, "start": round(bar * 4 + 2.5, 3), "duration": 0.45, "velocity": 88}
                 for bar in range(0, BARS, 2) for pitch in (62, 65, 69)]
        variant = json.loads(json.dumps(plan))
        variant.update({"version": f"loop_{name}", "total_bars": BARS, "bar_s": BAR_S,
                        "core": cores, "library": {"palette": {}, "placements": {}}})
        variant["synth"]["bass"].update({"notes": notes, "volume": 0.5})
        variant["synth"]["stabs"].update({"notes": stabs, "volume": 0.4})
        path = OUT / f"plan_{name}.json"
        path.write_text(json.dumps(variant, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  {name:16s} bajo {len(notes):3d} notas | stabs {len(stabs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
