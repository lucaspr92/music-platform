"""v6: the stab layer becomes a library element, so it can be chosen by measurement.

The Serum stabs play the right onsets but sit on a patch whose state cannot be verified and
which cannot be shaped (the plugin exposes one parameter). A Dm synth loop at 128 BPM is a real,
curated sound, so it can be scored like any other element. The synth stabs stay in the plan at
zero volume for the A/B.

The stab windows are the ones measured on the 6s guitar bucket, which is what that layer is.

Usage: python scripts/plan_replica_v6.py
"""
from __future__ import annotations
import argparse, json

from pathlib import Path

SCRATCH = Path(__file__).resolve().parents[1] / "logs"
V4 = SCRATCH / "tribal_replica_plan_v4.json"
GUITAR = SCRATCH / "reference_guitar_line.json"
OUT = SCRATCH / "tribal_replica_plan_v6.json"


def runs_from_presence(sections: list[dict], total_bars: int) -> list[dict]:
    """A section's end is the next section's start: the stem line artifacts carry no `bars`,
    and deriving `start + bars` silently produced zero-length runs."""
    ordered = sorted(sections, key=lambda s: float(s["start_bar"]))
    starts = [int(float(s["start_bar"])) for s in ordered]
    runs: list[dict] = []
    for index, section in enumerate(ordered):
        if not section.get("present"):
            continue
        start = starts[index]
        end_raw = section.get("end_bar")
        if end_raw is not None:
            end = int(float(end_raw))
        elif index + 1 < len(starts):
            end = starts[index + 1]
        else:
            end = total_bars
        end = min(end, total_bars)
        if runs and runs[-1]["end_bar"] >= start - 1:
            runs[-1]["end_bar"] = max(runs[-1]["end_bar"], end)
        else:
            runs.append({"start_bar": start, "end_bar": end})
    return [r for r in runs if r["end_bar"] > r["start_bar"]]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default=None, help="stab loop to place (optional)")
    args = parser.parse_args()

    plan = json.loads(V4.read_text(encoding="utf-8"))
    guitar = json.loads(GUITAR.read_text(encoding="utf-8"))
    total_bars = int(plan["total_bars"])
    plan["version"] = "6"

    placements = runs_from_presence(guitar["sections"], total_bars)
    plan["library"]["placements"]["stabs_loop"] = placements
    if args.path:
        plan["library"]["palette"]["stabs_loop"] = {"path": args.path, "bpm": 128.0}
    plan["synth"]["stabs"]["volume"] = 0.0
    plan["synth"]["stabs"]["muted_reason"] = "replaced by a library stab loop; kept for A/B"
    plan["revisions"] = {"from": "v4", "changes": [
        "stab layer becomes a library element placed on the measured 6s guitar windows",
        "Serum stabs muted to 0 volume for A/B"]}
    OUT.write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"plan v6 -> {OUT.name}")
    print(f"  stabs_loop en {len(placements)} tramos: "
          f"{[(r['start_bar'], r['end_bar']) for r in placements]}")
    print(f"  synth stabs volumen={plan['synth']['stabs']['volume']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
