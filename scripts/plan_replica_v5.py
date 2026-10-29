"""v5: hand percussion replaced by the measured best groove match at 128 BPM.

The role-keyword selection was a dead end: with only 67 PERCUSSION assets carrying a usable
fingerprint, matching the word "conga" mostly returned SALSA loops at the wrong tempo (130 and
100 BPM). Name is not evidence. What the ranking does say is that the Fiesta 4 Percussion family
at 128 BPM sits closest to the reference's intra-beat accent distribution (0.07-0.13 distance
against 0.78-0.93 for the keyword picks).

So the four hand-percussion tracks take the four best-scoring loops at 128 BPM, keeping the
density gating from the stems. The trade-off is explicit: they are four variants of one family,
which risks sounding alike, and the density gating is what keeps them from stacking.

Usage: python scripts/plan_replica_v5.py
"""
from __future__ import annotations
import json

from pathlib import Path

SCRATCH = Path(__file__).resolve().parents[1] / "logs"
V4 = SCRATCH / "tribal_replica_plan_v4.json"
SELECTION = SCRATCH / "percussion_selection_v5.json"
OUT = SCRATCH / "tribal_replica_plan_v5.json"
TARGET_BPM = 128.0
SLOTS = ("conga", "djembe", "bongo", "shaker")
MAX_TEMPO_DRIFT = 0.5


def main() -> int:
    plan = json.loads(V4.read_text(encoding="utf-8"))
    selection = json.loads(SELECTION.read_text(encoding="utf-8"))
    plan["version"] = "5"

    pool = [row for row in selection["ranking"]
            if abs(float(row["bpm"]) - TARGET_BPM) <= MAX_TEMPO_DRIFT]
    if len(pool) < len(SLOTS):
        raise SystemExit(f"NOT_ENOUGH_MATCHES: {len(pool)} at {TARGET_BPM} BPM, need {len(SLOTS)}")

    changes = {}
    for slot, row in zip(SLOTS, pool[:len(SLOTS)]):
        previous = plan["library"]["palette"].get(slot, {})
        plan["library"]["palette"][slot] = {"path": row["path"], "bpm": row["bpm"],
                                           "groove_score": row["score"],
                                           "shape_distance": row["shape_distance"]}
        changes[slot] = {"from": Path(previous.get("path", "")).name if previous else None,
                         "to": row["file"], "score": row["score"]}

    plan["revisions"] = {"from": "v4", "changes": [
        "hand percussion re-selected by measured groove distance, all at 128 BPM",
        "role keywords discarded: they returned salsa loops at the wrong tempo"]}
    plan["selection_notes"] = {
        "candidates_with_valid_fingerprint": selection["candidates_valid"],
        "target_profile_16ths": selection["reference_profile"],
        "trade_off": "four variants of one family; density gating keeps them from stacking",
    }
    OUT.write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"plan v5 -> {OUT.name}")
    for slot, change in changes.items():
        print(f"  {slot:7s} {str(change['from'])[:40]:40s} -> {change['to'][:40]}  "
              f"(score {change['score']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
