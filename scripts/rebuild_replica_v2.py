"""Rebuild the replica in Live from plan v2 (the stem-upgraded arrangement map).

v2 states placements explicitly, so this is simpler than v1's run-merging: for every
element it deletes the existing REPLICA track from the highest index down and recreates it
from the plan. Deleting and recreating (rather than editing clips in place) is deliberate:
the arrangement is data, and a track rebuilt from the plan cannot drift from it.

Requires Live running with the AbletonMCP control surface loaded.

Usage: python scripts/rebuild_replica_v2.py [--plan logs/tribal_replica_plan_v2.json]
"""
from __future__ import annotations
import json, sys, time

from pathlib import Path

# Scratch artifacts stay in the gitignored logs/ dir, resolved from the repo root so
# the script runs from any working directory.
SCRATCH = Path(__file__).resolve().parents[1] / "logs"

from copilot.daw.ableton_tcp import AbletonTcpAdapter

VOLUMES = {"kick": .75, "perc": .52, "hats": .46, "shaker": .40, "conga": .46,
           "djembe": .46, "bongo": .40, "bass": .70, "hook": .50}
PREFIX = "REPLICA - "


def main(plan_path: Path) -> int:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    daw = AbletonTcpAdapter()
    daw.connect()
    report: dict = {"status": "RUNNING", "plan": str(plan_path), "version": plan.get("version"),
                    "elements": {}}
    started = time.perf_counter()
    try:
        daw.set_tempo(float(plan["bpm"]))
        snapshot = daw.snapshot(include_notes=False)
        # delete existing replica tracks, highest index first so indices stay valid
        existing = sorted((t for t in snapshot.tracks if t.name.startswith(PREFIX)),
                          key=lambda t: t.index, reverse=True)
        for track in existing:
            daw._command("delete_track", {"track_index": track.index}, side_effect=True)
        report["deleted_tracks"] = [t.name for t in existing]

        for element, runs in plan["placements"].items():
            choice = plan["palette"].get(element)
            if not choice:
                continue
            index = int(daw.create_audio_track(PREFIX + element)["index"])
            clip = daw.create_audio_clip(index, 0, choice["path"])
            daw.set_clip_warping(index, 0, True)
            length_beats = float(clip.get("length") or 16.0)
            daw.set_clip_loop(index, 0, 0.0, length_beats, True)
            step = min(max(4.0, round(length_beats)), 32.0)
            placed = []
            for run in runs:
                beat, end_beat = run["start_bar"] * 4.0, run["end_bar"] * 4.0
                while beat < end_beat - 1e-6:
                    daw.duplicate_clip_to_arrangement(index, 0, float(beat))
                    placed.append(round(beat, 2))
                    beat += step
            daw.set_mixer_volume(index, VOLUMES.get(element, .6))
            report["elements"][element] = {"track_index": index, "source": Path(choice["path"]).name,
                                           "runs": len(runs), "placed": len(placed),
                                           "volume": VOLUMES.get(element, .6)}
            print(f"  {element:7s} track {index:>2}  {len(runs):>2} tramos  {len(placed):>3} clips  "
                  f"{Path(choice['path']).name}")
        report["status"] = "APPLIED"
    except Exception as exc:
        report["status"] = "FAILED"
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report["seconds"] = round(time.perf_counter() - started, 1)
        try:
            daw.disconnect()
        except Exception:
            pass
        (SCRATCH / "replica_v2_build.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    path = Path(sys.argv[sys.argv.index("--plan") + 1]) if "--plan" in sys.argv else SCRATCH / "tribal_replica_plan_v2.json"
    raise SystemExit(main(path))
