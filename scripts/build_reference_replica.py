"""Finish the replica after the interrupted run: idempotent completion.

Reads what is ACTUALLY in the arrangement per REPLICA track, compares it against the
plan, and only adds the missing placements (and the missing tracks). Safe to re-run.
"""
from __future__ import annotations
import json, time
from pathlib import Path

# Scratch artifacts stay in the gitignored logs/ dir, resolved from the repo root so
# the script runs from any working directory.
SCRATCH = Path(__file__).resolve().parents[1] / "logs"


from copilot.daw.ableton_tcp import AbletonTcpAdapter

PLAN = (SCRATCH / "tribal_replica_plan.json")
REPORT = (SCRATCH / "tribal_replica_completion.json")
VOLUMES = {"kick": .75, "perc": .52, "hats": .46, "shaker": .40, "conga": .46,
           "djembe": .46, "bongo": .40, "bass": .70, "hook": .50}
PREFIX = "REPLICA - "


def runs_for(element: str, parts: list[dict]) -> list[list[float]]:
    runs: list[list[float]] = []
    for part in parts:
        if not part["elements"].get(element):
            continue
        if runs and abs(runs[-1][1] - part["start_bar"]) < 0.5:
            runs[-1][1] = part["end_bar"]
        else:
            runs.append([part["start_bar"], part["end_bar"]])
    return runs


def desired_placements(element: str, parts: list[dict], step: float) -> list[float]:
    out = []
    for start_bar, end_bar in runs_for(element, parts):
        beat, end_beat = start_bar * 4.0, end_bar * 4.0
        while beat < end_beat - 1e-6:
            out.append(round(beat, 2))
            beat += step
    return out


def main() -> int:
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    daw = AbletonTcpAdapter()
    daw.connect()
    report: dict = {"status": "RUNNING", "elements": {}}
    started = time.perf_counter()
    try:
        snapshot = daw.snapshot(include_notes=False)
        existing = {t.name: t.index for t in snapshot.tracks}
        arranged = daw.get_arrangement_clips().get("clips") or []
        have: dict[int, set] = {}
        for clip in arranged:
            have.setdefault(int(clip.get("track_index")), set()).add(round(float(clip.get("start_time")), 2))
        for element, choice in plan["palette"].items():
            name = PREFIX + element
            row: dict = {"source": Path(choice["path"]).name}
            if name in existing:
                index = existing[name]
                row["reused_track"] = index
                length_beats = 16.0
                for clip in arranged:
                    if int(clip.get("track_index")) == index:
                        span = float(clip["end_time"]) - float(clip["start_time"])
                        length_beats = max(4.0, min(round(span), 32.0))
                        break
            else:
                index = int(daw.create_audio_track(name)["index"])
                row["created_track"] = index
                clip = daw.create_audio_clip(index, 0, choice["path"])
                daw.set_clip_warping(index, 0, True)
                daw.set_clip_loop(index, 0, 0.0, float(clip.get("length") or 16.0), True)
                daw.set_mixer_volume(index, VOLUMES.get(element, .6))
                length_beats = max(4.0, min(round(float(clip.get("length") or 16.0)), 32.0))
            want = desired_placements(element, plan["parts"], length_beats)
            already = have.get(index, set())
            covered = len(already.intersection(set(want)))
            missing = [beat for beat in want if beat not in already]
            for beat in missing:
                daw.duplicate_clip_to_arrangement(index, 0, float(beat))
            row.update(track_index=index, wanted=len(want), had=covered,
                       added=len(missing), volume=VOLUMES.get(element, .6))
            report["elements"][element] = row
            print(f"  {element:7s} track {index:>2}  ya tenia {covered:>3}/{len(want):<3} -> agregue {len(missing)}")
        report["status"] = "COMPLETED"
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
        REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
