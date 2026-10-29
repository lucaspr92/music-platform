"""Build plan v3 in Live: library loops for percussion, Serum 2 for bass and stabs.

The two synth parts are MIDI written from measured onsets, and the preset is applied through
a browser URI (query:Plugins#...), which is the only form the remote script can resolve.

Usage: python scripts/build_replica_v3.py [--plan logs/tribal_replica_plan_v3.json]
"""
from __future__ import annotations
import json, sys, time

from pathlib import Path

from copilot.daw.ableton_tcp import AbletonTcpAdapter
from copilot.schemas.session import MidiNote

SCRATCH = Path(__file__).resolve().parents[1] / "logs"
PREFIX = "REPLICA - "
VOLUMES = {"kick": .75, "perc": .52, "hats": .46, "shaker": .40, "conga": .46,
           "djembe": .46, "bongo": .40, "bass": .70, "stabs": .45}


def to_notes(items: list[dict]) -> list[MidiNote]:
    return [MidiNote(pitch=int(n["pitch"]), start_time=round(float(n["start"]), 6),
                     duration=round(float(n["duration"]), 6), velocity=int(n["velocity"]))
            for n in items]


def main(plan_path: Path, append: bool = False) -> int:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    daw = AbletonTcpAdapter()
    daw.connect()
    started = time.perf_counter()
    report: dict = {"status": "RUNNING", "plan": str(plan_path), "version": plan.get("version"),
                    "tracks": {}, "notes_written": {}}
    try:
        total_beats = float(plan["total_bars"]) * 4.0
        daw.set_tempo(float(plan["bpm"]))

        snap = daw.snapshot(include_notes=False)
        existing = {track.name for track in snap.tracks}
        if append:
            # Append mode: never delete. A rebuild wipes manual edits and, if it fails halfway,
            # leaves the set in an in-doubt state that has already cost us one session.
            present = [name for name in existing if name.startswith(PREFIX)]
            print(f"  append: {len(present)} tracks REPLICA ya presentes; solo se agrega lo que falta")
        else:
            for track in sorted((t for t in snap.tracks if t.name.startswith(PREFIX)),
                                key=lambda t: t.index, reverse=True):
                daw._command("delete_track", {"track_index": track.index}, side_effect=True)

        palette, placements = plan["library"]["palette"], plan["library"]["placements"]
        for element, runs in placements.items():
            choice = palette.get(element)
            if not choice:
                continue
            if append and (PREFIX + element) in existing:
                print(f"  {element:10s} ya existe, se deja como esta")
                report["tracks"][element] = {"status": "kept"}
                continue
            index = int(daw.create_audio_track(PREFIX + element)["index"])
            clip = daw.create_audio_clip(index, 0, choice["path"])
            daw.set_clip_warping(index, 0, True)
            length_beats = float(clip.get("length") or 16.0)
            daw.set_clip_loop(index, 0, 0.0, length_beats, True)
            step = min(max(4.0, round(length_beats)), 32.0)
            placed = 0
            for run in runs:
                beat = run["start_bar"] * 4.0
                end_beat = (run["end_bar"] if run.get("end_bar") is not None
                            else plan["total_bars"]) * 4.0
                while beat < end_beat - 1e-6:
                    daw.duplicate_clip_to_arrangement(index, 0, float(beat))
                    placed += 1
                    beat += step
            daw.set_mixer_volume(index, float((plan.get("gains") or {}).get(element, VOLUMES.get(element, .55))))
            report["tracks"][element] = {"index": index, "clips": placed, "kind": "audio",
                                         "source": Path(choice["path"]).name}
            print(f"  {element:8s} track {index:>2}  {placed:>3} clips  {Path(choice['path']).name}")

        for element, info in (plan.get("core") or {}).items():
            if append and (PREFIX + element + " (core)") in existing:
                print(f"  core {element:6s} ya existe, se deja como esta")
                continue
            index = int(daw.create_audio_track(PREFIX + element + " (core)")["index"])
            daw.create_audio_clip(index, 0, info["wav"])
            daw.duplicate_clip_to_arrangement(index, 0, 0.0)
            daw.set_mixer_volume(index, float((plan.get("gains") or {}).get(f"core_{element}", VOLUMES.get(element, .6))))
            report["tracks"][f"core_{element}"] = {"index": index, "kind": "audio_core",
                                                   "source": Path(info["wav"]).name}
            print(f"  core    {element:8s} track {index:>2}  1 clip  {Path(info['wav']).name}")

        for name, spec in plan["synth"].items():
            if append and spec.get("track") in existing:
                print(f"  synth {name:5s} ya existe, se deja como esta")
                continue
            index = int(daw.create_midi_track(spec["track"])["index"])
            load = daw.load_instrument_or_effect(index, spec["device_uri"])
            device_name = load.get("item_name")
            preset = None
            if spec.get("preset_uri"):
                preset = daw.load_device_preset(index, 0, spec["preset_uri"])
            daw.create_midi_clip(index, 0, total_beats)
            notes = to_notes(spec["notes"])
            daw.replace_clip_notes(index, 0, notes)
            readback = daw.get_clip_notes(index, 0)
            written = readback.get("notes") or readback.get("items") or []
            daw.duplicate_clip_to_arrangement(index, 0, 0.0)
            # a synth part may carry an explicit volume, e.g. 0.0 to keep it available for an
            # A/B in Live without it being heard
            daw.set_mixer_volume(index, float(spec.get("volume", VOLUMES.get(name, .5))))
            report["tracks"][name] = {"index": index, "kind": "midi", "device": device_name,
                                      "preset": (preset or {}).get("preset_name"),
                                      "notes_sent": len(notes), "notes_readback": len(written)}
            report["notes_written"][name] = len(written)
            print(f"  {name:8s} track {index:>2}  {device_name}  preset={(preset or {}).get('preset_name')}  "
                  f"notas {len(notes)} enviadas / {len(written)} leidas")
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
        (SCRATCH / "replica_v3_build.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    path = Path(sys.argv[sys.argv.index("--plan") + 1]) if "--plan" in sys.argv else SCRATCH / "tribal_replica_plan_v3.json"
    append = "--append" in sys.argv
    raise SystemExit(main(path, append))
