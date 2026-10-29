"""Assemble the core from the measured winners, then master against the reference.

Every choice here comes from a scored search, never from a preference:
  logs/core_search_<role>.json holds the candidates and their in-context scores, and the winner
  is the first row after sorting. The 909 hat won its search, so it is kept and the report says so.

Stages: core -> plan v8 -> offline render -> matchering -> score. Both scores are printed so the
mastering contribution stays visible instead of silently folded into one number.

Usage: python scripts/build_core_final.py
"""
from __future__ import annotations
import json
import subprocess
import sys

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRATCH = ROOT / "logs"
PY = sys.executable
ROLES = ("kick", "hats", "clap")


def run(args: list[str]) -> str:
    result = subprocess.run(args, cwd=ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(f"FALLO {' '.join(args)}\n{result.stdout[-700:]}\n{result.stderr[-700:]}")
    return result.stdout


def score(candidate: Path, json_path: Path) -> dict:
    run([PY, "scripts/similarity_to_reference.py", "--candidate", str(candidate),
         "--json", str(json_path)])
    return json.loads(json_path.read_text(encoding="utf-8"))


def main() -> int:
    winners: dict[str, dict] = {}
    for role in ROLES:
        path = SCRATCH / f"core_search_{role}.json"
        if not path.is_file():
            raise SystemExit(f"falta {path}: corre primero search_core_oneshot.py --role {role}")
        rows = json.loads(path.read_text(encoding="utf-8"))
        winners[role] = rows[0]["option"]
        print(f"  {role:5s} ganador: {winners[role]['label'][:50]}  "
              f"(score {rows[0]['total']:.4f})")

    final = SCRATCH / "core_final"
    final.mkdir(parents=True, exist_ok=True)
    args = [PY, "scripts/render_percussion_core.py", "--out-dir", str(final)]
    for role in ROLES:
        args += [f"--{role}", winners[role]["source"]]
    run(args)

    plan = json.loads((SCRATCH / "tribal_replica_plan_v7.json").read_text(encoding="utf-8"))
    plan["version"] = "v8"
    for role in ROLES:
        plan["core"][role]["wav"] = str(final / f"core_{role}.wav")
        plan["core"][role]["source"] = winners[role]["label"]
        plan["core"][role]["chosen_in_context"] = True
    note = "v8: core one-shots chosen by measuring the whole track, not the sample in isolation"
    if isinstance(plan.get("revisions"), dict):
        plan["revisions"]["v8"] = note
    else:
        plan.setdefault("revisions", []).append(note)
    plan_path = SCRATCH / "tribal_replica_plan_v8.json"
    plan_path.write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")

    rendered = SCRATCH / "render_8.wav"
    run([PY, "scripts/render_plan_audio.py", "--plan", str(plan_path), "--out", str(rendered)])
    plain = score(rendered, SCRATCH / "similarity_v8.json")
    print()
    print("=== v8 sin master ===")
    for name, value in plain["dimensions"].items():
        print(f"  {name:16s} {value}")
    print(f"  {'TOTAL':16s} {plain['total']:.4f}")

    import matchering as mg
    matched = SCRATCH / "matched_8.wav"
    mg.process(target=rendered,
               reference=Path.home() / "music-platform-runtime/stems-src/reference_tudm_44100_2ch.wav",
               results=[mg.pcm24(str(matched))])
    mastered = score(matched, SCRATCH / "similarity_v8_mastered.json")
    print()
    print("=== v8 masterizado con matchering ===")
    for name, value in mastered["dimensions"].items():
        print(f"  {name:16s} {value}")
    print(f"  {'TOTAL':16s} {mastered['total']:.4f}")
    print()
    print(f"mastering: {plain['total']:.4f} -> {mastered['total']:.4f} "
          f"({mastered['total'] - plain['total']:+.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
