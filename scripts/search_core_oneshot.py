"""Pick the core's one-shot by measuring the whole track, not the sample in isolation.

The timbre ranking says which one-shot sounds closest, but a sample that wins on its own can lose
in context: it may fight the hats, sit wrong under the bass, or decay into the next hit. So each
candidate is rendered into the reference-timed core, substituted into the plan, scored, and only
the measured winner is kept.

Usage: python scripts/search_core_oneshot.py --role kick --top 3
"""
from __future__ import annotations
import json
import subprocess
import sys

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRATCH = ROOT / "logs"
PY = sys.executable
CURRENT = {"kick": "Samples/One Shots/Drums/Kick/Kick 909 1.aif",
           "hats": "Samples/One Shots/Drums/Hihat/Hihat Closed 909.aif",
           "clap": "Samples/One Shots/Drums/Clap/Clap 909.aif"}


def run(args: list[str]) -> str:
    result = subprocess.run(args, cwd=ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(f"FALLO {' '.join(args)}\n{result.stdout[-800:]}\n{result.stderr[-800:]}")
    return result.stdout


def score(plan_path: Path, out_dir: Path) -> dict:
    rendered = out_dir / "render.wav"
    run([PY, "scripts/render_plan_audio.py", "--plan", str(plan_path), "--out", str(rendered)])
    run([PY, "scripts/similarity_to_reference.py", "--candidate", str(rendered),
         "--json", str(out_dir / "similarity.json")])
    return json.loads((out_dir / "similarity.json").read_text(encoding="utf-8"))


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", required=True, choices=list(CURRENT))
    parser.add_argument("--top", type=int, default=3)
    parser.add_argument("--plan", default=str(SCRATCH / "tribal_replica_plan_v7.json"))
    args = parser.parse_args()
    base_plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    ranking = json.loads((SCRATCH / "oneshot_timbre_ranking.json").read_text(encoding="utf-8"))

    options = [{"label": "actual (909)", "source": CURRENT[args.role], "distance": None}]
    for row in ranking["rankings"][args.role][: args.top]:
        options.append({"label": row["file"], "source": row["path"],
                        "distance": row["timbre_distance"]})

    results = []
    for index, option in enumerate(options):
        name = f"{args.role}_{index}"
        out_dir = SCRATCH / "core_search" / name
        out_dir.mkdir(parents=True, exist_ok=True)
        run([PY, "scripts/render_percussion_core.py", f"--{args.role}", option["source"],
             "--out-dir", str(out_dir)])
        plan = json.loads(json.dumps(base_plan))
        plan["core"][args.role]["wav"] = str(out_dir / f"core_{args.role}.wav")
        plan["core"][args.role]["source"] = option["source"]
        plan_path = out_dir / "plan.json"
        plan_path.write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")
        report = score(plan_path, out_dir)
        total = float(report["total"])
        dimensions = report.get("dimensions", {})
        results.append({"option": option, "total": total,
                        "dimensions": {k: v for k, v in dimensions.items()}})
        print(f"  {total:.4f}  (timbre_dist={option['distance']})  {option['label'][:52]}")

    results.sort(key=lambda row: -row["total"])
    print()
    print(f"=== {args.role}: mejor en contexto ===")
    for row in results[:4]:
        print(f"  {row['total']:.4f}  {row['option']['label'][:56]}")
    (SCRATCH / f"core_search_{args.role}.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print()
    print("GANADOR:", results[0]["option"]["label"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
