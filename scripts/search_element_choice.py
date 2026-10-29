"""Choose an element by scoring the whole track with each candidate, not the loop in isolation.

The percussion experiment proved the point: a loop's own fingerprint did not predict how the
assembled track scored. So each candidate is rendered into the full plan and scored against the
reference, and the winner is the one that scores best in context.

Usage: python scripts/search_element_choice.py --plan <plan.json> --element bass --role BASS
"""
from __future__ import annotations
import argparse, json, re, subprocess, sys, tempfile

from pathlib import Path

SCRATCH = Path(__file__).resolve().parents[1] / "logs"
INDEX = Path.home() / "Library/Application Support/MusicPlatform/sample-library-index.json"
SEARCH_DIR = SCRATCH / "search"


def candidates_by_filename(role: str, key_pattern: str, bpm: int = 128) -> list[dict]:
    assets = list(json.loads(INDEX.read_text(encoding="utf-8"))["assets"].values())
    out = []
    for asset in assets:
        if str(asset.get("semantic_role")).upper() != role.upper():
            continue
        name = Path(asset["relative_path"]).name
        if str(bpm) not in name or not re.search(key_pattern, name.lower()):
            continue
        out.append({"path": asset["path"], "file": name})
    return out


def score_plan(plan: dict, tag: str) -> dict:
    SEARCH_DIR.mkdir(parents=True, exist_ok=True)
    plan_path = SEARCH_DIR / f"plan_{tag}.json"
    render_path = SEARCH_DIR / f"render_{tag}.wav"
    score_path = SEARCH_DIR / f"score_{tag}.json"
    plan_path.write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")
    subprocess.run([sys.executable, str(SCRATCH.parent / "scripts/render_plan_audio.py"),
                    "--plan", str(plan_path), "--out", str(render_path)],
                   check=True, capture_output=True)
    subprocess.run([sys.executable, str(SCRATCH.parent / "scripts/similarity_to_reference.py"),
                    "--candidate", str(render_path), "--json", str(score_path)],
                   check=True, capture_output=True)
    return json.loads(score_path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--element", required=True)
    parser.add_argument("--role", required=True)
    parser.add_argument("--key-pattern", default=r"dm|d min")
    args = parser.parse_args()

    base = json.loads(args.plan.read_text(encoding="utf-8"))
    candidates = candidates_by_filename(args.role, args.key_pattern)
    print(f"{args.element}: {len(candidates)} candidatos por etiqueta de archivo "
          f"(rol {args.role}, 128 BPM, key /{args.key_pattern}/)")

    results = []
    for candidate in candidates:
        plan = json.loads(json.dumps(base))
        plan["library"]["palette"][args.element] = {"path": candidate["path"], "bpm": 128.0}
        tag = re.sub(r"[^A-Za-z0-9]+", "_", Path(candidate["file"]).stem)[-40:]
        report = score_plan(plan, tag)
        results.append({"file": candidate["file"], "path": candidate["path"],
                        "total": report["total"], "dimensions": report["dimensions"]})
        print(f"  {report['total']:.4f}  {candidate['file'][:58]}")

    results.sort(key=lambda item: -item["total"])
    print()
    print("mejor candidato:", results[0]["file"], "->", results[0]["total"])
    out = SCRATCH / f"search_{args.element}.json"
    out.write_text(json.dumps({"element": args.element, "role": args.role,
                               "results": results}, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    print("->", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
