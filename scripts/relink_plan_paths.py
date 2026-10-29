"""Re-point a plan's sample paths at THIS machine's copy of the packs.

Plans store absolute paths, and those paths are whoever built the plan's. A plan built on one
machine will not resolve on another, even with the same sample packs installed.

This looks every palette entry up by FILENAME in the local sample index and rewrites the path, so a
plan can travel between machines. It reports anything it cannot find instead of silently leaving a
broken path in place.

Usage:
  python scripts/relink_plan_paths.py --plan plans/v15_fx_music.json --write plans/v15_fx_music.json
"""
from __future__ import annotations
import json

from pathlib import Path

INDEX = Path.home() / "Library/Application Support/MusicPlatform/sample-library-index.json"


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    parser.add_argument("--write", default=None, help="write here; default is stdout only")
    argv = parser.parse_args()
    plan = json.loads(Path(argv.plan).read_text(encoding="utf-8"))
    if not INDEX.is_file():
        raise SystemExit(f"no sample index at {INDEX}")
    by_name = {}
    for asset in json.loads(INDEX.read_text(encoding="utf-8"))["assets"].values():
        by_name.setdefault(Path(asset["relative_path"]).name, asset["path"])

    palette = (plan.get("library") or {}).get("palette") or {}
    found, missing = 0, []
    for element, choice in palette.items():
        name = choice.get("filename") or Path(str(choice.get("path", ""))).name
        resolved = by_name.get(name)
        if resolved:
            if choice.get("path") != resolved:
                print(f"  {element:12s} {name[:44]} -> resuelto")
            choice["path"] = resolved
            found += 1
        else:
            missing.append((element, name))
    print(f"resueltos: {found} | faltantes: {len(missing)}")
    for element, name in missing:
        print(f"  FALTA en el indice local: {element} -> {name}")
    if argv.write:
        Path(argv.write).write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")
        print("->", argv.write)
    return 0 if not missing else 1


if __name__ == "__main__":
    raise SystemExit(main())
