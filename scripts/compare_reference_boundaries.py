"""Compare stored reference boundaries with local human marks, without audio or Live."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from copilot.audio.msaf_reference_boundaries import analyze_msaf_reference_boundaries
from copilot.audio.reference_boundary_comparison import compare_reference_boundaries
from copilot.schemas.music_analysis import MusicAnalysisPack


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", type=Path, required=True, help="persisted MusicAnalysisPack JSON")
    parser.add_argument(
        "--marks", type=Path, required=True,
        help="JSON object with marked_qn and optional alternatives_qn by provider",
    )
    parser.add_argument("--tolerance-qn", type=float, default=4.0)
    parser.add_argument("--msaf-python", type=Path, help="isolated local Python with MSAF installed")
    parser.add_argument("--scratch-root", type=Path, help="existing scratch directory for temporary audio")
    args = parser.parse_args()
    if (args.msaf_python is None) != (args.scratch_root is None):
        parser.error("--msaf-python and --scratch-root must be provided together")

    pack = MusicAnalysisPack.model_validate_json(args.pack.read_text(encoding="utf-8"))
    marks = json.loads(args.marks.read_text(encoding="utf-8"))
    if not isinstance(marks, dict) or not isinstance(marks.get("marked_qn"), list):
        raise ValueError("REFERENCE_HUMAN_MARKS_INVALID")
    alternatives = marks.get("alternatives_qn", {})
    if not isinstance(alternatives, dict) or any(
        not isinstance(name, str) or not isinstance(values, list)
        for name, values in alternatives.items()
    ):
        raise ValueError("REFERENCE_ALTERNATIVES_INVALID")
    if args.msaf_python is not None:
        if "msaf" in alternatives:
            raise ValueError("MSAF_BOUNDARIES_ALREADY_SUPPLIED")
        msaf = analyze_msaf_reference_boundaries(
            pack, python_executable=args.msaf_python, scratch_root=args.scratch_root,
        )
        alternatives["msaf"] = msaf["boundaries_qn"]
    report = compare_reference_boundaries(
        pack, marks["marked_qn"],
        alternatives_qn=alternatives,
        tolerance_qn=args.tolerance_qn,
    )
    if args.msaf_python is not None:
        report["provider_metadata"] = {"msaf": msaf}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
