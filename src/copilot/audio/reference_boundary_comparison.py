"""Read-only comparison of reference structure against human-marked boundaries."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from copilot.schemas.music_analysis import MusicAnalysisPack


def _interior_boundaries(
    values: Sequence[float], *, start_qn: float, end_qn: float, source: str,
) -> list[float]:
    boundaries = [float(value) for value in values]
    if any(not math.isfinite(value) or not start_qn < value < end_qn for value in boundaries):
        raise ValueError(f"REFERENCE_BOUNDARY_OUTSIDE_REGION:{source}")
    if any(left >= right for left, right in zip(boundaries, boundaries[1:])):
        raise ValueError(f"REFERENCE_BOUNDARIES_NOT_INCREASING:{source}")
    return boundaries


def _compare(
    predicted: list[float], marked: list[float], tolerance_qn: float,
) -> dict[str, Any]:
    # Match monotonically: a detected boundary can explain at most one mark.
    # Maximize true positives first and minimize total timing error second.
    rows: list[list[tuple[int, float, tuple[tuple[float, float], ...]]]] = [
        [(0, 0.0, ()) for _ in range(len(marked) + 1)]
        for _ in range(len(predicted) + 1)
    ]
    for i in range(len(predicted) - 1, -1, -1):
        for j in range(len(marked) - 1, -1, -1):
            options = [rows[i + 1][j], rows[i][j + 1]]
            error = abs(predicted[i] - marked[j])
            if error <= tolerance_qn:
                count, total_error, pairs = rows[i + 1][j + 1]
                options.append((count + 1, total_error + error, ((predicted[i], marked[j]),) + pairs))
            rows[i][j] = max(options, key=lambda result: (result[0], -result[1]))
    count, total_error, pairs = rows[0][0]
    matched_predictions = {pair[0] for pair in pairs}
    matched_marks = {pair[1] for pair in pairs}
    return {
        "matched": [{"predicted_qn": pred, "marked_qn": mark, "error_qn": abs(pred - mark)}
                    for pred, mark in pairs],
        "true_positives": count,
        "false_positives": len(predicted) - count,
        "missed_marks": len(marked) - count,
        "precision": count / len(predicted) if predicted else None,
        "recall": count / len(marked) if marked else None,
        "mean_error_qn": total_error / count if count else None,
        "unmatched_predictions_qn": [value for value in predicted if value not in matched_predictions],
        "unmatched_marks_qn": [value for value in marked if value not in matched_marks],
    }


def compare_reference_boundaries(
    pack: MusicAnalysisPack,
    marked_qn: Sequence[float],
    *,
    alternatives_qn: Mapping[str, Sequence[float]] | None = None,
    tolerance_qn: float = 4.0,
) -> dict[str, Any]:
    """Compare change points in one analysis region without assigning quality.

    Inputs are interior boundaries in the pack's absolute quarter-note timeline.
    Alternatives may be produced by MSAF later; this function never runs a
    provider, decodes audio or treats fixed measurement windows as sections.
    """
    if not math.isfinite(tolerance_qn) or tolerance_qn <= 0:
        raise ValueError("REFERENCE_BOUNDARY_TOLERANCE_INVALID")
    if not pack.structural_regions:
        raise ValueError("REFERENCE_STRUCTURAL_REGIONS_MISSING")
    regions = sorted(pack.structural_regions, key=lambda region: region.start_beat)
    try:
        start_qn = float(pack.timeline.get("start_qn", regions[0].start_beat))
        end_qn = float(pack.timeline.get("end_qn", regions[-1].end_beat))
    except (TypeError, ValueError) as exc:
        raise ValueError("REFERENCE_REGION_INVALID") from exc
    if not math.isfinite(start_qn) or not math.isfinite(end_qn) or start_qn >= end_qn:
        raise ValueError("REFERENCE_REGION_INVALID")
    if regions[0].start_beat < start_qn - 1e-3 or regions[-1].end_beat > end_qn + 1e-3:
        raise ValueError("REFERENCE_STRUCTURAL_REGIONS_OUTSIDE_TIMELINE")
    if any(
        not math.isclose(left.end_beat, right.start_beat, abs_tol=1e-6)
        for left, right in zip(regions, regions[1:])
    ):
        raise ValueError("REFERENCE_STRUCTURAL_REGIONS_DISCONTINUOUS")
    marked = _interior_boundaries(
        marked_qn, start_qn=start_qn, end_qn=end_qn, source="HUMAN",
    )
    local = _interior_boundaries(
        [region.start_beat for region in regions[1:]],
        start_qn=start_qn, end_qn=end_qn, source="LOCAL_ANALYZER",
    )
    sources: dict[str, list[float]] = {"local_analyzer": local}
    for name, boundaries in (alternatives_qn or {}).items():
        if not name or name == "local_analyzer":
            raise ValueError("REFERENCE_BOUNDARY_SOURCE_INVALID")
        sources[name] = _interior_boundaries(
            boundaries, start_qn=start_qn, end_qn=end_qn, source=name,
        )
    return {
        "status": "BOUNDARIES_COMPARED",
        "no_write": True,
        "reference_state_token": pack.tokens.reference_state_token,
        "audio_sha256": pack.provenance.get("audio_sha256"),
        "analyzer_ids": pack.analyzer_ids,
        "region_qn": {"start": start_qn, "end": end_qn},
        "tolerance_qn": tolerance_qn,
        "human_boundaries_qn": marked,
        "providers": {
            name: {"boundaries_qn": boundaries, **_compare(boundaries, marked, tolerance_qn)}
            for name, boundaries in sources.items()
        },
        "limitations": [
            "Human marks are subjective section boundaries, not measurements of musical quality.",
            "Matching boundaries does not establish source identity, hook function or danceability.",
        ],
    }
