"""Fail-closed delivery gate; structural and audio facts are not artistic claims."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from copilot.producer.goal import ProducerGoal
from copilot.producer.track_spec import TrackSpec
from copilot.musicplan.score_compiler import ScoreBinding, verify_score_geometry


def section_window_specs(spec: TrackSpec) -> dict[str, dict[str, tuple[float, float]]]:
    """Three bounded independent windows including both sides of transitions."""
    cursor = 0.0
    windows = {}
    for section in spec.sections:
        duration = float(section.bars * 4)
        width = min(4.0, duration / 3.0)
        end = cursor + duration
        middle = cursor + (duration - width) / 2
        windows[section.name] = {
            "opening": (cursor, cursor + width),
            "middle": (middle, middle + width),
            "ending": (end - width, end),
        }
        cursor = end
    return windows


def verify_arrangement_timeline(
    spec: TrackSpec, *, tracks: dict[str, int], clips: list[dict[str, Any]],
    score_bindings: list[ScoreBinding] | None = None,
) -> list[str]:
    """Check the persisted clip geometry, not merely SafeWrite-created IDs."""
    if score_bindings is not None:
        expected_pairs = {
            (section.name, role) for section in spec.sections for role in section.active_roles
        }
        bound_pairs = {(binding.section, binding.role) for binding in score_bindings}
        if expected_pairs != bound_pairs:
            return ["SCORE_ARRANGEMENT_ROLES_NOT_BOUND"]
        if not score_bindings or not math.isclose(
            max(binding.start_qn + binding.duration_qn for binding in score_bindings),
            spec.duration_bars * 4, abs_tol=1e-3, rel_tol=0,
        ):
            return ["ARRANGEMENT_DURATION_MISMATCH"]
        return verify_score_geometry(score_bindings, tracks=tracks, clips=clips)
    reasons: list[str] = []
    parsed: list[tuple[int, float, float]] = []
    for clip in clips:
        try:
            index = clip["track_index"]
            start = float(clip["start_time"])
            length = float(clip["length"])
        except (KeyError, TypeError, ValueError):
            return ["ARRANGEMENT_CLIP_GEOMETRY_UNAVAILABLE"]
        if (
            not isinstance(index, int) or isinstance(index, bool)
            or not math.isfinite(start) or not math.isfinite(length)
            or start < 0 or length <= 0
        ):
            return ["ARRANGEMENT_CLIP_GEOMETRY_INVALID"]
        if "end_time" in clip:
            try:
                end_time = float(clip["end_time"])
            except (TypeError, ValueError):
                return ["ARRANGEMENT_CLIP_GEOMETRY_INVALID"]
            if not math.isfinite(end_time) or not math.isclose(
                end_time, start + length, abs_tol=1e-3,
            ):
                return ["ARRANGEMENT_CLIP_GEOMETRY_INVALID"]
        parsed.append((index, start, start + length))
    if not parsed:
        return ["ARRANGEMENT_CLIPS_MISSING"]
    total = spec.duration_bars * 4.0
    if not math.isclose(max(end for _, _, end in parsed), total, abs_tol=1e-3):
        reasons.append("ARRANGEMENT_DURATION_MISMATCH")
    cursor = 0.0
    for section in spec.sections:
        end = cursor + section.bars * 4.0
        for role in section.active_roles:
            index = tracks.get(role)
            intervals = sorted(
                (max(start, cursor), min(stop, end))
                for row_index, start, stop in parsed
                if row_index == index and start < end and stop > cursor
            )
            if not intervals:
                reasons.append(f"ARRANGEMENT_ROLE_MISSING:{section.name}:{role}")
                continue
            covered_until = cursor
            for start, stop in intervals:
                if start > covered_until + 1e-3:
                    break
                covered_until = max(covered_until, stop)
            if covered_until < end - 1e-3:
                reasons.append(
                    f"ARRANGEMENT_KICK_GAP:{section.name}" if role == "Kick"
                    else f"ARRANGEMENT_ROLE_GAP:{section.name}:{role}"
                )
        cursor = end
    return reasons


def evaluate_delivery(
    *,
    goal: ProducerGoal,
    spec: TrackSpec,
    project_path: Path,
    project_identity: str,
    reopened_identity: str | None,
    tempo_bpm: float,
    arrangement: list[dict[str, Any]],
    arrangement_geometry: list[str] | None = None,
    captures: dict[str, dict[str, Any]],
    role_captures: dict[str, dict[str, Any]] | None = None,
    actions: list[dict[str, Any]],
    critique_verdict: str | None,
    transport_stopped: bool,
    unresolved_transactions: list[str] | None = None,
) -> dict[str, Any]:
    """Demand independent readbacks for each section and durable delivery."""
    failures: list[str] = []
    try:
        goal.validate_track_spec(spec)
    except ValueError as exc:
        failures.append(str(exc))
    if not project_path.is_file() or project_path.suffix.lower() != ".als":
        failures.append("SAVED_ALS_NOT_VERIFIED")
    if not project_identity or reopened_identity != project_identity:
        failures.append("REOPENED_PROJECT_IDENTITY_NOT_VERIFIED")
    if not math.isclose(tempo_bpm, goal.bpm, abs_tol=1e-6):
        failures.append("LIVE_TEMPO_MISMATCH")
    expected = {
        (section.name, role)
        for section in spec.sections for role in section.active_roles
    }
    placed = {
        (str(row.get("section")), str(row.get("track")))
        for row in arrangement if row.get("status") == "VERIFIED"
    }
    if expected - placed:
        failures.append("ARRANGEMENT_SECTIONS_NOT_VERIFIED")
    if arrangement_geometry is None:
        failures.append("ARRANGEMENT_GEOMETRY_NOT_VERIFIED")
    else:
        failures.extend(arrangement_geometry)
    seen_ids: set[str] = set()
    for section in spec.sections:
        audio = captures.get(section.name)
        if not audio or not audio.get("capture_id") or not audio.get("path"):
            failures.append(f"SECTION_AUDIO_NOT_VERIFIED:{section.name}")
        elif not Path(str(audio["path"])).is_file() or not (
            float(audio.get("rms") or 0) > 0
            and 0 < float(audio.get("peak") or 0) < 1
        ):
            failures.append(f"SECTION_SILENT_OR_CLIPPING:{section.name}")
        for position, (start, end) in section_window_specs(spec)[section.name].items():
            window = (audio or {}).get("windows", {}).get(position)
            try:
                valid = bool(
                    window and window.get("capture_id")
                    and window["capture_id"] not in seen_ids
                    and Path(str(window["path"])).is_file()
                    and math.isclose(float(window["start_qn"]), start, abs_tol=1e-3)
                    and math.isclose(float(window["end_qn"]), end, abs_tol=1e-3)
                    and float(window["rms"]) > 0
                    and 0 < float(window["peak"]) < 1
                    and window.get("project_identity") == project_identity
                )
            except (KeyError, TypeError, ValueError, OverflowError):
                valid = False
            if not valid:
                failures.append(f"SECTION_WINDOW_NOT_VERIFIED:{section.name}:{position}")
            else:
                seen_ids.add(window["capture_id"])
    required_roles = {"Kick", "Bass", str(spec.hook_role or "")}
    for role in sorted(required_roles):
        audio = (role_captures or {}).get(role)
        if (
            not audio or audio.get("ok") is not True
            or audio.get("signal_status") != "HAS_SIGNAL"
            or audio.get("restore", {}).get("ok") is not True
            or not audio.get("audio_sha256")
            or not Path(str(audio.get("wav_path") or "")).is_file()
        ):
            failures.append(f"ROLE_SOURCE_AUDIO_NOT_VERIFIED:{role}")
    if not actions:
        failures.append("NO_VERIFIED_PRODUCTION_ACTIONS")
    elif any(row.get("status") != "VERIFIED" for row in actions):
        failures.append("ESSENTIAL_ACTION_UNVERIFIED")
    if any(row.get("experimental") is True for row in actions):
        failures.append("HERMES_MIDI_PHRASE_CERTIFICATION_PENDING")
    if unresolved_transactions:
        failures.append("UNRESOLVED_TRANSACTIONS")
    if critique_verdict != "finalize":
        failures.append("CRITIQUE_NOT_FINALIZED")
    if not transport_stopped:
        failures.append("TRANSPORT_NOT_STOPPED")
    return {
        "status": "COMPLETE" if not failures else "DRAFT",
        "reasons": list(dict.fromkeys(failures)),
        "expected_sections": len(spec.sections),
        "verified_sections": sum(
            section.name in captures and
            f"SECTION_AUDIO_NOT_VERIFIED:{section.name}" not in failures and
            f"SECTION_SILENT_OR_CLIPPING:{section.name}" not in failures and
            not any(
                item.startswith(f"SECTION_WINDOW_NOT_VERIFIED:{section.name}:")
                for item in failures
            )
            for section in spec.sections
        ),
        "quantization_seconds": goal.quantization_seconds,
        "artistic_quality_human_verified": False,
    }
