"""TECH_HOUSE_PRODUCTION_KIT_V1 — build the 16-bar groove in real Live.

Order: new empty 126 BPM working copy -> one compound MusicPlan
(CREATE_TRACK + preset DEVICE_LOAD + CREATE_PATTERN + DUPLICATE per role) ->
per-drum SAMPLE_LOAD -> SET_TRACK_VOLUME gain staging -> authoritative readback
-> Master capture through the verified V5 stereo path -> report.

Every musical write goes through ProductionCompiler -> SafeWrite. Presets play
as shipped: no processing is added unless a technical problem requires it.
Nothing here judges how the result sounds; that is HUMAN_AUDITION.
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from copilot.daw.state_tokens import attach_tokens
from copilot.importing.fast_lab_sound_once import _apply, _plan
from copilot.importing.new_project_v1 import _empty_template, open_new_project, prepare_new_project
from copilot.importing.working_copy_manager_v1 import create_working_copy, is_copilot_working_copy
from copilot.musicplan import (
    build_create_track_action,
    build_device_load_action,
    build_duplicate_clip_to_arrangement_action,
    build_pattern_action,
    build_sample_load_action,
    build_set_track_volume_action,
)
from copilot.musicplan.full_groove_lab_v1 import _same_notes
from copilot.platform.runtime_paths import ableton_core_library, runtime_root
from copilot.producer import tech_house_kit_v1 as kit
from copilot.runtime.production_compiler import ProductionCompiler
from copilot.runtime.safe_write import build_safe_write_executor
from copilot.schemas.musicplan import DiagnosisBinding, MusicPlan, PlanIntentClass, PlanStatus, VolumeOperation
from copilot.schemas.session import MidiNote, TrackState

DEFAULT_OUT = runtime_root() / "production-kit" / "tech-house-v1"
DEFAULT_TEMPLATE = runtime_root() / "templates" / "empty-126bpm" / "empty-126bpm.als"
DEFAULT_CORE_LIBRARY = ableton_core_library()
WAV_NAME = "tech_house_production_kit_v1.wav"
SILENCE_RMS = 1e-5
CLIP_PEAK = 1.0
# Starting gain stage (Live normalized fader, 0.85 = unity). Chosen for
# headroom on a seven-track sum with no limiter; not a mix decision.
GAIN_STAGE = {
    "Kick": 0.75, "Clap": 0.66, "Closed Hat": 0.55, "Open Hat": 0.55,
    "Perc": 0.52, "Bass": 0.70, "Stab": 0.58,
}
HEADROOM_STEP = 0.06  # one uniform reduction if the Master clips


def _slug(name: str) -> str:
    stem, dot, ext = name.rpartition(".")
    safe = "".join(ch.lower() if ch.isalnum() else "_" for ch in stem).strip("_")
    return f"kit_v1_{safe}.{ext.lower()}"


def stage_one_shots(definition: dict[str, Any], working_root: Path) -> dict[str, dict[str, str]]:
    """Copy drum one-shots into the project with a digest check. Never overwrite."""
    folder = working_root / "Samples" / "KIT_V1"
    folder.mkdir(parents=True, exist_ok=True)
    staged: dict[str, dict[str, str]] = {}
    for row in definition["sounds"]:
        if row["source_type"] != "one_shot_sample":
            continue
        dest = folder / _slug(row["browser_name"])
        if dest.exists():
            raise FileExistsError(f"KIT_SAMPLE_ALREADY_STAGED: {dest}")
        shutil.copy2(row["path"], dest)
        if kit.sha256_file(dest) != row["sha256"]:
            dest.unlink()
            raise ValueError(f"KIT_SAMPLE_DIGEST_MISMATCH: {row['role']}")
        staged[row["role"]] = {"path": str(dest), "uri": f"Samples/KIT_V1/{dest.name}", "sha256": row["sha256"]}
    return staged


def _preset_uri(daw, browser_name: str) -> str:
    stem = browser_name.rsplit(".", 1)[0]
    found = daw.search_browser(stem, "all")
    rows = [row for row in (found.get("items") or found.get("results") or [])
            if row.get("name") == browser_name and row.get("is_loadable")]
    if len(rows) != 1:
        raise RuntimeError(f"PRESET_NOT_UNAMBIGUOUS: {browser_name} matches={len(rows)}")
    return str(rows[0]["uri"])


def _music_plan(session, actions: list, plan_id: str, note: str) -> MusicPlan:
    return MusicPlan(
        plan_id=plan_id, status=PlanStatus.READY_FOR_EXECUTION,
        intent_class=PlanIntentClass.CONTROLLED_ENGINEERING_VALIDATION,
        diagnosis=DiagnosisBinding(diagnosis_id=f"{kit.MILESTONE}:groove-plan",
                                   diagnosis_status="STRUCTURED_GROOVE_PLAN", diagnosis_accepted=True),
        project_state_token=session.project_token or "", audible_state_token=session.audible_token or "",
        evidence_refs=[kit.MILESTONE], actions=actions, notes=[note],
        created_at=datetime.now(timezone.utc).isoformat(),
    )


def channel_metrics(path: Path) -> dict[str, Any]:
    audio, rate = sf.read(str(path), always_2d=True, dtype="float64")
    out: dict[str, Any] = {"path": str(path), "sha256": kit.sha256_file(path), "channels": int(audio.shape[1]),
                           "sample_rate": int(rate), "duration_s": len(audio) / rate}
    if audio.shape[1] != 2:
        return out
    left, right = audio[:, 0], audio[:, 1]
    rms = lambda channel: float(np.sqrt(np.mean(channel * channel)))
    peak = float(np.max(np.abs(audio)))
    out.update({
        "left_rms": rms(left), "right_rms": rms(right),
        "left_peak": float(np.max(np.abs(left))), "right_peak": float(np.max(np.abs(right))),
        "peak": peak, "peak_dbfs": (20 * np.log10(peak) if peak > 0 else None),
        "left_nonzero_fraction": float(np.mean(left != 0)), "right_nonzero_fraction": float(np.mean(right != 0)),
        "lr_correlation": float(np.corrcoef(left, right)[0, 1]) if rms(left) > 0 and rms(right) > 0 else None,
        "lr_bit_identical": bool(np.array_equal(left, right)),
    })
    try:
        import pyloudnorm

        out["lufs_integrated"] = float(pyloudnorm.Meter(rate).integrated_loudness(audio))
    except Exception as exc:  # noqa: BLE001 — loudness is informative only
        out["lufs_integrated"] = None
        out["lufs_error"] = str(exc)[:160]
    return out


def technical_audio_ok(metrics: dict[str, Any]) -> dict[str, bool]:
    """Technical gates only (channels, signal, clipping). Never a taste verdict."""
    stereo = metrics.get("channels") == 2
    return {
        "stereo": stereo,
        "left_signal": stereo and metrics["left_rms"] > SILENCE_RMS and metrics["left_nonzero_fraction"] > 0,
        "right_signal": stereo and metrics["right_rms"] > SILENCE_RMS and metrics["right_nonzero_fraction"] > 0,
        "not_bit_identical": stereo and metrics["lr_bit_identical"] is False,
        "no_clipping": stereo and metrics["peak"] < CLIP_PEAK,
    }


def _capture_master(daw, *, start_beat: float, end_beat: float, tempo: float, revision: int) -> dict[str, Any]:
    from copilot.audio.batch_capture import capture_parallel_pass
    from copilot.audio.live3r_trust import _recorders
    from copilot.schemas.observation import SignalPoint

    recorder = _recorders(kick_index=-1, bass_index=-1, kick_id="unused", bass_id="unused",
                          kick_channel="unused", bass_channel="unused",
                          main_point=SignalPoint.MAIN_FINAL)[:1]
    one = capture_parallel_pass(daw, start_beat=start_beat, end_beat=end_beat, fire_tracks=[],
                                recorders=recorder, tempo=tempo, session_revision=revision,
                                transport="arrangement")
    asset = one["assets"]["master"]
    final = Path(asset.analysis_file_path or asset.file_path)
    raw = Path(str(getattr(asset, "raw_file_path", "") or final.with_name(f"{final.stem}_raw.wav")))
    return {"final": final, "raw": raw, "timings": one.get("timings") or {}}


def readback(daw, executor, plan: dict[str, Any], *, preset_names: dict[str, str] | None = None) -> dict[str, Any]:
    session = executor.tools.get_session_snapshot()
    attach_tokens(session)
    arrangement = daw.get_arrangement_clips().get("clips") or []
    tracks: dict[str, Any] = {}
    ok = True
    for item in kit.KIT:
        track = session.track_by_name(item.track_name)
        if track is None:
            tracks[item.role] = {"exists": False}
            ok = False
            continue
        expected = [MidiNote(**row) for row in plan["roles"][item.role]["notes"]]
        observed = daw.get_clip_notes(track.index, 0)
        exact = _same_notes(observed.get("notes") or [], expected)
        placed = [clip for clip in arrangement if clip.get("track_index") == track.index]
        devices = [{"name": device.name, "class": device.class_name} for device in track.devices]
        if item.is_one_shot:
            device_ok = any(device["class"] in {"OriginalSimpler", "Simpler"} or "Simpler" in device["name"]
                            for device in devices) and len(devices) == 1
        else:
            expected_name = (preset_names or {}).get(item.role, item.browser_name).rsplit(".", 1)[0]
            device_ok = len(devices) == 1 and devices[0]["name"] == expected_name
        placement_ok = bool(placed) and min(float(c["start_time"]) for c in placed) == 0.0 and \
            max(float(c["end_time"]) for c in placed) >= kit.CLIP_BEATS
        row = {
            "exists": True, "index": track.index, "role": track.role,
            "devices": devices, "device_ok": device_ok,
            "session_clip_length": observed.get("length"),
            "requested_notes": len(expected), "readback_notes": len(observed.get("notes") or []),
            "notes_exact": exact,
            "arrangement_clips": [{"start": c.get("start_time"), "end": c.get("end_time")} for c in placed],
            "arrangement_ok": placement_ok,
            "volume": track.mixer.volume, "mute": track.mixer.mute, "solo": track.mixer.solo,
        }
        ok = ok and exact and device_ok and placement_ok and observed.get("length") == kit.CLIP_BEATS
        tracks[item.role] = row
    return {
        "ok": ok and session.transport.tempo == kit.TEMPO_BPM,
        "tempo": session.transport.tempo,
        "meter": [session.transport.signature_numerator, session.transport.signature_denominator],
        "project_identity": session.project_identity,
        "project_path": session.project_path,
        "track_count": len(session.tracks),
        "tracks": tracks,
    }


def run(*, template: Path, out: Path, core_library: Path, stab_fallback: bool = False) -> dict[str, Any]:
    from copilot.audio.capture_journal_recovery import unresolved_capture_journals
    from copilot.audio.live_capture import ensure_master_tap
    from copilot.audio.tap_trust import inventory_taps
    from copilot.daw.ableton_tcp import AbletonTcpAdapter
    from copilot.runtime.environment_autonomy_v1 import ensure_ableton_ready

    for sub in ("plan", "evidence", "audio", "report", "workspace"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    if (out / "report" / "tech_house_production_kit_v1.json").exists():
        raise FileExistsError("KIT_REPORT_ALREADY_EXISTS")
    report: dict[str, Any] = {"milestone": kit.MILESTONE, "started_at": datetime.now(timezone.utc).isoformat(),
                              "musical_writes_path": "MusicPlan -> ProductionCompiler -> SafeWrite",
                              "sidechain": "BLOCKED_BY_CONTROL_SURFACE", "automation": "DEFERRED",
                              "semantic_controls": "DEFERRED", "processing": [],
                              "human_audition": "PENDING"}

    definition = kit.kit_definition(core_library)
    plan = kit.groove_plan()
    (out / "plan" / "kit_definition.json").write_text(json.dumps(definition, indent=2), encoding="utf-8")
    (out / "plan" / "groove_plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")

    workspace = out / ("workspace-stab-fallback" if stab_fallback else "workspace")
    if _empty_template(template):
        copy = prepare_new_project(template_als=template, workspace=workspace)
        if copy.get("status") != "CREATED":
            raise RuntimeError(f"NEW_PROJECT_BLOCKED: {copy}")
        opened = open_new_project(copy)
        expected_baseline_tracks = 0
    else:
        # Live 12 rejects a zero-player-track ALS. A saved default set with
        # four untouched placeholder tracks is the smallest valid template.
        copy = create_working_copy(source_als=template, project_root=template.parent,
                                   copy_scope="project_directory", workspace=workspace)
        if copy.get("status") not in {"CREATED", "REUSED"}:
            raise RuntimeError(f"DEFAULT_TEMPLATE_COPY_BLOCKED: {copy}")
        ready = ensure_ableton_ready(working_als=copy["working_als"])
        session_ready = ready.get("session") or {}
        if ready.get("status") != "PROJECT_READY" or session_ready.get("track_count") != 4:
            raise RuntimeError(f"DEFAULT_TEMPLATE_NOT_READY: {ready.get('status')} {ready.get('reason')} {session_ready}")
        opened = {"status": "OPENED_DEFAULT_4", "working_als": copy["working_als"],
                  "project_identity": session_ready.get("project_identity"), "readiness": ready}
        expected_baseline_tracks = 4
    report["project_open"] = {k: v for k, v in opened.items() if k != "readiness"}
    if opened.get("status") not in {"OPENED_EMPTY", "OPENED_DEFAULT_4"}:
        raise RuntimeError(f"NEW_PROJECT_NOT_OPENED: {opened}")
    working_als = Path(copy["working_als"])
    if not is_copilot_working_copy(working_als):
        raise RuntimeError("WORKING_COPY_REQUIRED")
    staged = stage_one_shots(definition, Path(copy["working_root"]))
    report["staged_samples"] = staged
    preset_names = {"Stab": kit.sound("Stab").fallback_browser_name} if stab_fallback else {}
    report["technical_fallbacks"] = ({"Stab": {"from": kit.sound("Stab").browser_name,
                                                   "to": preset_names["Stab"],
                                                   "reason": "FIRST_PRESET_LOAD_TIMEOUT"}}
                                     if stab_fallback else {})

    daw = AbletonTcpAdapter()
    daw.connect()
    try:
        journal_root = out / "evidence" / "safe_write"
        executor = build_safe_write_executor(daw, journal_path=journal_root / "safe-write.jsonl",
                                             persist_dir=journal_root / "prestate")
        unresolved = [row for row in executor.recover() if row["recovery"] != "VERIFIED"]
        if unresolved:
            raise RuntimeError(f"UNRESOLVED_SAFEWRITE_JOURNAL: {unresolved}")
        session = executor.tools.get_session_snapshot()
        attach_tokens(session)
        if session.transport.tempo != kit.TEMPO_BPM or session.transport.signature_numerator != 4 \
                or session.transport.signature_denominator != 4:
            raise RuntimeError(f"TEMPO_OR_METER_MISMATCH: {session.transport.tempo}")
        if len(session.tracks) != expected_baseline_tracks:
            raise RuntimeError("PROJECT_BASELINE_TRACK_COUNT_MISMATCH")
        if expected_baseline_tracks:
            expected = [("1-MIDI", "midi"), ("2-MIDI", "midi"),
                        ("3-Audio", "audio"), ("4-Audio", "audio")]
            observed = [(track.name, track.role) for track in session.tracks]
            if observed != expected or any(track.clips or track.devices for track in session.tracks):
                raise RuntimeError(f"DEFAULT_TEMPLATE_NOT_PRISTINE: {observed}")
        report["baseline_track_count"] = expected_baseline_tracks
        pid = session.project_identity or ""

        # 1. One compound intent: tracks, presets, patterns, placement.
        actions = []
        preset_uris: dict[str, str] = {}
        for item in kit.KIT:
            virtual = TrackState(stable_id="", index=-1, name=item.track_name, role="midi")
            actions.append(build_create_track_action(project_identity=pid, track_name=item.track_name,
                                                     track_kind="midi", reason=f"{kit.MILESTONE} {item.role}",
                                                     evidence_refs=[kit.MILESTONE]))
            if not item.is_one_shot:
                selected_name = preset_names.get(item.role, item.browser_name)
                preset_uris[item.role] = _preset_uri(daw, selected_name)
                actions.append(build_device_load_action(track=virtual, project_identity=pid,
                                                        device_name=selected_name.rsplit(".", 1)[0],
                                                        device_uri=preset_uris[item.role],
                                                        reason=f"Core Library preset for {item.role}",
                                                        evidence_refs=[kit.MILESTONE]))
            notes = [MidiNote(**row) for row in plan["roles"][item.role]["notes"]]
            actions.append(build_pattern_action(track=virtual, project_identity=pid, clip_index=0,
                                                length_beats=kit.CLIP_BEATS, notes=notes,
                                                reason=f"16-bar {item.role} part", evidence_refs=[kit.MILESTONE]))
            actions.append(build_duplicate_clip_to_arrangement_action(track=virtual, project_identity=pid,
                                                                       clip_index=0, destination_time=0.0,
                                                                       length=None,
                                                                       reason=f"Place {item.role} at bar 1",
                                                                       evidence_refs=[kit.MILESTONE]))
        report["preset_uris"] = preset_uris
        compound = _music_plan(session, actions, f"tech_house_kit_v1_{session.revision}",
                               "Seven-role 16-bar groove; no musical quality claim.")
        compiled = ProductionCompiler().compile(compound, session=session)
        if compiled.status != "COMPILED" or compiled.intent is None:
            raise RuntimeError(f"COMPOUND_COMPILE_FAILED: {compiled.reasons}")
        (out / "plan" / "musicplan_compound.json").write_text(compound.model_dump_json(indent=2), encoding="utf-8")
        result = executor.run(compiled.intent)
        report["compound"] = {"ok": result.ok, "musical_writes": result.musical_writes,
                              "certified_actions": len(compiled.certified_action_ids),
                              "failure": str(result.failure) if result.failure else None, "error": result.error}
        if not result.ok:
            raise RuntimeError(f"COMPOUND_SAFEWRITE_FAILED: {report['compound']}")

        # 2. Drum one-shots into Simpler, one certified intent each.
        report["sample_loads"] = {}
        for item in kit.KIT:
            if not item.is_one_shot:
                continue
            session = executor.tools.get_session_snapshot()
            attach_tokens(session)
            track = session.track_by_name(item.track_name)
            action = build_sample_load_action(track=track, project_identity=session.project_identity or "",
                                              clip_index=0, sample_uri=staged[item.role]["uri"],
                                              reason=f"Load {item.browser_name}", evidence_refs=[kit.MILESTONE],
                                              session_incarnation_id=session.session_incarnation_id or "")
            report["sample_loads"][item.role] = _apply(
                executor, _plan(session, [action], kit.MILESTONE, f"kit_{item.role.replace(' ', '_')}_sample",
                                target_track=track), session)

        # 3. Gain staging, then capture; one uniform headroom step if it clips.
        def set_volumes(offset: float) -> dict[str, Any]:
            applied = {}
            for item in kit.KIT:
                session_now = executor.tools.get_session_snapshot()
                attach_tokens(session_now)
                track = session_now.track_by_name(item.track_name)
                target = round(max(0.0, GAIN_STAGE[item.role] - offset), 4)
                action = build_set_track_volume_action(
                    track=track, project_identity=session_now.project_identity or "",
                    operation=VolumeOperation.SET, expected_before=float(track.mixer.volume),
                    target_value=target, reason="Gain staging for headroom (technical, not a mix decision)",
                    evidence_refs=[kit.MILESTONE], session_incarnation_id=session_now.session_incarnation_id or "")
                applied[item.role] = {"target": target, **_apply(
                    executor, _plan(session_now, [action], kit.MILESTONE,
                                    f"kit_{item.role.replace(' ', '_')}_vol", target_track=track), session_now)}
            return applied

        report["gain_stage"] = {"pass_1": set_volumes(0.0)}
        ensure_master_tap(daw)
        session = executor.tools.get_session_snapshot()
        attach_tokens(session)
        attempts = []
        capture = None
        metrics = None
        for attempt, offset in enumerate((0.0, HEADROOM_STEP), start=1):
            if attempt == 2:
                report["gain_stage"]["pass_2"] = set_volumes(offset)
                session = executor.tools.get_session_snapshot()
                attach_tokens(session)
            capture = _capture_master(daw, start_beat=0.0, end_beat=kit.CLIP_BEATS,
                                      tempo=float(session.transport.tempo), revision=int(session.revision))
            metrics = {"raw": channel_metrics(capture["raw"]), "final": channel_metrics(capture["final"])}
            attempts.append({"attempt": attempt, "headroom_offset": offset,
                             "final_peak": metrics["final"].get("peak")})
            if technical_audio_ok(metrics["final"])["no_clipping"]:
                break
        report["capture_attempts"] = attempts

        # 4. Authoritative readback after everything is written.
        rb = readback(daw, executor, plan, preset_names=preset_names)
        (out / "evidence" / "ableton_readback.json").write_text(json.dumps(rb, indent=2, default=str),
                                                                encoding="utf-8")
        report["readback_ok"] = rb["ok"]

        # 5. Terminal state.
        after = daw.snapshot(include_notes=False)
        report["terminal"] = {
            "transport_playing": after.transport.playing,
            "taps": [{"track": r.get("track_name"), "slot": r.get("slot"), "rec": r.get("rec")}
                     for r in inventory_taps(daw)],
            "open_capture_journals": len(unresolved_capture_journals()),
        }
    finally:
        daw.disconnect()

    wav = out / "audio" / WAV_NAME
    shutil.copy2(capture["final"], wav)
    shutil.copy2(capture["raw"], out / "audio" / "tech_house_production_kit_v1_raw.wav")
    metrics["delivered"] = channel_metrics(wav)
    gates = technical_audio_ok(metrics["delivered"])
    metrics["technical_gates"] = gates
    metrics["note"] = "Peak/RMS/LUFS detect technical problems only; they are not a quality judgment."
    (out / "evidence" / "capture_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    terminal = report["terminal"]
    terminal_ok = (terminal["transport_playing"] is False and terminal["open_capture_journals"] == 0
                   and all(float(t["rec"] or 0) == 0 for t in terminal["taps"]))
    report["technical"] = "VERIFIED" if (report["compound"]["ok"] and report["readback_ok"]
                                          and all(gates.values()) and terminal_ok) else "FAILED"
    report["technical_gates"] = {**gates, "readback": report["readback_ok"], "terminal": terminal_ok}
    report["artifacts"] = {"wav": str(wav), "working_als": str(working_als)}
    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    (out / "report" / "tech_house_production_kit_v1.json").write_text(json.dumps(report, indent=2, default=str),
                                                                      encoding="utf-8")
    (out / "report" / "tech_house_production_kit_v1.md").write_text(render_markdown(report, metrics),
                                                                    encoding="utf-8")
    return report


def render_markdown(report: dict[str, Any], metrics: dict[str, Any]) -> str:
    """Human-readable mirror of the JSON report. States facts, never taste."""
    delivered = metrics.get("delivered") or {}
    gates = report.get("technical_gates") or {}
    lines = [
        f"# {kit.MILESTONE}",
        "",
        f"- TECHNICAL: **{report.get('technical')}**",
        f"- HUMAN_AUDITION: **{report.get('human_audition')}**",
        f"- WAV: `{(report.get('artifacts') or {}).get('wav')}`",
        f"- Live set (working copy): `{(report.get('artifacts') or {}).get('working_als')}`",
        f"- {kit.BARS} bars, {kit.TEMPO_BPM:g} BPM, {kit.METER[0]}/{kit.METER[1]}, {kit.KEY}",
        f"- Write path: {report.get('musical_writes_path')}",
        "",
        "## Sounds",
        "",
        "| Role | Source | Device |",
        "|---|---|---|",
    ]
    lines += [f"| {item.role} | {item.browser_name} | {item.ableton_device} |" for item in kit.KIT]
    lines += ["", f"Provenance: {kit.PROVENANCE_CORE_LIBRARY}", "", "## Technical gates", ""]
    lines += [f"- {name}: {'PASS' if ok else 'FAIL'}" for name, ok in gates.items()]
    lines += [
        "",
        "## Capture (technical measurements only)",
        "",
        f"- Peak: {delivered.get('peak_dbfs')} dBFS",
        f"- RMS L/R: {delivered.get('left_rms')} / {delivered.get('right_rms')}",
        f"- Integrated loudness: {delivered.get('lufs_integrated')} LUFS",
        f"- L/R correlation: {delivered.get('lr_correlation')}",
        f"- Capture attempts: {len(report.get('capture_attempts') or [])}",
        "",
        str(metrics.get("note") or ""),
        "",
        "## Not done in this milestone",
        "",
        f"- Sidechain: {report.get('sidechain')} (bass onsets avoid every kick instead)",
        f"- Automation: {report.get('automation')}",
        f"- Semantic controls: {report.get('semantic_controls')}",
        f"- Added processing: {report.get('processing') or 'none (presets as shipped)'}",
        "- Save: not programmatic; save the set in Live with Ctrl+S.",
        "",
        "Sound quality is not claimed here. It is the human audition's call.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=kit.MILESTONE)
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--core-library", type=Path, default=DEFAULT_CORE_LIBRARY)
    parser.add_argument("--stab-fallback", action="store_true",
                        help="Use documented Chord Dub Pluck fallback after a recorded Stab load timeout")
    args = parser.parse_args()
    report = run(template=args.template, out=args.out, core_library=args.core_library,
                 stab_fallback=args.stab_fallback)
    print(json.dumps({key: report.get(key) for key in ("technical", "technical_gates", "readback_ok",
                                                         "human_audition", "artifacts")}, indent=2))
    return 0 if report["technical"] == "VERIFIED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
