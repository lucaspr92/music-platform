"""One real V2 A/B pass on the already-owned Live session; no V1 mutation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import time

from copilot.core_v2.direct import AbletonBatch, DirectAbletonExecutor, DirectAbletonSession, TrackBatch
from copilot.daw.detect import detect_ableton
from copilot.integration.tech_house_kit_direct_v1 import _capture, metrics as wav_metrics, VOLUMES
from copilot.platform.runtime_paths import runtime_root
from copilot.producer import tech_house_kit_v1 as v1, tech_house_kit_v2 as v2
from copilot.schemas.session import MidiNote


RUNTIME = runtime_root()
A_DIR = RUNTIME / "production-kit" / "tech-house-direct-v1"
B_DIR = RUNTIME / "production-kit" / "tech-house-direct-v2"
A_WAV = A_DIR / "audio" / "tech_house_production_kit_v1_direct.wav"
B_WAV = B_DIR / "audio" / "tech_house_production_kit_v2.wav"
OLD_NAMES = ["FAST V2 - Kick", "FAST V2 - Hat", "FAST V2 - Bass"] + [
    f"KIT DIRECT V1 - {sound.role}" for sound in v1.KIT
]
DEFAULT_NAMES = ["1-MIDI", "2-MIDI", "3-Audio", "4-Audio"]


def make_batch(plan: dict) -> AbletonBatch:
    specs = []
    for sound in v2.KIT:
        specs.append(TrackBatch(
            name=f"KIT DIRECT V2 - {sound.role}", source_name=sound.browser_name,
            source_kind="sample" if sound.is_one_shot else "device",
            notes=tuple(MidiNote(**row) for row in plan["roles"][sound.role]["notes"]),
            volume=VOLUMES[sound.role], fallback_source_name=None,
        ))
    return AbletonBatch(v1.TEMPO_BPM, v1.CLIP_BEATS, tuple(specs))


def _validate_batch(batch: AbletonBatch) -> None:
    if batch.tempo != 126 or batch.length_beats != 64 or len(batch.tracks) != 7:
        raise ValueError("INVALID_KIT_V2_BATCH")
    if [t.name for t in batch.tracks] != [f"KIT DIRECT V2 - {s.role}" for s in v2.KIT]:
        raise ValueError("INVALID_KIT_V2_ROLES")
    for track in batch.tracks:
        if not track.source_name or not track.notes or track.fallback_source_name:
            raise ValueError("INVALID_KIT_V2_SOURCE_OR_NOTES")
        if not 0 < track.volume <= .85 or any(
            n.start_time < 0 or n.start_time + n.duration > 64.000001 for n in track.notes
        ):
            raise ValueError("INVALID_KIT_V2_NOTE_BOUNDARY")


def _preflight(session: DirectAbletonSession) -> tuple[dict[int, bool], str]:
    detected = detect_ableton()
    if not detected.process_running or not detected.process_pid:
        raise RuntimeError("ABLETON_NOT_RUNNING")
    session.pid = detected.process_pid
    session.daw.connect()
    session.daw.get_session_info()  # handshake, not just an open TCP port
    path = str(session.daw.get_session_path().get("path") or "")
    session.initial_path = path
    if not path or not Path(path).resolve().is_relative_to(RUNTIME.resolve()):
        raise RuntimeError("UNRECOGNIZED_LIVE_PROJECT")
    snapshot = session.daw.snapshot(include_notes=False)
    arranged = session.daw.get_arrangement_clips().get("clips") or []
    if [t.name for t in snapshot.tracks] != DEFAULT_NAMES + OLD_NAMES:
        raise RuntimeError("UNKNOWN_OR_CHANGED_LIVE_TRACKS")
    if len(arranged) != 10:
        raise RuntimeError("UNKNOWN_OR_CHANGED_ARRANGEMENT")
    for index, track in enumerate(snapshot.tracks):
        if index < 4:
            if track.devices or track.clips:
                raise RuntimeError("DEFAULT_TRACK_HAS_USER_WORK")
            continue
        if track.role != "midi" or len(track.devices) != 1 or len(track.clips) != 1:
            raise RuntimeError("PRIOR_OWNED_TRACK_CHANGED")
        clips = [a for a in arranged if a.get("track_index") == track.index]
        expected_end = 16 if index < 7 else 64
        if (len(clips) != 1 or abs(float(clips[0]["start_time"])) > 1e-6
                or abs(float(clips[0]["end_time"]) - expected_end) > 1e-6):
            raise RuntimeError("PRIOR_OWNED_ARRANGEMENT_CHANGED")
    if abs(snapshot.transport.tempo - 126) > .01:
        raise RuntimeError("TEMPO_CHANGED")
    a_report = json.loads((A_DIR / "evidence" / "direct_execution_report.json").read_text(encoding="utf-8"))
    a_hash = hashlib.sha256(A_WAV.read_bytes()).hexdigest()
    if a_report.get("technical") != "VERIFIED" or a_hash != a_report["audio"]["sha256"]:
        raise RuntimeError("BASELINE_A_WAV_CHANGED")
    return {t.index: t.mixer.mute for t in snapshot.tracks[4:]}, a_hash


def _note_key(note: MidiNote | dict) -> tuple:
    if isinstance(note, dict):
        return (int(note["pitch"]), round(float(note["start_time"]), 6),
                round(float(note["duration"]), 6), int(note["velocity"]))
    return (note.pitch, round(note.start_time, 6), round(note.duration, 6), note.velocity)


def _apply(session: DirectAbletonSession, batch: AbletonBatch, old_mutes: dict[int, bool]) -> dict:
    _validate_batch(batch)
    executor = DirectAbletonExecutor(session)
    daw = session.daw
    if str(daw.get_session_path().get("path") or "") != session.initial_path:
        raise RuntimeError("SESSION_CHANGED_BEFORE_V2")
    daw.stop_playback()
    for index in old_mutes:
        daw.set_track_mute(index, True)
    muted = daw.snapshot(include_notes=False)
    if any(not t.mixer.mute for t in muted.tracks[4:14]):
        raise RuntimeError("PRIOR_TRACK_MUTE_READBACK_FAILED")
    created = {}
    for spec in batch.tracks:
        index = int(daw.create_midi_track(spec.name)["index"])
        created[spec.name] = index
        loaded = executor._load_with_fallback(index, spec)
        if loaded != spec.source_name:
            raise RuntimeError(f"V2_SOURCE_SUBSTITUTED: {spec.name}")
        daw.create_midi_clip(index, 0, 64)
        daw.replace_clip_notes(index, 0, list(spec.notes))
        daw.duplicate_clip_to_arrangement(index, 0, 0)
        daw.set_mixer_volume(index, spec.volume)
    final = daw.snapshot(include_notes=False)
    arranged = daw.get_arrangement_clips().get("clips") or []
    if abs(final.transport.tempo - 126) > .01 or len(final.tracks) != 21:
        raise RuntimeError("V2_TEMPO_OR_TRACK_READBACK_FAILED")
    details = {}
    for spec in batch.tracks:
        track = next((t for t in final.tracks if t.name == spec.name), None)
        if track is None or track.index != created[spec.name] or not track.devices:
            raise RuntimeError(f"V2_TRACK_OR_DEVICE_READBACK_FAILED: {spec.name}")
        observed = daw.get_clip_notes(track.index, 0).get("notes") or []
        clip = [a for a in arranged if a.get("track_index") == track.index]
        if (sorted(map(_note_key, observed)) != sorted(map(_note_key, spec.notes))
                or len(clip) != 1 or abs(float(clip[0]["start_time"])) > 1e-6
                or abs(float(clip[0]["end_time"]) - 64) > 1e-6
                or abs(track.mixer.volume - spec.volume) > 1e-4):
            raise RuntimeError(f"V2_CLIP_OR_VOLUME_READBACK_FAILED: {spec.name}")
        details[spec.name] = {"index": track.index, "source": spec.source_name,
                              "devices": [d.name for d in track.devices],
                              "note_count": len(observed), "arrangement": clip,
                              "volume": track.mixer.volume}
    return {"tempo": final.transport.tempo, "tracks": details,
            "track_count": len(final.tracks), "timeouts": executor.timeouts}


def _restore_mutes(session: DirectAbletonSession, old_mutes: dict[int, bool]) -> bool:
    daw = session.daw
    daw.stop_playback()
    current = daw.snapshot(include_notes=False)
    for track in current.tracks:
        if track.name.startswith("KIT DIRECT V2 - "):
            daw.set_track_mute(track.index, True)
    for index, mute in old_mutes.items():
        daw.set_track_mute(index, mute)
    observed = daw.snapshot(include_notes=False)
    return (all(observed.tracks[index].mixer.mute == mute for index, mute in old_mutes.items())
            and all(t.mixer.mute for t in observed.tracks if t.name.startswith("KIT DIRECT V2 - ")))


def run() -> dict:
    if B_WAV.exists():
        raise FileExistsError("V2_WAV_ALREADY_EXISTS")
    start = time.perf_counter()
    for folder in (B_DIR / "audio", B_DIR / "plan", B_DIR / "evidence", B_DIR / "report"):
        folder.mkdir(parents=True, exist_ok=True)
    core_library = Path(json.loads((A_DIR / "plan" / "kit_definition.json").read_text(encoding="utf-8"))["core_library"])
    definition = v2.kit_definition(core_library)
    plan = v2.groove_plan()
    batch = make_batch(plan)
    _validate_batch(batch)
    (B_DIR / "plan" / "kit_definition.json").write_text(json.dumps(definition, indent=2), encoding="utf-8")
    (B_DIR / "plan" / "groove_plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    report: dict = {"milestone": v2.MILESTONE, "technical": "RUNNING",
                    "human_audition": "PENDING", "baseline_a": str(A_WAV), "candidate_b": str(B_WAV),
                    "fast_direct_used": True, "safe_write_used": False, "save_used": False,
                    "sidechain": "DEFERRED", "automation": "DEFERRED"}
    session = DirectAbletonSession()
    old_mutes: dict[int, bool] = {}
    try:
        old_mutes, report["baseline_a_sha256"] = _preflight(session)
        report["ableton_pid"] = session.pid
        report["readback"] = _apply(session, batch, old_mutes)
        session.daw.set_current_song_time(0)
        session.daw.start_playback()
        time.sleep(2)
        session.daw.stop_playback()
        report["play_stop"] = "VERIFIED"
        attempts = []
        for attempt in range(2):
            source = _capture(session)
            measured = wav_metrics(source)
            attempts.append({"attempt": attempt + 1, "metrics": measured})
            if measured["peak"] < 1:
                break
            if attempt == 0:
                for track in report["readback"]["tracks"].values():
                    session.daw.set_mixer_volume(track["index"], max(.05, track["volume"] - .06))
        report["capture_attempts"] = attempts
        if not (measured["channels"] == 2 and 29 < measured["duration_s"] < 32
                and measured["left_rms"] > 1e-5 and measured["right_rms"] > 1e-5
                and measured["peak"] < 1 and not measured["lr_identical"]):
            raise RuntimeError("V2_CAPTURE_TECHNICAL_GATE_FAILED")
        shutil.copy2(source, B_WAV)
        report["audio"] = wav_metrics(B_WAV)
        if hashlib.sha256(A_WAV.read_bytes()).hexdigest() != report["baseline_a_sha256"]:
            raise RuntimeError("BASELINE_A_CHANGED_DURING_V2")
        report["technical"] = "VERIFIED"
    except Exception as exc:
        report["technical"] = "FAILED"
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        try:
            if old_mutes:
                report["mutes_restored"] = _restore_mutes(session, old_mutes)
                if not report["mutes_restored"]:
                    report["technical"] = "FAILED"
                    report["error"] = "MUTE_RESTORE_READBACK_FAILED"
        except Exception as exc:
            report["technical"] = "FAILED"
            report["mute_restore_error"] = f"{type(exc).__name__}: {exc}"
        report["rpc_count"] = session.daw.tcp_stats().get("total")
        report["elapsed_seconds"] = round(time.perf_counter() - start, 3)
        (B_DIR / "evidence" / "direct_execution_report.json").write_text(
            json.dumps(report, indent=2, default=str), encoding="utf-8")
        (B_DIR / "report" / "tech_house_kit_v2.json").write_text(
            json.dumps(report, indent=2, default=str), encoding="utf-8")
        session.close()
    if report["technical"] != "VERIFIED":
        raise RuntimeError(report.get("error", "V2_TECHNICAL_GATE_FAILED"))
    return report


if __name__ == "__main__":
    done = run()
    print(json.dumps({k: done.get(k) for k in ("technical", "audio", "elapsed_seconds", "rpc_count", "mutes_restored")}, indent=2))
