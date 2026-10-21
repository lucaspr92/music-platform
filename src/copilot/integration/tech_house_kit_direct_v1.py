"""Run the existing Tech House Kit through the verified fast direct path."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
import soundfile as sf

from copilot.audio.batch_capture import capture_parallel_pass
from copilot.audio.live3r_trust import _recorders
from copilot.audio.live_capture import ensure_master_tap
from copilot.core_v2 import AbletonBatch, DirectAbletonExecutor, DirectAbletonSession, TrackBatch
from copilot.platform.runtime_paths import ableton_core_library, runtime_root
from copilot.producer import tech_house_kit_v1 as kit
from copilot.schemas.observation import SignalPoint
from copilot.schemas.session import MidiNote


OUT = runtime_root() / "production-kit" / "tech-house-direct-v1"
WAV = OUT / "audio" / "tech_house_production_kit_v1_direct.wav"
VOLUMES = {"Kick": .75, "Clap": .66, "Closed Hat": .55, "Open Hat": .55,
           "Perc": .52, "Bass": .70, "Stab": .58}


def make_batch(plan: dict) -> AbletonBatch:
    specs = []
    for item in kit.KIT:
        specs.append(TrackBatch(
            name=f"KIT DIRECT V1 - {item.role}", source_name=item.browser_name,
            fallback_source_name=item.fallback_browser_name,
            source_kind="sample" if item.is_one_shot else "device",
            notes=tuple(MidiNote(**row) for row in plan["roles"][item.role]["notes"]),
            volume=VOLUMES[item.role],
        ))
    return AbletonBatch(tempo=kit.TEMPO_BPM, length_beats=kit.CLIP_BEATS,
                        tracks=tuple(specs))


def metrics(path: Path) -> dict:
    audio, rate = sf.read(path, always_2d=True, dtype="float64")
    left, right = audio[:, 0], audio[:, 1]
    out = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
           "duration_s": len(audio) / rate, "sample_rate": rate, "channels": audio.shape[1],
           "left_rms": float(np.sqrt(np.mean(left * left))),
           "right_rms": float(np.sqrt(np.mean(right * right))),
           "left_peak": float(np.max(np.abs(left))),
           "right_peak": float(np.max(np.abs(right))),
           "peak": float(np.max(np.abs(audio))),
           "lr_correlation": float(np.corrcoef(left, right)[0, 1]),
           "lr_identical": bool(np.array_equal(left, right))}
    try:
        import pyloudnorm
        out["lufs_integrated"] = float(pyloudnorm.Meter(rate).integrated_loudness(audio))
    except Exception:
        out["lufs_integrated"] = None
    return out


def _capture(session: DirectAbletonSession) -> Path:
    ensure_master_tap(session.daw)
    snapshot = session.daw.snapshot(include_notes=False)
    recorder = _recorders(kick_index=-1, bass_index=-1, kick_id="unused", bass_id="unused",
                          kick_channel="unused", bass_channel="unused",
                          main_point=SignalPoint.MAIN_FINAL)[:1]
    result = capture_parallel_pass(session.daw, start_beat=0.0, end_beat=kit.CLIP_BEATS,
                                   fire_tracks=[], recorders=recorder, tempo=kit.TEMPO_BPM,
                                   session_revision=snapshot.revision, transport="arrangement")
    asset = result["assets"]["master"]
    return Path(asset.analysis_file_path or asset.file_path)


def run() -> dict:
    if WAV.exists():
        raise FileExistsError("DIRECT_KIT_WAV_ALREADY_EXISTS")
    for folder in (OUT / "audio", OUT / "plan", OUT / "evidence", OUT / "report"):
        folder.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    definition = kit.kit_definition(ableton_core_library())
    plan = kit.groove_plan()
    (OUT / "plan" / "kit_definition.json").write_text(json.dumps(definition, indent=2), encoding="utf-8")
    (OUT / "plan" / "groove_plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    report: dict = {"milestone": "TECH_HOUSE_PRODUCTION_KIT_V1_DIRECT", "technical": "RUNNING",
                    "human_audition": "PENDING", "safe_write_used": False, "template_used": False,
                    "manifest_used": False, "save_used": False, "sidechain": "DEFERRED",
                    "automation": "DEFERRED", "processing": "NONE"}
    session = DirectAbletonSession()
    try:
        session.connect(allowed_smoke_root=OUT.parents[1])
        report["ableton_pid"] = session.pid
        report["owned_smoke_reused"] = session.owned_smoke
        executor = DirectAbletonExecutor(session)
        result = executor.apply(make_batch(plan))
        report["readback"] = result
        (OUT / "evidence" / "ableton_readback.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8")
        session.daw.set_current_song_time(0.0)
        session.daw.start_playback()
        time.sleep(2.0)
        session.daw.stop_playback()
        report["play_stop"] = "VERIFIED"
        attempts = []
        for attempt in range(2):
            source = _capture(session)
            probe = metrics(source)
            attempts.append({"attempt": attempt + 1, "metrics": probe})
            if probe["peak"] < 1.0:
                break
            if attempt == 0:
                for row in result["tracks"].values():
                    session.daw.set_mixer_volume(row["index"], max(.05, row["volume"] - .06))
        report["capture_attempts"] = attempts
        shutil.copy2(source, WAV)
        audio = metrics(WAV)
        report["audio"] = audio
        (OUT / "evidence" / "capture_metrics.json").write_text(
            json.dumps(audio, indent=2), encoding="utf-8")
        if not (audio["channels"] == 2 and 29 < audio["duration_s"] < 32
                and audio["left_rms"] > 1e-5 and audio["right_rms"] > 1e-5
                and audio["peak"] < 1.0 and not audio["lr_identical"]):
            raise RuntimeError("DIRECT_KIT_CAPTURE_GATE_FAILED")
        report["technical"] = "VERIFIED"
        report["rpc_total"] = session.daw.tcp_stats()["total"]
        report["timeouts"] = executor.timeouts
    except Exception as exc:
        report["technical"] = "FAILED"
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        try:
            session.daw.stop_playback()
        except Exception:
            pass
        report["elapsed_seconds"] = round(time.perf_counter() - started, 3)
        (OUT / "evidence" / "direct_execution_report.json").write_text(
            json.dumps(report, indent=2, default=str), encoding="utf-8")
        (OUT / "report" / "tech_house_production_kit_v1_direct.json").write_text(
            json.dumps(report, indent=2, default=str), encoding="utf-8")
        (OUT / "report" / "tech_house_production_kit_v1_direct.md").write_text(
            f"# Tech House Kit Direct V1\n\nTechnical: {report['technical']}\n\n"
            f"Human audition: PENDING\n\nWAV: {WAV if WAV.exists() else 'none'}\n\n"
            "No musical quality judgment is made here.\n", encoding="utf-8")
        session.close()
    return report


if __name__ == "__main__":
    result = run()
    print(json.dumps({"technical": result["technical"], "audio": result.get("audio"),
                      "elapsed_seconds": result["elapsed_seconds"], "rpc_total": result.get("rpc_total")}, indent=2))
