"""Bounded, real-Live three-track smoke for the direct executor."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import time

import numpy as np
import soundfile as sf

from copilot.audio.batch_capture import capture_parallel_pass
from copilot.audio.live3r_trust import _recorders
from copilot.audio.live_capture import ensure_master_tap
from copilot.core_v2.direct import AbletonBatch, DirectAbletonExecutor, DirectAbletonSession, TrackBatch
from copilot.platform.runtime_paths import runtime_root
from copilot.schemas.observation import SignalPoint
from copilot.schemas.session import MidiNote


OUT = runtime_root() / "core-v2-smoke"
WAV = OUT / "fast_direct_executor_v1.wav"


def batch() -> AbletonBatch:
    note = lambda pitch, start, duration, velocity: MidiNote(
        pitch=pitch, start_time=start, duration=duration, velocity=velocity)
    return AbletonBatch(
        tempo=126.0, length_beats=16.0,
        tracks=(
            TrackBatch("FAST V2 - Kick", "Kick 909 1.aif", "sample",
                       tuple(note(60, float(beat), 0.25, 110) for beat in range(16)), 0.72),
            TrackBatch("FAST V2 - Hat", "Hihat Closed 909.aif", "sample",
                       tuple(note(60, beat + 0.5, 0.15, 75) for beat in range(16)), 0.52),
            TrackBatch("FAST V2 - Bass", "Basic FM House Bass.adg", "device",
                       tuple(note(43 if beat % 4 else 46, beat + 0.75, 0.22, 92)
                             for beat in (0, 2, 4, 6, 8, 10, 12, 14)), 0.65),
        ),
    )


def wav_metrics(path: Path) -> dict:
    audio, rate = sf.read(path, always_2d=True, dtype="float64")
    rms = np.sqrt(np.mean(audio * audio, axis=0))
    peaks = np.max(np.abs(audio), axis=0)
    return {"path": str(path), "sample_rate": rate, "duration_s": len(audio) / rate,
            "channels": audio.shape[1], "left_rms": float(rms[0]),
            "right_rms": float(rms[1]) if len(rms) > 1 else 0.0,
            "left_peak": float(peaks[0]), "right_peak": float(peaks[1]) if len(peaks) > 1 else 0.0,
            "peak": float(np.max(peaks)),
            "lr_identical": bool(audio.shape[1] == 2 and np.array_equal(audio[:, 0], audio[:, 1]))}


def run() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    if WAV.exists():
        raise FileExistsError("SMOKE_WAV_ALREADY_EXISTS")
    started = time.perf_counter()
    session = DirectAbletonSession()
    report: dict = {"milestone": "FAST_DIRECT_EXECUTOR_V1", "status": "RUNNING",
                    "safe_write_used": False, "manifest_used": False, "template_used": False,
                    "save_used": False}
    try:
        session.connect()
        report["ableton_pid"] = session.pid
        executor = DirectAbletonExecutor(session)
        report["readback"] = executor.apply(batch())
        session.daw.set_current_song_time(0.0)
        session.daw.start_playback()
        time.sleep(2.0)
        session.daw.stop_playback()
        report["play_stop"] = True
        report["master_tap"] = ensure_master_tap(session.daw)
        snapshot = session.daw.snapshot(include_notes=False)
        recorder = _recorders(kick_index=-1, bass_index=-1, kick_id="unused", bass_id="unused",
                              kick_channel="unused", bass_channel="unused",
                              main_point=SignalPoint.MAIN_FINAL)[:1]
        captured = capture_parallel_pass(
            session.daw, start_beat=0.0, end_beat=16.0, fire_tracks=[], recorders=recorder,
            tempo=126.0, session_revision=snapshot.revision, transport="arrangement",
        )
        asset = captured["assets"]["master"]
        source = Path(asset.analysis_file_path or asset.file_path)
        shutil.copy2(source, WAV)
        report["capture"] = wav_metrics(WAV)
        metrics = report["capture"]
        if not (metrics["channels"] == 2 and metrics["duration_s"] > 0
                and metrics["left_rms"] > 1e-5 and metrics["right_rms"] > 1e-5
                and metrics["peak"] < 1.0):
            raise RuntimeError("CAPTURE_TECHNICAL_GATE_FAILED")
        report["status"] = "VERIFIED"
        report["rpc_total"] = session.daw.tcp_stats()["total"]
    except Exception as exc:
        report["status"] = "FAILED"
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        try:
            session.daw.stop_playback()
        except Exception:
            pass
        report["elapsed_seconds"] = round(time.perf_counter() - started, 3)
        (OUT / "direct_execution_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        session.close()
    return report


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, default=str))
