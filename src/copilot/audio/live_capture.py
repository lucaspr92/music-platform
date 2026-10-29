from __future__ import annotations

import os
import shutil
import socket
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import soundfile as sf
from pydantic import BaseModel, Field

from copilot.audio.measure import measure_audio
from copilot.audio.semantics import DEFAULT_PREROLL_BEATS, region_windows
from copilot.daw.ableton_tcp import AbletonTcpAdapter
from copilot.daw.adapter import DawError
from copilot.midi.time import bars_to_beats
from copilot.schemas.observation import (
    CaptureView,
    MusicObservation,
    ObservationSource,
    SignalPoint,
    TailPolicy,
)
from copilot.schemas.session import MidiNote

TAP_NAME = "Copilot Audio Tap"
MASTER_INDEX = -1
STAGING_NAME = "_next.wav"
STAGING_KICK = "_next_kick.wav"
STAGING_BASS = "_next_bass.wav"
from copilot.platform.system import default_capture_dir

DEFAULT_CAPTURE_DIR = default_capture_dir()
DEVICE_SOURCE = (
    Path(__file__).resolve().parents[3] / "devices" / "Copilot Audio Tap.amxd"
)
USER_LIBRARY_TAP = (
    Path.home()
    / "Documents"
    / "Ableton"
    / "User Library"
    / "Presets"
    / "Audio Effects"
    / "Max Audio Effect"
    / "Copilot"
    / "Copilot Audio Tap.amxd"
)
TAP_UDP_PORT = 19877
EXPECTED_TAP_PROTOCOL = 3
TAP_READBACK_TIMEOUT_S = 10.0
TAP_READBACK_POLL_S = 0.10
SONG_TIME_RESTORE_TOLERANCE_BEATS = 0.08  # ~40 ms at 120 BPM; not sample-accurate.
SILENCE_RMS = 1.0e-4
SILENCE_PEAK = 1.0e-3
ONSET_PEAK = 3.0e-3
DURATION_TOLERANCE_S = 0.050
TEMPO_EPS_BPM = 0.05
SOURCE_TRACK_NAME = "LIVE2 Source"
SOURCE_CLIP_NAME = "LIVE2 4 Bars"
SILENT_TRACK_NAME = "LIVE21 Silent"
REGION_LOW_PITCH = 48
REGION_HIGH_PITCH = 84


class AudioCaptureError(DawError):
    def __init__(
        self,
        code: str,
        message: str,
        *,
        details: dict[str, object] | None = None,
    ) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.details = dict(details or {})


class StageTimer:
    def __init__(self) -> None:
        self._last = time.perf_counter()
        self.marks: dict[str, float] = {}

    def mark(self, name: str) -> None:
        now = time.perf_counter()
        self.marks[name] = self.marks.get(name, 0.0) + (now - self._last)
        self._last = now


@dataclass
class CaptureContext:
    """One session read, reused while session_revision stays valid.

    Mixer restore is skipped only when the mutation plan did not touch mixer.
    Routing restore still runs for tracks that this plan actually rerouted.
    """

    session: object
    track_infos: dict[int, dict[str, object]]
    master_info: dict[str, object]
    mixer: list[dict[str, object]]
    outputs: list[dict[str, object]]
    tempo: float | None = None
    sample_rate: int | None = None
    tap_device: dict[str, object] | None = None
    rec_param_index: int | None = None
    off_main_tap: dict[str, object] | None = None
    capture_track_index: int | None = None
    sends_only_name: str | None = None
    return_volumes: list[tuple[int, float]] = field(default_factory=list)
    revision: int | None = None


class AudioAsset(BaseModel):
    capture_id: str
    raw_file_path: str
    analysis_file_path: str
    file_path: str
    source: str = "MASTER"
    source_type: str = "MASTER"
    source_stable_id: str | None = None
    requested_start_beat: float
    requested_end_beat: float
    start_beat: float
    end_beat: float
    sample_rate: int
    channels: int
    raw_duration: float
    analysis_duration: float
    duration: float
    session_revision: int
    tempo: float | None = None
    requested_start_qn: float | None = None
    requested_end_qn: float | None = None
    transport_start_estimate_qn: float | None = None
    capture_arm_time: float | None = None
    analysis_start_offset: float | None = None
    first_poll_qn: float | None = None
    capture_file_start_time: float | None = None
    transport_target_qn: float | None = None
    verified_transport_start_qn: float | None = None
    verification_wall_time: float | None = None
    transport_primitive_version: str | None = None
    signature_numerator: int | None = None
    signature_denominator: int | None = None
    transport_start_observed: float | None = None
    transport_stop_observed: float | None = None
    trim_start_samples: int = 0
    trim_end_samples: int = 0
    rms: float | None = None
    peak: float | None = None
    size_bytes: int = 0
    capture_view: CaptureView | None = None
    observation_source: ObservationSource | None = None
    signal_point: SignalPoint | None = None
    signal_point_label: str | None = None
    tail_policy: TailPolicy = TailPolicy.STRICT_REGION
    request_start_beat: float | None = None
    request_end_beat: float | None = None
    capture_start_beat: float | None = None
    capture_end_beat: float | None = None
    analysis_start_beat: float | None = None
    analysis_end_beat: float | None = None
    preroll_beats: float = 0.0
    session_revision_at_start: int | None = None
    session_revision_at_end: int | None = None
    capture_quality: str = "OK"
    finite_samples: bool = True
    expected_frames: int | None = None
    actual_frames: int | None = None
    stage_timings: dict[str, float] = Field(default_factory=dict)
    slot: int | None = None
    pass_id: str | None = None
    routing: str | None = None
    returns_included: bool | None = None


class WavCheck(BaseModel):
    exists: bool
    size_bytes: int
    channels: int | None = None
    sample_rate: int | None = None
    duration: float | None = None
    rms: float | None = None
    peak: float | None = None
    subtype: str | None = None
    ok: bool = False
    errors: list[str] = Field(default_factory=list)
    finite_samples: bool = True
    frames: int | None = None
    # A stereo file can carry one silent channel and still look healthy: the channel COUNT and
    # the MONO downmix both pass. These per-channel figures are what make that visible.
    channel_rms: list[float] | None = None
    channel_peak: list[float] | None = None
    stereo_correlation: float | None = None


def capture_dir() -> Path:
    raw = os.environ.get("MUSICCOPILOT_CAPTURE_DIR", str(DEFAULT_CAPTURE_DIR))
    path = Path(raw)
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise AudioCaptureError(
            "CANNOT_CREATE_OUTPUT", f"cannot create capture dir {path}: {exc}"
        ) from exc
    return path


def install_audio_tap_device() -> dict[str, str | bool]:
    from copilot.importing.m4l_runtime_v1 import ensure_m4l_runtime

    if not DEVICE_SOURCE.is_file():
        raise AudioCaptureError(
            "TAP_MISSING", f"device source missing: {DEVICE_SOURCE}"
        )
    result = ensure_m4l_runtime(source=DEVICE_SOURCE)
    if result.get("status") not in {"INSTALLED", "ALREADY_CURRENT", "UPDATED"}:
        raise AudioCaptureError(
            "TAP_MISSING",
            str(result.get("reason") or result.get("status") or "M4L runtime provisioning failed"),
        )
    dest = Path(str(result.get("installed") or USER_LIBRARY_TAP))
    capture_dir()
    return {
        "source": str(DEVICE_SOURCE),
        "installed": str(dest),
        "exists": dest.is_file(),
        "size": dest.stat().st_size if dest.is_file() else 0,
        "provisioning": str(result.get("status")),
    }


def rebuild_audio_tap_device() -> dict[str, object]:
    """Write the current Slot-enabled maxpat into the unfrozen M4L container."""
    import importlib.util

    builder = DEVICE_SOURCE.parent / "build_amxd.py"
    spec = importlib.util.spec_from_file_location("copilot_build_amxd", builder)
    if spec is None or spec.loader is None:
        raise AudioCaptureError("TAP_MISSING", f"cannot load {builder}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    maxpat = DEVICE_SOURCE.parent / "Copilot Audio Tap.maxpat"
    size = int(module.build_audio_effect_amxd(maxpat, DEVICE_SOURCE))
    installed = install_audio_tap_device()
    return {"rebuilt_bytes": size, "maxpat": str(maxpat), **installed}


def tap_parameter_names(daw: AbletonTcpAdapter, track_index: int) -> list[str]:
    device = find_tap(daw, track_index)
    if device is None:
        return []
    params = daw.get_device_parameters(track_index, int(device["index"]))
    return [str(item.get("name") or "") for item in params.get("parameters") or []]


def tap_has_slot(daw: AbletonTcpAdapter, track_index: int) -> bool:
    return any(name.lower() == "slot" for name in tap_parameter_names(daw, track_index))


def tap_requires_refresh(
    daw: AbletonTcpAdapter,
    track_index: int,
    device_index: int,
    *,
    device_name: str | None = None,
) -> dict[str, object]:
    """Detect an instantiated legacy tap that cannot use the current writer.

    V5 also fixes stereo input wiring. Protocol 4 alone cannot distinguish
    a compiled V4 instance with a silent right channel from the corrected tap.
    """
    params = daw.get_device_parameters(track_index, int(device_index))
    slot_max: float | None = None
    protocol: float | None = None
    for item in params.get("parameters") or []:
        name = str(item.get("name") or "").lower()
        if name == "slot" and item.get("max") is not None:
            slot_max = float(item["max"])
        elif name == "tapprotocol" and item.get("value") is not None:
            protocol = float(item["value"])
    if device_name is None:
        device = find_tap(daw, track_index)
        device_name = str((device or {}).get("name") or "")
    stale = ("Copilot Audio Tap 5" not in device_name) or (slot_max is not None and slot_max < 8.0) or (
        protocol is not None and protocol < 4.0
    )
    return {
        "stale": stale,
        "slot_max": slot_max,
        "tap_protocol": protocol,
        "required_slot_max": 8,
        "required_tap_protocol": 4,
        "device_name": device_name,
    }


def tap_protocol_version(daw: AbletonTcpAdapter, track_index: int) -> int | None:
    device = find_tap(daw, track_index)
    if device is None:
        return None
    params = daw.get_device_parameters(track_index, int(device["index"]))
    for item in params.get("parameters") or []:
        if str(item.get("name") or "").lower() == "tapprotocol":
            return int(round(float(item.get("value") or 0.0)))
    return None


def verify_slot_roundtrip(
    daw: AbletonTcpAdapter, track_index: int, slot: int
) -> dict[str, object]:
    written = set_tap_slot(daw, track_index, slot)
    if written.get("slot") is None:
        raise AudioCaptureError("PROTOCOL_INCOMPATIBLE", f"Slot missing on {track_index}")
    device = find_tap(daw, track_index)
    if device is None:
        raise AudioCaptureError("TAP_MISSING", f"no tap on {track_index}")
    params = daw.get_device_parameters(track_index, int(device["index"]))
    read = None
    for item in params.get("parameters") or []:
        if str(item.get("name") or "").lower() == "slot":
            read = float(item.get("value") or 0.0)
            break
    ok = read is not None and int(round(read)) == int(slot)
    if not ok:
        raise AudioCaptureError(
            "PROTOCOL_INCOMPATIBLE",
            f"Slot write {slot} read-back {read} on track {track_index}",
        )
    return {"slot": slot, "read": read, "ok": True}


def _exclusive_open_ok(path: Path) -> dict[str, object]:
    from copilot.platform.files import exclusive_open_ok

    return exclusive_open_ok(path)


def wav_shared_read_ok(path: Path) -> dict[str, object]:
    from copilot.platform.files import shared_read_ok

    return shared_read_ok(path)


def wav_lock_owners(path: Path) -> list[dict[str, object]]:
    from copilot.platform.files import lock_owners

    return lock_owners(path)


def wait_until_wav_shared_readable(
    paths: list[Path],
    *,
    timeout_s: float = 12.0,
    interval_s: float = 0.05,
) -> dict[str, Any]:
    """Bounded wait until each path is shared-readable. Exclusive release is not required.

    Rec=0 is necessary but not sufficient. If the writer never yields a readable
    file, fail as CAPTURE_FINALIZATION_TIMEOUT — never as TAP_STALE_ARMED.
    """
    deadline = time.time() + timeout_s
    last: list[dict[str, object]] = []
    observations: dict[str, dict[str, object]] = {
        str(path): {
            "path": str(path),
            "samples": 0,
            "saw_exists": False,
            "saw_readable": False,
            "saw_nonzero": False,
            "max_size": 0,
            "last": None,
        }
        for path in paths
    }
    started = time.perf_counter()
    while time.time() < deadline:
        last = []
        ready = True
        for path in paths:
            row = dict(wav_shared_read_ok(path))
            row["path"] = str(path)
            row["exclusive"] = _exclusive_open_ok(path).get("exclusive")
            last.append(row)
            summary = observations[str(path)]
            summary["samples"] = int(summary["samples"]) + 1
            summary["saw_exists"] = bool(summary["saw_exists"] or row.get("exists"))
            summary["saw_readable"] = bool(
                summary["saw_readable"] or row.get("readable")
            )
            size = row.get("size")
            if isinstance(size, (int, float)):
                summary["max_size"] = max(int(summary["max_size"]), int(size))
                summary["saw_nonzero"] = bool(
                    summary["saw_nonzero"] or int(size) > 0
                )
            summary["last"] = dict(row)
            if not row.get("exists") or not row.get("readable"):
                ready = False
        if ready:
            return {
                "ok": True,
                "waited_s": time.perf_counter() - started,
                "rows": last,
                "exclusive_required": False,
            }
        time.sleep(interval_s)
    summary_rows = list(observations.values())
    saw_any_file = any(bool(row["saw_exists"]) for row in summary_rows)
    saw_any_readable = any(bool(row["saw_readable"]) for row in summary_rows)
    if not saw_any_file:
        failure_code = "STAGING_FILE_NOT_CREATED"
    elif not saw_any_readable:
        failure_code = "STAGING_FILE_NOT_SHARED_READABLE"
    else:
        failure_code = "CAPTURE_FINALIZATION_TIMEOUT"
    payload = {
        "ok": False,
        "waited_s": time.perf_counter() - started,
        "rows": last,
        "observation_summary": summary_rows,
        "timeout_s": timeout_s,
        "exclusive_required": False,
    }
    raise AudioCaptureError(
        failure_code,
        "CAPTURE_FINALIZATION_TIMEOUT: staging WAV never became shared-readable: "
        + str(payload),
        details={"stage": "staging_finalize", **payload},
    )


def verify_rec_close_releases_handles(
    daw: AbletonTcpAdapter,
    recorders: list[dict[str, object]],
) -> dict[str, object]:
    """Rec=1 then Rec=0; staging files must be exclusively openable."""
    recorder_indexes = {int(rec["tap_track_index"]) for rec in recorders}
    disabled_foreign: list[int] = []
    count = int(daw.health().get("track_count") or 0)
    for index in list(range(count)) + [MASTER_INDEX]:
        if index in recorder_indexes:
            continue
        try:
            if find_tap(daw, index) is None:
                continue
            set_tap_recording(daw, False, index, broadcast_udp=False)
            set_tap_enabled(daw, index, False)
            disabled_foreign.append(index)
        except (DawError, AudioCaptureError):
            continue
    rows: list[dict[str, object]] = []
    try:
        for rec in recorders:
            set_tap_slot(daw, int(rec["tap_track_index"]), int(rec["slot"]))
            set_tap_recording(
                daw, False, int(rec["tap_track_index"]), broadcast_udp=False
            )
        time.sleep(0.15)
        for rec in recorders:
            set_tap_recording(
                daw, True, int(rec["tap_track_index"]), broadcast_udp=False
            )
        time.sleep(0.45)
        for rec in recorders:
            set_tap_recording(
                daw, False, int(rec["tap_track_index"]), broadcast_udp=False
            )
        time.sleep(0.6)
        ok = True
        for rec in recorders:
            path = staging_path(str(rec["staging"]))
            probe = _exclusive_open_ok(path)
            if not probe["exists"] or not probe["exclusive"]:
                ok = False
            rows.append(
                {
                    "key": rec.get("key"),
                    "track_index": rec["tap_track_index"],
                    "slot": rec["slot"],
                    "path": str(path),
                    **probe,
                }
            )
        return {"ok": ok, "files": rows}
    finally:
        for index in disabled_foreign:
            try:
                set_tap_enabled(daw, index, True)
            except (DawError, AudioCaptureError):
                pass


def master_chain_signal_point(daw: AbletonTcpAdapter) -> SignalPoint:
    devices = [
        str(item.get("name") or "")
        for item in (daw.get_master_info().get("devices") or [])
    ]
    if devices and TAP_NAME.lower() in devices[-1].lower():
        return SignalPoint.MAIN_FINAL
    return SignalPoint.MAIN_NOT_FINAL


def unique_capture_path() -> Path:
    root = capture_dir()
    for _ in range(8):
        capture_id = uuid4().hex[:12]
        path = root / f"capture_{capture_id}.wav"
        if not path.exists():
            return path
    raise AudioCaptureError("CANNOT_CREATE_OUTPUT", "could not allocate unique name")


def staging_path(name: str = STAGING_NAME) -> Path:
    return capture_dir() / name


def _file_locked(exc: OSError) -> bool:
    return getattr(exc, "winerror", None) == 32 or exc.errno in {11, 13, 16}


def take_staging(staging: Path, dest: Path) -> str:
    """Move staging to dest. Copy if Max still holds the file open."""
    last: OSError | None = None
    for _ in range(6):
        try:
            staging.replace(dest)
            return "moved"
        except OSError as exc:
            last = exc
            if not _file_locked(exc):
                raise
            time.sleep(0.15)
    try:
        shutil.copy2(staging, dest)
        return "copied_while_locked"
    except OSError as exc:
        raise AudioCaptureError("FILE_LOCKED", f"{staging} -> {dest}: {exc}") from last


def _clear_staging(name: str = STAGING_NAME) -> None:
    path = staging_path(name)
    if not path.exists():
        return
    for _ in range(6):
        try:
            if path.stat().st_size == 0:
                path.unlink()
            else:
                orphan = capture_dir() / f"capture_orphan_{uuid4().hex[:8]}.wav"
                path.replace(orphan)
            return
        except OSError as exc:
            if not _file_locked(exc):
                raise
            time.sleep(0.15)
    # Max sfrecord~ keeps the file open after Rec=0. Next open overwrites.


def tap_from_info(info: dict[str, object] | None) -> dict[str, object] | None:
    if not info:
        return None
    for device in info.get("devices") or []:
        name = str(device.get("name") or "")
        if TAP_NAME.lower() in name.lower():
            return device
    return None


def mixer_from_track_infos(
    infos: dict[int, dict[str, object]],
) -> list[dict[str, object]]:
    states: list[dict[str, object]] = []
    for index in sorted(infos):
        info = infos[index]
        states.append(
            {
                "index": index,
                "name": info.get("name"),
                "mute": bool(info.get("mute", False)),
                "solo": bool(info.get("solo", False)),
                "arm": bool(info.get("arm", False)),
                "monitoring": info.get("monitoring"),
            }
        )
    return states


def outputs_from_track_infos(
    infos: dict[int, dict[str, object]],
) -> list[dict[str, object]]:
    states: list[dict[str, object]] = []
    for index in sorted(infos):
        info = infos[index]
        states.append(
            {
                "index": index,
                "name": info.get("name"),
                "type": info.get("output_routing_type"),
                "channel": info.get("output_routing_channel"),
            }
        )
    return states


def load_capture_context(daw: AbletonTcpAdapter) -> CaptureContext:
    session = daw.snapshot(include_notes=False)
    infos = {int(k): dict(v) for k, v in daw.last_track_infos.items()}
    master = daw.last_master_info or daw.get_master_info()
    daw.last_master_info = master
    off_main = None
    for index, info in infos.items():
        if tap_from_info(info) is None:
            continue
        if info.get("is_midi_track") and not info.get("is_audio_track"):
            continue
        off_main = {"index": index, "name": info.get("name")}
        break
    capture_index = None
    capture = session.track_by_name("Copilot Capture")
    if capture is not None:
        capture_index = capture.index
    returns = daw.last_return_tracks or daw.get_return_tracks()
    daw.last_return_tracks = returns
    return_volumes = [
        (int(item["index"]), float(item.get("volume") or 0.0))
        for item in returns.get("return_tracks") or []
    ]
    playback = daw.last_playback or daw.get_playback_position()
    live_sr = playback.get("sample_rate")
    return CaptureContext(
        session=session,
        track_infos=infos,
        master_info=master,
        mixer=mixer_from_track_infos(infos),
        outputs=outputs_from_track_infos(infos),
        tempo=float(playback.get("tempo") or 0.0) or None,
        sample_rate=int(live_sr) if live_sr else None,
        tap_device=tap_from_info(master),
        off_main_tap=off_main,
        capture_track_index=capture_index,
        return_volumes=return_volumes,
        revision=session.revision,
    )


def find_taps_on_track(
    daw: AbletonTcpAdapter, track_index: int, *, refresh: bool = False
) -> list[dict[str, object]]:
    info = None
    if not refresh:
        if track_index == MASTER_INDEX and daw.last_master_info:
            info = daw.last_master_info
        elif track_index in daw.last_track_infos:
            info = daw.last_track_infos[track_index]
    if info is None:
        info = (
            daw.get_master_info()
            if track_index == MASTER_INDEX
            else daw.get_track_info(track_index)
        )
    found: list[dict[str, object]] = []
    for device in info.get("devices") or []:
        name = str(device.get("name") or "")
        if TAP_NAME.lower() in name.lower():
            found.append(device)
    return found


def find_tap(
    daw: AbletonTcpAdapter, track_index: int = MASTER_INDEX
) -> dict[str, object] | None:
    taps = find_taps_on_track(daw, track_index)
    return next((tap for tap in taps if "Copilot Audio Tap 5" in str(tap.get("name") or "")), None) or (taps[0] if taps else None)


def wait_for_tap_readback(
    daw: AbletonTcpAdapter,
    track_index: int,
    *,
    timeout_s: float = TAP_READBACK_TIMEOUT_S,
    poll_s: float = TAP_READBACK_POLL_S,
) -> list[dict[str, object]]:
    """Poll Live until a loaded tap is observable or a deadline expires.

    Browser/device loading is asynchronous. A fixed post-load sleep makes
    readiness depend on the host machine's timing. The authoritative signal
    is the next Live readback, so convergence is bounded by a monotonic
    deadline and returns a typed failure when it does not occur.
    """
    deadline = time.monotonic() + max(0.0, float(timeout_s))
    while True:
        found = find_taps_on_track(daw, track_index, refresh=True)
        if found:
            return found
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        time.sleep(min(max(0.01, float(poll_s)), remaining))
    where = "Master" if track_index == MASTER_INDEX else f"track {track_index}"
    raise AudioCaptureError(
        "TAP_READBACK_TIMEOUT",
        f"Copilot Audio Tap did not become observable on {where} before deadline",
    )


def find_master_tap(daw: AbletonTcpAdapter) -> dict[str, object] | None:
    return find_tap(daw, MASTER_INDEX)


def master_tap_position(daw: AbletonTcpAdapter) -> dict[str, object]:
    info = daw.last_master_info or daw.get_master_info()
    daw.last_master_info = info
    devices = list(info.get("devices") or [])
    tap = None
    for device in devices:
        if TAP_NAME.lower() in str(device.get("name") or "").lower():
            tap = device
    last_index = len(devices) - 1 if devices else -1
    tap_index = int(tap["index"]) if tap is not None else None
    return {
        "tap": tap,
        "tap_index": tap_index,
        "device_count": len(devices),
        "last_index": last_index,
        "is_last": tap_index is not None and tap_index == last_index,
        "devices": [str(d.get("name")) for d in devices],
    }


def assert_master_tap_final(daw: AbletonTcpAdapter) -> dict[str, object]:
    pos = master_tap_position(daw)
    if pos["tap"] is None:
        raise AudioCaptureError("TAP_MISSING", "Copilot Audio Tap is not on Master")
    if not pos["is_last"]:
        raise AudioCaptureError(
            "CAPTURE_SIGNAL_POINT_AMBIGUOUS",
            "MASTER_CONTEXT_FINAL requires Copilot Audio Tap as the last Main device; "
            f"tap_index={pos['tap_index']} devices={pos['devices']}",
        )
    return pos


def count_copilot_taps(daw: AbletonTcpAdapter) -> dict[str, object]:
    master = find_taps_on_track(daw, MASTER_INDEX)
    tracks: list[dict[str, object]] = []
    count = int(daw.health().get("track_count") or 0)
    extras = 0
    for index in range(count):
        taps = find_taps_on_track(daw, index)
        if taps:
            info = daw.get_track_info(index)
            tracks.append({"index": index, "name": info.get("name"), "count": len(taps)})
            extras += len(taps)
    return {
        "master": len(master),
        "tracks": tracks,
        "total": len(master) + extras,
    }


def ensure_master_tap(daw: AbletonTcpAdapter) -> dict[str, object]:
    existing = find_master_tap(daw)
    replaced_stale = False
    if existing is not None:
        freshness = tap_requires_refresh(
            daw, MASTER_INDEX, int(existing["index"]), device_name=str(existing.get("name") or "")
        )
        if not freshness["stale"]:
            return {"already_loaded": True, "device": existing, "freshness": freshness}
        replaced_stale = True
    install_audio_tap_device()
    uri = _find_tap_uri(daw)
    if not uri:
        if replaced_stale:
            raise AudioCaptureError(
                "TAP_REFRESH_BLOCKED",
                "stale Copilot Audio Tap found but no current browser URI is available",
            )
        if find_master_tap(daw) is None:
            raise AudioCaptureError(
                "TAP_MISSING",
                "Copilot Audio Tap is not on Master and was not found in the Live browser. "
                "Drop devices/Copilot Audio Tap.amxd onto Master once.",
            )
        return {"already_loaded": True, "device": find_master_tap(daw)}
    if replaced_stale and "Copilot%20Audio%20Tap%205.amxd" not in uri and "Copilot Audio Tap 5.amxd" not in uri:
        raise AudioCaptureError("TAP_REFRESH_BLOCKED", "stereo-correct V5 browser URI is unavailable")
    loaded = daw.load_instrument_or_effect(MASTER_INDEX, uri)
    if loaded.get("error"):
        loaded = daw.load_browser_item(MASTER_INDEX, uri)
    if loaded.get("error"):
        raise AudioCaptureError("TAP_REFRESH_BLOCKED", str(loaded.get("error")))
    deadline = time.monotonic() + TAP_READBACK_TIMEOUT_S
    device = None
    while time.monotonic() < deadline:
        devices = find_taps_on_track(daw, MASTER_INDEX, refresh=True)
        device = next((row for row in devices if "Copilot Audio Tap 5" in str(row.get("name") or "")), None)
        if device is not None:
            break
        time.sleep(TAP_READBACK_POLL_S)
    if device is None:
        raise AudioCaptureError("TAP_READBACK_TIMEOUT", "V5 tap did not appear; prior tap remains installed")
    if replaced_stale and existing is not None:
        prior = next((row for row in devices if row.get("name") == existing.get("name")), None)
        if prior is not None:
            daw.delete_device(MASTER_INDEX, int(prior["index"]))
            device = next((row for row in find_taps_on_track(daw, MASTER_INDEX, refresh=True) if "Copilot Audio Tap 5" in str(row.get("name") or "")), None)
            if device is None:
                raise AudioCaptureError("TAP_REFRESH_BLOCKED", "V5 tap missing after legacy removal")
    return {
        "already_loaded": False,
        "device": device,
        "load": loaded,
        "replaced_stale": replaced_stale,
    }


def _find_tap_uri(daw: AbletonTcpAdapter) -> str | None:
    from copilot.importing.m4l_runtime_v1 import find_canonical_tap_uri

    return find_canonical_tap_uri(daw)


def _rec_parameter(
    daw: AbletonTcpAdapter, device_index: int, track_index: int = MASTER_INDEX
) -> dict[str, object] | None:
    params = daw.get_device_parameters(track_index, device_index)
    for param in params.get("parameters") or []:
        name = str(param.get("name") or "")
        if name.lower() in {"rec", "record"}:
            return param
    return None


def _send_tap_udp(value: int) -> None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.sendto(str(value).encode("ascii"), ("127.0.0.1", TAP_UDP_PORT))
    finally:
        sock.close()


def set_tap_enabled(
    daw: AbletonTcpAdapter, track_index: int, enabled: bool
) -> dict[str, object]:
    device = find_tap(daw, track_index)
    if device is None:
        raise AudioCaptureError("TAP_MISSING", f"no tap on track {track_index}")
    params = daw.get_device_parameters(track_index, int(device["index"]))
    for param in params.get("parameters") or []:
        if str(param.get("name") or "").lower() in {"device on", "on"}:
            daw.set_device_parameter(
                track_index,
                int(device["index"]),
                int(param["index"]),
                1.0 if enabled else 0.0,
            )
            return {"enabled": enabled, "parameter": param.get("name")}
    raise AudioCaptureError("TAP_MISSING", "Device On not found on Copilot Audio Tap")


def set_tap_recording(
    daw: AbletonTcpAdapter,
    recording: bool,
    track_index: int = MASTER_INDEX,
    context: CaptureContext | None = None,
    *,
    broadcast_udp: bool = True,
    device_index: int | None = None,
    rec_param_index: int | None = None,
) -> dict[str, object]:
    index = device_index
    if index is None:
        device = None
        if (
            context is not None
            and track_index == MASTER_INDEX
            and context.tap_device is not None
        ):
            device = context.tap_device
        else:
            device = find_tap(daw, track_index)
        if device is None:
            where = "Master" if track_index == MASTER_INDEX else f"track {track_index}"
            raise AudioCaptureError(
                "TAP_MISSING", f"Copilot Audio Tap is not on {where}"
            )
        index = int(device["index"])
    param_index = rec_param_index
    if param_index is None and (
        context is not None
        and track_index == MASTER_INDEX
        and context.rec_param_index is not None
    ):
        param_index = context.rec_param_index
    if param_index is None:
        param = _rec_parameter(daw, index, track_index)
        if param is not None:
            param_index = int(param["index"])
            if context is not None and track_index == MASTER_INDEX:
                context.rec_param_index = param_index
    sent = {"udp": recording if broadcast_udp else None, "tap_track_index": track_index}
    if param_index is not None:
        written = daw.set_device_parameter(
            track_index, index, param_index, 1.0 if recording else 0.0
        )
        sent["parameter"] = "Rec"
        sent["readback"] = written.get("value")
        sent["device_index"] = index
        sent["parameter_index"] = param_index
    if broadcast_udp:
        _send_tap_udp(1 if recording else 0)
    return sent


def _unknown_command(exc: DawError) -> bool:
    message = str(exc).lower()
    return "unknown command" in message or "unsupported command" in message


def _batch_control_available(daw: AbletonTcpAdapter) -> bool:
    return daw.batch_control_live is not False


def _note_batch_control(daw: AbletonTcpAdapter, live: bool) -> None:
    daw.batch_control_live = live


def _apply_rec_items_serial(
    daw: AbletonTcpAdapter, items: list[dict[str, Any]]
) -> dict[str, float]:
    armed_at: dict[str, float] = {}
    for item in items:
        daw.set_device_parameter(
            int(item["track_index"]),
            int(item["device_index"]),
            int(item["parameter_index"]),
            float(item["value"]),
        )
        armed_at[str(item["id"])] = time.time()
    return armed_at


def set_taps_recording(
    daw: AbletonTcpAdapter,
    recorders: list[dict[str, Any]],
    recording: bool,
    *,
    by_track: dict[int, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Arm/disarm several taps. Batch if the loaded Remote Script supports it."""
    items: list[dict[str, Any]] = []
    skipped: list[str] = []
    want = 1.0 if recording else 0.0
    by_track = by_track or {}
    for rec in recorders:
        track_index = int(rec["tap_track_index"])
        row = by_track.get(track_index)
        current = None if row is None else row.get("rec")
        if (
            not recording
            and current is not None
            and float(current) < 0.5
            and rec.get("force_rec") is not True
        ):
            skipped.append(str(rec.get("key") or track_index))
            continue
        device_index = rec.get("device_index")
        if device_index is None and row is not None:
            device_index = row.get("device_index")
        param_index = rec.get("rec_param_index")
        if param_index is None and row is not None:
            param_index = row.get("rec_param_index")
        if device_index is None or param_index is None:
            set_tap_recording(
                daw,
                recording,
                track_index,
                broadcast_udp=False,
                device_index=None if device_index is None else int(device_index),
                rec_param_index=None if param_index is None else int(param_index),
            )
            continue
        items.append(
            {
                "id": str(rec.get("key") or track_index),
                "track_index": track_index,
                "device_index": int(device_index),
                "parameter_index": int(param_index),
                "value": want,
            }
        )
    if not items:
        return {
            "ok": True,
            "mode": "skipped",
            "skipped": skipped,
            "results": [],
            "armed_at": {},
        }
    if not _batch_control_available(daw):
        armed_at = _apply_rec_items_serial(daw, items)
        return {
            "ok": True,
            "mode": "serial",
            "skipped": skipped,
            "results": items,
            "armed_at": armed_at,
        }
    try:
        result = daw.set_device_parameters(items)
        _note_batch_control(daw, True)
        mode = "batch"
        t_batch = time.time()
        armed_at = {str(item["id"]): t_batch for item in items}
    except DawError as exc:
        if not _unknown_command(exc):
            raise
        _note_batch_control(daw, False)
        armed_at = _apply_rec_items_serial(daw, items)
        return {
            "ok": True,
            "mode": "serial",
            "skipped": skipped,
            "results": items,
            "armed_at": armed_at,
        }
    if not result.get("ok"):
        failed = [row for row in result.get("results") or [] if not row.get("ok")]
        raise AudioCaptureError(
            "TAP_CONTROL_FAILED",
            "partial Rec write: " + str(failed),
        )
    for row in result.get("results") or []:
        value = row.get("value")
        if value is None:
            continue
        if abs(float(value) - want) > 0.1:
            raise AudioCaptureError(
                "TAP_CONTROL_FAILED",
                f"Rec read-back {value} wanted {want} on {row.get('id')}",
            )
    return {**result, "mode": mode, "skipped": skipped, "armed_at": armed_at}


def set_tap_slot(
    daw: AbletonTcpAdapter,
    track_index: int,
    slot: int,
    *,
    device_index: int | None = None,
    parameter_index: int | None = None,
) -> dict[str, object]:
    index = device_index
    if index is None:
        device = find_tap(daw, track_index)
        if device is None:
            raise AudioCaptureError("TAP_MISSING", f"no tap on track {track_index}")
        index = int(device["index"])
    param_index = parameter_index
    if param_index is None:
        params = daw.get_device_parameters(track_index, index)
        for param in params.get("parameters") or []:
            if str(param.get("name") or "").lower() == "slot":
                param_index = int(param["index"])
                break
    if param_index is None:
        return {"slot": None, "missing": True}
    daw.set_device_parameter(track_index, index, param_index, float(slot))
    read = daw.get_device_parameter(track_index, index, param_index)
    value = read.get("value")
    return {
        "slot": slot,
        "parameter_index": param_index,
        "device_index": index,
        "readback": value,
        "ok": value is not None and int(round(float(value))) == int(slot),
    }


def validate_wav(
    path: Path,
    *,
    expected_sr: int | None,
    expected_duration: float,
    require_signal: bool,
) -> WavCheck:
    errors: list[str] = []
    if not path.is_file():
        return WavCheck(exists=False, size_bytes=0, errors=["file missing"], ok=False)
    size = path.stat().st_size
    if size <= 0:
        return WavCheck(
            exists=True,
            size_bytes=0,
            errors=["zero-byte file"],
            ok=False,
        )
    try:
        data, sample_rate = sf.read(str(path), always_2d=True)
        subtype = None
        with sf.SoundFile(str(path)) as handle:
            subtype = handle.subtype
        channels = int(data.shape[1])
        duration = float(len(data) / sample_rate)
        frames = int(len(data))
        finite = bool(np.isfinite(data).all())
        mono = np.mean(data, axis=1)
        rms = float(np.sqrt(np.mean(mono**2))) if finite else float("nan")
        peak = float(np.max(np.abs(mono))) if finite else float("nan")
        channel_rms = [float(np.sqrt(np.mean(data[:, i] ** 2))) for i in range(channels)] if finite else []
        channel_peak = [float(np.max(np.abs(data[:, i]))) for i in range(channels)] if finite else []
        stereo_correlation: float | None = None
        if channels == 2 and finite:
            left, right = data[:, 0], data[:, 1]
            if float(np.std(left)) > 0.0 and float(np.std(right)) > 0.0:
                stereo_correlation = float(np.corrcoef(left, right)[0, 1])
    except Exception as exc:  # noqa: BLE001 - report observed failure
        try:
            with wave.open(str(path), "rb") as handle:
                channels = handle.getnchannels()
                sample_rate = handle.getframerate()
                frames = handle.getnframes()
            return WavCheck(
                exists=True,
                size_bytes=size,
                channels=channels,
                sample_rate=sample_rate,
                duration=frames / float(sample_rate) if sample_rate else None,
                errors=[f"soundfile failed: {exc}"],
                ok=False,
            )
        except Exception as wave_exc:
            return WavCheck(
                exists=True,
                size_bytes=size,
                errors=[f"cannot open wav: {exc}; {wave_exc}"],
                ok=False,
            )
    if not finite:
        errors.append("non-finite samples (NaN/Inf)")
    if channels != 2:
        errors.append(f"channels {channels} != 2")
    if expected_sr and sample_rate != expected_sr:
        errors.append(f"sample_rate {sample_rate} != Live {expected_sr}")
    if duration < max(0.5, expected_duration * 0.75):
        errors.append(
            f"duration {duration:.3f}s shorter than expected {expected_duration:.3f}s"
        )
    if require_signal:
        if rms <= SILENCE_RMS:
            errors.append(f"silent unexpected capture rms={rms}")
        if peak <= SILENCE_PEAK:
            errors.append(f"silent unexpected capture peak={peak}")
        # per-channel: a silent channel is a routing failure, not a quiet take
        for index, (ch_rms, ch_peak) in enumerate(zip(channel_rms, channel_peak)):
            if ch_rms <= SILENCE_RMS or ch_peak <= SILENCE_PEAK:
                errors.append(
                    f"channel {index} is silent (rms={ch_rms:.2e}, peak={ch_peak:.2e})")
        if channels >= 2 and stereo_correlation is None:
            errors.append(
                "a channel is constant: stereo correlation is undefined, capture routing suspect")
    return WavCheck(
        exists=True,
        size_bytes=size,
        channels=channels,
        sample_rate=int(sample_rate),
        duration=duration,
        rms=rms,
        peak=peak,
        subtype=subtype,
        ok=not errors,
        errors=errors,
        finite_samples=finite,
        channel_rms=channel_rms,
        channel_peak=channel_peak,
        stereo_correlation=stereo_correlation,
        frames=frames,
    )


def beats_to_seconds(beats: float, tempo: float) -> float:
    if tempo <= 0:
        raise AudioCaptureError("TEMPO_AUTOMATION_UNSUPPORTED", f"invalid tempo {tempo}")
    return float(beats) * 60.0 / float(tempo)


def assert_constant_tempo(
    daw: AbletonTcpAdapter, start_beat: float, end_beat: float
) -> float:
    saved = _transport_snapshot(daw)
    probes = [start_beat, (start_beat + end_beat) / 2.0, end_beat]
    tempos: list[float] = []
    try:
        for beat in probes:
            daw.set_current_song_time(beat)
            time.sleep(0.02)
            tempos.append(float(daw.get_playback_position().get("tempo") or 0.0))
    finally:
        daw.set_current_song_time(float(saved["current_song_time"]))
    if any(tempo <= 0 for tempo in tempos):
        raise AudioCaptureError("TEMPO_AUTOMATION_UNSUPPORTED", f"tempo probes={tempos}")
    if max(tempos) - min(tempos) > TEMPO_EPS_BPM:
        raise AudioCaptureError(
            "TEMPO_AUTOMATION_UNSUPPORTED",
            f"tempo changes across region: {tempos}",
        )
    return tempos[0]


def find_onset_samples(data: np.ndarray, threshold: float = ONSET_PEAK) -> int | None:
    mono = np.mean(data, axis=1) if data.ndim == 2 else data
    hits = np.where(np.abs(mono) > threshold)[0]
    if len(hits) == 0:
        return None
    return int(hits[0])


def write_analysis_wav(
    raw_path: Path,
    dest_path: Path,
    *,
    start_beat: float,
    end_beat: float,
    tempo: float,
    play_offset_seconds: float,
    align: str = "transport",
) -> dict[str, float | int]:
    data, sample_rate = sf.read(str(raw_path), always_2d=True)
    onset = find_onset_samples(data)
    onset_offset = None if onset is None else onset / float(sample_rate)
    if align == "onset" and onset_offset is not None:
        play_offset_seconds = onset_offset
    start_s = play_offset_seconds + beats_to_seconds(start_beat, tempo)
    end_s = play_offset_seconds + beats_to_seconds(end_beat, tempo)
    start_i = max(0, int(round(start_s * sample_rate)))
    end_i = min(len(data), int(round(end_s * sample_rate)))
    missing_s = end_s - (end_i / float(sample_rate))
    if missing_s > 0.002:
        raise AudioCaptureError(
            "SHORT_CAPTURE",
            f"raw ends at {len(data) / sample_rate:.3f}s, need {end_s:.3f}s",
        )
    if end_i <= start_i:
        raise AudioCaptureError("SHORT_CAPTURE", "empty analysis region after trim")
    trimmed = data[start_i:end_i]
    subtype = "FLOAT"
    with sf.SoundFile(str(raw_path)) as handle:
        subtype = handle.subtype
    sf.write(str(dest_path), trimmed, sample_rate, subtype=subtype)
    return {
        "trim_start_samples": start_i,
        "trim_end_samples": end_i,
        "analysis_duration": len(trimmed) / float(sample_rate),
        "sample_rate": int(sample_rate),
        "channels": int(trimmed.shape[1]),
        "play_offset_seconds": play_offset_seconds,
        "onset_samples": -1 if onset is None else onset,
        "onset_offset_seconds": -1.0 if onset_offset is None else float(onset_offset),
        "align": align,
    }


def _mixer_snapshot(
    daw: AbletonTcpAdapter,
    infos: dict[int, dict[str, object]] | None = None,
) -> list[dict[str, object]]:
    if infos is not None:
        return mixer_from_track_infos(infos)
    count = int(daw.health().get("track_count") or 0)
    states: list[dict[str, object]] = []
    for index in range(count):
        info = daw.get_track_info(index)
        states.append(
            {
                "index": index,
                "name": info.get("name"),
                "mute": bool(info.get("mute", False)),
                "solo": bool(info.get("solo", False)),
                "arm": bool(info.get("arm", False)),
                "monitoring": info.get("monitoring"),
            }
        )
    return states


def _restore_mixer(
    daw: AbletonTcpAdapter,
    before: list[dict[str, object]],
    *,
    touched: bool = True,
) -> list[str]:
    # Restore only when the mutation plan actually changed mixer.
    if not touched:
        return []
    errors: list[str] = []
    for state in before:
        index = int(state["index"])
        info = daw.get_track_info(index)
        if bool(info.get("solo", False)) != bool(state["solo"]):
            daw.set_track_solo(index, bool(state["solo"]))
        if bool(info.get("mute", False)) != bool(state["mute"]):
            daw.set_track_mute(index, bool(state["mute"]))
        if bool(info.get("arm", False)) != bool(state["arm"]):
            try:
                daw.set_track_arm(index, bool(state["arm"]))
            except DawError:
                errors.append(f"arm unrestored track {index}")
        want_mon = state.get("monitoring")
        if want_mon and info.get("monitoring") and info.get("monitoring") != want_mon:
            try:
                daw.set_track_monitoring(index, str(want_mon))
            except DawError:
                errors.append(f"monitoring unrestored track {index}")
        check = daw.get_track_info(index)
        if bool(check.get("solo", False)) != bool(state["solo"]):
            errors.append(f"solo unrestored track {index}")
        if bool(check.get("mute", False)) != bool(state["mute"]):
            errors.append(f"mute unrestored track {index}")
        if bool(check.get("arm", False)) != bool(state["arm"]):
            errors.append(f"arm unrestored track {index}")
    return errors


def _exclusive_solo(daw: AbletonTcpAdapter, track_index: int) -> None:
    for state in _mixer_snapshot(daw):
        want = int(state["index"]) == track_index
        if bool(state["solo"]) != want:
            daw.set_track_solo(int(state["index"]), want)


def _fire_session_clips(daw: AbletonTcpAdapter, track_indexes: list[int]) -> dict[str, Any]:
    if not track_indexes:
        return {"ok": True, "mode": "none", "results": []}
    clips = [
        {"id": str(index), "track_index": int(index), "clip_index": 0}
        for index in track_indexes
    ]
    if not _batch_control_available(daw):
        for index in track_indexes:
            daw.fire_clip(index, 0)
        return {"ok": True, "mode": "serial", "results": clips}
    try:
        result = daw.fire_clips(clips)
        _note_batch_control(daw, True)
        mode = "batch"
    except DawError as exc:
        if not _unknown_command(exc):
            raise
        _note_batch_control(daw, False)
        for index in track_indexes:
            daw.fire_clip(index, 0)
        return {"ok": True, "mode": "serial", "results": clips}
    if not result.get("ok"):
        failed = [row for row in result.get("results") or [] if not row.get("ok")]
        raise AudioCaptureError("TRANSPORT_FAILED", "partial fire: " + str(failed))
    return {**result, "mode": mode}


def _stop_session_clips(daw: AbletonTcpAdapter, track_indexes: list[int]) -> dict[str, Any]:
    if not track_indexes:
        return {"ok": True, "mode": "none", "results": []}
    clips = [
        {"id": str(index), "track_index": int(index), "clip_index": 0}
        for index in track_indexes
    ]
    if not _batch_control_available(daw):
        for index in track_indexes:
            try:
                daw.stop_clip(index, 0)
            except DawError:
                pass
        return {"ok": True, "mode": "serial", "results": clips}
    try:
        result = daw.stop_clips(clips)
        _note_batch_control(daw, True)
        mode = "batch"
    except DawError as exc:
        if not _unknown_command(exc):
            raise
        _note_batch_control(daw, False)
        for index in track_indexes:
            try:
                daw.stop_clip(index, 0)
            except DawError:
                pass
        return {"ok": True, "mode": "serial", "results": clips}
    if not result.get("ok"):
        failed = [row for row in result.get("results") or [] if not row.get("ok")]
        raise AudioCaptureError("TRANSPORT_FAILED", "partial stop: " + str(failed))
    return {**result, "mode": mode}


def _playing_clip_tracks(
    daw: AbletonTcpAdapter, track_indexes: list[int]
) -> list[int]:
    playing: list[int] = []
    for index in track_indexes:
        info = daw.last_track_infos.get(int(index)) or {}
        slots = info.get("clip_slots") or []
        clip = (slots[0].get("clip") if slots else None) or {}
        if clip.get("is_playing") or clip.get("is_triggered"):
            playing.append(int(index))
    return playing


def _wav_poll_status(
    path: Path,
    *,
    min_duration_s: float | None,
    expected_sr: int | None = None,
    expected_channels: int = 2,
) -> dict[str, object]:
    """Header/frame probe. Duration reached is not sufficient by itself."""
    if not path.is_file():
        return {"candidate": False, "reason": "missing", "frames": None, "size": 0}
    size = path.stat().st_size
    if size <= 0:
        return {"candidate": False, "reason": "empty", "frames": None, "size": 0}
    try:
        info = sf.info(str(path))
    except (OSError, RuntimeError, ValueError) as exc:
        return {
            "candidate": False,
            "reason": f"header:{exc}",
            "frames": None,
            "size": size,
        }
    frames = int(info.frames)
    if int(info.channels) != expected_channels:
        return {
            "candidate": False,
            "reason": f"channels {info.channels}",
            "frames": frames,
            "size": size,
        }
    if expected_sr and int(info.samplerate) != expected_sr:
        return {
            "candidate": False,
            "reason": f"sr {info.samplerate}",
            "frames": frames,
            "size": size,
        }
    duration = frames / float(info.samplerate) if info.samplerate else 0.0
    if min_duration_s is not None and duration + 0.05 < min_duration_s:
        return {
            "candidate": False,
            "reason": "short",
            "frames": frames,
            "size": size,
            "duration": duration,
        }
    return {
        "candidate": True,
        "frames": frames,
        "size": size,
        "duration": duration,
        "channels": int(info.channels),
        "samplerate": int(info.samplerate),
    }


def _wait_for_wav(
    path: Path,
    timeout: float,
    *,
    min_duration_s: float | None = None,
    expected_sr: int | None = None,
    expected_channels: int = 2,
) -> None:
    deadline = time.time() + timeout
    last_frames: int | None = None
    stable = 0
    last_status: dict[str, object] = {}
    while time.time() < deadline:
        status = _wav_poll_status(
            path,
            min_duration_s=min_duration_s,
            expected_sr=expected_sr,
            expected_channels=expected_channels,
        )
        last_status = status
        frames = status.get("frames")
        if status.get("candidate") and isinstance(frames, int) and frames > 0:
            if frames == last_frames:
                stable += 1
                if stable >= 2:
                    try:
                        data, _sr = sf.read(str(path), always_2d=True)
                    except (OSError, RuntimeError, ValueError):
                        stable = 0
                        last_frames = None
                        time.sleep(0.05)
                        continue
                    if not bool(np.isfinite(data).all()):
                        raise AudioCaptureError(
                            "CAPTURE_QUALITY_WARNING",
                            f"non-finite samples while finalizing {path}",
                        )
                    return
            else:
                last_frames = frames
                stable = 0
        else:
            last_frames = frames if isinstance(frames, int) else None
            stable = 0
        time.sleep(0.05)
    if not path.is_file():
        raise AudioCaptureError("CANNOT_CREATE_OUTPUT", f"staging wav missing: {path}")
    if path.stat().st_size <= 0:
        raise AudioCaptureError("ZERO_BYTE", f"staging wav is empty: {path}")
    reason = last_status.get("reason") or "unstable frames"
    raise AudioCaptureError(
        "SHORT_CAPTURE",
        f"wav not finalized: {reason} frames={last_status.get('frames')}",
    )


def _transport_snapshot(daw: AbletonTcpAdapter) -> dict[str, object]:
    pos = daw.get_playback_position()
    return {
        "is_playing": bool(pos.get("is_playing")),
        "current_song_time": float(pos.get("current_song_time", 0.0)),
        "loop": bool(pos.get("loop", False)),
        "loop_start": float(pos.get("loop_start", 0.0)),
        "loop_length": float(pos.get("loop_length", 0.0)),
        "tempo": float(pos.get("tempo", 120.0)),
        "sample_rate": pos.get("sample_rate"),
        "signature_numerator": int(pos.get("signature_numerator", 4)),
        "signature_denominator": int(pos.get("signature_denominator", 4)),
    }


def _restore_transport(
    daw: AbletonTcpAdapter,
    saved: dict[str, object],
    source_track: int | None,
    *,
    strict: bool = True,
) -> dict[str, object]:
    mutations: list[str] = []
    try:
        daw.stop_playback()
        mutations.append("stop_playback")
        if source_track is not None:
            try:
                daw.stop_clip(source_track, 0)
                mutations.append(f"stop_clip:{source_track}")
            except DawError:
                pass
        target_time = float(saved["current_song_time"])
        want_playing = bool(saved["is_playing"])
        daw.set_current_song_time(target_time)
        mutations.append("set_current_song_time")
        loop_length = float(saved["loop_length"])
        if loop_length > 0:
            daw.set_arrangement_loop(
                float(saved["loop_start"]),
                float(saved["loop_start"]) + loop_length,
                bool(saved["loop"]),
            )
            mutations.append("set_arrangement_loop")
        if want_playing:
            daw.start_playback()
            mutations.append("start_playback")
    except DawError:
        try:
            daw.stop_playback()
            mutations.append("stop_playback_fallback")
        except DawError:
            pass
    readback = daw.get_playback_position()
    time_error = abs(
        float(readback.get("current_song_time", 0.0)) - float(saved["current_song_time"])
    )
    playing_ok = bool(readback.get("is_playing")) == bool(saved["is_playing"])
    if time_error > SONG_TIME_RESTORE_TOLERANCE_BEATS:
        try:
            daw.set_current_song_time(float(saved["current_song_time"]))
            mutations.append("set_current_song_time_retry")
            readback = daw.get_playback_position()
            time_error = abs(
                float(readback.get("current_song_time", 0.0))
                - float(saved["current_song_time"])
            )
            playing_ok = bool(readback.get("is_playing")) == bool(saved["is_playing"])
        except DawError:
            pass
    ok = playing_ok and time_error <= SONG_TIME_RESTORE_TOLERANCE_BEATS
    result = {
        "ok": ok,
        "saved": {
            "is_playing": bool(saved["is_playing"]),
            "current_song_time": float(saved["current_song_time"]),
        },
        "readback": {
            "is_playing": bool(readback.get("is_playing")),
            "current_song_time": readback.get("current_song_time"),
        },
        "song_time_error_beats": time_error,
        "tolerance_beats": SONG_TIME_RESTORE_TOLERANCE_BEATS,
        "mutations": mutations,
        "transport_mutations": len(mutations),
    }
    if not ok and strict:
        raise AudioCaptureError(
            "STATE_RESTORE_FAILED",
            "transport read-back mismatch: " + str(result),
        )
    return result


def _session_fingerprint(
    daw: AbletonTcpAdapter,
    target_index: int | None,
    mixer: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    pos = daw.get_playback_position()
    mixer = mixer if mixer is not None else _mixer_snapshot(daw)
    target_name = None
    if target_index is not None:
        for state in mixer:
            if int(state["index"]) == target_index:
                target_name = state.get("name")
                break
    return {
        "tempo": round(float(pos.get("tempo") or 0.0), 4),
        "signature": [
            int(pos.get("signature_numerator", 4)),
            int(pos.get("signature_denominator", 4)),
        ],
        "track_count": len(mixer),
        "mixer": [
            {
                "index": s["index"],
                "name": s["name"],
                "mute": s["mute"],
                "solo": s["solo"],
            }
            for s in mixer
        ],
        "target_index": target_index,
        "target_name": target_name,
    }


def _observe_session_end(daw: AbletonTcpAdapter, session, *, mixer_mutated: bool):
    """Validate revision without a full N+1 snapshot when mixer was not mutated."""
    info = daw.get_session_info()
    track_count = int(info.get("track_count", -1))
    tempo = float(info.get("tempo") or 0.0)
    same_count = track_count == len(session.tracks)
    same_tempo = abs(tempo - float(session.transport.tempo)) <= TEMPO_EPS_BPM
    same_sig = int(info.get("signature_numerator", 0)) == session.transport.signature_numerator
    if same_count and same_tempo and same_sig and not mixer_mutated:
        return session
    return daw.snapshot(include_notes=False)


def _clips_from_info(info: dict[str, object] | None) -> bool:
    slots = (info or {}).get("clip_slots") or []
    return bool(slots and slots[0].get("has_clip"))


def capture_audio_segment(
    daw: AbletonTcpAdapter,
    source: str,
    start_beat: float,
    end_beat: float,
    *,
    require_signal: bool = True,
    fire_track: int | None = None,
    fire_tracks: list[int] | None = None,
    tap_track_index: int = MASTER_INDEX,
    isolate_solo: bool | None = None,
    fire_all: bool = False,
    observation_source: ObservationSource | None = None,
    capture_view: CaptureView | None = None,
    signal_point: SignalPoint | None = None,
    signal_point_label: str | None = None,
    preroll_beats: float = DEFAULT_PREROLL_BEATS,
    tail_policy: TailPolicy = TailPolicy.STRICT_REGION,
    context: CaptureContext | None = None,
    known_tempo: float | None = None,
    broadcast_udp: bool = True,
    staging_name: str = STAGING_NAME,
) -> AudioAsset:
    if end_beat <= start_beat:
        raise AudioCaptureError("SHORT_CAPTURE", "end_beat must be after start_beat")
    stages = StageTimer()
    if context is not None and tap_track_index == MASTER_INDEX and context.tap_device:
        tap = context.tap_device
        tap_before = context.master_info
    else:
        tap = find_tap(daw, tap_track_index)
        tap_before = (
            daw.get_master_info()
            if tap_track_index == MASTER_INDEX
            else daw.get_track_info(tap_track_index)
        )
    if tap is None:
        where = "Master" if tap_track_index == MASTER_INDEX else f"track {tap_track_index}"
        raise AudioCaptureError("TAP_MISSING", f"Copilot Audio Tap is not on {where}")
    if context is not None:
        mixer_before = list(context.mixer)
        session = context.session
    else:
        mixer_before = _mixer_snapshot(daw)
        stages.mark("mixer_snapshot")
        session = daw.snapshot(include_notes=False)
        stages.mark("session_snapshot")
    revision_start = session.revision
    source_type = "MASTER"
    source_stable_id: str | None = None
    fire_indexes: list[int] = []
    saved: dict[str, object] | None = None
    infos = context.track_infos if context is not None else None
    if source == "MASTER":
        if fire_tracks:
            fire_indexes = list(fire_tracks)
        elif fire_all or fire_track is None:
            for state in mixer_before:
                index = int(state["index"])
                info = infos.get(index) if infos is not None else daw.get_track_info(index)
                if _clips_from_info(info):
                    fire_indexes.append(index)
        else:
            fire_indexes = [fire_track]
    else:
        try:
            track = session.track_by_id(source)
        except KeyError as exc:
            raise AudioCaptureError(
                "TARGET_AMBIGUOUS", f"unknown track stable_id {source}"
            ) from exc
        source_type = "TRACK"
        source_stable_id = track.stable_id
        if fire_tracks:
            fire_indexes = list(fire_tracks)
        elif fire_all:
            for state in mixer_before:
                index = int(state["index"])
                info = infos.get(index) if infos is not None else daw.get_track_info(index)
                if _clips_from_info(info):
                    fire_indexes.append(index)
        else:
            fire_indexes = [track.index]
    windows = region_windows(
        start_beat,
        end_beat,
        preroll_beats=preroll_beats,
        tail_policy=tail_policy,
    )
    if observation_source is None:
        observation_source = (
            ObservationSource.MASTER_CONTEXT
            if source_type == "MASTER"
            else ObservationSource.TRACK_ISOLATED
        )
    if signal_point is None:
        signal_point = (
            SignalPoint.MASTER if source_type == "MASTER" else SignalPoint.UNKNOWN
        )
    use_solo = isolate_solo
    if use_solo is None:
        use_solo = source_type == "TRACK" and not fire_all
    # Exclusive solo is the only mixer mutation this function performs.
    mixer_mutated = bool(use_solo)

    analysis_path = unique_capture_path()
    capture_id = analysis_path.stem.removeprefix("capture_")
    raw_path = capture_dir() / f"capture_{capture_id}_raw.wav"
    _clear_staging(staging_name)
    stage = staging_path(staging_name)
    t_play: float | None = None
    transport_start = None
    transport_stop = None
    t_rec_stop = 0.0
    tempo = 0.0
    expected_duration = 0.0
    live_sr_int: int | None = None
    asset: AudioAsset | None = None
    try:
        if mixer_mutated and source_type == "TRACK" and source_stable_id:
            _exclusive_solo(daw, session.track_by_id(source_stable_id).index)
        stages.mark("transport_setup")
        fingerprint_start = _session_fingerprint(
            daw,
            None if source_type == "MASTER" else session.track_by_id(source).index,
            mixer=mixer_before,
        )
        stages.mark("fingerprint")
        if known_tempo is not None:
            tempo = float(known_tempo)
            saved = _transport_snapshot(daw)
        else:
            tempo = assert_constant_tempo(daw, start_beat, end_beat)
            saved = _transport_snapshot(daw)
        stages.mark("tempo_probes")
        expected_duration = beats_to_seconds(end_beat - start_beat, tempo)
        live_sr = saved.get("sample_rate") or (context.sample_rate if context else None)
        live_sr_int = int(live_sr) if live_sr else None
        set_tap_recording(
            daw,
            False,
            tap_track_index,
            context=context,
            broadcast_udp=broadcast_udp,
        )
        time.sleep(0.1)
        _clear_staging(staging_name)
        daw.stop_playback()
        for index in fire_indexes:
            try:
                daw.stop_clip(index, 0)
            except DawError:
                pass
        daw.set_current_song_time(0.0)
        set_tap_recording(
            daw,
            True,
            tap_track_index,
            context=context,
            broadcast_udp=broadcast_udp,
        )
        time.sleep(0.08)
        if fire_indexes:
            _fire_session_clips(daw, fire_indexes)
        else:
            daw.start_playback()
        deadline = time.time() + 2.0
        while time.time() < deadline:
            pos = daw.get_playback_position()
            if pos.get("is_playing"):
                t_play = time.time()
                transport_start = float(pos.get("current_song_time", 0.0))
                break
            time.sleep(0.02)
        stages.mark("pre_roll")
        if t_play is None:
            raise AudioCaptureError("SHORT_CAPTURE", "transport did not start")
        wait_s = beats_to_seconds(end_beat, tempo) + 0.20
        time.sleep(max(0.0, wait_s - (time.time() - t_play)))
        stages.mark("record")
        transport_stop = float(daw.get_playback_position().get("current_song_time", 0.0))
        set_tap_recording(
            daw,
            False,
            tap_track_index,
            context=context,
            broadcast_udp=broadcast_udp,
        )
        t_rec_stop = time.time()
        time.sleep(0.15)
        daw.stop_playback()
        for index in fire_indexes:
            try:
                daw.stop_clip(index, 0)
            except DawError:
                pass
        stages.mark("recorder_stop")
        stages.marks["stop"] = stages.marks["recorder_stop"]
        _wait_for_wav(
            stage,
            timeout=8.0,
            min_duration_s=expected_duration,
            expected_sr=live_sr_int,
        )
        stages.mark("wav_finalize")
        if raw_path.exists() or analysis_path.exists():
            analysis_path = unique_capture_path()
            capture_id = analysis_path.stem.removeprefix("capture_")
            raw_path = capture_dir() / f"capture_{capture_id}_raw.wav"
        stage.replace(raw_path)
        raw_check = validate_wav(
            raw_path,
            expected_sr=live_sr_int,
            expected_duration=expected_duration,
            require_signal=False,
        )
        if not raw_check.exists or (raw_check.size_bytes or 0) <= 0:
            raise AudioCaptureError("ZERO_BYTE", str(raw_path))
        play_offset = 0.0
        if t_play is not None and raw_check.duration:
            t_rec_start = t_rec_stop - float(raw_check.duration)
            play_offset = max(0.0, t_play - t_rec_start)
        trim = write_analysis_wav(
            raw_path,
            analysis_path,
            start_beat=start_beat,
            end_beat=end_beat,
            tempo=tempo,
            play_offset_seconds=play_offset,
        )
        analysis_check = validate_wav(
            analysis_path,
            expected_sr=int(trim["sample_rate"]),
            expected_duration=expected_duration,
            require_signal=require_signal,
        )
        if abs(float(analysis_check.duration or 0.0) - expected_duration) > DURATION_TOLERANCE_S:
            raise AudioCaptureError(
                "SHORT_CAPTURE",
                f"analysis duration {analysis_check.duration:.4f}s "
                f"!= {expected_duration:.4f}s ±{DURATION_TOLERANCE_S}s",
            )
        if require_signal and any("silent" in err for err in analysis_check.errors):
            raise AudioCaptureError("SILENT_UNEXPECTED", "; ".join(analysis_check.errors))
        if not analysis_check.finite_samples:
            raise AudioCaptureError("CAPTURE_QUALITY_WARNING", "NaN/Inf in analysis WAV")
        stages.mark("wav_validation")
        session_end = _observe_session_end(
            daw, session, mixer_mutated=mixer_mutated
        )
        fingerprint_end = (
            fingerprint_start
            if not mixer_mutated
            else _session_fingerprint(
                daw,
                None if source_type == "MASTER" else session.track_by_id(source).index,
                mixer=mixer_before,
            )
        )
        quality = "OK"
        if fingerprint_end != fingerprint_start:
            quality = "CAPTURE_MUTATED"
        expected_frames = int(round(expected_duration * int(analysis_check.sample_rate or 0)))
        actual_frames = analysis_check.frames
        onset_off = float(trim.get("onset_offset_seconds") or -1)
        if (
            quality == "OK"
            and onset_off >= 0
            and abs(onset_off - play_offset) > 0.15
        ):
            quality = "CAPTURE_QUALITY_WARNING"
        if (
            quality == "OK"
            and actual_frames is not None
            and abs(actual_frames - expected_frames) > max(1, int((analysis_check.sample_rate or 44100) * 0.002))
        ):
            quality = "CAPTURE_QUALITY_WARNING"
        stages.mark("session_revision_check")
        stages.marks["file_finalize"] = (
            stages.marks.get("wav_finalize", 0.0)
            + stages.marks.get("wav_validation", 0.0)
            + stages.marks.get("session_revision_check", 0.0)
        )
        asset = AudioAsset(
            capture_id=capture_id,
            raw_file_path=str(raw_path),
            analysis_file_path=str(analysis_path),
            file_path=str(analysis_path),
            source="MASTER" if source_type == "MASTER" else source_stable_id or source,
            source_type=source_type,
            source_stable_id=source_stable_id,
            requested_start_beat=start_beat,
            requested_end_beat=end_beat,
            start_beat=start_beat,
            end_beat=end_beat,
            sample_rate=int(analysis_check.sample_rate or 0),
            channels=int(analysis_check.channels or 0),
            raw_duration=float(raw_check.duration or 0.0),
            analysis_duration=float(analysis_check.duration or 0.0),
            duration=float(analysis_check.duration or 0.0),
            session_revision=session.revision,
            tempo=tempo,
            signature_numerator=int(saved["signature_numerator"]),
            signature_denominator=int(saved["signature_denominator"]),
            transport_start_observed=transport_start,
            transport_stop_observed=transport_stop,
            trim_start_samples=int(trim["trim_start_samples"]),
            trim_end_samples=int(trim["trim_end_samples"]),
            rms=analysis_check.rms,
            peak=analysis_check.peak,
            size_bytes=analysis_check.size_bytes,
            capture_view=capture_view
            or (
                CaptureView.MASTER_CONTEXT
                if observation_source is ObservationSource.MASTER_CONTEXT
                else CaptureView.TRACK_ISOLATED
                if observation_source is ObservationSource.TRACK_ISOLATED
                else CaptureView.TRACK_CONTEXT_REMOVAL
                if observation_source
                in {
                    ObservationSource.TRACK_IN_MIX_CONTEXT,
                    ObservationSource.TRACK_CONTEXT_REMOVAL,
                }
                else None
            ),
            observation_source=observation_source,
            signal_point=signal_point,
            signal_point_label=signal_point_label,
            tail_policy=tail_policy,
            request_start_beat=float(windows["request_start_beat"]),
            request_end_beat=float(windows["request_end_beat"]),
            capture_start_beat=float(windows["capture_start_beat"]),
            capture_end_beat=float(windows["capture_end_beat"]),
            analysis_start_beat=float(windows["analysis_start_beat"]),
            analysis_end_beat=float(windows["analysis_end_beat"]),
            preroll_beats=float(windows["preroll_beats"]),
            session_revision_at_start=revision_start,
            session_revision_at_end=session_end.revision,
            capture_quality=quality,
            finite_samples=bool(analysis_check.finite_samples),
            expected_frames=expected_frames,
            actual_frames=actual_frames,
            stage_timings=dict(stages.marks),
        )
        return asset
    except (ConnectionError, OSError, TimeoutError) as exc:
        raise AudioCaptureError("REMOTE_DISCONNECT", str(exc)) from exc
    finally:
        t_restore = time.perf_counter()
        restore_errors = _restore_mixer(daw, mixer_before, touched=mixer_mutated)
        if saved is not None:
            restore = _restore_transport(
                daw, saved, fire_indexes[0] if fire_indexes else None, strict=False
            )
            if not restore.get("ok"):
                restore_errors.append("transport read-back mismatch")
        # Read-back: one Live query after capture, not a full mixer sweep.
        tap_after = (
            daw.get_master_info()
            if tap_track_index == MASTER_INDEX
            else daw.get_track_info(tap_track_index)
        )
        before_count = tap_before.get("device_count")
        if before_count is None:
            before_count = len(tap_before.get("devices") or [])
        after_count = tap_after.get("device_count")
        if after_count is None:
            after_count = len(tap_after.get("devices") or [])
        if after_count != before_count:
            restore_errors.append("tap host device_count changed")
        if tap_from_info(tap_after) is None:
            restore_errors.append("tap missing after capture")
        try:
            set_tap_recording(
                daw,
                False,
                tap_track_index,
                context=context,
                broadcast_udp=broadcast_udp,
            )
        except DawError:
            _send_tap_udp(0)
        if asset is not None:
            restore_s = time.perf_counter() - t_restore
            asset.stage_timings["state_restore"] = restore_s
            asset.stage_timings["readback_verification"] = restore_s
        if restore_errors:
            raise AudioCaptureError(
                "REMOTE_DISCONNECT",
                "project state not restored: " + "; ".join(restore_errors),
            )


def capture_master_segment(
    daw: AbletonTcpAdapter,
    start_beat: float,
    end_beat: float,
    *,
    source_track: int | None = None,
    require_signal: bool = True,
) -> AudioAsset:
    return capture_audio_segment(
        daw,
        "MASTER",
        start_beat,
        end_beat,
        require_signal=require_signal,
        fire_track=source_track,
    )


def observe_asset(asset: AudioAsset, region: str, tempo: float) -> MusicObservation:
    path = asset.analysis_file_path or asset.file_path
    data, sample_rate = sf.read(path, always_2d=True)
    samples = data.T
    label = asset.source_type if asset.source_type else "MASTER"
    return measure_audio(
        samples,
        int(sample_rate),
        source=f"LIVE_{label}:{asset.capture_id}",
        region=region,
        tempo_hint_bpm=tempo,
        capture_view=asset.capture_view,
        observation_source=asset.observation_source,
        signal_point=asset.signal_point,
        signal_point_label=asset.signal_point_label,
        tail_policy=asset.tail_policy,
    )


def ensure_internal_source(daw: AbletonTcpAdapter) -> dict[str, object]:
    session = daw.snapshot(include_notes=False)
    existing = session.track_by_name(SOURCE_TRACK_NAME)
    if existing is not None and existing.clips:
        daw.replace_clip_notes(existing.index, 0, _region_marker_notes())
        session = daw.snapshot(include_notes=False)
        track = session.track_by_name(SOURCE_TRACK_NAME)
        return {
            "created": False,
            "track_index": existing.index,
            "track_name": existing.name,
            "stable_id": track.stable_id if track else "",
        }
    if existing is None:
        created = daw.create_midi_track(SOURCE_TRACK_NAME, -1)
        track_index = int(created["index"])
    else:
        track_index = existing.index
    _load_stock_instrument(daw, track_index)
    info = daw.get_track_info(track_index)
    slots = info.get("clip_slots") or []
    has_clip = bool(slots and slots[0].get("has_clip"))
    if not has_clip:
        daw.create_midi_clip(track_index, 0, 16.0)
        daw.set_clip_name(track_index, 0, SOURCE_CLIP_NAME)
    daw.replace_clip_notes(track_index, 0, _region_marker_notes())
    session = daw.snapshot(include_notes=False)
    track = session.track_by_name(SOURCE_TRACK_NAME)
    return {
        "created": True,
        "track_index": track_index,
        "track_name": SOURCE_TRACK_NAME,
        "stable_id": track.stable_id if track else "",
    }


def _region_marker_notes() -> list[MidiNote]:
    return [
        MidiNote(pitch=REGION_LOW_PITCH, start_time=0.0, duration=8.0, velocity=110),
        MidiNote(pitch=REGION_HIGH_PITCH, start_time=8.0, duration=8.0, velocity=110),
    ]


def ensure_silent_track(daw: AbletonTcpAdapter) -> dict[str, object]:
    session = daw.snapshot(include_notes=False)
    existing = session.track_by_name(SILENT_TRACK_NAME)
    if existing is None:
        created = daw.create_midi_track(SILENT_TRACK_NAME, -1)
        track_index = int(created["index"])
        daw.create_midi_clip(track_index, 0, 16.0)
        daw.set_clip_name(track_index, 0, "Silence")
        daw.replace_clip_notes(track_index, 0, [])
        created_flag = True
    else:
        track_index = existing.index
        created_flag = False
    session = daw.snapshot(include_notes=False)
    track = session.track_by_name(SILENT_TRACK_NAME)
    return {
        "created": created_flag,
        "track_index": track_index,
        "track_name": SILENT_TRACK_NAME,
        "stable_id": track.stable_id if track else "",
    }


def _load_stock_instrument(daw: AbletonTcpAdapter, track_index: int) -> dict[str, object]:
    info = daw.get_track_info(track_index)
    for device in info.get("devices") or []:
        class_name = str(device.get("class_name") or "")
        if class_name and class_name != "MidiEffectGroupDevice":
            return {"already_present": True, "device": device}
    for query, category in (
        ("Drift", "instruments"),
        ("Analog", "instruments"),
        ("Operator", "instruments"),
        ("Grand Piano", "instruments"),
    ):
        found = daw.search_browser(query, category)
        for item in found.get("results") or []:
            if not item.get("is_loadable"):
                continue
            uri = item.get("uri")
            if not uri:
                continue
            try:
                loaded = daw.load_browser_item(track_index, str(uri))
            except DawError:
                loaded = daw.load_instrument_or_effect(track_index, str(uri))
            if not loaded.get("error"):
                return {"loaded": loaded, "query": query}
    raise AudioCaptureError(
        "TAP_MISSING",
        "could not load a stock Live instrument for the test source",
    )


def four_bar_region(daw: AbletonTcpAdapter) -> tuple[float, float, float]:
    pos = daw.get_playback_position()
    tempo = float(pos.get("tempo") or 120.0)
    num = int(pos.get("signature_numerator") or 4)
    den = int(pos.get("signature_denominator") or 4)
    length = bars_to_beats(4, num, den)
    return 0.0, length, tempo
