from __future__ import annotations

import inspect
import json
import os
import socket
from collections import Counter
from pathlib import Path
from typing import Any
from uuid import uuid4

from copilot.daw.adapter import DawAdapter, DawError
from copilot.daw.identities import IdentityRegistry
from copilot.daw.protocol import (
    CAPABILITIES,
    ProtocolError,
    handshake_payload,
    require_capability,
    require_local_host,
)
from copilot.daw.state_hash import ObservedRevision
from copilot.daw.timeouts import DEFAULT_TIMEOUTS, TimeoutPolicy
from copilot.daw.write import WriteInDoubt
from copilot.schemas.session import (
    ClipState,
    DeviceState,
    DeviceParameter,
    MidiNote,
    MixerState,
    RoutingState,
    SendState,
    SessionState,
    TrackState,
    TransportState,
)

DEFAULT_HOST = os.environ.get("ABLETON_MCP_HOST", "127.0.0.1")
DEFAULT_PORT = int(os.environ.get("ABLETON_MCP_PORT", "9877"))


def _is_transport_failure(exc: BaseException) -> bool:
    message = str(exc)
    return any(
        token in message
        for token in (
            "Timeout waiting for Ableton",
            "socket disconnect",
            "BLOCKED_BY_ENVIRONMENT",
        )
    )


class AbletonTcpAdapter(DawAdapter):
    """Typed wrapper around the jpoindexter/ableton-mcp Remote Script protocol."""

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        timeouts: TimeoutPolicy | None = None,
    ) -> None:
        require_local_host(host)
        self.host = host
        self.port = port
        self._device_uri_cache: dict[str, str] = {}
        self.timeouts = timeouts or DEFAULT_TIMEOUTS
        self._sock: socket.socket | None = None
        self.ids = IdentityRegistry()
        self.observed = ObservedRevision()
        self.session_incarnation_id = f"sess_{uuid4().hex[:12]}"
        self.capabilities: set[str] = set()
        self.handshake_info: dict[str, Any] = {}
        self.strict_capabilities = True
        self._recv_buf = ""
        self.tcp_counts: Counter[str] = Counter()
        self.snapshot_calls = 0
        self.last_track_infos: dict[int, dict[str, Any]] = {}
        self.last_master_info: dict[str, Any] | None = None
        self.last_playback: dict[str, Any] | None = None
        self.last_return_tracks: dict[str, Any] | None = None
        self.last_topology: dict[str, Any] | None = None
        self.last_project: dict[str, Any] | None = None
        self._project_cached: dict[str, Any] | None = None
        self.snapshot_source: str | None = None
        self.profile_track_info = False
        self.track_info_sites: Counter[str] = Counter()
        self.batch_control_live: bool | None = None

    def reset_tcp_stats(self) -> None:
        self.tcp_counts = Counter()
        self.snapshot_calls = 0
        self.track_info_sites = Counter()

    def tcp_stats(self) -> dict[str, Any]:
        return {
            "total": int(sum(self.tcp_counts.values())),
            "by_type": dict(self.tcp_counts.most_common()),
            "full_session_snapshots": int(self.snapshot_calls),
            "get_track_info": int(self.tcp_counts.get("get_track_info", 0)),
            "get_tracks_info": int(self.tcp_counts.get("get_tracks_info", 0)),
            "get_capture_topology": int(self.tcp_counts.get("get_capture_topology", 0)),
            "snapshot_source": self.snapshot_source,
            "get_track_info_sites": dict(self.track_info_sites.most_common()),
        }

    def connect(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(self.timeouts.connect)
        try:
            sock.connect((self.host, self.port))
        except OSError as exc:
            sock.close()
            raise DawError(
                "BLOCKED_BY_ENVIRONMENT: Ableton Remote Script is not reachable "
                f"at {self.host}:{self.port}"
            ) from exc
        self._sock = sock
        self._recv_buf = ""
        try:
            hello = self._command("protocol_hello", side_effect=False)
            self.handshake_info = hello
            self.capabilities = set(hello.get("capabilities") or [])
        except DawError as exc:
            if isinstance(exc, ProtocolError) or _is_transport_failure(exc):
                self.disconnect()
                raise
            self.handshake_info = {
                **handshake_payload(),
                "mode": "LEGACY",
            }
            self.capabilities = set(CAPABILITIES)

    def disconnect(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            finally:
                self._sock = None

    def health(self) -> dict[str, Any]:
        result = self._command("health_check")
        result["backend"] = "ableton-tcp"
        return result

    def probe(self) -> dict[str, Any]:
        health = self.health()
        session = self.snapshot()
        return {
            "verification_class": "LIVE_VERIFIED",
            "status": "connected",
            "backend": "ableton-tcp",
            "host": self.host,
            "port": self.port,
            "health": health,
            "tempo": session.transport.tempo,
            "playing": session.transport.playing,
            "signature": [
                session.transport.signature_numerator,
                session.transport.signature_denominator,
            ],
            "track_count": len(session.tracks),
            "tracks": [
                {
                    "stable_id": track.stable_id,
                    "index": track.index,
                    "name": track.name,
                    "role": track.role,
                    "clips": [
                        {
                            "stable_id": clip.stable_id,
                            "slot_index": clip.slot_index,
                            "name": clip.name,
                            "note_count": len(clip.notes),
                        }
                        for clip in track.clips
                    ],
                    "devices": [
                        {"stable_id": device.stable_id, "name": device.name}
                        for device in track.devices
                    ],
                }
                for track in session.tracks
            ],
        }

    def get_session_info(self) -> dict[str, Any]:
        return self._command("get_session_info")

    def _ensure_project(self) -> dict[str, Any]:
        if self.last_project and (
            self.last_project.get("path") or self.last_project.get("name")
        ):
            return self.last_project
        if self._project_cached is not None:
            self.last_project = self._project_cached
            return self._project_cached
        try:
            self._project_cached = self.get_session_path()
        except DawError:
            self._project_cached = {}
        self.last_project = self._project_cached
        return self.last_project

    def snapshot(self, *, include_notes: bool = True) -> SessionState:
        self.snapshot_calls += 1
        topology = self._safe_command("get_capture_topology", default=None)
        if isinstance(topology, dict) and isinstance(topology.get("tracks"), list):
            self.snapshot_source = "topology"
            self.last_topology = topology
            return self._session_from_topology(topology, include_notes=include_notes)
        batched = self._safe_command("get_tracks_info", default=None)
        session = self._command("get_session_info")
        playback = self._command("get_playback_position")
        if isinstance(batched, dict) and isinstance(batched.get("tracks"), list):
            self.snapshot_source = "tracks_info"
            infos = {int(item["index"]): item for item in batched["tracks"]}
            return self._session_from_infos(
                session, playback, infos, include_notes=include_notes
            )
        self.snapshot_source = "serial"
        names = self._safe_command("get_all_track_names", default={})
        track_count = int(session.get("track_count", 0))
        raw_names = names.get("names") if isinstance(names, dict) else None
        infos: dict[int, dict[str, Any]] = {}
        for index in range(track_count):
            info = self._command("get_track_info", {"track_index": index})
            if raw_names and index < len(raw_names) and not info.get("name"):
                info["name"] = raw_names[index]
            infos[index] = info
        return self._session_from_infos(
            session, playback, infos, include_notes=include_notes
        )

    def _session_from_topology(
        self, topology: dict[str, Any], *, include_notes: bool
    ) -> SessionState:
        session = topology.get("session") or {}
        playback = topology.get("playback") or {}
        self.last_master_info = topology.get("master")
        self.last_playback = playback
        returns = topology.get("return_tracks")
        self.last_return_tracks = returns if isinstance(returns, dict) else None
        project = topology.get("project")
        if isinstance(project, dict):
            self.last_project = project
        infos = {
            int(item["index"]): item
            for item in topology.get("tracks") or []
            if "index" in item
        }
        return self._session_from_infos(
            session, playback, infos, include_notes=include_notes
        )

    def _session_from_infos(
        self,
        session: dict[str, Any],
        playback: dict[str, Any],
        infos: dict[int, dict[str, Any]],
        *,
        include_notes: bool,
    ) -> SessionState:
        tracks: list[TrackState] = []
        clips: dict[int, list[ClipState]] = {}
        devices: dict[int, list[DeviceState]] = {}
        self.last_track_infos = dict(infos)
        self.last_playback = playback
        for index in sorted(infos):
            info = infos[index]
            role = "midi" if info.get("is_midi_track") else "audio"
            name = info.get("name") or f"Track {index}"
            tracks.append(
                TrackState(
                    stable_id=name,
                    index=index,
                    name=name,
                    role=role,
                    mixer=MixerState(
                        volume=float(info.get("volume", 0.85)),
                        pan=float(info.get("panning", 0.0)),
                        mute=bool(info.get("mute", False)),
                        solo=bool(info.get("solo", False)),
                        arm=bool(info.get("arm", False)),
                    ),
                    routing=RoutingState(
                        input_type=str(info.get("input_routing_type") or ""),
                        input_channel=str(info.get("input_routing_channel") or ""),
                        output_type=str(info.get("output_routing_type") or ""),
                        output_channel=str(info.get("output_routing_channel") or ""),
                        monitoring=str(info.get("monitoring") or ""),
                    ),
                    sends=[
                        SendState(
                            index=int(send.get("index", send.get("send_index", i))),
                            name=str(send.get("name") or ""),
                            value=float(send.get("value", send.get("level", 0.0)) or 0.0),
                        )
                        for i, send in enumerate(info.get("sends") or [])
                    ],
                    grouped=bool(info.get("is_grouped", False)),
                    foldable=bool(info.get("is_foldable", False)),
                )
            )
            clips[index] = []
            for slot in info.get("clip_slots", []):
                clip = slot.get("clip")
                if not slot.get("has_clip") or not clip:
                    continue
                notes: list[MidiNote] = []
                if include_notes and info.get("is_midi_track"):
                    try:
                        note_data = self._command(
                            "get_clip_notes",
                            {"track_index": index, "clip_index": slot["index"]},
                        )
                        notes = [MidiNote(**note) for note in note_data.get("notes", [])]
                    except DawError:
                        notes = []
                clips[index].append(
                    ClipState(
                        stable_id="",
                        slot_index=int(slot["index"]),
                        name=clip.get("name", ""),
                        length_beats=float(clip.get("length", 0.0)),
                        is_midi=bool(info.get("is_midi_track")),
                        notes=notes,
                        sample_uri=clip.get("sample_uri") or clip.get("sample_path") or None,
                    )
                )
            devices[index] = [
                DeviceState(
                    stable_id="",
                    index=int(device.get("index", 0)),
                    name=device.get("name", ""),
                    class_name=device.get("class_name", ""),
                    enabled=bool(device.get("enabled", device.get("is_active", True))),
                    parameters=[
                        DeviceParameter(
                            index=int(parameter.get("index", p_i)),
                            name=str(parameter.get("name") or ""),
                            value=float(parameter.get("value", 0.0) or 0.0),
                            min=float(parameter.get("min", 0.0) or 0.0),
                            max=float(parameter.get("max", 1.0) or 1.0),
                        )
                        for p_i, parameter in enumerate(device.get("parameters") or [])
                    ],
                )
                for device in info.get("devices", [])
            ]
        attached = self.ids.attach(tracks, clips, devices)
        project = self._ensure_project()
        built = SessionState(
            daw="ableton",
            connected=True,
            session_incarnation_id=self.session_incarnation_id,
            project_path=project.get("path") or None,
            project_name=project.get("name") or None,
            transport=TransportState(
                tempo=float(session.get("tempo", playback.get("tempo", 120.0))),
                signature_numerator=int(
                    session.get(
                        "signature_numerator",
                        playback.get("signature_numerator", 4),
                    )
                ),
                signature_denominator=int(
                    session.get(
                        "signature_denominator",
                        playback.get("signature_denominator", 4),
                    )
                ),
                playing=bool(playback.get("is_playing", False)),
                position_beats=float(playback.get("current_song_time", 0.0)),
            ),
            tracks=attached,
        )
        return self.observed.observe(built)

    def create_midi_track(self, name: str, index: int = -1) -> dict[str, Any]:
        created = self._command("create_midi_track", {"index": index}, side_effect=True)
        track_index = int(created["index"])
        # Remote Script ignores name; wrap that gap ourselves.
        if name:
            self._command(
                "set_track_name",
                {"track_index": track_index, "name": name},
                side_effect=True,
            )
            created["name"] = name
        return created

    def delete_track(self, track_index: int) -> dict[str, Any]:
        return self._command(
            "delete_track", {"track_index": track_index}, side_effect=True
        )

    def set_track_name(self, track_index: int, name: str) -> dict[str, Any]:
        return self._command(
            "set_track_name",
            {"track_index": track_index, "name": name},
            side_effect=True,
        )

    def set_mixer_volume(self, track_index: int, volume: float) -> dict[str, Any]:
        return self._command(
            "set_track_volume",
            {"track_index": track_index, "volume": volume},
            side_effect=True,
        )

    def create_midi_clip(
        self, track_index: int, clip_index: int, length_beats: float
    ) -> dict[str, Any]:
        return self._command(
            "create_clip",
            {
                "track_index": track_index,
                "clip_index": clip_index,
                "length": length_beats,
            },
            side_effect=True,
        )

    def delete_clip(self, track_index: int, clip_index: int) -> dict[str, Any]:
        return self._command(
            "delete_clip",
            {"track_index": track_index, "clip_index": clip_index},
            side_effect=True,
        )

    def set_clip_name(
        self, track_index: int, clip_index: int, name: str
    ) -> dict[str, Any]:
        return self._command(
            "set_clip_name",
            {"track_index": track_index, "clip_index": clip_index, "name": name},
            side_effect=True,
        )

    def replace_clip_notes(
        self, track_index: int, clip_index: int, notes: list[MidiNote]
    ) -> dict[str, Any]:
        payload = [note.model_dump() for note in notes]
        return self._command(
            "add_notes_to_clip",
            {
                "track_index": track_index,
                "clip_index": clip_index,
                "notes": payload,
            },
            side_effect=True,
        )

    def get_clip_notes(self, track_index: int, clip_index: int) -> dict[str, Any]:
        return self._command(
            "get_clip_notes",
            {"track_index": track_index, "clip_index": clip_index},
        )

    def set_device_parameter(
        self,
        track_index: int,
        device_index: int,
        parameter_index: int,
        value: float,
    ) -> dict[str, Any]:
        return self._command(
            "set_device_parameter",
            {
                "track_index": track_index,
                "device_index": device_index,
                "parameter_index": parameter_index,
                "value": value,
            },
            side_effect=True,
        )

    def set_device_parameters(self, items: list[dict[str, Any]]) -> dict[str, Any]:
        return self._command(
            "set_device_parameters",
            {"items": items},
            side_effect=True,
        )

    def get_master_info(self) -> dict[str, Any]:
        info = self._command("get_master_info")
        self.last_master_info = info
        return info

    def get_device_parameters(
        self, track_index: int, device_index: int
    ) -> dict[str, Any]:
        return self._command(
            "get_device_parameters",
            {"track_index": track_index, "device_index": device_index},
        )

    def get_device_parameter(
        self, track_index: int, device_index: int, parameter_index: int
    ) -> dict[str, Any]:
        try:
            return self._command(
                "get_device_parameter",
                {
                    "track_index": track_index,
                    "device_index": device_index,
                    "parameter_index": parameter_index,
                },
            )
        except DawError as exc:
            message = str(exc).lower()
            if "unknown command" not in message and "unsupported command" not in message:
                raise
            params = self.get_device_parameters(track_index, device_index)
            for item in params.get("parameters") or []:
                if int(item.get("index", -1)) == int(parameter_index):
                    return item
            raise

    def get_track_info(self, track_index: int) -> dict[str, Any]:
        if self.profile_track_info:
            frame = inspect.stack()[1]
            site = f"{Path(frame.filename).name}:{frame.lineno}:{frame.function}"
            self.track_info_sites[site] += 1
        info = self._command("get_track_info", {"track_index": track_index})
        self.last_track_infos[int(track_index)] = info
        return info

    def get_tracks_info(self, indices: list[int] | None = None) -> dict[str, Any]:
        params: dict[str, Any] = {}
        if indices is not None:
            params["indices"] = [int(index) for index in indices]
        result = self._command("get_tracks_info", params)
        for item in result.get("tracks") or []:
            if "index" in item:
                self.last_track_infos[int(item["index"])] = item
        return result

    def get_capture_topology(self) -> dict[str, Any]:
        result = self._command("get_capture_topology")
        self.last_topology = result
        if isinstance(result.get("master"), dict):
            self.last_master_info = result["master"]
        if isinstance(result.get("playback"), dict):
            self.last_playback = result["playback"]
        if isinstance(result.get("return_tracks"), dict):
            self.last_return_tracks = result["return_tracks"]
        for item in result.get("tracks") or []:
            if "index" in item:
                self.last_track_infos[int(item["index"])] = item
        return result

    def get_playback_position(self) -> dict[str, Any]:
        return self._command("get_playback_position")

    def set_track_mute(self, track_index: int, mute: bool) -> dict[str, Any]:
        return self._command(
            "set_track_mute",
            {"track_index": track_index, "mute": mute},
            side_effect=True,
        )

    def set_track_solo(self, track_index: int, solo: bool) -> dict[str, Any]:
        return self._command(
            "set_track_solo",
            {"track_index": track_index, "solo": solo},
            side_effect=True,
        )

    def start_playback(self) -> dict[str, Any]:
        return self._command("start_playback", side_effect=True)

    def stop_playback(self) -> dict[str, Any]:
        return self._command("stop_playback", side_effect=True)

    def set_current_song_time(self, time: float) -> dict[str, Any]:
        return self._command(
            "set_current_song_time",
            {"time": float(time)},
            side_effect=True,
        )

    def jump_to_time(self, time: float) -> dict[str, Any]:
        return self._command(
            "jump_to_time",
            {"time": float(time)},
            side_effect=True,
        )

    def scrub_by(self, delta: float) -> dict[str, Any]:
        return self._command(
            "scrub_by",
            {"delta": float(delta)},
            side_effect=True,
        )

    def continue_playing(self) -> dict[str, Any]:
        return self._command("continue_playing", side_effect=True)

    def start_playback_at_qn(self, time: float) -> dict[str, Any]:
        return self._command(
            "start_playback_at_qn",
            {"time": float(time), "qn": float(time)},
            side_effect=True,
        )

    def set_arrangement_loop(
        self, start: float, end: float, enabled: bool
    ) -> dict[str, Any]:
        return self._command(
            "set_arrangement_loop",
            {"start": start, "end": end, "enabled": enabled},
            side_effect=True,
        )

    def set_tempo(self, tempo: float) -> dict[str, Any]:
        return self._command("set_tempo", {"tempo": tempo}, side_effect=True)

    def create_audio_track(self, name: str, index: int = -1) -> dict[str, Any]:
        created = self._command(
            "create_audio_track", {"index": index}, side_effect=True
        )
        track_index = int(created["index"])
        if name:
            self._command(
                "set_track_name",
                {"track_index": track_index, "name": name},
                side_effect=True,
            )
            created["name"] = name
        return created

    def get_available_inputs(self, track_index: int) -> dict[str, Any]:
        return self._command("get_available_inputs", {"track_index": track_index})

    def get_available_outputs(self, track_index: int) -> dict[str, Any]:
        return self._command("get_available_outputs", {"track_index": track_index})

    def get_track_input_routing(self, track_index: int) -> dict[str, Any]:
        return self._command("get_track_input_routing", {"track_index": track_index})

    def get_track_output_routing(self, track_index: int) -> dict[str, Any]:
        return self._command("get_track_output_routing", {"track_index": track_index})

    def get_track_available_input_types(self, track_index: int) -> dict[str, Any]:
        return self._command("get_track_available_input_types", {"track_index": track_index})

    def get_track_available_output_types(self, track_index: int) -> dict[str, Any]:
        return self._command("get_track_available_output_types", {"track_index": track_index})

    def get_session_automation_record(self) -> dict[str, Any]:
        return self._command("get_session_automation_record", {})

    def get_clip_automation(
        self, track_index: int, clip_index: int, parameter_name: str
    ) -> dict[str, Any]:
        return self._command(
            "get_clip_automation",
            {
                "track_index": track_index,
                "clip_index": clip_index,
                "parameter_name": parameter_name,
            },
        )

    def set_track_input_routing(
        self,
        track_index: int,
        routing_type: str,
        routing_channel: str = "",
    ) -> dict[str, Any]:
        return self._command(
            "set_track_input_routing",
            {
                "track_index": track_index,
                "routing_type": routing_type,
                "routing_channel": routing_channel,
            },
            side_effect=True,
        )

    def set_track_output_routing(
        self,
        track_index: int,
        routing_type: str,
        routing_channel: str = "",
    ) -> dict[str, Any]:
        return self._command(
            "set_track_output_routing",
            {
                "track_index": track_index,
                "routing_type": routing_type,
                "routing_channel": routing_channel,
            },
            side_effect=True,
        )

    def save_session(self) -> dict[str, Any]:
        return self._command("save", {}, side_effect=True)

    def set_device_input_routing(
        self, track_index: int, device_index: int, routing_type: str, routing_channel: str = ""
    ) -> dict[str, Any]:
        return self._command(
            "set_device_input_routing",
            {
                "track_index": track_index,
                "device_index": device_index,
                "routing_type": routing_type,
                "routing_channel": routing_channel,
            },
            side_effect=True,
        )

    def set_track_monitoring(self, track_index: int, monitoring: str) -> dict[str, Any]:
        return self._command(
            "set_track_monitoring",
            {"track_index": track_index, "monitoring": monitoring},
            side_effect=True,
        )

    def get_track_monitoring(self, track_index: int) -> dict[str, Any]:
        return self._command("get_track_monitoring", {"track_index": track_index})

    def get_track_delay(self, track_index: int) -> dict[str, Any]:
        return self._command("get_track_delay", {"track_index": track_index})

    def get_session_path(self) -> dict[str, Any]:
        return self._command("get_session_path")

    def set_track_arm(self, track_index: int, arm: bool) -> dict[str, Any]:
        return self._command(
            "set_track_arm",
            {"track_index": track_index, "arm": arm},
            side_effect=True,
        )

    def set_return_volume(self, return_index: int, volume: float) -> dict[str, Any]:
        return self._command(
            "set_return_volume",
            {"return_index": return_index, "volume": volume},
            side_effect=True,
        )

    def set_send_level(
        self, track_index: int, send_index: int, level: float
    ) -> dict[str, Any]:
        return self._command(
            "set_send_level",
            {
                "track_index": track_index,
                "send_index": send_index,
                "level": level,
            },
            side_effect=True,
        )

    def get_send_level(self, track_index: int, send_index: int) -> dict[str, Any]:
        return self._command(
            "get_send_level",
            {"track_index": track_index, "send_index": send_index},
        )

    def get_track_sends(self, track_index: int, *, limit: int = 16) -> list[dict[str, Any]]:
        cached = self.last_track_infos.get(int(track_index))
        if cached and isinstance(cached.get("sends"), list):
            return list(cached["sends"])
        batched = self._safe_command(
            "get_track_sends", {"track_index": track_index}, default=None
        )
        if isinstance(batched, dict) and isinstance(batched.get("sends"), list):
            return list(batched["sends"])
        rows: list[dict[str, Any]] = []
        for send_index in range(limit):
            row = self.get_send_level(track_index, send_index)
            if row.get("error"):
                break
            rows.append({"send_index": send_index, **row})
        return rows

    def get_return_tracks(self) -> dict[str, Any]:
        return self._command("get_return_tracks")

    def create_return_track(self) -> dict[str, Any]:
        return self._command("create_return_track", side_effect=True)

    def create_group_track(
        self, track_indices: list[int], name: str
    ) -> dict[str, Any]:
        return self._command(
            "create_group_track",
            {"track_indices": track_indices, "name": name},
            side_effect=True,
        )

    def set_signature(self, numerator: int, denominator: int) -> dict[str, Any]:
        return self._command(
            "set_signature",
            {"numerator": numerator, "denominator": denominator},
            side_effect=True,
        )

    def get_clip_warp_info(
        self, track_index: int, clip_index: int
    ) -> dict[str, Any]:
        return self._command(
            "get_clip_warp_info",
            {"track_index": track_index, "clip_index": clip_index},
        )

    def set_clip_warp_mode(
        self, track_index: int, clip_index: int, warp_mode: str
    ) -> dict[str, Any]:
        return self._command(
            "set_clip_warp_mode",
            {
                "track_index": track_index,
                "clip_index": clip_index,
                "warp_mode": warp_mode,
            },
            side_effect=True,
        )

    def create_audio_clip(
        self, track_index: int, clip_index: int, file_path: str
    ) -> dict[str, Any]:
        return self._command(
            "create_audio_clip",
            {
                "track_index": track_index,
                "clip_index": clip_index,
                "file_path": file_path,
            },
            side_effect=True,
        )

    def set_clip_warping(
        self, track_index: int, clip_index: int, warping: bool
    ) -> dict[str, Any]:
        return self._command(
            "set_clip_warping",
            {
                "track_index": track_index,
                "clip_index": clip_index,
                "warping": warping,
            },
            side_effect=True,
        )

    def set_clip_loop(
        self,
        track_index: int,
        clip_index: int,
        loop_start: float,
        loop_end: float,
        looping: bool,
    ) -> dict[str, Any]:
        return self._command(
            "set_clip_loop",
            {
                "track_index": track_index,
                "clip_index": clip_index,
                "loop_start": loop_start,
                "loop_end": loop_end,
                "looping": looping,
            },
            side_effect=True,
        )

    def duplicate_clip_to_arrangement(
        self,
        track_index: int,
        clip_index: int,
        destination_time: float,
        length: float | None = None,
    ) -> dict[str, Any]:
        return self._command(
            "duplicate_clip_to_arrangement",
            {
                "track_index": track_index,
                "clip_index": clip_index,
                "destination_time": destination_time,
                "length": length,
            },
            side_effect=True,
        )

    def get_arrangement_clips(self) -> dict[str, Any]:
        return self._command("get_arrangement_clips")

    def delete_arrangement_clips(self, clip_ids: list[str]) -> dict[str, Any]:
        return self._command("delete_arrangement_clips", {"clip_ids": list(clip_ids)}, side_effect=True)

    def fire_clip(self, track_index: int, clip_index: int) -> dict[str, Any]:
        return self._command(
            "fire_clip",
            {"track_index": track_index, "clip_index": clip_index},
            side_effect=True,
        )

    def fire_clips(self, clips: list[dict[str, Any]]) -> dict[str, Any]:
        return self._command("fire_clips", {"clips": clips}, side_effect=True)

    def stop_clip(self, track_index: int, clip_index: int) -> dict[str, Any]:
        return self._command(
            "stop_clip",
            {"track_index": track_index, "clip_index": clip_index},
            side_effect=True,
        )

    def stop_clips(self, clips: list[dict[str, Any]]) -> dict[str, Any]:
        return self._command("stop_clips", {"clips": clips}, side_effect=True)

    def bridge_command(
        self,
        command_type: str,
        params: dict[str, Any] | None = None,
        *,
        side_effect: bool | None = None,
    ) -> dict[str, Any]:
        if side_effect is None:
            side_effect = not (
                command_type.startswith("get_")
                or command_type in {"health_check", "protocol_hello"}
            )
        return self._command(command_type, params or {}, side_effect=bool(side_effect))

    def search_browser(self, query: str, category: str = "all") -> dict[str, Any]:
        return self._command(
            "search_browser", {"query": query, "category": category}
        )

    def browse_path(self, path: list[str]) -> dict[str, Any]:
        return self._command("browse_path", {"path": path})

    def _resolve_device_uri_by_search(self, name: str) -> str | None:
        """Resolve a device URI from browser search categories.

        `search_browser(category=all)` in the vendored bridge omits plugins, so
        we probe additional categories and keep deterministic ranking.
        """
        normalized = " ".join(name.casefold().split())
        best: tuple[int, int, str] | None = None
        for category in ("audio_effects", "midi_effects", "instruments", "drums", "sounds", "all"):
            try:
                sr = self.search_browser(name, category)
            except DawError:
                continue
            results = sr.get("results", []) if isinstance(sr, dict) else []
            for row in results:
                if not bool(row.get("is_device")):
                    continue
                uri = str(row.get("uri") or "")
                if not uri:
                    continue
                row_name = " ".join(str(row.get("name") or "").casefold().split())
                exact = 1 if row_name == normalized else 0
                contains = 1 if (normalized in row_name or row_name in normalized) else 0
                score = (exact, contains, uri)
                if best is None or score > best:
                    best = score
        return None if best is None else best[2]

    def _resolve_device_uri_by_plugins_browse(self, name: str) -> str | None:
        """Fallback plugin resolver via browse_path(['plugins']) tree walk."""
        normalized = " ".join(name.casefold().split())
        stack: list[list[str]] = [["plugins"]]
        visited: set[tuple[str, ...]] = set()
        matches: list[tuple[int, str]] = []
        max_depth = 6
        max_nodes = 3000
        scanned = 0
        while stack and scanned < max_nodes:
            path = stack.pop()
            key = tuple(path)
            if key in visited:
                continue
            visited.add(key)
            try:
                payload = self.browse_path(path)
            except DawError:
                continue
            items = payload.get("items", []) if isinstance(payload, dict) else []
            for item in items:
                scanned += 1
                item_name = str(item.get("name") or "")
                item_norm = " ".join(item_name.casefold().split())
                is_folder = bool(item.get("is_folder"))
                is_loadable = bool(item.get("is_loadable"))
                uri = str(item.get("uri") or "")
                if uri and is_loadable and (normalized == item_norm or normalized in item_norm or item_norm in normalized):
                    exact = 2 if item_norm == normalized else 1
                    matches.append((exact, uri))
                if is_folder and len(path) < max_depth and item_name:
                    stack.append(path + [item_name])
        if not matches:
            return None
        matches.sort(key=lambda row: (row[0], row[1]), reverse=True)
        return matches[0][1]

    def load_instrument_or_effect(
        self, track_index: int, uri: str
    ) -> dict[str, Any]:
        # The live bridge needs a browser query URI (e.g. "query:AudioFx#EQ%20Eight").
        # Accept a bare name ("EQ Eight") OR a mock-style path ("devices/audio-effects/EQ Eight").
        if not uri.startswith("query:"):
            name = uri.rsplit("/", 1)[-1] if "/" in uri else uri
            cache_key = " ".join(name.casefold().split())
            cached = self._device_uri_cache.get(cache_key)
            if not cached:
                resolved = self._resolve_device_uri_by_search(name)
                if not resolved:
                    resolved = self._resolve_device_uri_by_plugins_browse(name)
                self._device_uri_cache[cache_key] = resolved or uri
            uri = self._device_uri_cache[cache_key]
        return self._command(
            "load_instrument_or_effect",
            {"track_index": track_index, "uri": uri},
            side_effect=True,
        )

    def load_browser_item(
        self, track_index: int, item_uri: str, clip_index: int | None = None
    ) -> dict[str, Any]:
        # The live bridge loads a BROWSER item, not a file path. Resolve local
        # local sample paths to a browser URI
        # by searching the filename stem across all categories (incl. Places).
        # Non-query URIs are library-relative sample paths: navigate the user Places
        # by path (O(depth), fast) instead of a full recursive browser search.
        if item_uri and not item_uri.startswith("query:"):
            value = str(item_uri).replace("\\", "/")
            parts = [part for part in value.strip("/").split("/") if part]
            if (
                not parts
                or value.startswith("/")
                or ":" in parts[0]
                or any(part in {".", ".."} for part in parts)
            ):
                raise DawError("browser sample path must be a safe relative library path")
            return self._command(
                "load_browser_item_by_path",
                {
                    "track_index": track_index,
                    "rel_path": item_uri,
                    "clip_index": clip_index if clip_index is not None else 0,
                },
                side_effect=True,
            )
        params: dict[str, Any] = {"track_index": track_index, "item_uri": item_uri}
        if clip_index is not None:
            params["clip_index"] = clip_index
        return self._command("load_browser_item", params, side_effect=True)

    def get_device_by_name(self, track_index: int, device_name: str) -> dict[str, Any]:
        return self._command(
            "get_device_by_name",
            {"track_index": track_index, "device_name": device_name},
        )

    def load_device_preset(
        self, track_index: int, device_index: int, preset_uri: str
    ) -> dict[str, Any]:
        return self._command(
            "load_device_preset",
            {
                "track_index": track_index,
                "device_index": device_index,
                "preset_uri": preset_uri,
            },
            side_effect=True,
        )

    def delete_device(self, track_index: int, device_index: int) -> dict[str, Any]:
        return self._command(
            "delete_device",
            {"track_index": track_index, "device_index": device_index},
            side_effect=True,
        )

    def move_device(
        self, track_index: int, device_index: int, new_index: int
    ) -> dict[str, Any]:
        return self._command(
            "move_device",
            {
                "track_index": track_index,
                "device_index": device_index,
                "new_index": new_index,
            },
            side_effect=True,
        )

    def move_device_right(
        self, track_index: int, device_index: int
    ) -> dict[str, Any]:
        return self._command(
            "move_device_right",
            {"track_index": track_index, "device_index": device_index},
            side_effect=True,
        )

    def _safe_command(
        self, command_type: str, params: dict[str, Any] | None = None, default: Any = None
    ) -> Any:
        try:
            return self._command(command_type, params)
        except DawError:
            return default

    def _command(
        self,
        command_type: str,
        params: dict[str, Any] | None = None,
        *,
        side_effect: bool = False,
    ) -> dict[str, Any]:
        if self._sock is None:
            raise DawError("Not connected")
        if (
            command_type != "protocol_hello"
            and self.capabilities
            and self.strict_capabilities
        ):
            require_capability(self.capabilities, command_type)
        self.tcp_counts[command_type] += 1
        request_id = f"req_{uuid4().hex[:12]}"
        heavy_read = command_type in {
            "get_session_info",
            "get_capture_topology",
            "get_tracks_info",
            "get_track_info",
            "get_all_track_names",
            "get_session_path",
        }
        if side_effect:
            timeout = self.timeouts.simple_mutation
        elif heavy_read:
            timeout = self.timeouts.large_operation
        else:
            timeout = self.timeouts.read
        payload = json.dumps(
            {
                "type": command_type,
                "params": params or {},
                "request_id": request_id,
            }
        ).encode("utf-8")
        try:
            self._sock.sendall(payload)
        except OSError as exc:
            raise DawError("socket disconnect while sending") from exc
        self._sock.settimeout(timeout)
        try:
            response = self._recv_json()
        except TimeoutError as exc:
            if side_effect:
                raise WriteInDoubt(command_type, request_id) from exc
            raise DawError("Timeout waiting for Ableton") from exc
        except OSError as exc:
            if side_effect:
                raise WriteInDoubt(command_type, request_id) from exc
            raise DawError("socket disconnect") from exc
        if not isinstance(response, dict):
            raise ProtocolError("unexpected response")
        if "request_id" in response and response["request_id"] != request_id:
            raise ProtocolError(
                f"wrong request id: expected {request_id} got {response['request_id']}"
            )
        if response.get("status") == "error":
            raise DawError(response.get("message", "Unknown Ableton error"))
        result = response.get("result", {})
        if not isinstance(result, dict):
            return {"value": result}
        return result

    def _recv_json(self) -> dict[str, Any]:
        decoder = json.JSONDecoder()
        assert self._sock is not None
        while True:
            if self._recv_buf:
                try:
                    parsed, index = decoder.raw_decode(self._recv_buf)
                    self._recv_buf = self._recv_buf[index:].lstrip()
                    if not isinstance(parsed, dict):
                        raise ProtocolError("unexpected response")
                    return parsed
                except json.JSONDecodeError:
                    pass
            chunk = self._sock.recv(8192)
            if not chunk:
                if self._recv_buf:
                    raise ProtocolError("truncated packet")
                raise DawError("Empty response from Ableton")
            self._recv_buf += chunk.decode("utf-8")
