from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class TransportState(BaseModel):
    tempo: float = 120.0
    signature_numerator: int = 4
    signature_denominator: int = 4
    playing: bool = False
    position_beats: float = 0.0


class MixerState(BaseModel):
    volume: float = 0.85
    pan: float = 0.0
    mute: bool = False
    solo: bool = False
    arm: bool = False


class MidiNote(BaseModel):
    pitch: int
    start_time: float
    duration: float
    velocity: int = 100
    mute: bool = False


class ClipState(BaseModel):
    stable_id: str
    slot_index: int
    name: str
    length_beats: float
    is_midi: bool = True
    notes: list[MidiNote] = Field(default_factory=list)
    sample_uri: str | None = None
    is_audio: bool = False


class DeviceParameter(BaseModel):
    index: int
    name: str
    value: float
    min: float = 0.0
    # ``None`` means the bridge did not provide a usable maximum. Never
    # synthesize a range: parameter writes must defer until bounds are known.
    max: float | None = None


class DeviceState(BaseModel):
    stable_id: str
    index: int
    name: str
    class_name: str = ""
    enabled: bool = True
    parameters: list[DeviceParameter] = Field(default_factory=list)
    sample_uri: str | None = None


class RoutingState(BaseModel):
    input_type: str = ""
    input_channel: str = ""
    output_type: str = ""
    output_channel: str = ""
    monitoring: str = ""


class SendState(BaseModel):
    index: int = 0
    name: str = ""
    value: float = 0.0


class TrackState(BaseModel):
    stable_id: str
    index: int
    name: str
    role: Literal["midi", "audio", "return", "master", "unknown"] = "unknown"
    mixer: MixerState = Field(default_factory=MixerState)
    clips: list[ClipState] = Field(default_factory=list)
    devices: list[DeviceState] = Field(default_factory=list)
    routing: RoutingState = Field(default_factory=RoutingState)
    sends: list[SendState] = Field(default_factory=list)
    grouped: bool = False
    foldable: bool = False
    clip_slot_count: int | None = Field(default=None, ge=0)
    empty_clip_slots: list[int] | None = None


class SessionState(BaseModel):
    revision: int = 0
    state_hash: str = ""
    session_incarnation_id: str = ""
    daw: str = "ableton"
    connected: bool = False
    transport: TransportState = Field(default_factory=TransportState)
    tracks: list[TrackState] = Field(default_factory=list)
    selected_track_id: str | None = None
    selected_clip_id: str | None = None
    project_path: str | None = None
    project_name: str | None = None
    project_identity: str = ""
    project_token: str = ""
    audible_token: str = ""

    def track_by_id(self, stable_id: str) -> TrackState:
        for track in self.tracks:
            if track.stable_id == stable_id:
                return track
        raise KeyError(f"Unknown track id: {stable_id}")

    def track_by_name(self, name: str) -> TrackState | None:
        matches = [track for track in self.tracks if track.name == name]
        if len(matches) != 1:
            return None
        return matches[0]

    def clip_by_id(self, stable_id: str) -> tuple[TrackState, ClipState]:
        for track in self.tracks:
            for clip in track.clips:
                if clip.stable_id == stable_id:
                    return track, clip
        raise KeyError(f"Unknown clip id: {stable_id}")
