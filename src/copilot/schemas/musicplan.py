"""MusicPlan V1 typed contracts. No free-form execution. No Ableton imports."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, Field

from copilot.schemas.session import MidiNote

SCHEMA_VERSION = "musicplan-v1"


class PlanStatus(StrEnum):
    DRAFT = "DRAFT"
    VALIDATED = "VALIDATED"
    STALE = "STALE"
    REJECTED = "REJECTED"
    READY_FOR_EXECUTION = "READY_FOR_EXECUTION"
    EXECUTED = "EXECUTED"
    VERIFIED = "VERIFIED"
    ROLLED_BACK = "ROLLED_BACK"
    FAILED = "FAILED"
    IN_DOUBT = "IN_DOUBT"


class ActionType(StrEnum):
    SET_TRACK_VOLUME = "SET_TRACK_VOLUME"


class ProductionActionKind(StrEnum):
    """Producer intent vocabulary; certification is owned by the runtime."""

    SET_TRACK_VOLUME = "SET_TRACK_VOLUME"
    LOAD_SAMPLE = "LOAD_SAMPLE"
    DUPLICATE_CLIP_TO_ARRANGEMENT = "DUPLICATE_CLIP_TO_ARRANGEMENT"
    LOAD_DEVICE = "LOAD_DEVICE"
    DEVICE_TWEAK = "DEVICE_TWEAK"
    DEVICE_LOAD = "DEVICE_LOAD"
    SAMPLE_SWAP = "SAMPLE_SWAP"
    CREATE_TRACK = "CREATE_TRACK"
    SAMPLE_LOAD = "SAMPLE_LOAD"
    CREATE_PATTERN = "CREATE_PATTERN"
    CREATE_MIDI_PHRASE = "CREATE_MIDI_PHRASE"
    SET_TRACK_MUTE = "SET_TRACK_MUTE"
    SET_TRACK_ROUTING = "SET_TRACK_ROUTING"
    SET_DEVICE_ROUTING = "SET_DEVICE_ROUTING"


class VolumeOperation(StrEnum):
    SET = "SET"
    DELTA = "DELTA"


class PlanIntentClass(StrEnum):
    AUTONOMOUS_MUSICAL_IMPROVEMENT = "AUTONOMOUS_MUSICAL_IMPROVEMENT"
    CONTROLLED_ENGINEERING_VALIDATION = "CONTROLLED_ENGINEERING_VALIDATION"
    ABSTENTION = "ABSTENTION"


class ActionTarget(BaseModel):
    """Durable target identity. RuntimeObjectId is never durable identity.

    `ref` is a PersistentObjectRef payload (dict) to keep schemas daw-free.
    """

    ref: dict[str, Any]
    runtime_id: dict[str, Any] | None = None
    resolve_status: str | None = None
    track_index_locator: int | None = None


class ActionPrecondition(BaseModel):
    code: str
    detail: str = ""
    required: bool = True
    satisfied: bool | None = None
    observed: Any = None
    expected: Any = None


class ExpectedEffect(BaseModel):
    affected_target: str
    direction: str
    description: str
    measurement_to_compare_after: str = ""
    limitations: list[str] = Field(default_factory=list)


class ExecutionVerificationSpec(BaseModel):
    """Did Ableton reach the requested parameter state?"""

    kind: Literal["EXECUTION"] = "EXECUTION"
    parameter: str = "track.mixer.volume"
    expected_after: float
    unit: str = "ableton_volume"
    tolerance: float = 1e-4
    requires_readback: bool = True


class MusicalVerificationSpec(BaseModel):
    """Did resulting audio improve? Recapture later — not run in V1 dry-run."""

    kind: Literal["MUSICAL"] = "MUSICAL"
    requires_recapture: bool = True
    comparison: str = ""
    deferred: bool = True
    note: str = "Musical verification is out of scope for MusicPlan V1 dry-run."


class VerificationSpec(BaseModel):
    execution: ExecutionVerificationSpec
    musical: MusicalVerificationSpec


class RollbackSpec(BaseModel):
    """Must be prepared from authoritative pre-write readback before READY."""

    parameter: str = "track.mixer.volume"
    unit: str = "ableton_volume"
    restore_value: float
    source: str = "authoritative_prewrite_readback"
    prepared: bool = True


class VolumeActionParams(BaseModel):
    kind: Literal["volume"] = "volume"
    operation: VolumeOperation
    unit: Literal["ableton_volume"] = "ableton_volume"
    target_value: float | None = None
    delta: float | None = None
    expected_before: float
    intended_after: float
    allowed_min: float = 0.0
    allowed_max: float = 1.0
    readback_tolerance: float = 0.02


class DeviceTweakActionParams(BaseModel):
    kind: Literal["device_tweak"] = "device_tweak"
    device_index: int
    parameter_name: str
    unit: str = ""
    expected_before: float
    intended_after: float
    allowed_min: float | None = None
    allowed_max: float | None = None
    readback_tolerance: float = 0.001


class DeviceLoadActionParams(BaseModel):
    kind: Literal["device_load"] = "device_load"
    device_name: str
    device_uri: str | None = None
    device_index_hint: int = -1


class SampleSwapActionParams(BaseModel):
    kind: Literal["sample_swap"] = "sample_swap"
    clip_index: int
    sample_uri: str
    previous_sample_uri: str | None = None


class CreateTrackActionParams(BaseModel):
    kind: Literal["create_track"] = "create_track"
    track_name: str
    track_kind: Literal["audio", "midi"] = "audio"
    index_hint: int = -1


class SampleLoadActionParams(BaseModel):
    kind: Literal["sample_load"] = "sample_load"
    clip_index: int
    sample_uri: str


class ArrangementDuplicateActionParams(BaseModel):
    kind: Literal["duplicate_clip_to_arrangement"] = "duplicate_clip_to_arrangement"
    clip_index: int
    destination_time: float
    length: float | None = None


class PatternActionParams(BaseModel):
    kind: Literal["create_pattern"] = "create_pattern"
    clip_index: int
    length_beats: float
    notes: list[MidiNote] = Field(default_factory=list)


class SetTrackMuteActionParams(BaseModel):
    kind: Literal["set_track_mute"] = "set_track_mute"
    mute: bool


class SetTrackRoutingActionParams(BaseModel):
    kind: Literal["set_track_routing"] = "set_track_routing"
    routing_type: str
    routing_channel: str = ""


class SetDeviceRoutingActionParams(BaseModel):
    kind: Literal["set_device_routing"] = "set_device_routing"
    device_index: int
    routing_type: str = "Track"
    routing_channel: str = ""


ActionParams = Annotated[
    Union[
        VolumeActionParams,
        DeviceTweakActionParams,
        DeviceLoadActionParams,
        SampleSwapActionParams,
        CreateTrackActionParams,
        SampleLoadActionParams,
        ArrangementDuplicateActionParams,
        PatternActionParams,
        SetTrackMuteActionParams,
        SetTrackRoutingActionParams,
        SetDeviceRoutingActionParams,
    ],
    Field(discriminator="kind"),
]


class PlanAction(BaseModel):
    action_id: str
    action_type: ProductionActionKind
    target: ActionTarget
    params: ActionParams
    reason: str
    evidence_refs: list[str] = Field(default_factory=list)
    preconditions: list[ActionPrecondition] = Field(default_factory=list)
    expected_effect: ExpectedEffect
    verification: VerificationSpec | None = None
    rollback: RollbackSpec | None = None
    reversible: bool = True


class DiagnosisBinding(BaseModel):
    diagnosis_id: str
    diagnosis_revision: int | None = None
    diagnosis_status: str
    diagnosis_accepted: bool
    cause_status: str | None = None
    region_id: str | None = None
    artifact: str | None = None


class MusicPlan(BaseModel):
    plan_id: str
    schema_version: str = SCHEMA_VERSION
    status: PlanStatus = PlanStatus.DRAFT
    intent_class: PlanIntentClass = PlanIntentClass.AUTONOMOUS_MUSICAL_IMPROVEMENT
    diagnosis: DiagnosisBinding
    project_state_token: str
    audible_state_token: str
    target_state_tokens: dict[str, str] = Field(default_factory=dict)
    evidence_refs: list[str] = Field(default_factory=list)
    actions: list[PlanAction] = Field(default_factory=list)
    created_at: str
    notes: list[str] = Field(default_factory=list)
    rejection_reason: str | None = None
    gate: dict[str, Any] = Field(default_factory=dict)


class CompiledExecutionEnvelope(BaseModel):
    """Typed production-write envelope. Not free-form commands. Not executed here."""

    envelope_id: str
    plan_id: str
    action_id: str
    action_type: ActionType
    transaction_intent: str
    resolved_track_index: int
    runtime_id: dict[str, Any] | None = None
    expected_before: float
    requested_after: float
    unit: str = "ableton_volume"
    readback_required: bool = True
    rollback_value: float
    verification: VerificationSpec
    project_state_token: str
    audible_state_token: str
    target_state_token: str
    musical_writes_if_executed: int = 1
    dry_run_only: bool = True


class DryRunResult(BaseModel):
    status: str
    plan_id: str
    plan_status: PlanStatus
    gate: dict[str, Any] = Field(default_factory=dict)
    target_resolution: dict[str, Any] = Field(default_factory=dict)
    live_tokens: dict[str, str] = Field(default_factory=dict)
    current_volume: float | None = None
    preconditions: list[ActionPrecondition] = Field(default_factory=list)
    rollback: RollbackSpec | None = None
    compiled: CompiledExecutionEnvelope | None = None
    would_mutate: bool = False
    musical_writes: int = 0
    detail: str = ""
    artifact: str | None = None


class PlannedAction(BaseModel):
    operation: str
    target: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)


class LegacyMusicPlan(BaseModel):
    goal: str
    target: str | None = None
    constraints: list[str] = Field(default_factory=list)
    musical_intent: str = ""
    actions: list[PlannedAction] = Field(default_factory=list)
    validation_rules: list[str] = Field(default_factory=list)
    expected_effect: str = ""
