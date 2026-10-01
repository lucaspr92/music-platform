"""Plan extra MIDI phrases in observed empty slots, never existing-clip edits."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from copilot.musicplan import build_pattern_action
from copilot.musicplan.score_compiler import ScoreDeferred, require_score_session
from copilot.producer.arrangement_score import ArrangementScore, PhraseSlotBinding, midi_note_signature
from copilot.producer.midi_phrase_policy import (
    MIDI_PHRASE_CAPABILITIES, empty_phrase_slot_blocker, phrase_preservation_token,
)
from copilot.producer.track_spec import TrackSpec
from copilot.schemas.musicplan import PlanAction, ProductionActionKind
from copilot.schemas.session import SessionState


class PhraseAssignment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: str
    phrase_id: str
    track_stable_id: str
    clip_index: int = Field(ge=0)
    action_id: str | None = None
    initial_clip_stable_id: str | None = None


class PhraseMaterialCompilation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    actions: list[PlanAction] = Field(default_factory=list)
    assignments: list[PhraseAssignment] = Field(default_factory=list)
    deferred: list[ScoreDeferred] = Field(default_factory=list)
    excluded_slots: dict[str, set[int]] = Field(default_factory=dict)
    preservation_token: str
    no_write: Literal[True] = True


def prepare_score_phrase_material(
    *, score: ArrangementScore, spec: TrackSpec, sample_map: dict[str, str],
    session: SessionState, owned_midi_track_ids: frozenset[str],
    verified_material_roles: set[str], verified_material_devices: dict[str, str],
    capabilities: frozenset[str],
    enabled: bool = False, max_new_phrases: int = 128,
) -> PhraseMaterialCompilation:
    score.validate_against(spec, sample_map=sample_map)
    require_score_session(session, spec)
    if max_new_phrases < 1:
        raise ValueError("MIDI_PHRASE_BUDGET_INVALID")
    result = PhraseMaterialCompilation(preservation_token="")
    starts = score.section_starts(spec)
    events = sorted(
        (event for event in score.events if event.phrase_id is not None),
        key=lambda event: (starts[event.section] + event.offset_qn, event.event_id),
    )
    seen = set()
    initial_phrase_ids = {}
    for event in events:
        if event.phrase_id is None:
            raise ValueError("MIDI_PHRASE_ID_REQUIRED")
        initial_phrase_ids.setdefault(event.role, event.phrase_id)
    assigned_shapes: dict[tuple, PhraseAssignment] = {}
    available_slots: dict[str, list[int]] = {}
    for event in events:
        if event.phrase_id is None:
            raise ValueError("MIDI_PHRASE_ID_REQUIRED")
        key = (event.role, event.phrase_id)
        if key in seen:
            continue
        seen.add(key)
        phrase = score.phrases[event.phrase_id]

        def defer(reason: str, missing: list[str] | None = None) -> None:
            result.deferred.append(ScoreDeferred(
                event_id=event.event_id, section=event.section, role=event.role,
                reason=reason, required_capabilities=missing or [],
            ))

        matches = [track for track in session.tracks if track.name == event.role]
        if (
            len(matches) != 1 or matches[0].stable_id not in owned_midi_track_ids
            or event.role not in verified_material_roles
        ):
            defer("MIDI_PHRASE_OWNED_VERIFIED_MATERIAL_REQUIRED")
            continue
        track = matches[0]
        expected_device = verified_material_devices.get(event.role)
        if not expected_device or sum(
            device.stable_id == expected_device for device in track.devices
        ) != 1:
            defer("MIDI_PHRASE_VERIFIED_INSTRUMENT_MISSING")
            continue
        initial = [clip for clip in track.clips if clip.slot_index == 0]
        if len(initial) != 1 or not initial[0].stable_id:
            defer("MIDI_PHRASE_INITIAL_CLIP_UNRESOLVED")
            continue
        if not score.phrases[initial_phrase_ids[event.role]].matches_clip(initial[0]):
            defer("MIDI_PHRASE_INITIAL_READBACK_MISMATCH")
            continue
        shape = (event.role, phrase.length_qn, tuple(midi_note_signature(phrase.midi_notes())))
        previous = assigned_shapes.get(shape)
        if previous is not None:
            result.assignments.append(previous.model_copy(update={"phrase_id": event.phrase_id}))
            continue
        if phrase.matches_clip(initial[0]):
            assignment = PhraseAssignment(
                role=event.role, phrase_id=event.phrase_id, track_stable_id=track.stable_id,
                clip_index=0, initial_clip_stable_id=initial[0].stable_id,
            )
        else:
            if not enabled:
                defer("MIDI_PHRASE_EXPERIMENT_DISABLED")
                continue
            missing = sorted(MIDI_PHRASE_CAPABILITIES - capabilities)
            if missing:
                defer("MIDI_PHRASE_CAPABILITIES_UNAVAILABLE", missing)
                continue
            if track.stable_id not in available_slots:
                available_slots[track.stable_id] = sorted(track.empty_clip_slots or [])
            slots = available_slots[track.stable_id]
            if not slots:
                defer(
                    "MIDI_PHRASE_SLOT_INVENTORY_UNAVAILABLE"
                    if track.empty_clip_slots is None else "MIDI_PHRASE_EMPTY_SLOTS_EXHAUSTED"
                )
                continue
            index = slots[0]
            blocker = empty_phrase_slot_blocker(track, index)
            if blocker:
                defer(blocker)
                continue
            if len(result.actions) >= max_new_phrases:
                raise ValueError("MIDI_PHRASE_ACTION_BUDGET_EXCEEDED")
            slots.pop(0)
            action = build_pattern_action(
                track=track, project_identity=session.project_identity, clip_index=index,
                length_beats=phrase.length_qn, notes=phrase.midi_notes(),
                reason=f"Producer additional MIDI phrase: {event.role}/{event.phrase_id}",
                evidence_refs=[event.source_sha256],
                session_incarnation_id=session.session_incarnation_id,
            ).model_copy(update={"action_type": ProductionActionKind.CREATE_MIDI_PHRASE})
            result.actions.append(action)
            result.excluded_slots.setdefault(track.stable_id, set()).add(index)
            assignment = PhraseAssignment(
                role=event.role, phrase_id=event.phrase_id,
                track_stable_id=track.stable_id, clip_index=index, action_id=action.action_id,
            )
        result.assignments.append(assignment)
        assigned_shapes[shape] = assignment
    result.preservation_token = phrase_preservation_token(
        session, excluded_slots=result.excluded_slots,
    )
    return result


def phrase_material_blocker(
    material: PhraseMaterialCompilation, *, score: ArrangementScore,
    session: SessionState, verified_clip_ids: dict[str, str],
) -> str | None:
    if material.preservation_token != phrase_preservation_token(
        session, excluded_slots=material.excluded_slots,
    ):
        return "MIDI_PHRASE_MATERIAL_BASELINE_CHANGED"
    for assignment in material.assignments:
        if assignment.action_id is None:
            continue
        tracks = [track for track in session.tracks if track.stable_id == assignment.track_stable_id]
        if len(tracks) != 1:
            return "MIDI_PHRASE_TARGET_CHANGED"
        clips = [clip for clip in tracks[0].clips if clip.slot_index == assignment.clip_index]
        expected_id = verified_clip_ids.get(assignment.action_id)
        if expected_id is None:
            if clips:
                return "MIDI_PHRASE_RESERVED_SLOT_CHANGED"
        elif (
            len(clips) != 1 or clips[0].stable_id != expected_id
            or not score.phrases[assignment.phrase_id].matches_clip(clips[0])
        ):
            return "MIDI_PHRASE_VERIFIED_CLIP_CHANGED"
    return None


def bind_verified_phrase_slots(
    material: PhraseMaterialCompilation, *, score: ArrangementScore,
    sample_map: dict[str, str], session: SessionState, verified_clip_ids: dict[str, str],
) -> list[PhraseSlotBinding]:
    blocker = phrase_material_blocker(
        material, score=score, session=session, verified_clip_ids=verified_clip_ids,
    )
    if blocker:
        raise ValueError(blocker)
    bindings = []
    for assignment in material.assignments:
        expected_id = (
            verified_clip_ids.get(assignment.action_id)
            if assignment.action_id is not None else assignment.initial_clip_stable_id
        )
        if expected_id is None:
            continue
        track = session.track_by_id(assignment.track_stable_id)
        clips = [clip for clip in track.clips if clip.slot_index == assignment.clip_index]
        if (
            len(clips) != 1 or clips[0].stable_id != expected_id
            or not score.phrases[assignment.phrase_id].matches_clip(clips[0])
        ):
            raise ValueError("MIDI_PHRASE_BINDING_READBACK_MISMATCH")
        bindings.append(PhraseSlotBinding(
            role=assignment.role, phrase_id=assignment.phrase_id, clip_index=assignment.clip_index,
            track_stable_id=track.stable_id, clip_stable_id=expected_id,
            source_sha256=sample_map[assignment.role], project_identity=session.project_identity,
            session_incarnation_id=session.session_incarnation_id,
        ))
    return bindings
