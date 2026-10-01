"""Bind score placements to observed clips through existing MusicPlan builders."""

from __future__ import annotations

import math
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from copilot.daw.state_tokens import target_token, token_of
from copilot.musicplan import (
    build_create_track_action, build_duplicate_clip_to_arrangement_action,
    build_pattern_action, build_sample_load_action,
)
from copilot.producer.arrangement_score import (
    ArrangementScore, PhraseSlotBinding, PlaybackIntent, TransitionKind,
)
from copilot.producer.track_spec import TrackSpec
from copilot.schemas.musicplan import MusicPlan, PlanAction, ProductionActionKind
from copilot.schemas.session import MidiNote, RoutingState, SessionState, TrackState


class ScoreDeferred(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_id: str
    section: str
    role: str
    reason: str
    status: Literal["EXECUTION_DEFERRED"] = "EXECUTION_DEFERRED"
    required_capabilities: list[str] = Field(default_factory=list)


class ScoreBinding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action_id: str
    event_id: str
    section: str
    role: str
    track_stable_id: str
    source_clip_stable_id: str
    source_clip_index: int = Field(default=0, ge=0)
    phrase_id: str | None = None
    source_sha256: str
    project_identity: str
    session_incarnation_id: str
    source_track_token: str
    tempo_bpm: float
    start_qn: float
    duration_qn: float


class ScoreCompilation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    actions: list[PlanAction] = Field(default_factory=list)
    bindings: list[ScoreBinding] = Field(default_factory=list)
    deferred: list[ScoreDeferred] = Field(default_factory=list)
    no_write: Literal[True] = True

    @property
    def status(self) -> str:
        return "PLANNED_WITH_DEFERRED_ACTIONS" if self.deferred else "PLACEMENTS_BOUND"


def initial_score_patterns(
    score: ArrangementScore, spec: TrackSpec,
) -> dict[str, dict[str, Any]]:
    """The earliest phrase per role is the initial clip, not a deterministic template."""
    starts = score.section_starts(spec)
    patterns = {}
    events = sorted(score.events, key=lambda event: (starts[event.section] + event.offset_qn, event.event_id))
    for event in events:
        if event.phrase_id is not None and event.role not in patterns:
            phrase = score.phrases[event.phrase_id]
            patterns[event.role] = {
                "length_beats": phrase.length_qn,
                "notes": [note.model_dump(mode="json") for note in phrase.notes],
            }
    midi_roles = set(patterns)
    if any(event.role in midi_roles and event.phrase_id is None for event in events):
        raise ValueError("SCORE_ROLE_MIXES_MIDI_AND_AUDIO_PLAYBACK")
    return patterns


def build_initial_score_material(
    *, base_plan: MusicPlan, score: ArrangementScore, spec: TrackSpec,
    session: SessionState,
) -> MusicPlan:
    """Reuse canonical builders for initial sample-backed MIDI phrases."""
    patterns = initial_score_patterns(score, spec)
    roles = {role for section in spec.sections for role in section.active_roles}
    loads = {
        str(action.target.ref.get("name")): action
        for action in base_plan.actions if action.action_type is ProductionActionKind.SAMPLE_LOAD
    }
    actions = []
    for role in sorted(roles):
        load = loads.get(role)
        if load is None:
            raise ValueError(f"SCORE_MATERIAL_LOAD_MISSING:{role}")
        if role not in patterns:
            actions.extend(
                action for action in base_plan.actions
                if str(action.target.ref.get("name")) == role
                and action.action_type in {
                    ProductionActionKind.CREATE_TRACK, ProductionActionKind.SAMPLE_LOAD,
                }
            )
            continue
        track = TrackState(
            stable_id="", index=-1, name=role, role="midi",
            routing=RoutingState(output_type="Main", monitoring="in"),
        )
        pattern = patterns[role]
        refs = list(load.evidence_refs)
        actions.extend([
            build_create_track_action(
                project_identity=session.project_identity, track_name=role,
                track_kind="midi", reason=f"Producer score initial MIDI role: {role}",
                evidence_refs=refs,
            ),
            build_sample_load_action(
                track=track, project_identity=session.project_identity, clip_index=0,
                sample_uri=load.params.sample_uri,
                reason=f"Producer score sample instrument: {role}", evidence_refs=refs,
            ),
            build_pattern_action(
                track=track, project_identity=session.project_identity, clip_index=0,
                length_beats=pattern["length_beats"],
                notes=[MidiNote.model_validate(note) for note in pattern["notes"]],
                reason=f"Producer score initial phrase: {role}", evidence_refs=refs,
            ),
        ])
    return base_plan.model_copy(update={"actions": actions})


def _score_source_token(track: TrackState) -> str:
    return token_of({
        "target_token": target_token(track),
        "clip_sources": sorted(
            (clip.slot_index, clip.stable_id, clip.sample_uri, clip.is_audio)
            for clip in track.clips
        ),
        "device_sources": sorted(
            (device.index, device.stable_id, device.sample_uri)
            for device in track.devices
        ),
    })


def require_score_session(session: SessionState, spec: TrackSpec) -> None:
    if (
        not session.connected or not session.project_path
        or not session.project_path.lower().endswith(".als")
        or not session.project_identity or not session.project_token
        or not session.audible_token or not session.session_incarnation_id
    ):
        raise ValueError("SCORE_AUTHORITATIVE_SESSION_REQUIRED")
    if (
        not math.isclose(session.transport.tempo, spec.bpm, abs_tol=1e-6, rel_tol=0)
        or (session.transport.signature_numerator, session.transport.signature_denominator)
        != (spec.meter_numerator, spec.meter_denominator)
    ):
        raise ValueError("SCORE_SESSION_TIMING_MISMATCH")


def compile_score_placements(
    *, score: ArrangementScore, spec: TrackSpec, sample_map: dict[str, str],
    session: SessionState, capabilities: set[str] | frozenset[str] | None,
    evidence_refs: list[str], verified_material_roles: set[str],
    max_actions: int = 2048,
    verified_phrase_slots: list[PhraseSlotBinding] | None = None,
) -> ScoreCompilation:
    """No RPCs or writes: the existing executor must compile each action fresh."""
    score.validate_against(spec, sample_map=sample_map)
    require_score_session(session, spec)
    if max_actions < 1:
        raise ValueError("SCORE_ACTION_BUDGET_INVALID")
    result = ScoreCompilation()
    starts = score.section_starts(spec)
    required = {"clip.create", "clip.delete", "session.read"}
    missing = sorted(required - (capabilities or set()))
    phrase_slots = {}
    for binding in verified_phrase_slots or []:
        key = (binding.role, binding.phrase_id)
        if key in phrase_slots:
            raise ValueError("SCORE_PHRASE_BINDINGS_AMBIGUOUS")
        phrase_slots[key] = binding

    for event in score.events:
        def defer(reason: str, required_capabilities: list[str] | None = None) -> None:
            result.deferred.append(ScoreDeferred(
                event_id=event.event_id, section=event.section, role=event.role,
                reason=reason, required_capabilities=required_capabilities or [],
            ))

        if missing:
            defer("SCORE_PLACEMENT_CAPABILITIES_UNAVAILABLE", missing)
            continue
        if event.role not in verified_material_roles:
            defer("SCORE_SOURCE_MATERIAL_NOT_VERIFIED")
            continue
        matches = [track for track in session.tracks if track.name == event.role]
        if len(matches) != 1 or not matches[0].stable_id:
            defer("SCORE_TARGET_NOT_RESOLVED")
            continue
        track = matches[0]
        clip_index = 0
        phrase_binding = None
        if event.phrase_id is not None and verified_phrase_slots is not None:
            phrase_binding = phrase_slots.get((event.role, event.phrase_id))
            if phrase_binding is None or (
                phrase_binding.track_stable_id != track.stable_id
                or phrase_binding.project_identity != session.project_identity
                or phrase_binding.session_incarnation_id != session.session_incarnation_id
                or phrase_binding.source_sha256 != event.source_sha256
            ):
                defer("SCORE_MIDI_PHRASE_SOURCE_NOT_VERIFIED")
                continue
            clip_index = phrase_binding.clip_index
        clips = [clip for clip in track.clips if clip.slot_index == clip_index]
        if len(clips) != 1 or not clips[0].stable_id:
            defer("SCORE_SOURCE_CLIP_NOT_RESOLVED")
            continue
        clip = clips[0]
        if phrase_binding is not None and clip.stable_id != phrase_binding.clip_stable_id:
            defer("SCORE_MIDI_PHRASE_SOURCE_CHANGED")
            continue
        length = clip.length_beats
        if not math.isfinite(length) or length <= 0:
            defer("SCORE_SOURCE_CLIP_LENGTH_INVALID")
            continue
        if event.playback is PlaybackIntent.MIDI_PHRASE:
            if event.phrase_id is None:
                raise ValueError("SCORE_PHRASE_REFERENCE_MISMATCH")
            phrase = score.phrases[event.phrase_id]
            if not phrase.matches_clip(clip):
                defer("SCORE_DISTINCT_MIDI_PHRASE_NOT_CERTIFIED")
                continue
        elif not clip.is_audio or clip.is_midi:
            defer("SCORE_AUDIO_SOURCE_CLIP_NOT_VERIFIED")
            continue
        repetitions = event.duration_qn / length
        count = round(repetitions)
        if (
            count < 1 or not math.isclose(repetitions, count, abs_tol=1e-6)
            or (event.playback is PlaybackIntent.SOURCE_ONCE and count != 1)
        ):
            defer("SCORE_SOURCE_CLIP_TRIM_OR_STRETCH_NOT_CERTIFIED")
            continue
        if len(result.actions) + count > max_actions:
            raise ValueError("SCORE_PLACEMENT_ACTION_BUDGET_EXCEEDED")
        for repetition in range(count):
            start = starts[event.section] + event.offset_qn + repetition * length
            action = build_duplicate_clip_to_arrangement_action(
                track=track, project_identity=session.project_identity, clip_index=clip_index,
                destination_time=start, length=None,
                reason=f"REAL_LUCAS arrangement {event.section} (score event {event.event_id})",
                evidence_refs=list(dict.fromkeys([*evidence_refs, event.source_sha256])),
                session_incarnation_id=session.session_incarnation_id,
            )
            result.actions.append(action)
            result.bindings.append(ScoreBinding(
                action_id=action.action_id, event_id=event.event_id,
                section=event.section, role=event.role, track_stable_id=track.stable_id,
                source_clip_stable_id=clip.stable_id, source_sha256=event.source_sha256,
                source_clip_index=clip_index, phrase_id=event.phrase_id,
                project_identity=session.project_identity,
                session_incarnation_id=session.session_incarnation_id,
                source_track_token=_score_source_token(track),
                tempo_bpm=spec.bpm,
                start_qn=start, duration_qn=length,
            ))
    for cue in score.transitions:
        if cue.kind is TransitionKind.AUTOMATION:
            result.deferred.append(ScoreDeferred(
                event_id=f"transition:{cue.after_section}", section=cue.after_section,
                role="TRANSITION", reason="SCORE_TRANSITION_AUTOMATION_NOT_CERTIFIED",
            ))
    return result


def score_binding_blocker(binding: ScoreBinding, session: SessionState) -> str | None:
    """Check the bound source against the executor's fresh pre-write view."""
    if (
        not session.connected or session.project_identity != binding.project_identity
        or session.session_incarnation_id != binding.session_incarnation_id
    ):
        return "SCORE_BOUND_SESSION_CHANGED"
    if (
        not math.isclose(session.transport.tempo, binding.tempo_bpm, abs_tol=1e-6, rel_tol=0)
        or (session.transport.signature_numerator, session.transport.signature_denominator)
        != (4, 4)
    ):
        return "SCORE_BOUND_TIMING_CHANGED"
    matches = [track for track in session.tracks if track.stable_id == binding.track_stable_id]
    if len(matches) != 1 or matches[0].name != binding.role:
        return "SCORE_BOUND_TARGET_CHANGED"
    track = matches[0]
    clips = [clip for clip in track.clips if clip.slot_index == binding.source_clip_index]
    if (
        len(clips) != 1 or clips[0].stable_id != binding.source_clip_stable_id
        or _score_source_token(track) != binding.source_track_token
    ):
        return "SCORE_BOUND_SOURCE_CHANGED"
    return None


def verify_score_geometry(
    bindings: list[ScoreBinding], *, tracks: dict[str, int],
    clips: list[dict[str, Any]],
) -> list[str]:
    """Match exact observed clips to bound events, including intentional spaces."""
    if not bindings:
        return ["SCORE_ARRANGEMENT_BINDINGS_MISSING"]
    expected = []
    for binding in bindings:
        index = tracks.get(binding.role)
        if index is None:
            return ["SCORE_ARRANGEMENT_TRACK_MISSING"]
        expected.append((index, binding.start_qn, binding.duration_qn))
    selected = {index for index, _, _ in expected}
    observed = []
    for clip in clips:
        try:
            index = clip["track_index"]
            start = float(clip["start_time"])
            length = float(clip["length"])
        except (KeyError, TypeError, ValueError, OverflowError):
            return ["SCORE_ARRANGEMENT_GEOMETRY_UNAVAILABLE"]
        if (
            type(index) is not int or not math.isfinite(start)
            or not math.isfinite(length) or start < 0 or length <= 0
        ):
            return ["SCORE_ARRANGEMENT_GEOMETRY_INVALID"]
        if "end_time" in clip:
            try:
                end = float(clip["end_time"])
            except (TypeError, ValueError, OverflowError):
                return ["SCORE_ARRANGEMENT_GEOMETRY_INVALID"]
            if not math.isfinite(end) or not math.isclose(
                end, start + length, abs_tol=1e-3, rel_tol=0,
            ):
                return ["SCORE_ARRANGEMENT_GEOMETRY_INVALID"]
        if index in selected:
            observed.append((index, start, length))
    if len(expected) != len(observed):
        return ["SCORE_ARRANGEMENT_CLIP_COUNT_MISMATCH"]
    if any(
        wanted[0] != actual[0]
        or not math.isclose(wanted[1], actual[1], abs_tol=1e-3)
        or not math.isclose(wanted[2], actual[2], abs_tol=1e-3)
        for wanted, actual in zip(sorted(expected), sorted(observed))
    ):
        return ["SCORE_ARRANGEMENT_EVENT_GEOMETRY_MISMATCH"]
    return []
