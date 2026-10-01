"""Concrete musical events over TrackSpec; producer intent, never Live authority."""

from __future__ import annotations

from enum import StrEnum
import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from copilot.producer.track_spec import TrackSpec
from copilot.schemas.session import ClipState, MidiNote


class ScoreNote(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    pitch: int = Field(ge=0, le=127, strict=True)
    start_time: float = Field(ge=0)
    duration: float = Field(gt=0)
    velocity: int = Field(ge=1, le=127, strict=True)


class MidiPhrase(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    length_qn: float = Field(gt=0, le=256)
    notes: list[ScoreNote] = Field(min_length=1, max_length=2048)
    reason: str = Field(min_length=1, max_length=1000)

    @model_validator(mode="after")
    def bounded_notes(self) -> "MidiPhrase":
        if any(note.start_time + note.duration > self.length_qn for note in self.notes):
            raise ValueError("SCORE_NOTE_OUTSIDE_PHRASE")
        return self

    def midi_notes(self) -> list[MidiNote]:
        return [MidiNote(**note.model_dump()) for note in self.notes]

    def matches_clip(self, clip: ClipState) -> bool:
        return (
            clip.is_midi and not clip.is_audio
            and math.isclose(clip.length_beats, self.length_qn, abs_tol=1e-6, rel_tol=0)
            and midi_note_signature(clip.notes) == midi_note_signature(self.midi_notes())
        )


def midi_note_signature(notes: list[MidiNote]) -> list[tuple[int, float, float, int, bool]]:
    return sorted(
        (note.pitch, round(note.start_time, 6), round(note.duration, 6), note.velocity, note.mute)
        for note in notes
    )


class PlaybackIntent(StrEnum):
    SOURCE_ONCE = "SOURCE_ONCE"
    REPEAT_SOURCE = "REPEAT_SOURCE"
    MIDI_PHRASE = "MIDI_PHRASE"


class ScoreEvent(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, allow_inf_nan=False)

    event_id: str = Field(min_length=1, max_length=128)
    section: str = Field(min_length=1, max_length=64)
    role: str = Field(min_length=1, max_length=64)
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    offset_qn: float = Field(ge=0)
    duration_qn: float = Field(gt=0)
    playback: PlaybackIntent
    phrase_id: str | None = Field(default=None, min_length=1, max_length=128)
    reason: str = Field(min_length=1, max_length=1000)

    @model_validator(mode="after")
    def phrase_matches_intent(self) -> "ScoreEvent":
        if (self.playback is PlaybackIntent.MIDI_PHRASE) != (self.phrase_id is not None):
            raise ValueError("SCORE_PHRASE_REFERENCE_MISMATCH")
        return self


class PhraseSlotBinding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: str
    phrase_id: str
    clip_index: int = Field(ge=0)
    track_stable_id: str
    clip_stable_id: str
    source_sha256: str
    project_identity: str
    session_incarnation_id: str


class TransitionKind(StrEnum):
    ROLE_CHANGE = "ROLE_CHANGE"
    SCORE_EVENT = "SCORE_EVENT"
    AUTOMATION = "AUTOMATION"


class ScoreTransition(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    after_section: str = Field(min_length=1, max_length=64)
    kind: TransitionKind
    event_id: str | None = None
    reason: str = Field(min_length=1, max_length=1000)

    @model_validator(mode="after")
    def concrete_event(self) -> "ScoreTransition":
        if (self.kind is TransitionKind.SCORE_EVENT) != (self.event_id is not None):
            raise ValueError("SCORE_TRANSITION_EVENT_MISMATCH")
        return self


class ArrangementScore(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["arrangement-score-v1"] = "arrangement-score-v1"
    phrases: dict[str, MidiPhrase] = Field(default_factory=dict, max_length=128)
    events: list[ScoreEvent] = Field(min_length=1, max_length=2048)
    transitions: list[ScoreTransition] = Field(default_factory=list, max_length=32)
    no_write: Literal[True] = True

    @model_validator(mode="after")
    def unique_events(self) -> "ArrangementScore":
        ids = [event.event_id for event in self.events]
        if len(ids) != len(set(ids)):
            raise ValueError("SCORE_DUPLICATE_EVENT_ID")
        used_phrases = {event.phrase_id for event in self.events if event.phrase_id is not None}
        if used_phrases != set(self.phrases):
            raise ValueError("SCORE_PHRASES_UNRESOLVED_OR_UNUSED")
        boundaries = [cue.after_section for cue in self.transitions]
        if len(boundaries) != len(set(boundaries)):
            raise ValueError("SCORE_DUPLICATE_TRANSITION")
        return self

    def validate_against(self, spec: TrackSpec, *, sample_map: dict[str, str]) -> None:
        if (spec.meter_numerator, spec.meter_denominator) != (4, 4):
            raise ValueError("SCORE_METER_NOT_SUPPORTED")
        sections = {section.name: section for section in spec.sections}
        covered: set[tuple[str, str]] = set()
        intervals: dict[tuple[str, str], list[tuple[float, float]]] = {}
        for event in self.events:
            section = sections.get(event.section)
            if section is None or event.role not in section.active_roles:
                raise ValueError("SCORE_EVENT_OUTSIDE_ACTIVE_SECTION_ROLE")
            if event.source_sha256 != sample_map.get(event.role):
                raise ValueError("SCORE_EVENT_SOURCE_NOT_SELECTED")
            if event.offset_qn + event.duration_qn > section.bars * 4:
                raise ValueError("SCORE_EVENT_OUTSIDE_SECTION")
            if event.phrase_id is not None:
                phrase = self.phrases[event.phrase_id]
                if not _whole_repetitions(event.duration_qn, phrase.length_qn):
                    raise ValueError("SCORE_PHRASE_PARTIAL_REPETITION")
            key = (event.section, event.role)
            covered.add(key)
            intervals.setdefault(key, []).append(
                (event.offset_qn, event.offset_qn + event.duration_qn)
            )
        expected = {
            (section.name, role) for section in spec.sections for role in section.active_roles
        }
        if expected != covered:
            raise ValueError("SCORE_ACTIVE_ROLE_HAS_NO_EVENT")
        starts = self.section_starts(spec)
        end = max(
            starts[event.section] + event.offset_qn + event.duration_qn
            for event in self.events
        )
        if abs(end - spec.duration_bars * 4) > 1e-6:
            raise ValueError("SCORE_DURATION_MISMATCH")
        for spans in intervals.values():
            ordered = sorted(spans)
            if any(right[0] < left[1] - 1e-6 for left, right in zip(ordered, ordered[1:])):
                raise ValueError("SCORE_EVENTS_OVERLAP")
        events = {event.event_id: event for event in self.events}
        for cue in self.transitions:
            if cue.after_section not in sections:
                raise ValueError("SCORE_TRANSITION_SECTION_UNKNOWN")
            index = next(i for i, section in enumerate(spec.sections) if section.name == cue.after_section)
            current = spec.sections[index]
            following = spec.sections[index + 1] if index + 1 < len(spec.sections) else None
            if cue.kind is TransitionKind.ROLE_CHANGE:
                next_roles = set(following.active_roles) if following else set()
                if set(current.active_roles) == next_roles:
                    raise ValueError("SCORE_TRANSITION_HAS_NO_ROLE_CHANGE")
            if cue.kind is TransitionKind.SCORE_EVENT:
                event = events.get(cue.event_id)
                if event is None or event.section not in {
                    current.name, following.name if following else current.name,
                }:
                    raise ValueError("SCORE_TRANSITION_EVENT_NOT_AT_BOUNDARY_SECTIONS")
        planned_boundaries = {cue.after_section for cue in self.transitions}
        if any(section.transition.strip() and section.name not in planned_boundaries for section in spec.sections):
            raise ValueError("SCORE_DECLARED_TRANSITION_HAS_NO_DECISION")

    def section_starts(self, spec: TrackSpec) -> dict[str, float]:
        starts = {}
        cursor = 0.0
        for section in spec.sections:
            starts[section.name] = cursor
            cursor += section.bars * 4
        return starts


def _whole_repetitions(duration: float, source_length: float) -> bool:
    import math

    return source_length > 0 and round(duration / source_length) >= 1 and math.isclose(
        duration / source_length, round(duration / source_length), abs_tol=1e-6,
    )
