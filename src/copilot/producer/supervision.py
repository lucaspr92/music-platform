"""Targeted revision intent and human project review, not a write authority."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from copilot.daw.state_tokens import target_token, token_of
from copilot.producer.state import ProducerPhase, ProducerState
from copilot.schemas.session import SessionState, TrackState


def _preservation_token(track: TrackState) -> str:
    return token_of({
        "target_token": target_token(track),
        "device_samples": [device.sample_uri for device in track.devices],
        "clip_samples": [clip.sample_uri for clip in track.clips],
    })


def _instrument_token(track: TrackState) -> str:
    return token_of({"devices": [
        device.model_dump(mode="json", exclude={"stable_id", "index"})
        for device in track.devices
    ]})


class RevisionRegion(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    start_qn: float = Field(ge=0)
    end_qn: float = Field(gt=0)

    @model_validator(mode="after")
    def ordered(self) -> "RevisionRegion":
        if self.end_qn <= self.start_qn:
            raise ValueError("REVISION_REGION_INVALID")
        return self


class ScopedRevisionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    instruction: str = Field(min_length=1, max_length=2000)
    project_identity: str = Field(min_length=1)
    project_token: str = Field(min_length=1)
    audible_token: str = Field(min_length=1)
    session_incarnation_id: str = Field(min_length=1)
    target_track_ids: list[str] = Field(min_length=1, max_length=16)
    regions: list[RevisionRegion] = Field(min_length=1, max_length=32)
    preserve_instruments: bool = True

    @model_validator(mode="after")
    def unique_targets(self) -> "ScopedRevisionRequest":
        if any(not target.strip() for target in self.target_track_ids):
            raise ValueError("REVISION_TARGET_ID_REQUIRED")
        if len(set(self.target_track_ids)) != len(self.target_track_ids):
            raise ValueError("REVISION_DUPLICATE_TARGET")
        ordered = sorted(self.regions, key=lambda region: region.start_qn)
        if any(right.start_qn < left.end_qn for left, right in zip(ordered, ordered[1:])):
            raise ValueError("REVISION_OVERLAPPING_REGIONS")
        return self

    def validate_session(self, session: SessionState) -> None:
        if (
            not session.connected or not session.project_path
            or not session.project_path.lower().endswith(".als")
        ):
            raise ValueError("REVISION_DURABLE_CONNECTED_SESSION_REQUIRED")
        if session.project_identity != self.project_identity:
            raise ValueError("REVISION_PROJECT_MISMATCH")
        if (
            session.project_token != self.project_token
            or session.audible_token != self.audible_token
            or session.session_incarnation_id != self.session_incarnation_id
        ):
            raise ValueError("REVISION_STALE_REQUEST")
        for stable_id in self.target_track_ids:
            matches = [track for track in session.tracks if track.stable_id == stable_id]
            if len(matches) != 1 or matches[0].role not in {"audio", "midi"}:
                raise ValueError("REVISION_TARGET_NOT_RESOLVED")


class ScopedRevisionProposal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request: ScopedRevisionRequest
    target_tokens: dict[str, str]
    preserved_track_tokens: dict[str, str]
    preserved_target_instrument_tokens: dict[str, str] = Field(default_factory=dict)
    musical_proposal: str = Field(min_length=1, max_length=4000)
    status: Literal["PROPOSED_NO_WRITE"] = "PROPOSED_NO_WRITE"
    no_write: Literal[True] = True
    limitations: list[str] = Field(default_factory=lambda: [
        "Existing-track phrase replacement is not certified by this proposal.",
        "Arrangement readback and instrument preservation need the certified executor.",
        "No mutation, rollback, disk save or audible judgment was performed.",
    ])

    def validate_preserved_tracks(self, after: SessionState) -> None:
        if (
            after.project_identity != self.request.project_identity
            or after.session_incarnation_id != self.request.session_incarnation_id
        ):
            raise ValueError("REVISION_POST_STATE_IDENTITY_MISMATCH")
        ids = [track.stable_id for track in after.tracks]
        expected = set(self.target_tokens) | set(self.preserved_track_tokens)
        if len(ids) != len(set(ids)) or set(ids) != expected:
            raise ValueError("REVISION_TRACK_SET_CHANGED")
        for track in after.tracks:
            expected_token = self.preserved_track_tokens.get(track.stable_id)
            if expected_token is not None and _preservation_token(track) != expected_token:
                raise ValueError(f"REVISION_OUT_OF_SCOPE_CHANGE:{track.stable_id}")
            instrument = self.preserved_target_instrument_tokens.get(track.stable_id)
            if instrument is not None and _instrument_token(track) != instrument:
                raise ValueError(f"REVISION_INSTRUMENT_CHANGED:{track.stable_id}")


def propose_scoped_revision(
    *, request: ScopedRevisionRequest, session: SessionState,
    state: ProducerState, musical_proposal: str,
) -> tuple[ScopedRevisionProposal, ProducerState]:
    request.validate_session(session)
    if state.project_identity != session.project_identity:
        raise ValueError("PRODUCER_STATE_PROJECT_IDENTITY_MISMATCH")
    ids = [track.stable_id for track in session.tracks]
    if any(not stable_id for stable_id in ids) or len(ids) != len(set(ids)):
        raise ValueError("REVISION_TRACK_IDENTITIES_REQUIRED")
    selected = set(request.target_track_ids)
    proposal = ScopedRevisionProposal(
        request=request, musical_proposal=musical_proposal,
        target_tokens={
            track.stable_id: _preservation_token(track)
            for track in session.tracks if track.stable_id in selected
        },
        preserved_track_tokens={
            track.stable_id: _preservation_token(track)
            for track in session.tracks if track.stable_id not in selected
        },
        preserved_target_instrument_tokens={
            track.stable_id: _instrument_token(track)
            for track in session.tracks
            if track.stable_id in selected and request.preserve_instruments
        },
    )
    updated = state.record(
        "SCOPED_REVISION_PROPOSED", phase=ProducerPhase.ADJUSTING,
        payload=proposal.model_dump(mode="json"),
    )
    updated.decisions["scoped_revision"] = proposal.model_dump(mode="json")
    return proposal, updated


class RetentionDecision(StrEnum):
    KEEP = "KEEP"
    RETOUCH = "RETOUCH"
    REPLACE = "REPLACE"


class ReviewedElement(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    role_or_section: str = Field(min_length=1, max_length=128)
    decision: RetentionDecision
    reason: str = Field(default="", max_length=1000)


class TrackSupervision(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    schema_version: Literal["track-supervision-v1"] = "track-supervision-v1"
    project_identity: str = Field(min_length=1)
    reviewed_als_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    elements: list[ReviewedElement] = Field(min_length=1, max_length=128)
    overall_judgment: str = Field(min_length=1, max_length=2000)
    human_retained_fraction: float | None = Field(default=None, ge=0, le=1)
    source: Literal["HUMAN"] = "HUMAN"


def record_track_supervision(
    state: ProducerState, review: TrackSupervision,
) -> ProducerState:
    """Store the user's verdict without certifying musical or disk completion."""
    if state.project_identity != review.project_identity:
        raise ValueError("SUPERVISION_PROJECT_MISMATCH")
    updated = state.record(
        "TRACK_SUPERVISION_RECORDED", payload=review.model_dump(mode="json"),
    )
    updated.observations["track_supervision"] = review.model_dump(mode="json")
    return updated
