"""Planning and ledger checks without mock Live, providers or audio."""

import pytest

from copilot.producer.state import ProducerPhase, ProducerState
from copilot.producer.supervision import (
    ScopedRevisionRequest, TrackSupervision, propose_scoped_revision,
    record_track_supervision,
)
from copilot.schemas.session import SessionState, TrackState


def session():
    return SessionState(
        connected=True, project_path="working.als",
        project_identity="project", project_token="current",
        audible_token="audible", session_incarnation_id="incarnation",
        tracks=[
            TrackState(stable_id="bass", index=0, name="Bass", role="midi"),
            TrackState(stable_id="drums", index=1, name="Drums", role="audio"),
        ],
    )


def request():
    return ScopedRevisionRequest(
        instruction="Change the bass line",
        project_identity="project", project_token="current",
        audible_token="audible", session_incarnation_id="incarnation",
        target_track_ids=["bass"], regions=[{"start_qn": 0, "end_qn": 32}],
    )


def state():
    return ProducerState(
        session_id="producer", project_identity="project",
        decisions={"existing_choice": "preserve"},
    )


def test_scoped_proposal_does_not_execute_or_replace_existing_decisions():
    proposal, updated = propose_scoped_revision(
        request=request(), session=session(), state=state(),
        musical_proposal="More space between bass notes while preserving its instrument",
    )
    assert proposal.status == "PROPOSED_NO_WRITE"
    assert proposal.no_write
    assert updated.decisions["existing_choice"] == "preserve"
    assert updated.phase is ProducerPhase.ADJUSTING
    proposal.validate_preserved_tracks(session())


def test_stale_request_cannot_overwrite_manual_changes():
    current = session().model_copy(update={"audible_token": "human-edited"})
    with pytest.raises(ValueError, match="STALE_REQUEST"):
        propose_scoped_revision(
            request=request(), session=current, state=state(), musical_proposal="new bass",
        )


def test_ambiguous_stable_id_fails_closed():
    current = session()
    current.tracks.append(current.tracks[0].model_copy())
    with pytest.raises(ValueError, match="TARGET_NOT_RESOLVED"):
        propose_scoped_revision(
            request=request(), session=current, state=state(), musical_proposal="new bass",
        )


def test_unrelated_track_changes_are_rejected():
    proposal, _ = propose_scoped_revision(
        request=request(), session=session(), state=state(), musical_proposal="new bass",
    )
    after = session()
    after.tracks[1].mixer.volume = .2
    with pytest.raises(ValueError, match="OUT_OF_SCOPE_CHANGE:drums"):
        proposal.validate_preserved_tracks(after)


def test_human_effectiveness_does_not_mark_track_complete():
    review = TrackSupervision(
        project_identity="project", reviewed_als_sha256="a" * 64,
        elements=[{"role_or_section": "Bass", "decision": "RETOUCH"}],
        overall_judgment="I would finish this with minor changes",
        human_retained_fraction=.75,
    )
    updated = record_track_supervision(state(), review)
    assert updated.phase is ProducerPhase.PLANNING
    assert updated.observations["track_supervision"]["human_retained_fraction"] == .75
    assert state().observations == {}
