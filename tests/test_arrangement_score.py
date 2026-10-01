"""Pure score contracts for Hermes QA; no DAW/provider mocks or audio."""

import pytest

from copilot.integration.autonomous_producer_alpha_v1 import LucasPlanningProviderAdapter
from copilot.musicplan import build_create_track_action, build_sample_load_action
from copilot.musicplan.score_compiler import (
    build_initial_score_material, compile_score_placements, initial_score_patterns,
    score_binding_blocker, verify_score_geometry,
)
from copilot.producer.arrangement_score import ArrangementScore, MidiPhrase
from copilot.producer.quality_gate import verify_arrangement_timeline
from copilot.producer.track_spec import TrackSpec
from copilot.schemas.musicplan import DiagnosisBinding, MusicPlan, ProductionActionKind
from copilot.schemas.session import ClipState, SessionState, TrackState


HASH = "a" * 64
VOCAL_HASH = "b" * 64
CAPABILITIES = {"clip.create", "clip.delete", "session.read"}


def spec(*, sparse_vocal=False):
    return TrackSpec(
        bpm=127, duration_bars=4,
        sections=[
            {"name": "Intro", "bars": 2, "energy": .4, "active_roles": ["Kick"]},
            {
                "name": "Drop", "bars": 2, "energy": .8,
                "active_roles": ["Kick", "Vocal"] if sparse_vocal else ["Kick"],
            },
        ],
    )


def score(*, sparse_vocal=False, distinct_phrase=False):
    phrase = {
        "length_qn": 8, "reason": "Original rhythm",
        "notes": [{"pitch": 60, "start_time": 0, "duration": .25, "velocity": 100}],
    }
    phrases = {"initial": phrase}
    if distinct_phrase:
        phrases["changed"] = {
            **phrase,
            "notes": [{"pitch": 62, "start_time": 1, "duration": .25, "velocity": 90}],
        }
    events = [
        {
            "event_id": "intro-kick", "section": "Intro", "role": "Kick",
            "source_sha256": HASH, "offset_qn": 0, "duration_qn": 8,
            "playback": "MIDI_PHRASE", "phrase_id": "initial", "reason": "Opening pulse",
        },
        {
            "event_id": "drop-kick", "section": "Drop", "role": "Kick",
            "source_sha256": HASH, "offset_qn": 0, "duration_qn": 8,
            "playback": "MIDI_PHRASE",
            "phrase_id": "changed" if distinct_phrase else "initial", "reason": "Drop pulse",
        },
    ]
    if sparse_vocal:
        events.append({
            "event_id": "vocal-entry", "section": "Drop", "role": "Vocal",
            "source_sha256": VOCAL_HASH, "offset_qn": 6, "duration_qn": 1,
            "playback": "SOURCE_ONCE", "reason": "One late vocal, with space afterward",
        })
    return ArrangementScore(phrases=phrases, events=events)


def session():
    return SessionState(
        connected=True, project_path="working.als", project_identity="project",
        project_token="project-token", audible_token="audible-token",
        session_incarnation_id="incarnation",
        transport={"tempo": 127},
        tracks=[
            TrackState(
                stable_id="kick-track", index=0, name="Kick", role="midi",
                clips=[ClipState(
                    stable_id="kick-source", slot_index=0, name="Kick",
                    length_beats=8, notes=score().phrases["initial"].midi_notes(),
                )],
            ),
            TrackState(
                stable_id="vocal-track", index=1, name="Vocal", role="audio",
                clips=[ClipState(
                    stable_id="vocal-source", slot_index=0, name="Vocal",
                    length_beats=1, is_audio=True, is_midi=False,
                )],
            ),
        ],
    )


def compile_score(*, musical_score=None, musical_spec=None, current=None, **kwargs):
    return compile_score_placements(
        score=musical_score or score(), spec=musical_spec or spec(),
        sample_map={"Kick": HASH, "Vocal": VOCAL_HASH},
        session=current or session(), evidence_refs=[HASH],
        capabilities=kwargs.pop("capabilities", CAPABILITIES),
        verified_material_roles=kwargs.pop("verified_material_roles", {"Kick", "Vocal"}),
        **kwargs,
    )


def geometry(compiled):
    indexes = {"Kick": 0, "Vocal": 1}
    return [
        {
            "track_index": indexes[binding.role],
            "start_time": binding.start_qn, "length": binding.duration_qn,
        }
        for binding in compiled.bindings
    ]


def test_phrases_reject_notes_outside_declared_duration():
    with pytest.raises(ValueError, match="NOTE_OUTSIDE_PHRASE"):
        MidiPhrase(
            length_qn=4, reason="Invalid",
            notes=[{"pitch": 60, "start_time": 3.9, "duration": 1, "velocity": 100}],
        )


def test_score_rejects_unselected_sources_missing_roles_and_overlaps():
    payload = score().model_dump(mode="json")
    payload["events"][0]["source_sha256"] = VOCAL_HASH
    with pytest.raises(ValueError, match="SOURCE_NOT_SELECTED"):
        ArrangementScore.model_validate(payload).validate_against(spec(), sample_map={"Kick": HASH})
    with pytest.raises(ValueError, match="ACTIVE_ROLE_HAS_NO_EVENT"):
        score().validate_against(spec(sparse_vocal=True), sample_map={"Kick": HASH})
    payload = score().model_dump(mode="json")
    payload["events"].append({**payload["events"][0], "event_id": "overlapping"})
    with pytest.raises(ValueError, match="EVENTS_OVERLAP"):
        ArrangementScore.model_validate(payload).validate_against(spec(), sample_map={"Kick": HASH})


def test_unfinished_timeline_is_not_a_complete_score():
    payload = score().model_dump(mode="json")
    payload["events"][1].update(playback="SOURCE_ONCE", phrase_id=None, duration_qn=4)
    with pytest.raises(ValueError, match="DURATION_MISMATCH"):
        ArrangementScore.model_validate(payload).validate_against(spec(), sample_map={"Kick": HASH})


def test_repetition_uses_real_eight_qn_clip_not_bridge_four_qn_tiling():
    compiled = compile_score()
    assert compiled.status == "PLACEMENTS_BOUND"
    assert compiled.no_write
    assert [action.params.destination_time for action in compiled.actions] == [0, 8]
    assert all(action.params.length is None for action in compiled.actions)
    assert [binding.duration_qn for binding in compiled.bindings] == [8, 8]
    assert session().tracks[0].clips[0].length_beats == 8


def test_sparse_vocal_geometry_is_explicit_and_not_continuous_coverage():
    compiled = compile_score(
        musical_score=score(sparse_vocal=True), musical_spec=spec(sparse_vocal=True),
    )
    rows = geometry(compiled)
    assert rows[-1] == {"track_index": 1, "start_time": 14, "length": 1}
    assert verify_arrangement_timeline(
        spec(sparse_vocal=True), tracks={"Kick": 0, "Vocal": 1},
        clips=rows, score_bindings=compiled.bindings,
    ) == []
    assert verify_arrangement_timeline(
        spec(sparse_vocal=True), tracks={"Kick": 0, "Vocal": 1}, clips=rows,
    ) == ["ARRANGEMENT_ROLE_GAP:Drop:Vocal"]


def test_distinct_phrase_is_deferred_without_generic_substitution():
    compiled = compile_score(musical_score=score(distinct_phrase=True))
    assert len(compiled.actions) == 1
    assert compiled.deferred[0].event_id == "drop-kick"
    assert compiled.deferred[0].reason == "SCORE_DISTINCT_MIDI_PHRASE_NOT_CERTIFIED"
    assert set(initial_score_patterns(score(distinct_phrase=True), spec())) == {"Kick"}


@pytest.mark.parametrize(
    ("kwargs", "reason"),
    [
        ({"capabilities": set()}, "SCORE_PLACEMENT_CAPABILITIES_UNAVAILABLE"),
        ({"verified_material_roles": set()}, "SCORE_SOURCE_MATERIAL_NOT_VERIFIED"),
    ],
)
def test_unavailable_capabilities_or_material_never_produce_actions(kwargs, reason):
    compiled = compile_score(**kwargs)
    assert not compiled.actions
    assert {row.reason for row in compiled.deferred} == {reason}


def test_source_trimming_and_action_budget_are_not_silently_approximated():
    current = session()
    current.tracks[1].clips[0].length_beats = 2
    compiled = compile_score(
        musical_score=score(sparse_vocal=True), musical_spec=spec(sparse_vocal=True),
        current=current,
    )
    assert compiled.deferred[0].reason == "SCORE_SOURCE_CLIP_TRIM_OR_STRETCH_NOT_CERTIFIED"
    with pytest.raises(ValueError, match="ACTION_BUDGET_EXCEEDED"):
        compile_score(max_actions=1)


def test_source_changes_between_binding_and_execution_fail_closed():
    binding = compile_score().bindings[0]
    assert score_binding_blocker(binding, session()) is None
    current = session()
    current.tracks[0].clips[0].notes[0].pitch = 61
    assert score_binding_blocker(binding, current) == "SCORE_BOUND_SOURCE_CHANGED"
    current = session()
    current.tracks[0].clips[0].sample_uri = "different-source.wav"
    assert score_binding_blocker(binding, current) == "SCORE_BOUND_SOURCE_CHANGED"
    current = session().model_copy(update={"project_identity": "other"})
    assert score_binding_blocker(binding, current) == "SCORE_BOUND_SESSION_CHANGED"
    current = session()
    current.transport.tempo = 128
    assert score_binding_blocker(binding, current) == "SCORE_BOUND_TIMING_CHANGED"


def test_live_meter_must_match_the_score_before_placement():
    current = session()
    current.transport.signature_numerator = 3
    with pytest.raises(ValueError, match="SESSION_TIMING_MISMATCH"):
        compile_score(current=current)


def test_automation_is_reported_as_deferred_not_as_a_transition_write():
    payload = score().model_dump(mode="json")
    payload["transitions"] = [{
        "after_section": "Intro", "kind": "AUTOMATION", "reason": "Filter opening",
    }]
    compiled = compile_score(musical_score=ArrangementScore.model_validate(payload))
    assert compiled.deferred[0].reason == "SCORE_TRANSITION_AUTOMATION_NOT_CERTIFIED"
    assert all(
        action.action_type is ProductionActionKind.DUPLICATE_CLIP_TO_ARRANGEMENT
        for action in compiled.actions
    )


def test_wrong_geometry_or_inconsistent_end_time_is_rejected():
    compiled = compile_score()
    rows = geometry(compiled)
    rows[0]["start_time"] = 1
    assert verify_score_geometry(
        compiled.bindings, tracks={"Kick": 0}, clips=rows,
    ) == ["SCORE_ARRANGEMENT_EVENT_GEOMETRY_MISMATCH"]
    rows = geometry(compiled)
    rows[0]["end_time"] = 100
    assert verify_score_geometry(
        compiled.bindings, tracks={"Kick": 0}, clips=rows,
    ) == ["SCORE_ARRANGEMENT_GEOMETRY_INVALID"]


def test_initial_material_uses_score_notes_and_existing_builders():
    current = session()
    target = TrackState(stable_id="", index=-1, name="Kick", role="audio")
    base = MusicPlan(
        plan_id="score-plan", created_at="2026-10-01T00:00:00Z",
        diagnosis=DiagnosisBinding(
            diagnosis_id="intent", diagnosis_status="intent", diagnosis_accepted=True,
        ),
        project_state_token=current.project_token, audible_state_token=current.audible_token,
        actions=[
            build_create_track_action(
                project_identity="project", track_name="Kick", track_kind="audio",
                reason="Initial source", evidence_refs=[HASH],
            ),
            build_sample_load_action(
                track=target, project_identity="project", clip_index=0,
                sample_uri="kick.wav", reason="Selected source", evidence_refs=[HASH],
            ),
        ],
    )
    result = build_initial_score_material(
        base_plan=base, score=score(), spec=spec(), session=current,
    )
    assert [action.action_type for action in result.actions] == [
        ProductionActionKind.CREATE_TRACK, ProductionActionKind.SAMPLE_LOAD,
        ProductionActionKind.CREATE_PATTERN,
    ]
    assert result.actions[0].params.track_kind == "midi"
    assert result.actions[2].params.notes == score().phrases["initial"].midi_notes()
    assert len(base.actions) == 2


def test_complete_track_provider_schema_requires_score_and_resolves_refs():
    legacy = LucasPlanningProviderAdapter._schema()
    assert "arrangement_score" not in legacy["properties"]
    schema = LucasPlanningProviderAdapter._schema(complete_track=True)
    assert {"arrangement_score", "track_spec"} <= set(schema["required"])

    def check_refs(value):
        if isinstance(value, dict):
            if "$ref" in value:
                assert value["$ref"].startswith("#/$defs/")
                assert value["$ref"].split("/")[-1] in schema["$defs"]
            for nested in value.values():
                check_refs(nested)
        elif isinstance(value, list):
            for nested in value:
                check_refs(nested)

    check_refs(schema)
