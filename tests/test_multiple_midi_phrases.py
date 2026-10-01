"""Hermes pure-contract QA for experimental phrases; no DAW/provider mocks."""

import pytest

from copilot.daw.state_tokens import attach_tokens
from copilot.daw.object_ref import ref_from_track
from copilot.integration.lucas_core_v1 import _single_action_plan
from copilot.musicplan.score_compiler import compile_score_placements, score_binding_blocker
from copilot.musicplan.score_material import (
    bind_verified_phrase_slots, phrase_material_blocker, prepare_score_phrase_material,
)
from copilot.producer.arrangement_score import ArrangementScore
from copilot.producer.midi_phrase_policy import MIDI_PHRASE_CAPABILITIES, phrase_preservation_token
from copilot.producer.track_spec import TrackSpec
from copilot.runtime.production_compiler import ProductionCompiler
from copilot.runtime.safe_write import action_is_certified_for_intent
from copilot.schemas.safe_write import KIND_EXPERIMENTAL_MIDI_PHRASE_V1
from copilot.schemas.session import ClipState, DeviceState, SessionState, TrackState


HASH = "a" * 64


def musical_spec():
    return TrackSpec(
        bpm=127, sections=[
            {"name": "Intro", "bars": 2, "energy": .4, "active_roles": ["Bass"]},
            {"name": "Drop", "bars": 2, "energy": .8, "active_roles": ["Bass"]},
        ],
    )


def musical_score(*, identical=False):
    first = {
        "length_qn": 8, "reason": "Original bass",
        "notes": [{"pitch": 36, "start_time": 1, "duration": .5, "velocity": 100}],
    }
    second = first if identical else {
        **first, "notes": [{"pitch": 38, "start_time": 2, "duration": .25, "velocity": 90}],
    }
    return ArrangementScore(
        phrases={"first": first, "second": second},
        events=[
            {
                "event_id": "intro", "section": "Intro", "role": "Bass",
                "source_sha256": HASH, "offset_qn": 0, "duration_qn": 8,
                "playback": "MIDI_PHRASE", "phrase_id": "first", "reason": "Intro bass",
            },
            {
                "event_id": "drop", "section": "Drop", "role": "Bass",
                "source_sha256": HASH, "offset_qn": 0, "duration_qn": 8,
                "playback": "MIDI_PHRASE", "phrase_id": "second", "reason": "Changed bass",
            },
        ],
    )


def observed_session():
    current = SessionState(
        connected=True, project_path="working.als", project_identity="project",
        session_incarnation_id="incarnation", transport={"tempo": 127},
        tracks=[
            TrackState(
                stable_id="bass", index=0, name="Bass", role="midi",
                clip_slot_count=3, empty_clip_slots=[1, 2],
                clips=[ClipState(
                    stable_id="initial", slot_index=0, name="Initial bass", length_beats=8,
                    notes=musical_score().phrases["first"].midi_notes(),
                )],
                devices=[DeviceState(
                    stable_id="instrument", index=0, name="Simpler", sample_uri="bass.wav",
                )],
            ),
            TrackState(stable_id="other", index=1, name="Other", role="audio"),
        ],
    )
    attach_tokens(current)
    return current


def prepare(*, current=None, score=None, enabled=True, owned=frozenset({"bass"}),
            capabilities=MIDI_PHRASE_CAPABILITIES, **kwargs):
    return prepare_score_phrase_material(
        score=score or musical_score(), spec=musical_spec(), sample_map={"Bass": HASH},
        session=current or observed_session(), owned_midi_track_ids=owned,
        verified_material_roles={"Bass"}, verified_material_devices={"Bass": "instrument"},
        capabilities=capabilities, enabled=enabled, **kwargs,
    )


def phrase_readback():
    current = observed_session()
    current.tracks[0].clips.append(ClipState(
        stable_id="new-phrase", slot_index=1, name="New bass", length_beats=8,
        notes=musical_score().phrases["second"].midi_notes(),
    ))
    current.tracks[0].empty_clip_slots = [2]
    attach_tokens(current)
    return current


def test_one_instrument_two_phrases_and_no_writes_during_planning():
    current = observed_session()
    material = prepare(current=current)
    assert material.no_write
    assert len(material.actions) == 1
    assert [(item.phrase_id, item.clip_index) for item in material.assignments] == [
        ("first", 0), ("second", 1),
    ]
    assert len(current.tracks[0].clips) == 1
    assert current.tracks[0].devices[0].stable_id == "instrument"


def test_identical_phrases_reuse_verified_initial_source():
    material = prepare(score=musical_score(identical=True))
    assert not material.actions
    assert [item.clip_index for item in material.assignments] == [0, 0]


@pytest.mark.parametrize(
    ("kwargs", "reason"),
    [
        ({"enabled": False}, "MIDI_PHRASE_EXPERIMENT_DISABLED"),
        ({"capabilities": frozenset()}, "MIDI_PHRASE_CAPABILITIES_UNAVAILABLE"),
        ({"owned": frozenset()}, "MIDI_PHRASE_OWNED_VERIFIED_MATERIAL_REQUIRED"),
    ],
)
def test_disabled_unowned_or_unsupported_material_never_creates_actions(kwargs, reason):
    material = prepare(**kwargs)
    assert not material.actions
    assert {row.reason for row in material.deferred} == {reason}


def test_unknown_or_exhausted_slot_inventory_does_not_invent_scenes():
    current = observed_session()
    current.tracks[0].empty_clip_slots = None
    assert prepare(current=current).deferred[0].reason == "MIDI_PHRASE_SLOT_INVENTORY_UNAVAILABLE"
    current = observed_session()
    current.tracks[0].clip_slot_count = 1
    current.tracks[0].empty_clip_slots = []
    assert prepare(current=current).deferred[0].reason == "MIDI_PHRASE_EMPTY_SLOTS_EXHAUSTED"


def test_initial_phrase_or_verified_instrument_must_still_exist():
    current = observed_session()
    current.tracks[0].clips[0].notes[0].pitch = 39
    material = prepare(current=current)
    assert not material.actions
    assert {row.reason for row in material.deferred} == {"MIDI_PHRASE_INITIAL_READBACK_MISMATCH"}
    current = observed_session()
    current.tracks[0].devices = []
    material = prepare(current=current)
    assert not material.actions
    assert {row.reason for row in material.deferred} == {"MIDI_PHRASE_VERIFIED_INSTRUMENT_MISSING"}


def test_pending_slot_occupation_and_unrelated_changes_fail_closed():
    material = prepare()
    current = phrase_readback()
    assert phrase_material_blocker(
        material, score=musical_score(), session=current, verified_clip_ids={},
    ) == "MIDI_PHRASE_RESERVED_SLOT_CHANGED"
    current = observed_session()
    current.tracks[1].mixer.volume = .2
    assert phrase_material_blocker(
        material, score=musical_score(), session=current, verified_clip_ids={},
    ) == "MIDI_PHRASE_MATERIAL_BASELINE_CHANGED"


def test_verified_phrases_bind_to_distinct_slots_and_actual_clip_ids():
    material = prepare()
    proof = {material.actions[0].action_id: "new-phrase"}
    current = phrase_readback()
    bindings = bind_verified_phrase_slots(
        material, score=musical_score(), sample_map={"Bass": HASH},
        session=current, verified_clip_ids=proof,
    )
    compiled = compile_score_placements(
        score=musical_score(), spec=musical_spec(), sample_map={"Bass": HASH},
        session=current, capabilities=MIDI_PHRASE_CAPABILITIES,
        evidence_refs=[HASH], verified_material_roles={"Bass"}, verified_phrase_slots=bindings,
    )
    assert not compiled.deferred
    assert [action.params.clip_index for action in compiled.actions] == [0, 1]
    assert [binding.source_clip_stable_id for binding in compiled.bindings] == ["initial", "new-phrase"]
    assert score_binding_blocker(compiled.bindings[1], current) is None
    current.tracks[0].clips[1].notes[0].pitch = 39
    assert score_binding_blocker(compiled.bindings[1], current) == "SCORE_BOUND_SOURCE_CHANGED"


def test_unverified_source_cannot_be_adopted_just_because_notes_match():
    material = prepare()
    with pytest.raises(ValueError, match="RESERVED_SLOT_CHANGED"):
        bind_verified_phrase_slots(
            material, score=musical_score(), sample_map={"Bass": HASH},
            session=phrase_readback(), verified_clip_ids={},
        )


def test_preservation_includes_instruments_and_outside_regions():
    before = observed_session()
    after = phrase_readback()
    excluded = {"bass": {1}}
    assert phrase_preservation_token(before, excluded_slots=excluded) == (
        phrase_preservation_token(after, excluded_slots=excluded)
    )
    after.tracks[0].devices[0].sample_uri = "another-instrument.wav"
    assert phrase_preservation_token(before, excluded_slots=excluded) != (
        phrase_preservation_token(after, excluded_slots=excluded)
    )


def test_default_compiler_does_not_certify_or_enable_new_phrase_action():
    current = observed_session()
    action = prepare(current=current).actions[0]
    plan = _single_action_plan(action, session=current, plan_id="extra-phrase")
    rejected = ProductionCompiler().compile(plan, session=current)
    assert rejected.status == "PLAN_REJECTED"
    assert rejected.reasons == ("MIDI_PHRASE_EXPERIMENT_DISABLED",)
    experimental = ProductionCompiler(
        experimental_midi_track_ids=frozenset({"bass"}),
        negotiated_capabilities=MIDI_PHRASE_CAPABILITIES,
    ).compile(plan, session=current)
    assert experimental.status == "COMPILED_EXPERIMENTAL"
    assert experimental.intent.kind == KIND_EXPERIMENTAL_MIDI_PHRASE_V1
    assert not experimental.intent.executions[0].certified
    assert not experimental.certified_action_ids
    assert not action_is_certified_for_intent(experimental.intent, "CREATE_MIDI_PHRASE")


def test_compiler_rejects_occupied_slot_even_with_experiment_authorization():
    current = phrase_readback()
    action = prepare().actions[0]
    action = action.model_copy(update={"target": action.target.model_copy(update={
        "ref": ref_from_track(
            current.tracks[0], project_identity=current.project_identity,
        ).model_dump(mode="json"),
    })})
    plan = _single_action_plan(action, session=current, plan_id="occupied-phrase")
    result = ProductionCompiler(
        experimental_midi_track_ids=frozenset({"bass"}),
        negotiated_capabilities=MIDI_PHRASE_CAPABILITIES,
    ).compile(plan, session=current)
    assert result.reasons == ("MIDI_PHRASE_SLOT_OCCUPIED",)
