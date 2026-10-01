"""Pure contract checks; no DAW adapter, model provider or audio processing."""

import hashlib

import pytest

from copilot.integration.lucas_core_v1 import _grounded_intent, build_lucas_input
from copilot.musicplan.astra_plan import build_astra_prompt
from copilot.producer.context import (
    MusicalFamily, ProducerBrief, ProducerContext, lucas_producer_brief,
    prepare_producer_context,
)
from copilot.producer.track_spec import TrackSpec
from copilot.sample_library.schemas import (
    BpmEstimate, LibraryIndex, SampleAsset, SampleRole, SampleSetContext, SampleType,
)
from copilot.schemas.lucas_integration import ProjectContext, ReferenceContext, UserIntent


def reference(identity="reference"):
    return ReferenceContext(
        identity=identity, reference_state_token=f"reference:{identity}", tempo_bpm=127,
        evidence_refs=[f"evidence:{identity}"],
    )


def indexed_asset(tmp_path, *, role=SampleRole.KICK, kind=SampleType.ONE_SHOT, bpm=None):
    source = tmp_path / f"{role.value}.bin"
    source.write_bytes(role.value.encode("ascii"))
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    asset = SampleAsset(
        id=digest, sha256=digest, path=str(source), filename=source.name,
        library_root=str(tmp_path), relative_path=source.name,
        extension=".bin", size_bytes=source.stat().st_size,
        semantic_role=role, sample_type=kind, bpm=BpmEstimate(value=bpm),
    )
    return LibraryIndex(assets={digest: asset}), digest


def test_lucas_direction_is_opt_in_and_not_semantic_listening():
    brief = lucas_producer_brief()
    assert brief.artist_direction == ["Nacho Scoppa", "Jay de Lys"]
    assert brief.internal_audio_capture_authorized is False
    assert brief.final_render is False
    assert brief.authorized_sample_sha256 == []
    assert brief.style_context().provenance["semantic_listening"] is False


def test_initial_context_is_blocked_not_an_invented_project():
    context = prepare_producer_context(brief=lucas_producer_brief(), references=[])
    assert "REFERENCE_EVIDENCE_REQUIRED" in context.planning_blockers()
    assert "MATERIAL_REQUIRED:VOCALS" in context.planning_blockers()
    assert context.no_write


def test_one_shot_without_bpm_is_not_filtered_out(tmp_path):
    index, digest = indexed_asset(tmp_path)
    brief = ProducerBrief(
        intent="Complete drum arrangement", required_families=[MusicalFamily.DRUMS],
        authorized_sample_sha256=[digest],
    )
    context = prepare_producer_context(
        brief=brief, references=[reference()], index=index,
        authorized_root=tmp_path, bpm=127,
    )
    assert context.samples.roles["Kick"][0]["sha256"] == digest
    assert not context.planning_blockers()
    assert context.samples.roles["Kick"][0]["permission"] == "USER_AUTHORIZED_FOR_THIS_PRODUCTION"


def test_unapproved_material_is_not_selected(tmp_path):
    index, _ = indexed_asset(tmp_path)
    context = prepare_producer_context(
        brief=ProducerBrief(intent="Drums", required_families=[MusicalFamily.DRUMS]),
        references=[reference()], index=index, authorized_root=tmp_path,
    )
    assert not context.samples.candidates
    assert "MATERIAL_REQUIRED:Kick" in context.planning_blockers()


def test_whole_library_permission_does_not_require_per_sample_approval(tmp_path):
    index, digest = indexed_asset(tmp_path)
    brief = ProducerBrief(
        intent="Drums", required_families=[MusicalFamily.DRUMS],
        authorized_library_root=str(tmp_path),
    )
    context = prepare_producer_context(
        brief=brief, references=[reference()], index=index, authorized_root=tmp_path,
    )
    assert context.samples.roles["Kick"][0]["sha256"] == digest
    assert context.samples.roles["Kick"][0]["permission"] == "USER_AUTHORIZED_LIBRARY"
    assert not context.planning_blockers()


def test_library_permission_is_bound_to_its_root(tmp_path):
    index, _ = indexed_asset(tmp_path)
    other = tmp_path / "other"
    other.mkdir()
    with pytest.raises(ValueError, match="PERMISSION_ROOT_MISMATCH"):
        prepare_producer_context(
            brief=ProducerBrief(intent="Drums", authorized_library_root=str(other)),
            references=[reference()], index=index, authorized_root=tmp_path,
        )


def test_changed_bytes_fail_before_planning(tmp_path):
    index, digest = indexed_asset(tmp_path)
    brief = ProducerBrief(intent="Drums", authorized_sample_sha256=[digest])
    (tmp_path / "KICK.bin").write_bytes(b"changed")
    with pytest.raises(ValueError, match="DIGEST_MISMATCH"):
        prepare_producer_context(
            brief=brief, references=[reference()], index=index, authorized_root=tmp_path,
        )


def test_duplicate_reference_tokens_fail_closed():
    with pytest.raises(ValueError, match="DUPLICATE_REFERENCE"):
        ProducerContext(
            brief=lucas_producer_brief(), references=[reference(), reference()],
            samples=SampleSetContext(task_id="task"),
        )


def test_multiple_references_keep_their_evidence_and_no_listening_claim():
    context = prepare_producer_context(
        brief=lucas_producer_brief(), references=[reference("one"), reference("two")],
    )
    prompt = build_astra_prompt(
        candidates={}, intent="plan", production_context=context, strict=True,
    )
    assert "evidence:one" in prompt and "evidence:two" in prompt
    assert "No previews were rendered or auditioned" in prompt
    assert "no EDM risers" not in prompt
    assert "never claim listening" in prompt


def test_style_and_reference_facts_reach_core_planner_prompt():
    context = build_lucas_input(
        user_intent=UserIntent(description="Complete track"), reference=reference(),
        samples=SampleSetContext(task_id="task"), style=lucas_producer_brief().style_context(),
        project=ProjectContext(
            project_identity="project", project_token="token",
            audible_token="audible", session_incarnation_id="incarnation",
        ),
    )
    intent = _grounded_intent(context)
    assert "Nacho Scoppa" in intent
    assert "Percussion and bass should interact" in intent
    assert "evidence:reference" in intent


def test_required_families_cannot_be_silently_omitted(tmp_path):
    index, digest = indexed_asset(tmp_path)
    context = prepare_producer_context(
        brief=ProducerBrief(intent="Full track", authorized_sample_sha256=[digest]),
        references=[reference()], index=index, authorized_root=tmp_path,
    )
    spec = TrackSpec(bpm=127, sections=[
        {"name": "Intro", "bars": 8, "energy": .2, "active_roles": ["Kick"]},
    ])
    with pytest.raises(ValueError, match="REQUIRED_FAMILIES_MISSING"):
        context.validate_track_spec(spec)
