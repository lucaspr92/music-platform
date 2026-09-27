from pathlib import Path

import numpy as np
import soundfile as sf
import pytest

from copilot.audio.music_analyzer import analyze_reference_music
from copilot.daw.mock import MockAbletonAdapter
from copilot.daw.state_tokens import attach_tokens, target_token
from copilot.integration.lucas_core_v1 import (
    bound_plan_to_certified_actions,
    build_lucas_input,
    build_project_context,
    build_reference_context,
    constrain_plan_to_lucas_intent,
    execute_lucas_patch_contracts_through_core,
    execute_lucas_plan_through_core,
    normalize_sample_uri_for_working_copy,
    rebind_sample_load_action,
    rebind_sample_load_plan,
    run_lucas_critique,
    run_lucas_critique_with_provider_failover,
    run_lucas_planner,
)
from copilot.musicplan import build_create_track_action, build_pattern_action, build_sample_load_action
from copilot.sample_library.schemas import (
    AudioDescriptors,
    BpmEstimate,
    LibraryIndex,
    SampleAsset,
    SampleRole,
    SampleSetContext,
    SampleType,
)
from copilot.schemas.lucas_integration import StyleContext, UserIntent
from copilot.schemas.musicplan import (
    DiagnosisBinding,
    MusicPlan,
    PlanIntentClass,
    PlanStatus,
    ProductionActionKind,
)
from copilot.human_eval.store import now_iso
from copilot.reasoning.errors import ReasoningFailure
from copilot.reasoning.provider import FailingProvider, ScriptedProvider


def _reference(tmp_path: Path):
    sr = 16_000
    path = tmp_path / "reference.wav"
    time = np.arange(sr * 2) / sr
    sf.write(path, (0.2 * np.sin(2 * np.pi * 110 * time)).astype(np.float32), sr)
    return analyze_reference_music(
        path,
        reference_state_token="reference:integration",
        target_state_token="target:integration",
        tempo_bpm=126,
        use_cache=False,
    )


def _session():
    daw = MockAbletonAdapter()
    daw.connect()
    daw.create_audio_track("Existing")
    session = daw.snapshot()
    attach_tokens(session)
    return daw, session


def _index(tmp_path: Path) -> LibraryIndex:
    sample = tmp_path / "kick.wav"
    sf.write(sample, np.zeros(16_000, dtype=np.float32), 16_000)
    asset = SampleAsset(
        id="sha:kick",
        path=str(sample),
        filename="kick.wav",
        library_root=str(tmp_path),
        relative_path="Samples/Imported/kick.wav",
        extension=".wav",
        size_bytes=sample.stat().st_size,
        sha256="sha:kick",
        sample_type=SampleType.ONE_SHOT,
        semantic_role=SampleRole.KICK,
        bpm=BpmEstimate(value=126, confidence=1),
        descriptors=AudioDescriptors(duration_s=1),
    )
    return LibraryIndex(roots=[str(tmp_path)], assets={asset.sha256: asset})


def _plan(session, actions):
    return MusicPlan(
        plan_id="lucas-plan",
        status=PlanStatus.DRAFT,
        intent_class=PlanIntentClass.CONTROLLED_ENGINEERING_VALIDATION,
        diagnosis=DiagnosisBinding(
            diagnosis_id="lucas",
            diagnosis_status="SUPPORTED",
            diagnosis_accepted=True,
        ),
        project_state_token=session.project_token,
        audible_state_token=session.audible_token,
        created_at=now_iso(),
        actions=actions,
    )


def test_contexts_are_typed_no_write_and_preserve_reference_provenance(tmp_path: Path):
    _, session = _session()
    pack = _reference(tmp_path)
    reference = build_reference_context(pack)
    samples = SampleSetContext(
        task_id="integration",
        roles={"KICK": [{"id": "sha:kick", "filename": "kick.wav"}]},
        candidates=[{"id": "sha:kick", "filename": "kick.wav"}],
    )
    context = build_lucas_input(
        user_intent=UserIntent(description="build a restrained groove"),
        reference=reference,
        samples=samples,
        style=StyleContext(),
        project=build_project_context(session),
    )

    assert context.no_write is True
    assert context.reference.reference_state_token == "reference:integration"
    assert context.samples.stable_sample_ids == ["sha:kick"]
    assert context.project.tracks[0].stable_id == session.tracks[0].stable_id


def test_lucas_planner_output_is_typed_and_grounded(tmp_path: Path):
    _, session = _session()
    pack = _reference(tmp_path)
    samples = SampleSetContext(task_id="integration", candidates=[{"id": "sha:kick"}])
    input_context = build_lucas_input(
        user_intent=UserIntent(description="build a restrained groove"),
        reference=build_reference_context(pack),
        samples=samples,
        project=build_project_context(session),
    )
    index = _index(tmp_path)

    def fake_planner(**kwargs):
        action = build_create_track_action(
            project_identity=kwargs["session"].project_identity,
            track_name="Kick",
            track_kind="audio",
            reason="Lucas selects a kick role",
            evidence_refs=["sha:kick"],
        )
        return _plan(kwargs["session"], [action]), {"lucas": "stable-test"}

    run = run_lucas_planner(
        input_context=input_context,
        index=index,
        session=session,
        planner=fake_planner,
    )
    assert run.plan.schema_version == "musicplan-v1"
    assert "reference:integration" in run.plan.notes[-2]
    assert run.planner_metadata["lucas"] == "stable-test"


def test_lucas_critique_adapter_calls_lucas_surface_without_write(monkeypatch):
    _, session = _session()
    plan = _plan(session, [])
    observed = {}

    def fake_critique(**kwargs):
        observed.update(kwargs)
        from copilot.musicplan.critique import CritiqueResult

        return CritiqueResult(verdict="improve")

    monkeypatch.setattr("copilot.musicplan.critique.critique_track", fake_critique)
    result = run_lucas_critique(
        plan=plan,
        session=session,
        provider=object(),
        timeout_s=7.0,
    )

    assert result is not None
    assert result.verdict == "improve"
    assert observed["plan"] is plan
    assert observed["session"] is session
    assert observed["timeout_s"] == 7.0


def test_lucas_critique_provider_failover_preserves_contract_and_evidence():
    _, session = _session()
    plan = _plan(session, [])
    fallback = ScriptedProvider(
        {"*": '{"verdict":"improve","top_3_issues":[],"reasoning":"grounded fallback"}'},
        identity="configured-fallback",
    )
    primary = FailingProvider(ReasoningFailure.MODEL_TIMEOUT, "primary timeout")

    result = run_lucas_critique_with_provider_failover(
        plan=plan,
        session=session,
        providers=[primary, fallback],
        evidence_context={"pre_rms": 0.19, "post_rms": 0.17},
        timeout_s=1.0,
    )

    assert result["status"] == "CRITIQUE_COMPLETE"
    assert result["provider"] == "configured-fallback"
    assert result["result"].verdict == "improve"
    assert '"pre_rms": 0.19' in fallback.last_prompt
    assert result["attempts"][0]["status"] == "FAILED"
    assert result["attempts"][1]["status"] == "SUCCESS"


def test_lucas_critique_provider_failover_reports_all_failures_without_verdict():
    _, session = _session()
    plan = _plan(session, [])

    result = run_lucas_critique_with_provider_failover(
        plan=plan,
        session=session,
        providers=[FailingProvider(ReasoningFailure.MODEL_TIMEOUT, "timeout")],
        timeout_s=1.0,
    )

    assert result["status"] == "CRITIQUE_PROVIDER_UNAVAILABLE"
    assert result["result"] is None
    assert result["attempts"][0]["failure"]["type"] == "ProviderError"


def test_lucas_critique_invalid_provider_output_is_not_a_verdict():
    _, session = _session()
    plan = _plan(session, [])

    result = run_lucas_critique_with_provider_failover(
        plan=plan,
        session=session,
        providers=[ScriptedProvider({"*": "not-json"}, identity="invalid-provider")],
        timeout_s=1.0,
    )

    assert result["status"] == "CRITIQUE_PROVIDER_UNAVAILABLE"
    assert result["result"] is None
    assert result["attempts"][0]["failure"]["type"] == "INVALID_SCHEMA_OR_CRITIQUE_FAILURE"


def test_stale_project_context_fails_closed(tmp_path: Path):
    _, session = _session()
    pack = _reference(tmp_path)
    context = build_lucas_input(
        user_intent=UserIntent(description="build"),
        reference=build_reference_context(pack),
        samples=SampleSetContext(task_id="integration"),
        project=build_project_context(session).model_copy(update={"project_token": "stale"}),
    )
    with pytest.raises(ValueError, match="STALE_PLAN|PROJECT_MISMATCH"):
        run_lucas_planner(
            input_context=context,
            index=_index(tmp_path),
            session=session,
            planner=lambda **kwargs: (_plan(kwargs["session"], []), {}),
        )


def test_bounded_selection_reports_unsupported_actions_without_dropping_them():
    _, session = _session()
    track = session.tracks[0]
    create = build_create_track_action(
        project_identity=session.project_identity,
        track_name="Kick",
        track_kind="audio",
        reason="create kick",
        evidence_refs=["sha:kick"],
    )
    pattern = build_pattern_action(
        track=track,
        project_identity=session.project_identity,
        clip_index=0,
        length_beats=1,
        notes=[],
        reason="pattern is deferred",
        evidence_refs=[],
    )
    bounded = bound_plan_to_certified_actions(_plan(session, [create, pattern]), action_ids=[create.action_id])

    assert len(bounded.plan.actions) == 1
    assert bounded.accepted[0].action_id == create.action_id
    assert bounded.deferred[0].status == "DEFERRED_UNSUPPORTED"
    assert bounded.deferred[0].action_id == pattern.action_id


def test_sample_action_rebinds_to_authoritative_track_without_mutating_lucas_plan():
    _, session = _session()
    virtual = session.tracks[0].model_copy(update={"name": "Kick"})
    action = build_sample_load_action(
        track=virtual,
        project_identity=session.project_identity,
        clip_index=0,
        sample_uri="query:CurrentProject#Samples:Imported:kick.wav",
        reason="Lucas selected kick.wav",
        evidence_refs=["sha:kick"],
    )
    rebound = rebind_sample_load_action(action, track=session.tracks[0], session=session)

    assert action.target.ref["name"] == "Kick"
    assert rebound.action_id == action.action_id
    assert rebound.target.runtime_id["stable_id"] == session.tracks[0].stable_id
    assert rebound.params.sample_uri == action.params.sample_uri


def test_sample_plan_rebind_mints_authoritative_target_token():
    _, session = _session()
    virtual = session.tracks[0].model_copy(update={"name": "Kick"})
    action = build_sample_load_action(
        track=virtual,
        project_identity=session.project_identity,
        clip_index=0,
        sample_uri="query:CurrentProject#Samples:Imported:kick.wav",
        reason="Lucas selected kick.wav",
        evidence_refs=["sha:kick"],
    )
    plan = _plan(session, [action])
    rebound = rebind_sample_load_plan(plan, action, track=session.tracks[0], session=session)

    assert rebound.target_state_tokens == {
        session.tracks[0].name: target_token(session.tracks[0])
    }
    assert rebound.actions[0].target.runtime_id["stable_id"] == session.tracks[0].stable_id


def test_sample_uri_normalization_stays_in_core_adapter():
    assert normalize_sample_uri_for_working_copy("Samples/Imported/kick.wav") == (
        "query:CurrentProject#Samples:Imported:kick.wav"
    )
    assert normalize_sample_uri_for_working_copy("query:CurrentProject#Samples:Imported/kick.wav").startswith(
        "query:"
    )


def test_supported_lucas_plan_uses_core_safe_write_authority(tmp_path: Path):
    daw = MockAbletonAdapter()
    daw.connect()
    session = daw.snapshot()
    attach_tokens(session)
    action = build_create_track_action(
        project_identity=session.project_identity,
        track_name="Core Lucas Smoke",
        track_kind="audio",
        reason="supported Lucas create intent",
        evidence_refs=[],
    )
    report = execute_lucas_plan_through_core(
        plan=_plan(session, [action]),
        session=session,
        daw=daw,
        persist_dir=tmp_path,
        rollback_after=True,
    )

    assert report["status"] == "SAFE_WRITE_COMPLETE"
    assert [item["action_type"] for item in report["accepted"]] == ["CREATE_TRACK"]
    assert report["direct_lucas_writes"] == 0
    assert report["write_authority"] == "SafeWriteExecutor"
    assert report["rollback_verified"] is True
    assert daw.snapshot().track_by_name("Core Lucas Smoke") is None


def test_supported_lucas_parameter_intent_uses_core_safe_write(tmp_path: Path):
    daw = MockAbletonAdapter()
    daw.connect()
    daw.create_audio_track("Existing")
    daw.load_instrument_or_effect(0, "Operator")
    session = daw.snapshot()
    attach_tokens(session)

    report = execute_lucas_patch_contracts_through_core(
        contracts=[
            {
                "operation": "set_device_parameter",
                "track": "Existing",
                "device": "Operator",
                "writes": [{"index": 1, "value": 0.6}],
            }
        ],
        session=session,
        daw=daw,
        persist_dir=tmp_path,
    )

    assert report["status"] == "SAFE_WRITE_COMPLETE"
    assert report["accepted"][0]["action_type"] == "SET_DEVICE_PARAMETER"
    assert report["direct_lucas_writes"] == 0
    assert daw.snapshot().tracks[0].devices[1].parameters[1].value == pytest.approx(0.6)


def test_lucas_device_preset_intent_runs_through_core_safe_write(tmp_path: Path):
    daw = MockAbletonAdapter()
    daw.connect()
    daw.create_audio_track("Existing")
    daw.load_instrument_or_effect(0, "Operator")
    session = daw.snapshot()
    attach_tokens(session)

    report = execute_lucas_patch_contracts_through_core(
        contracts=[
            {
                "operation": "load_device_preset",
                "track": "Existing",
                "device": "Operator",
                "preset_uri": "query:UserPresets#Operator#Init",
            }
        ],
        session=session,
        daw=daw,
        persist_dir=tmp_path,
    )

    assert report["status"] == "SAFE_WRITE_COMPLETE"
    assert report["accepted"][0]["action_type"] == "LOAD_DEVICE_PRESET"
    assert report["direct_lucas_writes"] == 0


def test_vibe_production_path_has_no_direct_soniq_writer():
    import inspect

    from copilot import cli

    source = inspect.getsource(cli._vibe)
    assert "apply_patch_contract_auto_mode" not in source
    assert "copilot.producer.soniq_surface" not in source
    assert "execute_lucas_plan_through_core" in source
    assert "execute_lucas_patch_contracts_through_core" in source


def test_core_does_not_resurrect_lucas_omitted_tracks() -> None:
    daw = MockAbletonAdapter()
    daw.connect()
    session = daw.snapshot()
    attach_tokens(session)
    actions = [
        build_create_track_action(
            project_identity=session.project_identity,
            track_name=name,
            reason="fixture",
            evidence_refs=[],
        )
        for name in ("Clap", "Stab", "Guitar", "Sax")
    ]
    plan = _plan(session, actions)
    constrained, metadata = constrain_plan_to_lucas_intent(
        plan,
        {
            "selections": {"Clap": 1, "Stab": 1},
            "arrangement": [{"name": "DROP", "bars": 8, "active": ["Clap", "Stab"]}],
        },
    )
    names = [str(action.target.ref.get("name")) for action in constrained.actions]
    assert names == ["Clap", "Stab"]
    assert metadata["core_intent_gate"]["omitted_tracks"] == ["Guitar", "Sax"]
