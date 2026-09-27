"""Core-side adapter for the stable Lucas producer surface.

Lucas decides musical intent. This module only packages Core evidence, validates
the returned MusicPlan, and exposes explicit certified/unsupported action
results. It never receives a DAW object and never performs a write.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any

from copilot.schemas.advanced_perception import AdvancedPerceptionResult
from copilot.schemas.lucas_integration import (
    ContextFact,
    LucasProducerInput,
    ProjectContext,
    ProjectTrackContext,
    ReferenceContext,
    ReferenceSectionContext,
    SampleSetBoundary,
    StyleContext,
    UserIntent,
)
from copilot.schemas.music_analysis import MusicAnalysisPack
from copilot.schemas.musicplan import (
    DiagnosisBinding,
    MusicPlan,
    PlanAction,
    PlanIntentClass,
    ProductionActionKind,
    SCHEMA_VERSION,
)
from copilot.schemas.session import SessionState
from copilot.daw.state_tokens import attach_tokens, target_token
from copilot.human_eval.store import now_iso
from copilot.sample_library.schemas import LibraryIndex, SampleSetContext

CERTIFIED_ACTIONS = frozenset({
    ProductionActionKind.CREATE_TRACK,
    ProductionActionKind.SAMPLE_LOAD,
    ProductionActionKind.DUPLICATE_CLIP_TO_ARRANGEMENT,
    ProductionActionKind.LOAD_DEVICE,
    ProductionActionKind.DEVICE_LOAD,
    ProductionActionKind.DEVICE_TWEAK,
    ProductionActionKind.SET_TRACK_VOLUME,
})


@dataclass(frozen=True)
class PlannerRun:
    plan: MusicPlan
    planner_metadata: dict[str, Any]
    input_context: LucasProducerInput


@dataclass(frozen=True)
class ActionDisposition:
    action_id: str
    action_type: str
    status: str
    reason: str


@dataclass(frozen=True)
class BoundedPlan:
    plan: MusicPlan
    accepted: tuple[ActionDisposition, ...]
    deferred: tuple[ActionDisposition, ...]


def build_reference_context(
    pack: MusicAnalysisPack,
    perception: AdvancedPerceptionResult | None = None,
) -> ReferenceContext:
    """Convert immutable Core evidence into the bounded producer handoff."""
    sections = [
        ReferenceSectionContext(
            name=section.name,
            start_beat=section.start_beat,
            end_beat=section.end_beat,
            function=section.function,
            confidence=section.confidence,
            evidence_refs=list(section.evidence),
        )
        for section in pack.sections
    ]
    facts: list[ContextFact] = []
    for window in pack.windows:
        facts.extend([
            ContextFact(
                domain="energy",
                name=f"window[{window.index}].energy_db",
                value=window.energy_db,
                evidence_refs=list(window.evidence_refs),
                provenance=dict(window.provenance),
                limitations=list(window.limitations),
            ),
            ContextFact(
                domain="lowend",
                name=f"window[{window.index}].kick_bass_relationship",
                value={
                    "kick_energy": window.kick_energy,
                    "bass_energy": window.bass_energy,
                    "overlap_ratio": window.kick_bass_overlap_ratio,
                    "overlap_duration_s": window.kick_bass_overlap_duration_s,
                },
                evidence_refs=list(window.evidence_refs),
                provenance=dict(window.provenance),
                limitations=list(window.limitations),
            ),
            ContextFact(
                domain="groove",
                name=f"window[{window.index}].groove",
                value={
                    "onset_density_per_s": window.groove.onset_density_per_s,
                    "swing_ratio": window.groove.swing_ratio,
                    "offbeat_ratio": window.groove.offbeat_ratio,
                    "repetition_strength": window.groove.repetition_strength,
                },
                evidence_refs=list(window.evidence_refs),
                provenance=dict(window.provenance),
                limitations=list(window.limitations),
            ),
        ])
    if perception is not None:
        for provider in perception.providers:
            facts.append(ContextFact(
                domain="advanced_perception",
                name=f"provider.{provider.name}",
                value={
                    "available": provider.available,
                    "semantic": provider.semantic,
                    "version": provider.version,
                    "metadata": provider.metadata,
                },
                evidence_refs=["advanced_perception.provider_availability"],
                limitations=[provider.reason] if provider.reason else [],
            ))
    refs = list(dict.fromkeys([
        *pack.evidence_refs,
        *(ref for section in pack.sections for ref in section.evidence),
    ]))
    return ReferenceContext(
        reference_state_token=pack.tokens.reference_state_token,
        identity=str(pack.provenance.get("audio_sha256") or pack.tokens.reference_state_token),
        tempo_bpm=pack.tempo_bpm,
        sections=sections,
        facts=facts,
        evidence_refs=refs,
        provenance={
            "pack_schema": pack.schema_version,
            "audio_sha256": pack.provenance.get("audio_sha256"),
            "analyzer_ids": dict(pack.analyzer_ids),
        },
        limitations=list(dict.fromkeys(pack.limitations)),
    )


def build_sample_set_boundary(
    context: SampleSetContext,
    *,
    provenance: dict[str, Any] | None = None,
) -> SampleSetBoundary:
    ids = sorted({
        str(candidate["id"])
        for candidate in context.candidates
        if candidate.get("id")
    })
    return SampleSetBoundary(
        context=context,
        stable_sample_ids=ids,
        provenance=dict(provenance or {}),
    )


def build_project_context(session: SessionState) -> ProjectContext:
    return ProjectContext(
        project_identity=session.project_identity,
        project_token=session.project_token or session.project_identity,
        audible_token=session.audible_token,
        session_incarnation_id=session.session_incarnation_id,
        project_name=session.project_name,
        tracks=[
            ProjectTrackContext(
                stable_id=track.stable_id,
                name=track.name,
                role=track.role,
                index_locator=track.index,
                device_count=len(track.devices),
                clip_count=len(track.clips),
                volume=track.mixer.volume,
            )
            for track in session.tracks
        ],
        facts=[
            ContextFact(
                domain="project",
                name="transport.tempo",
                value=session.transport.tempo,
                provenance={"source": "SessionState"},
            ),
        ],
    )


def build_lucas_input(
    *,
    user_intent: UserIntent,
    reference: ReferenceContext,
    samples: SampleSetContext,
    project: ProjectContext,
    style: StyleContext | None = None,
) -> LucasProducerInput:
    return LucasProducerInput(
        user_intent=user_intent,
        reference=reference,
        samples=build_sample_set_boundary(samples),
        style=style or StyleContext(),
        project=project,
    )


def _grounded_intent(input_context: LucasProducerInput) -> str:
    sections = ", ".join(
        f"{section.name}:{section.start_beat:g}-{section.end_beat:g}"
        for section in input_context.reference.sections
    ) or "UNKNOWN"
    return (
        f"{input_context.user_intent.description}. "
        f"Reference tempo={input_context.reference.tempo_bpm:g} BPM; "
        f"observed sections={sections}; "
        f"reference_token={input_context.reference.reference_state_token}; "
        f"project_identity={input_context.project.project_identity}; "
        f"sample_candidates={len(input_context.samples.stable_sample_ids)}. "
        "Use only these grounded facts; creative choices remain producer intent."
    )


def run_lucas_planner(
    *,
    input_context: LucasProducerInput,
    index: LibraryIndex,
    session: SessionState,
    provider: Any = None,
    planner: Callable[..., tuple[MusicPlan, dict[str, Any]]] | None = None,
    plan_id: str = "lucas_core_integration_v1",
) -> PlannerRun:
    """Invoke the stable Lucas planner without exposing Core write authority."""
    if session.project_identity != input_context.project.project_identity:
        raise ValueError("PROJECT_MISMATCH: planner input is not for this project")
    planner_fn = planner
    if planner_fn is None:
        from copilot.musicplan.astra_plan import build_plan_from_prompt

        planner_fn = build_plan_from_prompt
    plan, metadata = planner_fn(
        index=index,
        session=session,
        intent=_grounded_intent(input_context),
        provider=provider,
        plan_id=plan_id,
    )
    validated = MusicPlan.model_validate(plan.model_dump(mode="json"))
    if validated.schema_version != SCHEMA_VERSION:
        raise ValueError(f"unsupported MusicPlan schema: {validated.schema_version}")
    expected_project_token = input_context.project.project_token
    if validated.project_state_token != expected_project_token:
        raise ValueError("STALE_PLAN: MusicPlan project token does not match ProjectContext")
    refs = list(dict.fromkeys([
        *validated.evidence_refs,
        *input_context.reference.evidence_refs,
    ]))
    validated = validated.model_copy(update={
        "evidence_refs": refs,
        "notes": [
            *validated.notes,
            f"Core reference handoff: {input_context.reference.reference_state_token}",
            "Core remains measurement and write authority; Lucas plan is intent only.",
        ],
    })
    constrained, constrained_metadata = constrain_plan_to_lucas_intent(validated, dict(metadata))
    return PlannerRun(plan=constrained, planner_metadata=constrained_metadata, input_context=input_context)


def constrain_plan_to_lucas_intent(
    plan: MusicPlan,
    metadata: dict[str, Any],
) -> tuple[MusicPlan, dict[str, Any]]:
    """Prevent the deterministic recipe from resurrecting omitted Lucas roles.

    ``build_plan_from_prompt`` historically starts from a broad recipe and
    overlays model selections.  That is safe for sample defaults, but not for
    executable track/device actions: an omitted Guitar/Sax must not reappear
    merely because the recipe contains those roles.  This Core-side gate uses
    only explicit Lucas metadata and never chooses a musical role itself.
    """
    selections = metadata.get("selections")
    arrangement = metadata.get("arrangement")
    track_spec = metadata.get("track_spec")
    if (
        not isinstance(selections, dict)
        and not isinstance(arrangement, list)
        and not isinstance(track_spec, dict)
    ):
        return plan, metadata
    allowed: set[str] = {str(name) for name in (selections or {})}
    for section in arrangement or []:
        if isinstance(section, dict):
            allowed.update(str(name) for name in (section.get("active") or []))
    track_spec_sections = track_spec.get("sections", []) if isinstance(track_spec, dict) else []
    for section in track_spec_sections:
        if isinstance(section, dict):
            allowed.update(str(name) for name in (section.get("active_roles") or []))
    if not allowed:
        return plan, metadata

    kept: list[PlanAction] = []
    omitted: list[str] = []
    for action in plan.actions:
        target_name = str((action.target.ref or {}).get("name") or "").strip()
        if target_name and target_name not in allowed:
            omitted.append(target_name)
            continue
        kept.append(action)
    omitted_names = sorted(set(omitted))
    if not omitted_names:
        return plan, metadata
    constrained = plan.model_copy(update={
        "actions": kept,
        "notes": [
            *plan.notes,
            "Core intent gate removed recipe actions for roles omitted by Lucas.",
        ],
    })
    return constrained, {
        **metadata,
        "core_intent_gate": {
            "allowed_tracks": sorted(allowed),
            "omitted_tracks": omitted_names,
            "removed_action_count": len(omitted),
        },
    }


def run_lucas_critique(
    *,
    plan: MusicPlan,
    session: SessionState,
    provider: Any = None,
    timeout_s: float = 120.0,
):
    """Invoke Lucas's advisory critique surface through a Core read-only boundary.

    The critique may use Astra internally, but it never receives a DAW adapter and
    never authorizes a write. Keeping this call in the Core adapter makes the
    planner and critique call chains explicit instead of calling the Lucas module
    ad hoc from a validation script.
    """
    from copilot.musicplan.critique import critique_track

    return critique_track(
        plan=plan,
        session=session,
        provider=provider,
        timeout_s=timeout_s,
    )


class _GroundedCritiqueProvider:
    """Provider adapter that adds immutable evidence without becoming a critic."""

    def __init__(self, inner: Any, evidence_context: dict[str, Any] | None) -> None:
        self.inner = inner
        self.evidence_context = dict(evidence_context or {})
        self.failure: dict[str, str] | None = None
        self.identity = str(getattr(inner, "identity", type(inner).__name__))
        self.version = str(getattr(inner, "version", "unknown"))

    def reason(self, prompt: str, *, timeout_s: float = 30.0) -> str:
        grounded = prompt
        if self.evidence_context:
            grounded += "\n\n=== IMMUTABLE PRE/POST EVIDENCE ===\n"
            grounded += json.dumps(self.evidence_context, sort_keys=True, default=str)
            grounded += "\nUse only this evidence; do not invent measurements."
        try:
            return self.inner.reason(grounded, timeout_s=timeout_s)
        except Exception as exc:  # provider failure is reported, never converted to a verdict
            self.failure = {
                "type": type(exc).__name__,
                "message": str(exc),
            }
            raise


def run_lucas_critique_with_provider_failover(
    *,
    plan: MusicPlan,
    session: SessionState,
    providers: Iterable[Any] | None = None,
    evidence_context: dict[str, Any] | None = None,
    timeout_s: float = 30.0,
) -> dict[str, Any]:
    """Run the unchanged Lucas critique contract over bounded providers.

    This is Core orchestration only. Lucas still owns the prompt, parser, and
    typed critique result; a provider can only return a valid result or fail.
    """
    if providers is None:
        from copilot.reasoning.provider import configured_http_provider

        configured = configured_http_provider()
        provider_list = [configured] if configured is not None else []
    else:
        provider_list = [provider for provider in providers if provider is not None]
    attempts: list[dict[str, Any]] = []
    if not provider_list:
        return {
            "status": "CRITIQUE_PROVIDER_UNAVAILABLE",
            "result": None,
            "attempts": attempts,
        }
    for provider in provider_list:
        adapter = _GroundedCritiqueProvider(provider, evidence_context)
        result = run_lucas_critique(
            plan=plan,
            session=session,
            provider=adapter,
            timeout_s=timeout_s,
        )
        if result is not None:
            return {
                "status": "CRITIQUE_COMPLETE",
                "result": result,
                "provider": adapter.identity,
                "provider_version": adapter.version,
                "attempts": [
                    *attempts,
                    {"provider": adapter.identity, "status": "SUCCESS"},
                ],
            }
        attempts.append({
            "provider": adapter.identity,
            "status": "FAILED",
            "failure": adapter.failure or {
                "type": "INVALID_SCHEMA_OR_CRITIQUE_FAILURE",
                "message": "Lucas critique returned no typed result",
            },
        })
    return {
        "status": "CRITIQUE_PROVIDER_UNAVAILABLE",
        "result": None,
        "attempts": attempts,
    }


def _single_action_plan(action: PlanAction, *, session: SessionState, plan_id: str) -> MusicPlan:
    """Wrap one Lucas intent in the canonical Core MusicPlan envelope."""
    target_tokens: dict[str, str] = {}
    name = str(action.target.ref.get("name", ""))
    track = session.track_by_name(name) if name else None
    if track is not None:
        target_tokens[track.name] = target_token(track)
    return MusicPlan(
        plan_id=plan_id,
        status="DRAFT",
        intent_class=PlanIntentClass.AUTONOMOUS_MUSICAL_IMPROVEMENT,
        diagnosis=DiagnosisBinding(
            diagnosis_id="lucas-core-execution",
            diagnosis_status="SUPPORTED",
            diagnosis_accepted=True,
        ),
        project_state_token=session.project_token or session.project_identity or "",
        audible_state_token=session.audible_token or "",
        target_state_tokens=target_tokens,
        created_at=now_iso(),
        actions=[action],
    )


def execute_core_action_through_safe_write(
    *,
    action: PlanAction,
    session: SessionState,
    daw,
    persist_dir: Path,
    rollback_after: bool = False,
) -> dict[str, Any]:
    """Compile and execute one already-grounded Lucas action through Core.

    The caller is responsible for resolving the action target against the
    authoritative session. This helper is deliberately single-action so every
    mix/master write has its own durable pre-state, readback, and rollback.
    """
    from copilot.runtime.production_compiler import ProductionCompiler
    from copilot.runtime.safe_write import build_safe_write_executor

    attach_tokens(session)
    single = _single_action_plan(
        action,
        session=session,
        plan_id=re.sub(r"[^A-Za-z0-9_.-]+", "_", f"core_{action.action_id}"),
    )
    compiled = ProductionCompiler().compile(single, session=session)
    if compiled.status != "COMPILED" or compiled.intent is None:
        return {
            "status": "EXECUTION_DEFERRED",
            "reason": "; ".join(compiled.reasons) or compiled.status,
            "action_type": action.action_type.value,
            "direct_lucas_writes": 0,
            "write_authority": "SafeWriteExecutor",
        }
    executor = build_safe_write_executor(
        daw,
        journal_path=persist_dir / f"{action.action_id}_safe_write.jsonl",
        persist_dir=persist_dir,
    )
    result = executor.run(compiled.intent)
    if not result.ok:
        return {
            "status": "SAFE_WRITE_FAILED",
            "reason": result.error or "SAFE_WRITE_FAILED",
            "action_type": action.action_type.value,
            "direct_lucas_writes": 0,
            "write_authority": "SafeWriteExecutor",
        }
    rollback_error = ""
    if rollback_after:
        rollback_error = executor._rollback_applied(result, compiled.intent)
    return {
        "status": "SAFE_WRITE_COMPLETE" if not rollback_error else "SAFE_WRITE_ROLLBACK_FAILED",
        "action_type": action.action_type.value,
        "readbacks": [row.model_dump(mode="json") for row in result.readbacks],
        "rollback_verified": rollback_after and not rollback_error,
        "rollback_error": rollback_error,
        "direct_lucas_writes": 0,
        "write_authority": "SafeWriteExecutor",
    }


def execute_lucas_plan_through_core(
    *,
    plan: MusicPlan,
    session: SessionState,
    daw,
    persist_dir: Path,
    rollback_after: bool = False,
) -> dict[str, Any]:
    """Execute only a bounded Lucas subset through Compiler -> SafeWrite.

    This is the production entrypoint used by the Core-owned CLI path. The
    planner may return a large plan, but this function deliberately executes
    only one create/sample pair and reports every other action as deferred.
    It never calls Lucas's DAW-facing surfaces.
    """
    from copilot.runtime.production_compiler import ProductionCompiler
    from copilot.runtime.safe_write import build_safe_write_executor

    attach_tokens(session)
    create = next(
        (a for a in plan.actions if a.action_type is ProductionActionKind.CREATE_TRACK),
        None,
    )
    sample = None
    if create is not None:
        track_name = str(create.params.track_name)
        sample = next(
            (
                a for a in plan.actions
                if a.action_type is ProductionActionKind.SAMPLE_LOAD
                and str(a.target.ref.get("name", "")) == track_name
            ),
            None,
        )

    selected_ids = [a.action_id for a in (create, sample) if a is not None]
    bounded = bound_plan_to_certified_actions(plan, action_ids=selected_ids)
    deferred = [
        {
            "action_id": item.action_id,
            "action_type": item.action_type,
            "status": "EXECUTION_DEFERRED",
            "reason": item.reason,
        }
        for item in bounded.deferred
    ]
    executor = build_safe_write_executor(
        daw,
        journal_path=persist_dir / "lucas_core_safe_write.jsonl",
        persist_dir=persist_dir,
    )
    compiler = ProductionCompiler()
    accepted: list[dict[str, Any]] = []
    applied: list[tuple[Any, Any]] = []
    current = session

    for action in (create, sample):
        if action is None:
            continue
        current = daw.snapshot()
        attach_tokens(current)
        executable = action
        if action.action_type is ProductionActionKind.SAMPLE_LOAD:
            if create is None:
                deferred.append({
                    "action_id": action.action_id,
                    "action_type": action.action_type.value,
                    "status": "EXECUTION_DEFERRED",
                    "reason": "sample target has no Core-created authoritative track",
                })
                continue
            target = current.track_by_name(str(create.params.track_name))
            if target is None:
                deferred.append({
                    "action_id": action.action_id,
                    "action_type": action.action_type.value,
                    "status": "EXECUTION_DEFERRED",
                    "reason": "created track was not found in authoritative readback",
                })
                continue
            executable = rebind_sample_load_action(
                action.model_copy(update={
                    "params": action.params.model_copy(update={
                        "sample_uri": normalize_sample_uri_for_working_copy(action.params.sample_uri),
                    })
                }),
                track=target,
                session=current,
            )
        single = _single_action_plan(
            executable,
            session=current,
            plan_id=re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{plan.plan_id}_{action.action_id}"),
        )
        compiled = compiler.compile(single, session=current)
        if compiled.status != "COMPILED" or compiled.intent is None:
            deferred.append({
                "action_id": action.action_id,
                "action_type": action.action_type.value,
                "status": "EXECUTION_DEFERRED",
                "reason": "; ".join(compiled.reasons) or compiled.status,
            })
            continue
        result = executor.run(compiled.intent)
        if not result.ok:
            return {
                "status": "SAFE_WRITE_FAILED",
                "accepted": accepted,
                "deferred": deferred,
                "error": result.error,
                "direct_lucas_writes": 0,
                "write_authority": "SafeWriteExecutor",
            }
        accepted.append({
            "action_id": action.action_id,
            "action_type": action.action_type.value,
            "status": "VERIFIED",
            "readbacks": [row.model_dump(mode="json") for row in result.readbacks],
        })

        if rollback_after:
            applied.append((result, compiled.intent))

    rollback_error = ""
    if rollback_after:
        for result, intent in reversed(applied):
            rollback_error = executor._rollback_applied(result, intent)
            if rollback_error:
                break

    return {
        "status": "SAFE_WRITE_COMPLETE" if not rollback_error else "SAFE_WRITE_ROLLBACK_FAILED",
        "accepted": accepted,
        "deferred": deferred,
        "after_track_count": len(daw.snapshot().tracks),
        "direct_lucas_writes": 0,
        "write_authority": "SafeWriteExecutor",
        "rollback_verified": rollback_after and not rollback_error,
        "rollback_error": rollback_error,
    }


def execute_lucas_patch_contracts_through_core(
    *,
    contracts: list[dict[str, Any]],
    session: SessionState,
    daw,
    persist_dir: Path,
) -> dict[str, Any]:
    """Map conservative parameter intents to DEVICE_TWEAK/SafeWrite.

    Presets, routing, WebSocket operations, and unknown units are explicitly
    deferred. This function is Core-owned and does not call ``soniq_surface``.
    """
    from copilot.musicplan import build_device_tweak_action
    from copilot.runtime.production_compiler import ProductionCompiler
    from copilot.runtime.safe_write import build_safe_write_executor

    attach_tokens(session)
    compiler = ProductionCompiler()
    executor = build_safe_write_executor(
        daw,
        journal_path=persist_dir / "lucas_core_patch_safe_write.jsonl",
        persist_dir=persist_dir,
    )
    accepted: list[dict[str, Any]] = []
    deferred: list[dict[str, Any]] = []
    current = session
    for cidx, contract in enumerate(contracts or []):
        operation = str(contract.get("operation", "parameter_patch"))
        if operation not in {"parameter_patch", "set_device_parameter", "load_device_preset", ""}:
            deferred.append({"index": cidx, "status": "EXECUTION_DEFERRED", "reason": "UNSUPPORTED_OPERATION"})
            continue
        track_name = str(contract.get("track", ""))
        device_name = str(contract.get("device", ""))
        current = daw.snapshot()
        attach_tokens(current)
        track = current.track_by_name(track_name)
        device = next((d for d in track.devices if d.name.lower() == device_name.lower()), None) if track else None
        if track is None or device is None:
            deferred.append({"index": cidx, "status": "EXECUTION_DEFERRED", "reason": "TARGET_NOT_FOUND"})
            continue
        if operation == "load_device_preset":
            from copilot.daw.identities import fingerprint_track
            from copilot.daw.object_ref import ref_from_track
            from copilot.schemas.safe_write import (
                KIND_PRODUCER_EXECUTION_V1,
                MutationExecution,
                MutationIntent,
                MutationRollback,
                MutationTarget,
                RollbackReversibility,
            )
            from copilot.schemas.transaction import TargetFingerprint, TargetLocator

            preset_uri = str(contract.get("preset_uri", "")).strip()
            if not preset_uri:
                deferred.append({"index": cidx, "status": "EXECUTION_DEFERRED", "reason": "PRESET_URI_REQUIRED"})
                continue
            action_id = f"lucas_preset_{cidx}"
            intent = MutationIntent(
                plan_id=f"lucas_patch_preset_{cidx}",
                kind=KIND_PRODUCER_EXECUTION_V1,
                user_intent="Lucas preset intent via Core SafeWrite",
                project_identity=current.project_identity or "",
                expected_revision=current.revision,
                expected_session_hash=current.state_hash,
                expected_project_token=current.project_token or "",
                expected_audible_token=current.audible_token or "",
                expected_incarnation_id=current.session_incarnation_id or "",
                targets=[
                    MutationTarget(
                        action_id=action_id,
                        ref=ref_from_track(track, project_identity=current.project_identity or "").model_dump(mode="json"),
                        stable_id=track.stable_id,
                        name_at_plan=track.name,
                        fingerprint=TargetFingerprint(**fingerprint_track(track)),
                        locator=TargetLocator(track_index=track.index, device_index=device.index),
                        session_incarnation_id=current.session_incarnation_id or "",
                    )
                ],
                executions=[
                    MutationExecution(
                        action_id=action_id,
                        action_type="LOAD_DEVICE_PRESET",
                        operation="load_device_preset",
                        arguments={"device_index": int(device.index), "preset_uri": preset_uri},
                        expected_before={"device_count": len(track.devices)},
                        expected_after={},
                        certified=True,
                        rollback=MutationRollback(
                            inverse_operation="restore_device_parameters",
                            inverse_params={"items": []},
                            reversibility=RollbackReversibility.INDEPENDENT,
                            prepared=True,
                        ),
                    )
                ],
            )
            result = executor.run(intent)
            if not result.ok:
                deferred.append({"index": cidx, "status": "EXECUTION_DEFERRED", "reason": result.error or "SAFE_WRITE_FAILED"})
                continue
            accepted.append({
                "index": cidx,
                "status": "VERIFIED",
                "action_type": "LOAD_DEVICE_PRESET",
                "readbacks": [row.model_dump(mode="json") for row in result.readbacks],
                "preset_uri": preset_uri,
            })
            continue
        constraints = dict(contract.get("constraints") or {})
        max_delta = float(constraints.get("max_delta_norm", 0.35))
        max_writes = max(1, min(16, int(constraints.get("max_writes", 8))))
        forbid_toggle = bool(constraints.get("forbid_device_on_toggle", True))
        for widx, write in enumerate(list(contract.get("writes") or [])[:max_writes]):
            try:
                pidx = int(write["index"]) if "index" in write else None
                param = next((p for p in device.parameters if p.index == pidx), None) if pidx is not None else next(
                    (p for p in device.parameters if p.name.lower() == str(write.get("name", "")).lower()), None
                )
                if param is None:
                    raise ValueError("PARAMETER_NOT_FOUND")
                normalized = float(write["value"])
                if not 0.0 <= normalized <= 1.0:
                    raise ValueError("VALUE_OUT_OF_RANGE")
                if forbid_toggle and (param.index == 0 or param.name.lower() == "device on"):
                    raise ValueError("DEVICE_TOGGLE_FORBIDDEN")
                span = float(param.max) - float(param.min)
                target = float(param.min) + normalized * span if span > 0 else normalized
                delta_norm = abs(target - float(param.value)) / span if span > 0 else abs(target - float(param.value))
                if delta_norm > max_delta:
                    raise ValueError("MAX_DELTA_EXCEEDED")
            except (KeyError, TypeError, ValueError) as exc:
                deferred.append({
                    "index": cidx,
                    "write": widx,
                    "status": "EXECUTION_DEFERRED",
                    "reason": str(exc),
                })
                continue
            action = build_device_tweak_action(
                track=track,
                project_identity=current.project_identity,
                device_index=device.index,
                parameter_name=param.name,
                expected_before=float(param.value),
                intended_after=target,
                unit="native",
                reason="Lucas parameter intent via Core SafeWrite",
                evidence_refs=[],
                session_incarnation_id=current.session_incarnation_id,
                allowed_min=float(param.min),
                allowed_max=float(param.max),
            )
            single = _single_action_plan(
                action,
                session=current,
                plan_id=f"lucas_patch_{cidx}_{widx}",
            )
            compiled = compiler.compile(single, session=current)
            if compiled.status != "COMPILED" or compiled.intent is None:
                deferred.append({"index": cidx, "write": widx, "status": "EXECUTION_DEFERRED", "reason": "; ".join(compiled.reasons)})
                continue
            result = executor.run(compiled.intent)
            if not result.ok:
                deferred.append({"index": cidx, "write": widx, "status": "EXECUTION_DEFERRED", "reason": result.error or "SAFE_WRITE_FAILED"})
                continue
            accepted.append({"index": cidx, "write": widx, "status": "VERIFIED", "action_type": "SET_DEVICE_PARAMETER"})
    return {
        "status": "SAFE_WRITE_COMPLETE" if accepted or not deferred else "EXECUTION_DEFERRED",
        "accepted": accepted,
        "deferred": deferred,
        "direct_lucas_writes": 0,
        "write_authority": "SafeWriteExecutor",
    }


def bound_plan_to_certified_actions(
    plan: MusicPlan,
    *,
    action_ids: Iterable[str],
) -> BoundedPlan:
    """Select an explicit bounded subset; every other action stays traceable."""
    requested = set(action_ids)
    known = {action.action_id for action in plan.actions}
    unknown = requested - known
    if unknown:
        raise ValueError(f"unknown action IDs in bounded selection: {sorted(unknown)}")
    accepted: list[ActionDisposition] = []
    deferred: list[ActionDisposition] = []
    selected: list[PlanAction] = []
    for action in plan.actions:
        if action.action_id in requested and action.action_type in CERTIFIED_ACTIONS:
            selected.append(action)
            accepted.append(ActionDisposition(
                action_id=action.action_id,
                action_type=action.action_type.value,
                status="ACCEPTED",
                reason="explicitly selected certified action",
            ))
        else:
            reason = (
                "not in bounded execution selection"
                if action.action_id not in requested
                else "action is outside certified producer vocabulary"
            )
            deferred.append(ActionDisposition(
                action_id=action.action_id,
                action_type=action.action_type.value,
                status="DEFERRED_UNSUPPORTED" if action.action_type not in CERTIFIED_ACTIONS else "DEFERRED",
                reason=reason,
            ))
    # Plan IDs are also used by SafeWrite for durable pre-state filenames.
    # Keep the derived ID portable across hosts; ':' is invalid in Windows
    # filenames and would fail before the first DAW mutation.
    bounded = plan.model_copy(update={
        "plan_id": f"{plan.plan_id}_bounded",
        "actions": selected,
    })
    return BoundedPlan(plan=bounded, accepted=tuple(accepted), deferred=tuple(deferred))


def rebind_sample_load_action(
    action: PlanAction,
    *,
    track,
    session: SessionState,
) -> PlanAction:
    """Bind Lucas's stable sample intent to Core's authoritative post-create track."""
    if action.action_type is not ProductionActionKind.SAMPLE_LOAD:
        raise ValueError("only SAMPLE_LOAD actions can be rebound")
    from copilot.musicplan import build_sample_load_action

    params = action.params
    rebound = build_sample_load_action(
        track=track,
        project_identity=session.project_identity,
        clip_index=int(params.clip_index),
        sample_uri=str(params.sample_uri),
        reason=action.reason,
        evidence_refs=list(action.evidence_refs),
        session_incarnation_id=session.session_incarnation_id,
    )
    return rebound.model_copy(update={"action_id": action.action_id})


def rebind_sample_load_plan(
    plan: MusicPlan,
    action: PlanAction,
    *,
    track,
    session: SessionState,
) -> MusicPlan:
    """Rebind a Lucas sample intent after Core authoritatively creates a track.

    A virtual Lucas track has no valid target-state token.  The token must be
    minted from the post-create authoritative readback before the sample load
    can pass the normal MusicPlan gate.
    """
    rebound = rebind_sample_load_action(action, track=track, session=session)
    return plan.model_copy(update={
        "actions": [rebound],
        "project_state_token": session.project_token or session.project_identity or "",
        "audible_state_token": session.audible_token or "",
        "target_state_tokens": {track.name: target_token(track)},
    })


def normalize_sample_uri_for_working_copy(sample_uri: str) -> str:
    """Translate Lucas library-relative paths to the typed Live browser URI."""
    value = str(sample_uri).replace("\\", "/")
    if value.startswith("query:CurrentProject#Samples:"):
        prefix, relative = value.split("#Samples:", 1)
        return f"{prefix}#Samples:{relative.strip('/').replace('/', ':')}"
    if value.startswith("query:") or value.startswith("browser://"):
        return value
    if value.startswith("Samples/"):
        value = value[len("Samples/"):]
    return f"query:CurrentProject#Samples:{value.lstrip('/').replace('/', ':')}"
