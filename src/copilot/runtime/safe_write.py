"""SAFE_WRITE_FOUNDATION_V2 — generic verified musical mutation architecture.

Wraps TransactionManager, DurableJournal, WriteInDoubt, and the frozen
SET_TRACK_VOLUME foundation. Producer Execution V1 is an additive intent kind
that uses this same executor; it does not create a second transaction system.
AnalyzeProject stays read-only.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from urllib.parse import unquote
from uuid import uuid4

from copilot.agent.journal import DurableJournal
from copilot.agent.recovery import classify_journal
from copilot.agent.tools import AgentTools
from copilot.agent.transactions import TransactionManager
from copilot.audio.m4l_control_contract_v1 import APPLY_RESULTS, APPLY_VOCABULARY
from copilot.daw.identities import fingerprint_track, fingerprints_equal
from copilot.daw.mutation_protocol import KIND_TEMPORARY_CAPTURE_HOST
from copilot.daw.object_ref import (
    PersistentObjectRef,
    ResolveStatus,
    ref_from_track,
    resolve_track,
)
from copilot.daw.state_tokens import attach_tokens
from copilot.daw.validation import validate_mixer_volume
from copilot.daw.write import ReconcileResult, WriteInDoubt
from copilot.musicplan.execute import diff_guard_state, snapshot_guard_state
from copilot.schemas.safe_write import (
    CERTIFIED_PRODUCTION_ACTION,
    CERTIFIED_PRODUCTION_ACTIONS,
    KIND_PRODUCER_EXECUTION_V1,
    KIND_EXPERIMENTAL_MIDI_PHRASE_V1,
    PRODUCER_CERTIFIED_ACTIONS,
    KIND_PRODUCTION_MUSICAL,
    MILESTONE,
    SCHEMA_VERSION,
    ApplyDecision,
    MutationExecution,
    MutationFailure,
    MutationIntent,
    MutationPhase,
    MutationPrecondition,
    MutationReadback,
    MutationResult,
    MutationRollback,
    MutationTarget,
    MutationVerification,
    RollbackReversibility,
)
from copilot.schemas.session import SessionState, TrackState
from copilot.producer.midi_phrase_policy import (
    MIDI_PHRASE_CAPABILITIES, empty_phrase_slot_blocker, phrase_from_arguments,
    phrase_preservation_token,
)
from copilot.schemas.transaction import (
    TargetFingerprint,
    TargetLocator,
    TransactionStatus,
)

# Capture-host compound batches are a different contract.
assert KIND_PRODUCTION_MUSICAL != KIND_TEMPORARY_CAPTURE_HOST
assert APPLY_VOCABULARY == (CERTIFIED_PRODUCTION_ACTION,)
assert set(APPLY_RESULTS) == {item.value for item in ApplyDecision}

VOLUME_READBACK_TOLERANCE = 1e-4


def _sample_uri_label(sample_uri: str) -> str:
    """Return the browser item's filename/stem from a URI-like reference."""
    value = unquote(str(sample_uri)).replace("\\", "/")
    value = value.split("#", 1)[-1]
    value = value.rsplit(":", 1)[-1].rsplit("/", 1)[-1]
    return Path(value).stem.casefold()


def _sample_device_matches(device, sample_uri: str) -> bool:
    """Match sparse Live readbacks without weakening identity checks.

    Some Live/Remote Script versions omit both ``device_index`` from the
    write response and ``sample_uri`` from device inventory. In that case
    the browser-loaded Simpler name is the available identifier. The caller
    still restricts this match to newly-created device stable IDs and
    requires uniqueness.
    """
    observed_uri = str(getattr(device, "sample_uri", None) or "")
    if observed_uri:
        return observed_uri == str(sample_uri)
    expected = _sample_uri_label(sample_uri)
    observed = Path(str(getattr(device, "name", ""))).stem.casefold()
    return bool(expected and observed and expected == observed)


def _sample_reference_matches(observed_uri: str | None, expected_uri: str) -> bool:
    observed = str(observed_uri or "").replace("\\", "/").casefold().strip()
    expected = str(expected_uri or "").replace("\\", "/").casefold().strip()
    if not observed or not expected:
        return False
    if observed == expected or observed.endswith("/" + expected):
        return True
    return Path(observed).name == Path(expected).name


def _track_fingerprint_key(track: TrackState) -> str:
    """Canonical structural identity used when a session incarnation changes."""
    return json.dumps(fingerprint_track(track), sort_keys=True, separators=(",", ":"))


def _is_live_sample_routing_normalization(
    before: dict[str, Any], after: dict[str, Any], target_name: str, row: dict[str, Any]
) -> bool:
    """Allow only Live's new-MIDI-track routing normalization for LOAD_SAMPLE."""
    before_routing = ((before.get("tracks") or {}).get(target_name) or {}).get("routing") or {}
    after_routing = ((after.get("tracks") or {}).get(target_name) or {}).get("routing") or {}
    before_target = (before.get("tracks") or {}).get(target_name) or {}
    after_target = (after.get("tracks") or {}).get(target_name) or {}
    if (
        row.get("field") == f"{target_name}.arm"
        and before_target.get("role") == "midi"
        and before_target.get("arm") is False
        and after_target.get("arm") is True
        and before_routing.get("output_type") == "No Output"
    ):
        return True
    if row.get("field") != f"{target_name}.routing":
        return False
    return (
        before_routing.get("output_type") == "No Output"
        and after_routing.get("output_type") == "Main"
        and before_routing.get("input_type") == after_routing.get("input_type")
        and before_routing.get("monitoring") == after_routing.get("monitoring")
    )


class WriteCancellation:
    """Cooperative cancel. After dispatch, unknown outcome is IN_DOUBT."""

    def __init__(self) -> None:
        self._requested = False
        self.reason = "CANCELLED"

    @property
    def requested(self) -> bool:
        return self._requested

    def cancel(self, reason: str = "CANCELLED") -> None:
        self._requested = True
        self.reason = reason


class SafeWriteRegistry:
    """In-process prepared-plan registry. In-flight writes cannot be superseded."""

    def __init__(self) -> None:
        self._prepared: dict[str, str] = {}

    def key_for(self, intent: MutationIntent) -> str:
        names = tuple(sorted(target.name_at_plan for target in intent.targets))
        return f"{intent.project_identity}|{intent.expected_incarnation_id}|{names}"

    def peek(self, intent: MutationIntent) -> str | None:
        return self._prepared.get(self.key_for(intent))

    def register_prepared(self, intent: MutationIntent) -> str | None:
        key = self.key_for(intent)
        previous = self._prepared.get(key)
        self._prepared[key] = intent.plan_id
        if previous and previous != intent.plan_id:
            return previous
        return None

    def release(self, intent: MutationIntent) -> None:
        key = self.key_for(intent)
        if self._prepared.get(key) == intent.plan_id:
            self._prepared.pop(key, None)

    def release_plan(self, plan_id: str) -> None:
        for key, value in list(self._prepared.items()):
            if value == plan_id:
                self._prepared.pop(key, None)


def certified_production_actions() -> frozenset[str]:
    return CERTIFIED_PRODUCTION_ACTIONS


def action_is_certified(action_type: str) -> bool:
    return action_type == CERTIFIED_PRODUCTION_ACTION


def action_is_certified_for_intent(intent: MutationIntent, action_type: str) -> bool:
    if intent.kind == KIND_PRODUCER_EXECUTION_V1:
        return action_type in PRODUCER_CERTIFIED_ACTIONS
    return action_is_certified(action_type)


def plan_compound_rollback(executions: list[MutationExecution]) -> dict[str, Any]:
    """Dependency-aware undo order for future compound production plans.

    Reverse apply order is the default. Explicit depends_on means the dependent
    step is undone first. NOT_INDEPENDENTLY_REVERSIBLE steps cannot be undone
    in isolation.
    """
    ids = [item.action_id for item in executions]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate action_id in mutation plan")
    index = {item.action_id: item for item in executions}
    incoming: dict[str, set[str]] = {action_id: set() for action_id in ids}
    outgoing: dict[str, set[str]] = {action_id: set() for action_id in ids}
    for position, action_id in enumerate(ids):
        step = index[action_id]
        for dependency in step.rollback.depends_on:
            if dependency not in incoming:
                raise ValueError(f"rollback depends_on unknown action {dependency}")
            # Undo dependent before dependency.
            incoming[dependency].add(action_id)
            outgoing[action_id].add(dependency)
        if position > 0:
            earlier = ids[position - 1]
            if earlier not in outgoing[action_id] and action_id not in outgoing[earlier]:
                incoming[earlier].add(action_id)
                outgoing[action_id].add(earlier)

    ready = [action_id for action_id in ids if not incoming[action_id]]
    # Prefer later-applied actions first (reverse apply) among ready nodes.
    apply_rank = {action_id: rank for rank, action_id in enumerate(ids)}
    order: list[str] = []
    while ready:
        ready.sort(key=lambda action_id: -apply_rank[action_id])
        current = ready.pop(0)
        order.append(current)
        for nxt in list(outgoing[current]):
            incoming[nxt].discard(current)
            outgoing[current].discard(nxt)
            if not incoming[nxt]:
                ready.append(nxt)

    if len(order) != len(ids):
        raise ValueError("cyclic rollback dependency")

    isolated_forbidden = [
        index[action_id].action_id
        for action_id in order
        if index[action_id].rollback.reversibility
        is RollbackReversibility.NOT_INDEPENDENTLY_REVERSIBLE
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "undo_action_ids": order,
        "apply_action_ids": ids,
        "not_independently_reversible": isolated_forbidden,
        "assumes_independent_reversibility": not isolated_forbidden,
        "nodes": [
            {
                "action_id": index[action_id].action_id,
                "action_type": index[action_id].action_type,
                "inverse_operation": index[action_id].rollback.inverse_operation,
                "inverse_params": dict(index[action_id].rollback.inverse_params),
                "reversibility": index[action_id].rollback.reversibility.value,
                "depends_on": list(index[action_id].rollback.depends_on),
            }
            for action_id in order
        ],
    }


def isolated_rollback_permitted(rollback_plan: dict[str, Any], action_id: str) -> bool:
    forbidden = set(rollback_plan.get("not_independently_reversible") or [])
    return action_id not in forbidden


def volume_intent(
    *,
    session: SessionState,
    track: TrackState,
    requested_after: float,
    plan_id: str | None = None,
    user_intent: str = "",
    decision_after_verify: ApplyDecision = ApplyDecision.KEEP,
) -> MutationIntent:
    """Build a certified SET_TRACK_VOLUME intent from an authoritative snapshot."""
    attach_tokens(session)
    requested_after = validate_mixer_volume(requested_after)
    before = float(track.mixer.volume)
    action_id = f"act_{uuid4().hex[:10]}"
    durable = ref_from_track(track, project_identity=session.project_identity or "")
    return MutationIntent(
        plan_id=plan_id or f"sw_{uuid4().hex[:12]}",
        kind=KIND_PRODUCTION_MUSICAL,
        user_intent=user_intent
        or f"{CERTIFIED_PRODUCTION_ACTION} {track.name} {before}->{requested_after}",
        project_identity=session.project_identity or "",
        expected_revision=session.revision,
        expected_session_hash=session.state_hash,
        expected_project_token=session.project_token or "",
        expected_audible_token=session.audible_token or "",
        expected_incarnation_id=session.session_incarnation_id or "",
        targets=[
            MutationTarget(
                action_id=action_id,
                ref=durable.model_dump(mode="json"),
                stable_id=track.stable_id,
                name_at_plan=track.name,
                fingerprint=TargetFingerprint(**fingerprint_track(track)),
                locator=TargetLocator(track_index=track.index),
                session_incarnation_id=session.session_incarnation_id or "",
            )
        ],
        preconditions=[
            MutationPrecondition(
                code="EXPECTED_VOLUME",
                expected=before,
                observed=before,
                satisfied=True,
                detail="authoritative pre-write volume",
            ),
            MutationPrecondition(
                code="ACTION_CERTIFIED",
                expected=CERTIFIED_PRODUCTION_ACTION,
                observed=CERTIFIED_PRODUCTION_ACTION,
                satisfied=True,
            ),
        ],
        executions=[
            MutationExecution(
                action_id=action_id,
                action_type=CERTIFIED_PRODUCTION_ACTION,
                operation="set_mixer_volume",
                arguments={"volume": requested_after},
                expected_before={"volume": before},
                expected_after={"volume": requested_after},
                certified=True,
                rollback=MutationRollback(
                    inverse_operation="set_mixer_volume",
                    inverse_params={"volume": before},
                    reversibility=RollbackReversibility.INDEPENDENT,
                    parameter="track.mixer.volume",
                    restore_value=before,
                    prepared=True,
                ),
            )
        ],
        decision_after_verify=decision_after_verify,
    )


def _volume_close(left: float, right: float, tol: float = VOLUME_READBACK_TOLERANCE) -> bool:
    return abs(float(left) - float(right)) <= tol


def _atomic_write(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)
    return path


class SafeWriteExecutor:
    """Canonical production-write lifecycle. Core owns journals and rollback."""

    def __init__(
        self,
        tools: AgentTools,
        *,
        journal: DurableJournal | None = None,
        registry: SafeWriteRegistry | None = None,
        persist_dir: Path | None = None,
        experimental_midi_track_ids: frozenset[str] = frozenset(),
    ) -> None:
        self.tools = tools
        self.transactions: TransactionManager = tools.transactions
        self.journal = journal if journal is not None else tools.transactions.journal
        self.registry = registry or SafeWriteRegistry()
        self.persist_dir = persist_dir
        self.experimental_midi_track_ids = experimental_midi_track_ids

    def _action_allowed(self, intent: MutationIntent, action_type: str) -> bool:
        if intent.kind == KIND_EXPERIMENTAL_MIDI_PHRASE_V1:
            return bool(self.experimental_midi_track_ids) and action_type == "CREATE_MIDI_PHRASE"
        return action_is_certified_for_intent(intent, action_type)

    def recover(self) -> list[dict[str, Any]]:
        """Crash recovery. Never auto-mutates. Never invents certainty."""
        if self.journal is None:
            return []
        return classify_journal(self.journal.read_all())

    def run(
        self,
        intent: MutationIntent,
        *,
        cancellation: WriteCancellation | None = None,
        stop_at: MutationPhase | None = None,
    ) -> MutationResult:
        result = MutationResult(plan_id=intent.plan_id, intent=intent)
        cancellation = cancellation or WriteCancellation()
        try:
            return self._run(intent, result, cancellation, stop_at)
        except WriteInDoubt as exc:
            return self._fail(
                result,
                MutationFailure.IN_DOUBT,
                MutationPhase.EXECUTE,
                str(exc),
                decision=ApplyDecision.IN_DOUBT,
            )
        except Exception as exc:  # noqa: BLE001
            if result.failure is not None:
                return result
            if self.transactions._open is not None:
                open_status = self.transactions._open.status
                if open_status in {TransactionStatus.SENT, TransactionStatus.IN_DOUBT}:
                    self.transactions.mark_in_doubt(str(exc))
                    return self._fail(
                        result,
                        MutationFailure.IN_DOUBT,
                        MutationPhase.EXECUTE,
                        str(exc),
                        decision=ApplyDecision.IN_DOUBT,
                    )
                self.transactions.abort(str(exc))
            return self._fail(
                result,
                MutationFailure.EXECUTION_FAILED,
                result.phase,
                str(exc),
            )

    def _run(
        self,
        intent: MutationIntent,
        result: MutationResult,
        cancellation: WriteCancellation,
        stop_at: MutationPhase | None,
    ) -> MutationResult:
        result.phase = MutationPhase.PLAN
        result.lifecycle.append(MutationPhase.PLAN.value)

        if intent.kind == KIND_EXPERIMENTAL_MIDI_PHRASE_V1 and (
            not self.experimental_midi_track_ids
            or len(intent.executions) != 1 or len(intent.targets) != 1
            or intent.executions[0].action_type != "CREATE_MIDI_PHRASE"
        ):
            return self._fail(
                result, MutationFailure.PRECONDITION_FAILED, MutationPhase.PLAN,
                "MIDI_PHRASE_EXPERIMENT_DISABLED_OR_INVALID",
            )

        if intent.kind == KIND_TEMPORARY_CAPTURE_HOST:
            return self._fail(
                result,
                MutationFailure.PRECONDITION_FAILED,
                MutationPhase.PLAN,
                "capture-host MutationBatch is not a musical production write",
            )

        session = self.tools.get_session_snapshot()
        attach_tokens(session)

        result.phase = MutationPhase.RECONCILE_PROJECT
        result.lifecycle.append(MutationPhase.RECONCILE_PROJECT.value)
        project_error = self._reconcile_project(intent, session)
        if project_error is not None:
            return self._fail(
                result, project_error[0], MutationPhase.RECONCILE_PROJECT, project_error[1]
            )

        if cancellation.requested:
            return self._cancel_before_write(intent, result, cancellation)

        result.phase = MutationPhase.RECONCILE_TARGETS
        result.lifecycle.append(MutationPhase.RECONCILE_TARGETS.value)
        resolved, target_error = self._reconcile_targets(intent, session)
        if target_error is not None:
            return self._fail(
                result, target_error[0], MutationPhase.RECONCILE_TARGETS, target_error[1]
            )

        result.phase = MutationPhase.VALIDATE_PRECONDITIONS
        result.lifecycle.append(MutationPhase.VALIDATE_PRECONDITIONS.value)
        pre_error = self._validate_preconditions(intent, session, resolved)
        if pre_error is not None:
            return self._fail(
                result,
                pre_error[0],
                MutationPhase.VALIDATE_PRECONDITIONS,
                pre_error[1],
            )

        if cancellation.requested:
            return self._cancel_before_write(intent, result, cancellation)

        try:
            rollback_plan = plan_compound_rollback(intent.executions)
        except ValueError as exc:
            return self._fail(
                result,
                MutationFailure.PRECONDITION_FAILED,
                MutationPhase.VALIDATE_PRECONDITIONS,
                str(exc),
            )
        result.rollback_plan = rollback_plan
        for step in intent.executions:
            step.rollback.prepared = True

        uncertified = [
            step.action_type
            for step in intent.executions
            if not self._action_allowed(intent, step.action_type)
        ]
        if uncertified:
            result.phase = MutationPhase.PERSIST_ROLLBACK
            result.lifecycle.append(MutationPhase.PERSIST_ROLLBACK.value)
            prestate_path = self._persist_prestate(
                intent, session, resolved, rollback_plan
            )
            result.prestate_path = str(prestate_path)
            return self._fail(
                result,
                MutationFailure.PRECONDITION_FAILED,
                MutationPhase.VALIDATE_PRECONDITIONS,
                "uncertified musical action: " + ",".join(uncertified),
            )

        result.phase = MutationPhase.PERSIST_ROLLBACK
        result.lifecycle.append(MutationPhase.PERSIST_ROLLBACK.value)
        prestate_path = self._persist_prestate(intent, session, resolved, rollback_plan)
        result.prestate_path = str(prestate_path)
        if self.journal is not None:
            result.journal_path = str(self.journal.path)

        previous = self.registry.peek(intent)
        if previous and previous != intent.plan_id:
            if self.transactions._open is not None:
                if self.transactions._open.status in {
                    TransactionStatus.SENT,
                    TransactionStatus.APPLIED,
                    TransactionStatus.IN_DOUBT,
                }:
                    return self._fail(
                        result,
                        MutationFailure.STALE_PLAN,
                        MutationPhase.PREPARED,
                        f"in-flight {previous} cannot be superseded",
                    )
                self.transactions.mark_superseded(f"superseded by {intent.plan_id}")
            self.registry.release_plan(previous)

        txn = self.transactions.begin(intent.user_intent or intent.plan_id, session)
        result.transaction_id = txn.transaction_id
        for step in intent.executions:
            target = next(item for item in intent.targets if item.action_id == step.action_id)
            self.transactions.plan_write(
                command_id=f"prep_{txn.transaction_id}_{step.action_id}",
                operation=step.operation,
                expected_revision=session.revision,
                before=step.expected_before,
                expected_after=step.expected_after,
                target_stable_id=target.stable_id,
                target_fingerprint=target.fingerprint.model_dump(),
                target_name_at_apply=target.name_at_plan,
            )

        self.transactions.mark_prepared()
        result.phase = MutationPhase.PREPARED
        result.lifecycle.append(MutationPhase.PREPARED.value)
        self.registry.register_prepared(intent)

        if stop_at is MutationPhase.PREPARED:
            return result

        if cancellation.requested:
            self.registry.release(intent)
            return self._cancel_open(result, cancellation)

        guards = {
            target.name_at_plan: self._guard_for_target(
                session,
                target.name_at_plan,
                next(step for step in intent.executions if step.action_id == target.action_id),
            )
            for target in intent.targets
        }

        result.phase = MutationPhase.EXECUTE
        result.lifecycle.append(MutationPhase.EXECUTE.value)
        applied: list[str] = []
        unknown: list[str] = []
        not_attempted = [step.action_id for step in intent.executions]
        for step in intent.executions:
            if cancellation.requested:
                result.applied_action_ids = list(applied)
                result.not_attempted_action_ids = list(not_attempted)
                self.registry.release(intent)
                if applied:
                    rollback_error = self._rollback_applied(result, intent)
                    if rollback_error:
                        return self._fail(
                            result,
                            MutationFailure.ROLLBACK_FAILED,
                            MutationPhase.ROLLBACK,
                            rollback_error,
                        )
                    return self._fail(
                        result,
                        MutationFailure.CANCELLED,
                        MutationPhase.ROLLBACK,
                        cancellation.reason,
                        decision=ApplyDecision.ROLLBACK,
                    )
                return self._cancel_open(result, cancellation)
            not_attempted.remove(step.action_id)
            try:
                write_result = self._execute_certified_step(step, resolved[step.action_id], session)
                if step.action_type == "LOAD_DEVICE":
                    step.expected_after["device_stable_id"] = write_result["device_stable_id"]
                    step.rollback.inverse_params["device_stable_id"] = write_result["device_stable_id"]
                if step.action_type == "LOAD_SAMPLE":
                    for key in ("clip_stable_id", "device_stable_id"):
                        if key in write_result:
                            step.expected_after[key] = write_result[key]
                            step.rollback.inverse_params[key] = write_result[key]
                if step.action_type == "DUPLICATE_CLIP_TO_ARRANGEMENT":
                    step.expected_after["arrangement_clip_ids"] = list(write_result["arrangement_clip_ids"])
                    step.rollback.inverse_params["arrangement_clip_ids"] = list(write_result["arrangement_clip_ids"])
                if step.action_type == "CREATE_TRACK":
                    created = self._created_track_from_result(write_result)
                    target = next(item for item in intent.targets if item.action_id == step.action_id)
                    target.stable_id = created.stable_id
                    target.name_at_plan = created.name
                    target.locator = TargetLocator(track_index=created.index)
                    target.fingerprint = TargetFingerprint(**fingerprint_track(created))
                    target.ref = ref_from_track(
                        created, project_identity=session.project_identity or ""
                    ).model_dump(mode="json")
                    resolved[step.action_id] = created
                    # Bind dependent CREATE_PATTERN targets to the exact
                    # authoritative track just created.  A pattern can never
                    # resolve against a user track by name before this point.
                    for dependent in intent.executions:
                        if step.action_id not in dependent.rollback.depends_on:
                            continue
                        dependent_target = next(
                            item for item in intent.targets if item.action_id == dependent.action_id
                        )
                        dependent_target.stable_id = created.stable_id
                        dependent_target.name_at_plan = created.name
                        dependent_target.locator = TargetLocator(track_index=created.index)
                        dependent_target.fingerprint = TargetFingerprint(**fingerprint_track(created))
                        dependent_target.ref = ref_from_track(
                            created, project_identity=session.project_identity or ""
                        ).model_dump(mode="json")
                        resolved[dependent.action_id] = created
                        # A newly-created MIDI track may receive a Live default
                        # device.  Capture that authoritative post-create
                        # inventory as the baseline for a dependent LOAD_DEVICE
                        # step; otherwise verification would misclassify the
                        # default device as an unrelated mutation.
                        if dependent.action_type == "LOAD_DEVICE":
                            guards[dependent_target.name_at_plan]["device_ids"] = [
                                item.stable_id for item in created.devices
                            ]
            except WriteInDoubt as exc:
                unknown.append(step.action_id)
                unknown.extend(not_attempted)
                result.unknown_action_ids = list(unknown)
                result.applied_action_ids = list(applied)
                result.not_attempted_action_ids = list(not_attempted)
                self.registry.release(intent)
                if self.transactions._open is not None:
                    self.transactions.mark_in_doubt(str(exc))
                return self._fail(
                    result,
                    MutationFailure.IN_DOUBT,
                    MutationPhase.EXECUTE,
                    str(exc),
                    decision=ApplyDecision.IN_DOUBT,
                )
            except Exception as exc:  # noqa: BLE001
                result.applied_action_ids = list(applied)
                result.not_attempted_action_ids = list(not_attempted)
                self.registry.release(intent)
                if applied:
                    rollback_error = self._rollback_applied(result, intent)
                    if rollback_error:
                        return self._fail(
                            result,
                            MutationFailure.ROLLBACK_FAILED,
                            MutationPhase.ROLLBACK,
                            f"{exc}; {rollback_error}",
                        )
                    return self._fail(
                        result,
                        MutationFailure.PARTIAL_FAILURE,
                        MutationPhase.EXECUTE,
                        str(exc),
                    )
                if self.transactions._open is not None:
                    if self.transactions._open.status == TransactionStatus.SENT:
                        self.transactions.mark_in_doubt(str(exc))
                        return self._fail(
                            result,
                            MutationFailure.IN_DOUBT,
                            MutationPhase.EXECUTE,
                            str(exc),
                            decision=ApplyDecision.IN_DOUBT,
                        )
                    self.transactions.abort(str(exc))
                return self._fail(
                    result,
                    MutationFailure.EXECUTION_FAILED,
                    MutationPhase.EXECUTE,
                    str(exc),
                )
            applied.append(step.action_id)
            result.musical_writes += 1
        result.applied_action_ids = list(applied)

        result.phase = MutationPhase.READBACK
        result.lifecycle.append(MutationPhase.READBACK.value)
        after = self.tools.get_session_snapshot()
        attach_tokens(after)
        readbacks, readback_error = self._readback(intent, after, resolved)
        result.readbacks = readbacks
        if readback_error is not None:
            self.registry.release(intent)
            rollback_error = self._rollback_applied(result, intent)
            failure = (
                MutationFailure.ROLLBACK_FAILED
                if rollback_error
                else readback_error[0]
            )
            detail = readback_error[1]
            if rollback_error:
                detail = f"{detail}; {rollback_error}"
            return self._fail(result, failure, MutationPhase.READBACK, detail)

        result.phase = MutationPhase.RECONCILE
        result.lifecycle.append(MutationPhase.RECONCILE.value)
        for step in intent.executions:
            outcome = self.transactions.reconcile(
                operation=step.operation,
                before=step.expected_before,
                expected_after=step.expected_after,
                session=after,
            )
            if outcome is not ReconcileResult.SATISFIED:
                self.registry.release(intent)
                if self.transactions._open is not None:
                    self.transactions.mark_in_doubt(
                        f"reconcile {outcome.value} for {step.action_id}"
                    )
                return self._fail(
                    result,
                    MutationFailure.IN_DOUBT,
                    MutationPhase.RECONCILE,
                    f"reconcile {outcome.value}",
                    decision=ApplyDecision.IN_DOUBT,
                )

        result.phase = MutationPhase.VERIFY
        result.lifecycle.append(MutationPhase.VERIFY.value)
        verification, verify_error = self._verify(intent, guards, after)
        result.verification = verification
        if verify_error is not None:
            self.registry.release(intent)
            rollback_error = self._rollback_applied(result, intent)
            failure = (
                MutationFailure.ROLLBACK_FAILED
                if rollback_error
                else verify_error[0]
            )
            detail = verify_error[1]
            if rollback_error:
                detail = f"{detail}; {rollback_error}"
            return self._fail(result, failure, MutationPhase.VERIFY, detail)

        if cancellation.requested:
            intent = intent.model_copy(
                update={"decision_after_verify": ApplyDecision.ROLLBACK}
            )

        if intent.decision_after_verify is ApplyDecision.ROLLBACK:
            result.phase = MutationPhase.ROLLBACK
            result.lifecycle.append(MutationPhase.ROLLBACK.value)
            self.registry.release(intent)
            rollback_error = self._rollback_applied(result, intent, already_recorded=True)
            if rollback_error:
                return self._fail(
                    result,
                    MutationFailure.ROLLBACK_FAILED,
                    MutationPhase.ROLLBACK,
                    rollback_error,
                )
            restored = self.tools.get_session_snapshot()
            attach_tokens(restored)
            result.phase = MutationPhase.ROLLBACK_READBACK
            result.lifecycle.append(MutationPhase.ROLLBACK_READBACK.value)
            rb_readbacks, rb_error = self._rollback_readback(intent, restored)
            result.readbacks.extend(rb_readbacks)
            if rb_error is not None:
                return self._fail(
                    result, rb_error[0], MutationPhase.ROLLBACK_READBACK, rb_error[1]
                )
            result.phase = MutationPhase.ROLLBACK_VERIFY
            result.lifecycle.append(MutationPhase.ROLLBACK_VERIFY.value)
            rb_verify, rb_verify_error = self._verify_restore(intent, guards, restored)
            result.rollback_verification = rb_verify
            if rb_verify_error is not None:
                return self._fail(
                    result,
                    rb_verify_error[0],
                    MutationPhase.ROLLBACK_VERIFY,
                    rb_verify_error[1],
                )
            result.ok = True
            result.decision = ApplyDecision.ROLLBACK
            result.journal_terminal_state = (
                self.transactions.history[-1].status.value
                if self.transactions.history
                else TransactionStatus.ROLLED_BACK.value
            )
            result.open_transaction = self.transactions._open is not None
            if cancellation.requested:
                result.ok = False
                result.failure = MutationFailure.CANCELLED
                result.lifecycle.append(MutationFailure.CANCELLED.value)
            return result

        self.registry.release(intent)
        self.transactions.commit(
            {
                "SAFE_WRITE_FOUNDATION_V2": "KEEP",
                "readbacks": [item.model_dump(mode="json") for item in readbacks],
            },
            session=after,
        )
        result.phase = MutationPhase.KEEP
        result.lifecycle.append(MutationPhase.KEEP.value)
        result.ok = True
        result.decision = ApplyDecision.KEEP
        result.journal_terminal_state = TransactionStatus.VERIFIED.value
        result.open_transaction = self.transactions._open is not None
        return result

    def _reconcile_project(
        self, intent: MutationIntent, session: SessionState
    ) -> tuple[MutationFailure, str] | None:
        if intent.project_identity and (session.project_identity or "") != intent.project_identity:
            return (
                MutationFailure.PROJECT_MISMATCH,
                f"planned {intent.project_identity!r} live {session.project_identity!r}",
            )
        if (
            intent.expected_incarnation_id
            and session.session_incarnation_id
            and intent.expected_incarnation_id != session.session_incarnation_id
        ):
            return (
                MutationFailure.STALE_PLAN,
                "session incarnation changed",
            )
        if (
            intent.expected_revision is not None
            and session.revision != intent.expected_revision
        ):
            return (
                MutationFailure.STALE_PLAN,
                f"revision {session.revision} != {intent.expected_revision}",
            )
        if intent.expected_session_hash and session.state_hash != intent.expected_session_hash:
            return (MutationFailure.STALE_PLAN, "session hash mismatch")
        if intent.expected_project_token and (
            (session.project_token or "") != intent.expected_project_token
        ):
            return (MutationFailure.STALE_PLAN, "PROJECT_STATE_TOKEN mismatch")
        if intent.expected_audible_token and (
            (session.audible_token or "") != intent.expected_audible_token
        ):
            return (MutationFailure.STALE_PLAN, "AUDIBLE_STATE_TOKEN mismatch")
        return None

    def _reconcile_targets(
        self, intent: MutationIntent, session: SessionState
    ) -> tuple[dict[str, TrackState], tuple[MutationFailure, str] | None]:
        resolved: dict[str, TrackState] = {}
        for target in intent.targets:
            step = next((item for item in intent.executions if item.action_id == target.action_id), None)
            if step is not None and step.action_type == "CREATE_TRACK":
                # CREATE_TRACK has no pre-existing track target. The project
                # identity and session tokens are the preconditions; the
                # created track receives its PersistentObjectRef after write.
                resolved[target.action_id] = None  # type: ignore[assignment]
                continue
            if step is not None and step.action_type in {"CREATE_PATTERN", "LOAD_DEVICE", "DUPLICATE_CLIP_TO_ARRANGEMENT"}:
                dependencies = set(step.rollback.depends_on)
                if dependencies:
                    dependency = next((item for item in dependencies if any(
                        candidate.action_id == item and candidate.action_type == "CREATE_TRACK"
                        for candidate in intent.executions
                    )), None)
                    dependency_step = next(
                        (item for item in intent.executions if item.action_id == dependency), None
                    )
                    if dependency_step is not None and dependency_step.action_type == "CREATE_TRACK":
                        # This is intentionally unresolved until the CREATE_TRACK
                        # readback binds the dependent target.
                        resolved[target.action_id] = None  # type: ignore[assignment]
                        continue
            ref = PersistentObjectRef.model_validate(target.ref)
            outcome = resolve_track(session, ref)
            if outcome.status is ResolveStatus.PROJECT_MISMATCH:
                return {}, (MutationFailure.PROJECT_MISMATCH, outcome.reason)
            if outcome.status is ResolveStatus.TARGET_AMBIGUOUS:
                return {}, (MutationFailure.TARGET_AMBIGUOUS, outcome.reason)
            if outcome.status is ResolveStatus.TARGET_NOT_FOUND:
                return {}, (MutationFailure.TARGET_NOT_FOUND, outcome.reason)
            if outcome.status is not ResolveStatus.RESOLVED or outcome.track_index is None:
                return {}, (MutationFailure.TARGET_NOT_FOUND, outcome.reason)
            track = session.tracks[outcome.track_index]
            target.stable_id = track.stable_id
            target.locator = TargetLocator(track_index=track.index)
            target.fingerprint = TargetFingerprint(**fingerprint_track(track))
            resolved[target.action_id] = track
        return resolved, None

    def _guard_for_target(
        self, session: SessionState, target_name: str, step: MutationExecution
    ) -> dict[str, Any]:
        if step.action_type == "CREATE_TRACK":
            return {
                "track_ids": [item.stable_id for item in session.tracks],
                "track_fingerprints": [_track_fingerprint_key(item) for item in session.tracks],
            }
        if (
            step.action_type in {"CREATE_PATTERN", "LOAD_DEVICE", "DUPLICATE_CLIP_TO_ARRANGEMENT"}
            and session.track_by_name(target_name) is None
            and step.rollback.depends_on
        ):
            guard = {
                "track_ids": [item.stable_id for item in session.tracks],
                "track_fingerprints": [_track_fingerprint_key(item) for item in session.tracks],
                "pattern_absent": True,
                "device_ids": [],
            }
            if step.action_type == "DUPLICATE_CLIP_TO_ARRANGEMENT":
                guard["arrangement_ids"] = [
                    str(item.get("id", ""))
                    for item in self.tools.daw.get_arrangement_clips().get("clips", [])
                ]
            return guard
        track = session.track_by_name(target_name)
        if track is None:
            return snapshot_guard_state(session, target_name)
        guard = snapshot_guard_state(session, target_name)
        if step.action_type == "CREATE_MIDI_PHRASE":
            guard["phrase_preservation_token"] = phrase_preservation_token(
                session, excluded_slots={track.stable_id: {int(step.arguments["clip_index"])}},
            )
        if step.action_type in {"LOAD_DEVICE", "SET_DEVICE_PARAMETER"}:
            guard["device_ids"] = [item.stable_id for item in track.devices]
        if step.action_type == "SET_DEVICE_PARAMETER":
            guard["device_values"] = {
                f"{device.index}:{parameter.index}": float(parameter.value)
                for device in track.devices
                for parameter in device.parameters
            }
        if step.action_type == "LOAD_SAMPLE":
            guard["clip_ids"] = [item.stable_id for item in track.clips]
            guard["clip_slots"] = [item.slot_index for item in track.clips]
            guard["clip_samples"] = {
                str(item.slot_index): item.sample_uri for item in track.clips
            }
            guard["device_ids"] = [item.stable_id for item in track.devices]
        if step.action_type == "DUPLICATE_CLIP_TO_ARRANGEMENT":
            arrangement = self.tools.daw.get_arrangement_clips()
            guard["arrangement_ids"] = [str(item.get("id", "")) for item in arrangement.get("clips", [])]
        return guard

    def _validate_preconditions(
        self,
        intent: MutationIntent,
        _session: SessionState,
        resolved: dict[str, TrackState],
    ) -> tuple[MutationFailure, str] | None:
        if not intent.executions:
            return (MutationFailure.PRECONDITION_FAILED, "empty mutation plan")
        multi_create_count = min(
            (
                int(step.expected_before.get("track_count", len(_session.tracks)))
                for step in intent.executions
                if step.action_type == "CREATE_TRACK"
                and step.expected_after.get("multi_intent")
            ),
            default=None,
        )
        for step in intent.executions:
            if not self._action_allowed(intent, step.action_type):
                continue
            track = resolved[step.action_id]
            if step.action_type == "CREATE_MIDI_PHRASE":
                if track is None or track.stable_id not in self.experimental_midi_track_ids:
                    return (MutationFailure.PRECONDITION_FAILED, "MIDI_PHRASE_TRACK_NOT_OWNED")
                if not MIDI_PHRASE_CAPABILITIES <= set(getattr(self.tools.daw, "capabilities", ())):
                    return (MutationFailure.PRECONDITION_FAILED, "MIDI_PHRASE_CAPABILITIES_UNAVAILABLE")
                try:
                    phrase_from_arguments(step.arguments)
                    blocker = empty_phrase_slot_blocker(track, step.arguments["clip_index"])
                except (KeyError, TypeError, ValueError) as exc:
                    return (MutationFailure.PRECONDITION_FAILED, f"MIDI_PHRASE_INVALID: {exc}")
                if blocker:
                    return (MutationFailure.PRECONDITION_FAILED, blocker)
                observed = phrase_preservation_token(
                    _session, excluded_slots={track.stable_id: {step.arguments["clip_index"]}},
                )
                if (
                    observed != step.expected_before.get("preservation_token")
                    or not step.rollback.prepared or step.rollback.inverse_operation != "delete_clip"
                    or step.expected_before.get("clip_exists") is not False
                    or step.operation != "create_pattern"
                ):
                    return (MutationFailure.PRECONDITION_FAILED, "MIDI_PHRASE_PRESTATE_CHANGED")
                continue
            if step.action_type == "CREATE_TRACK":
                expected_track_count = (
                    multi_create_count
                    if step.expected_after.get("multi_intent")
                    else int(step.expected_before.get("track_count", len(_session.tracks)))
                )
                if len(_session.tracks) != expected_track_count:
                    return (MutationFailure.PRECONDITION_FAILED, "track count changed before CREATE_TRACK")
                if any(item.name == str(step.arguments.get("name", "")) for item in _session.tracks):
                    return (MutationFailure.PRECONDITION_FAILED, "track name already exists")
                if not step.rollback.prepared:
                    return (MutationFailure.PRECONDITION_FAILED, f"rollback not prepared for {step.action_id}")
                continue
            if step.action_type == "CREATE_PATTERN":
                if track is not None:
                    return (MutationFailure.PRECONDITION_FAILED, "pattern target already exists")
                if not step.arguments.get("notes"):
                    return (MutationFailure.PRECONDITION_FAILED, "pattern notes are required")
                if float(step.arguments.get("length_beats", 0)) <= 0:
                    return (MutationFailure.PRECONDITION_FAILED, "pattern length must be positive")
                if not step.rollback.prepared:
                    return (MutationFailure.PRECONDITION_FAILED, f"rollback not prepared for {step.action_id}")
                continue
            if step.action_type == "LOAD_DEVICE":
                if track is None:
                    if len(step.rollback.depends_on) == 1:
                        continue
                    return (MutationFailure.TARGET_NOT_FOUND, "device target track missing")
                if len(track.devices) != int(step.expected_before.get("device_count", len(track.devices))):
                    return (MutationFailure.PRECONDITION_FAILED, "device inventory changed before LOAD_DEVICE")
                if not step.rollback.prepared:
                    return (MutationFailure.PRECONDITION_FAILED, f"rollback not prepared for {step.action_id}")
                continue
            if step.action_type == "DUPLICATE_CLIP_TO_ARRANGEMENT":
                if track is None:
                    dependency_types = {
                        candidate.action_type for candidate in intent.executions
                        if candidate.action_id in step.rollback.depends_on
                    }
                    if dependency_types != {"CREATE_TRACK", "CREATE_PATTERN"}:
                        return (MutationFailure.TARGET_NOT_FOUND, "arrangement target track missing")
                else:
                    clip_index = int(step.arguments["clip_index"])
                    if not any(item.slot_index == clip_index for item in track.clips):
                        return (MutationFailure.PRECONDITION_FAILED, "source clip missing")
                try:
                    self.tools.daw.get_arrangement_clips()
                except Exception as exc:  # noqa: BLE001
                    return (MutationFailure.PRECONDITION_FAILED, f"arrangement readback unavailable: {exc}")
                if not step.rollback.prepared:
                    return (MutationFailure.PRECONDITION_FAILED, f"rollback not prepared for {step.action_id}")
                continue
            if step.action_type == "LOAD_SAMPLE":
                if track is None:
                    return (MutationFailure.TARGET_NOT_FOUND, "sample target track missing")
                if not step.arguments.get("sample_uri"):
                    return (MutationFailure.PRECONDITION_FAILED, "sample URI is required")
                clip_index = int(step.arguments["clip_index"])
                if track.role != "audio":
                    if not step.rollback.prepared:
                        return (MutationFailure.PRECONDITION_FAILED, f"rollback not prepared for {step.action_id}")
                    continue
                clip = next((item for item in track.clips if item.slot_index == clip_index), None)
                if clip is not None:
                    return (MutationFailure.PRECONDITION_FAILED, f"clip slot occupied: {clip_index}")
                if not step.rollback.prepared:
                    return (MutationFailure.PRECONDITION_FAILED, f"rollback not prepared for {step.action_id}")
                continue
            if step.action_type == "SET_DEVICE_PARAMETER":
                if track is None:
                    return (MutationFailure.TARGET_NOT_FOUND, "parameter target track missing")
                device_index = int(step.arguments["device_index"])
                parameter_index = int(step.arguments["parameter_index"])
                if device_index < 0 or device_index >= len(track.devices) or parameter_index < 0 or parameter_index >= len(track.devices[device_index].parameters):
                    return (MutationFailure.TARGET_NOT_FOUND, "parameter identity missing")
                if abs(float(track.devices[device_index].parameters[parameter_index].value) - float(step.expected_before["value"])) > 1e-3:
                    return (MutationFailure.PRECONDITION_FAILED, "parameter pre-state mismatch")
                if not step.rollback.prepared:
                    return (MutationFailure.PRECONDITION_FAILED, f"rollback not prepared for {step.action_id}")
                continue
            if step.action_type != CERTIFIED_PRODUCTION_ACTION:
                continue
            expected = float(step.expected_before.get("volume", track.mixer.volume))
            observed = float(track.mixer.volume)
            if not _volume_close(expected, observed, 0.02):
                return (
                    MutationFailure.PRECONDITION_FAILED,
                    f"volume {observed} != expected {expected}",
                )
            try:
                validate_mixer_volume(step.arguments.get("volume"))
            except Exception as exc:  # noqa: BLE001
                return (MutationFailure.PRECONDITION_FAILED, str(exc))
            if not step.rollback.prepared and step.rollback.restore_value is None:
                return (
                    MutationFailure.PRECONDITION_FAILED,
                    f"rollback not prepared for {step.action_id}",
                )
        for item in intent.preconditions:
            if item.required and item.satisfied is False:
                return (MutationFailure.PRECONDITION_FAILED, item.code)
        return None

    def _persist_prestate(
        self,
        intent: MutationIntent,
        session: SessionState,
        resolved: dict[str, TrackState],
        rollback_plan: dict[str, Any],
    ) -> Path:
        directory = self.persist_dir
        if directory is None and self.journal is not None:
            directory = self.journal.path.parent
        if directory is None:
            directory = Path("logs") / "safe_write"
        payload = {
            "milestone": MILESTONE,
            "schema_version": SCHEMA_VERSION,
            "plan_id": intent.plan_id,
            "kind": intent.kind,
            "project_identity": session.project_identity,
            "session_revision": session.revision,
            "session_hash": session.state_hash,
            "session_incarnation_id": session.session_incarnation_id,
            "project_token": session.project_token,
            "audible_token": session.audible_token,
            "targets": [target.model_dump(mode="json") for target in intent.targets],
            "before": {
                action_id: (
                    {
                        "track_count": len(session.tracks),
                        "track_ids": [item.stable_id for item in session.tracks],
                    }
                    if track is None
                    else {
                        "name": track.name,
                        "index": track.index,
                        "stable_id": track.stable_id,
                        "volume": float(track.mixer.volume),
                    }
                )
                for action_id, track in resolved.items()
            },
            "rollback_plan": rollback_plan,
            "executions": [step.model_dump(mode="json") for step in intent.executions],
        }
        path = Path(directory) / f"{intent.plan_id}.prestate.json"
        return _atomic_write(path, payload)

    def _execute_certified_step(
        self,
        step: MutationExecution,
        track: TrackState,
        session: SessionState,
    ) -> dict[str, Any]:
        if step.action_type == "CREATE_TRACK":
            return self._execute_create_track(step, session)
        if step.action_type in {"CREATE_PATTERN", "CREATE_MIDI_PHRASE"}:
            return self._execute_create_pattern(step, track, session)
        if step.action_type == "LOAD_DEVICE":
            return self._execute_load_device(step, track, session)
        if step.action_type == "DUPLICATE_CLIP_TO_ARRANGEMENT":
            return self._execute_duplicate_clip_to_arrangement(step, track, session)
        if step.action_type == "LOAD_SAMPLE":
            return self._execute_load_sample(step, track, session)
        if step.action_type == "SET_DEVICE_PARAMETER":
            return self._execute_set_device_parameter(step, track, session)
        if step.action_type != CERTIFIED_PRODUCTION_ACTION:
            raise RuntimeError(f"uncertified action reached execute: {step.action_type}")
        volume = validate_mixer_volume(step.arguments["volume"])
        command_id = f"cmd_{uuid4().hex[:12]}"
        before_volume = float(track.mixer.volume)
        self.transactions.mark_sent(command_id, step.operation)
        try:
            result = self.tools.daw.set_mixer_volume(track.index, volume)
        except WriteInDoubt as exc:
            live = self.tools.get_session_snapshot()
            outcome = self.transactions.reconcile(
                operation=step.operation,
                before=step.expected_before,
                expected_after=step.expected_after,
                session=live,
            )
            if outcome is ReconcileResult.SATISFIED:
                result = {"reconciled": True, **step.expected_after}
            else:
                raise WriteInDoubt(step.operation, command_id) from exc
        live_track = self._track_named_or_index(track.name, track.index)
        self.transactions.record(
            target_stable_id=live_track.stable_id,
            target_locator_at_apply=TargetLocator(track_index=live_track.index),
            target_fingerprint=TargetFingerprint(**fingerprint_track(live_track)),
            target_name_at_apply=live_track.name,
            operation=step.operation,
            before={"volume": before_volume},
            after={"volume": result.get("volume", volume)},
            expected_after=step.expected_after,
            inverse_operation=step.rollback.inverse_operation,
            inverse_params=dict(step.rollback.inverse_params),
            command_id=command_id,
            expected_revision=session.revision,
        )
        return result

    def _execute_create_track(
        self, step: MutationExecution, session: SessionState
    ) -> dict[str, Any]:
        """Create, identify and journal exactly one new track."""
        command_id = f"cmd_{uuid4().hex[:12]}"
        operation = step.operation
        self.transactions.mark_sent(command_id, operation)
        # A compound multi-element intent may create several tracks.  The
        # precondition snapshot is intentionally shared, but the durable
        # journal must record the live count immediately before this step.
        live_before = self.tools.get_session_snapshot()
        before_ids = {track.stable_id for track in live_before.tracks}
        before_count = len(live_before.tracks)
        try:
            if operation == "create_audio_track":
                result = self.tools.daw.create_audio_track(
                    str(step.arguments["name"]), int(step.arguments.get("index", -1))
                )
            else:
                result = self.tools.daw.create_midi_track(
                    str(step.arguments["name"]), int(step.arguments.get("index", -1))
                )
        except WriteInDoubt as exc:
            live = self.tools.get_session_snapshot()
            created = [track for track in live.tracks if track.stable_id not in before_ids]
            if len(created) != 1:
                raise WriteInDoubt(operation, command_id) from exc
            result = {"reconciled": True, "index": created[0].index}
        live = self.tools.get_session_snapshot()
        created = [track for track in live.tracks if track.stable_id not in before_ids]
        if len(created) != 1:
            raise RuntimeError("CREATE_TRACK readback was ambiguous")
        track = created[0]
        self.transactions.record(
            target_stable_id=track.stable_id,
            target_locator_at_apply=TargetLocator(track_index=track.index),
            target_fingerprint=TargetFingerprint(**fingerprint_track(track)),
            target_name_at_apply=track.name,
            operation=operation,
            before={"track_count": before_count, "track_ids": sorted(before_ids)},
            after={"track_count": len(live.tracks), "stable_id": track.stable_id},
            expected_after={"track_count": before_count + 1, "stable_id": track.stable_id},
            inverse_operation="delete_track",
            inverse_params={},
            asset_id=track.stable_id,
            command_id=command_id,
            expected_revision=session.revision,
        )
        return {**result, "track_stable_id": track.stable_id, "track_index": track.index}

    def _execute_create_pattern(
        self, step: MutationExecution, track: TrackState | None, session: SessionState
    ) -> dict[str, Any]:
        """Create one editable MIDI clip and notes inside the SafeWrite txn."""
        if track is None:
            raise RuntimeError("CREATE_PATTERN target track is unresolved")
        command_id = f"cmd_{uuid4().hex[:12]}"
        self.transactions.mark_sent(command_id, step.operation)
        clip_index = int(step.arguments["clip_index"])
        length_beats = float(step.arguments["length_beats"])
        from copilot.schemas.session import MidiNote

        notes = [MidiNote.model_validate(note) for note in step.arguments.get("notes", [])]
        try:
            self.tools.daw.create_midi_clip(track.index, clip_index, length_beats)
            self.tools.daw.replace_clip_notes(track.index, clip_index, notes)
        except WriteInDoubt as exc:
            live = self.tools.get_session_snapshot()
            live_track = live.track_by_id(track.stable_id)
            clip = next((item for item in live_track.clips if item.slot_index == clip_index), None)
            if clip is None or len(clip.notes) != len(notes):
                raise WriteInDoubt(step.operation, command_id) from exc
        live = self.tools.get_session_snapshot()
        live_track = live.track_by_id(track.stable_id)
        clip = next((item for item in live_track.clips if item.slot_index == clip_index), None)
        if clip is None or len(clip.notes) != len(notes):
            raise RuntimeError("CREATE_PATTERN clip or notes missing on readback")
        self.transactions.record(
            target_stable_id=live_track.stable_id,
            target_locator_at_apply=TargetLocator(track_index=live_track.index, clip_index=clip_index),
            target_fingerprint=TargetFingerprint(**fingerprint_track(live_track)),
            target_name_at_apply=live_track.name,
            operation=step.operation,
            before={"clip_exists": False, "clip_index": clip_index},
            after={"clip_stable_id": clip.stable_id, "note_count": len(clip.notes)},
            expected_after={"clip_stable_id": clip.stable_id, "note_count": len(notes)},
            inverse_operation="delete_clip",
            inverse_params={},
            asset_id=clip.stable_id,
            command_id=command_id,
            expected_revision=session.revision,
        )
        step.expected_after["clip_stable_id"] = clip.stable_id
        return {"clip_stable_id": clip.stable_id, "clip_index": clip_index, "note_count": len(clip.notes)}

    def _created_track_from_result(self, result: dict[str, Any]) -> TrackState:
        stable_id = str(result.get("track_stable_id") or "")
        live = self.tools.get_session_snapshot()
        matches = [track for track in live.tracks if track.stable_id == stable_id]
        if len(matches) != 1:
            raise RuntimeError("created track stable identity could not be resolved")
        return matches[0]

    def _execute_load_device(
        self, step: MutationExecution, track: TrackState | None, session: SessionState
    ) -> dict[str, Any]:
        if track is None:
            raise RuntimeError("LOAD_DEVICE target track is unresolved")
        command_id = f"cmd_{uuid4().hex[:12]}"
        self.transactions.mark_sent(command_id, step.operation)
        before_ids = {device.stable_id for device in track.devices}
        try:
            result = self.tools.daw.load_instrument_or_effect(
                track.index, str(step.arguments["uri"])
            )
        except WriteInDoubt as exc:
            live = self.tools.get_session_snapshot()
            live_track = live.track_by_id(track.stable_id)
            created = [device for device in live_track.devices if device.stable_id not in before_ids]
            if len(created) != 1:
                raise WriteInDoubt(step.operation, command_id) from exc
            result = {"reconciled": True, "device_index": created[0].index}
        live = self.tools.get_session_snapshot()
        live_track = live.track_by_id(track.stable_id)
        created = [device for device in live_track.devices if device.stable_id not in before_ids]
        if len(created) != 1:
            raise RuntimeError("LOAD_DEVICE readback was ambiguous")
        device = created[0]
        self.transactions.record(
            target_stable_id=live_track.stable_id,
            target_locator_at_apply=TargetLocator(track_index=live_track.index, device_index=device.index),
            target_fingerprint=TargetFingerprint(**fingerprint_track(live_track)),
            target_name_at_apply=live_track.name,
            operation=step.operation,
            before={"device_count": len(track.devices), "device_ids": sorted(before_ids)},
            after={"device_count": len(live_track.devices), "device_stable_id": device.stable_id},
            expected_after={"device_count": len(track.devices) + 1, "device_stable_id": device.stable_id},
            inverse_operation="delete_device",
            inverse_params={},
            asset_id=device.stable_id,
            command_id=command_id,
            expected_revision=session.revision,
        )
        return {**result, "device_stable_id": device.stable_id, "device_index": device.index}

    def _execute_load_sample(self, step: MutationExecution, track: TrackState | None, session: SessionState) -> dict[str, Any]:
        if track is None:
            raise RuntimeError("LOAD_SAMPLE target track is unresolved")
        command_id = f"cmd_{uuid4().hex[:12]}"
        self.transactions.mark_sent(command_id, step.operation)
        clip_index = int(step.arguments["clip_index"])
        sample_uri = str(step.arguments["sample_uri"])
        before_device_ids = {item.stable_id for item in track.devices}
        try:
            result = self.tools.daw.load_browser_item(track.index, sample_uri, clip_index=clip_index)
        except WriteInDoubt as exc:
            live = self.tools.get_session_snapshot()
            live_track = live.track_by_id(track.stable_id)
            if track.role == "audio":
                clip = next((item for item in live_track.clips if item.slot_index == clip_index), None)
                if clip is None or not _sample_reference_matches(clip.sample_uri, sample_uri):
                    raise WriteInDoubt(step.operation, command_id) from exc
                return self._record_loaded_clip(step, session, command_id, live_track, clip, reconciled=True)
            devices = [
                item for item in live_track.devices
                if item.stable_id not in before_device_ids and _sample_device_matches(item, sample_uri)
            ]
            if len(devices) != 1:
                raise WriteInDoubt(step.operation, command_id) from exc
            return self._record_loaded_device_sample(step, session, command_id, live_track, devices[0], reconciled=True)
        live = self.tools.get_session_snapshot()
        live_track = live.track_by_id(track.stable_id)
        if track.role == "audio":
            clip = next((item for item in live_track.clips if item.slot_index == clip_index), None)
            if clip is None or not _sample_reference_matches(clip.sample_uri, sample_uri):
                raise RuntimeError("LOAD_SAMPLE clip missing or sample URI mismatched on readback")
            return self._record_loaded_clip(step, session, command_id, live_track, clip)
        device_index = result.get("device_index") if isinstance(result, dict) else None
        candidates = [
            item for item in live_track.devices
            if item.stable_id not in before_device_ids and _sample_device_matches(item, sample_uri)
        ]
        if device_index is not None:
            candidates = [item for item in candidates if item.index == int(device_index)]
        if len(candidates) != 1:
            raise RuntimeError("LOAD_SAMPLE Simpler missing on readback")
        device = candidates[0]
        return self._record_loaded_device_sample(step, session, command_id, live_track, device)

    def _record_loaded_clip(
        self, step: MutationExecution, session: SessionState, command_id: str,
        live_track: TrackState, clip, *, reconciled: bool = False,
    ) -> dict[str, Any]:
        step.expected_after["clip_stable_id"] = clip.stable_id
        self.transactions.record(
            target_stable_id=live_track.stable_id,
            target_locator_at_apply=TargetLocator(track_index=live_track.index, clip_index=clip.slot_index),
            target_fingerprint=TargetFingerprint(**fingerprint_track(live_track)), target_name_at_apply=live_track.name,
            operation=step.operation, before={"clip_index": clip.slot_index}, after={"clip_stable_id": clip.stable_id},
            expected_after={"clip_stable_id": clip.stable_id, "sample_uri": step.arguments["sample_uri"]},
            inverse_operation="delete_clip", inverse_params={}, asset_id=clip.stable_id,
            command_id=command_id, expected_revision=session.revision,
        )
        return {"clip_stable_id": clip.stable_id, "reconciled": reconciled}

    def _record_loaded_device_sample(
        self, step: MutationExecution, session: SessionState, command_id: str,
        live_track: TrackState, device, *, reconciled: bool = False,
    ) -> dict[str, Any]:
        step.expected_after["device_stable_id"] = device.stable_id
        self.transactions.record(
            target_stable_id=live_track.stable_id,
            target_locator_at_apply=TargetLocator(track_index=live_track.index, device_index=device.index),
            target_fingerprint=TargetFingerprint(**fingerprint_track(live_track)), target_name_at_apply=live_track.name,
            operation=step.operation, before={"device_count": len(live_track.devices) - 1},
            after={"device_stable_id": device.stable_id},
            expected_after={"device_stable_id": device.stable_id, "sample_uri": step.arguments["sample_uri"]},
            inverse_operation="delete_device", inverse_params={}, asset_id=device.stable_id,
            command_id=command_id, expected_revision=session.revision,
        )
        return {"device_stable_id": device.stable_id, "device_index": device.index, "reconciled": reconciled}

    def _execute_duplicate_clip_to_arrangement(
        self, step: MutationExecution, track: TrackState | None, session: SessionState
    ) -> dict[str, Any]:
        if track is None:
            raise RuntimeError("DUPLICATE_CLIP_TO_ARRANGEMENT target is unresolved")
        command_id = f"cmd_{uuid4().hex[:12]}"
        self.transactions.mark_sent(command_id, step.operation)
        before = self.tools.daw.get_arrangement_clips()
        before_ids = {str(item.get("id", "")) for item in before.get("clips", [])}
        try:
            result = self.tools.daw.duplicate_clip_to_arrangement(
                track.index,
                int(step.arguments["clip_index"]),
                float(step.arguments["destination_time"]),
                step.arguments.get("length"),
            )
        except WriteInDoubt as exc:
            after = self.tools.daw.get_arrangement_clips()
            created = [item for item in after.get("clips", []) if str(item.get("id", "")) not in before_ids]
            if not created:
                raise WriteInDoubt(step.operation, command_id) from exc
            result = {"reconciled": True}
        after = self.tools.daw.get_arrangement_clips()
        created = [item for item in after.get("clips", []) if str(item.get("id", "")) not in before_ids]
        ids = [str(item.get("id", "")) for item in created if item.get("id")]
        if not ids:
            raise RuntimeError("arrangement readback did not expose the created clip")
        self.transactions.record(
            target_stable_id=track.stable_id,
            target_locator_at_apply=TargetLocator(track_index=track.index),
            target_fingerprint=TargetFingerprint(**fingerprint_track(track)),
            target_name_at_apply=track.name,
            operation=step.operation,
            before={"arrangement_clip_ids": sorted(before_ids)},
            after={"arrangement_clip_ids": ids},
            expected_after={"arrangement_clip_ids": ids},
            inverse_operation="delete_arrangement_clips",
            inverse_params={"arrangement_clip_ids": ids},
            asset_id=ids[0],
            command_id=command_id,
            expected_revision=session.revision,
        )
        return {**result, "arrangement_clip_ids": ids}

    def _execute_set_device_parameter(self, step: MutationExecution, track: TrackState | None, session: SessionState) -> dict[str, Any]:
        if track is None:
            raise RuntimeError("SET_DEVICE_PARAMETER target track is unresolved")
        command_id = f"cmd_{uuid4().hex[:12]}"
        self.transactions.mark_sent(command_id, step.operation)
        di, pi, value = int(step.arguments["device_index"]), int(step.arguments["parameter_index"]), float(step.arguments["value"])
        try:
            result = self.tools.daw.set_device_parameter(track.index, di, pi, value)
        except WriteInDoubt as exc:
            live = self.tools.get_session_snapshot()
            live_track = live.track_by_id(track.stable_id)
            observed = live_track.devices[di].parameters[pi].value
            if abs(float(observed) - value) > 1e-3:
                raise WriteInDoubt(step.operation, command_id) from exc
            result = {"reconciled": True, "value": observed}
            value = float(observed)
        self.transactions.record(
            target_stable_id=track.stable_id,
            target_locator_at_apply=TargetLocator(track_index=track.index, device_index=di, parameter_index=pi),
            target_fingerprint=TargetFingerprint(**fingerprint_track(track)), target_name_at_apply=track.name,
            operation=step.operation, before=step.expected_before, after={"value": value}, expected_after={"value": value},
            inverse_operation="set_device_parameter", inverse_params={"value": step.expected_before["value"]},
            command_id=command_id, expected_revision=session.revision,
        )
        return {**result, "value": value}

    def _track_named_or_index(self, name: str, index: int) -> TrackState:
        session = self.tools.get_session_snapshot()
        track = session.track_by_name(name)
        if track is not None:
            return track
        if 0 <= index < len(session.tracks):
            return session.tracks[index]
        raise WriteInDoubt("set_mixer_volume")

    def _readback(
        self,
        intent: MutationIntent,
        session: SessionState,
        resolved: dict[str, TrackState],
    ) -> tuple[list[MutationReadback], tuple[MutationFailure, str] | None]:
        rows: list[MutationReadback] = []
        for step in intent.executions:
            planned = resolved[step.action_id]
            if step.action_type == "CREATE_TRACK":
                target = next(item for item in intent.targets if item.action_id == step.action_id)
                created = next((item for item in session.tracks if item.stable_id == target.stable_id), None)
                matched = created is not None
                rows.append(
                    MutationReadback(
                        action_id=step.action_id,
                        parameter="session.track",
                        expected=target.stable_id,
                        observed=None if created is None else created.stable_id,
                        matched=matched,
                        authoritative=True,
                    )
                )
                if not matched:
                    return rows, (MutationFailure.READBACK_MISMATCH, "created track missing on readback")
                continue
            if step.action_type == "CREATE_PATTERN":
                target = next(item for item in intent.targets if item.action_id == step.action_id)
                track = session.track_by_id(target.stable_id)
                clip_index = int(step.arguments["clip_index"])
                clip = next((item for item in track.clips if item.slot_index == clip_index), None)
                expected_count = int(step.expected_after.get("note_count", 0))
                matched = clip is not None and len(clip.notes) == expected_count
                rows.append(MutationReadback(
                    action_id=step.action_id,
                    parameter="clip.notes",
                    expected=expected_count,
                    observed=None if clip is None else len(clip.notes),
                    matched=matched,
                    authoritative=True,
                ))
                if not matched:
                    return rows, (MutationFailure.READBACK_MISMATCH, "pattern notes missing on readback")
                continue
            if step.action_type == "CREATE_MIDI_PHRASE":
                target = next(item for item in intent.targets if item.action_id == step.action_id)
                track = session.track_by_id(target.stable_id)
                clips = [clip for clip in track.clips if clip.slot_index == step.arguments["clip_index"]]
                clip = clips[0] if len(clips) == 1 else None
                matched = bool(
                    clip is not None
                    and clip.stable_id == step.expected_after.get("clip_stable_id")
                    and phrase_from_arguments(step.arguments).matches_clip(clip)
                )
                rows.append(MutationReadback(
                    action_id=step.action_id, parameter="clip.midi_phrase",
                    expected=step.arguments,
                    observed=clip.model_dump(mode="json") if clip is not None else None,
                    matched=matched, authoritative=True,
                ))
                if not matched:
                    return rows, (MutationFailure.READBACK_MISMATCH, "MIDI_PHRASE_CONTENT_MISMATCH")
                continue
            if step.action_type == "LOAD_DEVICE":
                target = next(item for item in intent.targets if item.action_id == step.action_id)
                track = session.track_by_id(target.stable_id)
                expected_id = str(step.expected_after.get("device_stable_id", ""))
                device = next((item for item in track.devices if item.stable_id == expected_id), None)
                matched = device is not None
                rows.append(MutationReadback(
                    action_id=step.action_id,
                    parameter="track.device",
                    expected=expected_id,
                    observed=None if device is None else device.stable_id,
                    matched=matched,
                    authoritative=True,
                ))
                if not matched:
                    return rows, (MutationFailure.READBACK_MISMATCH, "loaded device missing on readback")
                continue
            if step.action_type == "DUPLICATE_CLIP_TO_ARRANGEMENT":
                target = next(item for item in intent.targets if item.action_id == step.action_id)
                expected_ids = set(str(item) for item in step.expected_after.get("arrangement_clip_ids", []))
                actual_ids = {
                    str(item.get("id", ""))
                    for item in self.tools.daw.get_arrangement_clips().get("clips", [])
                }
                matched = bool(expected_ids) and expected_ids.issubset(actual_ids)
                rows.append(MutationReadback(action_id=step.action_id, parameter="arrangement.clip", expected=sorted(expected_ids), observed=sorted(actual_ids & expected_ids), matched=matched, authoritative=True))
                if not matched:
                    return rows, (MutationFailure.READBACK_MISMATCH, "arrangement clip missing on readback")
                continue
            if step.action_type == "LOAD_SAMPLE":
                target = next(item for item in intent.targets if item.action_id == step.action_id)
                track = session.track_by_id(target.stable_id)
                sample_uri = str(step.arguments["sample_uri"])
                if track.role == "audio":
                    clip = next((item for item in track.clips if item.slot_index == int(step.arguments["clip_index"])), None)
                    observed = None if clip is None else clip.sample_uri
                    matched = _sample_reference_matches(observed, sample_uri)
                    rows.append(MutationReadback(action_id=step.action_id, parameter="clip.sample", expected=sample_uri, observed=observed, matched=matched, authoritative=True))
                else:
                    device = next((item for item in track.devices if item.stable_id == str(step.expected_after.get("device_stable_id", ""))), None)
                    observed = None if device is None else (device.sample_uri or device.name)
                    matched = device is not None and _sample_device_matches(device, sample_uri)
                    rows.append(MutationReadback(action_id=step.action_id, parameter="device.sample", expected=sample_uri, observed=observed, matched=matched, authoritative=True))
                if not matched:
                    return rows, (MutationFailure.READBACK_MISMATCH, "loaded sample missing or mismatched on readback")
                continue
            if step.action_type == "SET_DEVICE_PARAMETER":
                target = next(item for item in intent.targets if item.action_id == step.action_id)
                track = session.track_by_id(target.stable_id)
                param = track.devices[int(step.arguments["device_index"])].parameters[int(step.arguments["parameter_index"])]
                expected_value = float(step.expected_after["value"])
                matched = abs(float(param.value) - expected_value) <= 1e-3
                rows.append(MutationReadback(action_id=step.action_id, parameter="device.parameter", expected=expected_value, observed=param.value, matched=matched, authoritative=True))
                if not matched:
                    return rows, (MutationFailure.READBACK_MISMATCH, "device parameter mismatch on readback")
                continue
            track = session.track_by_name(planned.name)
            if track is None:
                return rows, (
                    MutationFailure.IN_DOUBT,
                    f"target missing after write: {planned.name}",
                )
            expected = float(step.expected_after["volume"])
            observed = float(track.mixer.volume)
            matched = _volume_close(expected, observed)
            rows.append(
                MutationReadback(
                    action_id=step.action_id,
                    parameter="track.mixer.volume",
                    expected=expected,
                    observed=observed,
                    matched=matched,
                    authoritative=True,
                )
            )
            if not matched:
                return rows, (
                    MutationFailure.READBACK_MISMATCH,
                    f"{planned.name} volume {observed} != {expected}",
                )
        return rows, None

    def _verify(
        self,
        intent: MutationIntent,
        guards: dict[str, dict[str, Any]],
        after: SessionState,
    ) -> tuple[MutationVerification, tuple[MutationFailure, str] | None]:
        unexpected: list[dict[str, Any]] = []
        # A single multi-element intent is allowed to create several
        # Copilot-owned tracks.  Each CREATE_TRACK must see all of the
        # intent's new tracks as expected, not only itself.
        created_target_ids = {
            target.stable_id
            for target in intent.targets
            if any(
                step.action_id == target.action_id and step.action_type == "CREATE_TRACK"
                for step in intent.executions
            )
        }
        created_arrangement_ids = {
            str(item)
            for step in intent.executions
            if step.action_type == "DUPLICATE_CLIP_TO_ARRANGEMENT"
            for item in step.expected_after.get("arrangement_clip_ids", [])
        }
        for target in intent.targets:
            step = next(item for item in intent.executions if item.action_id == target.action_id)
            if step.action_type == "CREATE_TRACK":
                before_ids = set(guards[target.name_at_plan].get("track_ids", []))
                after_ids = {item.stable_id for item in after.tracks}
                if target.stable_id not in after_ids or (after_ids - before_ids - created_target_ids):
                    unexpected.append({"action_id": target.action_id, "kind": "unexpected_track_state"})
                continue
            if step.action_type == "CREATE_PATTERN":
                target_track = after.track_by_id(target.stable_id)
                clip_index = int(step.arguments["clip_index"])
                clip = next((item for item in target_track.clips if item.slot_index == clip_index), None)
                expected_count = int(step.expected_after.get("note_count", 0))
                if clip is None or len(clip.notes) != expected_count:
                    unexpected.append({"action_id": target.action_id, "kind": "unexpected_pattern_state"})
                continue
            if step.action_type == "CREATE_MIDI_PHRASE":
                expected = guards[target.name_at_plan]["phrase_preservation_token"]
                observed = phrase_preservation_token(
                    after, excluded_slots={target.stable_id: {int(step.arguments["clip_index"])}},
                )
                if observed != expected:
                    unexpected.append({"action_id": step.action_id, "kind": "MIDI_PHRASE_OUT_OF_SCOPE_CHANGE"})
                continue
            if step.action_type == "LOAD_DEVICE":
                before_ids = set(guards[target.name_at_plan].get("device_ids", []))
                track_after = after.track_by_id(target.stable_id)
                after_ids = {item.stable_id for item in track_after.devices}
                expected_id = str(step.expected_after.get("device_stable_id", ""))
                if expected_id not in after_ids or (after_ids - before_ids - {expected_id}):
                    unexpected.append({"action_id": target.action_id, "kind": "unexpected_device_state"})
                continue
            if step.action_type == "DUPLICATE_CLIP_TO_ARRANGEMENT":
                before_ids = set(guards[target.name_at_plan].get("arrangement_ids", []))
                expected_ids = set(str(item) for item in step.expected_after.get("arrangement_clip_ids", []))
                actual_ids = {
                    str(item.get("id", ""))
                    for item in self.tools.daw.get_arrangement_clips().get("clips", [])
                }
                if (
                    not expected_ids
                    or not expected_ids.issubset(actual_ids)
                    or (actual_ids - before_ids - created_arrangement_ids)
                ):
                    unexpected.append({"action_id": target.action_id, "kind": "unexpected_arrangement_state"})
                continue
            if step.action_type == "LOAD_SAMPLE":
                base = diff_guard_state(
                    guards[target.name_at_plan],
                    snapshot_guard_state(after, target.name_at_plan),
                    expected_volume_delta_target="__none__",
                )
                unexpected.extend(
                    row for row in base["unexpected_mutations"]
                    if not _is_live_sample_routing_normalization(
                        guards[target.name_at_plan],
                        snapshot_guard_state(after, target.name_at_plan),
                        target.name_at_plan,
                        row,
                    )
                )
                track_after = after.track_by_id(target.stable_id)
                if track_after.role == "audio":
                    before_ids = set(guards[target.name_at_plan].get("clip_ids", []))
                    after_ids = {item.stable_id for item in track_after.clips}
                    expected_id = str(step.expected_after.get("clip_stable_id", ""))
                    clip = next((item for item in track_after.clips if item.stable_id == expected_id), None)
                    if expected_id not in after_ids or (after_ids - before_ids - {expected_id}) or clip is None or not _sample_reference_matches(clip.sample_uri, str(step.arguments["sample_uri"])):
                        unexpected.append({"action_id": target.action_id, "kind": "unexpected_sample_state"})
                else:
                    before_ids = set(guards[target.name_at_plan].get("device_ids", []))
                    after_ids = {item.stable_id for item in track_after.devices}
                    expected_id = str(step.expected_after.get("device_stable_id", ""))
                    device = next((item for item in track_after.devices if item.stable_id == expected_id), None)
                    if expected_id not in after_ids or (after_ids - before_ids - {expected_id}) or device is None or not _sample_device_matches(device, str(step.arguments["sample_uri"])):
                        unexpected.append({"action_id": target.action_id, "kind": "unexpected_sample_state"})
                continue
            if step.action_type == "DUPLICATE_CLIP_TO_ARRANGEMENT":
                expected_ids = set(str(item) for item in step.expected_after.get("arrangement_clip_ids", []))
                actual_ids = {
                    str(item.get("id", ""))
                    for item in self.tools.daw.get_arrangement_clips().get("clips", [])
                }
                present = bool(expected_ids & actual_ids)
                rows.append(MutationReadback(action_id=step.action_id, parameter="arrangement.clip", expected=False, observed=sorted(expected_ids & actual_ids), matched=not present, authoritative=True, detail="rollback"))
                if present:
                    return rows, (MutationFailure.ROLLBACK_FAILED, "arrangement clip still present after rollback")
                continue
            if step.action_type == "SET_DEVICE_PARAMETER":
                base = diff_guard_state(
                    guards[target.name_at_plan],
                    snapshot_guard_state(after, target.name_at_plan),
                    expected_volume_delta_target="__none__",
                )
                unexpected.extend(base["unexpected_mutations"])
                track_after = after.track_by_id(target.stable_id)
                di = int(step.arguments["device_index"])
                pi = int(step.arguments["parameter_index"])
                actual = float(track_after.devices[di].parameters[pi].value)
                if abs(actual - float(step.expected_after["value"])) > 1e-3:
                    unexpected.append({"action_id": target.action_id, "kind": "unexpected_parameter_state"})
                before_devices = set(guards[target.name_at_plan].get("device_ids", []))
                if {item.stable_id for item in track_after.devices} != before_devices:
                    unexpected.append({"action_id": target.action_id, "kind": "unexpected_device_state"})
                continue
            post = snapshot_guard_state(after, target.name_at_plan)
            diff = diff_guard_state(
                guards[target.name_at_plan],
                post,
                expected_volume_delta_target=target.name_at_plan,
                expected_volume=float(step.expected_after["volume"]),
            )
            unexpected.extend(diff["unexpected_mutations"])
        verification = MutationVerification(
            ok=not unexpected,
            kind="EXECUTION",
            unexpected_mutations=unexpected,
            detail="only expected volume changed" if not unexpected else "unexpected mutation",
        )
        if unexpected:
            return verification, (
                MutationFailure.VERIFICATION_FAILED,
                "unexpected state mutation after write",
            )
        return verification, None

    def _rollback_readback(
        self, intent: MutationIntent, session: SessionState
    ) -> tuple[list[MutationReadback], tuple[MutationFailure, str] | None]:
        rows: list[MutationReadback] = []
        for step in intent.executions:
            target = next(item for item in intent.targets if item.action_id == step.action_id)
            if step.action_type == "CREATE_TRACK":
                present = any(item.stable_id == target.stable_id for item in session.tracks)
                rows.append(
                    MutationReadback(
                        action_id=step.action_id,
                        parameter="session.track",
                        expected=False,
                        observed=present,
                        matched=not present,
                        authoritative=True,
                        detail="rollback",
                    )
                )
                if present:
                    return rows, (MutationFailure.ROLLBACK_FAILED, "created track still present after rollback")
                continue
            if step.action_type == "CREATE_PATTERN":
                # The dependent clip is checked indirectly by the CREATE_TRACK
                # rollback below.  Keeping this branch avoids treating the
                # now-deleted track as a missing user target.
                continue
            if step.action_type == "CREATE_MIDI_PHRASE":
                track = session.track_by_id(target.stable_id)
                index = int(step.arguments["clip_index"])
                matched = (
                    not any(clip.slot_index == index for clip in track.clips)
                    and track.empty_clip_slots is not None and index in track.empty_clip_slots
                )
                rows.append(MutationReadback(
                    action_id=step.action_id, parameter="clip.slot_empty",
                    expected=True, observed=matched, matched=matched,
                    authoritative=True, detail="rollback",
                ))
                if not matched:
                    return rows, (MutationFailure.ROLLBACK_FAILED, "MIDI_PHRASE_SLOT_NOT_RESTORED")
                continue
            if step.action_type == "LOAD_DEVICE":
                target_track = session.track_by_id(target.stable_id)
                device_id = str(step.expected_after.get("device_stable_id", ""))
                present = any(item.stable_id == device_id for item in target_track.devices)
                rows.append(MutationReadback(
                    action_id=step.action_id,
                    parameter="track.device",
                    expected=False,
                    observed=present,
                    matched=not present,
                    authoritative=True,
                    detail="rollback",
                ))
                if present:
                    return rows, (MutationFailure.ROLLBACK_FAILED, "loaded device still present after rollback")
                continue
            if step.action_type == "LOAD_SAMPLE":
                target_track = session.track_by_id(target.stable_id)
                if target_track.role == "audio":
                    clip = next((item for item in target_track.clips if item.slot_index == int(step.arguments["clip_index"])), None)
                    present = clip is not None and clip.stable_id == str(step.expected_after.get("clip_stable_id", ""))
                    parameter = "clip.sample"
                else:
                    present = any(item.stable_id == str(step.expected_after.get("device_stable_id", "")) for item in target_track.devices)
                    parameter = "device.sample"
                rows.append(MutationReadback(action_id=step.action_id, parameter=parameter, expected=False, observed=present, matched=not present, authoritative=True, detail="rollback"))
                if present:
                    return rows, (MutationFailure.ROLLBACK_FAILED, "loaded sample still present after rollback")
                continue
            if step.action_type == "SET_DEVICE_PARAMETER":
                target_track = session.track_by_id(target.stable_id)
                param = target_track.devices[int(step.arguments["device_index"])].parameters[int(step.arguments["parameter_index"])]
                expected = step.rollback.restore_value if step.rollback.restore_value is not None else step.expected_before["value"]
                matched = abs(float(param.value) - float(expected)) <= 1e-3
                rows.append(MutationReadback(action_id=step.action_id, parameter="device.parameter", expected=expected, observed=param.value, matched=matched, authoritative=True, detail="rollback"))
                if not matched:
                    return rows, (MutationFailure.ROLLBACK_FAILED, "device parameter rollback mismatch")
                continue
            track = session.track_by_name(target.name_at_plan)
            if track is None:
                return rows, (
                    MutationFailure.ROLLBACK_FAILED,
                    f"target missing after rollback: {target.name_at_plan}",
                )
            expected = float(step.rollback.restore_value)
            observed = float(track.mixer.volume)
            matched = _volume_close(expected, observed)
            rows.append(
                MutationReadback(
                    action_id=step.action_id,
                    parameter="track.mixer.volume",
                    expected=expected,
                    observed=observed,
                    matched=matched,
                    authoritative=True,
                    detail="rollback",
                )
            )
            if not matched:
                return rows, (
                    MutationFailure.ROLLBACK_FAILED,
                    f"rollback readback {observed} != {expected}",
                )
        return rows, None

    def _verify_restore(
        self,
        intent: MutationIntent,
        guards: dict[str, dict[str, Any]],
        restored: SessionState,
    ) -> tuple[MutationVerification, tuple[MutationFailure, str] | None]:
        unexpected: list[dict[str, Any]] = []
        for target in intent.targets:
            step = next(item for item in intent.executions if item.action_id == target.action_id)
            if step.action_type == "CREATE_MIDI_PHRASE":
                observed = phrase_preservation_token(
                    restored, excluded_slots={target.stable_id: {int(step.arguments["clip_index"])}},
                )
                if observed != guards[target.name_at_plan]["phrase_preservation_token"]:
                    unexpected.append({"action_id": step.action_id, "kind": "MIDI_PHRASE_PRESERVATION_NOT_RESTORED"})
                continue
            if step.action_type == "CREATE_TRACK":
                before_ids = set(guards[target.name_at_plan].get("track_ids", []))
                after_ids = {item.stable_id for item in restored.tracks}
                if after_ids != before_ids:
                    before_fingerprints = sorted(guards[target.name_at_plan].get("track_fingerprints", []))
                    after_fingerprints = sorted(_track_fingerprint_key(item) for item in restored.tracks)
                    if before_fingerprints == after_fingerprints:
                        continue
                    unexpected.append({"action_id": target.action_id, "kind": "track_set_not_restored"})
                continue
            if step.action_type in {"CREATE_PATTERN", "LOAD_DEVICE", "DUPLICATE_CLIP_TO_ARRANGEMENT"} and step.rollback.depends_on:
                continue
            post = snapshot_guard_state(restored, target.name_at_plan)
            diff = diff_guard_state(
                guards[target.name_at_plan],
                post,
                expected_volume_delta_target="__none__",
            )
            unexpected.extend(diff["unexpected_mutations"])
        verification = MutationVerification(
            ok=not unexpected,
            kind="ROLLBACK",
            unexpected_mutations=unexpected,
        )
        if unexpected:
            return verification, (
                MutationFailure.ROLLBACK_FAILED,
                "restored state did not match pre-write snapshot",
            )
        return verification, None

    def _rollback_applied(
        self,
        result: MutationResult,
        intent: MutationIntent,
        *,
        already_recorded: bool = False,
    ) -> str:
        isolated = result.rollback_plan.get("not_independently_reversible") or []
        remaining = result.applied_action_ids or [
            step.action_id for step in intent.executions
        ]
        if isolated and any(action_id in remaining for action_id in isolated):
            group = set(result.rollback_plan.get("apply_action_ids") or remaining)
            if set(remaining) != group:
                return "not independently reversible; compound group incomplete"
        if self.transactions._open is not None:
            txn = self.transactions._open
            # A dependent clip/device changes the newly created track's
            # fingerprint while the compound transaction is still open. On
            # partial failure the ledger has not reached its commit-time
            # target refresh. Refresh only this exclusive new-track group,
            # and only when the last recorded content state still matches
            # authoritative Live readback. Existing user tracks stay strict.
            if txn.actions and not result.unknown_action_ids:
                created_tracks = [
                    action for action in txn.actions if action.inverse_operation == "delete_track"
                ]
                created_ids = {action.target_stable_id for action in created_tracks}
                if created_tracks and all(action.target_stable_id in created_ids for action in txn.actions):
                    current = self.tools.get_session_snapshot()
                    refreshable = True
                    for created in created_tracks:
                        track = next(
                            (item for item in current.tracks if item.stable_id == created.target_stable_id), None
                        )
                        content = next(
                            (
                                action for action in reversed(txn.actions)
                                if action.target_stable_id == created.target_stable_id
                                and action.inverse_operation in {"delete_clip", "delete_device", "delete_track"}
                            ),
                            None,
                        )
                        if not (
                            track is not None
                            and content is not None
                            and current.session_incarnation_id == txn.session_incarnation_id
                            and track.name == created.target_name_at_apply
                            and fingerprints_equal(
                                fingerprint_track(track), content.target_fingerprint.model_dump()
                            )
                        ):
                            refreshable = False
                            break
                    if refreshable:
                        self.transactions._refresh_targets(txn, current)
            txn = self.transactions.abort("safe-write rollback")
            if txn.status is TransactionStatus.IN_DOUBT:
                return txn.error or "rollback in doubt"
            if txn.status is TransactionStatus.ROLLED_BACK:
                return ""
            if txn.status is TransactionStatus.ROLLBACK_CONFLICT:
                return txn.error or "rollback conflict"
            if txn.status is TransactionStatus.FAILED:
                if not txn.actions:
                    return ""
                return txn.error or "rollback failed"
            return txn.error or txn.status.value
        last = self.transactions.last_own()
        if already_recorded or last is not None:
            txn = self.transactions.rollback_last()
            if txn.status is not TransactionStatus.ROLLED_BACK:
                return txn.error or txn.status.value
            return ""
        return ""

    def _cancel_before_write(
        self,
        intent: MutationIntent,
        result: MutationResult,
        cancellation: WriteCancellation,
    ) -> MutationResult:
        self.registry.release(intent)
        result.ok = False
        result.decision = ApplyDecision.ABSTAIN
        result.failure = MutationFailure.CANCELLED
        result.phase = MutationPhase.PREPARED
        result.error = cancellation.reason
        result.journal_terminal_state = TransactionStatus.CANCELLED.value
        result.open_transaction = self.transactions._open is not None
        result.lifecycle.append(MutationFailure.CANCELLED.value)
        return result

    def _cancel_open(
        self, result: MutationResult, cancellation: WriteCancellation
    ) -> MutationResult:
        if self.transactions._open is not None:
            self.transactions.mark_cancelled(cancellation.reason)
        result.ok = False
        result.decision = ApplyDecision.ABSTAIN
        result.failure = MutationFailure.CANCELLED
        result.error = cancellation.reason
        result.journal_terminal_state = TransactionStatus.CANCELLED.value
        result.open_transaction = self.transactions._open is not None
        result.lifecycle.append(MutationFailure.CANCELLED.value)
        return result

    def _fail(
        self,
        result: MutationResult,
        failure: MutationFailure,
        phase: MutationPhase,
        error: str,
        *,
        decision: ApplyDecision | None = None,
    ) -> MutationResult:
        result.ok = False
        result.failure = failure
        result.phase = phase
        result.error = error
        result.lifecycle.append(failure.value)
        if decision is not None:
            result.decision = decision
        elif failure is MutationFailure.IN_DOUBT:
            result.decision = ApplyDecision.IN_DOUBT
        elif failure is MutationFailure.CANCELLED:
            result.decision = ApplyDecision.ABSTAIN
        elif failure is MutationFailure.SUPERSEDED:
            result.decision = ApplyDecision.ABSTAIN
        else:
            result.decision = ApplyDecision.ROLLBACK
        if self.transactions.history:
            result.journal_terminal_state = self.transactions.history[-1].status.value
        elif self.transactions._open is not None:
            result.journal_terminal_state = self.transactions._open.status.value
        result.open_transaction = self.transactions._open is not None
        if failure is MutationFailure.IN_DOUBT:
            if result.journal_terminal_state == TransactionStatus.FAILED.value:
                raise AssertionError("IN_DOUBT must never be recorded as FAILED")
        return result


def build_safe_write_executor(
    daw, *, journal_path: Path, persist_dir: Path | None = None,
    experimental_midi_track_ids: frozenset[str] = frozenset(),
) -> SafeWriteExecutor:
    journal = DurableJournal(journal_path)
    tools = AgentTools(daw, TransactionManager(daw, journal=journal))
    return SafeWriteExecutor(
        tools, journal=journal, persist_dir=persist_dir or journal_path.parent,
        experimental_midi_track_ids=experimental_midi_track_ids,
    )
