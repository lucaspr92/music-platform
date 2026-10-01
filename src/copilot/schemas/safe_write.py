"""SAFE_WRITE_FOUNDATION_V2 typed contracts.

Generic verified musical-mutation architecture. Does not certify new actions.
ABLETON_MUTATION_PROTOCOL_V1 MutationBatch is not this contract.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from copilot.schemas.transaction import TargetFingerprint, TargetLocator

MILESTONE = "SAFE_WRITE_FOUNDATION_V2"
SCHEMA_VERSION = "safe-write-v2"

CERTIFIED_PRODUCTION_ACTION = "SET_TRACK_VOLUME"
CERTIFIED_PRODUCTION_ACTIONS = frozenset({CERTIFIED_PRODUCTION_ACTION})

KIND_PRODUCTION_MUSICAL = "PRODUCTION_MUSICAL"
KIND_PRODUCER_EXECUTION_V1 = "PRODUCER_EXECUTION_V1"
KIND_EXPERIMENTAL_MIDI_PHRASE_V1 = "EXPERIMENTAL_MIDI_PHRASE_V1"
PRODUCER_CERTIFIED_ACTIONS = frozenset({
    "CREATE_TRACK", "CREATE_PATTERN", "LOAD_SAMPLE", "DUPLICATE_CLIP_TO_ARRANGEMENT", "LOAD_DEVICE", "SET_DEVICE_PARAMETER", "SET_TRACK_VOLUME"
})


class MutationPhase(StrEnum):
    PLAN = "PLAN"
    RECONCILE_PROJECT = "RECONCILE_PROJECT"
    RECONCILE_TARGETS = "RECONCILE_TARGETS"
    VALIDATE_PRECONDITIONS = "VALIDATE_PRECONDITIONS"
    PERSIST_ROLLBACK = "PERSIST_ROLLBACK"
    PREPARED = "PREPARED"
    EXECUTE = "EXECUTE"
    READBACK = "READBACK"
    RECONCILE = "RECONCILE"
    VERIFY = "VERIFY"
    KEEP = "KEEP"
    ROLLBACK = "ROLLBACK"
    ROLLBACK_READBACK = "ROLLBACK_READBACK"
    ROLLBACK_VERIFY = "ROLLBACK_VERIFY"


class MutationFailure(StrEnum):
    TARGET_NOT_FOUND = "TARGET_NOT_FOUND"
    TARGET_AMBIGUOUS = "TARGET_AMBIGUOUS"
    PROJECT_MISMATCH = "PROJECT_MISMATCH"
    STALE_PLAN = "STALE_PLAN"
    PRECONDITION_FAILED = "PRECONDITION_FAILED"
    EXECUTION_FAILED = "EXECUTION_FAILED"
    PARTIAL_FAILURE = "PARTIAL_FAILURE"
    IN_DOUBT = "IN_DOUBT"
    READBACK_MISMATCH = "READBACK_MISMATCH"
    VERIFICATION_FAILED = "VERIFICATION_FAILED"
    ROLLBACK_FAILED = "ROLLBACK_FAILED"
    CANCELLED = "CANCELLED"
    SUPERSEDED = "SUPERSEDED"


class ApplyDecision(StrEnum):
    KEEP = "KEEP"
    ROLLBACK = "ROLLBACK"
    ABSTAIN = "ABSTAIN"
    IN_DOUBT = "IN_DOUBT"


class RollbackReversibility(StrEnum):
    INDEPENDENT = "INDEPENDENT"
    DEPENDENT = "DEPENDENT"
    NOT_INDEPENDENTLY_REVERSIBLE = "NOT_INDEPENDENTLY_REVERSIBLE"


class MutationTarget(BaseModel):
    """Durable target. Track index is a locator, never identity."""

    action_id: str
    ref: dict[str, Any] = Field(default_factory=dict)
    stable_id: str = ""
    name_at_plan: str = ""
    fingerprint: TargetFingerprint = Field(default_factory=TargetFingerprint)
    locator: TargetLocator | None = None
    session_incarnation_id: str = ""


class MutationPrecondition(BaseModel):
    code: str
    required: bool = True
    expected: Any = None
    observed: Any = None
    satisfied: bool | None = None
    detail: str = ""


class MutationRollback(BaseModel):
    """Durable inverse. Core owns rollback; Remote Script does not."""

    inverse_operation: str
    inverse_params: dict[str, Any] = Field(default_factory=dict)
    reversibility: RollbackReversibility = RollbackReversibility.INDEPENDENT
    depends_on: list[str] = Field(default_factory=list)
    parameter: str = ""
    restore_value: Any = None
    source: str = "authoritative_prewrite_readback"
    prepared: bool = False


class MutationExecution(BaseModel):
    action_id: str
    action_type: str
    operation: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    expected_before: dict[str, Any] = Field(default_factory=dict)
    expected_after: dict[str, Any] = Field(default_factory=dict)
    certified: bool = False
    rollback: MutationRollback = Field(default_factory=MutationRollback)


class MutationReadback(BaseModel):
    action_id: str
    parameter: str
    expected: Any = None
    observed: Any = None
    matched: bool = False
    authoritative: bool = True
    detail: str = ""


class MutationVerification(BaseModel):
    ok: bool = False
    kind: str = "EXECUTION"
    readbacks: list[MutationReadback] = Field(default_factory=list)
    unexpected_mutations: list[dict[str, Any]] = Field(default_factory=list)
    detail: str = ""


class MutationIntent(BaseModel):
    plan_id: str
    kind: str = KIND_PRODUCTION_MUSICAL
    user_intent: str = ""
    project_identity: str = ""
    expected_revision: int | None = None
    expected_session_hash: str = ""
    expected_project_token: str = ""
    expected_audible_token: str = ""
    expected_incarnation_id: str = ""
    targets: list[MutationTarget] = Field(default_factory=list)
    preconditions: list[MutationPrecondition] = Field(default_factory=list)
    executions: list[MutationExecution] = Field(default_factory=list)
    decision_after_verify: ApplyDecision = ApplyDecision.KEEP
    supersedes_plan_id: str = ""


class MutationResult(BaseModel):
    milestone: str = MILESTONE
    plan_id: str
    ok: bool = False
    decision: ApplyDecision = ApplyDecision.ABSTAIN
    failure: MutationFailure | None = None
    phase: MutationPhase = MutationPhase.PLAN
    lifecycle: list[str] = Field(default_factory=list)
    intent: MutationIntent | None = None
    rollback_plan: dict[str, Any] = Field(default_factory=dict)
    prestate_path: str = ""
    journal_path: str = ""
    transaction_id: str = ""
    journal_terminal_state: str = ""
    open_transaction: bool = False
    readbacks: list[MutationReadback] = Field(default_factory=list)
    verification: MutationVerification | None = None
    rollback_verification: MutationVerification | None = None
    applied_action_ids: list[str] = Field(default_factory=list)
    unknown_action_ids: list[str] = Field(default_factory=list)
    not_attempted_action_ids: list[str] = Field(default_factory=list)
    musical_writes: int = 0
    error: str = ""
    recovery: dict[str, Any] = Field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")
