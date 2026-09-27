from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
from urllib.parse import unquote
from uuid import uuid4

from copilot.agent.journal import DurableJournal
from copilot.agent.recovery import classify_journal
from copilot.daw.adapter import DawAdapter, DawError
from copilot.daw.identities import (
    fingerprint_track,
    fingerprints_equal,
    resolve_by_fingerprint,
)
from copilot.daw.write import ReconcileResult, spec_for
from copilot.schemas.session import MidiNote, SessionState, TrackState
from copilot.schemas.transaction import (
    AgentTransaction,
    TargetFingerprint,
    TargetLocator,
    TransactionAction,
    TransactionStatus,
)

logger = logging.getLogger("copilot.transactions")


def _sample_device_matches(device, sample_uri: str) -> bool:
    observed_uri = str(getattr(device, "sample_uri", None) or "")
    if observed_uri:
        return observed_uri == str(sample_uri)
    value = unquote(str(sample_uri)).replace("\\", "/")
    value = value.split("#", 1)[-1].rsplit(":", 1)[-1].rsplit("/", 1)[-1]
    expected = Path(value).stem.casefold()
    observed = Path(str(getattr(device, "name", ""))).stem.casefold()
    return bool(expected and observed and expected == observed)


def _sample_reference_matches(observed_uri: str | None, expected_uri: str) -> bool:
    observed = str(observed_uri or "").replace("\\", "/").casefold().strip()
    expected = str(expected_uri or "").replace("\\", "/").casefold().strip()
    if not observed or not expected:
        return False
    if observed == expected or observed.endswith("/" + expected):
        return True
    observed_name = Path(unquote(observed)).name
    expected_name = Path(unquote(expected)).name
    return bool(observed_name and expected_name and observed_name == expected_name)


class RollbackConflict(DawError):
    """Target cannot be resolved unambiguously. Nothing was mutated."""


class TransactionManager:
    """Records only this agent's actions and can roll them back exactly."""

    def __init__(
        self, daw: DawAdapter, journal: DurableJournal | None = None
    ) -> None:
        self.daw = daw
        self.journal = journal
        self.history: list[AgentTransaction] = []
        self._open: AgentTransaction | None = None

    def begin(self, user_intent: str, session: SessionState) -> AgentTransaction:
        if self._open is not None:
            raise DawError("A transaction is already open")
        txn = AgentTransaction(
            transaction_id=f"txn_{uuid4().hex[:12]}",
            user_intent=user_intent,
            session_revision=session.revision,
            session_hash=session.state_hash,
            session_incarnation_id=session.session_incarnation_id,
        )
        self._open = txn
        self._journal(
            kind="begin",
            status=TransactionStatus.PLANNED,
            intent=user_intent,
            expected_revision=session.revision,
            session_hash=session.state_hash,
        )
        logger.info("begin %s intent=%s", txn.transaction_id, user_intent)
        return txn

    def plan_write(
        self,
        *,
        command_id: str,
        operation: str,
        expected_revision: int,
        before: dict[str, Any],
        expected_after: dict[str, Any],
        target_stable_id: str = "",
        target_fingerprint: dict[str, Any] | None = None,
        target_name_at_apply: str = "",
    ) -> None:
        write_class, _permission = spec_for(operation)
        self._journal(
            kind="write",
            status=TransactionStatus.PLANNED,
            command_id=command_id,
            operation=operation,
            expected_revision=expected_revision,
            before=before,
            expected_after=expected_after,
            target_stable_id=target_stable_id,
            target_fingerprint=target_fingerprint or {},
            target_name_at_apply=target_name_at_apply,
            mutation_class=write_class.value,
        )

    def mark_sent(self, command_id: str, operation: str) -> None:
        if self._open is not None:
            self._open.status = TransactionStatus.SENT
        self._journal(
            kind="write",
            status=TransactionStatus.SENT,
            command_id=command_id,
            operation=operation,
        )

    def mark_prepared(self) -> None:
        """Persist the durable-prestate boundary before any musical write."""
        if self._open is None:
            raise DawError("No open transaction")
        self._open.status = TransactionStatus.PREPARED
        self._journal(kind="prepare", status=TransactionStatus.PREPARED)

    def mark_superseded(self, reason: str = "") -> AgentTransaction:
        """Close a prepared transaction without applying its write."""
        if self._open is None:
            raise DawError("No open transaction")
        self._open.status = TransactionStatus.SUPERSEDED
        self._open.error = reason
        txn = self._open
        self.history.append(txn)
        self._open = None
        self._journal(
            kind="supersede",
            status=TransactionStatus.SUPERSEDED,
            transaction_id=txn.transaction_id,
            session_incarnation_id=txn.session_incarnation_id,
            error=reason,
        )
        return txn

    def mark_in_doubt(self, error: str, command_id: str = "") -> AgentTransaction:
        if self._open is None:
            raise DawError("No open transaction")
        self._open.status = TransactionStatus.IN_DOUBT
        self._open.error = error
        txn = self._open
        self.history.append(txn)
        self._open = None
        self._journal(
            kind="write",
            status=TransactionStatus.IN_DOUBT,
            command_id=command_id,
            transaction_id=txn.transaction_id,
            session_incarnation_id=txn.session_incarnation_id,
            error=error,
        )
        logger.error("in_doubt %s: %s", txn.transaction_id, error)
        return txn

    def record(
        self,
        *,
        target_stable_id: str,
        target_locator_at_apply: TargetLocator,
        target_fingerprint: TargetFingerprint,
        target_name_at_apply: str,
        operation: str,
        before: dict[str, Any],
        after: dict[str, Any],
        inverse_operation: str,
        inverse_params: dict[str, Any],
        asset_id: str | None = None,
        command_id: str = "",
        expected_after: dict[str, Any] | None = None,
        expected_revision: int | None = None,
    ) -> None:
        if self._open is None:
            raise DawError("No open transaction")
        write_class, _permission = spec_for(operation)
        action = TransactionAction(
            target_stable_id=target_stable_id,
            target_locator_at_apply=target_locator_at_apply,
            target_fingerprint=target_fingerprint,
            target_name_at_apply=target_name_at_apply,
            operation=operation,
            before=before,
            after=after,
            expected_after=expected_after or after,
            inverse_operation=inverse_operation,
            inverse_params=inverse_params,
            command_id=command_id,
            session_incarnation_id=self._open.session_incarnation_id,
            expected_revision=expected_revision,
            mutation_class=write_class.value,
            status=TransactionStatus.APPLIED,
        )
        self._open.actions.append(action)
        self._open.status = TransactionStatus.APPLIED
        if asset_id:
            self._open.assets_created.append(asset_id)
        self._journal(
            kind="write",
            status=TransactionStatus.APPLIED,
            command_id=command_id,
            operation=operation,
            target_stable_id=target_stable_id,
            after=after,
        )

    def commit(
        self,
        verification: dict[str, Any] | None = None,
        session: SessionState | None = None,
    ) -> AgentTransaction:
        if self._open is None:
            raise DawError("No open transaction")
        if session is not None:
            self._refresh_targets(self._open, session)
        self._open.status = TransactionStatus.VERIFIED
        self._open.verification = verification or {}
        txn = self._open
        self.history.append(txn)
        self._open = None
        self._journal(
            kind="commit",
            status=TransactionStatus.VERIFIED,
            transaction_id=txn.transaction_id,
            session_incarnation_id=txn.session_incarnation_id,
            verification=verification or {},
        )
        logger.info("commit %s actions=%s", txn.transaction_id, len(txn.actions))
        return txn

    def _refresh_targets(self, txn: AgentTransaction, session: SessionState) -> None:
        for action in txn.actions:
            track = None
            if (
                action.session_incarnation_id == session.session_incarnation_id
                and action.target_stable_id
            ):
                try:
                    track = session.track_by_id(action.target_stable_id)
                except KeyError:
                    track = None
            if track is None:
                track = resolve_by_fingerprint(
                    session,
                    action.target_fingerprint.model_dump(),
                    stable_id=action.target_stable_id,
                    session_incarnation_id=action.session_incarnation_id,
                )
            if track is None:
                continue
            action.target_stable_id = track.stable_id
            action.target_fingerprint = TargetFingerprint(**fingerprint_track(track))
            action.target_locator_at_apply = TargetLocator(
                track_index=track.index,
                clip_index=action.target_locator_at_apply.clip_index,
                device_index=action.target_locator_at_apply.device_index,
                parameter_index=action.target_locator_at_apply.parameter_index,
            )
            action.target_name_at_apply = track.name

    def fail(self, error: str) -> AgentTransaction:
        if self._open is None:
            raise DawError("No open transaction")
        self._open.status = TransactionStatus.FAILED
        self._open.error = error
        txn = self._open
        self.history.append(txn)
        self._open = None
        self._journal(
            kind="fail",
            status=TransactionStatus.FAILED,
            transaction_id=txn.transaction_id,
            session_incarnation_id=txn.session_incarnation_id,
            error=error,
        )
        logger.error("fail %s: %s", txn.transaction_id, error)
        return txn

    def abort(self, error: str) -> AgentTransaction:
        if self._open is None:
            raise DawError("No open transaction")
        if self._open.status == TransactionStatus.IN_DOUBT:
            return self.mark_in_doubt(error)
        txn = self._open
        if not txn.actions:
            return self.fail(error)
        self._journal(kind="rollback", status=TransactionStatus.SENT, error=error)
        try:
            self._rollback_actions(txn)
        except RollbackConflict as exc:
            txn.status = TransactionStatus.ROLLBACK_CONFLICT
            txn.error = f"{error}; {exc}"
            self.history.append(txn)
            self._open = None
            self._journal(
                kind="rollback",
                status=TransactionStatus.ROLLBACK_CONFLICT,
                transaction_id=txn.transaction_id,
                session_incarnation_id=txn.session_incarnation_id,
                error=txn.error,
            )
            logger.error("abort %s ROLLBACK_CONFLICT", txn.transaction_id)
            return txn
        except Exception as exc:  # noqa: BLE001
            txn.status = TransactionStatus.FAILED
            txn.error = f"{error}; rollback incomplete: {exc}"
            self.history.append(txn)
            self._open = None
            self._journal(
                kind="rollback",
                status=TransactionStatus.FAILED,
                transaction_id=txn.transaction_id,
                session_incarnation_id=txn.session_incarnation_id,
                error=txn.error,
            )
            logger.error("abort %s FAILED", txn.transaction_id)
            return txn
        txn.status = TransactionStatus.ROLLED_BACK
        txn.error = error
        self.history.append(txn)
        self._open = None
        self._journal(
            kind="rollback",
            status=TransactionStatus.ROLLED_BACK,
            transaction_id=txn.transaction_id,
            session_incarnation_id=txn.session_incarnation_id,
            error=error,
        )
        logger.error("abort %s ROLLED_BACK", txn.transaction_id)
        return txn

    def last_own(self) -> AgentTransaction | None:
        for txn in reversed(self.history):
            if txn.status in {TransactionStatus.VERIFIED, TransactionStatus.APPLIED}:
                return txn
        return None

    def rollback_last(self) -> AgentTransaction:
        txn = self.last_own()
        if txn is None:
            raise DawError("No agent transaction to undo")
        logger.info("rollback %s", txn.transaction_id)
        self._journal(
            kind="rollback",
            status=TransactionStatus.SENT,
            transaction_id=txn.transaction_id,
        )
        try:
            self._rollback_actions(txn)
        except RollbackConflict as exc:
            txn.status = TransactionStatus.ROLLBACK_CONFLICT
            txn.error = str(exc)
            self._journal(
                kind="rollback",
                status=TransactionStatus.ROLLBACK_CONFLICT,
                transaction_id=txn.transaction_id,
                error=str(exc),
            )
            logger.error("rollback %s conflict: %s", txn.transaction_id, exc)
            return txn
        txn.status = TransactionStatus.ROLLED_BACK
        self._journal(
            kind="rollback",
            status=TransactionStatus.ROLLED_BACK,
            transaction_id=txn.transaction_id,
        )
        return txn

    def resolve_track(self, action: TransactionAction, session: SessionState) -> TrackState:
        expected = action.target_fingerprint.model_dump()
        track = resolve_by_fingerprint(
            session,
            expected,
            stable_id=action.target_stable_id,
            session_incarnation_id=action.session_incarnation_id,
        )
        if track is None:
            raise RollbackConflict(
                f"Cannot resolve target for {action.operation} "
                f"stable_id={action.target_stable_id!r} "
                f"name_at_apply={action.target_name_at_apply!r} "
                f"incarnation={action.session_incarnation_id!r}"
            )
        current = fingerprint_track(track)
        if not fingerprints_equal(current, expected):
            raise RollbackConflict(
                f"Fingerprint mismatch for {action.target_stable_id}: "
                f"expected={expected} actual={current}"
            )
        return track

    def resolve_locator(
        self, action: TransactionAction, track: TrackState
    ) -> TargetLocator:
        stored = action.target_locator_at_apply
        clip_index = stored.clip_index
        if clip_index is not None:
            slots = [clip.slot_index for clip in track.clips]
            if clip_index in slots:
                pass
            elif len(slots) == 1 and action.target_fingerprint.clip_slots == slots:
                clip_index = slots[0]
            else:
                raise RollbackConflict(
                    f"Cannot resolve current clip locator for {action.target_stable_id} "
                    f"(stored={stored.clip_index}, current_slots={slots})"
                )
        device_index = stored.device_index
        if device_index is not None:
            if device_index < 0 or device_index >= len(track.devices):
                raise RollbackConflict(
                    f"Cannot resolve current device locator for {action.target_stable_id} "
                    f"(stored={device_index}, device_count={len(track.devices)})"
                )
        return TargetLocator(
            track_index=track.index,
            clip_index=clip_index,
            device_index=device_index,
            parameter_index=stored.parameter_index,
        )

    def reconcile(
        self,
        *,
        operation: str,
        before: dict[str, Any],
        expected_after: dict[str, Any],
        session: SessionState,
    ) -> ReconcileResult:
        if operation in {"create_midi_track", "create_audio_track"}:
            before_count = int(before.get("track_count", 0))
            if len(session.tracks) == before_count + 1:
                return ReconcileResult.SATISFIED
            if len(session.tracks) == before_count:
                return ReconcileResult.ABSENT
            return ReconcileResult.AMBIGUOUS
        if operation == "load_instrument_or_effect":
            expected_id = str(expected_after.get("device_stable_id", ""))
            if expected_id and any(
                device.stable_id == expected_id
                for track in session.tracks
                for device in track.devices
            ):
                return ReconcileResult.SATISFIED
            before_count = int(before.get("device_count", 0))
            after_counts = [len(track.devices) for track in session.tracks]
            if any(count == before_count + 1 for count in after_counts):
                return ReconcileResult.AMBIGUOUS
            return ReconcileResult.ABSENT
        if operation == "load_browser_item":
            expected_uri = str(expected_after.get("sample_uri", ""))
            expected_clip_id = str(expected_after.get("clip_stable_id", ""))
            expected_device_id = str(expected_after.get("device_stable_id", ""))
            if expected_clip_id:
                matches = [
                    clip for track in session.tracks for clip in track.clips
                    if clip.stable_id == expected_clip_id
                ]
                return ReconcileResult.SATISFIED if len(matches) == 1 else (
                    ReconcileResult.ABSENT if not matches else ReconcileResult.AMBIGUOUS
                )
            if expected_device_id:
                matches = [
                    device for track in session.tracks for device in track.devices
                    if device.stable_id == expected_device_id
                ]
                return ReconcileResult.SATISFIED if len(matches) == 1 else (
                    ReconcileResult.ABSENT if not matches else ReconcileResult.AMBIGUOUS
                )
            matches = [
                clip for track in session.tracks for clip in track.clips
                if _sample_reference_matches(clip.sample_uri, expected_uri)
            ] + [
                device for track in session.tracks for device in track.devices
                if _sample_device_matches(device, expected_uri)
            ]
            if len(matches) == 1:
                return ReconcileResult.SATISFIED
            if not matches:
                return ReconcileResult.ABSENT
            return ReconcileResult.AMBIGUOUS
        if operation == "duplicate_clip_to_arrangement":
            expected_ids = set(str(item) for item in expected_after.get("arrangement_clip_ids", []))
            if not expected_ids:
                return ReconcileResult.AMBIGUOUS
            actual = self.daw.get_arrangement_clips().get("clips", [])
            actual_ids = {str(item.get("id", "")) for item in actual}
            if expected_ids.issubset(actual_ids):
                return ReconcileResult.SATISFIED
            if expected_ids.isdisjoint(actual_ids):
                return ReconcileResult.ABSENT
            return ReconcileResult.AMBIGUOUS
        if operation == "load_device_preset":
            expected_digest = str(expected_after.get("param_digest", ""))
            expected_device_id = str(expected_after.get("device_stable_id", ""))
            if expected_device_id:
                matches = [
                    device
                    for track in session.tracks
                    for device in track.devices
                    if device.stable_id == expected_device_id
                ]
                if len(matches) == 1:
                    return ReconcileResult.SATISFIED
                if not matches:
                    return ReconcileResult.ABSENT
                return ReconcileResult.AMBIGUOUS
            if not expected_digest:
                return ReconcileResult.AMBIGUOUS
            def _digest(params):
                rows = []
                for item in params:
                    try:
                        rows.append((int(item.index), round(float(item.value), 6)))
                    except Exception:
                        continue
                rows.sort(key=lambda r: r[0])
                return __import__("json").dumps(rows, separators=(",", ":"))
            matches = [
                device
                for track in session.tracks
                for device in track.devices
                if _digest(device.parameters) == expected_digest
            ]
            if len(matches) == 1:
                return ReconcileResult.SATISFIED
            if not matches:
                return ReconcileResult.ABSENT
            return ReconcileResult.AMBIGUOUS
        if operation in {"set_mixer_volume", "set_device_parameter", "set_track_name"}:
            return (
                ReconcileResult.SATISFIED
                if self._value_matches(session, expected_after)
                else ReconcileResult.ABSENT
            )
        if operation == "replace_clip_notes":
            expected = expected_after.get("note_count")
            if expected is None:
                return ReconcileResult.AMBIGUOUS
            actual = sum(len(clip.notes) for track in session.tracks for clip in track.clips)
            if actual == expected:
                return ReconcileResult.SATISFIED
            return ReconcileResult.ABSENT
        if operation == "create_pattern":
            expected_clip_id = str(expected_after.get("clip_stable_id", ""))
            matches = [
                clip
                for track in session.tracks
                for clip in track.clips
                if expected_clip_id and clip.stable_id == expected_clip_id
            ]
            if len(matches) == 1:
                return ReconcileResult.SATISFIED
            if matches:
                return ReconcileResult.AMBIGUOUS
            clip_index = expected_after.get("clip_index")
            note_count = expected_after.get("note_count")
            candidates = [
                clip
                for track in session.tracks
                for clip in track.clips
                if clip_index is not None
                and clip.slot_index == int(clip_index)
                and (note_count is None or len(clip.notes) == int(note_count))
            ]
            if len(candidates) == 1:
                return ReconcileResult.SATISFIED
            return ReconcileResult.ABSENT if not candidates else ReconcileResult.AMBIGUOUS
        return ReconcileResult.AMBIGUOUS

    def _value_matches(self, session: SessionState, expected_after: dict[str, Any]) -> bool:
        if "volume" in expected_after:
            volumes = [track.mixer.volume for track in session.tracks]
            return any(abs(volume - float(expected_after["volume"])) < 1e-6 for volume in volumes)
        if "name" in expected_after:
            return any(track.name == expected_after["name"] for track in session.tracks)
        if "value" in expected_after:
            values = [
                parameter.value
                for track in session.tracks
                for device in track.devices
                for parameter in device.parameters
            ]
            return any(abs(value - float(expected_after["value"])) < 1e-6 for values in [values] for value in values)
        return False

    def recovery_scan(self) -> list[dict[str, Any]]:
        if self.journal is None:
            return []
        return classify_journal(self.journal.read_all())

    def _rollback_actions(self, txn: AgentTransaction) -> None:
        session = self.daw.snapshot()
        resolved: list[tuple[TransactionAction, TargetLocator]] = []
        for action in reversed(txn.actions):
            track = self.resolve_track(action, session)
            resolved.append((action, self.resolve_locator(action, track)))
        for action, locator in resolved:
            self._apply_inverse(action, locator)

    def _apply_inverse(self, action: TransactionAction, locator: TargetLocator) -> None:
        op = action.inverse_operation
        params = action.inverse_params
        if op == "delete_track":
            self.daw.delete_track(locator.track_index)
            return
        if op == "set_track_name":
            self.daw.set_track_name(locator.track_index, str(params["name"]))
            return
        if op == "delete_clip":
            if locator.clip_index is None:
                raise RollbackConflict("Clip inverse missing current clip locator")
            self.daw.delete_clip(locator.track_index, locator.clip_index)
            return
        if op == "replace_clip_notes":
            if locator.clip_index is None:
                raise RollbackConflict("Note inverse missing current clip locator")
            notes = [MidiNote(**note) for note in params.get("notes", [])]
            self.daw.replace_clip_notes(locator.track_index, locator.clip_index, notes)
            return
        if op == "set_mixer_volume":
            self.daw.set_mixer_volume(locator.track_index, float(params["volume"]))
            return
        if op == "set_track_mute":
            self.daw.set_track_mute(locator.track_index, bool(params["mute"]))
            return
        if op == "set_track_output_routing":
            self.daw.set_track_output_routing(
                locator.track_index,
                str(params["output_type"]),
                str(params.get("output_channel", "")),
            )
            return
        if op == "set_device_input_routing":
            if locator.device_index is None:
                raise RollbackConflict("Device input routing inverse missing current device locator")
            self.daw.set_device_input_routing(
                locator.track_index,
                locator.device_index,
                str(params["input_type"]),
                str(params.get("input_channel", "")),
            )
            return
        if op == "set_device_parameter":
            if locator.device_index is None or locator.parameter_index is None:
                raise RollbackConflict("Device inverse missing current locator")
            self.daw.set_device_parameter(
                locator.track_index,
                locator.device_index,
                locator.parameter_index,
                float(params["value"]),
            )
            return
        if op == "restore_device_parameters":
            if locator.device_index is None:
                raise RollbackConflict("Device parameter-restore inverse missing current device locator")
            for item in params.get("items", []):
                self.daw.set_device_parameter(
                    locator.track_index,
                    locator.device_index,
                    int(item["parameter_index"]),
                    float(item["value"]),
                )
            return
        if op == "delete_device":
            if locator.device_index is None:
                raise RollbackConflict("Device inverse missing current device locator")
            self.daw.delete_device(locator.track_index, locator.device_index)
            return
        if op == "load_browser_item":
            self.daw.load_browser_item(
                locator.track_index, str(params["item_uri"]), clip_index=locator.clip_index
            )
            return
        if op == "delete_arrangement_clips":
            self.daw.delete_arrangement_clips([str(item) for item in params.get("arrangement_clip_ids", [])])
            return
        raise DawError(f"Unsupported inverse: {op}")

    def _journal(self, **record: Any) -> None:
        if self.journal is None:
            return
        if self._open is not None:
            record.setdefault("transaction_id", self._open.transaction_id)
            record.setdefault("session_incarnation_id", self._open.session_incarnation_id)
        self.journal.append(record)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        identities = []
        if hasattr(self.daw, "ids"):
            identities = self.daw.ids.dump()
        payload = {
            "transactions": [txn.model_dump() for txn in self.history],
            "identities": identities,
        }
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(__import__("json").dumps(payload, indent=2), encoding="utf-8")
        tmp.replace(path)

    def load(self, path: Path) -> None:
        if not path.exists():
            return
        raw = __import__("json").loads(path.read_text(encoding="utf-8"))
        if isinstance(raw, list):
            self.history = [AgentTransaction.model_validate(item) for item in raw]
            return
        self.history = [
            AgentTransaction.model_validate(item) for item in raw.get("transactions", [])
        ]
        if hasattr(self.daw, "ids"):
            self.daw.ids.load(raw.get("identities", []))
