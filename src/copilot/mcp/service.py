"""MCP application boundary; the existing Core remains the only write authority."""

from __future__ import annotations

import hashlib
import json
import math
import os
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse
from uuid import uuid4

from copilot.daw.ableton_tcp import AbletonTcpAdapter
from copilot.daw.adapter import DawError
from copilot.daw.object_ref import ref_from_track
from copilot.daw.state_tokens import attach_tokens, target_token
from copilot.importing.working_copy_manager_v1 import is_copilot_working_copy
from copilot.mcp.contracts import DeviceSelection, ServerConfig, SessionExpectation
from copilot.musicplan import build_device_tweak_action
from copilot.producer.context import ProducerContext
from copilot.producer.preset_catalog import PresetCatalog, PresetSelectionRequest
from copilot.producer.soniq_surface import (
    detect_surface_completeness,
    read_soniq_ws_surface,
)
from copilot.runtime.production_compiler import ProductionCompiler
from copilot.runtime.safe_write import build_safe_write_executor
from copilot.schemas.musicplan import (
    DeviceLoadActionParams,
    DiagnosisBinding,
    MusicPlan,
    PlanIntentClass,
    ProductionActionKind,
    SampleLoadActionParams,
)
from copilot.schemas.session import DeviceState, SessionState, TrackState


class MusicPlatformService:
    def __init__(self, config: ServerConfig) -> None:
        self.config = config
        self._lock = threading.RLock()
        self._daw: AbletonTcpAdapter | None = None
        if config.allow_writes:
            if config.working_copy is None:
                raise ValueError("MCP_WORKING_COPY_REQUIRED_FOR_WRITES")
            if not is_copilot_working_copy(config.working_copy):
                raise ValueError("MCP_PROTECTED_WORKING_COPY_REQUIRED")

    def status(self) -> dict[str, Any]:
        return {
            "status": "CONFIGURED_NOT_LIVE_VERIFIED",
            "transport": "stdio",
            "surfaces": ["AbletonMCP", "Serum 2 / Soniq"],
            "writes_enabled": self.config.allow_writes,
            "soniq_ws_configured": bool(os.environ.get("SONIQ_WS_URL", "").strip()),
            "producer_context_configured": self.config.producer_context is not None,
            "preset_catalog_configured": self.config.preset_catalog is not None,
            "write_authority": "ProductionCompiler -> SafeWrite -> DawAdapter",
            "experimental_midi_phrases": False,
            "soniq_ws_writes": False,
            "captures": False,
            "renders": False,
            "musical_writes": 0,
        }

    @contextmanager
    def _connection(self) -> Iterator[AbletonTcpAdapter]:
        # A single lock covers reads too: one socket cannot multiplex TCP frames.
        with self._lock:
            try:
                if self._daw is None:
                    self._daw = AbletonTcpAdapter(host="127.0.0.1", port=9877)
                    self._daw.connect()
                daw = self._daw
                if daw.handshake_info.get("mode") == "LEGACY":
                    raise DawError("MCP_AUTHORITATIVE_HANDSHAKE_REQUIRED")
                if "session.read" not in daw.capabilities:
                    raise DawError("MCP_BRIDGE_CAPABILITY_SESSION_READ_REQUIRED")
                # Refresh the adapter's project readback for every tool call.
                daw.last_project = daw.get_session_path()
                yield daw
            except (DawError, OSError):
                self.close()
                raise

    def close(self) -> None:
        with self._lock:
            if self._daw is not None:
                self._daw.disconnect()
                self._daw = None

    def _snapshot(self, daw: AbletonTcpAdapter) -> SessionState:
        session = attach_tokens(daw.snapshot())
        if not session.connected or not session.session_incarnation_id:
            raise DawError("MCP_AUTHORITATIVE_SESSION_REQUIRED")
        if self.config.working_copy is not None:
            observed = Path(session.project_path).resolve() if session.project_path else None
            if observed != self.config.working_copy.resolve():
                raise DawError("PROJECT_MISMATCH")
        return session

    def get_session(self, *, include_notes: bool = False) -> dict[str, Any]:
        with self._connection() as daw:
            session = self._snapshot(daw)
            view = session.model_dump(mode="json")
            for track in view["tracks"]:
                for clip in track["clips"]:
                    clip["note_count"] = len(clip["notes"])
                    if not include_notes:
                        clip.pop("notes")
                for device in track["devices"]:
                    device["parameter_count"] = len(device.pop("parameters"))
            return {
                "session": view,
                "notes_included": include_notes,
                "parameter_values": "USE_SONIQ_READ_SURFACE",
                "expectation": SessionExpectation.from_session(session).model_dump(),
                "targets": {
                    track.stable_id: {
                        "ref": ref_from_track(
                            track, project_identity=session.project_identity,
                        ).model_dump(mode="json"),
                        "target_token": target_token(track),
                    }
                    for track in session.tracks
                },
                "capabilities": sorted(daw.capabilities),
                "musical_writes": 0,
            }

    def get_arrangement(self) -> dict[str, Any]:
        with self._connection() as daw:
            session = self._snapshot(daw)
            return {
                "expectation": SessionExpectation.from_session(session).model_dump(),
                "arrangement": daw.get_arrangement_clips(),
                "musical_writes": 0,
            }

    def search_browser(self, query: str, category: str = "all") -> dict[str, Any]:
        if not query.strip() or len(query) > 240 or len(category) > 80:
            raise ValueError("MCP_BROWSER_QUERY_INVALID")
        with self._connection() as daw:
            self._snapshot(daw)
            return {"items": daw.search_browser(query, category), "musical_writes": 0}

    @staticmethod
    def _device(
        session: SessionState, selection: DeviceSelection,
    ) -> tuple[TrackState, DeviceState]:
        tracks = [t for t in session.tracks if t.stable_id == selection.track_stable_id]
        if len(tracks) != 1:
            raise DawError("TARGET_NOT_FOUND" if not tracks else "TARGET_AMBIGUOUS")
        devices = [
            d for d in tracks[0].devices if d.stable_id == selection.device_stable_id
        ]
        if len(devices) != 1:
            raise DawError("DEVICE_NOT_FOUND" if not devices else "TARGET_AMBIGUOUS")
        return tracks[0], devices[0]

    def read_surface(
        self, selection: DeviceSelection, *, offset: int = 0, limit: int = 100,
    ) -> dict[str, Any]:
        if offset < 0 or not 1 <= limit <= 256:
            raise ValueError("MCP_PARAMETER_PAGE_INVALID")
        with self._connection() as daw:
            session = self._snapshot(daw)
            track, device = self._device(session, selection)
            if session.track_by_name(track.name) is None or sum(
                d.name == device.name for d in track.devices
            ) != 1:
                raise DawError("TARGET_AMBIGUOUS")
            surface = detect_surface_completeness(
                daw, session=session, track_name=track.name, device_name=device.name,
            )
            schema = surface.pop("schema")
            rows = schema["parameters"]
            return {
                "selection": selection.model_dump(),
                "expectation": SessionExpectation.from_session(session).model_dump(),
                "surface": surface,
                "parameters": rows[offset:offset + limit],
                "total": len(rows),
                "next_offset": offset + limit if offset + limit < len(rows) else None,
                "authority": "ABLETON_READBACK",
                "musical_writes": 0,
            }

    def read_full_surface(
        self, selection: DeviceSelection, *, offset: int = 0, limit: int = 100,
    ) -> dict[str, Any]:
        if offset < 0 or not 1 <= limit <= 256:
            raise ValueError("MCP_PARAMETER_PAGE_INVALID")
        with self._connection() as daw:
            session = self._snapshot(daw)
            track, device = self._device(session, selection)
            if session.track_by_name(track.name) is None or sum(
                d.name == device.name for d in track.devices
            ) != 1:
                raise DawError("TARGET_AMBIGUOUS")
            surface = read_soniq_ws_surface(
                track_name=track.name, device_name=device.name,
            )
            rows = surface.pop("parameters")
            return {
                "selection": selection.model_dump(),
                "surface": surface,
                "parameters": rows[offset:offset + limit],
                "total": len(rows),
                "next_offset": offset + limit if offset + limit < len(rows) else None,
                "authority": "EXTERNAL_SONIQ_READ_ONLY",
                "limitation": "Soniq has no verified project/device identity binding; "
                              "these values cannot authorize a Live write.",
                "musical_writes": 0,
            }

    def producer_context(self) -> dict[str, Any]:
        if self.config.producer_context is None:
            raise ValueError("MCP_PRODUCER_CONTEXT_NOT_CONFIGURED")
        context = ProducerContext.model_validate_json(
            self.config.producer_context.read_text(encoding="utf-8"),
        )
        return context.prompt_payload()

    def select_presets(self, request: PresetSelectionRequest) -> dict[str, Any]:
        if self.config.preset_catalog is None:
            raise ValueError("MCP_PRESET_CATALOG_NOT_CONFIGURED")
        catalog = PresetCatalog.model_validate_json(
            self.config.preset_catalog.read_text(encoding="utf-8"),
        )
        return catalog.select(request).model_dump(mode="json")

    def _require_write_session(
        self, session: SessionState, expected: SessionExpectation,
    ) -> None:
        if not self.config.allow_writes:
            raise ValueError("MCP_WRITES_DISABLED")
        expected.require_current(session)
        if (
            not session.project_path
            or not session.project_path.lower().endswith(".als")
            or not is_copilot_working_copy(session.project_path)
        ):
            raise ValueError("MCP_PROTECTED_WORKING_COPY_REQUIRED")

    def _authorize_material(self, plan: MusicPlan) -> None:
        for action in plan.actions:
            if action.action_type is ProductionActionKind.SAMPLE_LOAD:
                if not isinstance(action.params, SampleLoadActionParams):
                    raise ValueError("MCP_ACTION_PARAMS_MISMATCH")
                action.params.sample_uri = str(self._authorize_sample(action.params.sample_uri))
            elif action.action_type in {
                ProductionActionKind.LOAD_DEVICE, ProductionActionKind.DEVICE_LOAD,
            }:
                if not isinstance(action.params, DeviceLoadActionParams):
                    raise ValueError("MCP_ACTION_PARAMS_MISMATCH")
                uri = action.params.device_uri
                name = unquote(action.params.device_name).strip()
                forbidden = (".als", ".wav", ".aif", ".aiff", ".flac", ".mp3", ".ogg")
                if not name or any(part in name for part in ("\\", "/", ":")):
                    raise ValueError("MCP_DEVICE_NAME_MUST_NOT_BE_A_PATH_OR_URI")
                if name.casefold().endswith(forbidden):
                    raise ValueError("MCP_DEVICE_NAME_IS_NOT_AN_INSTRUMENT_OR_EFFECT")
                if not uri:
                    continue
                if unquote(uri).casefold().endswith(forbidden):
                    raise ValueError("MCP_DEVICE_URI_IS_NOT_A_PRESET")
                if self.config.preset_catalog is None:
                    raise ValueError("MCP_DEVICE_URI_REQUIRES_LOCAL_PRESET_CATALOG")
                catalog = PresetCatalog.model_validate_json(
                    self.config.preset_catalog.read_text(encoding="utf-8"),
                )
                if not any(p.uri == uri for p in catalog.presets):
                    raise ValueError("MCP_DEVICE_URI_NOT_IN_LOCAL_PRESET_CATALOG")

    def _authorize_sample(self, uri: str) -> Path:
        if self.config.producer_context is None:
            raise ValueError("MCP_SAMPLE_PERMISSION_CONTEXT_REQUIRED")
        context = ProducerContext.model_validate_json(
            self.config.producer_context.read_text(encoding="utf-8"),
        )
        root_value = context.brief.authorized_library_root or self.config.sample_root
        if not root_value:
            raise ValueError("MCP_SAMPLE_AUTHORIZED_ROOT_REQUIRED")
        root = Path(root_value).resolve(strict=True)
        parsed = urlparse(uri)
        if parsed.scheme not in {"", "file"} and not (
            len(parsed.scheme) == 1 and len(uri) >= 3 and uri[1:3] in {":\\", ":/"}
        ):
            raise ValueError("MCP_SAMPLE_LOCAL_FILE_REQUIRED")
        if parsed.scheme == "file":
            if parsed.netloc not in {"", "localhost"}:
                raise ValueError("MCP_SAMPLE_LOCAL_FILE_REQUIRED")
            value = unquote(parsed.path)
            if len(value) >= 3 and value[0] == "/" and value[2] == ":":
                value = value[1:]
        else:
            value = uri
        candidate_path = Path(value)
        source = (
            candidate_path if candidate_path.is_absolute() else root / candidate_path
        ).resolve(strict=True)
        if not source.is_file() or not source.is_relative_to(root):
            raise ValueError("MCP_SAMPLE_OUTSIDE_AUTHORIZED_ROOT")
        if source.suffix.casefold() not in {".wav", ".aif", ".aiff", ".flac", ".mp3", ".ogg"}:
            raise ValueError("MCP_SAMPLE_AUDIO_FILE_REQUIRED")
        candidates = [
            row for row in context.samples.candidates
            if isinstance(row.get("relative_path"), str)
            and (root / row["relative_path"]).resolve() == source
        ]
        if len(candidates) != 1:
            raise ValueError("MCP_SAMPLE_NOT_IN_AUTHORIZED_SHORTLIST")
        with source.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        if digest != candidates[0]["sha256"]:
            raise ValueError("MCP_SAMPLE_DIGEST_MISMATCH")
        return source

    def _execute(
        self, daw: AbletonTcpAdapter, session: SessionState, plan: MusicPlan,
    ) -> dict[str, Any]:
        # Model-supplied IDs must never become journal/prestate filenames.
        safe_plan = plan.model_copy(deep=True, update={"plan_id": f"mcp_{uuid4().hex}"})
        self._authorize_material(safe_plan)
        compiled = ProductionCompiler().compile(safe_plan, session=session)
        if compiled.status != "COMPILED" or compiled.intent is None:
            return {
                "status": "EXECUTION_DEFERRED",
                "ok": False,
                "reasons": list(compiled.reasons) or [compiled.status],
                "musical_writes": 0,
            }
        executor = build_safe_write_executor(
            daw, journal_path=self.config.state_dir / "safe_write.jsonl",
            persist_dir=self.config.state_dir,
        )
        unresolved = [
            row for row in executor.recover() if row["recovery"] != "VERIFIED"
        ]
        if unresolved:
            return {
                "status": "RECOVERY_REQUIRED", "ok": False,
                "recovery": unresolved, "musical_writes": 0,
            }
        result = executor.run(compiled.intent)
        return {
            "status": "VERIFIED" if result.ok else result.decision.value,
            "ok": result.ok,
            "result": result.to_dict(),
            "musical_verification": "NOT_PERFORMED",
            "capture_performed": False,
            "render_performed": False,
        }

    def apply_plan(
        self, plan: MusicPlan, expected: SessionExpectation,
    ) -> dict[str, Any]:
        if not self.config.allow_writes:
            raise ValueError("MCP_WRITES_DISABLED")
        if len(plan.actions) > 64:
            raise ValueError("MCP_ACTION_BUDGET_EXCEEDED")
        json.dumps(plan.model_dump(mode="json"), allow_nan=False)
        with self._connection() as daw:
            session = self._snapshot(daw)
            self._require_write_session(session, expected)
            if (
                plan.project_state_token != expected.project_token
                or plan.audible_state_token != expected.audible_token
            ):
                raise ValueError("MCP_PLAN_EXPECTATION_MISMATCH")
            return self._execute(daw, session, plan)

    def set_parameter(
        self, selection: DeviceSelection, expected: SessionExpectation, *,
        parameter_index: int, parameter_name: str, expected_before: float,
        value: float, reason: str,
    ) -> dict[str, Any]:
        if not self.config.allow_writes:
            raise ValueError("MCP_WRITES_DISABLED")
        if not reason.strip() or not all(map(math.isfinite, (expected_before, value))):
            raise ValueError("MCP_PARAMETER_REQUEST_INVALID")
        with self._connection() as daw:
            session = self._snapshot(daw)
            self._require_write_session(session, expected)
            track, device = self._device(session, selection)
            rows = [p for p in device.parameters if p.index == parameter_index]
            if len(rows) != 1:
                raise ValueError("MCP_PARAMETER_NOT_UNIQUELY_EXPOSED_IN_ABLETON")
            parameter = rows[0]
            if parameter.name.casefold() != parameter_name.casefold():
                raise ValueError("MCP_PARAMETER_NAME_INDEX_MISMATCH")
            if sum(p.name.casefold() == parameter.name.casefold() for p in device.parameters) != 1:
                raise ValueError("MCP_PARAMETER_NAME_AMBIGUOUS")
            if (
                parameter.max is None
                or not all(map(math.isfinite, (parameter.min, parameter.max)))
                or not parameter.min <= value <= parameter.max
            ):
                raise ValueError("MCP_PARAMETER_RANGE_UNKNOWN_OR_INVALID")
            if abs(parameter.value - expected_before) > 1e-6:
                raise ValueError("MCP_PARAMETER_EXPECTED_VALUE_MISMATCH")
            action = build_device_tweak_action(
                track=track, project_identity=session.project_identity,
                device_index=device.index, parameter_name=parameter.name,
                expected_before=parameter.value, intended_after=value,
                allowed_min=parameter.min, allowed_max=parameter.max,
                reason=reason, evidence_refs=[],
                session_incarnation_id=session.session_incarnation_id,
            )
            plan = MusicPlan(
                plan_id=f"mcp_{uuid4().hex}",
                intent_class=PlanIntentClass.CONTROLLED_ENGINEERING_VALIDATION,
                diagnosis=DiagnosisBinding(
                    diagnosis_id="mcp-explicit-parameter-request",
                    diagnosis_status="NOT_APPLICABLE",
                    diagnosis_accepted=False,
                ),
                project_state_token=session.project_token,
                audible_state_token=session.audible_token,
                target_state_tokens={track.name: target_token(track)},
                actions=[action],
                created_at=datetime.now(timezone.utc).isoformat(),
            )
            return self._execute(daw, session, plan)
