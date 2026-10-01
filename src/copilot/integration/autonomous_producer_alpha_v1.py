"""Bounded real-Live Alpha orchestration.

This module is intentionally a thin Core coordinator.  Lucas remains the
source of musical decisions; Core owns context assembly, identity, compilation,
SafeWrite, readback, capture, and artifact persistence.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from copilot.audio.advanced_perception_v1 import run_advanced_perception
from copilot.audio.music_analyzer import analyze_reference_music
from copilot.daw.ableton_tcp import AbletonTcpAdapter
from copilot.daw.state_tokens import attach_tokens
from copilot.human_eval.store import atomic_write, now_iso
from copilot.integration.lucas_core_v1 import (
    _single_action_plan,
    build_lucas_input,
    build_project_context,
    build_reference_context,
    normalize_sample_uri_for_working_copy,
    rebind_sample_load_action,
    run_lucas_critique_with_provider_failover,
    run_lucas_planner,
)
from copilot.integration.mixing_mastering_v1 import capture_and_analyze_master
from copilot.importing.working_copy_manager_v1 import is_copilot_working_copy
from copilot.musicplan import build_duplicate_clip_to_arrangement_action
from copilot.producer.goal import ProducerGoal
from copilot.producer.context import ProducerContext
from copilot.producer.arrangement_score import ArrangementScore, PhraseSlotBinding
from copilot.musicplan.score_compiler import (
    ScoreBinding, ScoreCompilation, compile_score_placements, score_binding_blocker,
)
from copilot.musicplan.score_material import (
    PhraseMaterialCompilation, bind_verified_phrase_slots, phrase_material_blocker,
    prepare_score_phrase_material,
)
from copilot.producer.state import ProducerPhase, ProducerState, ProducerStateStore
from copilot.sample_library.library_v1 import (
    build_sample_set_context,
    index_library,
    load_index,
)
from copilot.sample_library.schemas import LibraryIndex, SampleRole
from copilot.schemas.lucas_integration import StyleContext, UserIntent
from copilot.schemas.musicplan import MusicPlan, ProductionActionKind
from copilot.reasoning.provider import configured_http_provider

MILESTONE = "AUTONOMOUS_PRODUCER_ALPHA_V1"
REAL_LUCAS = "REAL_LUCAS"
CONTROLLED_FIXTURE = "CONTROLLED_FIXTURE"
SUPPORTED_ALPHA_ACTIONS = frozenset(
    {
        ProductionActionKind.CREATE_TRACK,
        ProductionActionKind.SAMPLE_LOAD,
        ProductionActionKind.LOAD_SAMPLE,
        ProductionActionKind.LOAD_DEVICE,
        ProductionActionKind.DEVICE_LOAD,
        ProductionActionKind.DEVICE_TWEAK,
        ProductionActionKind.SET_TRACK_VOLUME,
        ProductionActionKind.DUPLICATE_CLIP_TO_ARRANGEMENT,
        ProductionActionKind.CREATE_PATTERN,
    }
)


class RealLucasRequired(RuntimeError):
    """Raised when the stable planner did not produce a real model plan."""


class LucasReasoningOutputAdapter:
    """Translate the configured Astra reasoning envelope to Lucas's stable JSON.

    The provider is allowed to return the repository's grounded reasoning
    envelope.  The adapter only unwraps model-authored candidate strategies;
    it never chooses samples, invents sections, or applies Core taste logic.
    """

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.identity = str(getattr(inner, "identity", type(inner).__name__))
        self.version = str(getattr(inner, "version", "unknown"))
        self.response_adapter = "lucas_reasoning_candidate_strategies_v1"
        self.last_raw = ""

    @staticmethod
    def _decode(value: Any) -> Any:
        if not isinstance(value, str):
            return value
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value

    @classmethod
    def unwrap(cls, raw: str) -> str:
        payload = json.loads(raw)
        if "selections" in payload or "arrangement" in payload:
            return raw
        strategies = payload.get("candidate_strategies") or []
        output: dict[str, Any] = {}
        for row in strategies:
            if not isinstance(row, dict):
                continue
            name = str(row.get("strategy") or "")
            value = cls._decode(row.get("reason"))
            if name in {"selections", "arrangement", "patch_contracts"}:
                output[name] = value
            elif name == "reasoning":
                output["reasoning"] = value
            elif "candidate" in name.lower():
                selections: dict[str, int] = {}
                for match in re.finditer(r"([A-Za-z][A-Za-z ]*?)\s+candidate\s+(\d+)", name, re.I):
                    track = " ".join(match.group(1).strip().split())
                    track = re.sub(r"^(select|choose|and)\s+", "", track, flags=re.I).strip()
                    if track:
                        selections[track] = int(match.group(2))
                if selections:
                    output.setdefault("selections", {}).update(selections)
            elif ":" in name and "bar" in name.lower():
                from copilot.musicplan.arrangement import ALL_TRACKS

                sections: list[dict[str, Any]] = []
                for match in re.finditer(
                    r"([^:]+):\s*(\d+)\s+bars?,\s*(?:active\s+)?([^\.]+)",
                    name,
                    re.I,
                ):
                    active = []
                    active_text = match.group(3)
                    for track_name in sorted(ALL_TRACKS, key=len, reverse=True):
                        if re.search(rf"(?<![A-Za-z]){re.escape(track_name)}(?![A-Za-z])", active_text, re.I):
                            active.append(track_name)
                    active.sort(key=lambda item: active_text.lower().find(item.lower()))
                    if active:
                        sections.append({
                            "name": re.sub(r"[^A-Z0-9]+", "_", match.group(1).upper()).strip("_") or "SECTION",
                            "bars": int(match.group(2)),
                            "active": active,
                        })
                if sections:
                    output.setdefault("arrangement", []).extend(sections)
        if not any(key in output for key in ("selections", "arrangement", "patch_contracts")):
            raise RealLucasRequired("LUCAS_PROVIDER_OUTPUT_HAS_NO_PRODUCER_STRATEGY")
        return json.dumps(output)

    def reason(self, prompt: str, *, timeout_s: float = 30.0) -> str:
        raw = self.inner.reason(prompt, timeout_s=timeout_s)
        self.last_raw = raw
        return self.unwrap(raw)


class LucasPlanningProviderAdapter:
    """Use the configured provider with Lucas's producer-output contract.

    The generic Astra provider deliberately requests the diagnosis schema. That
    schema is correct for reasoning/evidence calls but cannot carry Lucas's
    selections and arrangement. This adapter keeps the same configured model,
    endpoint, and credentials while requesting the stable Lucas JSON object.
    """

    def __init__(self, inner: Any, *, complete_track: bool = False) -> None:
        self.inner = inner
        self.identity = str(getattr(inner, "identity", type(inner).__name__))
        self.version = str(getattr(inner, "version", "unknown"))
        self.response_adapter = "lucas_producer_plan_schema_v1"
        self.last_raw = ""
        self.complete_track = complete_track

    @staticmethod
    def _schema(*, complete_track: bool = False) -> dict[str, Any]:
        schema = {
            "type": "object",
            "properties": {
                "selections": {"type": "object", "additionalProperties": {"type": "integer"}},
                "selection_reasons": {"type": "object", "additionalProperties": {"type": "string"}},
                "rejected_candidates": {
                    "type": "object",
                    "additionalProperties": {
                        "type": "object", "additionalProperties": {"type": "string"},
                    },
                },
                "patterns": {
                    "type": "object",
                    "additionalProperties": {
                        "type": "object",
                        "properties": {
                            "length_beats": {"type": "number"},
                            "notes": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "pitch": {"type": "integer"},
                                        "start_time": {"type": "number"},
                                        "duration": {"type": "number"},
                                        "velocity": {"type": "integer"},
                                    },
                                    "required": ["pitch", "start_time", "duration", "velocity"],
                                },
                            },
                        },
                        "required": ["length_beats", "notes"],
                    },
                },
                "producer_criteria": {
                    "type": "object",
                    "properties": {
                        "primary_hook": {"type": "string"},
                        "hook_role": {"type": "string"},
                        "uncertainty": {"type": "string"},
                        "sections": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "section_name": {"type": "string"},
                                    "perceptual_goal": {"type": "string"},
                                    "lead_role": {"type": "string"},
                                    "low_end_owner": {"type": "string"},
                                    "space_roles": {"type": "array", "items": {"type": "string"}},
                                    "hook_usage": {"type": "string"},
                                    "energy_rationale": {"type": "string"},
                                    "variation_hypothesis": {"type": "string"},
                                    "claim_kind": {"type": "string"},
                                    "evidence_refs": {"type": "array", "items": {"type": "string"}},
                                },
                                "required": [
                                    "section_name", "perceptual_goal", "lead_role",
                                    "low_end_owner", "space_roles", "hook_usage",
                                    "energy_rationale", "variation_hypothesis",
                                    "claim_kind", "evidence_refs",
                                ],
                                "additionalProperties": False,
                            },
                        },
                    },
                    "required": ["primary_hook", "hook_role", "uncertainty", "sections"],
                    "additionalProperties": False,
                },
                "mix_decisions": {
                    "type": "object",
                    "properties": {
                        name: {
                            "type": "object",
                            "properties": {
                                "decision": {"type": "string"},
                                "reason": {"type": "string"},
                                "hypothesis": {"type": "string"},
                                "objective": {"type": "string"},
                                "action": {
                                    "type": "object",
                                    "properties": {
                                        "track_name": {"type": "string"},
                                        "control": {"type": "string"},
                                        "band": {"type": "integer"},
                                        "value": {"type": "number"},
                                        "unit": {"type": "string"},
                                    },
                                    "required": ["track_name", "control", "value", "unit"],
                                    "additionalProperties": False,
                                },
                            },
                            "required": ["decision", "reason"],
                        } for name in ("eq_eight", "limiter")
                    },
                    "required": ["eq_eight", "limiter"],
                },
                "arrangement": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "bars": {"type": "integer"},
                            "active": {"type": "array", "items": {"type": "string"}},
                        },
                        "required": ["name", "bars", "active"],
                        "additionalProperties": False,
                    },
                },
                "reasoning": {"type": "string"},
                "track_spec": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "intent": {"type": "string"},
                        "bpm": {"type": "number"},
                        "duration_bars": {"type": "integer"},
                        "primary_hook": {"type": "string"},
                        "hook_role": {"type": "string"},
                        "style": {"type": "string"},
                        "sections": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string"},
                                    "bars": {"type": "integer"},
                                    "energy": {"type": "number"},
                                    "active_roles": {"type": "array", "items": {"type": "string"}},
                                    "variation": {"type": "string"},
                                    "transition": {"type": "string"},
                                },
                                "required": ["name", "bars", "energy", "active_roles"],
                            },
                        },
                    },
                    "required": ["bpm", "sections"],
                },
                "patch_contracts": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "track": {"type": "string"},
                            "device": {"type": "string"},
                            "writes": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "index": {"type": "integer"},
                                        "name": {"type": "string"},
                                        "value": {"type": "number"},
                                    },
                                    "required": ["index", "name", "value"],
                                    "additionalProperties": False,
                                },
                            },
                            "constraints": {
                                "type": "object",
                                "properties": {
                                    "max_delta_norm": {"type": "number"},
                                    "max_writes": {"type": "integer"},
                                    "forbid_device_on_toggle": {"type": "boolean"},
                                },
                                "required": ["max_delta_norm", "max_writes", "forbid_device_on_toggle"],
                                "additionalProperties": False,
                            },
                        },
                        "required": ["track", "device", "writes", "constraints"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": [
                "selections", "selection_reasons", "rejected_candidates",
                "arrangement", "reasoning", "patch_contracts", "producer_criteria",
                "mix_decisions",
            ],
            "additionalProperties": False,
        }
        if complete_track:
            score_schema = ArrangementScore.model_json_schema()
            schema["$defs"] = score_schema.pop("$defs", {})
            schema["properties"]["arrangement_score"] = score_schema
            schema["required"].extend(["track_spec", "arrangement_score"])
        return schema

    def reason(self, prompt: str, *, timeout_s: float = 30.0) -> str:
        model = str(getattr(self.inner, "_model", ""))
        if "astra" not in model.lower():
            raw = self.inner._reason_chat_json_object(prompt, timeout_s=timeout_s)
            self.last_raw = raw
            return raw
        payload = {
            "model": model,
            "reasoning": {"effort": os.environ.get("COPILOT_REASONING_EFFORT") or "high"},
            "input": [
                {"role": "system", "content": "Return only the Lucas producer JSON object. No Ableton operations."},
                {"role": "user", "content": prompt},
            ],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "LucasProducerPlan",
                    "schema": self._schema(complete_track=self.complete_track),
                    "strict": False,
                }
            },
        }
        raw_payload = self.inner._post("/responses", payload, timeout_s=timeout_s)
        from copilot.reasoning.provider import _extract_responses_text

        text = _extract_responses_text(raw_payload)
        if not text:
            raise RealLucasRequired("LUCAS_PRODUCER_PLAN_RESPONSE_EMPTY")
        self.last_raw = text
        return text


def discover_reference(evidence: Path, *, project_path: Path | None = None) -> Path:
    """Find an existing WAV for analysis without copying it into Ableton."""
    candidates: list[Path] = []
    for root in (evidence / "captures", Path.home() / "CopilotProjects" / "captures"):
        if root.is_dir():
            candidates.extend(
                path for path in root.glob("*.wav")
                if not path.name.lower().endswith("_raw.wav")
                and not path.name.lower().startswith("_next")
                and path.stat().st_size > 44
            )
    if project_path is not None:
        candidates.extend(
            path for path in project_path.parent.glob("*.wav")
            if not path.name.lower().endswith("_raw.wav")
            and path.stat().st_size > 44
        )
    if not candidates:
        raise FileNotFoundError("NO_REAL_REFERENCE_AUDIO_DISCOVERED")
    # Capture scratch files can be newer than the last valid take and still be
    # zero-byte/invalid WAVs.  Select the newest file that the frozen analyzer
    # can actually decode, preserving deterministic discovery without trusting
    # filename recency alone.
    import soundfile as sf

    readable: list[Path] = []
    for path in sorted(candidates, key=lambda item: item.stat().st_mtime_ns, reverse=True):
        try:
            if int(sf.info(path).frames) > 0:
                readable.append(path)
        except Exception:
            continue
    if not readable:
        raise FileNotFoundError("NO_READABLE_REAL_REFERENCE_AUDIO_DISCOVERED")
    return readable[0]


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _plan_action_name(action: Any) -> str:
    return str(getattr(action.action_type, "value", action.action_type))


def _actualize_action(action: Any, current: Any) -> tuple[Any | None, str | None]:
    name = str(action.target.ref.get("name") or "")
    track = current.track_by_name(name) if name else None
    if action.action_type is ProductionActionKind.CREATE_TRACK:
        if track is not None:
            return None, "TARGET_ALREADY_EXISTS"
        return action, None
    if track is None:
        return None, "TARGET_NOT_FOUND_OR_AMBIGUOUS"
    from copilot.daw.object_ref import ref_from_track, runtime_from_track

    target = action.target.model_copy(update={
        "ref": ref_from_track(track, project_identity=current.project_identity).model_dump(mode="json"),
        "runtime_id": runtime_from_track(track, session_incarnation_id=current.session_incarnation_id).model_dump(mode="json"),
        "track_index_locator": track.index,
    })
    actual = action.model_copy(update={"target": target})
    if action.action_type is ProductionActionKind.SAMPLE_LOAD:
        sample_uri = str(actual.params.sample_uri).replace("\\", "/")
        if track.role == "audio":
            # Live's audio-track loader resolves a project-relative path through
            # the browser path command.  The query URI form is for instruments
            # and effects loaded on MIDI tracks; using it on an audio track
            # leaves the clip absent on authoritative readback.
            if sample_uri.startswith("Samples/"):
                sample_uri = sample_uri[len("Samples/"):]
            parts = [part for part in sample_uri.strip("/").split("/") if part]
            if (
                not parts
                or sample_uri.startswith("/")
                or ":" in parts[0]
                or any(part in {".", ".."} for part in parts)
            ):
                return None, "SAMPLE_PATH_OUTSIDE_AUTHORIZED_LIBRARY"
            actual = actual.model_copy(update={
                "params": actual.params.model_copy(update={"sample_uri": sample_uri})
            })
        else:
            sample_uri = normalize_sample_uri_for_working_copy(sample_uri)
        actual = rebind_sample_load_action(
            actual.model_copy(update={
                "params": actual.params.model_copy(update={
                    "sample_uri": sample_uri,
                })
            }),
            track=track,
            session=current,
        )
    if action.action_type in {
        ProductionActionKind.LOAD_DEVICE,
        ProductionActionKind.DEVICE_LOAD,
    } and bool(getattr(track, "grouped", False)):
        return None, "TARGET_TRACK_GROUPED_NOT_VISIBLE_FOR_DEVICE_LOAD"
    return actual, None


def _execute_one(
    action: Any, *, daw: Any, persist_dir: Path,
    score_binding: ScoreBinding | None = None,
    phrase_material: PhraseMaterialCompilation | None = None,
    musical_score: ArrangementScore | None = None,
    verified_phrase_clip_ids: dict[str, str] | None = None,
    experimental_midi_track_ids: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    from copilot.daw.state_tokens import attach_tokens
    from copilot.runtime.production_compiler import ProductionCompiler
    from copilot.runtime.safe_write import build_safe_write_executor

    # SafeWrite's authoritative precondition snapshot includes the complete
    # session payload. Compile against that same view; the lightweight
    # include_notes=False view can carry a different observed revision.
    current = daw.snapshot()
    attach_tokens(current)
    if action.action_type is ProductionActionKind.CREATE_MIDI_PHRASE:
        blocker = (
            "MIDI_PHRASE_CONTEXT_REQUIRED"
            if phrase_material is None or musical_score is None else
            phrase_material_blocker(
                phrase_material, score=musical_score, session=current,
                verified_clip_ids=verified_phrase_clip_ids or {},
            )
        )
        if blocker:
            return {
                "action_id": action.action_id, "action_type": _plan_action_name(action),
                "status": "EXECUTION_DEFERRED", "reason": blocker,
            }
    if score_binding is not None:
        blocker = score_binding_blocker(score_binding, current)
        if blocker is not None:
            return {
                "action_id": action.action_id,
                "action_type": _plan_action_name(action),
                "status": "EXECUTION_DEFERRED", "reason": blocker,
            }
    planned_name = str(action.target.ref.get("name") or "")
    planned_track = current.track_by_name(planned_name) if planned_name else None
    if action.action_type is ProductionActionKind.SAMPLE_LOAD and planned_track is not None:
        if planned_track.role == "audio" and "browser.load" not in getattr(daw, "capabilities", set()):
            return {
                "action_id": action.action_id,
                "action_type": _plan_action_name(action),
                "status": "EXECUTION_DEFERRED",
                "reason": "BRIDGE_CAPABILITY_BROWSER_LOAD_UNAVAILABLE",
            }
        if planned_track.role != "audio" and planned_track.devices:
            return {
                "action_id": action.action_id,
                "action_type": _plan_action_name(action),
                "status": "EXECUTION_DEFERRED",
                "reason": "TARGET_MIDI_TRACK_ALREADY_HAS_DEVICE_READBACK_AMBIGUOUS",
            }
    actual, reason = _actualize_action(action, current)
    if actual is None:
        return {
            "action_id": action.action_id,
            "action_type": _plan_action_name(action),
            "status": "EXECUTION_DEFERRED",
            "reason": reason,
        }
    single = _single_action_plan(actual, session=current, plan_id=f"alpha_{action.action_id}")
    compiler = ProductionCompiler()
    if action.action_type is ProductionActionKind.CREATE_MIDI_PHRASE:
        compiler = ProductionCompiler(
            experimental_midi_track_ids=experimental_midi_track_ids,
            negotiated_capabilities=frozenset(getattr(daw, "capabilities", ())),
        )
    compiled = compiler.compile(single, session=current)
    if compiled.status not in {"COMPILED", "COMPILED_EXPERIMENTAL"} or compiled.intent is None:
        return {
            "action_id": action.action_id,
            "action_type": _plan_action_name(action),
            "status": "EXECUTION_DEFERRED",
            "reason": "; ".join(compiled.reasons) or compiled.status,
        }
    executor_args = {}
    if action.action_type is ProductionActionKind.CREATE_MIDI_PHRASE:
        executor_args["experimental_midi_track_ids"] = experimental_midi_track_ids
    executor = build_safe_write_executor(
        daw, journal_path=persist_dir / f"{action.action_id}_safe_write.jsonl",
        persist_dir=persist_dir, **executor_args,
    )
    result = executor.run(compiled.intent)
    row = {
        "action_id": action.action_id,
        "action_type": _plan_action_name(action),
        "status": "VERIFIED" if result.ok else "FAILED",
        "readbacks": [item.model_dump(mode="json") for item in result.readbacks],
    }
    if not result.ok:
        row["error"] = result.error or "SAFE_WRITE_FAILED"
    if action.action_type is ProductionActionKind.SAMPLE_LOAD and result.ok:
        loaded_id = compiled.intent.executions[0].expected_after.get("device_stable_id")
        if loaded_id:
            row["loaded_device_stable_id"] = loaded_id
    if action.action_type is ProductionActionKind.CREATE_MIDI_PHRASE:
        row["experimental"] = True
        row["certification"] = "HERMES_VALIDATION_PENDING"
    return row


def _arrangement_actions(
    plan: MusicPlan, metadata: dict[str, Any], daw: Any, *, strict: bool = False,
    missing: list[str] | None = None,
    score_compilations: list[ScoreCompilation] | None = None,
    verified_material_roles: set[str] | None = None,
    verified_phrase_slots: list[PhraseSlotBinding] | None = None,
) -> list[Any]:
    sections = metadata.get("arrangement") or []
    if not sections:
        return []
    actions: list[Any] = []
    cursor = 0.0
    session = daw.snapshot()
    attach_tokens(session)
    if metadata.get("arrangement_score") is not None:
        from copilot.producer.track_spec import TrackSpec

        compiled = compile_score_placements(
            score=ArrangementScore.model_validate(metadata["arrangement_score"]),
            spec=TrackSpec.model_validate(metadata["track_spec"]),
            sample_map=metadata["sample_map"], session=session,
            capabilities=getattr(daw, "capabilities", None),
            evidence_refs=list(plan.evidence_refs),
            verified_material_roles=verified_material_roles or set(),
            verified_phrase_slots=verified_phrase_slots,
        )
        if score_compilations is None:
            raise ValueError("SCORE_COMPILATION_REPORT_REQUIRED")
        score_compilations.append(compiled)
        return compiled.actions
    for section in sections:
        if isinstance(section, dict):
            section_name = str(section.get("name") or "SECTION")
            bars = int(section.get("bars") or 0)
            active_names = list(section.get("active") or [])
        else:
            section_name = str(section.name)
            bars = int(section.bars)
            active_names = list(section.active)
        if bars <= 0:
            if strict:
                raise ValueError(f"ARRANGEMENT_INVALID_SECTION: {section_name}")
            continue
        length = float(bars * 4)
        for name in active_names:
            track = session.track_by_name(str(name))
            if track is None or not any(clip.slot_index == 0 for clip in track.clips):
                if strict:
                    if missing is None:
                        raise ValueError(f"ARRANGEMENT_SOURCE_CLIP_MISSING: {section_name}:{name}")
                    missing.append(f"{section_name}:{name}")
                continue
            actions.append(build_duplicate_clip_to_arrangement_action(
                track=track,
                project_identity=session.project_identity,
                clip_index=0,
                destination_time=cursor,
                length=length,
                reason=f"REAL_LUCAS arrangement {section_name} ({bars} bars)",
                evidence_refs=list(plan.evidence_refs),
                session_incarnation_id=session.session_incarnation_id,
            ))
        cursor += length
    return actions


def _execute_midi_pattern(
    create: Any, pattern: Any, *, daw: Any, persist_dir: Path
) -> dict[str, Any]:
    """Use the compiler's single certified create+pattern transaction."""
    from copilot.runtime.production_compiler import ProductionCompiler
    from copilot.runtime.safe_write import build_safe_write_executor

    current = daw.snapshot()
    attach_tokens(current)
    combined = _single_action_plan(
        create, session=current, plan_id=f"alpha_pattern_{create.action_id}"
    ).model_copy(update={"actions": [create, pattern]})
    compiled = ProductionCompiler().compile(combined, session=current)
    if compiled.status != "COMPILED" or compiled.intent is None:
        return {
            "action_id": pattern.action_id, "action_type": "CREATE_PATTERN",
            "status": "EXECUTION_DEFERRED",
            "reason": "; ".join(compiled.reasons) or compiled.status,
        }
    result = build_safe_write_executor(
        daw,
        journal_path=persist_dir / f"{create.action_id}_pattern_safe_write.jsonl",
        persist_dir=persist_dir,
    ).run(compiled.intent)
    return {
        "action_id": pattern.action_id, "action_type": "CREATE_PATTERN",
        "compound_action_ids": [create.action_id, pattern.action_id],
        "status": "VERIFIED" if result.ok else "FAILED",
        "error": None if result.ok else result.error or "SAFE_WRITE_FAILED",
        "readbacks": [item.model_dump(mode="json") for item in result.readbacks],
        "created_track_stable_id": next(
            (
                item.observed for item in result.readbacks
                if item.action_id == create.action_id and item.parameter == "session.track"
                and item.matched and isinstance(item.observed, str)
            ),
            None,
        ) if result.ok else None,
    }


def _uncertified_phrase_variations(spec: Any, plan: MusicPlan) -> list[dict[str, Any]]:
    """A repeated slot-0 MIDI pattern cannot substantiate a distinct phrase."""
    midi_roles = {
        str(action.target.ref.get("name") or "")
        for action in plan.actions
        if action.action_type is ProductionActionKind.CREATE_PATTERN
    }
    return [
        {
            "action_id": f"phrase_variation:{section.name}:{role}",
            "action_type": "CREATE_PATTERN",
            "status": "EXECUTION_DEFERRED",
            "reason": "PHRASE_MIDI_VARIATION_NOT_CERTIFIED",
            "section": section.name,
            "role": role,
            "source": "PRODUCER_SECTION_VARIATION",
        }
        for section in spec.sections
        if section.variation.strip()
        for role in section.active_roles
        if role in midi_roles
    ]


def _build_sample_index(project_path: Path, evidence: Path) -> tuple[LibraryIndex, dict[str, Any]]:
    roots = [project_path.parent / "Samples"]
    roots = [root for root in roots if root.is_dir()]
    if not roots:
        raise FileNotFoundError("PROJECT_SAMPLE_LIBRARY_MISSING")
    path = evidence / "alpha_sample_library_index.json"
    counts = index_library(roots, path)
    index = load_index(path)
    if index is None:
        raise RuntimeError("SAMPLE_LIBRARY_INDEX_FAILED")
    return index, counts


def _stage_selected_samples(
    *, index: LibraryIndex, selections: dict[str, str],
    authorized_root: Path, project_path: Path,
) -> list[dict[str, Any]]:
    """Materialize only selected, digest-verified indexed assets in this copy."""
    from copilot.sample_library.schemas import AssetStatus

    root = authorized_root.resolve(strict=True)
    samples = (project_path.parent / "Samples").resolve()
    staged: list[dict[str, Any]] = []
    for role, digest in selections.items():
        asset = index.assets.get(digest)
        if asset is None or asset.status is not AssetStatus.INDEXED:
            raise ValueError(f"SELECTED_ASSET_NOT_INDEXED: {role}")
        source = Path(asset.path).resolve(strict=True)
        relative = Path(asset.relative_path.replace("\\", "/"))
        if (
            not source.is_file() or not source.is_relative_to(root)
            or relative.is_absolute() or ".." in relative.parts
        ):
            raise ValueError(f"SELECTED_ASSET_OUTSIDE_AUTHORIZED_LIBRARY: {role}")
        if relative.parts and relative.parts[0].casefold() == "samples":
            relative = Path(*relative.parts[1:])
        target = (samples / relative).resolve()
        if not target.is_relative_to(samples):
            raise ValueError(f"SELECTED_ASSET_TARGET_UNSAFE: {role}")
        sha = hashlib.sha256(source.read_bytes()).hexdigest()
        if sha != digest:
            raise ValueError(f"SELECTED_ASSET_DIGEST_MISMATCH: {role}")
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_file():
            if hashlib.sha256(target.read_bytes()).hexdigest() != digest:
                raise ValueError(f"SELECTED_ASSET_COLLISION: {role}")
        else:
            shutil.copy2(source, target)
            if hashlib.sha256(target.read_bytes()).hexdigest() != digest:
                raise ValueError(f"SELECTED_ASSET_COPY_MISMATCH: {role}")
        staged.append({
            "role": role, "sha256": digest, "path": str(target),
            "license": asset.provenance.get("license", "UNKNOWN"),
        })
    return staged


def _capture_sections(
    *, daw: Any, session: Any, spec: Any,
) -> dict[str, dict[str, Any]]:
    """Capture bounded Main excerpts at the model's actual section boundaries."""
    from copilot.audio.live_capture import capture_master_segment
    from copilot.producer.quality_gate import section_window_specs

    observed: dict[str, dict[str, Any]] = {}
    for section, windows in section_window_specs(spec).items():
        captured = {}
        for position, (start, end) in windows.items():
            asset = capture_master_segment(daw, start, end, require_signal=True)
            if (
                not asset.capture_id or not asset.file_path.is_file()
                or not (0 < float(asset.rms or 0))
                or not (0 < float(asset.peak or 0) < 1.0)
            ):
                raise RuntimeError(f"SECTION_AUDIO_SILENT_OR_CLIPPING: {section}:{position}")
            current = daw.snapshot(include_notes=False)
            attach_tokens(current)
            if current.project_identity != session.project_identity:
                raise RuntimeError("PROJECT_IDENTITY_CHANGED_DURING_LISTENING")
            captured[position] = {
                "capture_id": asset.capture_id, "path": str(asset.file_path),
                "rms": float(asset.rms), "peak": float(asset.peak),
                "start_qn": start, "end_qn": end,
                "project_identity": current.project_identity,
            }
        observed[section] = {**captured["opening"], "windows": captured}
    return observed


def _capture_goal_sources(
    *, daw: Any, session: Any, spec: Any, evidence: Path,
) -> dict[str, dict[str, Any]]:
    from copilot.audio.capture_preflight import generic_capture_preflight
    from copilot.audio.session_diagnose import preflight_session
    from copilot.audio.source_capture_pool_v1 import capture_source_post_mixer_ref
    from copilot.daw.object_ref import ref_from_track

    preflight = generic_capture_preflight(preflight_session(daw))
    if not preflight.get("pass"):
        failures = "; ".join(str(item) for item in preflight.get("missing") or [])
        raise RuntimeError("GENERIC_CAPTURE_PREFLIGHT_FAILED: " + (failures or "UNKNOWN"))

    wanted = {"Kick", "Bass", spec.hook_role}
    captures: dict[str, dict[str, Any]] = {}
    timeline = {}
    cursor = 0.0
    for section in spec.sections:
        for role in section.active_roles:
            timeline.setdefault(role, (cursor, cursor + section.bars * 4.0))
        cursor += section.bars * 4.0
    for role in sorted(wanted):
        track = session.track_by_name(role)
        if track is None or role not in timeline:
            captures[role] = {"ok": False, "reason": "SOURCE_TRACK_OR_SECTION_MISSING"}
            continue
        start, section_end = timeline[role]
        result = capture_source_post_mixer_ref(
            daw, session=session,
            preflight=preflight,
            target_ref=ref_from_track(track, project_identity=session.project_identity),
            start_qn=start, end_qn=min(start + 16.0, section_end),
            region_id=f"producer_{role.lower().replace(' ', '_')}",
            tempo=session.transport.tempo,
            dest_root=evidence / "sources",
        )
        current = daw.snapshot(include_notes=False)
        attach_tokens(current)
        if current.project_identity != session.project_identity:
            raise RuntimeError("PROJECT_IDENTITY_CHANGED_DURING_SOURCE_CAPTURE")
        captures[role] = result
    return captures


def _execute_goal_mix(
    *, daw: Any, spec: Any, decisions: dict[str, dict[str, Any]],
    persist_dir: Path, project_identity: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Audition typed model intent using the existing MusicPlan/SafeWrite mixer."""
    import numpy as np
    import soundfile as sf

    from copilot.audio.live_capture import capture_master_segment
    from copilot.integration.mixing_mastering_v1 import execute_lucas_mix_master_iteration

    selected = {name: row for name, row in decisions.items() if row["decision"] == "APPLY"}
    if not selected:
        return {"status": "NO_MIX_WRITE_REQUESTED"}, []

    def measure(path: Path) -> dict[str, float]:
        with sf.SoundFile(path) as source:
            data = source.read(
                frames=min(source.frames, source.samplerate * 16),
                always_2d=True, dtype="float32",
            )
            sr = source.samplerate
        mono = data.mean(axis=1)
        if not len(mono) or not np.isfinite(mono).all():
            raise RuntimeError("MIX_AUDIO_UNMEASURABLE")
        rms = float(np.sqrt(np.mean(mono ** 2)))
        if rms <= 1e-6:
            raise RuntimeError("MIX_AUDIO_SILENT")
        spectrum = np.abs(np.fft.rfft(mono)) ** 2
        frequencies = np.fft.rfftfreq(len(mono), 1 / sr)
        total = float(np.sum(spectrum))
        return {
            "crest": float(np.max(np.abs(mono))) / rms,
            "low_band": float(np.sum(spectrum[frequencies < 150]) / total),
            "high_band": float(np.sum(spectrum[frequencies > 5000]) / total),
            "rms": rms,
        }

    def observe() -> dict[str, Any]:
        audio = capture_master_segment(
            daw, 0.0, min(16.0, float(spec.sections[0].bars * 4)),
            require_signal=True,
        )
        current = daw.snapshot()
        attach_tokens(current)
        if (
            current.project_identity != project_identity
            or not audio.capture_id or not audio.file_path.is_file()
            or not 0 < float(audio.rms or 0)
            or not 0 < float(audio.peak or 0) < 1
        ):
            raise RuntimeError("MIX_AUDIO_OR_PROJECT_IDENTITY_UNVERIFIED")
        return {
            "capture_id": audio.capture_id, "path": str(audio.file_path),
            "rms": float(audio.rms), "peak": float(audio.peak),
            "sha256": hashlib.sha256(audio.file_path.read_bytes()).hexdigest(),
            "physical": measure(audio.file_path),
        }

    before = observe()
    captures: dict[str, dict[str, Any]] = {"before": before, "comparison": before}

    def post_apply(phase, plan, pre, post):
        if pre.project_identity != project_identity or post.project_identity != project_identity:
            return {"decision": "ROLLBACK", "reason": "PROJECT_IDENTITY_CHANGED"}
        if not plan.actions:
            return {"decision": "KEEP", "reason": "NO_ACTIONS"}
        after = observe()
        previous = captures["comparison"]
        if (
            after["capture_id"] == previous["capture_id"]
            or after["sha256"] == previous["sha256"]
        ):
            return {"decision": "ROLLBACK", "reason": "MIX_CAPTURE_NOT_FRESH"}
        captures[phase] = after
        objective = selected["eq_eight" if phase == "mix" else "limiter"]["objective"]
        dimension = {
            "reduce_low_band": "low_band",
            "reduce_high_band": "high_band",
            "reduce_peak": "crest",
        }[objective]
        baseline = previous["physical"][dimension]
        observed = after["physical"][dimension]
        if not observed < baseline * 0.99:
            return {"decision": "ROLLBACK", "reason": "MIX_OBJECTIVE_NOT_IMPROVED",
                    "before": baseline, "after": observed, "objective": objective}
        if (
            phase == "mix"
            and after["physical"]["crest"] < previous["physical"]["crest"] / 1.413
        ):
            return {"decision": "ROLLBACK", "reason": "MIX_TRANSIENT_LOSS"}
        captures["comparison"] = after
        return {
            "decision": "KEEP", "reason": "MEASURED_OBJECTIVE_IMPROVED",
            "objective": objective, "before": baseline, "after": observed,
            "capture": after,
            "limitation": "Factual spectral/crest proxy; no semantic musical verdict.",
        }

    strategy = {
        "mix": {"parameter_actions": []},
        "master": {"parameter_actions": []},
    }
    for device, decision in selected.items():
        action = decision["action"]
        strategy["mix" if device == "eq_eight" else "master"]["parameter_actions"].append({
            "operation": "set_typed_parameter",
            "track_name": action["track_name"],
            "device_name": "EQ Eight" if device == "eq_eight" else "Limiter",
            "control": action["control"],
            "band": action.get("band"),
            "value": action["value"],
            "unit": action["unit"],
            "reason": decision["reason"],
            "evidence_refs": [before["capture_id"]],
        })
    session = daw.snapshot()
    attach_tokens(session)
    result = execute_lucas_mix_master_iteration(
        strategy=strategy, daw=daw, session=session, persist_dir=persist_dir,
        post_apply=post_apply,
    )
    rows = []
    for device in selected:
        phase = result["phases"].get("mix" if device == "eq_eight" else "master", {})
        accepted = (
            phase.get("writes_verified") == 1
            and phase.get("decision") == "KEEP"
            and not phase.get("deferred")
            and phase.get("status") == "SAFE_WRITE_COMPLETE"
        )
        deferred = phase.get("deferred") or []
        rows.append({
            "action_id": f"mix:{device}", "action_type": "DEVICE_TWEAK",
            "status": "VERIFIED" if accepted else (
                "EXECUTION_DEFERRED" if deferred and not phase.get("writes_verified") else "FAILED"
            ),
            "reason": (
                None if accepted else
                deferred[0]["reason"] if deferred else
                phase.get("error") or phase.get("rollback_error") or phase.get("post_apply", {}).get("reason")
                or "MIX_NOT_RETAINED"
            ),
            "source": "REAL_LUCAS_MIX_DECISION",
            "readbacks": phase.get("writes", []),
        })
    return {"result": result, "captures": captures}, rows


def run_alpha(
    *, evidence: Path = Path("logs"),
    goal: ProducerGoal | None = None,
    expected_project_path: Path | None = None,
    library_index: LibraryIndex | None = None,
    authorized_library_root: Path | None = None,
    opened_project: dict[str, Any] | None = None,
    reference_audio: Path | None = None,
    allow_ui_save: bool = False,
    production_context: ProducerContext | None = None,
    enable_midi_phrases: bool = False,
) -> dict[str, Any]:
    """Run a bounded pass; goal mode never uses a deterministic fallback."""
    evidence.mkdir(parents=True, exist_ok=True)
    started = now_iso()
    artifact_path = evidence / "autonomous_producer_alpha_v1.json"
    report: dict[str, Any] = {
        "milestone": MILESTONE,
        "started_at": started,
        "MUSICAL_WRITES": 0,
        "strategy_provenance": REAL_LUCAS,
        "lucas_owned_files_modified": 0,
        "experimental_midi_phrases_enabled": enable_midi_phrases,
    }
    if goal is not None:
        report["goal"] = goal.model_dump(mode="json")
        report["status"] = "BLOCKED"
    if production_context is not None:
        report["production_context"] = production_context.model_dump(mode="json")
        blockers = production_context.planning_blockers()
        if goal is None:
            blockers.append("PRODUCER_COMPLETE_TRACK_GOAL_REQUIRED")
        if reference_audio is not None:
            blockers.append("PRODUCER_REFERENCE_INPUT_AMBIGUOUS")
        if not production_context.brief.internal_audio_capture_authorized:
            blockers.append("PRODUCER_NO_CAPTURE_DELIVERY_NOT_CERTIFIED")
        if blockers:
            report.update(status="BLOCKED", reason=";".join(blockers))
            atomic_write(artifact_path, report)
            return report
    producer_state: ProducerState | None = None
    producer_state_store: ProducerStateStore | None = None
    from copilot.importing.new_project_v1 import UiSaveRunBudget

    ui_save_budget = UiSaveRunBudget(enabled=allow_ui_save)

    def request_ui_save(opened: dict[str, Any]) -> dict[str, Any]:
        return ui_save_budget.save(opened)

    def persist_producer_state(next_state: ProducerState) -> None:
        nonlocal producer_state
        if producer_state_store is None:
            raise RuntimeError("PRODUCER_STATE_STORE_NOT_READY")
        producer_state = next_state
        path = producer_state_store.save(next_state)
        report["producer_state"] = {
            "session_id": next_state.session_id,
            "path": str(path),
            "phase": next_state.phase.value,
            "event_count": len(next_state.events),
        }

    daw = AbletonTcpAdapter()
    try:
        daw.connect()
        session = daw.snapshot(include_notes=False)
        attach_tokens(session)
        if not session.project_path or not is_copilot_working_copy(session.project_path):
            raise RuntimeError("CONTROLLED_WORKING_COPY_REQUIRED")
        if goal is not None:
            if expected_project_path is None or Path(session.project_path).resolve() != expected_project_path.resolve():
                raise RuntimeError("PRODUCER_WORKING_COPY_MISMATCH")
            if (
                opened_project is None
                or opened_project.get("status") != "OPENED_EMPTY"
                or opened_project.get("project_identity") != session.project_identity
                or Path(str(opened_project.get("working_als") or "")).resolve() != expected_project_path.resolve()
                or (opened_project.get("readiness") or {}).get("launch", {}).get("launch") != "started"
                or not (opened_project.get("readiness") or {}).get("launch", {}).get("process_lifecycle", {}).get("owned")
            ):
                raise RuntimeError("PRODUCER_OWNED_SAVE_CAPABLE_SESSION_REQUIRED")
            if session.tracks:
                raise RuntimeError("PRODUCER_TEMPLATE_NOT_EMPTY")
            if not abs(float(session.transport.tempo) - goal.bpm) < 1e-6:
                raise RuntimeError("PRODUCER_TEMPLATE_TEMPO_MISMATCH_NO_CERTIFIED_TEMPO_WRITE")
        report["SESSION_READY"] = "VERIFIED"
        report["project"] = {
            "path": session.project_path,
            "identity": session.project_identity,
            "token": session.project_token,
            "track_count_before": len(session.tracks),
        }
        if goal is not None:
            from copilot.audio.cross_project_bootstrap_v1 import bootstrap_project

            bootstrap = bootstrap_project(
                daw, evidence=evidence / "capture_bootstrap.json",
                journal_dir=evidence / "capture_bootstrap_journals",
            )
            report["capture_bootstrap"] = bootstrap
            if bootstrap.get("CROSS_PROJECT_BOOTSTRAP_V1") != "VERIFIED":
                raise RuntimeError(f"CAPTURE_BOOTSTRAP_NOT_VERIFIED: {bootstrap.get('reason') or bootstrap.get('status')}")
            session = daw.snapshot(include_notes=False)
            attach_tokens(session)
            if session.project_identity != report["project"]["identity"]:
                raise RuntimeError("PROJECT_IDENTITY_CHANGED_DURING_BOOTSTRAP")
        state_key = hashlib.sha256(
            f"autonomous-alpha:{session.project_identity}".encode("utf-8")
        ).hexdigest()[:20]
        producer_state_store = ProducerStateStore(evidence / "producer_state")
        producer_state = producer_state_store.load(state_key)
        if producer_state is None:
            producer_state = ProducerState(
                session_id=state_key,
                project_identity=session.project_identity,
            )
        persist_producer_state(
            producer_state.record(
                "SESSION_OBSERVED",
                phase=ProducerPhase.PLANNING,
                payload={
                    "tempo_bpm": session.transport.tempo,
                    "track_count": len(session.tracks),
                },
            )
        )

        project_path = Path(session.project_path)
        if goal is not None:
            if library_index is None or authorized_library_root is None:
                raise ValueError("PRODUCER_AUTHORIZED_INDEXED_LIBRARY_REQUIRED")
            index = library_index
            index_counts = {"indexed": len(index.assets), "source": str(authorized_library_root)}
        else:
            index, index_counts = _build_sample_index(project_path, evidence)
        if production_context is not None:
            pack = None
            reference_path = None
            reference = production_context.references[0]
        elif goal is None:
            reference_path = Path(os.environ["COPILOT_ALPHA_REFERENCE"]) if os.environ.get("COPILOT_ALPHA_REFERENCE") else discover_reference(evidence, project_path=project_path)
            pack = analyze_reference_music(
                reference_path,
                reference_state_token=f"reference:alpha:{reference_path.stem}",
                target_state_token=f"target:alpha:{session.project_identity}",
                tempo_bpm=float(session.transport.tempo),
                use_cache=False,
            )
            reference = build_reference_context(pack)
        else:
            from copilot.schemas.lucas_integration import ReferenceContext

            reference_path = reference_audio
            if reference_path is not None:
                if not reference_path.is_file() or reference_path.suffix.lower() not in {
                    ".wav", ".flac", ".aiff", ".aif",
                }:
                    raise ValueError("PRODUCER_REFERENCE_AUDIO_UNSUPPORTED")
                pack = analyze_reference_music(
                    reference_path,
                    reference_state_token=f"external-comparison:{reference_path.stem}",
                    target_state_token=f"target:{session.project_identity}",
                    tempo_bpm=goal.bpm, use_cache=False,
                )
                reference = build_reference_context(pack)
            else:
                pack = None
                reference = ReferenceContext(
                    reference_state_token=f"no-reference:{session.project_identity}",
                    identity="NO_USER_REFERENCE",
                    tempo_bpm=goal.bpm,
                    limitations=["No audio reference supplied; style is producer intent."],
                )
        roles = [
            SampleRole.KICK, SampleRole.CLAP, SampleRole.CLOSED_HAT,
            SampleRole.SHAKER, SampleRole.PERCUSSION, SampleRole.TOP_LOOP,
            SampleRole.BASS, SampleRole.VOCAL, SampleRole.SYNTH, SampleRole.FX,
            SampleRole.IMPACT, SampleRole.TEXTURE,
        ]
        samples = production_context.samples if production_context is not None else build_sample_set_context(
            index, task_id="autonomous-producer-alpha-v1", wanted_roles=roles,
            per_role=3, bpm=float(session.transport.tempo),
        )
        context = build_lucas_input(
            user_intent=UserIntent(description=goal.prompt if goal else (
                "CONTROLLED_ALPHA_INTENT: create and develop approximately 32 bars "
                "of an electronic track using the available project, real sample "
                "library, and reference context, with a coherent groove, bass/low-end "
                "role, musical texture, and basic mix balance. You must make the "
                "musical decisions and return a non-empty 5-to-8-section arrangement."
            ), requested_bpm=goal.bpm if goal else None),
            reference=reference,
            samples=samples,
            project=build_project_context(session),
            style=production_context.brief.style_context() if production_context else StyleContext(),
        )
        provider = configured_http_provider()
        if provider is None:
            raise RealLucasRequired("REAL_LUCAS_PROVIDER_UNAVAILABLE")
        planner_provider = LucasPlanningProviderAdapter(
            provider, complete_track=production_context is not None,
        )
        def strict_planner(**kwargs):
            from copilot.musicplan.astra_plan import build_plan_from_prompt

            return build_plan_from_prompt(
                **kwargs, goal=goal,
                preview_root=evidence / "sample_ab",
                authorized_library_root=authorized_library_root,
            )

        planner_run = run_lucas_planner(
            input_context=context,
            index=index,
            session=session,
            provider=planner_provider,
            plan_id="autonomous_producer_alpha_v1",
            planner=strict_planner if goal else None,
            production_context=production_context,
        )
        report["lucas"] = {
            "entrypoint": "build_plan_from_prompt",
            "strategy_provenance": REAL_LUCAS if planner_run.planner_metadata.get("astra_used") else CONTROLLED_FIXTURE,
            "provider": planner_provider.identity,
            "provider_version": planner_provider.version,
            "response_adapter": planner_provider.response_adapter,
            "metadata": planner_run.planner_metadata,
        }
        if not planner_run.planner_metadata.get("astra_used"):
            report["lucas"]["provider_raw_response"] = planner_provider.last_raw
            raise RealLucasRequired("LUCAS_PLANNER_FELL_BACK_TO_DETERMINISTIC_RECIPE")
        if production_context is not None:
            report["production_context"] = planner_run.planner_metadata["production_context"]
        plan = planner_run.plan
        if goal is not None:
            from copilot.producer.track_spec import TrackSpec

            spec = TrackSpec.model_validate(planner_run.planner_metadata["track_spec"])
            goal.validate_track_spec(spec)
            report["track_spec"] = spec.model_dump(mode="json")
            report["mix_decisions"] = planner_run.planner_metadata["mix_decisions"]
            report["producer_criteria"] = planner_run.planner_metadata["producer_criteria"]
            report["selected_samples"] = _stage_selected_samples(
                index=index,
                selections=planner_run.planner_metadata["sample_map"],
                authorized_root=authorized_library_root,
                project_path=project_path,
            )
            report["sample_comparisons"] = planner_run.planner_metadata["sample_comparisons"]
            producer_state = producer_state.model_copy(update={
                "decisions": {
                    **producer_state.decisions,
                    "track_spec": report["track_spec"],
                    "sample_selections": planner_run.planner_metadata["sample_map"],
                    "selection_reasons": planner_run.planner_metadata["selection_reasons"],
                    "rejected_candidates": planner_run.planner_metadata["rejected_candidates"],
                    "mix_decisions": report["mix_decisions"],
                    "producer_criteria": report["producer_criteria"],
                    "production_context": report.get("production_context"),
                    "arrangement_score": planner_run.planner_metadata.get("arrangement_score"),
                },
            })
        persist_producer_state(
            producer_state.record(
                "PLAN_ACCEPTED",
                phase=ProducerPhase.EXECUTING,
                payload={
                    "plan_id": plan.plan_id,
                    "action_count": len(plan.actions),
                    "arrangement_sections": len(planner_run.planner_metadata.get("arrangement") or []),
                    "track_spec_present": bool(planner_run.planner_metadata.get("track_spec")),
                },
            )
        )
        report.update({
            "reference": {
                "path": str(reference_path) if reference_path else None,
                "identity": reference.identity,
                "rights_state": (
                    "PROVIDED_READ_ONLY_CONTEXT" if production_context
                    else "UNKNOWN_EXTERNAL_RECORDING" if pack else "NO_REFERENCE"
                ),
                "role": (
                    "provided_reference_evidence_only" if production_context
                    else "aggregate_energy_groove_mix_comparison_only" if goal and pack
                    else "groove_lowend_texture_analysis_only" if pack else "NONE"
                ),
                "pack": pack.model_dump(mode="json") if pack else None,
            },
            "sample_set_context": (
                report["production_context"]["samples"] if production_context
                else samples.model_dump(mode="json")
            ),
            "sample_index": index_counts,
            "style_context": context.style.model_dump(mode="json"),
            "reference_contexts": (
                [ref.model_dump(mode="json") for ref in production_context.references]
                if production_context else [reference.model_dump(mode="json")]
            ),
            "lucas": {
                "entrypoint": "build_plan_from_prompt",
                "strategy_provenance": REAL_LUCAS,
                "provider": planner_provider.identity,
                "provider_version": planner_provider.version,
                "response_adapter": planner_provider.response_adapter,
                "metadata": planner_run.planner_metadata,
                "plan": plan.model_dump(mode="json"),
            },
        })

        persist_dir = evidence / "autonomous_producer_alpha_v1" / "safe_write"
        persist_dir.mkdir(parents=True, exist_ok=True)
        dispositions: list[dict[str, Any]] = []
        report["execution"] = {"actions": dispositions}
        midi_patterns = {
            str(action.target.ref.get("name")): action
            for action in plan.actions
            if action.action_type is ProductionActionKind.CREATE_PATTERN
        }
        consumed_patterns: set[str] = set()
        ordered = sorted(
            plan.actions,
            key=lambda action: 0 if action.action_type is ProductionActionKind.CREATE_TRACK else 1 if action.action_type in {ProductionActionKind.SAMPLE_LOAD, ProductionActionKind.LOAD_SAMPLE} else 2,
        )
        for action in ordered:
            if action.action_type is ProductionActionKind.CREATE_PATTERN and action.action_id in consumed_patterns:
                continue
            if action.action_type not in SUPPORTED_ALPHA_ACTIONS:
                dispositions.append({
                    "action_id": action.action_id,
                    "action_type": _plan_action_name(action),
                    "status": "EXECUTION_DEFERRED",
                    "reason": "UNSUPPORTED_OR_NOT_CERTIFIED",
                })
                continue
            pattern = midi_patterns.get(str(action.target.ref.get("name")))
            if action.action_type is ProductionActionKind.CREATE_TRACK and pattern is not None:
                row = _execute_midi_pattern(action, pattern, daw=daw, persist_dir=persist_dir)
                consumed_patterns.add(pattern.action_id)
            else:
                row = _execute_one(action, daw=daw, persist_dir=persist_dir)
            row["source"] = "REAL_LUCAS_MUSICPLAN"
            dispositions.append(row)
            if row["status"] in {"VERIFIED", "FAILED"}:
                report["MUSICAL_WRITES"] += 1

        missing_sources: list[str] = []
        score_compilations: list[ScoreCompilation] = []
        material_action_roles = {
            action.action_id: str(action.target.ref.get("name") or "")
            for action in plan.actions if action.action_type is ProductionActionKind.SAMPLE_LOAD
        }
        verified_material_roles = {
            material_action_roles[row["action_id"]]
            for row in dispositions
            if row["status"] == "VERIFIED" and row["action_id"] in material_action_roles
        }
        verified_material_devices = {
            material_action_roles[row["action_id"]]: row["loaded_device_stable_id"]
            for row in dispositions if row["status"] == "VERIFIED"
            and row["action_id"] in material_action_roles and row.get("loaded_device_stable_id")
        }
        verified_phrase_slots = None
        if enable_midi_phrases and planner_run.planner_metadata.get("arrangement_score") is not None:
            if any(row["status"] == "FAILED" for row in dispositions):
                report.update(status="DRAFT", reason="INITIAL_MATERIAL_FAILED_BEFORE_MIDI_PHRASES")
                persist_producer_state(producer_state.record(
                    "MIDI_PHRASES_NOT_ATTEMPTED", phase=ProducerPhase.DRAFT,
                    detail=report["reason"],
                ))
                return report
            score = ArrangementScore.model_validate(planner_run.planner_metadata["arrangement_score"])
            owned_midi_ids = frozenset(
                row["created_track_stable_id"] for row in dispositions
                if row["status"] == "VERIFIED" and row.get("created_track_stable_id")
            )
            material_session = daw.snapshot()
            attach_tokens(material_session)
            phrase_material = prepare_score_phrase_material(
                score=score, spec=spec, sample_map=planner_run.planner_metadata["sample_map"],
                session=material_session, owned_midi_track_ids=owned_midi_ids,
                verified_material_roles=verified_material_roles,
                verified_material_devices=verified_material_devices,
                capabilities=frozenset(getattr(daw, "capabilities", ())), enabled=True,
            )
            report["phrase_material"] = phrase_material.model_dump(mode="json")
            producer_state = producer_state.model_copy(update={"observations": {
                **producer_state.observations, "phrase_material": report["phrase_material"],
            }})
            persist_producer_state(producer_state.record(
                "EXPERIMENTAL_MIDI_PHRASES_PLANNED", phase=ProducerPhase.EXECUTING,
                payload={"new_phrase_count": len(phrase_material.actions)},
            ))
            dispositions.extend({
                **deferred.model_dump(mode="json"),
                "action_id": f"phrase:{deferred.event_id}",
                "action_type": "CREATE_MIDI_PHRASE",
                "source": "SCORE_PHRASE_MATERIAL",
            } for deferred in phrase_material.deferred)
            verified_clip_ids: dict[str, str] = {}
            for action in phrase_material.actions:
                row = _execute_one(
                    action, daw=daw, persist_dir=persist_dir,
                    phrase_material=phrase_material, musical_score=score,
                    verified_phrase_clip_ids=verified_clip_ids,
                    experimental_midi_track_ids=owned_midi_ids,
                )
                row["source"] = "SCORE_PHRASE_MATERIAL"
                dispositions.append(row)
                if row["status"] in {"VERIFIED", "FAILED"}:
                    report["MUSICAL_WRITES"] += 1
                if row["status"] == "FAILED":
                    report.update(status="DRAFT", reason=row.get("error"))
                    persist_producer_state(producer_state.record(
                        "EXPERIMENTAL_MIDI_PHRASE_FAILED", phase=ProducerPhase.DRAFT,
                        detail=report["reason"] or "SAFE_WRITE_FAILED",
                    ))
                    return report
                if row["status"] == "VERIFIED":
                    matching = [
                        item["observed"]["stable_id"] for item in row["readbacks"]
                        if item["parameter"] == "clip.midi_phrase" and item["matched"]
                        and isinstance(item["observed"], dict) and item["observed"].get("stable_id")
                    ]
                    if len(matching) != 1:
                        raise RuntimeError("MIDI_PHRASE_VERIFIED_READBACK_MISSING")
                    verified_clip_ids[action.action_id] = matching[0]
            phrase_session = daw.snapshot()
            attach_tokens(phrase_session)
            verified_phrase_slots = bind_verified_phrase_slots(
                phrase_material, score=score, sample_map=planner_run.planner_metadata["sample_map"],
                session=phrase_session, verified_clip_ids=verified_clip_ids,
            )
            report["verified_phrase_slots"] = [
                binding.model_dump(mode="json") for binding in verified_phrase_slots
            ]
            producer_state = producer_state.model_copy(update={"observations": {
                **producer_state.observations,
                "verified_phrase_slots": report["verified_phrase_slots"],
            }})
            persist_producer_state(producer_state.record(
                "MIDI_PHRASE_READBACKS_OBSERVED", phase=ProducerPhase.EXECUTING,
                payload={"verified_phrase_count": len(verified_phrase_slots)},
            ))
        arrangement_actions = _arrangement_actions(
            plan, planner_run.planner_metadata, daw, strict=goal is not None,
            missing=missing_sources,
            score_compilations=score_compilations,
            verified_material_roles=verified_material_roles,
            verified_phrase_slots=verified_phrase_slots,
        )
        score_compilation = score_compilations[0] if score_compilations else None
        binding_by_action = {
            binding.action_id: binding
            for binding in (score_compilation.bindings if score_compilation else [])
        }
        if score_compilation is not None:
            report["arrangement_score_compilation"] = {
                **score_compilation.model_dump(mode="json"), "status": score_compilation.status,
            }
            producer_state = producer_state.model_copy(update={
                "observations": {
                    **producer_state.observations,
                    "arrangement_score_compilation": report["arrangement_score_compilation"],
                },
            })
            persist_producer_state(producer_state.record(
                "ARRANGEMENT_SCORE_BOUND",
                phase=ProducerPhase.EXECUTING,
                payload={
                    "placement_count": len(score_compilation.bindings),
                    "deferred_count": len(score_compilation.deferred),
                },
            ))
            dispositions.extend({
                **deferred.model_dump(mode="json"),
                "action_id": f"score:{deferred.event_id}",
                "action_type": "ARRANGEMENT_SCORE",
                "source": "REAL_LUCAS_ARRANGEMENT_SCORE",
            } for deferred in score_compilation.deferred)
        arrangement_rows: list[dict[str, Any]] = []
        for action in arrangement_actions:
            binding = binding_by_action.get(action.action_id)
            if binding is not None:
                row = _execute_one(
                    action, daw=daw, persist_dir=persist_dir, score_binding=binding,
                )
            else:
                row = _execute_one(action, daw=daw, persist_dir=persist_dir)
            row["source"] = "REAL_LUCAS_ARRANGEMENT"
            if binding is not None:
                row["score_binding"] = binding.model_dump(mode="json")
            dispositions.append(row)
            arrangement_rows.append({
                "section": (
                    binding.section if binding else
                    action.reason.split(" arrangement ", 1)[-1].split(" (", 1)[0]
                ),
                "track": binding.role if binding else str(action.target.ref.get("name") or ""),
                "status": row["status"],
            })
            if row["status"] in {"VERIFIED", "FAILED"}:
                report["MUSICAL_WRITES"] += 1
        for source in missing_sources:
            dispositions.append({
                "action_id": f"missing:{source}",
                "action_type": "DUPLICATE_CLIP_TO_ARRANGEMENT",
                "status": "EXECUTION_DEFERRED",
                "reason": f"ARRANGEMENT_SOURCE_CLIP_MISSING: {source}",
                "source": "REAL_LUCAS_ARRANGEMENT",
            })
        if goal is not None:
            phrase_gaps = (
                _uncertified_phrase_variations(spec, plan)
                if score_compilation is None else []
            )
            dispositions.extend(phrase_gaps)
            report["phrase_variations"] = {
                "status": (
                    "DRAFT" if phrase_gaps else
                    "CONCRETE_SCORE_PLANNED" if score_compilation is not None else
                    "NO_MIDI_VARIATION_CLAIMED"
                ),
                "unverified": phrase_gaps,
            }
        report["execution"] = {
            "actions": dispositions,
            "executable_verified": sum(row.get("status") == "VERIFIED" for row in dispositions),
            "deferred": sum(row.get("status") == "EXECUTION_DEFERRED" for row in dispositions),
            "failed": sum(row.get("status") == "FAILED" for row in dispositions),
            "direct_lucas_writes": 0,
            "direct_soniq_writes": 0,
            "safe_write_authorities": 1,
        }
        persist_producer_state(
            producer_state.record(
                "EXECUTION_OBSERVED",
                phase=ProducerPhase.OBSERVING,
                payload={
                    "verified": report["execution"]["executable_verified"],
                    "deferred": report["execution"]["deferred"],
                    "failed": report["execution"]["failed"],
                },
            )
        )
        if report["execution"]["failed"]:
            report["status"] = "DRAFT" if goal else "FAILED_EXECUTION"
            return report
        if goal is not None:
            if not allow_ui_save:
                save = request_ui_save(opened_project)
                report["save"] = save
                report["persistence"] = "PERSISTENCE_PENDING"
                report["status"] = "DRAFT"
                report["blockers"] = [save["reason"]]
                persist_producer_state(
                    producer_state.model_copy(update={
                        "pending_issues": report["blockers"],
                        "stop_reason": report["blockers"][0],
                    }).record(
                        "SAVE_NOT_ATTEMPTED",
                        phase=ProducerPhase.DRAFT,
                        detail=report["blockers"][0],
                    )
                )
                return report
            daw.disconnect()
            save = request_ui_save(opened_project)
            report["save"] = save
            if save.get("status") != "SAVED_REOPENED":
                report["status"] = "DRAFT"
                report["blockers"] = [save.get("reason") or "SAVE_NOT_VERIFIED"]
                persist_producer_state(
                    producer_state.model_copy(update={
                        "pending_issues": report["blockers"],
                        "stop_reason": report["blockers"][0],
                    }).record(
                        "SAVE_BLOCKED",
                        phase=ProducerPhase.DRAFT,
                        detail=report["blockers"][0],
                    )
                )
                return report
            daw = AbletonTcpAdapter()
            daw.connect()
            reopened = daw.snapshot(include_notes=False)
            attach_tokens(reopened)
            if (
                reopened.project_identity != session.project_identity
                or Path(str(reopened.project_path or "")).resolve() != project_path.resolve()
                or len(reopened.tracks) != save.get("track_count")
            ):
                raise RuntimeError("REOPENED_PROJECT_SNAPSHOT_MISMATCH")
            if report["execution"]["deferred"]:
                report["status"] = "DRAFT"
                report["blockers"] = ["ESSENTIAL_ACTION_DEFERRED"]
                persist_producer_state(
                    producer_state.model_copy(update={
                        "pending_issues": [
                            row["reason"] for row in dispositions
                            if row["status"] == "EXECUTION_DEFERRED"
                        ],
                        "stop_reason": "ESSENTIAL_ACTION_DEFERRED",
                    }).record(
                        "ESSENTIAL_ACTIONS_DEFERRED",
                        phase=ProducerPhase.DRAFT,
                        payload={"deferred": report["execution"]["deferred"]},
                    )
                )
                return report
            if any(row["decision"] == "APPLY" for row in report["mix_decisions"].values()):
                report["mix_pass"], mix_rows = _execute_goal_mix(
                    daw=daw, spec=spec, decisions=report["mix_decisions"],
                    persist_dir=persist_dir, project_identity=session.project_identity,
                )
                dispositions.extend(mix_rows)
                report["MUSICAL_WRITES"] += sum(
                    row["status"] in {"VERIFIED", "FAILED"} for row in mix_rows
                )
                report["execution"]["deferred"] += sum(
                    row["status"] == "EXECUTION_DEFERRED" for row in mix_rows
                )
                report["execution"]["failed"] += sum(
                    row["status"] == "FAILED" for row in mix_rows
                )
                report["execution"]["executable_verified"] += sum(
                    row["status"] == "VERIFIED" for row in mix_rows
                )
                persist_producer_state(
                    producer_state.record(
                        "TYPED_MIX_OBSERVED",
                        phase=ProducerPhase.OBSERVING,
                        payload={
                            "actions": mix_rows,
                            "capture_ids": {
                                key: row["capture_id"]
                                for key, row in report["mix_pass"]["captures"].items()
                            },
                        },
                    )
                )
                mix_unverified = any(row["status"] != "VERIFIED" for row in mix_rows)
                if mix_unverified:
                    report["status"] = "DRAFT"
                    report["blockers"] = [
                        row["reason"] or "MIX_NOT_VERIFIED" for row in mix_rows
                        if row["status"] != "VERIFIED"
                    ]
                    persist_producer_state(
                        producer_state.model_copy(update={
                            "pending_issues": report["blockers"],
                            "stop_reason": report["blockers"][0],
                        }).record("TYPED_MIX_BLOCKED", phase=ProducerPhase.DRAFT)
                    )
                    if not any(row["status"] == "VERIFIED" for row in mix_rows):
                        return report
                from copilot.producer.blind_review import prepare_blind_review

                mix_captures = report["mix_pass"]["captures"]
                latest = mix_captures.get("master") or mix_captures.get("mix")
                try:
                    report["blind_review"] = prepare_blind_review(
                        mix_captures["before"], latest,
                        destination=evidence / "blind_review",
                    )
                except (ValueError, OSError) as exc:
                    report["blind_review"] = {
                        "status": "NOT_AVAILABLE",
                        "reason": f"{type(exc).__name__}: {exc}",
                        "artistic_quality_human_verified": False,
                    }
                previous = report["save"]
                ready = previous.get("readiness") or {}
                daw.disconnect()
                mix_save = request_ui_save({
                    "status": "OPENED_EMPTY",
                    "working_als": str(project_path),
                    "project_identity": session.project_identity,
                    "saved_sha256": previous["sha256"],
                    "process_pid": ready.get("same_process_pid"),
                    "readiness": ready,
                })
                report["mix_save"] = mix_save
                if mix_save.get("status") != "SAVED_REOPENED":
                    report["status"] = "DRAFT"
                    report["blockers"] = [mix_save.get("reason") or "MIX_SAVE_NOT_VERIFIED"]
                    persist_producer_state(
                        producer_state.model_copy(update={
                            "pending_issues": report["blockers"],
                            "stop_reason": report["blockers"][0],
                        }).record("MIX_SAVE_BLOCKED", phase=ProducerPhase.DRAFT)
                    )
                    return report
                report["save"] = mix_save
                daw = AbletonTcpAdapter()
                daw.connect()
                verified_mix = daw.snapshot(include_notes=False)
                attach_tokens(verified_mix)
                if verified_mix.project_identity != session.project_identity:
                    raise RuntimeError("MIX_REOPEN_IDENTITY_MISMATCH")
                if mix_unverified:
                    return report

        after_session = daw.snapshot(include_notes=False)
        attach_tokens(after_session)
        section_captures = (
            _capture_sections(daw=daw, session=after_session, spec=spec)
            if goal is not None else {}
        )
        source_captures = (
            _capture_goal_sources(
                daw=daw, session=after_session, spec=spec, evidence=evidence
            )
            if goal is not None else {}
        )
        if goal is not None:
            report["section_captures"] = section_captures
            report["source_captures"] = source_captures
            persist_producer_state(
                producer_state.record(
                    "SECTIONS_LISTENED",
                    phase=ProducerPhase.CRITIQUING,
                    payload={
                        "section_capture_ids": {
                            name: row["capture_id"]
                            for name, row in section_captures.items()
                        },
                        "source_statuses": {
                            name: row.get("signal_status", row.get("reason"))
                            for name, row in source_captures.items()
                        },
                    },
                )
            )
        audio = capture_and_analyze_master(
            daw, after_session, label="alpha_initial_pass",
        )
        perception = run_advanced_perception(
            audio["music_analysis"], audio_path=Path(audio["path"]),
        )
        post_context = {
            "project_identity": after_session.project_identity,
            "plan_id": plan.plan_id,
            "audio": {
                "capture_id": audio["capture_id"], "path": audio["path"],
                "rms": audio["rms"], "peak": audio["peak"], "region": audio["region"],
            },
            "section_captures": section_captures,
            "source_captures": source_captures,
            "music_analysis": audio["music_analysis"].model_dump(mode="json"),
            "physical_dsp": _jsonable(audio["dsp"]),
            "advanced_perception": perception.model_dump(mode="json"),
            "reference_identity": reference.identity,
            "limitations": list(dict.fromkeys(reference.limitations + (list(pack.limitations) if pack else []))),
        }
        report["post_change_context"] = post_context
        critique = run_lucas_critique_with_provider_failover(
            plan=plan,
            session=after_session,
            evidence_context=post_context,
            timeout_s=30.0,
        )
        revisions: list[dict[str, Any]] = []
        if goal is not None and critique.get("status") == "CRITIQUE_COMPLETE":
            from copilot.musicplan import build_set_track_volume_action
            from copilot.producer.revision import volume_revision_action
            from copilot.schemas.musicplan import VolumeOperation

            for iteration in range(1, 4):
                result = critique["result"]
                if result.verdict == "finalize":
                    break
                if result.verdict != "improve" or not result.top_3_issues:
                    report["revision_stop_reason"] = "CRITIQUE_HAS_NO_ACTIONABLE_ISSUE"
                    break
                priority_issue = sorted(
                    result.top_3_issues, key=lambda item: item.priority,
                )[0]
                current = daw.snapshot()
                attach_tokens(current)
                if current.project_identity != session.project_identity:
                    report["revision_stop_reason"] = "PROJECT_IDENTITY_CHANGED"
                    break
                request = (
                    "Return ONLY a JSON object with exactly track_name, target_volume "
                    "(native Ableton mixer 0..1, within 0.08 of current), and reason. "
                    "Propose one evidence-backed set_track_volume change, or return "
                    "an empty object if volume cannot address the issue. No LOM, Python, "
                    "routing, EQ or device commands. "
                    f"Single highest priority issue: {json.dumps(priority_issue.model_dump())}. "
                    f"Tracks: {json.dumps({t.name: t.mixer.volume for t in current.tracks if t.role in {'audio', 'midi'}})}. "
                    f"Captures: {json.dumps({name: {'rms': row['rms'], 'peak': row['peak']} for name, row in section_captures.items()})}"
                )
                try:
                    reason_fn = (
                        getattr(provider, "reason_json_object", None)
                        or getattr(provider, "_reason_chat_json_object", None)
                        or provider.reason
                    )
                    raw = reason_fn(request, timeout_s=30.0)
                    revision = volume_revision_action(
                        raw, session=current,
                        evidence_refs=[row["capture_id"] for row in section_captures.values()],
                    )
                except (ValueError, TypeError, KeyError, TimeoutError) as exc:
                    report["revision_stop_reason"] = f"REVISION_PROVIDER_OR_CONTRACT_FAILED: {exc}"
                    break
                row = _execute_one(revision, daw=daw, persist_dir=persist_dir)
                revisions.append({"iteration": iteration, "write": row})
                if row["status"] != "VERIFIED":
                    report["revision_stop_reason"] = row.get("error") or row.get("reason", "REVISION_UNVERIFIED")
                    break
                report["MUSICAL_WRITES"] += 1
                new_session = daw.snapshot(include_notes=False)
                attach_tokens(new_session)
                new_captures = _capture_sections(daw=daw, session=new_session, spec=spec)
                new_critique = run_lucas_critique_with_provider_failover(
                    plan=plan, session=new_session, providers=[provider],
                    evidence_context={"section_captures": new_captures, "previous": section_captures},
                    timeout_s=30.0,
                )
                better = (
                    new_critique.get("status") == "CRITIQUE_COMPLETE"
                    and (
                        new_critique["result"].verdict == "finalize"
                        or (
                            new_critique["result"].verdict == "improve"
                            and not any(
                                issue.area == priority_issue.area
                                and issue.issue == priority_issue.issue
                                for issue in new_critique["result"].top_3_issues
                            )
                            and len(new_critique["result"].top_3_issues) < len(result.top_3_issues)
                        )
                    )
                    and all(
                        new_captures[name]["windows"][position]["rms"]
                        >= old["rms"] * 0.5
                        for name, section in section_captures.items()
                        for position, old in section["windows"].items()
                    )
                )
                if not better:
                    before = current.track_by_name(str(revision.target.ref.get("name")))
                    fresh = daw.snapshot()
                    attach_tokens(fresh)
                    changed = fresh.track_by_name(before.name)
                    restore = build_set_track_volume_action(
                        track=changed, project_identity=fresh.project_identity,
                        operation=VolumeOperation.SET,
                        expected_before=changed.mixer.volume,
                        target_value=before.mixer.volume,
                        reason="Restore pre-revision volume; critique found no evidence of improvement",
                        evidence_refs=[row["capture_id"] for row in section_captures.values()],
                        session_incarnation_id=fresh.session_incarnation_id,
                    )
                    undo = _execute_one(restore, daw=daw, persist_dir=persist_dir)
                    revisions[-1]["rollback"] = undo
                    report["revision_stop_reason"] = (
                        "NO_MEASURABLE_IMPROVEMENT"
                        if undo["status"] == "VERIFIED"
                        else "REVISION_ROLLBACK_UNVERIFIED"
                    )
                    break
                section_captures = new_captures
                source_captures = _capture_goal_sources(
                    daw=daw, session=new_session, spec=spec, evidence=evidence
                )
                critique = new_critique
                revisions[-1]["result"] = "IMPROVED_WITH_CAPTURE_AND_CRITIQUE"
                persist_producer_state(
                    producer_state.model_copy(update={"iteration": iteration}).record(
                        "REVISION_ACCEPTED",
                        phase=ProducerPhase.ADJUSTING,
                        payload={"iteration": iteration, "action_id": revision.action_id,
                                 "resolved_issue": priority_issue.model_dump(mode="json")},
                    )
                )
            report["revisions"] = revisions
            report["section_captures"] = section_captures
            report["source_captures"] = source_captures
            if any(row.get("result") == "IMPROVED_WITH_CAPTURE_AND_CRITIQUE" for row in revisions):
                previous = report["save"]
                ready = previous.get("readiness") or {}
                reopened_owned = {
                    "status": "OPENED_EMPTY",
                    "working_als": str(project_path),
                    "project_identity": session.project_identity,
                    "saved_sha256": previous["sha256"],
                    "process_pid": ready.get("same_process_pid"),
                    "readiness": ready,
                }
                daw.disconnect()
                revision_save = request_ui_save(reopened_owned)
                report["revision_save"] = revision_save
                if revision_save.get("status") != "SAVED_REOPENED":
                    report["status"] = "DRAFT"
                    report["blockers"] = [revision_save.get("reason") or "REVISION_SAVE_NOT_VERIFIED"]
                    return report
                report["save"] = revision_save
                daw = AbletonTcpAdapter()
                daw.connect()
                confirmed = daw.snapshot(include_notes=False)
                attach_tokens(confirmed)
                if confirmed.project_identity != session.project_identity:
                    raise RuntimeError("REVISION_REOPEN_IDENTITY_MISMATCH")
        report["lucas_feedback"] = critique
        report["critique_provider_limited"] = critique.get("status") != "CRITIQUE_COMPLETE"
        final_phase = (
            ProducerPhase.ABSTAINED
            if report["critique_provider_limited"]
            else ProducerPhase.DRAFT if goal is not None else ProducerPhase.COMPLETE
        )
        persist_producer_state(
            producer_state.record(
                "CRITIQUE_OBSERVED",
                phase=final_phase,
                detail=(
                    "provider unavailable; no musical verdict fabricated"
                    if report["critique_provider_limited"]
                    else "typed critique completed"
                ),
                payload={"status": critique.get("status")},
            )
        )
        if report["critique_provider_limited"]:
            report["status"] = "PRODUCTION_PASS_VERIFIED / REVISION_PROVIDER_LIMITED"
        else:
            report["status"] = "PRODUCTION_PASS_VERIFIED"
        report["final"] = {
            "original_untouched": True,
            "direct_lucas_writes": 0,
            "direct_soniq_writes": 0,
            "safe_write_authorities": 1,
            "transport_stopped": not bool(after_session.transport.playing),
        }
        if goal is not None:
            from copilot.producer.quality_gate import (
                evaluate_delivery, verify_arrangement_timeline,
            )

            final_session = daw.snapshot(include_notes=False)
            attach_tokens(final_session)
            if final_session.project_identity != session.project_identity:
                raise RuntimeError("PROJECT_IDENTITY_CHANGED_BEFORE_DELIVERY")
            geometry = verify_arrangement_timeline(
                spec,
                tracks={track.name: track.index for track in final_session.tracks},
                clips=daw.get_arrangement_clips().get("clips", []),
                score_bindings=score_compilation.bindings if score_compilation else None,
            )
            if score_compilation is not None and (
                final_session.transport.signature_numerator,
                final_session.transport.signature_denominator,
            ) != (spec.meter_numerator, spec.meter_denominator):
                geometry.append("SCORE_LIVE_METER_MISMATCH")
            report["arrangement_geometry"] = {
                "status": "VERIFIED" if not geometry else "BLOCKED",
                "reasons": geometry,
                "source": "authoritative_reopened_live_clip_readback",
            }
            revision_actions = [
                item
                for revision in revisions
                for item in (revision.get("write"), revision.get("rollback"))
                if item is not None
            ]
            report["final"]["transport_stopped"] = not final_session.transport.playing
            report["quality_gate"] = evaluate_delivery(
                goal=goal, spec=spec, project_path=project_path,
                project_identity=final_session.project_identity,
                reopened_identity=(
                    final_session.project_identity
                    if report.get("save", {}).get("status") == "SAVED_REOPENED"
                    else None
                ),
                tempo_bpm=final_session.transport.tempo,
                arrangement=arrangement_rows, captures=section_captures,
                arrangement_geometry=geometry,
                role_captures=source_captures,
                actions=[*dispositions, *revision_actions],
                critique_verdict=(
                    critique["result"].verdict
                    if critique.get("status") == "CRITIQUE_COMPLETE" else None
                ),
                transport_stopped=not final_session.transport.playing,
            )
            report["status"] = report["quality_gate"]["status"]
            report["blockers"] = report["quality_gate"]["reasons"]
            persist_producer_state(
                producer_state.model_copy(update={
                    "pending_issues": report["blockers"],
                    "stop_reason": report["blockers"][0] if report["blockers"] else None,
                }).record(
                    "PRODUCTION_COMPLETE" if report["status"] == "COMPLETE" else "PRODUCTION_DRAFT",
                    phase=ProducerPhase.COMPLETE if report["status"] == "COMPLETE" else ProducerPhase.DRAFT,
                    detail="Delivery gates checked against project, audio, actions and critique",
                    payload={"reasons": report["blockers"]},
                )
            )
        return report
    except Exception as exc:
        if goal is None:
            raise
        report["status"] = "DRAFT" if report["MUSICAL_WRITES"] else "BLOCKED"
        report["blockers"] = [f"{type(exc).__name__}: {exc}"]
        if producer_state is not None:
            persist_producer_state(
                producer_state.model_copy(update={
                    "pending_issues": report["blockers"],
                    "stop_reason": report["blockers"][0],
                }).record(
                    "PRODUCTION_STOPPED",
                    phase=ProducerPhase.DRAFT if report["MUSICAL_WRITES"] else ProducerPhase.BLOCKED,
                    detail=str(exc)[:1000],
                )
            )
        return report
    finally:
        try:
            report["finished_at"] = now_iso()
            artifact_path.write_text(
                json.dumps(_jsonable(report), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception as exc:  # artifact persistence must not hide run status
            report["artifact_error"] = f"{type(exc).__name__}: {exc}"
        finally:
            daw.disconnect()


def main() -> int:
    report = run_alpha()
    print(json.dumps(_jsonable(report), indent=2, ensure_ascii=False))
    return 0 if str(report.get("status", "")).startswith(("VERIFIED", "PRODUCTION_PASS_VERIFIED")) else 2


if __name__ == "__main__":
    raise SystemExit(main())
