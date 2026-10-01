"""Pure contracts/SDK registration cases for Hermes; no mocked or real DAW calls."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from pydantic import ValidationError

from copilot.mcp.contracts import DeviceSelection, ServerConfig, SessionExpectation
from copilot.mcp.service import MusicPlatformService
from copilot.schemas.session import SessionState


def test_status_is_configuration_not_live_verification() -> None:
    service = MusicPlatformService(ServerConfig())
    status = service.status()
    assert status["status"] == "CONFIGURED_NOT_LIVE_VERIFIED"
    assert status["writes_enabled"] is False
    assert status["soniq_ws_writes"] is False
    assert status["musical_writes"] == 0
    assert service._daw is None


def test_write_opt_in_requires_protected_working_copy() -> None:
    with pytest.raises(ValueError, match="MCP_WORKING_COPY_REQUIRED"):
        MusicPlatformService(ServerConfig(allow_writes=True))
    with pytest.raises(ValueError, match="MCP_PROTECTED_WORKING_COPY_REQUIRED"):
        MusicPlatformService(ServerConfig(
            allow_writes=True, working_copy=Path("missing-project.als"),
        ))


def test_disabled_write_fails_without_backend_connection() -> None:
    service = MusicPlatformService(ServerConfig())
    with pytest.raises(ValueError, match="MCP_WRITES_DISABLED"):
        service.set_parameter(
            DeviceSelection(track_stable_id="t", device_stable_id="d"),
            SessionExpectation(
                project_identity="p", session_incarnation_id="i",
                project_token="project", audible_token="audible",
            ),
            parameter_index=0, parameter_name="Macro",
            expected_before=0.5, value=0.4, reason="request",
        )
    assert service._daw is None


@pytest.mark.parametrize(
    ("field", "error"),
    [
        ("project_identity", "PROJECT_MISMATCH"),
        ("session_incarnation_id", "SESSION_INCARNATION_MISMATCH"),
        ("project_token", "STALE_PLAN"),
        ("audible_token", "STALE_PLAN"),
    ],
)
def test_expectation_checks_identity_and_tokens(field: str, error: str) -> None:
    session = SessionState(
        project_identity="p", session_incarnation_id="i",
        project_token="project", audible_token="audible",
    )
    expected = SessionExpectation.from_session(session)
    expected.require_current(session)
    with pytest.raises(ValueError, match=error):
        expected.require_current(session.model_copy(update={field: "changed"}))


def test_unknown_fields_cannot_enable_writes() -> None:
    with pytest.raises(ValidationError):
        DeviceSelection.model_validate({
            "track_stable_id": "t", "device_stable_id": "d", "allow_writes": True,
        })


def test_real_sdk_registers_typed_tool_catalog_without_backend() -> None:
    pytest.importorskip("mcp.server.mcpserver")
    from copilot.mcp.server import create_server

    tools = asyncio.run(create_server(ServerConfig()).list_tools())
    assert len(tools) == 10
    mutations = {
        tool.name for tool in tools if not tool.annotations.read_only_hint
    }
    assert mutations == {"ableton_apply_plan", "soniq_set_parameter"}
    for tool in tools:
        assert tool.input_schema["type"] == "object"
        assert tool.annotations.open_world_hint is False
        _require_resolved_refs(tool.input_schema, tool.input_schema)


def _require_resolved_refs(node: object, root: dict) -> None:
    if isinstance(node, dict):
        if "$ref" in node:
            assert node["$ref"].startswith("#/")
            target = root
            for part in node["$ref"][2:].split("/"):
                target = target[part.replace("~1", "/").replace("~0", "~")]
        for value in node.values():
            _require_resolved_refs(value, root)
    elif isinstance(node, list):
        for value in node:
            _require_resolved_refs(value, root)
