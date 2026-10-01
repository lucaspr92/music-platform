"""Stdio MCP server exposing both integrations without launching Live or a model."""

import argparse
import json
import logging
import sys
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

from copilot.daw.adapter import DawError
from copilot.mcp.contracts import DeviceSelection, ServerConfig, SessionExpectation
from copilot.mcp.service import MusicPlatformService
from copilot.producer.preset_catalog import PresetSelectionRequest
from copilot.schemas.musicplan import MusicPlan

if TYPE_CHECKING:
    from mcp.server import MCPServer


INSTRUCTIONS = """Use these MCP tools directly; do not issue shell or raw bridge commands.
AbletonMCP and Serum 2/Soniq share the existing Core write authority.
Start with runtime_status and ableton_get_session. Copy its expectation and
stable IDs, references and target tokens; never invent identities or freshness.
Writes require a human-configured --allow-writes and protected --working-copy.
Use ableton_apply_plan for supported MusicPlan operations and
soniq_set_parameter for parameters actually exposed by Ableton.
Soniq WebSocket reads are external observations, never write preconditions.
Unsupported actions defer explicitly. Never retry an IN_DOUBT write.
No tool renders, captures, claims musical improvement or certifies a finished ALS.
"""


def create_server(config: ServerConfig) -> "MCPServer[Any]":
    try:
        from mcp.server import MCPServer
        from mcp.server.mcpserver.exceptions import ToolError
        from mcp_types import CallToolResult, TextContent, ToolAnnotations
    except ImportError as exc:
        raise RuntimeError(
            'MCP_SDK2_REQUIRED: from the repository root run '
            'python -m pip install -e ".[mcp,soniq]"',
        ) from exc

    service = MusicPlatformService(config)

    @asynccontextmanager
    async def lifespan(_server: "MCPServer[Any]") -> AsyncIterator[None]:
        try:
            yield
        finally:
            service.close()

    server = MCPServer(
        "music-platform", title="AbletonMCP + Serum 2 / Soniq",
        instructions=INSTRUCTIONS, log_level="WARNING", lifespan=lifespan,
    )
    read_only = ToolAnnotations(
        read_only_hint=True, destructive_hint=False,
        idempotent_hint=True, open_world_hint=False,
    )
    write = ToolAnnotations(
        read_only_hint=False, destructive_hint=True,
        idempotent_hint=False, open_world_hint=False,
    )

    def invoke(call: Callable[[], dict[str, Any]]) -> dict[str, Any]:
        try:
            payload = call()
            json.dumps(payload, allow_nan=False)
            return payload
        except (DawError, ValueError, OSError, RuntimeError) as exc:
            raise ToolError(str(exc)) from exc

    def mutation(call: Callable[[], dict[str, Any]]) -> CallToolResult:
        payload = invoke(call)
        return CallToolResult(
            content=[TextContent(type="text", text=json.dumps(payload, allow_nan=False))],
            structured_content=payload, is_error=not payload["ok"],
        )

    @server.tool(annotations=read_only)
    def runtime_status() -> dict[str, Any]:
        """Read local MCP policy/configuration. Does not connect to Live or Soniq."""
        return service.status()

    @server.tool(annotations=read_only)
    def ableton_get_session(include_notes: bool = False) -> dict[str, Any]:
        """Read Live metadata, capabilities, stable targets and current tokens.

        Notes are opt-in; plugin values are read separately with soniq_read_surface.
        Omitted details are not empty clips or empty plugin surfaces.
        """
        return invoke(lambda: service.get_session(include_notes=include_notes))

    @server.tool(annotations=read_only)
    def ableton_get_arrangement() -> dict[str, Any]:
        """Read existing Arrangement clips without playing, capturing or changing them."""
        return invoke(service.get_arrangement)

    @server.tool(annotations=read_only)
    def ableton_search_browser(query: str, category: str = "all") -> dict[str, Any]:
        """Search the existing Live browser. Results are metadata, not sample permission."""
        return invoke(lambda: service.search_browser(query, category))

    @server.tool(annotations=write)
    def ableton_apply_plan(plan: MusicPlan, expected: SessionExpectation) -> CallToolResult:
        """Apply a supported typed MusicPlan through Core/SafeWrite, with exact readback.

        Copy tokens/ref data from ableton_get_session. Core supports single certified
        actions and its existing MIDI compound shapes, not arbitrary mixed batches.
        Device URIs require a local preset catalog; samples require a permission context.
        Failure and IN_DOUBT return MCP errors with the full Core result; do not retry.
        """
        return mutation(lambda: service.apply_plan(plan, expected))

    @server.tool(annotations=read_only)
    def soniq_read_surface(
        selection: DeviceSelection, offset: int = 0, limit: int = 100,
    ) -> dict[str, Any]:
        """Read Serum/plugin parameters exposed by Ableton, with completeness and pagination."""
        return invoke(lambda: service.read_surface(selection, offset=offset, limit=limit))

    @server.tool(annotations=read_only)
    def soniq_read_full_surface(
        selection: DeviceSelection, offset: int = 0, limit: int = 100,
    ) -> dict[str, Any]:
        """Read the existing local SONIQ_WS_URL bridge. Never authorizes parameter writes."""
        return invoke(lambda: service.read_full_surface(selection, offset=offset, limit=limit))

    @server.tool(annotations=write)
    def soniq_set_parameter(
        selection: DeviceSelection, expected: SessionExpectation,
        parameter_index: int, parameter_name: str,
        expected_before: float, value: float, reason: str,
    ) -> CallToolResult:
        """Set one uniquely exposed Ableton plugin parameter in its native range through Core.

        Use stable device IDs and Ableton's matching name/index/current value,
        not a Soniq-only index.
        Full Soniq WS patch/preset writes remain uncertified and are not exposed.
        """
        return mutation(lambda: service.set_parameter(
            selection, expected, parameter_index=parameter_index,
            parameter_name=parameter_name, expected_before=expected_before,
            value=value, reason=reason,
        ))

    @server.tool(annotations=read_only)
    def soniq_select_presets(request: PresetSelectionRequest) -> dict[str, Any]:
        """Select compatible presets from the locally configured catalog; does not load them."""
        return invoke(lambda: service.select_presets(request))

    @server.tool(annotations=read_only)
    def producer_get_context() -> dict[str, Any]:
        """Read configured reference evidence, preferences and permitted sample shortlists."""
        return invoke(service.producer_context)

    return server


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="copilot-mcp", description="Local AbletonMCP + Serum 2/Soniq tools over stdio.",
    )
    parser.add_argument("--allow-writes", action="store_true")
    parser.add_argument("--working-copy", type=Path)
    parser.add_argument("--state-dir", type=Path, default=Path("logs") / "mcp")
    parser.add_argument("--producer-context", type=Path)
    parser.add_argument("--preset-catalog", type=Path)
    parser.add_argument("--sample-root", type=Path)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, stream=sys.stderr, force=True)
    try:
        server = create_server(ServerConfig(
            allow_writes=args.allow_writes, working_copy=args.working_copy,
            state_dir=args.state_dir, producer_context=args.producer_context,
            preset_catalog=args.preset_catalog, sample_root=args.sample_root,
        ))
    except (ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    server.run(transport="stdio")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
