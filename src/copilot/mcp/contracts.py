"""Typed MCP requests; neither these contracts nor their validation contact Live."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from copilot.schemas.session import SessionState


@dataclass(frozen=True)
class ServerConfig:
    allow_writes: bool = False
    working_copy: Path | None = None
    state_dir: Path = Path("logs") / "mcp"
    producer_context: Path | None = None
    preset_catalog: Path | None = None
    sample_root: Path | None = None


class SessionExpectation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    project_identity: str = Field(min_length=1)
    session_incarnation_id: str = Field(min_length=1)
    project_token: str = Field(min_length=1)
    audible_token: str = Field(min_length=1)

    @classmethod
    def from_session(cls, session: SessionState) -> SessionExpectation:
        return cls.model_validate({
            field: getattr(session, field) for field in cls.model_fields
        })

    def require_current(self, session: SessionState) -> None:
        if self.project_identity != session.project_identity:
            raise ValueError("PROJECT_MISMATCH")
        if self.session_incarnation_id != session.session_incarnation_id:
            raise ValueError("SESSION_INCARNATION_MISMATCH")
        if (
            self.project_token != session.project_token
            or self.audible_token != session.audible_token
        ):
            raise ValueError("STALE_PLAN")


class DeviceSelection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    track_stable_id: str = Field(min_length=1)
    device_stable_id: str = Field(min_length=1)
