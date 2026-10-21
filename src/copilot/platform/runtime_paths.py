"""Host-owned runtime and Ableton resource locations.

Application code asks this module for a per-machine location instead of carrying
a drive letter or a user directory. The Windows defaults keep the historical
layout used by existing evidence paths, and both values accept an environment
override so a machine can relocate its runtime without a code edit.
"""

from __future__ import annotations

import os
from pathlib import Path

RUNTIME_ROOT_VARIABLE = "MUSIC_PLATFORM_RUNTIME"
CORE_LIBRARY_VARIABLE = "ABLETON_CORE_LIBRARY"

WINDOWS_RUNTIME_ROOT = Path(r"D:\music-platform-runtime")
WINDOWS_CORE_LIBRARY = Path(
    r"C:\ProgramData\Ableton\Live 12 Trial\Resources\Core Library"
)


def runtime_root() -> Path:
    """Per-machine root for renders, plans, reports and evidence."""
    override = os.environ.get(RUNTIME_ROOT_VARIABLE)
    if override:
        return Path(override).expanduser()
    if os.name == "nt":
        return WINDOWS_RUNTIME_ROOT
    return Path.home() / "music-platform-runtime"


def ableton_core_library() -> Path:
    """Ableton Core Library used to resolve native device presets."""
    override = os.environ.get(CORE_LIBRARY_VARIABLE)
    if override:
        return Path(override).expanduser()
    if os.name == "nt":
        return WINDOWS_CORE_LIBRARY
    return macos_core_library()


def macos_core_library() -> Path:
    """First installed Live bundle that exposes a Core Library."""
    for root in (Path("/Applications"), Path.home() / "Applications"):
        if not root.is_dir():
            continue
        for bundle in sorted(root.glob("Ableton Live *.app")):
            candidate = bundle / "Contents" / "App-Resources" / "Core Library"
            if candidate.is_dir():
                return candidate
    return Path(
        "/Applications/Ableton Live 12 Suite.app/Contents/App-Resources/Core Library"
    )
