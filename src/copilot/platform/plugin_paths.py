"""Host-owned plugin installation directories.

Plugin discovery needs the real VST3/VST/AU folders of a machine, and those
locations are platform knowledge. They live here so producer modules ask for a
path instead of carrying drive letters; entries are data only, and callers test
existence themselves.
"""

from __future__ import annotations

import os
from pathlib import Path

WINDOWS_VST3_DIRECTORY = Path(r"C:\Program Files\Common Files\VST3")
WINDOWS_VST2_DIRECTORIES = (
    Path(r"C:\Program Files\VstPlugins"),
    Path(r"C:\Program Files\Steinberg\VstPlugins"),
)
MACOS_SYSTEM_VST3_DIRECTORY = Path("/Library/Audio/Plug-Ins/VST3")
MACOS_SYSTEM_VST2_DIRECTORY = Path("/Library/Audio/Plug-Ins/VST")
MACOS_SYSTEM_AU_DIRECTORY = Path("/Library/Audio/Plug-Ins/Components")


def is_windows() -> bool:
    return os.name == "nt"


def vst3_plugin_directories() -> list[Path]:
    """VST3 directories for this host, system scope before user scope."""
    if is_windows():
        return [WINDOWS_VST3_DIRECTORY]
    home = Path.home()
    return [
        MACOS_SYSTEM_VST3_DIRECTORY,
        home / "Library" / "Audio" / "Plug-Ins" / "VST3",
    ]


def vst2_plugin_directories() -> list[Path]:
    """VST2 directories for this host, system scope before user scope."""
    if is_windows():
        return list(WINDOWS_VST2_DIRECTORIES)
    home = Path.home()
    return [
        MACOS_SYSTEM_VST2_DIRECTORY,
        home / "Library" / "Audio" / "Plug-Ins" / "VST",
    ]


def au_component_directories() -> list[Path]:
    """Audio Unit component directories for this host (macOS only)."""
    if is_windows():
        return []
    home = Path.home()
    return [
        MACOS_SYSTEM_AU_DIRECTORY,
        home / "Library" / "Audio" / "Plug-Ins" / "Components",
    ]
