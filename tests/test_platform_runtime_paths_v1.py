"""Platform-owned runtime/plugin locations keep the historical layout."""

from __future__ import annotations

import os
from pathlib import Path

from copilot.platform.plugin_paths import (
    WINDOWS_VST2_DIRECTORIES,
    WINDOWS_VST3_DIRECTORY,
    au_component_directories,
    vst2_plugin_directories,
    vst3_plugin_directories,
)
from copilot.platform.runtime_paths import (
    CORE_LIBRARY_VARIABLE,
    RUNTIME_ROOT_VARIABLE,
    WINDOWS_CORE_LIBRARY,
    WINDOWS_RUNTIME_ROOT,
    ableton_core_library,
    runtime_root,
)
from copilot.producer.serum2_catalog import default_plugin_candidates


def test_runtime_root_honours_environment_override(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv(RUNTIME_ROOT_VARIABLE, str(tmp_path / "runtime"))
    assert runtime_root() == tmp_path / "runtime"


def test_runtime_root_default_preserves_windows_layout(monkeypatch) -> None:
    monkeypatch.delenv(RUNTIME_ROOT_VARIABLE, raising=False)
    monkeypatch.setattr(os, "name", "nt")
    assert runtime_root() == WINDOWS_RUNTIME_ROOT


def test_runtime_root_default_is_portable_elsewhere(monkeypatch) -> None:
    monkeypatch.delenv(RUNTIME_ROOT_VARIABLE, raising=False)
    monkeypatch.setattr(os, "name", "posix")
    assert runtime_root() == Path.home() / "music-platform-runtime"


def test_core_library_honours_environment_override(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv(CORE_LIBRARY_VARIABLE, str(tmp_path / "Core Library"))
    assert ableton_core_library() == tmp_path / "Core Library"


def test_core_library_default_preserves_windows_layout(monkeypatch) -> None:
    monkeypatch.delenv(CORE_LIBRARY_VARIABLE, raising=False)
    monkeypatch.setattr(os, "name", "nt")
    assert ableton_core_library() == WINDOWS_CORE_LIBRARY


def test_core_library_default_is_absolute_elsewhere(monkeypatch) -> None:
    monkeypatch.delenv(CORE_LIBRARY_VARIABLE, raising=False)
    monkeypatch.setattr(os, "name", "posix")
    resolved = ableton_core_library()
    assert resolved.is_absolute()
    assert resolved.name == "Core Library"


def test_plugin_directories_follow_the_host(monkeypatch) -> None:
    monkeypatch.setattr(os, "name", "nt")
    assert vst3_plugin_directories() == [WINDOWS_VST3_DIRECTORY]
    assert vst2_plugin_directories() == list(WINDOWS_VST2_DIRECTORIES)
    assert au_component_directories() == []

    monkeypatch.setattr(os, "name", "posix")
    assert vst3_plugin_directories()[0] == Path("/Library/Audio/Plug-Ins/VST3")
    assert vst2_plugin_directories()[0] == Path("/Library/Audio/Plug-Ins/VST")
    assert au_component_directories()[0] == Path("/Library/Audio/Plug-Ins/Components")


def test_serum_candidates_keep_platform_install_locations() -> None:
    paths = [str(path) for _plugin, _format, path in default_plugin_candidates()]
    assert str(WINDOWS_VST3_DIRECTORY / "Serum2.vst3") in paths
    assert str(WINDOWS_VST3_DIRECTORY / "Serum 2.vst3") in paths
    assert str(WINDOWS_VST2_DIRECTORIES[0] / "Serum.dll") in paths
    assert str(WINDOWS_VST2_DIRECTORIES[1] / "Serum.dll") in paths
