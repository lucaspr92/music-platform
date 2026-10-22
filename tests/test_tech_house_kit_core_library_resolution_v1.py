"""The kit's bank-relative paths must resolve on any host, not only Windows."""
from pathlib import Path

import pytest

from copilot.platform.runtime_paths import ableton_core_library
from copilot.producer import tech_house_kit_v1 as v1

# Live documents its factory content with Windows separators; on macOS a
# backslash is an ordinary filename character, which is why raw joins failed.
BACKSLASH = chr(92)


def test_both_separator_styles_resolve_to_the_same_file(tmp_path):
    posix = v1.resolve_relative(tmp_path, "Samples/One Shots/Drums")
    windows = v1.resolve_relative(tmp_path, BACKSLASH.join(("Samples", "One Shots", "Drums")))
    assert posix == windows
    assert posix == tmp_path / "Samples" / "One Shots" / "Drums"


def test_no_declared_sound_hides_a_separator_inside_a_filename(tmp_path):
    for item in v1.KIT:
        for relative in (item.relative_path, item.fallback_relative_path):
            resolved = v1.resolve_relative(tmp_path, relative)
            assert BACKSLASH not in resolved.name
            assert "/" not in resolved.name


@pytest.mark.skipif(not ableton_core_library().is_dir(), reason="Ableton Core Library not installed")
def test_every_declared_sound_exists_in_the_core_library():
    definition = v1.kit_definition(ableton_core_library())
    assert len(definition["sounds"]) == len(v1.KIT) == 7
    for row in definition["sounds"]:
        assert row["bytes"] > 0 and len(row["sha256"]) == 64
        assert BACKSLASH not in Path(row["path"]).name
