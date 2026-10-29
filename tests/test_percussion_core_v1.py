"""The percussive core must be plausible and must declare what was assumed.

A core that claims to be measured when it is not is worse than no core: it would let a guess
be read as evidence. The clap is the case in point.
"""
import json
from pathlib import Path

import pytest
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
SCRATCH = ROOT / "logs"
CORE = SCRATCH / "percussion_core.json"
STRUCTURE = SCRATCH / "reference_structure.json"


@pytest.fixture(scope="module")
def core():
    if not CORE.is_file():
        pytest.skip("run scripts/render_percussion_core.py first")
    return json.loads(CORE.read_text(encoding="utf-8"))


def test_kick_and_hats_are_measured_and_the_clap_is_declared_assumed(core):
    assert core["elements"]["kick"]["measured"] is True
    assert core["elements"]["hats"]["measured"] is True
    assert core["elements"]["clap"]["measured"] is False
    assert "ASSUMED" in core["clap_pattern"].upper()


def test_rates_are_musically_plausible(core):
    kick = core["elements"]["kick"]["per_bar"]
    hats = core["elements"]["hats"]["per_bar"]
    clap = core["elements"]["clap"]["per_bar"]
    assert 1.0 <= kick <= 6.0, f"a kick at {kick}/bar is not a plausible four-to-the-floor family"
    assert 4.0 <= hats <= 16.0, f"hats at {hats}/bar is outside a 16th-note ceiling"
    assert 1.0 <= clap <= 3.0, f"a backbeat at {clap}/bar is not a backbeat"


def test_core_wavs_match_the_bar_grid(core):
    structure = json.loads(STRUCTURE.read_text(encoding="utf-8"))
    expected = int(structure["total_bars"]) * float(structure["bar_s"])
    for element, info in core["elements"].items():
        path = Path(info["wav"])
        assert path.is_file(), f"{element}: missing {path}"
        frames, sr = sf.info(str(path)).frames, sf.info(str(path)).samplerate
        duration = frames / sr
        assert abs(duration - expected) < 3.0, f"{element}: {duration:.1f}s vs {expected:.1f}s"
