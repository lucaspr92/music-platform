"""The similarity metric must be able to say 1.0 and must be able to say 'nothing alike'.

A metric that scores everything high is worse than no metric: it certifies work that is not
done. The white-noise control exists for that reason.
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
REFERENCE = Path.home() / "music-platform-runtime/stems-src/reference_tudm_44100_2ch.wav"
NOISE = ROOT / "logs" / "noise_control.wav"

sys.path.insert(0, str(SCRIPTS))
import similarity_to_reference as sim  # noqa: E402


def test_identical_distributions_score_one():
    assert sim.l1_to_score([1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]) == 1.0
    assert sim.l1_to_score({"a": 1.0}, {"a": 1.0}) == 1.0


def test_disjoint_distributions_score_zero():
    assert sim.l1_to_score([1.0, 0.0], [0.0, 1.0]) == 0.0
    assert sim.l1_to_score({"a": 1.0, "b": 0.0}, {"a": 0.0, "b": 1.0}) == 0.0


def test_scores_are_bounded_and_monotone():
    near = sim.l1_to_score([0.5, 0.5, 0.0], [0.5, 0.4, 0.1])
    far = sim.l1_to_score([0.5, 0.5, 0.0], [0.0, 0.0, 1.0])
    assert 0.0 <= far < near <= 1.0


def test_bandpass_keeps_the_band_and_rejects_the_rest():
    import numpy as np
    sr = 44100
    t = np.arange(sr) / sr
    inside = np.sin(2 * np.pi * 80 * t).astype(np.float32)      # kick band
    outside = np.sin(2 * np.pi * 8000 * t).astype(np.float32)   # air band
    filtered_inside = sim.band_filter(inside, 40, 120)
    filtered_outside = sim.band_filter(outside, 40, 120)
    assert np.sqrt(np.mean(filtered_inside ** 2)) > 10 * np.sqrt(np.mean(filtered_outside ** 2))


@pytest.mark.skipif(not REFERENCE.is_file(), reason="reference audio not available")
def test_reference_against_itself_is_perfect():
    out = ROOT / "logs" / "similarity_test_sanity.json"
    subprocess.run([sys.executable, str(SCRIPTS / "similarity_to_reference.py"),
                    "--candidate", str(REFERENCE), "--json", str(out)],
                   check=True, capture_output=True)
    report = json.loads(out.read_text())
    assert report["total"] == 1.0
    for name, value in report["dimensions"].items():
        if value is not None:
            assert value == 1.0, f"{name} should be 1.0 against itself"


@pytest.mark.skipif(not (REFERENCE.is_file() and NOISE.is_file()),
                    reason="needs reference and noise control")
def test_white_noise_scores_low():
    out = ROOT / "logs" / "similarity_test_noise.json"
    subprocess.run([sys.executable, str(SCRIPTS / "similarity_to_reference.py"),
                    "--candidate", str(NOISE), "--json", str(out)],
                   check=True, capture_output=True)
    report = json.loads(out.read_text())
    assert report["total"] < 0.45, "a metric that scores noise highly cannot guide anything"
