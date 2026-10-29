"""The replication score must not reward level or mix.

These tests encode the failure of the previous scorer: it moved +0.037 when the render was only
mastered, with no musical change. A replication score that cannot pass these checks is not usable
for a replication-first workflow.
"""
import numpy as np
import pytest

from scripts.similarity_replication import (bar_and_section_grid, score_element,
                                            sixteenth_grid)

SR = 44100
BPM = 128.0
BAR_S = 4 * 60.0 / BPM
TOTAL_BARS = 8
SECTIONS = [(0, 4), (4, 8)]


def pattern_signal(shift_sixteenths: int = 0, gain: float = 1.0) -> np.ndarray:
    """Four-on-the-floor kick: an impulse on 16ths 1, 5, 9, 13 of every bar."""
    total = int(TOTAL_BARS * BAR_S * SR) + SR
    buffer = np.zeros(total, dtype=np.float32)
    six_s = BAR_S / 16.0
    click = np.hanning(400).astype(np.float32)
    for bar in range(TOTAL_BARS):
        for beat in range(4):
            step = (beat * 4 + shift_sixteenths) % 16
            start = int((bar * BAR_S + step * six_s) * SR)
            buffer[start:start + len(click)] += click * gain
    return buffer


def test_identical_signal_scores_one():
    signal = pattern_signal()
    result = score_element(signal, signal, SR, BAR_S, TOTAL_BARS, SECTIONS)
    assert result["score"] == pytest.approx(1.0, abs=1e-6)
    assert result["presence"] == 1.0
    assert result["pattern"] == 1.0


def test_level_change_does_not_change_the_score():
    signal = pattern_signal()
    quiet = pattern_signal(gain=0.35)
    loud = score_element(signal, signal, SR, BAR_S, TOTAL_BARS, SECTIONS)
    soft = score_element(signal, quiet, SR, BAR_S, TOTAL_BARS, SECTIONS)
    assert soft["score"] == pytest.approx(loud["score"], abs=1e-6), (
        "a level change must not move the replication score: that was the old scorer's bug")


def test_a_shifted_pattern_loses_pattern_agreement():
    signal = pattern_signal()
    shifted = pattern_signal(shift_sixteenths=2)
    result = score_element(signal, shifted, SR, BAR_S, TOTAL_BARS, SECTIONS)
    assert result["pattern"] < 0.95
    assert result["score"] < 1.0


def test_noise_scores_low():
    rng = np.random.default_rng(7)
    signal = pattern_signal()
    noise = rng.normal(0, 0.05, signal.size).astype(np.float32)
    result = score_element(signal, noise, SR, BAR_S, TOTAL_BARS, SECTIONS)
    assert result["score"] < 0.5, result


def test_sixteenth_grid_counts_a_four_on_the_floor():
    grid = sixteenth_grid(pattern_signal(), SR, BAR_S, TOTAL_BARS)
    loud = {index for index, value in enumerate(grid) if value > 0}
    assert loud == {0, 4, 8, 12}, grid


def test_plan_grid_falls_back_to_one_section():
    plan = {"bar_s": BAR_S, "total_bars": 8}
    bar_s, total_bars, sixteenths, sections = bar_and_section_grid(plan)
    assert bar_s == BAR_S and total_bars == 8 and sixteenths == 16
    assert sections == [(0, 8)]
