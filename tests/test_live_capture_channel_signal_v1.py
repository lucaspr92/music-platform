"""A stereo capture with a silent channel must FAIL validation.

The bug this locks down: a file with 2 channels and a silent right channel passed every check,
because the validator looked at the channel COUNT and at the rms of the MONO downmix. Captures
like that silently poison loudness, width and correlation measurements.
"""
import numpy as np
import pytest
import soundfile as sf

from copilot.audio.live_capture import SILENCE_RMS, validate_wav

SR = 48000


def _write(path, audio):
    sf.write(str(path), audio.astype(np.float32), SR, subtype="FLOAT")
    return path


def test_stereo_with_a_silent_right_channel_fails(tmp_path):
    t = np.arange(SR) / SR
    tone = 0.3 * np.sin(2 * np.pi * 220 * t)
    audio = np.stack([tone, np.zeros_like(tone)], axis=1)
    path = _write(tmp_path / "silent_right.wav", audio)

    check = validate_wav(path, expected_sr=SR, expected_duration=1.0, require_signal=True)

    assert check.channels == 2
    assert not check.ok
    assert any("channel 1 is silent" in error for error in check.errors), check.errors
    assert check.channel_peak is not None and check.channel_peak[1] == pytest.approx(0.0)


def test_stereo_with_a_constant_channel_reports_undefined_correlation(tmp_path):
    t = np.arange(SR) / SR
    left = 0.3 * np.sin(2 * np.pi * 220 * t)
    audio = np.stack([left, np.full_like(left, 0.25)], axis=1)
    path = _write(tmp_path / "constant_right.wav", audio)

    check = validate_wav(path, expected_sr=SR, expected_duration=1.0, require_signal=True)

    assert not check.ok
    assert check.stereo_correlation is None
    assert any("constant" in error for error in check.errors), check.errors


def test_healthy_stereo_passes_and_reports_correlation(tmp_path):
    t = np.arange(SR) / SR
    left = 0.3 * np.sin(2 * np.pi * 220 * t)
    right = 0.25 * np.sin(2 * np.pi * 330 * t)
    path = _write(tmp_path / "healthy.wav", np.stack([left, right], axis=1))

    check = validate_wav(path, expected_sr=SR, expected_duration=1.0, require_signal=True)

    assert check.ok, check.errors
    assert check.channel_rms is not None and min(check.channel_rms) > SILENCE_RMS
    assert check.stereo_correlation is not None


def test_a_legitimately_quiet_capture_is_not_judged_when_signal_is_not_required(tmp_path):
    t = np.arange(SR) / SR
    tone = 0.3 * np.sin(2 * np.pi * 220 * t)
    audio = np.stack([tone, np.zeros_like(tone)], axis=1)
    path = _write(tmp_path / "silent_right.wav", audio)

    check = validate_wav(path, expected_sr=SR, expected_duration=1.0, require_signal=False)

    assert not any("silent" in error for error in check.errors), check.errors
