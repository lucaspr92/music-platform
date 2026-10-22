"""The groove fingerprint is measured once and stored: re-runs must reuse it.

These tests inject the compute functions, so they assert the caching contract
without needing Essentia, librosa or TensorFlow on the machine.
"""
from pathlib import Path

import pytest

from copilot.sample_library.groove_v1 import GROOVE_VERSION, attach_groove, compute_groove
from copilot.sample_library.library_v1 import load_index, save_index
from copilot.sample_library.schemas import GrooveFingerprint, LibraryIndex, SampleAsset


def _index(tmp_path: Path, count: int = 2) -> Path:
    assets = {}
    for i in range(count):
        sha = f"{i:064d}"
        assets[sha] = SampleAsset(
            id=f"asset_{i}", path=f"/library/pack/loop {i}.wav", filename=f"loop {i}.wav",
            library_root="/library/pack", relative_path=f"loop {i}.wav", extension=".wav",
            size_bytes=100, sha256=sha,
        )
    path = tmp_path / "index.json"
    save_index(LibraryIndex(roots=["/library/pack"], assets=assets), path)
    return path


def _row(**over):
    row = {"version": GROOVE_VERSION, "bpm": 128.0, "beat_confidence": 0.8, "onsets": 10,
           "onsets_per_s": 5.0, "percussive_over_harmonic": 0.6,
           "percussive_band_share_db": {"drum_body": -4.8},
           "grid": {"sixteenths": [0.25, 0.25, 0.25, 0.25]}, "key": "D minor",
           "key_strength": 0.8, "clipping": False, "computed_at": "2026-01-01T00:00:00+00:00",
           "genre_status": "SKIPPED", "genre_discogs400": []}
    row.update(over)
    return row


def test_second_run_recomputes_nothing(tmp_path):
    path = _index(tmp_path)
    calls = []

    def fake_compute(file_path, *, with_genre=False):
        calls.append(file_path)
        return _row()

    first = attach_groove(path, compute=fake_compute)
    assert first["COMPUTED"] == 2 and first["REUSED"] == 0
    assert len(calls) == 2

    second = attach_groove(path, compute=fake_compute)
    assert second["REUSED"] == 2 and second["COMPUTED"] == 0
    assert len(calls) == 2, "a stored fingerprint must not be recomputed"

    index = load_index(path)
    assert index.groove_version == GROOVE_VERSION
    assert all(a.groove and a.groove.version == GROOVE_VERSION for a in index.assets.values())


def test_a_stale_version_is_recomputed(tmp_path):
    path = _index(tmp_path, count=1)
    index = load_index(path)
    sha = next(iter(index.assets))
    index.assets[sha].groove = GrooveFingerprint(version="groove-v0")
    save_index(index, path)

    calls = []
    report = attach_groove(path, compute=lambda p, **k: (calls.append(p), _row())[1])
    assert report["COMPUTED"] == 1 and len(calls) == 1


def test_genre_is_added_without_redoing_dsp(tmp_path):
    path = _index(tmp_path, count=1)
    dsp_calls, genre_calls = [], []

    report = attach_groove(path, compute=lambda p, **k: (dsp_calls.append(p), _row())[1])
    assert report["COMPUTED"] == 1 and dsp_calls

    report = attach_groove(
        path,
        with_genre=True,
        compute=lambda p, **k: (dsp_calls.append(p), _row())[1],
        compute_genre_fn=lambda p: (genre_calls.append(p), ("OK", [{"label": "Electronic---Tribal House", "p": 0.2}]))[1],
    )
    assert report["GENRE_ONLY"] == 1
    assert len(dsp_calls) == 1, "the DSP fingerprint was already stored"
    assert len(genre_calls) == 1
    index = load_index(path)
    assert next(iter(index.assets.values())).groove.genre_status == "OK"


def test_unavailable_models_are_not_retried_forever(tmp_path):
    path = _index(tmp_path, count=1)
    attach_groove(path, compute=lambda p, **k: _row())
    calls = []
    report = attach_groove(path, with_genre=True,
                           compute_genre_fn=lambda p: (calls.append(p), ("UNAVAILABLE", [{"error": "no models"}]))[1])
    assert report["GENRE_UNAVAILABLE"] == 1 and len(calls) == 1
    report = attach_groove(path, with_genre=True,
                           compute_genre_fn=lambda p: (calls.append(p), ("UNAVAILABLE", []))[1])
    assert len(calls) == 1, "UNAVAILABLE is a decision, not a retry loop"


@pytest.mark.skipif(not Path("/Volumes/Lucas/Samples").is_dir(), reason="local library not mounted")
def test_real_compute_returns_the_documented_shape(tmp_path):
    sample = next(p for p in Path("/Volumes/Lucas/Samples").rglob("*.wav"))
    row = compute_groove(str(sample))
    assert row["version"] == GROOVE_VERSION
    assert set(row["percussive_band_share_db"]) == {"kick_body", "drum_body", "snare_clap", "attack", "air"}
    assert row["grid"]["sixteenths"] and len(row["grid"]["sixteenths"]) == 4
    assert row["genre_status"] == "SKIPPED" and row["genre_discogs400"] == []
