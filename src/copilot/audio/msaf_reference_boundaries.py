"""Optional, locally isolated MSAF section-boundary comparison."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import soundfile as sf

from copilot.schemas.music_analysis import MusicAnalysisPack

_SUPPORTED_AUDIO = {".wav", ".flac", ".aif", ".aiff"}
_MAX_SOURCE_BYTES = 512 * 1024 * 1024
_SCRATCH_MARGIN_BYTES = 512 * 1024 * 1024
_MAX_REGION_SECONDS = 120.0
_MSAF_WORKER = """
import json
import sys
import msaf

times, _ = msaf.process(
    sys.argv[1], feature="mfcc", framesync=True,
    boundaries_id="foote", labels_id=None, hier=False,
    plot=False, sonify_bounds=False,
)
with open(sys.argv[2], "w", encoding="utf-8") as output:
    json.dump(
        {"version": msaf.__version__, "seconds": [float(t) for t in times]},
        output, allow_nan=False,
    )
"""


def _copy_region(source: Path, destination: Path, *, pack: MusicAnalysisPack) -> float:
    start_qn = float(pack.timeline["start_qn"])
    end_qn = float(pack.timeline["end_qn"])
    if not (
        math.isfinite(pack.tempo_bpm) and pack.tempo_bpm > 0
        and math.isfinite(start_qn) and math.isfinite(end_qn) and end_qn > start_qn
    ):
        raise ValueError("MSAF_REFERENCE_TIMELINE_INVALID")
    if (end_qn - start_qn) * 60.0 / pack.tempo_bpm > _MAX_REGION_SECONDS:
        raise ValueError("MSAF_REFERENCE_REGION_TOO_LONG")
    if pack.mode == "SELECTED_REGION":
        region = pack.provenance.get("source_region")
        if not isinstance(region, dict):
            raise ValueError("MSAF_SOURCE_REGION_MISMATCH")
        try:
            region_matches = all(
                math.isclose(float(region[key]), value, abs_tol=1e-6)
                for key, value in (("start_qn", start_qn), ("end_qn", end_qn))
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("MSAF_SOURCE_REGION_MISMATCH") from exc
        if not region_matches:
            raise ValueError("MSAF_SOURCE_REGION_MISMATCH")
    elif pack.mode != "WHOLE_TRACK":
        raise ValueError("MSAF_REFERENCE_MODE_UNSUPPORTED")
    with sf.SoundFile(source) as audio:
        sample_rate = audio.samplerate
        if sample_rate <= 0 or audio.frames <= 0:
            raise ValueError("MSAF_REFERENCE_AUDIO_INVALID")
        start_frame = round(start_qn * 60.0 / pack.tempo_bpm * sample_rate)
        end_frame = round(end_qn * 60.0 / pack.tempo_bpm * sample_rate)
        if pack.mode == "WHOLE_TRACK" and start_qn != 0:
            raise ValueError("MSAF_WHOLE_TRACK_ORIGIN_INVALID")
        if start_frame < 0 or end_frame > audio.frames + 1 or end_frame <= start_frame:
            raise ValueError("MSAF_REFERENCE_REGION_OUTSIDE_AUDIO")
        audio.seek(start_frame)
        remaining = min(end_frame, audio.frames) - start_frame
        duration_s = remaining / sample_rate
        with sf.SoundFile(
            destination, mode="w", samplerate=sample_rate,
            channels=audio.channels, subtype="FLOAT",
        ) as staged:
            while remaining > 0:
                chunk = audio.read(min(remaining, 65536), dtype="float32", always_2d=True)
                if not len(chunk):
                    raise ValueError("MSAF_REFERENCE_AUDIO_TRUNCATED")
                staged.write(chunk)
                remaining -= len(chunk)
    return duration_s


def analyze_msaf_reference_boundaries(
    pack: MusicAnalysisPack,
    *,
    python_executable: Path,
    scratch_root: Path,
) -> dict[str, Any]:
    """Run MSAF/Foote in an isolated local Python on a disposable audio copy.

    MSAF writes features and estimations alongside its input. Staging under a
    private ``audio`` child keeps both those writes and its random seed away
    from the user's original file and the running producer process.
    """
    source_path = pack.provenance.get("audio_path")
    expected_sha = pack.provenance.get("audio_sha256")
    if not source_path or not isinstance(expected_sha, str) or len(expected_sha) != 64:
        raise ValueError("MSAF_SOURCE_PROVENANCE_REQUIRED")
    source = Path(source_path).resolve(strict=True)
    if not source.is_file() or source.suffix.lower() not in _SUPPORTED_AUDIO:
        raise ValueError("MSAF_SOURCE_FORMAT_UNSUPPORTED")
    if source.stat().st_size > _MAX_SOURCE_BYTES:
        raise ValueError("MSAF_SOURCE_TOO_LARGE")
    with source.open("rb") as audio_file:
        actual_sha = hashlib.file_digest(audio_file, "sha256").hexdigest()
    if actual_sha != expected_sha:
        raise ValueError("MSAF_SOURCE_DIGEST_MISMATCH")
    interpreter = Path(python_executable).resolve(strict=True)
    if not interpreter.is_file():
        raise ValueError("MSAF_PYTHON_UNAVAILABLE")
    try:
        start_qn = float(pack.timeline["start_qn"])
        end_qn = float(pack.timeline["end_qn"])
        timeline_tempo = float(pack.timeline["tempo_bpm"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("MSAF_REFERENCE_TIMELINE_INVALID") from exc
    if (
        not all(math.isfinite(value) for value in (start_qn, end_qn, timeline_tempo))
        or start_qn < 0 or end_qn <= start_qn
        or not math.isclose(timeline_tempo, pack.tempo_bpm, abs_tol=1e-6)
    ):
        raise ValueError("MSAF_REFERENCE_TIMELINE_INVALID")
    duration_s = (end_qn - start_qn) * 60.0 / pack.tempo_bpm
    if duration_s > _MAX_REGION_SECONDS:
        raise ValueError("MSAF_REFERENCE_REGION_TOO_LONG")
    scratch = Path(scratch_root).resolve(strict=True)
    with sf.SoundFile(source) as audio:
        staged_bytes = math.ceil(duration_s * audio.samplerate) * audio.channels * 4
    if not scratch.is_dir() or shutil.disk_usage(scratch).free < (
        source.stat().st_size + staged_bytes + _SCRATCH_MARGIN_BYTES
    ):
        raise ValueError("MSAF_SCRATCH_SPACE_INSUFFICIENT")

    with tempfile.TemporaryDirectory(prefix="reference-msaf-", dir=scratch) as temp:
        work = Path(temp)
        audio_dir = work / "audio"
        audio_dir.mkdir()
        staged_audio = audio_dir / "reference.wav"
        duration_s = _copy_region(source, staged_audio, pack=pack)
        with source.open("rb") as audio_file:
            if hashlib.file_digest(audio_file, "sha256").hexdigest() != expected_sha:
                raise ValueError("MSAF_SOURCE_CHANGED_DURING_STAGING")
        result_path = work / "result.json"
        environment = os.environ.copy()
        environment.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
        try:
            result = subprocess.run(
                [str(interpreter), "-I", "-c", _MSAF_WORKER, str(staged_audio), str(result_path)],
                cwd=work, capture_output=True, text=True, timeout=600, check=False,
                env=environment,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("MSAF_PROCESS_TIMEOUT") from exc
        if result.returncode != 0:
            raise RuntimeError(
                f"MSAF_PROCESS_FAILED:{result.returncode}:"
                f"{result.stderr[-2000:]}"
            )
        if not result_path.is_file():
            raise RuntimeError("MSAF_RESULT_MISSING")
        payload = json.loads(result_path.read_text(encoding="utf-8"))

    seconds = payload.get("seconds")
    if not isinstance(seconds, list) or len(seconds) < 2:
        raise ValueError("MSAF_BOUNDARIES_INVALID")
    times = [float(value) for value in seconds]
    if (
        any(not math.isfinite(t) for t in times)
        or any(a >= b for a, b in zip(times, times[1:]))
        or abs(times[0]) > 0.1
        or abs(times[-1] - duration_s) > 1.0
    ):
        raise ValueError("MSAF_BOUNDARIES_INVALID")
    boundaries = [start_qn + t * pack.tempo_bpm / 60.0 for t in times[1:-1]]
    if any(not start_qn < value < end_qn for value in boundaries):
        raise ValueError("MSAF_BOUNDARIES_OUTSIDE_REGION")
    return {
        "boundaries_qn": boundaries,
        "provider": "msaf.foote.mfcc.framesync",
        "version": str(payload["version"]),
        "reference_state_token": pack.tokens.reference_state_token,
        "source_sha256": actual_sha,
        "limitations": [
            "MSAF structural boundaries are not instrument labels or musical quality judgments.",
            "An isolated local MSAF interpreter must be provisioned separately.",
        ],
        "no_write": True,
    }
