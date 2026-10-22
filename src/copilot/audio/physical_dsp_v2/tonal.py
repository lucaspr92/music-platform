"""TONAL FACTS: chroma, key candidates, tonality confidence. Never one certain key."""

from __future__ import annotations

from typing import Any

import numpy as np

from copilot.audio.physical_dsp_v2.audio import AudioBuffer
from copilot.audio.physical_dsp_v2.cache import cache_key, load_cached, store_cached
from copilot.audio.physical_dsp_v2.contract import ANALYZER_VERSION, KEY_CANDIDATE_COUNT, STFT_NPERSEG
from copilot.audio.physical_dsp_v2.observation import build_observation, merge_quality
from copilot.audio.physical_dsp_v2.providers import essentia_status, librosa_status, scipy_status
from copilot.audio.physical_dsp_v2.signal import chroma_vector, key_candidates, spectral_flatness, stft_power
from copilot.schemas.dsp import (
    AnalyzerFamily,
    DspLimitation,
    DspQuality,
    DspSubject,
    MeasuredValue,
    TimeSpan,
)


def analyze_tonal(
    buffer: AudioBuffer,
    *,
    subject: DspSubject,
    time_span: TimeSpan,
    params: dict[str, Any] | None = None,
    use_cache: bool = True,
    cache_dir=None,
) -> Any:
    params = dict(params or {})
    params.setdefault("nperseg", STFT_NPERSEG)
    params.setdefault("key_candidate_count", KEY_CANDIDATE_COUNT)
    key = cache_key(buffer.artifact_hash, AnalyzerFamily.TONAL.value, ANALYZER_VERSION, {
        **params,
        "start_s": time_span.start_s,
        "end_s": time_span.end_s,
        "granularity": time_span.granularity.value,
    })
    if use_cache:
        hit = load_cached(key, cache_dir)
        if hit is not None:
            return hit

    limitations = [
        DspLimitation("KEY_IS_CANDIDATE_SET_NOT_CERTAIN", "Never collapse to one certain key."),
        DspLimitation("CHROMA_FROM_STFT_KRUMHANSL_SCHMUCKLER", "STFT chroma + KS profiles. Not a licensed key model."),
        essentia_status().limitation,
        librosa_status().limitation,
    ]
    quality = DspQuality.OK
    sc = scipy_status()
    if not sc.available:
        return build_observation(
            observation_id=key,
            analyzer_id=AnalyzerFamily.TONAL.value,
            subject=subject,
            time_span=time_span,
            values=[],
            quality=DspQuality.UNSUPPORTED,
            limitations=[item for item in (sc.limitation, *limitations) if item is not None],
            provider_id="scipy",
            provider_version=sc.version,
            method="unavailable",
            params=params,
            source_artifact_hash=buffer.artifact_hash,
            cache_key=key,
        )

    freqs, times, power, stft_lims, stft_q = stft_power(buffer.mono, buffer.sample_rate, nperseg=params["nperseg"])
    limitations.extend(stft_lims)
    quality = merge_quality(quality, stft_q)
    chroma = chroma_vector(freqs, power, buffer.sample_rate)
    candidates = key_candidates(chroma, top_n=int(params["key_candidate_count"]))
    if not candidates:
        limitations.append(DspLimitation("NO_TONAL_ENERGY", "chroma energy is zero"))
        quality = merge_quality(quality, DspQuality.LIMITED)
        tonality = 0.0
        salience = 0.0
    else:
        best = float(candidates[0]["score"])
        second = float(candidates[1]["score"]) if len(candidates) > 1 else 0.0
        tonality = float((best - second) / (abs(best) + 1e-9))
        salience = float(np.max(chroma) / (np.mean(chroma) + 1e-12)) if float(np.sum(chroma)) else 0.0

    flux = 0.0
    if power.ndim == 2 and power.shape[1] >= 2:
        chroma_frames = []
        for i in range(power.shape[1]):
            chroma_frames.append(chroma_vector(freqs, power[:, i : i + 1], buffer.sample_rate))
        mat = np.vstack(chroma_frames)
        diffs = np.linalg.norm(np.diff(mat, axis=0), axis=1)
        flux = float(np.mean(diffs)) if len(diffs) else 0.0
    flat = spectral_flatness(power)
    tonal_likelihood = None if flat is None else float(max(0.0, min(1.0, 1.0 - flat)))
    if flat is None:
        limitations.append(DspLimitation("TONAL_LIKELIHOOD_FLATNESS_UNRESOLVED", ""))
        quality = merge_quality(quality, DspQuality.LIMITED)

    values = [
        MeasuredValue("chroma", {str(i): float(chroma[i]) for i in range(12)}, unit="ratio"),
        MeasuredValue("pitch_salience", salience, unit="ratio", confidence=None if candidates else 0.0),
        MeasuredValue("key_candidates", candidates),
        MeasuredValue("tonality_confidence", tonality, unit="ratio", confidence=tonality if candidates else 0.0),
        MeasuredValue("harmonic_change", flux, unit="ratio"),
        MeasuredValue("spectral_flatness", flat, unit="ratio"),
        MeasuredValue("tonal_likelihood", tonal_likelihood, unit="ratio"),
        MeasuredValue("non_tonal_likelihood", None if tonal_likelihood is None else float(1.0 - tonal_likelihood), unit="ratio"),
        MeasuredValue("selected_key", None),
    ]
    obs = build_observation(
        observation_id=key,
        analyzer_id=AnalyzerFamily.TONAL.value,
        subject=subject,
        time_span=time_span,
        values=values,
        quality=quality,
        limitations=[item for item in limitations if item is not None],
        provider_id="numpy+scipy",
        provider_version=sc.version,
        method="stft_chroma+krumhansl_schmuckler_candidates",
        params=params,
        source_artifact_hash=buffer.artifact_hash,
        cache_key=key,
    )
    if use_cache:
        store_cached(obs, cache_dir)
    return obs
