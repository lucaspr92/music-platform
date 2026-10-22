"""Groove fingerprint for library assets: measured once, stored in the index.

Essentia and librosa are OPT-IN extras, so nothing here can hard-fail an index run:
a machine without them still indexes samples, it just cannot fill the groove block.

Why it is stored, and how it stays correct:
  * content-addressed by the asset sha256 -> duplicates and moved files share one analysis
  * tempo: measured on 60 labelled loops, PercivalBpmEstimator beats multifeature (61.7% vs
    51.7% exact) and both crush librosa (20%) and the index autocorrelation (32%) on short
    loops, so short material uses Percival and long material keeps multifeature
  * the pack's own tempo label is recorded next to the measurement, never silently merged
  * versioned by GROOVE_VERSION -> a change in the method invalidates the cache on purpose
  * the genre block is TensorFlow inference, so its status is recorded (OK / SKIPPED /
    UNAVAILABLE / FAILED) and a machine without the models never retries the same file
  * DSP and genre are separate computations, so adding genre to an asset that already
    has its DSP fingerprint costs only the inference
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

GROOVE_VERSION = "groove-v2"  # v2: Percival for short material, label + method recorded
BANDS = ((60, 200), (200, 500), (500, 2000), (2000, 6000), (6000, 16000))
BAND_NAMES = ("kick_body", "drum_body", "snare_clap", "attack", "air")
MODELS_DIR = Path.home() / "music-platform-runtime" / "models" / "essentia"


def _label_bpm(filename: str) -> float | None:
    """The tempo the pack states in the filename ('095bpm', '128 BPM'), if any.

    Recorded as evidence next to the measurement: on 60 labelled loops no estimator
    got past 62% exact, so the label is the most trustworthy tempo available for a loop.
    """
    import re

    for token in re.findall(r"\d{3}", filename):
        value = float(token)
        if 60.0 <= value <= 190.0:
            return value
    return None


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def providers_available() -> tuple[bool, str]:
    try:
        import essentia  # noqa: F401
        import librosa  # noqa: F401
    except Exception as exc:  # pragma: no cover - depends on the host
        return False, f"{type(exc).__name__}: {exc}"
    return True, ""


def models_available() -> tuple[bool, str]:
    needed = ("discogs-effnet-bs64-1.pb", "genre_discogs400-discogs-effnet-1.pb",
              "genre_discogs400-discogs-effnet-1.json")
    missing = [name for name in needed if not (MODELS_DIR / name).is_file()]
    if missing:
        return False, "MODELS_MISSING: " + ", ".join(missing)
    return True, ""


def _band_share(signal: Any, sr: int, np: Any) -> dict[str, float]:
    spec = np.abs(np.fft.rfft(signal * np.hanning(len(signal))))
    freqs = np.fft.rfftfreq(len(signal), 1 / sr)
    power = spec ** 2
    total = float(power.sum()) or 1.0
    out: dict[str, float] = {}
    for name, (lo, hi) in zip(BAND_NAMES, BANDS):
        mask = np.logical_and(freqs >= lo, freqs < hi)
        energy = float(power[mask].sum()) or 1e-12
        out[name] = round(float(10 * np.log10(energy / total)), 2)
    return out


def _grid_profile(onset_times: Any, beats: Any, np: Any) -> dict:
    """Where hits land inside the beat: four sixteenth cells, normalised."""
    if onset_times.size < 4 or beats.size < 4:
        return {}
    period = float(np.median(np.diff(beats)))
    if period <= 0:
        return {}
    phase = ((onset_times - float(beats[0])) / period) % 1.0
    cell = np.minimum((phase * 4).astype(int), 3)
    hist = np.bincount(cell, minlength=4).astype(float)
    hist = hist / hist.sum()
    return {"sixteenths": [round(float(v), 3) for v in hist],
            "on_beat": round(float(hist[0]), 3),
            "offbeat_8th": round(float(hist[2]), 3),
            "syncopated_16ths": round(float(hist[1] + hist[3]), 3),
            "beat_period_s": round(period, 4)}


def compute_genre(path: str) -> tuple[str, list[dict]]:
    """TensorFlow genre inference. Returns (status, rows) and never raises."""
    ok, reason = models_available()
    if not ok:
        return "UNAVAILABLE", [{"error": reason}]
    try:
        import json

        import numpy as np
        import essentia.standard as es

        for attr in ("warningActive", "infoActive"):
            try:
                setattr(__import__("essentia").log, attr, False)
            except Exception:
                pass
        meta = json.loads((MODELS_DIR / "genre_discogs400-discogs-effnet-1.json").read_text(encoding="utf-8"))
        labels = meta.get("classes") or meta.get("labels") or []
        audio = es.MonoLoader(filename=str(path), sampleRate=16000)()
        embeddings = es.TensorflowPredictEffnetDiscogs(
            graphFilename=str(MODELS_DIR / "discogs-effnet-bs64-1.pb"), output="PartitionedCall:1")(audio)
        predictions = es.TensorflowPredict2D(
            graphFilename=str(MODELS_DIR / "genre_discogs400-discogs-effnet-1.pb"),
            input="serving_default_model_Placeholder", output="PartitionedCall:0")(embeddings)
        mean = np.mean(predictions, axis=0)
        order = np.argsort(mean)[::-1][:8]
        return "OK", [{"label": labels[i] if i < len(labels) else str(i), "p": round(float(mean[i]), 4)}
                      for i in order]
    except Exception as exc:
        return "FAILED", [{"error": f"{type(exc).__name__}: {exc}"}]


def compute_groove(path: str, *, with_genre: bool = False) -> dict:
    """DSP fingerprint (and optionally the genre block). Raises only on unreadable audio."""
    import numpy as np
    import soundfile as sf

    ok, reason = providers_available()
    if not ok:
        raise RuntimeError("GROOVE_PROVIDERS_UNAVAILABLE: " + reason)
    for attr in ("warningActive", "infoActive"):
        try:
            import essentia
            setattr(essentia.log, attr, False)
        except Exception:
            pass
    import essentia.standard as es
    import librosa

    data, sr = sf.read(str(path), always_2d=True, dtype="float32")
    mono = data.mean(axis=1).astype(np.float32)
    y, _ = librosa.load(str(path), sr=None, mono=True)
    duration = len(mono) / sr
    short_material = duration < 20.0
    if short_material:
        bpm = es.PercivalBpmEstimator()(mono)
        confidence = 0.0
        bpm_method = "essentia-percival"
    else:
        bpm, beats, confidence, _, _ = es.RhythmExtractor2013(method="multifeature")(mono)
        bpm_method = "essentia-multifeature"
    onsets, rate = es.OnsetRate()(mono)
    if short_material:
        _, beats = librosa.beat.beat_track(y=y, sr=sr, units="time")
        beats = np.asarray(beats, dtype=float)
    key, scale, strength = es.KeyExtractor()(mono)
    harmonic, percussive = librosa.effects.hpss(y, margin=3.0)
    rms = lambda signal: float(np.sqrt(np.mean(signal ** 2)))  # noqa: E731

    row = {
        "version": GROOVE_VERSION,
        "bpm": round(float(bpm), 2), "bpm_method": bpm_method, "bpm_label": _label_bpm(Path(path).name),
        "beat_confidence": round(float(confidence), 3),
        "onsets": int(len(onsets)), "onsets_per_s": round(float(rate), 2),
        "percussive_over_harmonic": round(rms(percussive) / (rms(harmonic) + 1e-12), 3),
        "percussive_band_share_db": _band_share(percussive, sr, np),
        "grid": _grid_profile(np.asarray(onsets, dtype=float), np.asarray(beats, dtype=float), np),
        "key": f"{key} {scale}", "key_strength": round(float(strength), 3),
        "clipping": bool(float(np.max(np.abs(data))) >= 1.0),
        "computed_at": _utc(),
    }
    if with_genre:
        status, rows = compute_genre(path)
        row["genre_status"] = status
        row["genre_discogs400"] = rows
    else:
        row["genre_status"] = "SKIPPED"
        row["genre_discogs400"] = []
    return row


def attach_groove(
    index_path: Path,
    *,
    only_shas: set[str] | None = None,
    with_genre: bool = False,
    compute: Callable[..., dict] | None = None,
    compute_genre_fn: Callable[[str], tuple[str, list[dict]]] | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    """Fill missing or stale groove fingerprints in the index and save it once.

    Re-runs are cheap by construction: a fresh fingerprint is REUSED, and asking for
    genre on an asset that already has DSP only runs the inference.
    """
    from copilot.sample_library.library_v1 import load_index, save_index
    from copilot.sample_library.schemas import GrooveFingerprint

    index = load_index(Path(index_path))
    if index is None:
        return {"status": "NO_INDEX", "index_path": str(index_path)}
    if compute is None:
        ok, reason = providers_available()
        if not ok:
            return {"status": "PROVIDERS_UNAVAILABLE", "reason": reason, "index_path": str(index_path)}
    fn = compute or compute_groove
    genre_fn = compute_genre_fn or compute_genre

    targets = [(sha, asset) for sha, asset in index.assets.items()
               if asset.status == "INDEXED" and (only_shas is None or sha in only_shas)]
    counts = {"COMPUTED": 0, "REUSED": 0, "GENRE_ONLY": 0, "GENRE_UNAVAILABLE": 0, "FAILED": 0}
    for done, (sha, asset) in enumerate(targets, start=1):
        stored = asset.groove
        fresh = stored is not None and stored.version == GROOVE_VERSION
        needs_genre = with_genre and (not fresh or stored.genre_status in {"SKIPPED", "FAILED"})
        if fresh and not needs_genre:
            counts["REUSED"] += 1
            if progress:
                progress(done, len(targets))
            continue
        try:
            if fresh:
                status, rows = genre_fn(asset.path)
                asset.groove = stored.model_copy(update={
                    "genre_status": status, "genre_discogs400": rows, "computed_at": _utc()})
                counts["GENRE_ONLY" if status == "OK" else "GENRE_UNAVAILABLE"] += 1
            else:
                row = fn(asset.path, with_genre=with_genre)
                asset.groove = GrooveFingerprint(**row)
                counts["COMPUTED"] += 1
        except Exception as exc:
            counts["FAILED"] += 1
            if stored is None:
                asset.groove = GrooveFingerprint(version=GROOVE_VERSION, genre_status="FAILED",
                                                 genre_discogs400=[{"error": f"{type(exc).__name__}: {exc}"}])
        if progress:
            progress(done, len(targets))
    index.groove_version = GROOVE_VERSION
    save_index(index, Path(index_path))
    return {"status": "ATTACHED", "index_path": str(index_path), "targets": len(targets),
            "groove_version": GROOVE_VERSION, **counts}
