"""Set element levels so the mix's spectral balance matches the reference, without clipping.

Each element is rendered alone to get its band-energy profile, then a non-negative least squares
solves for the gains that make the sum land on the reference's profile. Energy scales with the
square of the gain, so the solve happens in the energy domain and the result is square-rooted
back to gains.

This is a static mix: no automation, because the bridge cannot verify automation writes.

Usage: python scripts/optimize_levels.py --plan logs/tribal_replica_plan_v4.json
"""
from __future__ import annotations
import argparse, json

from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.optimize import nnls

SCRATCH = Path(__file__).resolve().parents[1] / "logs"
REFERENCE = Path.home() / "music-platform-runtime/stems-src/reference_tudm_44100_2ch.wav"
BANDS = ((40, 120), (200, 500), (500, 2000), (2000, 6000), (6000, 16000))
SR = 44100
CURRENT = {"kick": .75, "perc": .52, "hats": .46, "shaker": .40, "conga": .46,
           "djembe": .46, "bongo": .40, "bass": .70, "hook": .50,
           "core_kick": .75, "core_hats": .46, "core_clap": .55, "stabs_loop": .45}


def band_energy(signal: np.ndarray) -> np.ndarray:
    """Band energies of a whole signal, in the same five bands as the metric."""
    spectrum = np.abs(np.fft.rfft(signal)) ** 2
    freqs = np.fft.rfftfreq(len(signal), 1 / SR)
    return np.array([float(spectrum[np.logical_and(freqs >= lo, freqs < hi)].sum())
                     for lo, hi in BANDS])


def render_element(plan: dict, element: str, buffer_len: int) -> np.ndarray | None:
    """Render one element alone at unity gain, mirroring the plan's placement logic."""
    beat_s = 60.0 / float(plan["bpm"])
    buffer = np.zeros(buffer_len, dtype=np.float32)
    if element.startswith("core_"):
        info = plan.get("core", {}).get(element[5:])
        if not info:
            return None
        audio, sr = sf.read(info["wav"], always_2d=True, dtype="float32")
        clip = audio.mean(axis=1)
        stop = min(len(clip), buffer_len)
        buffer[:stop] += clip[:stop]
        return buffer
    choice = plan["library"]["palette"].get(element)
    if not choice:
        return None
    audio, sr = sf.read(choice["path"], always_2d=True, dtype="float32")
    clip = audio.mean(axis=1)
    clip_beats = len(clip) / sr / beat_s
    step = min(max(4.0, round(clip_beats)), 32.0)
    total_bars = int(plan["total_bars"])
    for run in plan["library"]["placements"].get(element, []):
        end = run["end_bar"] if run.get("end_bar") is not None else total_bars
        beat = float(run["start_bar"]) * 4.0
        while beat < float(end) * 4.0 - 1e-6:
            start = int(round(beat * beat_s * SR))
            stop = min(start + len(clip), buffer_len)
            if start < buffer_len:
                buffer[start:stop] += clip[: stop - start]
            beat += step
    return buffer


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, default=SCRATCH / "tribal_replica_plan_v4.json")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    beat_s = 60.0 / float(plan["bpm"])
    buffer_len = int(int(plan["total_bars"]) * 4 * beat_s * SR) + SR

    elements = list(plan["library"]["placements"]) + [f"core_{k}" for k in plan.get("core", {})]
    profiles = {}
    for element in elements:
        rendered = render_element(plan, element, buffer_len)
        if rendered is not None and np.any(rendered):
            profiles[element] = band_energy(rendered)
    print("elementos con perfil:", list(profiles))

    reference, sr = sf.read(str(REFERENCE), always_2d=False, dtype="float32")
    if reference.ndim > 1:
        reference = reference.mean(axis=1)
    target = band_energy(reference)
    target = target / target.sum()

    matrix = np.array([profiles[e] / profiles[e].sum() for e in profiles]).T
    # Unconstrained nnls drove the kick and hats to zero to nail the global band profile
    # (residual 0.0000) - a perfect fit that deletes the groove. Gains are therefore bounded
    # around the current musical values, and the kick keeps a hard floor: it is the anchor of
    # the track, not spectral filler.
    order = list(profiles)
    lower = np.array([(0.55 if element != "core_kick" else 0.75) ** 2 * CURRENT.get(element, .5) ** 2
                      for element in order])
    upper = np.array([1.40 ** 2 * CURRENT.get(element, .5) ** 2 for element in order])
    from scipy.optimize import lsq_linear
    solution = lsq_linear(matrix, target, bounds=(lower, upper), max_iter=500)
    gains_energy = solution.x
    gains = {element: float(np.sqrt(g)) for element, g in zip(order, gains_energy)}
    residual = float(np.linalg.norm(matrix @ gains_energy - target))

    mixed = np.zeros(buffer_len, dtype=np.float32)
    for element, gain in gains.items():
        mixed += render_element(plan, element, buffer_len) * gain
    peak = float(np.max(np.abs(mixed)) or 1.0)
    scale = 0.90 / peak if peak > 0.90 else 1.0
    gains = {element: round(gain * scale, 4) for element, gain in gains.items()}

    plan["gains"] = gains
    plan["mix_report"] = {"method": "nnls on band profiles", "residual": round(float(residual), 4),
                          "peak_before_scale": round(peak, 3), "applied_scale": round(scale, 3)}
    out = args.out or args.plan.with_name(args.plan.stem.replace("plan", "plan_mixed") + ".json")
    out.write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n{'elemento':12s} {'antes':>7} {'ahora':>7}")
    print("-" * 28)
    for element, gain in gains.items():
        print(f"{element:12s} {CURRENT.get(element, float('nan')):>7.2f} {gain:>7.3f}")
    print(f"\nresidual del ajuste espectral: {residual:.4f} | pico antes: {peak:.3f} | "
          f"escala aplicada: {scale:.3f}")
    print("->", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
