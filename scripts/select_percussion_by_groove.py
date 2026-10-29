"""Pick hand-percussion loops by measured groove distance, not by name.

The index's fingerprint stores the intra-beat distribution as four values (the four sixteenths
within a beat), so the reference's 16-step histogram is folded to the same four bins before
comparing. Tempo mismatch is penalised separately, because a loop at 100 BPM is a different
groove no matter how its sixteenths sit.

Only assets with a usable fingerprint take part, and the report says how many were dropped.

Usage: python scripts/select_percussion_by_groove.py
"""
from __future__ import annotations
import json

from pathlib import Path

import numpy as np

import essentia
import essentia.standard as es
from scipy.signal import butter, sosfiltfilt

essentia.log.warningActive = False

SCRATCH = Path(__file__).resolve().parents[1] / "logs"
DRUMS = Path.home() / "music-platform-runtime/stems/ft/htdemucs_ft/drums.wav"
INDEX = Path.home() / "Library/Application Support/MusicPlatform/sample-library-index.json"
OUT = SCRATCH / "percussion_selection_v5.json"
SR = 44100
TARGET_BPM = 128.0
ROLES = {"conga": ("conga", "congas"), "djembe": ("djembe", "jembe"),
         "bongo": ("bongo", "bongos", "timba"), "shaker": ("shaker", "shak", "cabasa", "guiro")}


def reference_body_profile() -> list[float]:
    """The reference's intra-beat accent distribution on the hand-drum band."""
    audio = np.asarray(es.MonoLoader(filename=str(DRUMS), resampleQuality=4)(), dtype=np.float32)
    nyquist = SR / 2.0
    sos = butter(4, [200 / nyquist, 500 / nyquist], btype="band", output="sos")
    band = sosfiltfilt(sos, audio).astype(np.float32)
    times, _rate = es.OnsetRate()(band)
    structure = json.loads((SCRATCH / "reference_structure.json").read_text(encoding="utf-8"))
    bar_s = float(structure["bar_s"])
    hist = np.zeros(16)
    for t in times:
        bar = int(t / bar_s)
        step = int(round((t - bar * bar_s) / (bar_s / 16.0))) % 16
        hist[step] += 1
    hist = hist / hist.sum() if hist.sum() else hist
    folded = [float(hist[i:i + 4].sum()) for i in range(0, 16, 4)]
    total = sum(folded) or 1.0
    return [round(v / total, 4) for v in folded]


def tv(a: list[float], b: list[float]) -> float:
    return float(sum(abs(x - y) for x, y in zip(a, b)) / 2.0)


def main() -> int:
    target = reference_body_profile()
    print("perfil de 16vos de la referencia (por beat):", target)

    assets = list(json.loads(INDEX.read_text(encoding="utf-8"))["assets"].values())
    valid = [a for a in assets
             if a.get("groove") and a["groove"].get("grid") and (a["groove"].get("bpm") or 0) > 0]
    perc = [a for a in valid if str(a.get("semantic_role")).upper() == "PERCUSSION"]
    print(f"assets: {len(assets)} | con grid valido: {len(valid)} | PERCUSSION validos: {len(perc)}")

    scored = []
    for asset in perc:
        groove = asset["groove"]
        sixteenths = groove["grid"].get("sixteenths") or []
        if len(sixteenths) != 4:
            continue
        shape_distance = tv(target, [float(x) for x in sixteenths])
        bpm = float(groove["bpm"])
        tempo_penalty = min(abs(bpm - TARGET_BPM) / 40.0, 1.0)
        scored.append({"path": asset["path"], "file": Path(asset["relative_path"]).name,
                       "bpm": round(bpm, 2), "role": str(asset.get("semantic_role")),
                       "shape_distance": round(shape_distance, 4),
                       "tempo_penalty": round(tempo_penalty, 4),
                       "score": round(1.0 - (0.7 * shape_distance + 0.3 * tempo_penalty), 4)})
    scored.sort(key=lambda item: -item["score"])

    print()
    print(f"{'score':>6} {'forma':>6} {'bpm':>7}  archivo")
    print("-" * 78)
    for row in scored[:14]:
        print(f"{row['score']:>6.3f} {row['shape_distance']:>6.3f} {row['bpm']:>7.2f}  {row['file'][:52]}")

    selection = {}
    for role, keywords in ROLES.items():
        pool = [row for row in scored if any(k in row["file"].lower() for k in keywords)]
        if pool:
            selection[role] = {**pool[0], "source": "role_keyword"}
        else:
            selection[role] = {**scored[0], "source": "global_best_no_role_match"}
        print(f"\n  {role:7s} -> {selection[role]['file'][:50]}  "
              f"(score {selection[role]['score']}, {selection[role]['source']})")

    (SCRATCH / "percussion_selection_v5.json").write_text(
        json.dumps({"reference_profile": target, "candidates_valid": len(perc),
                    "selection": selection, "ranking": scored[:40]},
                   indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
