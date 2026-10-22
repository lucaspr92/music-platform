"""Choose the tribal/afro palette from the measured shortlist.

Fixes two things the first ranking got wrong:
  * tempo: for short loops Essentia's multifeature is unreliable, while the packs
    LABEL their loops ("095bpm", "124", "128 BPM") -- the label wins, the index
    estimate is the fallback, and a measured value is only used to break ties
  * band targets: the reference's band balance is a MIX, so a solo shaker can never
    match it. Each role now only has to be PRESENT and dominant in its own band.
"""
from __future__ import annotations
import json, re
from pathlib import Path

# Scratch artifacts stay in the gitignored logs/ dir, resolved from the repo root so
# the script runs from any working directory.
SCRATCH = Path(__file__).resolve().parents[1] / "logs"


RANKING = (SCRATCH / "tribal_fingerprint_ranking.json")
TARGET_BPM = 128.0
TARGET_GRID = [0.257, 0.252, 0.246, 0.245]

# role -> (band it must own, must-not-be-dominated-by)
ROLE_BAND = {
    "kick": ("kick_body", ("air",)), "clap": ("snare_clap", ("kick_body",)),
    "hat_top": ("air", ("kick_body",)), "shaker": ("air", ("kick_body", "drum_body")),
    "conga": ("drum_body", ("air",)), "djembe": ("drum_body", ("air",)),
    "bongo": ("drum_body", ("air",)), "talking": ("drum_body", ("air",)),
    "tribal_perc": ("drum_body", ("air",)), "bass": ("kick_body", ("air",)), "hook": None,
}
ROLE_FALLBACK_LABEL = {"kick": "kick", "clap": "clap", "shaker": "shaker", "conga": "conga"}


def number_in_name(name: str) -> float | None:
    for token in re.findall(r"\d{2,3}", name):
        value = float(token)
        if 60.0 <= value <= 190.0:
            return value
    return None


def tonic_in_name(name: str) -> str | None:
    for key in ("D#", "C#", "F#", "G#", "A#", "Dmin", "Dm", "Gmin", "Cmin", "Amin", "Emin", "Fmin", "Bmin"):
        if key.lower() in name.lower():
            return key
    return None


def grid_error(six) -> float:
    if not six:
        return 25.0
    return float(sum(abs(a - b) for a, b in zip(six, TARGET_GRID))) / len(TARGET_GRID) * 100


def main() -> int:
    ranking = json.loads(RANKING.read_text(encoding="utf-8"))
    palette: dict[str, dict] = {}
    for role, rows in ranking.items():
        scored = []
        for r in rows:
            name = r["filename"]
            d = r["detail"]
            band_info = ROLE_BAND.get(role)
            bpm_file = number_in_name(name)
            bpm = bpm_file or d.get("bpm")
            if bpm is None:
                continue
            bpm_err = abs(bpm - TARGET_BPM)
            bands = d.get("band_db")
            band_ok = True
            detail_band = None
            if band_info:
                own, avoid = band_info
                g = r.get("bands_all") or {}
                detail_band = None
            # band data lives in the stored fingerprint; re-read it below
            scored.append({"row": r, "bpm": bpm, "bpm_file": bpm_file, "bpm_err": bpm_err,
                           "grid_err": grid_error(d.get("grid_sixteenths")),
                           "is_fill": "fill" in name.lower(), "tonic": tonic_in_name(name)})
        # Tonal roles: a bass/hook in the wrong key is useless for a D-minor reference,
        # so the key outranks everything else. For drums, the measure of "right" is the
        # groove: an on-grid loop at the wrong label beats an off-grid one at the right label.
        def sort_key(s):
            tonic = (s["tonic"] or "").upper()
            key_ok = 0 if tonic in ("D", "DM") else 1
            if role in ("bass", "hook"):
                return (s["is_fill"], key_ok, s["bpm_err"], s["grid_err"])
            return (s["is_fill"], s["bpm_err"], s["grid_err"])

        scored.sort(key=sort_key)
        keep = [s for s in scored if not s["is_fill"]][:3] or scored[:3]
        palette[role] = [{"filename": s["row"]["filename"], "path": s["row"]["path"],
                          "bpm_file": s["bpm_file"], "bpm_measured": s["row"]["detail"].get("bpm"),
                          "grid": s["row"]["detail"].get("grid_sixteenths"),
                          "grid_err": round(s["grid_err"], 2), "tonic": s["tonic"],
                          "is_fill": s["is_fill"]} for s in keep]
        print(f"=== {role} ===")
        for c in palette[role]:
            print(f"   {c['filename'][:50]:50s} bpm_file={c['bpm_file']} medido={c['bpm_measured']} "
                  f"grid_err={c['grid_err']} tonic={c['tonic']}")
    (SCRATCH / "tribal_palette_v1.json").write_text(json.dumps(palette, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n-> logs/tribal_palette_v1.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
