"""Index the tribal/afro/percussion + tech-house packs of the local sample library.

Uses the repo's own config/index helpers. Re-runnable: unchanged files are skipped
by size+mtime, so only new or modified assets get analysed.
"""
from __future__ import annotations
import json, sys
from pathlib import Path

# Scratch artifacts stay in the gitignored logs/ dir, resolved from the repo root so
# the script runs from any working directory.
SCRATCH = Path(__file__).resolve().parents[1] / "logs"


from copilot.sample_library.config import add_root, config_path, index_path
from copilot.sample_library.library_v1 import index_library

SAMPLES = Path("/Volumes/Lucas/Samples")
PACKS = [
    # tribal / afro / latin percussion  <- the point of this pass
    "Black.Octopus.Sound.Tribal.And.African.Percussion.(SCENE)-DISCOVER",
    "Basic Wavez - Organic Afro Fills",
    "Splice Originals - Criolla Latin Percussion",
    "wa_afro_latin_vol_01_free_samples",
    "Samplack Naito x Sen6 - Gingetsu Afro Pack",
    "Tomas Bisquierra - Percussive Tech House 4",
    "Hot Grooves - Latin Tech House 2",
    "Sample Tools by Cr2 - Latin House and Tech House Rhythms",
    "Esteban Calvet - Coppados Sample Pack 001",
    "Esteban Calvet - Coppados Sample Pack 002",
    "Esteban Calvet Coppados - Sample Pack 003",
    # tech house body (kick/bass/hook material)
    "Caravaca-Tech-House-Sessions-Vol.4",
    "Classmatic - Classmatic Sample Pack Vol.02",
    "Classmatic - Sample Pack Vol.1",
    "TOLROOM TECH HOUSE",
    "Zenhiser Spinshift - Tech House",
    "Phantom Sounds - Django Groovy Tech House",
    "Phantom Sounds - Fiesta Vol.4 Terrace Tech House Pack",
    "Production Music Live - Tech House Sound Pack Drift",
]


def main() -> int:
    cfg = config_path()
    roots = []
    for name in PACKS:
        path = SAMPLES / name
        if not path.is_dir():
            print(f"FALTA (salteado): {name}", file=sys.stderr)
            continue
        add_root(path, path=cfg)
        roots.append(path)
    print(f"roots a indexar: {len(roots)}")
    report = index_library(roots, index_path())
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print("index:", index_path())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
