"""PANNs where it actually answers 'what instrument': isolated stems and single loops.

On a full mix AudioSet reports the dominant genre (Music, House, Disco), which says nothing
about which instrument is playing. On a stem or a single loop the mixture stops dominating and
the classes do describe the source. That is what this measures.

Usage: ~/music-platform-runtime/venvs/demucs/bin/python scripts/tag_instruments_panns.py
"""
from __future__ import annotations
import json

from pathlib import Path

import julius
import numpy as np
import soundfile as sf
import torch

from panns_inference import AudioTagging, labels

SCRATCH = Path("/Volumes/Lucas/Proyectos Data/music-platform/logs")
STEMS = {
    "drums": Path.home() / "music-platform-runtime/stems/ft/htdemucs_ft/drums.wav",
    "bass": Path.home() / "music-platform-runtime/stems/ft/htdemucs_ft/bass.wav",
    "vocals": Path.home() / "music-platform-runtime/stems/ft/htdemucs_ft/vocals.wav",
    "other": Path.home() / "music-platform-runtime/stems/ft/htdemucs_ft/other.wav",
    "guitar": Path.home() / "music-platform-runtime/stems/6s/htdemucs_6s/guitar.wav",
    "piano": Path.home() / "music-platform-runtime/stems/6s/htdemucs_6s/piano.wav",
}
ROLE_KEYWORDS = {
    "kick": ("kick", "kik"), "hats": ("hat loop", "top loop", "hihat"),
    "conga": ("conga",), "djembe": ("djembe", "jembe"), "bongo": ("bongo", "timbale"),
    "shaker": ("shaker", "shak", "cabasa", "guiro"),
}
MAX_PER_ROLE = 10
SR = 32000


def tag(tagger, path: Path) -> tuple[list[dict], np.ndarray]:
    audio, sr = sf.read(str(path), always_2d=True, dtype="float32")
    mono = torch.from_numpy(audio.mean(axis=1))
    if sr != SR:
        mono = julius.resample_frac(mono, sr, SR)
    clipwise, embedding = tagger.inference(mono.numpy()[None, :])
    order = np.argsort(clipwise[0])[::-1][:8]
    return ([{"label": labels[int(i)], "p": round(float(clipwise[0][int(i)]), 4)} for i in order],
            embedding[0])


def main() -> int:
    tagger = AudioTagging(checkpoint_path=None, device="cpu")
    report: dict = {"stems": {}, "loops": {}}

    print("=== stems (instrumento aislado) ===")
    for name, path in STEMS.items():
        if not path.is_file():
            continue
        tops, _ = tag(tagger, path)
        report["stems"][name] = tops[:6]
        print(f"  {name:8s} " + " ".join(f"{t['label']}({t['p']:.2f})" for t in tops[:4]))

    # the analysis venv has no `copilot` package, so read the index artifact directly: the
    # script must run where torch lives, not where the repo package does
    (SCRATCH / "instrument_tags_panns.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    index_file = Path.home() / "Library/Application Support/MusicPlatform/sample-library-index.json"
    raw_assets = json.loads(index_file.read_text(encoding="utf-8"))["assets"]
    # the index maps sha -> asset, so iterating it directly would yield keys, not assets
    assets = list(raw_assets.values()) if isinstance(raw_assets, dict) else raw_assets
    print(f"indice: {len(assets)} assets")
    for role, keywords in ROLE_KEYWORDS.items():
        hits = []
        for asset in assets:
            rel = str(asset.get("relative_path") or asset.get("filename") or "").lower()
            if any(k in rel for k in keywords):
                hits.append((asset, rel))
        report["loops"][role] = []
        print(f"\n=== {role} ({len(hits)} candidatos, se taggean {min(len(hits), MAX_PER_ROLE)}) ===")
        for asset, rel in hits[:MAX_PER_ROLE]:
            tops, _ = tag(tagger, Path(asset["path"]))
            report["loops"][role].append({"file": Path(rel).name, "top": tops[:5]})
            print(f"  {Path(rel).name[:46]:46s} " +
                  " ".join(f"{t['label']}({t['p']:.2f})" for t in tops[:3]))

    (SCRATCH / "instrument_tags_panns.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n->", SCRATCH / "instrument_tags_panns.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
