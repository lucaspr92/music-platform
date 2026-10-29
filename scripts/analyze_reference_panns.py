"""PANNs: what instruments does the reference actually contain, section by section.

Band shares say where energy sits; PANNs (527 AudioSet classes) says WHAT is sounding -
Percussion, Drum, Rattle, Speech and so on. It also returns a 2048-d embedding per clip,
which is the timbre dimension the similarity metric will need.

Usage (runs in the analysis venv, which holds torch + panns):
  ~/music-platform-runtime/venvs/demucs/bin/python scripts/analyze_reference_panns.py
"""
from __future__ import annotations
import argparse, json

from pathlib import Path

import numpy as np
import soundfile as sf

from panns_inference import AudioTagging, labels

SCRATCH = Path("/Volumes/Lucas/Proyectos Data/music-platform/logs")
AUDIO = Path.home() / "music-platform-runtime/stems-src/reference_tudm_32000_1ch.wav"
TOP_K = 6


def top_tags(clipwise: np.ndarray, k: int = TOP_K) -> list[dict]:
    order = np.argsort(clipwise)[::-1][:k]
    return [{"label": labels[int(i)], "p": round(float(clipwise[int(i)]), 4)} for i in order]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", type=Path, default=AUDIO)
    parser.add_argument("--sections", type=Path, default=SCRATCH / "reference_structure.json")
    parser.add_argument("--json", type=Path, default=SCRATCH / "reference_panns.json")
    parser.add_argument("--embeddings", type=Path, default=SCRATCH / "reference_panns_embeddings.npy")
    args = parser.parse_args()

    audio, sr = sf.read(str(args.audio), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    tagger = AudioTagging(checkpoint_path=None, device="cpu")
    print(f"audio {len(audio) / sr:.1f}s @ {sr} Hz | {len(labels)} clases")

    clipwise, embedding = tagger.inference(audio[None, :])
    report: dict = {"audio": str(args.audio), "overall_top": top_tags(clipwise[0]),
                    "sections": []}
    print("tags globales:", ", ".join(f"{t['label']}={t['p']}" for t in report["overall_top"]))

    structure = json.loads(args.sections.read_text(encoding="utf-8"))
    vectors = []
    for section in structure.get("sections", []):
        a, b = int(float(section["start_s"]) * sr), int(float(section["end_s"]) * sr)
        chunk = audio[a:b]
        if chunk.size < sr // 2:
            continue
        with np.errstate(all="ignore"):
            sec_clipwise, sec_embedding = tagger.inference(chunk[None, :])
        report["sections"].append({
            "start_bar": section["start_bar"],
            "top": top_tags(sec_clipwise[0]),
            "rms_db": round(float(20 * np.log10(np.sqrt(np.mean(chunk ** 2)) + 1e-12)), 1),
        })
        vectors.append(sec_embedding[0])
    np.save(args.embeddings, np.asarray(vectors, dtype=np.float32))
    report["embedding_shape"] = list(np.asarray(vectors).shape)
    args.json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print()
    print(f"{'compas':>7} {'dB':>6}  instrumentos detectados")
    print("-" * 88)
    for row in report["sections"]:
        tags = " ".join(f"{t['label']}({t['p']:.2f})" for t in row["top"][:4])
        print(f"{row['start_bar']:>7.0f} {row['rms_db']:>6.1f}  {tags}")
    print()
    print(f"embeddings -> {args.embeddings} {report['embedding_shape']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
