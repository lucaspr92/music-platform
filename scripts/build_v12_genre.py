"""v12: written as a tribal tech house track, not derived from contaminated stems.

Why this discards the "measured parts" of v9-v11: 88% of the bass stem's onsets coincided with the
drums (median 11.6 ms apart) and 61% of its energy sat below 100 Hz, so that stem was the kick
bleeding. Per-instrument stems were no better: snare, toms and hats each showed 11-12 onsets per
bar, i.e. the same transients. Only two things measured reliably: the kick is four-on-the-floor
(1-5-9-13) and the sections are at bars 51 / 107 / 129 / 178.

So the parts are written to the genre's conventions instead:

  kick        1 5 9 13                       the one thing that WAS measured
  clap        5 13                           backbeat
  closed hat  16ths, open hat on every &     the genre's hat figure
  conga       4 6 11 14      \  syncopated call
  bongo       2 8 12 16      /  and answer
  djembe      7 15 + fill every 4th bar      the tribal accent
  bass        offbeat 8ths on D2, G1 every 4th bar   the bass marks the &, it does not run 16ths
  stabs       Dm once every 2 bars           space instead of saturation

Arrangement follows the measured sections. One-shots everywhere: no 126-BPM loops stretched to 128.

Usage: python scripts/build_v12_genre.py
"""
from __future__ import annotations
import json

from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from copilot.platform.runtime_paths import ableton_core_library
from copilot.producer import tech_house_kit_v1 as kit

ROOT = Path(__file__).resolve().parents[1]
SCRATCH = ROOT / "logs"
OUT = SCRATCH / "v12_core"
SR = 44100

SOURCES = {
    "kick":   "Samples/One Shots/Drums/Kick/Kick 909 1.aif",
    "clap":   "Samples/One Shots/Drums/Clap/Clap 909.aif",
    "hats":   "Samples/One Shots/Drums/Hihat/Hihat Closed 909.aif",
    "hats_open": "Samples/One Shots/Drums/Hihat/Hihat Open 909.aif",
    "conga":  "Samples/One Shots/Drums/Conga/Conga 808 Mid.aif",
    "bongo":  "Samples/One Shots/Drums/Bongo/Bongo C78 Hi.aif",
    "djembe": "Samples/One Shots/Drums/Misc Percussion/Djembe High Hard.aif",
}

# sixteenth steps (1-based) per element
PATTERNS = {
    "kick":   [(1, 1.0), (5, 1.0), (9, 1.0), (13, 1.0)],
    "clap":   [(5, 1.0), (13, 1.0)],
    "hats":   [(step, 0.55 if step % 2 else 0.7) for step in range(1, 17)],
    "hats_open": [(3, 0.6), (7, 0.6), (11, 0.6), (15, 0.6)],
    "conga":  [(4, 0.8), (6, 0.6), (11, 0.75), (14, 0.65)],
    "bongo":  [(2, 0.7), (8, 0.8), (12, 0.6), (16, 0.7)],
    "djembe": [(7, 0.9), (15, 0.95)],
}

# section gates: element -> [(first_bar, last_bar)] ; anything unlisted plays everywhere
GATES = {
    "kick":        [(17, 107), (129, 186)],
    "clap":        [(17, 107), (129, 193)],
    "conga":       [(0, 193)],
    "bongo":       [(17, 107), (117, 193)],
    "djembe":      [(33, 193)],
    "hats_open":   [(33, 193)],
}


def load_oneshot(relative: str) -> np.ndarray:
    path = kit.resolve_relative(ableton_core_library(), relative)
    if not path.is_file():
        raise FileNotFoundError(f"CORE_ONESHOT_MISSING: {path}")
    audio, sr = sf.read(str(path), always_2d=True, dtype="float32")
    mono = audio.mean(axis=1)
    if sr != SR:
        mono = librosa.resample(mono, orig_sr=sr, target_sr=SR)
    return mono.astype(np.float32)


def main() -> int:
    plan = json.loads((SCRATCH / "tribal_replica_plan_v11.json").read_text(encoding="utf-8"))
    bar_s = float(plan["bar_s"])
    total_bars = int(plan["total_bars"])
    OUT.mkdir(parents=True, exist_ok=True)
    track_s = total_bars * bar_s + 2.0

    rendered = {}
    for element, pattern in PATTERNS.items():
        sample = load_oneshot(SOURCES[element])
        buffer = np.zeros(int(track_s * SR), dtype=np.float32)
        gates = GATES.get(element)
        hits = 0
        for bar in range(total_bars):
            if gates and not any(first <= bar < last for first, last in gates):
                continue
            for step, velocity in pattern:
                start = int(round(((bar * 4) + (step - 1) * 0.25) * bar_s / 4 * SR))
                stop = min(start + len(sample), len(buffer))
                if 0 <= start < len(buffer):
                    buffer[start:stop] += sample[: stop - start] * velocity
                    hits += 1
        path = OUT / f"core_{element}.wav"
        sf.write(str(path), buffer, SR)
        rendered[element] = {"path": str(path), "hits": hits}
        print(f"  {element:10s} {hits:5d} golpes -> {path.name}")

    # merge closed + open hats into one element so the engine plays them together
    closed, open_hat = np.zeros(int(track_s * SR), dtype=np.float32), np.zeros(int(track_s * SR), dtype=np.float32)
    closed[:] = sf.read(rendered["hats"]["path"], always_2d=False)[0]
    open_hat[:] = sf.read(rendered["hats_open"]["path"], always_2d=False)[0]
    sf.write(OUT / "core_hats_all.wav", closed + open_hat * 0.8, SR)

    plan["version"] = "v12"
    plan["library"] = {"palette": {}, "placements": {}}
    plan["core"] = {
        "kick":   {"wav": rendered["kick"]["path"], "source": "909 kick, 1-5-9-13 (measured)", "measured": True},
        "clap":   {"wav": rendered["clap"]["path"], "source": "909 clap, backbeat 5-13", "measured": False},
        "hats":   {"wav": str(OUT / "core_hats_all.wav"), "source": "closed 16ths + open on every &", "measured": False},
        "conga":  {"wav": rendered["conga"]["path"], "source": "conga 808 mid, syncopated call 4-6-11-14", "measured": False},
        "bongo":  {"wav": rendered["bongo"]["path"], "source": "bongo C78 hi, answer 2-8-12-16", "measured": False},
        "djembe": {"wav": rendered["djembe"]["path"], "source": "djembe high, accents 7-15", "measured": False},
    }
    plan["gains"] = {"core_kick": 0.85, "core_clap": 0.55, "core_hats": 0.4,
                     "core_conga": 0.42, "core_bongo": 0.34, "core_djembe": 0.34}

    bass = []
    for bar in range(33, total_bars):
        if 107 <= bar < 129:
            continue
        for step in (3, 7, 11, 15):
            pitch = 38
            if step == 15 and bar % 4 == 0:
                pitch = 31          # the octave drop the reference really does
            bass.append({"pitch": pitch, "start": round(bar * 4 + (step - 1) * 0.25, 3),
                         "duration": 0.3, "velocity": 105 if step == 15 else 95})
    plan["synth"]["bass"].update({
        "notes": bass, "volume": 0.5, "measured_notes": len(bass),
        "source": "offbeat 8ths on D2, G1 on the last & of every 4th bar",
        "limits": ["patch unavailable: Serum 2 exposes one parameter"],
    })
    stabs = []
    for bar in range(33, total_bars, 2):
        if 107 <= bar < 121:
            continue
        for pitch in (62, 65, 69):
            stabs.append({"pitch": pitch, "start": round(bar * 4 + 2.5, 3),
                          "duration": 0.45, "velocity": 88})
    plan["synth"]["stabs"].update({
        "notes": stabs, "volume": 0.4, "measured_notes": len(stabs),
        "source": "Dm every 2 bars on the last &", "chord": "Dm",
    })
    note = "v12: written to genre conventions; contaminated stem-derived parts discarded"
    if isinstance(plan.get("revisions"), dict):
        plan["revisions"]["v12"] = note
    else:
        plan.setdefault("revisions", []).append(note)
    path = SCRATCH / "tribal_replica_plan_v12.json"
    path.write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"bajo: {len(bass)} notas (8vos contratiempo) | stabs: {len(stabs)}")
    print("->", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
