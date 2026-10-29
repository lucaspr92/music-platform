# Handoff: where this stands and how to continue

Branch `lucas/producer-serum2-v1`, published as `lucas/producer-serum2-v1-on-main`, PR #2.

## What exists right now

A 6:01 tribal tech house arrangement in Ableton, built from indexed loop-library material, at
128 BPM in D minor, with ten independent tracks:

| track | material | bars | volume |
|---|---|---|---|
| kick | PML_TH3_Kick_Loop_009_128BPM | 17-107, 129-186 | 0.75 |
| hats | Fiesta 4 Hat Loop 9 | 0-193 | 0.55 |
| perc_a | Fiesta 4 Percussion Loop 12 | 9-193 | 0.50 |
| perc_b | Fiesta 4 Percussion Loop 5 | 41-193 | 0.45 |
| shaker | Flamingo Shaker Loop 14 | 0-193 | 0.35 |
| bass | Fiesta 4 Bass Loop 23 (Dm) | 33-107, 129-193 | 0.40 |
| buildup | PML_TH3_Buildup_Loop_002_130BPM | 124.1-129 | 0.38 |
| fill | PML_TH3_Fill_Loop_002_128BPM_Ring | 104-107 | 0.42 |
| synth | Fiesta 4 Synth Loop 15 (Dm) | 51-107, 129-178 | 0.32 |
| vocal | Fiesta 4 Vocal Loop 3 | 59-107, 137-178 | 0.26 |

The arrangement has an arc: percussive intro, kick at 17, bass at 33, a real breakdown 107-129 with
no kick and no bass, drop at 129, and the kick out at 186.

Plans are versioned under `plans/`. `plans/v15_fx_music.json` is the current one.

## Reproduce it on another machine

```bash
# 1. resolve the sample paths for THIS machine (plans store absolute paths)
python scripts/relink_plan_paths.py --plan plans/v15_fx_music.json --write plans/v15_fx_music.json

# 2. open Ableton with the remote script listening on 127.0.0.1:9877, on a set you do not mind
#    gaining tracks, then:
python scripts/build_replica_v3.py --plan plans/v15_fx_music.json --append
```

Requirements: the sample packs used above must be installed and indexed at
`~/Library/Application Support/MusicPlatform/sample-library-index.json` (the same index the sample
browser uses). `relink_plan_paths.py` will name anything it cannot find instead of leaving a
broken path behind.

## The rule that matters: append, do not rebuild

`--append` adds only what the plan has and the set lacks, and never deletes. A rebuild deletes
every `REPLICA -` track: it wipes manual edits and, when it failed halfway once, it left the set in
an in-doubt state. Use `--append` by default. To replace one element, delete that single track and
append again.

## Known limits, so nobody is surprised

- **No per-track EQ from the bridge yet.** The bridge does not expose effect setters, so the
  spectral balance has to be done by hand in Live. Measured: the reference sits at sub-minus-mid
  +6.5 dB, and these raw loops at +19 dB, i.e. the midrange needs a lift and the bass a low cut.
- **`Impact.wav` and friends are 60-second collections, not single hits.** Do not place them as
  one-shots.
- **The Serum patch is not accessible:** the plugin exposes one parameter through the bridge, so
  the synth parts are library loops right now, not the reference's sound design.
- **Pitch/chord transcription from the stems is not reliable.** 88% of the bass stem's onsets
  coincide with the drums and 61% of its energy is below 100 Hz: it is kick bleed. Per-instrument
  stems leak too (snare, toms and hats each showed 11-12 onsets per bar). Do not build parts from
  those; `docs/REPLICATION_RESULTS_V1.md` has the numbers.
