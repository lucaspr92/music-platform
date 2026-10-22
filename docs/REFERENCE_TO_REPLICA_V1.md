# REFERENCE -> REPLICA V1

Measure a reference track, pick loops from the indexed local sample library by measured
evidence, and build an arrangement in Live that follows the reference's structure.

Everything here is measured. Nothing here claims taste: the numbers select candidates,
a human ear still decides.

## 1. Which library measures what, and why

| measurement | library | why that one |
|---|---|---|
| sections / change points | **librosa** (MFCC novelty + smoothed peaks, >= 8 bars apart) | frame-level spectral analysis is its job |
| percussive vs harmonic split | **librosa** `effects.hpss` | median-filter separation |
| band shares per section | numpy FFT over the separated components | deterministic and cheap |
| LUFS per section | **Essentia** `LoudnessEBUR128` | librosa does not implement EBU R128 |
| BPM + beat grid -> bars | **Essentia** `RhythmExtractor2013` | best on long material (see 3) |
| key | **Essentia** `KeyExtractor` | HPCP profile |
| style / genre | **Essentia + TensorFlow** (Discogs-EffNet + 400-style head) | librosa ships no models |
| 16th grid of loops | librosa `beat_track` | more stable on short loops |

Cross-library loudness and agreement probe: `scripts/probe_analysis_libraries.py`.
Essentia and pyloudnorm agree to <= 0.25 LU on real stereo and 0.037 LU on a synthetic
stereo control, so a loudness difference between them is never a defect signal.

## 2. The groove fingerprint is stored in the index (measured once)

`copilot/sample_library/groove_v1.py` writes a `groove` block into each indexed asset in
`sample-library-index.json`, keyed by the asset **sha256**:

```json
"groove": {
  "version": "groove-v2",
  "bpm": 96.6, "bpm_method": "essentia-percival", "bpm_label": 95.0,
  "percussive_over_harmonic": 0.799,
  "percussive_band_share_db": {"kick_body": -12.7, "drum_body": -3.48, "snare_clap": -3.41,
                               "attack": -14.3, "air": -23.76},
  "grid": {"sixteenths": [0.306, 0.278, 0.278, 0.139], "on_beat": 0.306},
  "key": "F minor", "key_strength": 0.617, "clipping": false,
  "genre_status": "OK",
  "genre_discogs400": [{"label": "Folk, World, & Country---African", "p": 0.117}]
}
```

Why it is never recomputed:

* keyed by sha256 -> duplicates and moved files share one analysis
* `GROOVE_VERSION` mismatch -> deliberate invalidation only (v1 -> v2 was one)
* the genre block is TensorFlow inference, so its status (`OK` / `SKIPPED` / `UNAVAILABLE`
  / `FAILED`) is recorded and a machine without the models never retries forever
* DSP and genre are separate computations: asking for genre on an asset that already has
  its DSP fingerprint costs only the inference

Measured on 176 candidate assets: **first run 119.3 s (176 COMPUTED), second run 0.4 s
(176 REUSED, 0 COMPUTED)**.

`scripts/rank_library_loops.py` fills the block for its candidates and then reads it back.

## 3. Tempo on short loops: no estimator is good, so the label is stored as evidence

60 library loops whose filename states their tempo:

| detector | exact | octave error | median error |
|---|---|---|---|
| Essentia `PercivalBpmEstimator` | **61.7%** | 1.7% | 0.36 BPM |
| Essentia `RhythmExtractor2013(multifeature)` | 51.7% | 1.7% | 0.91 |
| Essentia `degara` | 50.0% | 1.7% | 1.14 |
| index autocorrelation | 31.7% | 6.7% | 21.05 |
| librosa tempogram | 21.7% | 15.0% | 31.05 |
| librosa `beat_track` | 20.0% | 16.7% | 31.05 |

The estimators are calibrated for a whole track: a 6 s loop holds about 10 beats, so the
periodicity is genuinely ambiguous (95 / 190 / 47.5 are all defensible) and librosa's
tempogram has too few frames plus a `start_bpm=120` prior. On full material the two
agree (reference: Essentia 128.01, librosa 129.2). Hence: short material (< 20 s) uses
Percival, long material keeps multifeature, and the pack's own label is recorded next to
the measurement instead of being silently merged. Reproduction: `scripts/tempo_method_bench.py`.

## 4. The reference this was built from

`Gino Da Koda, Le Modo - Tudm feat. Le Modo (Original Mix)` (320 kbps, 6:02, both channels live):

* 128.01 BPM (conf 3.05), **D minor** (strength 0.855), -8.34 LUFS, LRA 4.19, 193 bars
* Discogs400: House 0.33 / Tech House 0.23 / Electro House 0.20 / **Tribal House 0.195**
* percussive identity: `drum_body` (200-500 Hz) -4.79 dB dominant, onsets 6.47/s,
  16th grid `[0.257, 0.252, 0.246, 0.245]` -> dense, even sixteenths
* 22 measured sections. Kick drops out at bars 51 and 107; the defining moment is the
  **22-bar breakdown at bar 107** (-11.1 LUFS, hand percussion plus bass, no kick, no hats);
  the drop returns at 129 with 6 x 8-bar blocks; the outro starts at 178.

## 5. How the replica is built

The arrangement is expressed as **clip presence**: each element track holds arrangement
clips only in the bars where the reference has that element. No automation, so no
envelope writes are needed and the result is fully reproducible from the plan JSON.

* loops with warping enabled; only loops within about 5% of the set tempo are eligible,
  because stretching a 95 BPM loop to 128 destroys the feel
* one loop per element, chosen by measured evidence (band it must own, 16th-grid distance
  from the reference, tempo, and for tonal roles the key): `scripts/rank_library_loops.py`
  then `scripts/pick_replica_palette.py` then `scripts/plan_reference_replica.py`
* `scripts/build_reference_replica.py` is **idempotent**: it reads what the arrangement
  actually contains per track and only adds what is missing, so an interrupted run is
  finished by re-running it
* verified by readback afterwards: 9/9 element tracks with the exact expected clip counts
  (kick 39, perc 49, hats 43, shaker 49, conga 49, djembe 49, bongo 49, bass 49, hook 23;
  409 arrangement clips)

Library samples are loaded by **absolute path** through `create_audio_clip`. Live's own
browser does not index the local packs, so `search_browser` finds factory hits only, and
`load_browser_item_by_path` deliberately refuses absolute paths.

## 6. Honest limits

* it reproduces structure and element entry/exit, **not the internal variation** of a
  6-minute record: loops repeat every 4-8 bars
* loudness cannot match: the reference is mastered at -8.34 LUFS, the set has no mix chain
* the kick pick won on its tempo label while its 16th grid is worse than the 124 BPM
  candidates: the one choice worth overriding by ear

## 7. Transitions in T2: what exists and what is missing

The bridge can already automate a filter: `set_clip_automation(track, clip, parameter_name,
envelope_data)` with `[{time, value}, ...]` points, plus `get_clip_automation` and
`get_clip_has_envelopes`. Probe result on a native device (`scripts/probe_device_automation.py`):
browser resolves `Auto Filter` unambiguously, the native device exposes **45 parameters**
(Frequency index 1, Resonance index 2), writing 4 envelope points returned
`{"points_added": 4}`, and the temporary track was deleted afterwards.

Missing, in order of importance:

1. **no public adapter wrapper** for `set_clip_automation` (only `get_clip_automation`
   exists), so the probe had to use a private `_command`
2. **no SafeWrite action type** for envelope writes, so a sweep is neither revertible nor
   routed through the single write authority
3. **the envelope readback does not return breakpoints** - only `has_automation: true`.
   The vendor code says "simplified - actual implementation would read breakpoints".
   Without this, verifying the shape of a sweep would be self-certification. This is the
   same reason `AUDIO_EFFECT_VERIFICATION` is `DEFERRED` in the musicplan layer
4. parameter lookup is by **name across all devices**, first match wins: fine with one
   device, wrong with Auto Filter plus EQ on the same track
5. `Frequency` is normalised 0..1, not Hz, so musical values need a mapping
6. sidechain ducking is routing (`input_routing_type` plus Post FX channel), not a
   parameter: out of scope so far

Tier 1 (transitions as audio clips, no new code) can use the indexed FX: 15 risers,
13 sweeps, 18 uplifters, 14 downlifters, 10 impacts, placed at the measured section
boundaries (bars 51, 107, 129, 186).

## 8. Running it

```
python scripts/index_local_sample_packs.py                       # index the local packs
python scripts/reference_groove_fingerprint.py <reference> --genre --json <out>
python scripts/reference_structure.py <reference> --json <out>
python scripts/rank_library_loops.py                             # fills and reads the groove cache
python scripts/pick_replica_palette.py
python scripts/plan_reference_replica.py
python scripts/build_reference_replica.py                        # idempotent Live writes
python scripts/probe_device_automation.py                        # T2 feasibility
python scripts/tempo_method_bench.py                             # the tempo evidence
```

Scratch artifacts (plans, rankings, probes) stay in the gitignored `logs/` directory and
are resolved from the repo root, so every script runs from any working directory.
