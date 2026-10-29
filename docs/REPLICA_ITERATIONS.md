# Replica iterations: measured against the reference

Every number below comes from `scripts/similarity_to_reference.py`, which scores a candidate
render per dimension and reports a weighted total. Two controls keep it honest:

- the reference scored against itself reads **1.000** on every dimension
- **white noise reads 0.323**, so the metric is able to say "nothing alike"

A metric without those two controls certifies work that is not done. An earlier version of this
one used the mean absolute difference over 16 normalised bins and scored noise at 0.94; it was
replaced by total variation plus an onset-density penalty.

## What the metric measures

| dimension | how | weight |
|---|---|---|
| groove_kick / body / clap / air | 16th-of-bar onset distribution per band, times a density ratio | 0.5 total |
| band_balance | per-8-bar share of energy in five bands, total variation | 0.25 |
| energy_curve | correlation of per-section RMS between candidate and reference | 0.25 |
| timbre | null: clip embeddings need the analysis venv | not counted |

Sections that are silent on either side are excluded: the reference file simply ends, and
comparing its tail against a render tail measures nothing.

## Iterations

| version | change | total | delta | verdict |
|---|---|---|---|---|
| v3.2 | loop bass restored, gated by the measured stem windows; synth bass muted | 0.608 | - | baseline |
| v4 | **rendered percussive core** (kick, hats, clap) from the reference's own measured onsets, replacing the kick and hat loops | **0.727** | **+0.119** | kept |
| v5 | hand percussion re-selected by measured groove distance | 0.718 | -0.009 | **rejected**: a loop's own fingerprint did not predict the assembled score |
| v4 + stabs loop | stab layer as a library element (Synth Loop 21) | 0.7325 | +0.0055 | marginal, left to taste |
| v4 + constrained nnls mix | gains solved to match the global band profile | 0.690 | -0.037 | **rejected**: it zeroed the kick and hats to fit the profile |
| v7 | v4 with the core at full gain (x0.8 headroom for the Live set) | **0.727** | same as v4 | final |

## Decisions that measurement took away from me

- **The kick loop was the problem.** Replacing loop-based kick and hats with a core rendered from
  the reference's own onsets is the single biggest gain (+0.119), and it is also the change that
  removes the "sounds generic" complaint.
- **A loop's fingerprint does not predict the track.** Ranking percussion by intra-beat accent
  distance looked principled and measured worse in context. Candidates are now scored by
  rendering the whole track, not by scoring the loop alone.
- **A global gain solve cannot fix a per-section balance.** The optimizer reached residual 0.0000
  by muting the kick and hats; with musical bounds it still measured worse, because the metric
  compares per section and a static gain is global.
- **The core wanted more level.** Full gain measured +0.027 over the previous Live levels.

## What is measured vs assumed

- kick and hats: **measured** (515 and 2300 onsets on the drums stem, silence-guarded)
- clap: **assumed** backbeat on 2 and 4. The clap cannot be isolated from this stem: every band
  of it yields about 11 onsets per bar because hat transients dominate, so no band and no
  threshold separates the clap. The report says `measured: false` for it.
- bass: loop chosen by vendor key label and measured in context; the index's own key detection is
  unreliable (`... Dm.wav` detected as "C major"), so the filename is the source of truth there.

## Blocked, with the reason

- **Plugin parameter surface**: Serum 2 exposes one parameter (`Device On`) in the bridge, so
  patches cannot be shaped and the reference's -23 cent tuning cannot be set.
- **Automation**: transitions need filter automation; the bridge writes it in a probe but the
  readback returns no breakpoints, so a write could not be verified. Not shipped.
- **Per-section balance**: needs automation, same blocker.

## Reproduce

```
python scripts/measure_rhythm_by_band.py          # onsets per percussive family
python scripts/render_percussion_core.py          # kick/hats/clap core
python scripts/plan_replica_v4.py                 # v4 plan
python scripts/render_plan_audio.py --plan logs/tribal_replica_plan_v4.json
python scripts/similarity_to_reference.py --candidate logs/render_4.wav
python scripts/build_replica_v3.py --plan logs/tribal_replica_plan_v7.json
pytest tests/test_similarity_to_reference_v1.py tests/test_percussion_core_v1.py
```

## v8: the sounds chosen by measurement, then mastered

| stage | total | change |
|---|---|---|
| v4/v7 core of 909 one-shots | 0.7270 | reference |
| kick one-shot chosen in context | 0.7621 | +0.035 |
| v8 (kick + clap chosen, 909 hat kept) | 0.7664 | +0.004 |
| **v8 + matchering** | **0.8031** | **+0.037** |

`matchering` is open source (MIT) reference-based mastering: it matches the target's loudness,
spectral balance, RMS and stereo width to the reference. It lifted `band_balance` 0.689 -> 0.761
and `energy_curve` 0.730 -> 0.885 without touching a single note, which is the clearest proof
that most of the remaining gap was mix, not arrangement.

Two lessons paid for with failures:

1. **A timbre ranking alone selects the wrong sample.** The closest one-shot by timbre distance
   (`Fiesta 4 Kick 15`, 0.411) scored 0.746 in context; the second closest (`HG LTH 2 Kick 27`,
   0.573) scored 0.762 and won. A sample can sound right alone and still fight the hats or decay
   into the next hit, so every element is chosen by rendering the whole track.
2. **A timbre ranking that ignores tempo and key invents nonsense.** Ranked by timbre alone the
   best bass was 123.5 BPM in F and the best synth was A#, against a 128 BPM reference in Dm.
   Musical constraints are applied first; timbre only orders what survives them.

Negative results kept as results: the 909 closed hat beat every timbre candidate (0.727 vs
0.696-0.702). A `Django Ride` ranked 0.473 by timbre, which is exactly the trap: a ride at 2300
hits per track smears. It was measured, it lost, it stayed out.

`log/` is gitignored: the JSON evidence is local and regenerated by the scripts above.
