# Replication-first plan (mix and master deferred)

Objective, set by Lucas: **replicate the reference as closely as possible before touching mix or
master**. Mix and master are a later stage and are explicitly out of scope here.

## Why the current score cannot drive this

`similarity_to_reference.py` averages six dimensions, two of which are mix (`band_balance`,
`energy_curve`). Mastering lifted the total from 0.7664 to 0.8031 without changing a single note,
which proves the score rewards mix. A replication score must be **mix-invariant**: it has to read
the same whether or not the track is mastered.

So stage R0 replaces the score, and the current 0.8031 is retired as a replication figure.

## The metric: per element, not per mix

For each element E (kick, snare/clap, hats, percussion, bass, stabs, fx):

| sub-metric | what it measures | how |
|---|---|---|
| presence | which bars the element is audible in | dB relative to that stem's own median, per bar (already built) |
| pattern | *where* inside the bar the hits fall | 16th-grid histogram agreement, total variation, per instrument (not per frequency band) |
| sequence | whether the bar-to-bar gesture repeats with the same shape | inter-bar distance between 16th vectors |
| notes (pitched only) | the actual notes | note F1 at 50 ms tolerance, standard MIR definition |
| timbre (isolated) | how close the sound is once isolated | MFCC + spectral profile cosine, per element |

`Replication = weighted mean of the element scores. No loudness, no band balance, no stereo width.`
`Mix = its own separate report, printed but never folded into Replication.`

**Acceptance for R0**: reference vs reference = 1.000 per element; the mastered render scores
**exactly the same** as the unmastered render (proving mix-invariance); white noise scores near 0.

## Stages

### R0 — replication metric and an honest baseline
Build `scripts/similarity_replication.py` as above, run it on the current v8 render, publish the
first replication number. Gate: the three checks at the end of the previous section.

### R1 — separate the drum kit by instrument (this removes the assumptions)
Today the clap is *assumed* (backbeat on 2 and 4) because band filtering cannot isolate it: every
band of the drums stem shows ~11 onsets/bar from hat transients. Per-instrument separation fixes
the root cause.

1. `drumsep` (pure DSP, repo venv) on the drums stem -> kick, snare, hihat, cymbals, toms.
2. Measure each stem: counts per bar and energy. Sanity: kick ~2.65/bar, hats ~11.9/bar.
3. If the DSP separation is not clean, escalate to `mdxnet-infer` with the DrumSep 6-stem
   checkpoint (torch already available in the demucs runtime venv).

Gate: the kick and hat stems independently reproduce the counts measured by band, and the snare
stem finally gives a measured clap instead of an assumption.

### R2 — transcribe the drums
Onsets per separated instrument -> 16th pattern, velocities from each stem's local peak, plus
open/closed hat state. Cross-check with `ADTOF-pytorch` if the two disagree.

Gate: the two methods agree on the kick and snare patterns; the transcribed pattern matches R1's
per-instrument histograms.

### R3 — transcribe the bassline into notes
The bass is currently a library loop, so it is the reference's *feel* only by accident. This stage
makes the part exact:

1. `torchcrepe` on the bass stem, fmin 30 Hz, fmax 300 Hz, 5 ms hop.
2. Segment into notes (periodicity threshold, median filter, onsets as boundaries, silence guard).
3. Cross-check against what is already measured: accents on 16ths 1, 2, 3, 7, 15 and a range
   A1..D2. A transcription that contradicts both is wrong, not the reference.
4. Build the part as MIDI on Serum 2 with those exact notes.

Gate: note F1 against a manually verified 8-bar window. The part is then exact even though the
patch is not available.

### R4 — stabs as events, not as a pattern
Essentia `ChordsDetection` (with the -23 cent tuning correction already applied) over the guitar
stem, indexed by that stem's onsets -> events of (time, chord, duration).

Gate: event-level agreement with the reference, not just chord histogram agreement.

### R5 — tribal percussion per element
Onsets from the separated toms/percussion, one-shot candidates chosen **by measuring the whole
track** (the method that already beat the timbre-only ranking once).

Gate: per-element pattern agreement.

### R6 — fx and transitions
Detect risers and impacts at the section boundaries already known (bars 51, 107, 129, 178) and
place them from the library FX role. Today the replica has none and the reference has them.

### R7 — the vocal, decided out loud
The reference carries a vocal from bar 25 to 161. Options: (a) instrumental replica, declared as
such, (b) a library vocal chop, (c) the reference's own stem, which is fine for an internal A/B
but is sampling the record and cannot ship. Lucas decides; the plan does not choose silently.

### R8 — close replication, then and only then mix and master
Publish the per-element table. Mix and master start after this, not before.

## Where the ceiling sits under this framing

1. **Element timbre**: a library kick is not their kick. Measurable per element from now on.
2. **The synth patches**: Serum 2 exposes one parameter through the bridge, so a transcribed part
   can be rhythmically and harmonically exact and still sound different. This is the hard cap and
   it is a bridge problem, not a music problem.
3. **The vocal**: unresolved until R7.

## Tooling, with install status verified on this machine

| job | tool | status |
|---|---|---|
| per-drum separation (cheap) | `drumsep` 0.1.0 | installs in the repo venv, no torch |
| per-drum separation (strong) | `mdxnet-infer` + DrumSep 6-stem | torch already in the demucs runtime venv |
| drum transcription | `ADTOF-pytorch` | same torch venv |
| bass pitch | `torchcrepe` | same torch venv |
| chords, onsets, beats, tuning | Essentia | already in use |
| structure | MSAF | already in use |
| all-in-one transcription | `omnizart` | **fails to build on py3.12** (TensorFlow); would need a 3.11 venv |
| audio similarity | `frechet-audio-distance` | rejected: it compares distributions of clips, not one track to another |
