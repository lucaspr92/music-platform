# Replication-first: what each phase measured

Scope, set by Lucas: replicate the reference before touching mix or master. Mix and master are
out of scope and were not performed in these phases.

## The honest baseline

`similarity_to_reference.py` (the old scorer) averaged in `band_balance` and `energy_curve`, which
are mix: mastering moved it 0.7664 -> 0.8031 without changing a note. It is retired as a
replication figure and kept only as a mix report.

`similarity_replication.py` replaces it: per element, level-invariant, no loudness, no band balance,
no stereo width.

| element | presence | pattern | timbre | score |
|---|---|---|---|---|
| kick | 0.747 | 0.867 | 0.836 | **0.841** |
| hihat | 0.956 | 0.821 | 0.646 | **0.798** |
| percussion | 0.793 | 0.684 | 0.805 | **0.730** |
| bass (library loop) | 0.628 | 0.404 | 0.761 | **0.526** |
| clap | 0.938 | 0.269 | 0.743 | 0.488 (contaminated) |

**REPLICATION = 0.6765.** Acceptance tests pass 6/6, including the one the old scorer failed:
a level change does not move the score. White noise scores below 0.5.

## Phase results

**R0 — metric.** Built and tested. Two defects were found by its own control tests before it was
trusted: `presence` gave a free 1.0 to a constant noise floor, and MFCC-only timbre scored noise at
0.53 against a kick. Fixed with spectral flatness/centroid/rolloff in the timbre vector and
weights that do not let presence carry an element.

**R1 — per-instrument drum separation.** `drumsep` on PyPI is an empty package by a different
author (no `separate`, no submodules); the real project is not what that name installs. Used
`mdxnet-infer` with the DrumSep 6-stem checkpoint instead, in the torch runtime venv.
Gate passed: kick **2.99/bar** (band measurement said 2.65) and hats **11.44/bar** (band said
11.92). The split reproduces the independent counts, so it is usable.
Honest limit: the 6-stem model targets a drum kit, so hand percussion lands in the `snare` (7.89
onsets/bar) and `toms` (7.82) buckets. **The clap still cannot be isolated**, by either method, so
backbeat 2-and-4 remains an assumption. `ride`/`crash` sit at -52/-48 dB i.e. bleed.

**R2 — drum transcription.** Covered by R1's measurement: per-instrument onsets per bar, from
separated sources rather than from frequency bands.

**R3 — the bassline into notes.** `torchcrepe` was rejected after it returned a constant 31.7 Hz on
a pure 220 Hz sine: its weights load under this runtime's torch but inference does not work, and no
torchaudio release matches torch 2.14.1 here. `librosa.pyin` passes the same control (220.1 Hz,
voiced 1.0) and is used, with `frame_length=4096` because 2048 cannot hold two periods of 30 Hz.
Result: **885 notes over 147 bars (6.02/bar)**, median 0.174 s, every 16th occupied, matching the
Essentia grid. Splitting only on pitch change merged repeated notes; onsets are a second boundary.
The part now drives Serum 2 and the loop is removed (plan v9).
Limit: the PART is exact by transcription, the PATCH is not available, so timbre still differs.
pyin octave jumps are kept raw rather than "corrected" by taste.

**R4 — stabs as events.** The per-onset chord transcription was built, tested against a second
source, and **rejected**: median match 0.577 on the guitar stem, 0.570 on the full mix, and the two
disagree on 43% of events. A stab attack is transient and its chroma is smeared, so triad-per-onset
is not resolvable this way. What is resolvable: the harmonic movement per section. It is not
uniform: bars 0-25 read G/Bm/Em/E while Dm only dominates from bar 25 (weights 260 and 536).
The rhythm of the harmonic layer (191 events, 0.99/bar) is measured and usable.

**R5 — tribal percussion.** Measured, not yet re-palettised: score 0.730 with pattern 0.684. The
loops are 126 BPM warped to 128, which is a plausible source of the pattern drift.

**R6 — fx and transitions.** 1 riser (bars 121-129, +22.5 dB, into the drop at 129) and 20
transients at boundaries, of which 12 are strong (>= 6 dB). The ones recurring every 8 bars are
crashes or bar accents, flagged as such rather than counted as impacts. The replica currently has
none of them.

**R7 — the vocal.** Measured: present bars 32-162, **14.66 events per active bar**, median gap
0.116 s (one 16th), voiced ratio 0.326, median pitch 410 Hz. That is not a sung phrase: it is a
16th-note chop or high percussion leaking into the vocals bucket. Decision: **instrumental replica,
declared as such**. Sampling the record cannot ship, and a library chop cannot be validated against
a layer whose own nature is ambiguous.

**R8 — not reached.** Mix and master stay untouched, per the scope.

## Where the replica stands, and the hard cap

The parts that were measured and rebuilt: kick (0.841), hihat (0.798), percussion (0.730), and a
bass part now exact by transcription. The cap is not arrangement any more: it is
**sound design**. Serum 2 exposes one parameter through the bridge, so a transcribed part can be
rhythmically and harmonically exact and still not sound like theirs. Until the bridge enumerates
plugin parameters, element timbre is the ceiling and it is measurable per element from now on.

## v9 built in Live: the bass is the transcribed part

Built into a clean set and verified by readback: tempo 128.01, 9 REPLICA tracks, 119 clips,
bass on Serum 2 with **885 notes written and 885 read back**, the library bass loop removed.

**The score must not be read as an improvement.** Replication over the four audio elements is
0.7142, and the baseline over those same four elements was also 0.7142: `(0.841 + 0.488 + 0.798 +
0.730) / 4`. Nothing about the audio changed, because v9 changed only the bass. What changed is
that the bass is no longer scored at 0.526 as a loop: its part is now exact by transcription and
its audio cannot be rendered offline (no offline Serum), so it is reported as PART-ONLY instead of
being scored as silence. Anything that presents 0.7142 against 0.6765 as progress is comparing
different element sets.

## Still open, in order

1. **R5 percussion**: 0.730 with pattern 0.684, and the loops are 126 BPM warped to 128 — the
   likely cause of the drift. One-shots placed on measured onsets is the next measured attempt.
2. **R6 fx placement**: 1 riser and 12 strong transients are detected and nothing is placed yet.
3. **R4 harmony**: per-onset chords are rejected; section-level chords are usable from bar 25.
4. **R8**: mix and master, untouched by scope.
