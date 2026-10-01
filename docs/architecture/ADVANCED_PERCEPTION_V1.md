# ADVANCED_PERCEPTION_V1

Advanced perception is a read-only provider boundary above the existing
`MusicAnalysisPack`. It adds observations to the EvidenceGraph; it does not
replace Physical DSP, Music Analyzer, EvidenceGraph, Astra, or producer
planning.

```text
MusicAnalysisPack
        |
        +-- local-music-analyzer provider (available)
        +-- CLAP embedding provider (real, optional dependency)
        +-- semantic-ear provider (availability-gated / optional)
        |
        +--> PerceptionObservation[]
        +--> EvidenceGraph fusion
        +--> ReferenceIntentBundle (read-only)
```

The local provider exposes existing Analyzer facts through a replaceable
provider contract and adds bounded structural observations such as
`section.non_template_boundary`. It does not invent source identities or
convert observations into production instructions.

CLAP is implemented through the Transformers ClapModel boundary using the
laion/clap-htsat-unfused checkpoint, pinned by default to revision
`8fa0f1c6d0433df6e97c127f64b2a1d6c0dcda8a`. It is loaded lazily, runs in
inference mode, and is cached per model/revision/device. Audio is decoded as float32,
folded to mono, resampled to 48 kHz, split into non-overlapping 10-second
windows, and L2-normalized. Long-audio scalar embeddings are the normalized
mean of window embeddings; window-level embeddings retain time provenance.
Audio/text comparisons require matching provider, model, revision, and
dimension; otherwise the result is NOT_COMPARABLE.

If CLAP or the semantic-ear provider is unavailable, the result contains an
explicit provider limitation. The deterministic embedding stub is never
treated as semantic.
Contradictory provider observations remain `CONTRADICT` in fusion output;
the system never selects one as ground truth merely to make a result pretty.

`ReferenceIntentBundle` stores distinct `REFERENCE_STATE_TOKEN` identities,
optional role/feature bindings, provenance, and conflicts. It never merges
facts from different references and never contains raw audio.

## Comparing reference boundaries (pending Hermes validation)

`copilot.audio.reference_boundary_comparison.compare_reference_boundaries`
compares interior structural-region boundaries against optional human marks and
alternative providers on the same absolute quarter-note timeline. It reports
one-to-one matches, false cuts, missed marks and timing error; the fixed
32-bar measurement windows are never counted as sections. A reference with
no human-marked changes may provide an empty mark list. The comparison is
read-only and makes no judgment about musical quality or danceability.
`scripts/compare_reference_boundaries.py` accepts a persisted
`MusicAnalysisPack` and a marks file of the form
`{"marked_qn": [32, 64], "alternatives_qn": {"msaf": [31, 68]}}`.
It prints a JSON comparison; quarter notes are absolute within the pack's
reference timeline. An `alternatives_qn` entry is supplied by the caller;
the script does not install MSAF. For a local MSAF comparison, omit the
manually supplied `msaf` entry and pass `--msaf-python <isolated-python>` plus
`--scratch-root <existing-directory>`. The optional Python environment must
be provisioned separately on Hermes: upstream MSAF is MIT licensed but pulls
older `librosa`, `cvxopt`, `vmo` and other dependencies; mixing them into the
project's Python 3.12 / NumPy 2 environment is not assumed safe. No new base
dependency is added.

The adapter runs MSAF's Foote novelty boundaries over frame-synchronous MFCC
in a bounded local subprocess. It checks the original audio hash against the
pack, stages at most 120 seconds of the exact reference region as temporary
floating-point WAV in a scratch directory (Foote's self-similarity matrix
grows quadratically with duration), and isolates MSAF's own features/estimation writes
there. Original audio is never handed to MSAF as an output path. Output is
converted from seconds to the pack's absolute quarter-note timeline; labels
are deliberately ignored. Missing interpreter, insufficient scratch space,
bad provenance, subprocess failure and malformed boundaries are errors, not
local-analyzer success. The caller supplies an actual comparison annotation;
no provider wins without comparing it with the baseline on the same audio.

The local Music Analyzer is the baseline. `librosa` is already used for local
musical facts; MSAF (MIT) is only a prospective structural-boundary comparator.
On Hermes, compare the same references and lightly annotated section changes
before considering MSAF as a provider. Essentia (AGPL-3.0) is not an
integration candidate without a demonstrated gap and license review. No
comparison against real annotations, MSAF execution or Live validation has
been performed yet.

| Source | Boundary mechanism | Existing authority | Limit |
|---|---|---|---|
| Music Analyzer | Local energy/spectral novelty peaks with minimum separation | StructuralRegion in the persisted pack | A change point is not a musical verdict. |
| MSAF Foote (optional) | Self-similarity novelty over frame-synchronous MFCC | Alternative boundaries only in the comparison report | Quadratic memory; isolated environment and at most 120 s per run. |

Example invocation **later on Hermes**, after an isolated MSAF Python is
available and Lucas has marked a few reference sections:

```text
$env:PYTHONPATH = (Resolve-Path .\src).Path
python scripts\compare_reference_boundaries.py --pack "<reference-pack.json>" --marks "<human-marks.json>" --msaf-python "<isolated-msaf-python.exe>" --scratch-root "<existing-scratch-folder>"
```

The report retains the source hash, reference token, MSAF version and separate
boundary scores. MSAF never replaces `MusicAnalysisPack.sections` or becomes
an Ableton/write authority; only a validated improvement against the marked
baseline would justify further integration.

Current honest terminal state on this machine:

```text
ADVANCED_PERCEPTION_V1 = VERIFIED / FROZEN
```

The local Analyzer and CLAP providers are real and read-only. A controlled
project capture was analyzed with CLAP audio embeddings, real audio-to-audio
and audio-to-text comparisons, EvidenceGraph insertion, cache/provenance checks, and
MUSICAL_WRITES = 0. The optional semantic-ear provider remains unavailable;
that is an explicit limitation, not a fabricated description of source
function or prominence.

The selected checkpoint is recorded as Apache-2.0 in its model card. The
dependency is optional (pip install -e .[perception]); no model weights or HF
cache are committed to this repository.
