# New tech-house project (experimental)

## Product target and present boundary

The intended handoff is **one saved, reopenable, editable Ableton `.als` with
a complete Arrangement**, not an exported/rendered track. The producer should
use supplied reference stems as read-only style evidence and authorized
assets from the indexed library for actual drums, bass, musical parts, vocals,
and effects. It should arrange intro through outro, make its own musical
choices, and let Lucas supervise the project afterward. A request such as
"change the bass line" must revise only the identified musical role and
regions while preserving unrelated work, then verify and save the revised
working copy. A missing usable vocal asset or unavailable Live edit/save
capability is an essential blocker, not permission to claim a finished track.
Reference stems must not be copied into the composition.

The quality target is **75% of the first complete track retained by Lucas
after supervision**: assess retained versus substantially replaced musical
decisions across roles and sections, weighted by artistic importance and
Lucas's overall judgment, rather than by raw clip count or the fraction of
attempts accepted. This target has not been measured or achieved. No automatic
quality gate may assert Lucas's verdict or confuse a structurally verified
Arrangement with musical effectiveness.

The command below is an *experimental partial route*, not that product.
It cannot currently certify loading all requested assets, changing an
existing MIDI bass phrase, or saving/reopening through the installed Live
bridge. Existing capture WAVs are internal verification artifacts, not final
renders or a substitute for the `.als` handoff; using capture in the intended
no-render workflow requires a separate decision. Do not label a `DRAFT` or
an on-disk but unverified `.als` as finished.

### Agent interface direction: MCP-first

Lucas's requested agent interface is the Ableton MCP: the LLM should make
musical decisions through its tools, not manage CLI scripts or internal
execution steps. Core validation, journals and rollback belong behind that
interface, without per-clip human approvals or a second Live write authority.
An MCP-facing tool must use the existing DawAdapter execution path; changing
the interface alone does not reduce Live round trips or certify more actions.

Existing integration work is already present in `main` and must be reused:

- AbletonMCP Remote Script: `vendor/abletonmcp_remote_script/AbletonMCP`.
- Typed Ableton TCP adapter: `src/copilot/daw/ableton_tcp.py`.
- Serum 2 / Soniq plugin surface: `src/copilot/producer/soniq_surface.py`,
  including parameter schemas, patches, preset snapshots and an optional
  WebSocket bridge configured through `SONIQ_WS_URL`.
- Role-compatible preset selection: `src/copilot/producer/preset_catalog.py`.

### Local MCP entrypoint (implemented, Hermes QA pending)

`copilot.mcp` exposes those integrations to an MCP host over **stdio**, using
the official Python SDK v2 (`mcp>=2.2,<3`). One server provides both families
of tools and serializes their Live RPCs; it does not install a second Ableton
Remote Script, start a replacement Soniq bridge, launch Live, or call a model.
Importing, starting and listing tools do not connect to either backend.
The TCP connection opens on a Live tool call, is reused serially, and closes
on shutdown or a bridge failure. Reconnection creates a new session incarnation,
so old observations cannot authorize subsequent writes.

```powershell
python -m pip install -e ".[mcp,soniq]"
python -m copilot.mcp --help
```

`config/mcp.servers.example.json` is an `mcpServers` configuration example
for a client that launches `copilot-mcp` from the installed environment.
VS Code uses the equivalent `servers` key in its MCP configuration.
Use that environment's executable path if the client's PATH cannot find it.
The LLM host must support MCP tools; no model-provider key belongs in this
configuration. Host-side tool approvals remain the client's policy.

| Tools | Actual behavior |
|---|---|
| `runtime_status` | Local policy and configured integrations; no connection or Live claim. |
| `ableton_get_session`, `ableton_get_arrangement`, `ableton_search_browser` | Fresh Live reads, stable IDs, references, tokens and negotiated capabilities. No playback or capture. |
| `ableton_apply_plan` | Typed MusicPlan through ProductionCompiler and SafeWrite. Existing single-action/compound shapes only; maximum 64 input actions. Unsupported actions return `EXECUTION_DEFERRED`, not a raw-command fallback. |
| `soniq_read_surface` | Paginated parameters visible through Ableton and the existing completeness assessment. |
| `soniq_read_full_surface` | Paginated schema/value reads from the configured local `SONIQ_WS_URL`; diagnostic-only, not a verified Live project/device binding or write authority. |
| `soniq_set_parameter` | One parameter actually exposed by Ableton, matching name/index, native bounds, expected-value guard and Core readback/rollback. |
| `soniq_select_presets`, `producer_get_context` | Existing local preset catalog and reference/preference/sample context; no model or audio processing. |

The session tool omits note bodies by default (`include_notes=True` opts in)
and returns plugin parameter counts, not thousands of parameter values.
Parameter values are read through the paginated surface tools; omitted fields
are explicitly marked, not represented as empty musical material.

Enable musical writes **locally**, never through an LLM tool parameter:

```powershell
copilot-mcp --allow-writes --working-copy "<protected-working-copy.als>" --state-dir "<local-journal-folder>"
```

Those options belong in the client configuration's `args`. The copy must have
the existing protected working-copy manifest and match Live's fresh path.
Each mutation also requires the expectation returned by `ableton_get_session`;
project, incarnation and state tokens must still match. No per-clip approval
is introduced by this server. The same Core journal is used across requests;
unresolved recovery blocks new writes, including after a server restart.
MCP mutation errors retain the full Core result and never retry `IN_DOUBT`.

Optional local arguments are `--producer-context`, `--preset-catalog` and
`--sample-root`. Sample loads require a permission-bound context shortlist,
an authorized root and a fresh SHA-256 match; a relative sample path resolves
inside that root. `--sample-root` locates explicitly hash-authorized samples,
not permission to use unlisted files. Explicit device/preset URIs must be in
the local configured preset catalog; this is not permission for an audio file
or original `.als`. Plain device-name loads retain existing Core validation.
Soniq WS requires the already configured local bridge and the `soniq` extra;
absence, incompatible schema, read timeout or incomplete values are errors.
There is no invented port or successful empty-surface fallback.

Full Soniq WS patch/preset writes remain outside the certified production
path, as described in `architecture/LUCAS_DELTA_RECONCILIATION_V1.md`; the MCP
does not expose that bypass. Experimental multiple-MIDI-phrase writes are
also not enabled here. This interface does not render, capture, save/reopen
an ALS, or certify musical quality, 75% effectiveness, or complete delivery.
It does not replace the existing Alpha planner with a new internal agent loop;
the configured MCP host's LLM invokes these tools directly.

Hermes QA: run the pure contracts/schema cases in `tests/test_mcp_interface.py`,
then connect a real MCP client and verify initialize/list-tools/status before
Live reads. Use `copilot.runtime.ensure_ableton_ready(...)` for normal environment
readiness before certification. On a protected working copy, verify one parameter change/readback,
stale expectation refusal, wrong-project refusal, sample hash refusal and
recovery blocking. Exercise actual Soniq reads and missing-bridge/timeouts;
Soniq-only values must never become Live write preconditions. Functional tests,
models, Live and audio have not been executed for this slice.

## Read-only producer context (implemented, Hermes validation pending)

`copilot.producer.context` adds a strict `ProducerBrief` and a `ProducerContext`
above the existing `ReferenceContext`, `StyleContext` and `SampleSetContext`.
The planner receives confirmed preferences, provisional hypotheses, open
decisions and independently identified references. Artist direction is not an
audio measurement or a promise of employment. Lucas's agreed brief is an
explicit opt-in, never a default for other users.

Prepare a planning packet without Live, model calls, audio decoding or rendering:

```text
python -m copilot.cli producer-context --lucas-brief --context-output "<new-context.json>"
python -m copilot.cli producer-context --brief-file "<brief.json>" --reference-context "<reference-context.json>" --reference-context "<second-reference-context.json>" --sample-index "<index.json>" --sample-root "<authorized library>" --context-output "<new-context.json>"
```

The output path must be a new JSON file, not an input file. The first command
intentionally produces a blocked context skeleton until reference evidence and
an authorized indexed library are supplied. Exit code 2 with explicit blockers
is not a successful production. Exit code 0 means `PLANNING_READY`, never that
an Ableton project has been delivered. Reference inputs are existing
`ReferenceContext` JSON, not stems or a `MusicAnalysisPack` in a different schema;
the established analyzer/Core mapper supplies those contexts later in Hermes.

Authorize the musical use of a whole library once with
`--authorize-library-use` and `--sample-root`, or bind its path in
`ProducerBrief.authorized_library_root`. There is no need to approve each
sample individually. For selective permissions, `authorized_sample_sha256`
can list individual source hashes instead. An indexed filename or license
string alone does not grant rights. Preparation checks index/path identity
and shortlisted source bytes without decoding samples or hashing the entire
library on every request. The shortlist includes source
hashes, descriptor/classification confidence and permission state. One-shots
are not discarded for lacking a BPM; BPM filtering is for loops. Conga/clave
text matching is a retrieval hint, not proof of instrument identity.

The goal-mode planner can consume `production_context` and revalidates its
shortlist against the current index. It requires every requested family to
appear in the complete `TrackSpec` and refuses deterministic fallbacks.
Core now passes the actual read-only style/reference/sample context to the
planner rather than only a tempo/section summary.

For the experimental Live command, `--producer-context "<context.json>"`
connects the packet to the existing orchestration. **The default no-capture
brief is blocked before opening Live** with
`PRODUCER_NO_CAPTURE_DELIVERY_NOT_CERTIFIED`: the current quality gate requires
internal captures, and this change does not bypass it. The separate
`internal_audio_capture_authorized` field permits technical captures only if
the user actually authorizes them; it never enables a final render.
Load, phrase-edit, save/reopen and full-track musical certification are still
pending in Hermes. No capability is inferred from this context alone.

## Concrete Arrangement score (implemented, Hermes validation pending)

Complete-track contextual planning now requires `ArrangementScore` alongside
`TrackSpec`. It identifies each section-relative event, selected source hash,
duration and playback intent, plus actual MIDI notes and explicit transition
decisions. Vocal/FX events may be sparse, not full-section loops. The contract
rejects missing active roles, overlaps, partial MIDI repetitions, unselected
sources, notes outside phrases and a score that ends before the requested
timeline. This is producer intent, not observed audio or artistic approval.
The legacy provider schema and continuous-coverage gate remain unchanged
when complete-track context/score is absent.

The earliest MIDI phrase of each role becomes its initial sample-backed
material through existing MusicPlan builders. After verified sample loading,
the score compiler binds placements to fresh, identified slot-0 clips and
their observed lengths/notes. It requires negotiated `clip.create`,
`clip.delete` and `session.read`; it never probes unknown commands.
One duplication per source-length placement uses `length=None`, avoiding
the bridge's four-QN tiling assumption for longer clips. Compilation is
bounded to 2048 placement actions; this is a safety ceiling, not a latency
target. Sparse placements and longer observed loops should reduce RPC count.

The Alpha persists score decisions and bindings/deferred reasons in its
existing ledger/report before placement, then uses only
ProductionCompiler -> SafeWrite. The executor rechecks bound project,
incarnation, track, source clip and observable source state against its
already-required fresh snapshot, including tempo and supported 4/4 meter;
changed sources or timing defer instead of silently
rebinding. Reopened Arrangement geometry must match bound placements exactly,
including intentional spaces, and reach the requested duration.

Without the experimental opt-in below, distinct later MIDI phrases remain
`EXECUTION_DEFERRED`; source trimming/stretching and automation remain deferred
in either mode. A generic repeated phrase is not substituted for them.
Deferred actions, missing captures or
unverified save/reopen still prevent `COMPLETE`. This development does not
certify source loading, no-capture delivery, phrase replacement, musical
quality or the 75% effectiveness target.

Hermes QA cases are prepared in `tests/test_arrangement_score.py`, without
DAW/provider mocks or audio processing. They cover source-length placement,
sparse vocals, strict contracts, changed sources, deferred phrases/automation,
capability/material guards, geometry, action budgets, initial MIDI material
and provider schema references. **They have not been run here.** Hermes should
first run those pure cases together with context/supervision and existing
producer regressions; only afterward, on an authorized working copy, verify
sample loading, source readback, real Arrangement geometry and persistence.
No test pass or complete `.als` delivery is claimed by this change.

### Multiple MIDI phrases on one track (experimental, default off)

`--enable-midi-phrases` on `produce-tech-house` opts into a **Hermes QA path**.
It prepares distinct score phrases in other freshly observed empty Session
slots on the same Copilot-created MIDI track; the sample instrument is not
reloaded. Identical note/length phrases reuse one verified source. This flag
does not authorize existing-clip replacement, scene creation, arbitrary user
tracks, captures or saving; all their existing guards still apply.

Core ownership comes from the initial SafeWrite track-creation readback, and
instrument identity comes from its verified sample-load result. The initial
phrase must still match authoritative notes. Empty-slot inventory is decoded
from the bridge's existing complete `clip_slots` readback; absent/incomplete
inventory is unknown, not permission to invent slots. Exhaustion defers,
and preparation is bounded to 128 additional phrase creations.

`CREATE_MIDI_PHRASE` is separate from the frozen `CREATE_PATTERN` compound.
The default ProductionCompiler rejects it. The opt-in compiler returns
`COMPILED_EXPERIMENTAL` with an uncertified action and a distinct experimental
intent kind; SafeWrite requires separately supplied Core-owned track IDs and
negotiated `session.read`, `clip.create`, `clip.delete`, `clip.read_notes` and
`clip.write_notes`. The production certification allowlists are not expanded.
The existing adapter/Remote Script `create_clip` and `add_notes_to_clip`
commands are reused sequentially; no new vendor command, MCP authority,
Remote Script transaction system or speculative concurrent RPC is introduced.

Before each creation, Core checks preserved material plus any already verified
new clips; a human-occupied reserved slot blocks rather than being adopted.
The canonical SafeWrite pre-state and inverse exist before mutation. Readback
checks exact phrase content/length and stable clip ID, not just note count,
and preservation covers instruments, other clips and other tracks. Rollback
must restore the empty slot and preserved state. A partial/ambiguous write
remains `IN_DOUBT` under the existing lifecycle; it is not retried or labeled
successful. New clips reach placements only through verified phrase-to-slot
bindings; placements no longer assume slot 0.

Experimental execution rows remain labeled `HERMES_VALIDATION_PENDING`.
The delivery gate explicitly prevents them from producing `COMPLETE` even
when technical readbacks succeed. Certification, audio suitability, durable
ALS delivery and the 75% artistic target remain separate unresolved boundaries.
`tests/test_multiple_midi_phrases.py` contains additional pure QA cases,
**written but not run here**.

In Hermes, first run the pure score/material/compiler regressions and existing
SafeWrite/adapter regressions. Then use an authorized working copy to check
two different bass phrases with the same instrument, exact Session and
Arrangement note/geometry readbacks, occupied-slot refusal, preservation,
KEEP and explicit ROLLBACK, and interrupted-write recovery. Finally assess
save/reopen and musical suitability separately. No testing is performed in
this development workspace.

## Targeted revisions and human supervision

`copilot.producer.supervision.propose_scoped_revision` binds a musical proposal
to supplied project/audible/incarnation tokens, stable track IDs and QN regions.
It preserves existing producer decisions and stores the proposal in the
existing ledger; callers persist it explicitly with `ProducerStateStore`.
`validate_preserved_tracks` checks unaffected track/sample state and, by
default, target instruments. These checks do not attest Arrangement geometry
or the unedited part of a target phrase. A fresh authoritative snapshot,
certified phrase replacement, SafeWrite readback/rollback and verified save
remain mandatory before calling any proposed revision applied.

Record a human review of the exact saved project without manipulating Live:

```text
python -m copilot.cli producer-supervise --review-file "<review.json>" --reviewed-als "<reviewed.als>" --producer-state-root "<existing ledger directory>" --producer-session-id "<existing session ID>"
```

The typed `TrackSupervision` requires project identity, reviewed ALS SHA-256,
an overall judgment and elements labeled `KEEP`, `RETOUCH` or `REPLACE`.
The command checks the supplied file hash and the ledger's project identity.
`human_retained_fraction` is an optional judgment supplied by the human;
it is not calculated from clip count or inferred from model responses.
Recording 0.75 does not certify disk persistence, mark the project `COMPLETE`,
or convert a one-time preference into a global rule. The file hash identifies
the reviewed version even after subsequent manual edits.

Pure contract regression cases are provided in
`tests/test_complete_track_context.py` and `tests/test_project_supervision.py`.
They contain no mock DAW/provider or audio processing. Running them and the
related existing suites is deferred to Hermes under the current instruction;
syntax/import/CLI checks are not full runtime or musical certification.

Run only with an explicitly authorized, **saved empty** Ableton template,
an indexed sample library, and a separate destination directory:

```text
python -m copilot.cli produce-tech-house --template "<empty.als>" --sample-index "<index.json>" --sample-root "<library folder>" --workspace "<output folder>" --prompt-file "<super-prompt.txt>"
```

Optionally add `--reference "<licensed-reference.wav>"` for read-only,
aggregate groove/energy/mix comparison. It is not a source of copied
notes, melodies, or stems, and the command never exports a final WAV.

The UTF-8 super prompt must contain exactly one `BPM: 127` and one
`Duración: 64 compases` (alternatively `Duration: 2:00` or
`Duración: 2 minutos 30 segundos` or `Duración: 120 segundos`).
Duration in time is quantized to the nearest
4/4 bar; the maximum deviation is half a bar. Fill in both values before
running. For example:

```text
BPM: 127
Duración: 64 compases
Produce an original tech-house groove. Choose kick, bass, percussion and a
distinctive hook from the indexed samples. Compare two or three candidates
per role and explain the choice and rejections using factual timbre and
transient evidence; do not claim semantic listening without a listener.
Develop kick/bass/percussion first. Describe the foreground, low-end owner
and deliberately unoccupied space for each section, with intentional
transitions and a variation hypothesis. Correct one prioritized problem at a
time and revert changes without measured improvement. Apply EQ Eight only
for a specific measured hypothesis, and Limiter last with a verified ceiling.
```

The supplied longer Spanish super prompt can be used as advisory creative
copy, but its BPM and duration placeholders must be replaced with exactly
one explicit value each. Its requests never authorize arbitrary model code
or override the runtime's measurements and fail-closed gates.

The command validates BPM/duration **before** opening Live, refuses model
fallbacks and unknown sample selections, creates a unique manifest-backed
copy, and only uses MusicPlan -> ProductionCompiler -> SafeWrite for musical
actions. It records its report under `logs/producer/<run-id>/report.json`.
The `.als` is the intended editable deliverable; internal WAV captures are
evidence, never a final export.

**Evidence and decisions:** each indexed candidate gets a reproducible
one-bar preview at matched peak and an equally gained single-bar context
proxy with a fixed Kick (or Bass) reference. Source SHA, type, BPM/pitch
confidence, transient/crest/spectrum and license state are reported along
with chosen and rejected candidates. The previews are **not** full groove
auditions in Live, and descriptors cannot judge musical fit or guarantee
license; if no semantic listening provider exists the report does not say
Astral heard them. A structured `producer_criteria` separates artistic
preferences and perceptual estimates from measured facts, checks section
roles against the actual plan and cites only selected sample hashes.
Section listening captures opening, middle and ending windows on both sides
of transitions; every window must have independent identity, timing,
non-silent, non-clipping evidence. An intentional silent break currently
cannot pass `COMPLETE` without a separate certified intentional-silence
contract.
**Phrase MIDI boundary:** the current compound compiler can create an editable
pattern on a *new* MIDI track, but it cannot atomically replace a phrase on an
existing sample-backed track while preserving its instrument and independently
verifying Arrangement playback/rollback. Repeated copies of slot 0 are not a
new phrase. If a model-specified section requests a MIDI `variation`, the
producer records `PHRASE_MIDI_VARIATION_NOT_CERTIFIED` as an essential
deferred action, saves the verified draft where possible and returns `DRAFT`,
never `COMPLETE`. A planning `variation_hypothesis` alone is not evidence that
a variation was rendered.

EQ Eight APPLY requires a stated `reduce_low_band` or `reduce_high_band`
hypothesis, and Limiter APPLY requires `reduce_peak`. The initial hypothesis
is based on sample descriptors, **not** a claim that the produced Live mix
was already heard. SafeWrite physical
readback and fresh before/after audio are followed by objective spectral or
crest comparison with limited transient loss; an unchanged or worsened
result rolls back. These are factual proxies, **not** proof of artistic
quality. Revisions target the highest-priority critique issue, compare
whole-section windows and stop after at most three attempts; provider
absence remains DRAFT. When comparable mix captures exist, an optional
`blind_review/review.json` contains randomized A/B previews normalized
to −18 dBFS RMS and six human questions; a separate
`answer_key.json` holds capture provenance. The human review is posterior
and does not gate technical completion. `artistic_quality_human_verified`
remains `false` until a human actually submits a review.

Example report fields (illustrative, not a completed Live run):

```json
{
  "status": "DRAFT",
  "MUSICAL_WRITES": 0,
  "sample_comparisons": {},
  "producer_criteria": null,
  "section_captures": {},
  "quality_gate": {
    "status": "DRAFT",
    "artistic_quality_human_verified": false
  },
  "blockers": ["BRIDGE_CAPABILITY_BROWSER_LOAD_UNAVAILABLE"]
}
```

**Current certification boundary:** the template must already have the
requested BPM; a tempo-change MusicPlan action is not certified. The installed
bridge may omit `browser.load` and exposes no verified Save command. Physical
EQ Eight/Limiter units must be reported by the connected bridge with
`device.physical_units_v1`; older installed scripts will defer requested
adjustments. The model may request one EQ Eight adjustment on an active
track and one Main Limiter ceiling adjustment with a typed physical value.
Only a native scale matching Live's own displayed unit permits the existing
MusicPlan → ProductionCompiler → SafeWrite path. Fresh parameter readback,
before/after Main captures and a second verified Save/reopen are required.
No recipe device chain is silently substituted. The blank template may have
empty Return tracks but no musical tracks, clips or Return devices.
A partial Live pass is `DRAFT`, and a failed
preflight is `BLOCKED`; neither is a finished track. No completion or
artistic-quality claim may be inferred from a file existing or from a
successful structural readback alone. A verified Save/reopen, audible
section-by-section and source checks, a bounded typed volume correction loop,
and real Live certification are required before the quality gate can return
`COMPLETE`. Live certification has not been performed on this change.
Arbitrary BPM remains blocked unless the empty saved template already matches
the requested BPM. Real Live proof needs a disposable authorized empty set,
an indexed licensed sample root, an available model provider, and a connected
bridge advertising actual `browser.load` and `device.physical_units_v1` when
those actions are requested. Two distinct prompt/project runs have not been
performed on this host; the offline mocks are not substitutes.
