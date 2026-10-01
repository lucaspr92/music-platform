# CAPABILITY_MAP.md

Machine source: `python -m copilot.cli capabilities`
(`src/copilot/audio/capability_matrix_v1.py`). This file is the human map.

Statuses: `VERIFIED` | `COMMAND_VERIFIED` | `IMPLEMENTED` | `IN_FLIGHT` |
`WAITING` | `DEFERRED` | `UNSUPPORTED`. Frozen means reopen only for a
reproducible bug.

## Empty-template new project

`copilot.importing.new_project_v1.prepare_new_project(template_als=..., workspace=...)`
requires an explicitly selected, saved, parseable empty `.als` with no musical
tracks, clips, Return devices or project Samples. Empty Return tracks do not
count as musical tracks. It creates a unique manifest-backed directory within
the authorized workspace. `open_new_project(copy)` uses
`ensure_ableton_ready`, requires an exact Live path, identity, zero tracks and
fresh versioned bridge readback. `OPENED_EMPTY` does not mean finished.
`browser_load_advertised` reflects only the *connected bridge's handshake*,
not the installed Live version; real loading and clip readback remain unverified.
The vendored script advertises `device.physical_units_v1`, but includes a
unit on an individual parameter only when it attests native/display parity.
The producer additionally checks device identity, parameter range, and
post-write readback; the installed script and Live remain unverified.

`save_new_project_via_windows_ui(opened)` is an opt-in, bounded Windows-only
fallback. It requires a newly launched *Copilot-owned* PID, exact manifest and
bridge project path, the expected executable and uniquely identifiable enabled
Live window, verified foreground HWND/PID, and no visible same-process modal.
Only then does it send Ctrl+S; it requires changed file mtime **and** digest,
rechecks bridge/window, posts WM_CLOSE to that one window (never a broad kill),
waits for that process to exit, and relaunches via `ensure_ableton_ready` with
distinct PID, exact path, identity and matching track count. Any uncertain
observation yields `BLOCKED`. An already-open or unowned Live process cannot
use this fallback. No Save As or modal acceptance is automated.

`reopen_and_verify_saved_project(opened)` attempts readiness after the caller
has closed Live externally; it gates a changed on-disk `.als`, a different
known Live PID, exact reopened path, matching identity and fresh bridge readback.
`verify_saved_project(opened, reopened=...)` is the lower-level check for a
readiness report obtained separately. The bridge has no implemented `save`
handler. The UI fallback is not certified on a real Live window yet; it blocks
where platform focus, title, PID or persistence cannot be proven. Neither a
keystroke nor a changed file alone establishes a finished editable project.

## Live session

| ID | Status | Notes |
|---|---|---|
| `LIVE_SESSION_READINESS_V1` | VERIFIED / FROZEN | TCP + hello/request-id + light snapshot + identity. Port ≠ session. |
| Working copy policy | VERIFIED | Require a manifest-backed working copy; refuse unprotected originals. Fixture ≠ musical holdout. |
| `CROSS_PROJECT_BOOTSTRAP_V1` | VERIFIED | Infra only. Second run `NO_CHANGES_REQUIRED`. |
| `PROJECT_READY_V1` | VERIFIED | Identity → bootstrap → preflight. |
| `PLATFORM_HARDCODE_AUDIT_V1` | VERIFIED / FROZEN | Platform audit closed on the controlled working copy. Trial-modal acknowledgement, manifest-backed project discovery, `PROJECT_READY`, real capture, deterministic Astra-timeout `ABSTAIN`, terminal restoration, and zero musical writes all passed. |

## Observation

| ID | Status | Notes |
|---|---|---|
| TapProtocol 3 | VERIFIED | Slots 0/1/2. Rec isolation. |
| `LIVE_CAPTURE_BASELINE` | VERIFIED | Real WAV + provenance. Constant tempo. |
| Alignment | LIMITED | ±52 ms musical / ±4 ms inter-view. Not sample-accurate. |
| `CAPTURE_BATCHING_V1` | VERIFIED / FROZEN | Legacy width 2 on TapProtocol 3. `PRE_ROLL_QN=16`. |
| `CAPTURE_HOST_BASELINE_V1` | VERIFIED / FROZEN | Canonical parked state. Live Ext. In+Main is not baseline. |
| `CAPTURE_SCALABILITY_V2` | VERIFIED / FROZEN | Groove Rider 4-source one-pass + Main. Pool counts only `HOST_AVAILABLE`. |
| `RPC_OPTIMIZATION_V2` | VERIFIED / FROZEN | Freshness domains + bulk reads. |
| `ABLETON_MUTATION_PROTOCOL_V1` | VERIFIED / FROZEN | Live Groove Rider compound prepare/restore. Sequential fallback kept. Not for musical writes. |
| `PHYSICAL_DSP_V2` | VERIFIED / FROZEN | Factual measurement layer. No mix-quality judgments. FullMix/LowEnd wrapped, not rewritten. |
| `MUSIC_ANALYZER_V1` | VERIFIED / FROZEN | Real WAV and real Ableton captured-project analysis produce persisted read-only evidence through `AudioAnalysisInput`; staging-WAV finalization, provenance, and terminal restoration passed. Scope limitations remain explicit in the pack. |
| `DEEP_CAUSAL_V2` | VERIFIED / FROZEN | Facts → observed events → read-only SessionState signal graph → candidate hypotheses → before/during/after evaluation. Real controlled-working-copy validation generated 8 events, 88 candidates and preserved unresolved alternatives with zero writes. |
| `ADVANCED_PERCEPTION_V1` | VERIFIED / FROZEN | Local Analyzer plus real optional LAION CLAP audio/text embeddings and EvidenceGraph fusion are read-only. Semantic-ear remains an explicit unavailable limitation. |
| `BASS_MUSICAL_MODEL_V1` | VERIFIED / FROZEN | Deterministic symbolic model over reconciled authoritative MIDI: pitch material, intervals, rhythmic cells, motifs, phrases, and explicit bass-only tonal abstention. No writes or model calls. |
| `HARMONIC_UNDERSTANDING_V1` | IMPLEMENTED / READ-ONLY | Fuses authoritative bass MIDI with optional OTHER/MUSIC-stem chroma into ranked chord hypotheses, harmonic rhythm, tonal candidates, and bass-role relationships. Stem purity is not promoted to ground truth; ambiguous windows and global tonality abstain. |
| `HARMONIC_HUMAN_REVIEW_PACKAGE_V1` | READY / AWAITING HUMAN | Reuses the exact harmonic artifact, verifies source hashes, produces 8 deterministic master/OTHER/BASS listening windows plus typed evidence and static review files. All verdicts remain `PENDING`; no musical correctness is self-certified. |
| `HARMONIC_REVIEW_AUDIO_USABILITY_FIX_V1` | VERIFIED / HUMAN AUDIBILITY PENDING | Context-first review copies use the full reference source, verified QN-origin mapping, deterministic -3 dBFS constant gain, per-file signal audit, and valid local HTML paths. Human playback remains the next boundary. |
| `CANONICAL_MUSIC_MODEL_VIEW_V1` | IMPLEMENTED / READ-ONLY | Thin provenance-preserving view over existing evidence/domain artifacts. First consumer is the Astra producer-planner context; no analyzer, SafeWrite, State Trust, or Ableton semantics changed. |

## Runtime / agent

| ID | Status | Notes |
|---|---|---|
| `PRODUCER_RUNTIME_V1` | VERIFIED / FROZEN | `producer.analyze_project`. Agent command count = 1. |
| Local AbletonMCP + Serum 2/Soniq interface | IMPLEMENTED / HERMES QA PENDING | `copilot-mcp` / `python -m copilot.mcp`: official SDK v2 stdio server with 10 typed tools over existing integrations. No connection at import/start/list; one serialized Live connection. Default read-only, local working-copy write opt-in, current tokens and canonical ProductionCompiler/SafeWrite journals. Native-exposed plugin parameter writes only; Soniq WS reads are diagnostic and WS writes remain unexposed. No Live/model/audio/functional-test execution or complete ALS certification. |
| `EVIDENCE_SYSTEM_V2` | VERIFIED / FROZEN | Graph above immutable packs. Fusion keeps contradictions. Limitations propagate. |
| `SAFE_WRITE_FOUNDATION_V2` | VERIFIED / FROZEN | Generic PLAN→readback→KEEP/ROLLBACK. `SET_TRACK_VOLUME` only certified musical action. Analyze stays read-only. |
| `PRODUCER_EXECUTION_V1` | VERIFIED / FROZEN | `MusicPlan → ProductionCompiler → SafeWrite` six-action minimum passed Live readback/rollback on the controlled working copy. |
| `REFERENCE_BASS_VARIATION_8_BAR` | VERIFIED / BOUNDED | One real `pista Project` run: reconciled MIDI pitches, 21-note rhythmic variant, new Operator track, Session + Arrangement clip at QN 160–192, isolated WAV `HAS_SIGNAL` at −21.14 dBFS, saved working copy. The earlier silent 32-bar preview was reclassified `FAILED`; human musical verdict is pending. Studio discard after process restart is not certified. |
| `MUSICAL_UNDERSTANDING_COMPLETION_P0` | VERIFIED / BOUNDED | Existing bass MIDI, groove/relationship, motif/phrase, arrangement and harmony artifacts project through `CanonicalMusicModelViewV2`; `MusicalVariationIntent` is deterministic and NO_WRITE. One real revalidation passed through SafeWrite/readback/preview and was rolled back to the original final state. Melody remains explicitly `INSUFFICIENT_EVIDENCE`; musical quality is not auto-certified. |
| `PRODUCT_CREATIVE_LOOP_V1` | VERIFIED / BOUNDED | Existing Rose Bass evidence projects through `CanonicalMusicModelViewV2` and `VariationIntent`; symbolic validation and per-event provenance are persisted before the existing ProductionCompiler/SafeWrite path. The inherited real Ableton revalidation verified readback, preview, KEEP/DISCARD mechanics, rollback, 0 model calls, and 0 unsafe writes. Human musical quality remains a review boundary. |
| `MULTI_VARIATION_V1` | VERIFIED / BOUNDED | Produce accepts 1/3/5 and emits distinct deterministic reference-bound strategies with independent review records. Five real Ableton candidates reached READY with signal-bearing previews, authoritative readback, 20 SafeWrite writes, and one shared session authority. Human musical review remains pending. |
| `CREATIVE_REFERENCE_LOOP_V1` | IMPLEMENTED / NEEDS_REAL_3_VARIATION_REVALIDATION | Existing reference-bound A/B/C generation now has an explicit musical `SELECT` boundary, `MUSICAL_ACCEPTED / DISK_SAVE_REQUIRED` state, preview comparison, and reject-all rollback composition. No save workaround or second write authority was added. |
| `MULTI_ELEMENT_CREATIVE_LOOP_V1` | VERIFIED / BOUNDED | One real 8-bar candidate composed evidence-bound BASS (21 notes), generic transient-timed DRUMS (75 notes), and provisional selected-chord HARMONIC support (4 notes). A single ProductionCompiler→SafeWrite transaction created 3 Copilot tracks, 3 native Operator devices, and 3 Arrangement clips with 12/12 authoritative readbacks. Combined Main preview is non-silent and persisted; human musical review remains pending. |
| `PROJECT_PERSISTENCE_AND_SAVE_POLICY_V1` | IMPLEMENTED / BLOCKED_LIVE_API_UNAVAILABLE | SafeWrite outcome, musical KEEP/DISCARD, and working-copy disk persistence are separate typed states with path/hash/mtime/live-token evidence. Candidates remain `PENDING/CANDIDATE_PENDING`; the installed Live/Remote Script surface has no verified programmatic save API, so durable KEEP fails closed instead of being fabricated. |
| `LUCAS_CORE_INTEGRATION_V1` | VERIFIED / FROZEN | Lucas `build_plan_from_prompt` and `critique_track` are called through typed Core handoff; Astra remains internal to Lucas planning. Real Live bounded run verified 2 execution writes, 2 rollback mutations, authoritative readback, and post-analysis with 0 writes. `SAMPLE_LOAD` is Lucas input vocabulary; compiler emits canonical SafeWrite `LOAD_SAMPLE`. No Lucas-owned files were changed. |
| `PRODUCER_INTELLIGENCE_P0` | IMPLEMENTED / NOT VERIFIED | Typed `TrackSpec` and persistent producer decision ledger are available. Astra `track_spec` responses can drive validated dynamic sections; Core intent gating accepts their roles. No new Ableton authority or automatic musical write path. |
| Complete-track producer context | IMPLEMENTED / HERMES VALIDATION PENDING | Opt-in artistic brief, independent reference contexts, permission/hash-bound role shortlists, full-family TrackSpec validation and planner handoff. `producer-context` is read-only; a no-capture production is explicitly blocked before Live. Scoped revision proposals and human ALS supervision reuse the producer ledger, not a second write authority. No complete ALS delivery or 75% artistic effectiveness is certified. |
| Complete-track Arrangement score | IMPLEMENTED / HERMES VALIDATION PENDING | Explicit section/role/sample events, MIDI phrases and transitions; source-length-aware placements through existing MusicPlan builders, ProductionCompiler and SafeWrite. Source bindings are persisted and rechecked; sparse vocal/FX geometry is explicit. Distinct later phrases, trim/stretch and automation defer. No tests, audio, provider or Live execution performed for this slice; durable complete ALS and musical effectiveness remain unverified. |
| Multiple MIDI phrases per track | IMPLEMENTED / EXPERIMENTAL / HERMES VALIDATION PENDING | Default-off `--enable-midi-phrases` prepares distinct phrases in observed empty slots of a readback-owned MIDI track, preserving the verified sample instrument. Experimental compiler/intent and SafeWrite opt-in use existing bridge commands; production certification lists remain unchanged. Exact notes/length, stable slot bindings and preservation/rollback guards are implemented. Experimental actions cannot yield `COMPLETE`; no tests or Live/audio/model execution performed here. |
| `MIXING_MASTERING_EXECUTION_V1` | VERIFIED / FROZEN | Core-owned bounded mix/master execution maps volume, device load, and parameter intent through ProductionCompiler and SafeWrite. Generic active-region selection fixed the silent-region bug; real working-copy validation produced non-silent comparable captures, measurable mix consequence, 4 verified writes, authoritative readback, rollback, and 0 direct Lucas writes. Validation strategy provenance was `CONTROLLED_FIXTURE`, not `build_plan_from_prompt` output; this freezes execution/audio verification, not autonomous Lucas mix/master planning. |
| `AUTONOMOUS_PRODUCER_ALPHA_V1` | PRODUCTION_PASS_VERIFIED / REVISION_PROVIDER_LIMITED | Real Lucas planner â†’ MusicPlan â†’ ProductionCompiler â†’ SafeWrite â†’ Live readback â†’ capture â†’ Analyzer/CLAP/Evidence completed on the controlled working copy. 23 writes verified, 29 actions explicitly deferred, direct Lucas/Soniq writes 0, unresolved IN_DOUBT 0. Audio sample placement remains deferred because the bridge does not advertise `browser.load`; critique provider timed out, so no revision was invented. |
| `LUCAS_POST_CHANGE_CRITIQUE_PROVIDER` | PROVIDER_LIMITED | The unchanged Lucas critique contract and Core bounded provider failover are implemented and tested. Real mix/master critique attempts used persisted evidence, but the only configured compatible provider (`gpt-6-astra`) timed out; no verdict was fabricated. |
| `FOUNDATION_INTEGRATION_CHECKPOINT_V1` | VERIFIED | DSP → EvidenceGraph adapter. AnalyzeProject remains read-only. Not a new feature. |
| `PRODUCER_ANALYZE_V1` | COMMAND_VERIFIED | Debug CLI. Fixture `INSUFFICIENT_EVIDENCE` on empty arrangement. |
| `PRODUCER_RUN_V1` | COMMAND_VERIFIED | Autonomous volume only on development working copy. |
| M4L control contract | DOCUMENTED / FROZEN | ANALYZE/STATUS/DIAGNOSIS/PROPOSED ACTION/APPLY/ROLLBACK. APPLY = `SET_TRACK_VOLUME` only. |
| State Trust | VERIFIED (tests + Live smoke) | Tokens, refs, stale plan. See `docs/core/STATE_TRUST.md`. |

## Reasoning

| ID | Status | Notes |
|---|---|---|
| Grounding contract | VERIFIED / FROZEN | No invented measurements/entities. Fail-closed. |
| `ASTRA_REASONING_V2` | VERIFIED / FROZEN | EvidenceView in, grounded MusicDiagnosis out. Astra is not measurement authority. |
| Astra on Groove Rider | RUN (engine) / pack replay BLOCKED | Last Live run stayed `INSUFFICIENT_EVIDENCE`. No persisted pack under `fixtures/frozen/`. |
| MusicPlan gate | VERIFIED | Closed unless evidence + policy allow. |

## Portability

| ID | Status | Notes |
|---|---|---|
| `ENVIRONMENT_AUTONOMY_POLICY_V1` | VERIFIED / CURRENT-HOST | Runtime platform discovery, idempotent Remote Script provisioning, native launch, bridge/readiness lifecycle, and working-copy reconciliation. |
| Windows/macOS/Linux installer | IMPLEMENTED | No secrets. Remote Script + M4L; platform paths are discovered at runtime. |
| Second-machine proof | WAITING | Needs a friend machine. |
| `REGRESSION_V1` | VERIFIED | Offline supported-envelope suite. |

## Waiting on a second real song

| ID | Status |
|---|---|
| `CROSS_PROJECT_MUSICAL_VALIDATION_V1` | WAITING_FOR_EXTERNAL_SONG |
| Musical generalization | UNPROVEN |

Development fixtures and untitled sets are not that test.

## Explicitly not capabilities yet

EQ, compressor, general MIDI editing, general arrangement editing, unbounded plugin control,
general web UI generation without bound reference artifacts, silent mock success, collapsing producer statuses, autonomous writes on
an unvalidated external song, Music Flamingo semantic-ear descriptions, and
autonomous Lucas mix/master planning from a real producer request.

Those must register on Producer Runtime when they exist. They do not get a
sidecar orchestrator.

## Physical capture ceiling (honest)

Today: discovered pool; Groove Rider certified 4 sources + Main in one pass.
`PRE_ROLL_QN=16`. Alignment `LIMITED ±52 ms`.
