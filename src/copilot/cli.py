from __future__ import annotations

import argparse
import json
from pathlib import Path

from copilot.agent.journal import DurableJournal
from copilot.agent.slices import (
    connect_live,
    connect_mock,
    run_create_c3_clip,
    run_undo_last,
)
from copilot.agent.tools import AgentTools
from copilot.agent.transactions import TransactionManager
from copilot.daw.ableton_tcp import AbletonTcpAdapter
from copilot.daw.ableton_tcp import DEFAULT_HOST as LIVE_DEFAULT_HOST
from copilot.daw.ableton_tcp import DEFAULT_PORT as LIVE_DEFAULT_PORT
from copilot.daw.adapter import DawError
from copilot.daw.detect import detect_ableton, live_block_status, write_detection
from copilot.daw.session_ready_v1 import (
    SESSION_READY,
    probe_session_ready,
    revalidate_project,
)
from copilot.audio.live2 import run_live2
from copilot.audio.live21 import run_live21
from copilot.audio.live22 import run_live22
from copilot.audio.live22e import run_live22e
from copilot.audio.live3 import run_live3
from copilot.audio.live3r import run_live3r
from copilot.audio.live3r_perf import run_live3r_perf
from copilot.audio.live3r_perf2 import run_live3r_perf2
from copilot.audio.live3r_perf3 import run_live3r_perf3
from copilot.audio.live3r_trust import run_live3r_trust
from copilot.audio.live3r_prod import run_live3r_prod
from copilot.audio.capture_alignment import run_alignment_certification
from copilot.audio.live_state import run_live_state
from copilot.daw.install_remote_script import install_remote_script
from copilot.logging_setup import configure_logging

# Production routing contracts (MusicPlan behavior not implemented here).
CANONICAL_PRE_WRITE_PATH = (
    "session-diagnose -> session-run1 -> session-run1-fullmix -> "
    "session-run1-astra-r2 -> session-run1-causal-trace -> "
    "session-run1-source-audio -> session-run1-strum-device"
)
CANONICAL_WRITE_PATH = "production-write"
LAB_COMMANDS = frozenset(
    {
        "mock-slice1",
        "live2",
        "live21",
        "live22",
        "live22e",
        "live3",
        "live3r",
        "live3r-perf",
        "live3r-perf2",
        "live3r-perf3",
        "live3r-trust",
        "live3r-prod",
        "live3r-align",
    }
)
CANONICAL_COMMANDS = (
    "install",
    "doctor",
    "onboard-project",
    "project-ready",
    "project-bootstrap",
    "analyze-project",
    "producer-analyze",
    "producer-run",
    "produce-tech-house",
    "producer-context",
    "producer-supervise",
    "cross-project-validate",
    "import-project",
    "regression-v1",
    "capabilities",
    "performance-report",
    "studio",
)
HELP_EPILOG = """
Canonical supported envelope:
  install                  (Windows or macOS local runtime; no musical writes)
  import-project "<folder>"
  analyze-project "<folder>"   (Producer Runtime: one call, owns the whole flow)
  doctor
  onboard-project          (alias of project-ready)
  project-bootstrap
  producer-analyze         (low-level debug path; prefer analyze-project)
  producer-run --mode analyze|autonomous
  producer-context         (read-only complete-track planning context; no Live/audio/model)
  producer-supervise       (record human ALS review in the producer ledger; no Live writes)
  cross-project-validate
  regression-v1
  capabilities
  performance-report       (read-only latency report; no musical writes)
  studio --port 8765      (local Music Studio vertical slice)

Lab runners require --lab and are not the supported envelope.
Live is ready only after SESSION_READY (not a listening port).
""".strip()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="AI Music Production Copilot — supported envelope CLI",
        epilog=HELP_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "command",
        choices=[
            "detect",
            "install-script",
            "probe",
            "slice1",
            "undo",
            "mock-slice1",
            "live2",
            "live21",
            "live22",
            "live22e",
            "live3",
            "live3r",
            "live3r-perf",
            "live3r-perf2",
            "live3r-perf3",
            "live3r-trust",
            "live3r-prod",
            "live3r-align",
            "live-state",
            "reason-eval",
            "reason-real",
            "reason-revalidate",
            "session-diagnose",
            "session-run1",
            "session-run1-astra",
            "session-run1-fullmix",
            "session-run1-astra-r2",
            "session-run1-grounding-causal",
            "session-run1-causal-trace",
            "session-run1-source-audio",
            "session-run1-strum-device",
            "session-seek-trust",
            "session-arrangement-start",
            "human-eval",
            "capture-journal-recover",
            "production-write",
            "project-bootstrap",
            "project-ready",
            "onboard-project",
            "producer-analyze",
            "analyze-project",
            "producer-run",
            "produce-tech-house",
            "producer-context",
            "producer-supervise",
            "cross-project-validate",
            "import-project",
            "install",
            "uninstall-copilot",
            "doctor",
            "regression-v1",
            "capabilities",
            "performance-report",
            "studio",
        ],
    )
    parser.add_argument("eval_argv", nargs="*", default=[])
    parser.add_argument("--prompt-file", default=None, help="produce-tech-house: UTF-8 super prompt file")
    parser.add_argument("--template", default=None, help="produce-tech-house: saved empty .als template")
    parser.add_argument("--sample-index", default=None, help="produce-tech-house: indexed sample-library JSON")
    parser.add_argument("--sample-root", default=None, help="produce-tech-house: authorized sample-library directory")
    parser.add_argument("--workspace", default=None, help="produce-tech-house: authorized output directory")
    parser.add_argument("--reference", default=None, help="produce-tech-house: optional read-only external comparison audio")
    parser.add_argument("--brief-file", default=None, help="producer-context: typed ProducerBrief JSON")
    parser.add_argument("--lucas-brief", action="store_true", help="producer-context: explicitly use Lucas's agreed direction")
    parser.add_argument("--authorize-library-use", action="store_true", help="producer-context: explicitly permit musical use of the supplied sample root")
    parser.add_argument("--reference-context", action="append", default=[], help="producer-context: existing ReferenceContext JSON (repeatable)")
    parser.add_argument("--context-output", default=None, help="producer-context: output JSON path")
    parser.add_argument("--producer-context", default=None, help="produce-tech-house: prepared complete-track context JSON")
    parser.add_argument(
        "--enable-midi-phrases", action="store_true",
        help="produce-tech-house: opt in to experimental empty-slot MIDI phrases; Hermes QA pending",
    )
    parser.add_argument("--review-file", default=None, help="producer-supervise: human TrackSupervision JSON")
    parser.add_argument("--producer-state-root", default=None, help="producer-supervise: existing ProducerStateStore directory")
    parser.add_argument("--producer-session-id", default=None, help="producer-supervise: existing producer session ID")
    parser.add_argument("--reviewed-als", default=None, help="producer-supervise: exact saved ALS evaluated by the human")
    parser.add_argument(
        "--allow-ui-save", action="store_true",
        help="produce-tech-house: explicitly allow one identity-verified Windows UI save/relaunch",
    )
    parser.add_argument("--track", default="AI Test")
    parser.add_argument("--log", default="logs/copilot.log")
    parser.add_argument(
        "--gate",
        choices=["smoke", "mini", "full"],
        default="smoke",
        help="reason-real only: smoke=1 call, mini=A-F x1, full=A-F x5",
    )
    parser.add_argument("--run", dest="eval_run", default=None)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument(
        "--studio-data-dir",
        default=None,
        help="studio: durable local data directory (default COPILOT_STUDIO_DATA_DIR or runtime/music-studio)",
    )
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument(
        "--lab",
        action="store_true",
        help="Required for legacy/lab capture runners (live3r*, mock-slice1, live2*).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="production-write: validate/compile only; ZERO musical mutations.",
    )
    parser.add_argument(
        "--plan",
        default=None,
        help="production-write: path to MusicPlan JSON.",
    )
    parser.add_argument(
        "--prepare-controlled-plan",
        action="store_true",
        help="production-write: build CONTROLLED_ENGINEERING_VALIDATION volume plan + dry-run.",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=-0.02,
        help="controlled plan volume delta in ableton_volume units (default -0.02).",
    )
    parser.add_argument(
        "--target-track",
        default=None,
        help="controlled plan target track name (default: first non-capture audio/midi track).",
    )
    parser.add_argument(
        "--mode",
        choices=["analyze", "autonomous"],
        default=None,
        help="producer-run: analyze (read-only) or autonomous.",
    )
    parser.add_argument(
        "--region",
        default=None,
        help="analyze-project / producer-analyze / producer-run: optional region id.",
    )
    parser.add_argument(
        "--start-qn",
        type=float,
        default=None,
        help="producer-analyze / producer-run: explicit region start.",
    )
    parser.add_argument(
        "--end-qn",
        type=float,
        default=None,
        help="producer-analyze / producer-run: explicit region end.",
    )
    parser.add_argument(
        "--trace-performance",
        action="store_true",
        help=(
            "producer-analyze / production-write: record an ExecutionSpan trace "
            "of the run (logs/performance/) including every Ableton round trip. "
            "Measurement only; does not change what the run does."
        ),
    )
    parser.add_argument(
        "--live-host",
        default=LIVE_DEFAULT_HOST,
        help="performance-report: Ableton Remote Script host.",
    )
    parser.add_argument(
        "--live-port",
        type=int,
        default=LIVE_DEFAULT_PORT,
        help="performance-report: Ableton Remote Script port (not the eval server --port).",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="performance-report: samples per measured operation (default 5).",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="performance-report: skip live Ableton reads.",
    )
    parser.add_argument(
        "--remove-venv",
        action="store_true",
        help="uninstall-copilot: also delete repo .venv",
    )
    parser.add_argument(
        "--remove-config",
        action="store_true",
        help="uninstall-copilot: also delete repo .env",
    )
    parser.add_argument(
        "--execute-controlled-write",
        action="store_true",
        help=(
            "production-write: run CONTROLLED_ENGINEERING_VALIDATION write+rollback loop. "
            "Not autonomous musical improvement."
        ),
    )
    parser.add_argument(
        "--source-plan",
        default="plan_82608749d172",
        help="source dry-run plan_id to link (immutable; not executed).",
    )
    parser.add_argument(
        "--source-envelope",
        default="env_5886f4fb0b",
        help="source dry-run envelope_id to link (immutable; not executed).",
    )
    parser.add_argument(
        "--expected-before",
        type=float,
        default=0.75,
        help="hard precondition: live target volume must match before write.",
    )
    parser.add_argument(
        "--audible-effect-verification",
        action="store_true",
        help=(
            "production-write: AUDIBLE_EFFECT_VERIFICATION_V1 "
            "(capture BEFORE/AFTER around controlled volume write + rollback)."
        ),
    )
    parser.add_argument(
        "--audible-effect-verification-v2",
        action="store_true",
        help=(
            "production-write: AUDIBLE_EFFECT_VERIFICATION_V2 "
            "(baseline-characterize -> freeze high-SNR SET_TRACK_VOLUME -> one write)."
        ),
    )
    parser.add_argument(
        "--first-autonomous-musical-improvement",
        action="store_true",
        help=(
            "production-write: FIRST_AUTONOMOUS_MUSICAL_IMPROVEMENT_V1 "
            "(one blind holdout observe->diagnose->gate->optional SET_TRACK_VOLUME)."
        ),
    )
    parser.add_argument(
        "--autonomous-musical-evaluation-v2",
        action="store_true",
        help=(
            "production-write: AUTONOMOUS_MUSICAL_EVALUATION_V2 "
            "(preselected HOLDOUT_320_352 blind evaluation)."
        ),
    )
    parser.add_argument(
        "--autonomous-musical-evaluation-v3",
        action="store_true",
        help=(
            "production-write: AUTONOMOUS_MUSICAL_EVALUATION_V3 "
            "(new independent holdout + arrangement-active source isolation)."
        ),
    )
    parser.add_argument(
        "--arrangement-active-source-isolation",
        action="store_true",
        help=(
            "production-write: ARRANGEMENT_ACTIVE_SOURCE_ISOLATION_V1 "
            "(Post Mixer views for arrangement-active sources; observation only)."
        ),
    )
    args = parser.parse_args(argv)
    logger = configure_logging(Path(args.log))
    evidence = Path("logs")
    evidence.mkdir(parents=True, exist_ok=True)

    if args.command in LAB_COMMANDS and not args.lab:
        print(
            json.dumps(
                {
                    "status": "LAB_FLAG_REQUIRED",
                    "command": args.command,
                    "detail": (
                        "Legacy/lab runners require --lab. "
                        f"CANONICAL_PRE_WRITE_PATH={CANONICAL_PRE_WRITE_PATH}; "
                        f"CANONICAL_WRITE_PATH={CANONICAL_WRITE_PATH}"
                    ),
                    "MUSICAL WRITES": 0,
                },
                indent=2,
            )
        )
        return 2

    if args.command == "production-write":
        return _production_write(
            evidence,
            logger,
            dry_run=bool(args.dry_run),
            plan_path=args.plan,
            prepare_controlled=bool(args.prepare_controlled_plan),
            execute_controlled=bool(args.execute_controlled_write),
            audible_effect=bool(args.audible_effect_verification),
            audible_effect_v2=bool(args.audible_effect_verification_v2),
            autonomous_improvement=bool(args.first_autonomous_musical_improvement),
            autonomous_evaluation_v2=bool(args.autonomous_musical_evaluation_v2),
            autonomous_evaluation_v3=bool(args.autonomous_musical_evaluation_v3),
            arrangement_source_isolation=bool(args.arrangement_active_source_isolation),
            trace_performance=bool(args.trace_performance),
            delta=float(args.delta),
            target_track=args.target_track,
            source_plan_id=str(args.source_plan),
            source_envelope_id=str(args.source_envelope),
            expected_before=float(args.expected_before),
        )

    if args.command == "project-bootstrap":
        return _project_bootstrap(evidence, logger)
    if args.command in {"project-ready", "onboard-project"}:
        return _project_ready(evidence, logger)
    if args.command == "analyze-project":
        return _analyze_project(
            evidence,
            logger,
            args.eval_argv,
            region_id=args.region,
            start_qn=args.start_qn,
            end_qn=args.end_qn,
        )
    if args.command == "producer-analyze":
        return _producer_analyze(
            evidence,
            logger,
            region_id=args.region,
            start_qn=args.start_qn,
            end_qn=args.end_qn,
            trace_performance=bool(args.trace_performance),
        )
    if args.command == "producer-run":
        return _producer_run(
            evidence,
            logger,
            mode=args.mode,
            region_id=args.region,
            start_qn=args.start_qn,
            end_qn=args.end_qn,
        )
    if args.command == "produce-tech-house":
        return _produce_tech_house(
            evidence,
            prompt_file=args.prompt_file,
            prompt_words=args.eval_argv,
            template=args.template,
            sample_index=args.sample_index,
            sample_root=args.sample_root,
            workspace=args.workspace,
            reference=args.reference,
            allow_ui_save=args.allow_ui_save,
            producer_context_file=args.producer_context,
            enable_midi_phrases=args.enable_midi_phrases,
        )
    if args.command == "producer-context":
        return _prepare_producer_context(
            brief_file=args.brief_file, lucas_brief=args.lucas_brief,
            reference_context_files=args.reference_context,
            sample_index=args.sample_index, sample_root=args.sample_root,
            output=args.context_output,
            authorize_library_use=args.authorize_library_use,
        )
    if args.command == "producer-supervise":
        return _record_producer_supervision(
            review_file=args.review_file, state_root=args.producer_state_root,
            session_id=args.producer_session_id, reviewed_als=args.reviewed_als,
        )
    if args.command == "cross-project-validate":
        return _cross_project_validate(evidence, logger)
    if args.command == "import-project":
        return _import_project(evidence, logger, args.eval_argv)
    if args.command == "install":
        from copilot.installing.second_machine_installer_v1 import run_installer

        report = run_installer(evidence=evidence)
        print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
        logger.info("install status=%s remote=%s m4l=%s", report.get("status"), report.get("REMOTE_SCRIPT"), report.get("M4L RUNTIME"))
        return 0 if report.get("status") == "VERIFIED" else 2
    if args.command == "uninstall-copilot":
        from copilot.installing.second_machine_installer_v1 import uninstall_copilot_owned

        report = uninstall_copilot_owned(
            remove_venv=bool(args.remove_venv),
            remove_config=bool(args.remove_config),
        )
        print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
        return 0 if report.get("status") == "VERIFIED" else 2
    if args.command == "doctor":
        return _doctor(evidence, logger)
    if args.command == "regression-v1":
        return _regression_v1(evidence, logger)
    if args.command == "capabilities":
        return _capabilities()
    if args.command == "performance-report":
        return _performance_report(
            evidence,
            logger,
            host=args.live_host,
            port=args.live_port,
            repeats=int(args.repeats),
            include_live=not args.offline,
        )

    if args.command == "studio":
        from copilot.studio.server import run_server

        return run_server(
            host=args.host,
            port=args.port,
            data_dir=Path(args.studio_data_dir) if args.studio_data_dir else None,
        )

    if args.command == "capture-journal-recover":
        return _capture_journal_recover(evidence, logger)

    if args.command == "detect":
        detection = detect_ableton()
        write_detection(evidence / "ableton_detect.json", detection)
        print(json.dumps(detection.to_dict(), indent=2))
        return 0 if detection.found else 2

    if args.command == "install-script":
        result = install_remote_script()
        (evidence / "remote_script_install.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
        print(json.dumps(result, indent=2))
        return 0 if result["status"] in {"INSTALLED", "ALREADY_CURRENT", "UPDATED"} else 2

    if args.command == "probe":
        return _probe(evidence, logger)

    if args.command == "live2":
        return _live2(evidence, logger)

    if args.command == "live21":
        return _live21(evidence, logger)

    if args.command == "live22":
        return _live22(evidence, logger)

    if args.command == "live22e":
        return _live22e(evidence, logger)

    if args.command == "live3":
        return _live3(evidence, logger)

    if args.command == "live3r":
        return _live3r(evidence, logger)

    if args.command == "live3r-perf":
        return _live3r_perf(evidence, logger)

    if args.command == "live3r-perf2":
        return _live3r_perf2(evidence, logger)

    if args.command == "live3r-perf3":
        return _live3r_perf3(evidence, logger)

    if args.command == "live3r-trust":
        return _live3r_trust(evidence, logger)

    if args.command == "live3r-prod":
        return _live3r_prod(evidence, logger)

    if args.command == "live3r-align":
        return _live3r_align(evidence, logger)

    if args.command == "live-state":
        return _live_state(evidence, logger)

    if args.command == "reason-eval":
        return _reason_eval(evidence, logger)

    if args.command == "reason-real":
        return _reason_real(evidence, logger, gate=args.gate)

    if args.command == "reason-revalidate":
        return _reason_revalidate(evidence, logger)

    if args.command == "session-diagnose":
        return _session_diagnose(evidence, logger)

    if args.command == "session-run1":
        return _session_run1(evidence, logger)

    if args.command == "session-run1-astra":
        return _session_run1_astra(evidence, logger)

    if args.command == "session-run1-fullmix":
        return _session_run1_fullmix(evidence, logger)

    if args.command == "session-run1-astra-r2":
        return _session_run1_astra_r2(evidence, logger)
    if args.command == "session-run1-grounding-causal":
        return _session_run1_grounding_causal(evidence, logger)
    if args.command == "session-run1-causal-trace":
        return _session_run1_causal_trace(evidence, logger)
    if args.command == "session-run1-source-audio":
        return _session_run1_source_audio(evidence, logger)
    if args.command == "session-run1-strum-device":
        return _session_run1_strum_device(evidence, logger)

    if args.command == "session-seek-trust":
        return _session_seek_trust(evidence, logger)

    if args.command == "session-arrangement-start":
        return _session_arrangement_start(evidence, logger)

    if args.command == "human-eval":
        from copilot.human_eval.cli import handle_human_eval

        return handle_human_eval(args, evidence)

    if args.command == "mock-slice1":
        daw = connect_mock()
        tools = AgentTools(daw, TransactionManager(daw))
        result = run_create_c3_clip(tools, args.track)
        result["verification_class"] = "MOCK_VERIFIED"
        result["backend"] = "mock"
        print(json.dumps(result, indent=2, default=str))
        return 0

    try:
        daw = connect_live()
    except DawError as exc:
        detection = detect_ableton()
        status = live_block_status(detection)
        payload = {
            "verification_class": status,
            "status": status,
            "error": str(exc),
            "required_action": _manual_control_surface_action()
            if status == "MANUAL_CONFIGURATION_REQUIRED"
            else None,
        }
        print(json.dumps(payload, indent=2))
        return 2
    store = evidence / "agent_transactions.json"
    journal = DurableJournal(evidence / "agent_journal.jsonl")
    txns = TransactionManager(daw, journal=journal)
    txns.load(store)
    recovery = txns.recovery_scan()
    if recovery:
        logger.info("journal recovery scan: %s", recovery)
    tools = AgentTools(daw, txns)
    logger.info("backend=ableton-tcp verification_class=LIVE")
    if args.command == "slice1":
        result = run_create_c3_clip(tools, args.track)
        result["verification_class"] = "LIVE_VERIFIED"
    else:
        result = run_undo_last(tools, args.track)
        result["verification_class"] = "LIVE_VERIFIED"
    txns.save(store)
    result["backend"] = "ableton-tcp"
    (evidence / f"live_{args.command}.json").write_text(
        json.dumps(result, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, default=str))
    return 0


def _live3r(evidence: Path, logger) -> int:
    detection = detect_ableton()
    adapter = AbletonTcpAdapter()
    try:
        adapter.connect()
        result = run_live3r(adapter, evidence)
        adapter.disconnect()
        print(json.dumps(
            {
                "phase": "LIVE-3R",
                "REAL_SESSION_LOW_END_DIAGNOSIS": result.get(
                    "REAL_SESSION_LOW_END_DIAGNOSIS"
                ),
                "LOW_END_DIAGNOSIS_FIXTURE_BASELINE": result.get(
                    "LOW_END_DIAGNOSIS_FIXTURE_BASELINE"
                ),
                "PRODUCTION_MUSIC_DIAGNOSIS": result.get("PRODUCTION_MUSIC_DIAGNOSIS"),
                "regions": {
                    key: {
                        "finding_types": (result.get(key) or {}).get("finding_types"),
                        "confidence": (result.get(key) or {}).get("confidence"),
                    }
                    for key in ("REGION_A", "REGION_B", "REGION_C")
                },
            },
            indent=2,
            default=str,
        ))
        return 0
    except Exception as exc:
        status = live_block_status(detection)
        payload = {"verification_class": status, "phase": "LIVE-3R", "error": str(exc)}
        (evidence / "live3r_reality_check.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )
        logger.error("LIVE-3R failed: %s", exc)
        print(json.dumps(payload, indent=2))
        return 2


def _live3r_perf(evidence: Path, logger) -> int:
    detection = detect_ableton()
    adapter = AbletonTcpAdapter()
    try:
        adapter.connect()
        result = run_live3r_perf(adapter, evidence)
        adapter.disconnect()
        print(
            json.dumps(
                {
                    "phase": result.get("phase"),
                    "region": result.get("region"),
                    "TOTAL": result.get("TOTAL"),
                    "CAPTURE": result.get("CAPTURE"),
                    "OVERHEAD": result.get("OVERHEAD"),
                    "DSP": result.get("DSP"),
                    "performance_gate": result.get("performance_gate"),
                    "DIAGNOSIS_RESULT_UNCHANGED": result.get(
                        "DIAGNOSIS_RESULT_UNCHANGED"
                    ),
                    "CAPTURE_PROVENANCE_INTACT": result.get(
                        "CAPTURE_PROVENANCE_INTACT"
                    ),
                    "STATE_RESTORE_INTACT": result.get("STATE_RESTORE_INTACT"),
                    "tcp_total": (result.get("tcp") or {}).get("total"),
                    "get_track_info": (result.get("tcp") or {}).get("get_track_info"),
                    "full_session_snapshots": result.get("full_session_snapshots"),
                    "top_3_remaining_bottlenecks": result.get(
                        "top_3_remaining_bottlenecks"
                    ),
                    "error": result.get("error"),
                },
                indent=2,
                default=str,
            )
        )
        if result.get("error"):
            return 2
        return 0
    except Exception as exc:
        status = live_block_status(detection)
        payload = {
            "verification_class": status,
            "phase": "PERFORMANCE CHECK",
            "error": str(exc),
        }
        (evidence / "live3r_perf.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )
        logger.error("LIVE-3R perf failed: %s", exc)
        print(json.dumps(payload, indent=2))
        return 2


def _live3r_perf2(evidence: Path, logger) -> int:
    detection = detect_ableton()
    adapter = AbletonTcpAdapter()
    try:
        adapter.connect()
        result = run_live3r_perf2(adapter, evidence)
        adapter.disconnect()
        print(
            json.dumps(
                {
                    "phase": result.get("phase"),
                    "TOTAL": result.get("TOTAL"),
                    "PASS COUNT": result.get("PASS COUNT"),
                    "performance_gate": result.get("performance_gate"),
                    "DIAGNOSIS_SAME_WAV_DETERMINISTIC": result.get(
                        "DIAGNOSIS_SAME_WAV_DETERMINISTIC"
                    ),
                    "CAPTURE_TO_CAPTURE_STABILITY": result.get(
                        "CAPTURE_TO_CAPTURE_STABILITY"
                    ),
                    "MINIMAL_EXPERIMENT": result.get("MINIMAL_EXPERIMENT"),
                    "probe_gates": (result.get("capture_track_probe") or {}).get(
                        "gates"
                    ),
                    "probe_host": (result.get("capture_track_probe") or {}).get("host"),
                    "probe_content": (result.get("capture_track_probe") or {}).get(
                        "content"
                    ),
                    "CAPTURE_PROVENANCE_INTACT": result.get(
                        "CAPTURE_PROVENANCE_INTACT"
                    ),
                    "STATE_RESTORE_INTACT": result.get("STATE_RESTORE_INTACT"),
                    "kick_signal_point": result.get("kick_signal_point"),
                    "finding_types": result.get("finding_types"),
                    "tcp_total": (result.get("tcp") or {}).get("total"),
                    "routing_mutations": result.get("routing_mutations"),
                    "STOP": result.get("STOP"),
                    "error": result.get("error"),
                },
                indent=2,
                default=str,
            )
        )
        if result.get("error") or result.get("STOP"):
            return 2 if result.get("error") else 0
        return 0
    except Exception as exc:
        status = live_block_status(detection)
        payload = {
            "verification_class": status,
            "phase": "PERF-2",
            "error": str(exc),
        }
        (evidence / "live3r_perf2.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )
        logger.error("PERF-2 failed: %s", exc)
        print(json.dumps(payload, indent=2))
        return 2


def _live3r_perf3(evidence: Path, logger) -> int:
    detection = detect_ableton()
    adapter = AbletonTcpAdapter()
    try:
        adapter.connect()
        result = run_live3r_perf3(adapter, evidence)
        adapter.disconnect()
        print(
            json.dumps(
                {
                    "phase": result.get("phase"),
                    "SLOT-ENABLED TAP": result.get("SLOT-ENABLED TAP"),
                    "RUNTIME PROTOCOL": result.get("RUNTIME PROTOCOL"),
                    "slot_roundtrip": result.get("slot_roundtrip"),
                    "capture_sends_all_silent": result.get(
                        "capture_sends_all_silent"
                    ),
                    "MULTI-TAP FILE ISOLATION": result.get("MULTI-TAP FILE ISOLATION"),
                    "PARALLEL PASS 1": result.get("PARALLEL PASS 1"),
                    "OFF-MAIN TRACK CAPTURE": result.get("OFF-MAIN TRACK CAPTURE"),
                    "UPDATED TAP PARAMETERS": result.get("UPDATED TAP PARAMETERS"),
                    "TWO-TAP RESULT": {
                        "ok": (result.get("TWO-TAP RESULT") or {}).get("ok"),
                        "pass_id": (result.get("TWO-TAP RESULT") or {}).get("pass_id"),
                    },
                    "PASS 1 RESULT": {
                        "ok": (result.get("PASS 1 RESULT") or {}).get("ok"),
                        "pass_id": (result.get("PASS 1 RESULT") or {}).get("pass_id"),
                    },
                    "SEMANTICS": result.get("SEMANTICS"),
                    "TOTAL TIME": result.get("TOTAL TIME") or result.get("TOTAL"),
                    "TOP REMAINING BOTTLENECK": result.get("TOP REMAINING BOTTLENECK"),
                    "routing_mutations": result.get("routing_mutations"),
                    "tcp_total": (result.get("tcp") or {}).get("total"),
                    "STOP": result.get("STOP"),
                    "error": result.get("error"),
                },
                indent=2,
                default=str,
            )
        )
        if result.get("error"):
            return 2
        return 0
    except Exception as exc:
        status = live_block_status(detection)
        payload = {
            "verification_class": status,
            "phase": "PERF-3",
            "error": str(exc),
        }
        (evidence / "live3r_perf3.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )
        logger.error("PERF-3 failed: %s", exc)
        print(json.dumps(payload, indent=2))
        return 2


def _live3r_trust(evidence: Path, logger) -> int:
    detection = detect_ableton()
    adapter = AbletonTcpAdapter()
    try:
        adapter.connect()
        result = run_live3r_trust(adapter, evidence)
        adapter.disconnect()
        print(
            json.dumps(
                {
                    "phase": result.get("phase"),
                    "TAP INVENTORY": {
                        "count": (result.get("TAP INVENTORY") or {}).get("count")
                    },
                    "SLOT COLLISION TEST": result.get("SLOT COLLISION TEST"),
                    "UDP CROSS-TALK TEST": {
                        "instance_control": (
                            result.get("UDP CROSS-TALK TEST") or {}
                        ).get("instance_control"),
                        "lom_ok": ((result.get("UDP CROSS-TALK TEST") or {}).get("lom") or {}).get(
                            "ok"
                        ),
                        "udp_ok": ((result.get("UDP CROSS-TALK TEST") or {}).get("udp") or {}).get(
                            "ok"
                        ),
                    },
                    "FILE OWNERSHIP TEST": result.get("FILE OWNERSHIP TEST"),
                    "MAIN SIGNAL POINT": (result.get("MAIN SIGNAL POINT") or {}).get(
                        "claim"
                    ),
                    "KICK/BASS ROUTING SEMANTICS": {
                        "kick": ((result.get("KICK/BASS ROUTING SEMANTICS") or {}).get("kick") or {}).get(
                            "claim"
                        ),
                        "bass": ((result.get("KICK/BASS ROUTING SEMANTICS") or {}).get("bass") or {}).get(
                            "claim"
                        ),
                    },
                    "PASS 1": {
                        "ok": (result.get("PASS 1") or {}).get("ok"),
                        "pass_id": (result.get("PASS 1") or {}).get("pass_id"),
                    },
                    "5x REPEATABILITY": (result.get("5x REPEATABILITY") or {}).get(
                        "claim"
                    ),
                    "ALIGNMENT RESULT": (result.get("ALIGNMENT RESULT") or {}).get(
                        "claim"
                    ),
                    "CAPTURE JOURNAL STATUS": result.get("CAPTURE JOURNAL STATUS"),
                    "STATE RESTORE": (result.get("STATE RESTORE") or {}).get("ok"),
                    "PERFORMANCE": result.get("PERFORMANCE"),
                    "STOP": result.get("STOP"),
                    "error": result.get("error"),
                },
                indent=2,
                default=str,
            )
        )
        if result.get("error"):
            return 2
        return 0
    except Exception as exc:
        status = live_block_status(detection)
        payload = {
            "verification_class": status,
            "phase": "MULTI-TAP TRUST",
            "error": str(exc),
        }
        (evidence / "live3r_trust.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )
        logger.error("TRUST failed: %s", exc)
        print(json.dumps(payload, indent=2))
        return 2


def _live3r_prod(evidence: Path, logger) -> int:
    detection = detect_ableton()
    adapter = AbletonTcpAdapter()
    try:
        adapter.connect()
        result = run_live3r_prod(adapter, evidence)
        adapter.disconnect()
        print(
            json.dumps(
                {
                    "phase": result.get("phase"),
                    "mode": result.get("mode"),
                    "RUNTIME": result.get("RUNTIME"),
                    "STOP": result.get("STOP"),
                    "PRODUCTION_PASS_TIME": result.get("PRODUCTION_PASS_TIME"),
                    "CERTIFICATION_SUITE_TIME": result.get("CERTIFICATION_SUITE_TIME"),
                    "GRADE": result.get("GRADE"),
                    "BENCHMARK": result.get("BENCHMARK"),
                    "TCP": result.get("TCP"),
                    "ROUTING_MUTATIONS": result.get("ROUTING_MUTATIONS"),
                    "TRANSPORT_MUTATIONS": result.get("TRANSPORT_MUTATIONS"),
                    "FIXED_SLEEPS_REMAINING": result.get("FIXED_SLEEPS_REMAINING"),
                    "BATCH": result.get("BATCH"),
                    "REGRESSION": result.get("REGRESSION"),
                    "CAPTURE JOURNAL STATUS": result.get("CAPTURE JOURNAL STATUS"),
                    "PASS 1": {
                        "ok": (result.get("PASS 1") or {}).get("ok"),
                        "pass_id": (result.get("PASS 1") or {}).get("pass_id"),
                    },
                    "error": result.get("error"),
                },
                indent=2,
                default=str,
            )
        )
        if result.get("error") or result.get("STOP"):
            return 2
        return 0
    except Exception as exc:
        status = live_block_status(detection)
        payload = {
            "verification_class": status,
            "phase": "PRODUCTION CAPTURE",
            "error": str(exc),
        }
        (evidence / "live3r_prod.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )
        logger.error("PRODUCTION capture failed: %s", exc)
        print(json.dumps(payload, indent=2))
        return 2


def _live3r_align(evidence: Path, logger) -> int:
    detection = detect_ableton()
    adapter = AbletonTcpAdapter()
    try:
        adapter.connect()
        result = run_alignment_certification(adapter, evidence)
        adapter.disconnect()
        (evidence / "live3r_align.json").write_text(
            json.dumps(result, indent=2, default=str), encoding="utf-8"
        )
        print(
            json.dumps(
                {
                    "phase": result.get("phase"),
                    "STOP": result.get("STOP"),
                    "ALIGNMENT_ENVELOPE": {
                        "claim": (result.get("ALIGNMENT_ENVELOPE") or {}).get("claim"),
                        "envelope": (result.get("ALIGNMENT_ENVELOPE") or {}).get(
                            "envelope"
                        ),
                        "sample_accurate": (result.get("ALIGNMENT_ENVELOPE") or {}).get(
                            "sample_accurate"
                        ),
                        "max_abs_error_ms": (result.get("ALIGNMENT_ENVELOPE") or {}).get(
                            "max_abs_error_ms"
                        ),
                    },
                    "capability_cache": result.get("capability_cache"),
                    "routing_restored": result.get("routing_restored"),
                },
                indent=2,
                default=str,
            )
        )
        envelope = result.get("ALIGNMENT_ENVELOPE") or {}
        if result.get("STOP") or envelope.get("claim") == "FAILED":
            return 2
        return 0
    except Exception as exc:
        status = live_block_status(detection)
        payload = {
            "verification_class": status,
            "phase": "ALIGNMENT CERTIFICATION",
            "error": str(exc),
        }
        (evidence / "live3r_align.json").write_text(
            json.dumps(payload, indent=2, default=str), encoding="utf-8"
        )
        logger.error("ALIGN failed: %s", exc)
        print(json.dumps(payload, indent=2, default=str))
        return 2


def _live_state(evidence: Path, logger) -> int:
    detection = detect_ableton()
    adapter = AbletonTcpAdapter()
    try:
        adapter.connect()
        result = run_live_state(adapter, evidence)
        adapter.disconnect()
        (evidence / "live_state.json").write_text(
            json.dumps(result, indent=2, default=str), encoding="utf-8"
        )
        print(
            json.dumps(
                {
                    "phase": result.get("phase"),
                    "ok": result.get("ok"),
                    "STOP": result.get("STOP"),
                    "ACCEPT": result.get("ACCEPT"),
                    "PROJECT": {
                        "identity_kind": (result.get("PROJECT") or {}).get(
                            "identity_kind"
                        ),
                        "path": (result.get("PROJECT") or {}).get("path"),
                    },
                    "PERF": result.get("PERF"),
                    "RECONNECT": result.get("RECONNECT"),
                },
                indent=2,
                default=str,
            )
        )
        if result.get("STOP") or not result.get("ok"):
            return 2
        return 0
    except Exception as exc:
        status = live_block_status(detection)
        payload = {
            "verification_class": status,
            "phase": "SESSION STATE TRUST",
            "error": str(exc),
        }
        (evidence / "live_state.json").write_text(
            json.dumps(payload, indent=2, default=str), encoding="utf-8"
        )
        logger.error("STATE TRUST failed: %s", exc)
        print(json.dumps(payload, indent=2, default=str))
        return 2


def _live3(evidence: Path, logger) -> int:
    detection = detect_ableton()
    adapter = AbletonTcpAdapter()
    try:
        adapter.connect()
        result = run_live3(adapter, evidence)
        adapter.disconnect()
        print(json.dumps(result, indent=2, default=str))
        ok = result.get("REAL LOW-END AUDIO") == "VERIFIED" and result.get(
            "STRUCTURED EVIDENCE"
        ) == "VERIFIED"
        return 0 if ok else 2
    except Exception as exc:
        status = live_block_status(detection)
        payload = {"verification_class": status, "phase": "LIVE-3", "error": str(exc)}
        (evidence / "live3_lowend_diagnosis.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )
        logger.error("LIVE-3 failed: %s", exc)
        print(json.dumps(payload, indent=2))
        return 2


def _live22e(evidence: Path, logger) -> int:
    detection = detect_ableton()
    adapter = AbletonTcpAdapter()
    try:
        adapter.connect()
        result = run_live22e(adapter, evidence)
        adapter.disconnect()
        print(json.dumps(result, indent=2, default=str))
        return 0 if result.get("SILENT BOUNDARY") != "FAILED" else 2
    except Exception as exc:
        status = live_block_status(detection)
        payload = {"verification_class": status, "phase": "LIVE-2.2e", "error": str(exc)}
        (evidence / "live22e_five_fixtures.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )
        logger.error("LIVE-2.2e failed: %s", exc)
        print(json.dumps(payload, indent=2))
        return 2


def _live22(evidence: Path, logger) -> int:
    detection = detect_ableton()
    adapter = AbletonTcpAdapter()
    try:
        adapter.connect()
        result = run_live22(adapter, evidence)
        adapter.disconnect()
        print(json.dumps(result, indent=2, default=str))
        failed = [
            key
            for key, value in result.items()
            if value == "FAILED"
            and key
            in {
                "CAPTURE SEMANTICS",
                "REGION ALIGNMENT",
                "TRACK ISOLATION",
                "SUPPORTED ENVELOPE",
            }
        ]
        return 0 if not failed else 2
    except Exception as exc:
        status = live_block_status(detection)
        payload = {
            "verification_class": status,
            "phase": "LIVE-2.2",
            "error": str(exc),
        }
        (evidence / "live22_capture_semantics.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )
        logger.error("LIVE-2.2 failed: %s", exc)
        print(json.dumps(payload, indent=2))
        return 2


def _live21(evidence: Path, logger) -> int:
    detection = detect_ableton()
    adapter = AbletonTcpAdapter()
    try:
        adapter.connect()
        result = run_live21(adapter, evidence)
        adapter.disconnect()
        print(json.dumps(result, indent=2, default=str))
        ok = (
            result.get("MASTER PRECISE CAPTURE") == "VERIFIED"
            and result.get("TRACK PRECISE CAPTURE") == "VERIFIED"
            and result.get("MUSIC OBSERVATION") == "VERIFIED"
        )
        return 0 if ok else 2
    except Exception as exc:
        status = live_block_status(detection)
        payload = {
            "verification_class": status,
            "MASTER PRECISE CAPTURE": "NOT_VERIFIED",
            "TRACK PRECISE CAPTURE": "NOT_VERIFIED",
            "error": str(exc),
        }
        (evidence / "live21_precise_capture.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )
        logger.error("LIVE-2.1 failed: %s", exc)
        print(json.dumps(payload, indent=2))
        return 2


def _live2(evidence: Path, logger) -> int:
    detection = detect_ableton()
    adapter = AbletonTcpAdapter()
    try:
        adapter.connect()
        result = run_live2(adapter, evidence)
        adapter.disconnect()
        print(json.dumps(result, indent=2, default=str))
        return 0 if result.get("AUDIO CAPTURE") == "VERIFIED" else 2
    except Exception as exc:
        status = live_block_status(detection)
        payload = {
            "verification_class": status,
            "AUDIO CAPTURE": "NOT_VERIFIED",
            "error": str(exc),
            "required_action": _manual_control_surface_action()
            if status == "MANUAL_CONFIGURATION_REQUIRED"
            else None,
        }
        (evidence / "live2_audio_capture.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )
        logger.error("LIVE-2 failed: %s", exc)
        print(json.dumps(payload, indent=2))
        return 2


def _probe(evidence: Path, logger) -> int:
    detection = detect_ableton()
    write_detection(evidence / "ableton_detect.json", detection)
    adapter = AbletonTcpAdapter()
    try:
        adapter.connect()
        if (adapter.handshake_info or {}).get("backend") == "mock":
            raise DawError("BLOCKED_BY_ENVIRONMENT: mock TCP backend is not Live")
        result = adapter.probe()
        adapter.disconnect()
        (evidence / "live_probe.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
        print(json.dumps(result, indent=2))
        return 0
    except Exception as exc:
        status = live_block_status(detection)
        payload = {
            "verification_class": status,
            "status": status,
            "error": str(exc),
            "required_action": _manual_control_surface_action()
            if status == "MANUAL_CONFIGURATION_REQUIRED"
            else None,
            "detection": detection.to_dict(),
        }
        (evidence / "live_probe.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )
        logger.error("LIVE probe failed: %s", exc)
        print(json.dumps(payload, indent=2))
        return 2


def _session_diagnose(evidence: Path, logger) -> int:
    from copilot.audio.session_diagnose import preflight_session, write_preflight
    from copilot.daw.ableton_tcp import AbletonTcpAdapter

    adapter = AbletonTcpAdapter()
    try:
        adapter.connect()
        report = preflight_session(adapter)
    finally:
        adapter.disconnect()
    path = write_preflight(report, evidence)
    logger.info(
        "session-diagnose preflight pass=%s missing=%s project=%s",
        report.get("pass"),
        report.get("missing"),
        report.get("project_name"),
    )
    print(json.dumps(
        {
            "pass": report.get("pass"),
            "status": report.get("status"),
            "missing": report.get("missing"),
            "target_source_unsupported": report.get("target_source_unsupported"),
            "kick_source_class": report.get("kick_source_class"),
            "kick_source": report.get("kick_source"),
            "bass_source": report.get("bass_source"),
            "project_ok": report.get("project_ok"),
            "working_copy": report.get("working_copy"),
            "original_open": report.get("original_open"),
            "live_set_path": report.get("live_set_path") or report.get("project_path"),
            "project_name": report.get("project_name"),
            "project_path": report.get("project_path"),
            "project_token": report.get("project_token"),
            "audible_token": report.get("audible_token"),
            "revision": report.get("revision"),
            "drums": report.get("drums"),
            "bass": report.get("bass"),
            "main": report.get("main"),
            "taps": report.get("taps"),
            "slot_collisions": report.get("slot_collisions"),
            "capture_hosts": report.get("capture_hosts"),
            "remote_script": report.get("remote_script"),
            "playing": report.get("playing"),
            "instruction": report.get("instruction"),
            "model_calls": report.get("model_calls"),
            "artifact": str(path),
        },
        indent=2,
        ensure_ascii=False,
    ))
    if not report.get("pass"):
        return 2
    return 0


def _session_run1_astra(evidence: Path, logger) -> int:
    from copilot.reasoning.session_astra import run_session_astra

    report = run_session_astra(
        evidence=evidence,
        source_run="session_run1",
        include_fullmix=False,
        diagnosis_revision=1,
    )
    return _print_astra_public(report, logger)


def _session_run1_fullmix(evidence: Path, logger) -> int:
    from copilot.audio.fullmix import run_session_fullmix

    report = run_session_fullmix(evidence=evidence, source_run="session_run1")
    logger.info(
        "session-run1-fullmix status=%s writes=%s artifact=%s",
        report.get("status"),
        report.get("MUSICAL WRITES"),
        report.get("artifact"),
    )
    public = {
        "status": report.get("status"),
        "analyzer_id": report.get("analyzer_id"),
        "analyzer_sha256": report.get("analyzer_sha256"),
        "configuration_hash": report.get("configuration_hash"),
        "MUSICAL WRITES": report.get("MUSICAL WRITES"),
        "ASTRA CALLS": report.get("ASTRA CALLS"),
        "artifact": report.get("artifact"),
        "regions": [
            {
                "region_id": row.get("region_id"),
                "audio_sha256": row.get("audio_sha256"),
                "duration_s": row.get("duration_s"),
                "energy_event_count": len(row.get("energy_events") or []),
                "energy_events": [
                    {
                        "kind": ev.get("kind"),
                        "start_s": round(float(ev.get("start_s") or 0.0), 3),
                        "end_s": round(float(ev.get("end_s") or 0.0), 3),
                        "duration_s": round(float(ev.get("duration_s") or 0.0), 3),
                        "relative_drop_db": round(float(ev.get("relative_drop_db") or 0.0), 2),
                        "similarity": ev.get("event_similarity_count"),
                        "period_s": ev.get("approx_period_s"),
                        "repetition_strength": ev.get("repetition_strength"),
                    }
                    for ev in (row.get("energy_events") or [])
                ],
                "transient_count": ((row.get("transient") or {}).get("transient_count")),
                "spectral_event_count": len(row.get("spectral_events") or []),
                "cache_hit": row.get("cache_hit"),
            }
            for row in report.get("regions") or []
        ],
    }
    print(json.dumps(public, indent=2, ensure_ascii=False))
    return 0 if report.get("MUSICAL WRITES") == 0 else 2


def _session_run1_astra_r2(evidence: Path, logger) -> int:
    from copilot.reasoning.session_astra import compare_astra_revisions, run_session_astra

    report = run_session_astra(
        evidence=evidence,
        source_run="session_run1",
        include_fullmix=True,
        diagnosis_revision=2,
        artifact_name="session_run1_astra_r2.json",
    )
    code = _print_astra_public(report, logger)
    if report.get("status") == "ASTRA COMPLETE":
        compare = compare_astra_revisions(evidence=evidence)
        print(json.dumps({"compare_artifact": compare.get("artifact"), "regions": compare.get("regions")}, indent=2, ensure_ascii=False))
    return code


def _session_run1_grounding_causal(evidence: Path, logger) -> int:
    """Freeze FullMix V1, fix/revalidate C grounding, gather causal evidence, Astra r3 C only."""
    from copilot.audio.fullmix import freeze_fullmix_v1
    from copilot.reasoning.session_astra import (
        revalidate_historical_r2_region,
        run_region_causal_astra,
    )

    freeze = freeze_fullmix_v1(evidence=evidence, source_run="session_run1")
    reval = revalidate_historical_r2_region(
        evidence=evidence,
        source_run="session_run1",
        region_id="REGION_C",
    )
    # Only call Astra for a new revision if historical r2 still cannot validly pass,
    # OR always for causal revision (r3) after evidence — r3 is a new revision with
    # causal context; r2 is never overwritten.
    causal_report = run_region_causal_astra(
        evidence=evidence,
        source_run="session_run1",
        region_id="REGION_C",
        diagnosis_revision=3,
    )
    out = {
        "FULLMIX V1 FROZEN": freeze.get("status"),
        "analyzer_id": freeze.get("analyzer_id"),
        "configuration_hash": freeze.get("configuration_hash"),
        "analyzer_hash": freeze.get("analyzer_hash"),
        "energy_reference_method": freeze.get("energy_reference_method"),
        "RUN1 MARKED CALIBRATION": freeze.get("run_classification"),
        "R2_C_BEFORE": reval.get("R2_C_BEFORE"),
        "R2_C_AFTER": reval.get("R2_C_AFTER"),
        "R2_remaining_reason": reval.get("remaining_reason"),
        "grounding_revalidation_artifact": reval.get("artifact"),
        "freeze_artifact": freeze.get("artifact"),
        "causal_astra": {
            "status": causal_report.get("status"),
            "diagnosis_revision": causal_report.get("diagnosis_revision"),
            "ASTRA CALLS": causal_report.get("ASTRA CALLS"),
            "MUSICAL WRITES": causal_report.get("MUSICAL WRITES"),
            "MUSICPLAN_GATE": causal_report.get("MUSICPLAN_GATE"),
            "artifact": causal_report.get("artifact"),
            "accepted": (causal_report.get("regions") or [{}])[0].get("accepted"),
            "failure": (causal_report.get("regions") or [{}])[0].get("failure"),
            "category": ((causal_report.get("regions") or [{}])[0].get("output") or {}).get(
                "category"
            ),
            "status_diag": ((causal_report.get("regions") or [{}])[0].get("output") or {}).get(
                "status"
            ),
            "summary": ((causal_report.get("regions") or [{}])[0].get("output") or {}).get(
                "summary"
            ),
            "requested_evidence_from_r2": causal_report.get("requested_evidence_from_r2"),
            "evidence_obtained_summary": causal_report.get("evidence_obtained_summary"),
        },
    }
    summary_path = evidence / "session_run1_grounding_causal_summary.json"
    summary_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(
        "grounding-causal freeze=%s r2_after=%s r3_accepted=%s gate=%s writes=%s",
        freeze.get("status"),
        reval.get("R2_C_AFTER"),
        out["causal_astra"]["accepted"],
        causal_report.get("MUSICPLAN_GATE"),
        causal_report.get("MUSICAL WRITES"),
    )
    print(json.dumps(out, indent=2, ensure_ascii=False))
    if causal_report.get("status") == "REAL_MODEL_UNAVAILABLE":
        return 2
    if causal_report.get("MUSICAL WRITES") != 0:
        return 2
    return 0


def _session_run1_causal_trace(evidence: Path, logger) -> int:
    """Causal Trace V1 + Astra r4 for REGION_C only. No MusicPlan auto-build."""
    from copilot.reasoning.session_astra import run_region_causal_trace_astra

    report = run_region_causal_trace_astra(
        evidence=evidence,
        source_run="session_run1",
        region_id="REGION_C",
        diagnosis_revision=4,
    )
    row = (report.get("regions") or [{}])[0]
    out = report.get("output") if False else (row.get("output") or {})
    public = {
        "status": report.get("status"),
        "EVENT_QN": {
            "event_id": (report.get("primary_gap_event") or {}).get("event_id"),
            "audio_s": [
                (report.get("primary_gap_event") or {}).get("event_audio_start_s"),
                (report.get("primary_gap_event") or {}).get("event_audio_end_s"),
            ],
            "qn": [
                (report.get("primary_gap_event") or {}).get("event_start_qn"),
                (report.get("primary_gap_event") or {}).get("event_end_qn"),
            ],
            "kind": (report.get("primary_gap_event") or {}).get("kind"),
        },
        "ruled_out": report.get("ruled_out"),
        "supported_cause_candidates": report.get("supported_cause_candidates"),
        "next_evidence": report.get("next_evidence"),
        "live_reconciliation": {
            "status": (report.get("live_reconciliation") or {}).get("status"),
            "ok": (report.get("live_reconciliation") or {}).get("ok"),
        },
        "ASTRA_R4": {
            "accepted": row.get("accepted"),
            "failure": row.get("failure"),
            "category": out.get("category"),
            "status": out.get("status"),
            "summary": out.get("summary"),
            "confidence": out.get("confidence"),
        },
        "MUSICPLAN_GATE": report.get("MUSICPLAN_GATE"),
        "MUSICPLAN_GATE_REASON": report.get("MUSICPLAN_GATE_REASON"),
        "MUSICAL WRITES": report.get("MUSICAL WRITES"),
        "ASTRA CALLS": report.get("ASTRA CALLS"),
        "causal_trace_artifact": report.get("causal_trace_artifact"),
        "artifact": report.get("artifact"),
    }
    summary_path = evidence / "session_run1_causal_trace_summary.json"
    summary_path.write_text(json.dumps(public, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(
        "causal-trace r4 accepted=%s gate=%s writes=%s next=%s",
        row.get("accepted"),
        report.get("MUSICPLAN_GATE"),
        report.get("MUSICAL WRITES"),
        report.get("next_evidence"),
    )
    print(json.dumps(public, indent=2, ensure_ascii=False))
    if report.get("status") in {"REAL_MODEL_UNAVAILABLE", "PROJECT_STATE_CONFLICT"}:
        return 2
    if report.get("MUSICAL WRITES") != 0:
        return 2
    return 0


def _session_run1_source_audio(evidence: Path, logger) -> int:
    """SOURCE AUDIO TRACE V1 (Core) + Astra r5 (reasoning). No MusicPlan auto-build."""
    from copilot.audio.source_audio_trace import run_source_audio_trace
    from copilot.daw.ableton_tcp import AbletonTcpAdapter
    from copilot.reasoning.session_astra import run_region_source_audio_astra

    adapter = AbletonTcpAdapter()
    adapter.connect()
    try:
        source_payload = run_source_audio_trace(
            adapter,
            evidence=evidence,
            source_run="session_run1",
        )
        live_project = source_payload.get("project_token")
        live_audible = source_payload.get("audible_token")
    finally:
        try:
            adapter.disconnect()
        except Exception:
            pass

    report = run_region_source_audio_astra(
        source_payload=source_payload,
        evidence=evidence,
        source_run="session_run1",
        region_id="REGION_C",
        diagnosis_revision=5,
        live_project_token=live_project,
        live_audible_token=live_audible,
    )
    row = (report.get("regions") or [{}])[0]
    out = row.get("output") or {}
    obs = {
        o.get("track"): {
            "signal_class": o.get("signal_class"),
            "midi_audio_relation": o.get("midi_audio_relation"),
            "event_rms": o.get("event_rms"),
            "event_peak": o.get("event_peak"),
            "relative_drop_db": o.get("relative_drop_db"),
            "main_event_signal_class": o.get("main_event_signal_class"),
        }
        for o in (report.get("observations") or [])
    }
    public = {
        "status": report.get("status"),
        "event_qn_range": report.get("event_qn_range"),
        "event_audio_range_s": report.get("event_audio_range_s"),
        "Strum": obs.get("Strum"),
        "Transform Seed": obs.get("Transform Seed"),
        "Rainstorm": obs.get("Rainstorm"),
        "decision": report.get("decision"),
        "ASTRA_R5": {
            "accepted": row.get("accepted"),
            "failure": row.get("failure"),
            "category": out.get("category"),
            "status": out.get("status"),
            "summary": out.get("summary"),
            "confidence": out.get("confidence"),
        },
        "MUSICPLAN_GATE": report.get("MUSICPLAN_GATE"),
        "MUSICPLAN_GATE_REASON": report.get("MUSICPLAN_GATE_REASON"),
        "MUSICAL WRITES": report.get("MUSICAL WRITES"),
        "AUDIBLE_MIX_MUTATIONS": report.get("AUDIBLE_MIX_MUTATIONS"),
        "ASTRA CALLS": report.get("ASTRA CALLS"),
        "temp_routing_restores": [
            {"track": p.get("track"), "restore_ok": (p.get("restore") or {}).get("ok")}
            for p in (report.get("passes") or [])
        ],
        "source_audio_artifact": report.get("source_audio_artifact"),
        "artifact": report.get("artifact"),
    }
    summary_path = evidence / "session_run1_source_audio_summary.json"
    summary_path.write_text(json.dumps(public, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(
        "source-audio r5 accepted=%s gate=%s writes=%s audible_mut=%s decision=%s",
        row.get("accepted"),
        report.get("MUSICPLAN_GATE"),
        report.get("MUSICAL WRITES"),
        report.get("AUDIBLE_MIX_MUTATIONS"),
        (report.get("decision") or {}).get("code"),
    )
    print(json.dumps(public, indent=2, ensure_ascii=False))
    if report.get("status") in {
        "REAL_MODEL_UNAVAILABLE",
        "PROJECT_STATE_CONFLICT",
        "PRE-FLIGHT STOP",
    }:
        return 2
    if report.get("MUSICAL WRITES") != 0 or report.get("AUDIBLE_MIX_MUTATIONS") != 0:
        return 2
    if any(p.get("restore_ok") is False for p in public["temp_routing_restores"]):
        return 2
    return 0


def _session_run1_strum_device(evidence: Path, logger) -> int:
    """STRUM DEVICE TRACE V1 (Core) + Astra r6 (reasoning). No MusicPlan auto-build."""
    from copilot.audio.strum_device_trace import run_strum_device_trace
    from copilot.daw.ableton_tcp import AbletonTcpAdapter
    from copilot.reasoning.session_astra import run_region_strum_device_astra

    adapter = AbletonTcpAdapter()
    adapter.connect()
    try:
        strum_payload = run_strum_device_trace(
            adapter,
            evidence=evidence,
            source_run="session_run1",
        )
        snap = strum_payload.get("state_snapshot") or {}
        live_project = snap.get("PROJECT_STATE_TOKEN")
        live_audible = snap.get("AUDIBLE_STATE_TOKEN")
        live_target = snap.get("TARGET_STATE_TOKEN")
    finally:
        try:
            adapter.disconnect()
        except Exception:
            pass

    report = run_region_strum_device_astra(
        strum_payload=strum_payload,
        evidence=evidence,
        source_run="session_run1",
        region_id="REGION_C",
        diagnosis_revision=6,
        live_project_token=live_project,
        live_audible_token=live_audible,
        live_target_token=live_target,
    )
    row = (report.get("regions") or [{}])[0]
    out = row.get("output") or {}
    cause = report.get("cause_result") or {}
    public = {
        "status": report.get("status"),
        "event_qn_range": report.get("event_qn_range"),
        "state_snapshot": report.get("state_snapshot"),
        "cause_status": cause.get("status"),
        "cause_detail": cause.get("detail"),
        "cause_leads": cause.get("leads"),
        "ruled_out": cause.get("ruled_out"),
        "automation_highlights": [
            {
                "parameter_identity": a.get("parameter_identity"),
                "parameter_raw": a.get("parameter_raw"),
                "amplitude_causality": a.get("amplitude_causality"),
                "value_before": a.get("value_before"),
                "value_during": a.get("value_during"),
                "value_after": a.get("value_after"),
                "changed_across_event": a.get("changed_across_event"),
            }
            for a in (report.get("automation_candidates") or [])
            if a.get("changed_across_event") or a.get("amplitude_causality") in {"KNOWN", "PLAUSIBLE"}
        ],
        "ASTRA_R6": {
            "accepted": row.get("accepted"),
            "failure": row.get("failure"),
            "category": out.get("category"),
            "status": out.get("status"),
            "summary": out.get("summary"),
            "confidence": out.get("confidence"),
        },
        "MUSICPLAN_GATE": report.get("MUSICPLAN_GATE"),
        "MUSICPLAN_GATE_REASON": report.get("MUSICPLAN_GATE_REASON"),
        "MUSICAL WRITES": report.get("MUSICAL WRITES"),
        "ASTRA CALLS": report.get("ASTRA CALLS"),
        "strum_device_artifact": report.get("strum_device_artifact"),
        "artifact": report.get("artifact"),
    }
    summary_path = evidence / "session_run1_strum_device_summary.json"
    summary_path.write_text(json.dumps(public, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(
        "strum-device r6 accepted=%s gate=%s cause=%s writes=%s",
        row.get("accepted"),
        report.get("MUSICPLAN_GATE"),
        cause.get("status"),
        report.get("MUSICAL WRITES"),
    )
    print(json.dumps(public, indent=2, ensure_ascii=False))
    if report.get("status") in {"REAL_MODEL_UNAVAILABLE", "PRE-FLIGHT STOP"}:
        return 2
    if report.get("MUSICAL WRITES") != 0:
        return 2
    return 0


def _print_astra_public(report: dict, logger) -> int:
    logger.info(
        "session-astra status=%s rev=%s astra=%s writes=%s artifact=%s",
        report.get("status"),
        report.get("diagnosis_revision"),
        report.get("ASTRA CALLS"),
        report.get("MUSICAL WRITES"),
        report.get("artifact"),
    )
    public = {
        "status": report.get("status"),
        "diagnosis_revision": report.get("diagnosis_revision"),
        "ASTRA CALLS": report.get("ASTRA CALLS"),
        "MUSICAL WRITES": report.get("MUSICAL WRITES"),
        "MusicPlan": report.get("MusicPlan"),
        "provider": report.get("provider"),
        "provider_version": report.get("provider_version"),
        "prompt_version": report.get("prompt_version"),
        "schema_version": report.get("schema_version"),
        "include_fullmix": report.get("include_fullmix"),
        "artifact": report.get("artifact"),
        "regions": [
            {
                "region_id": row.get("region_id"),
                "accepted": row.get("accepted"),
                "failure": row.get("failure"),
                "summary": ((row.get("diagnosis") or {}).get("user_facing") or (row.get("output") or {}).get("summary")),
                "category": (row.get("output") or {}).get("category"),
                "status": (row.get("output") or {}).get("status") or (row.get("diagnosis") or {}).get("status"),
                "human_overall_feel": (row.get("human") or {}).get("overall_feel"),
                "human_would_change": (row.get("human") or {}).get("would_change"),
                "human_notes": (row.get("human") or {}).get("notes"),
            }
            for row in report.get("regions") or []
        ],
    }
    if report.get("status") == "REAL_MODEL_UNAVAILABLE":
        public["reason"] = report.get("reason")
    print(json.dumps(public, indent=2, ensure_ascii=False))
    if report.get("status") == "REAL_MODEL_UNAVAILABLE":
        return 2
    if report.get("MUSICAL WRITES") != 0:
        return 2
    return 0


def _session_run1(evidence: Path, logger) -> int:
    from copilot.audio.session_run1 import run_session_run1
    from copilot.daw.ableton_tcp import AbletonTcpAdapter

    adapter = AbletonTcpAdapter()
    try:
        adapter.connect()
        report = run_session_run1(adapter, evidence)
    finally:
        adapter.disconnect()
    logger.info(
        "session-run1 status=%s astra=%s writes=%s artifact=%s",
        report.get("status"),
        report.get("ASTRA CALLS"),
        report.get("MUSICAL WRITES"),
        evidence / "session_run1.json",
    )
    public = {key: value for key, value in report.items() if key != "DSP OBSERVATIONS"}
    public["DSP OBSERVATIONS"] = [
        {
            "region": row.get("region"),
            "analyzer_id": row.get("analyzer_id"),
            "analyzer_sha256": row.get("analyzer_sha256"),
            "features": row.get("features"),
            "observation_views": list((row.get("music_observation") or {}).keys()),
        }
        for row in report.get("DSP OBSERVATIONS") or []
    ]
    print(json.dumps(public, indent=2, ensure_ascii=False, default=str))
    status = str(report.get("status") or "")
    if status.startswith("RUN 1 COMPLETE"):
        return 0
    return 2


def _session_arrangement_start(evidence: Path, logger) -> int:
    from copilot.audio.arrangement_activity import inspect_tempo_contract, map_lowend_candidates
    from copilot.audio.arrangement_seek import TRANSPORT_PRIMITIVE_VERSION, run_atomic_transport_trust
    from copilot.audio.session_diagnose import WORKING_COPY_CANDIDATE
    from copilot.importing.working_copy_manager_v1 import find_working_copy
    from copilot.daw.ableton_tcp import AbletonTcpAdapter

    adapter = AbletonTcpAdapter()
    try:
        adapter.connect()
        hello = adapter.handshake_info or {}
        als = Path(str(hello.get("path") or WORKING_COPY_CANDIDATE))
        try:
            session = adapter.snapshot(include_notes=False)
            if session.project_path:
                als = Path(session.project_path)
        except DawError:
            pass
        als = find_working_copy(als) or find_working_copy(WORKING_COPY_CANDIDATE) or Path(WORKING_COPY_CANDIDATE)
        tempo = inspect_tempo_contract(als)
        payload = {
            "phase": "ARRANGEMENT PLAYBACK AT QN — PRODUCTION FIX",
            "ATOMIC TRANSPORT IMPLEMENTATION": {
                "command": "start_playback_at_qn",
                "live_side": "start_playing() then current_song_time=target in one main-thread callback",
                "verification": "next_tick via schedule_message(1)",
                "transport_primitive_version": TRANSPORT_PRIMITIVE_VERSION,
                "hello_transport_primitive_version": hello.get("transport_primitive_version"),
                "control_surface_loaded": hello.get("transport_primitive_version")
                == TRANSPORT_PRIMITIVE_VERSION,
            },
            "TEMPO CONTRACT": tempo,
            "MUSICAL WRITES": 0,
            "ASTRA CALLS": 0,
            "DSP": 0,
            "NO CAPTURE": True,
        }
        if not tempo.get("ok"):
            payload["status"] = tempo.get("status") or "TEMPO_MAPPING_UNAVAILABLE"
            payload["TRANSPORT"] = {"ok": False, "skipped": True}
        elif hello.get("transport_primitive_version") != TRANSPORT_PRIMITIVE_VERSION:
            payload["status"] = "CONTROL_SURFACE_RELOAD_REQUIRED"
            payload["instruction"] = (
                "install-script copied AbletonMCP. Toggle Control Surface AbletonMCP Off/On, "
                "then re-run session-arrangement-start. Do not capture."
            )
            payload["TRANSPORT"] = {"ok": False, "skipped": True}
        else:
            transport = run_atomic_transport_trust(adapter, evidence=evidence)
            payload["TRANSPORT"] = transport
            payload["32 RESULT"] = next(
                (row for row in transport.get("rows") or [] if row.get("target_qn") == 32.0), None
            )
            payload["172 RESULT"] = next(
                (row for row in transport.get("rows") or [] if row.get("target_qn") == 172.0), None
            )
            payload["292 RESULT"] = next(
                (row for row in transport.get("rows") or [] if row.get("target_qn") == 292.0), None
            )
            payload["IMPLIED START ERRORS"] = transport.get("implied_start_errors_qn")
            payload["status"] = transport.get("TRANSPORT_START_PROVENANCE")
        payload["READ-ONLY ACTIVITY MAP"] = map_lowend_candidates(als)
        payload["CANDIDATE REAL REGIONS"] = (payload["READ-ONLY ACTIVITY MAP"] or {}).get(
            "candidates"
        )
    finally:
        adapter.disconnect()
    path = evidence / "arrangement_playback_at_qn.json"
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    payload["artifact"] = str(path)
    logger.info(
        "session-arrangement-start status=%s astra=0 dsp=0 artifact=%s",
        payload.get("status"),
        path,
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))
    if payload.get("status") == "VERIFIED":
        return 0
    return 2
    from copilot.audio.arrangement_seek import run_seek_trust
    from copilot.audio.session_run1 import run_region1_recapture
    from copilot.daw.ableton_tcp import AbletonTcpAdapter

    adapter = AbletonTcpAdapter()
    try:
        adapter.connect()
        seek = run_seek_trust(adapter, evidence)
        payload = {"SEEK": seek}
        if seek.get("ok"):
            payload["REGION_1 RECAPTURE RESULT"] = run_region1_recapture(adapter, evidence)
        else:
            payload["REGION_1 RECAPTURE RESULT"] = {
                "status": "SKIPPED",
                "reason": "ARRANGEMENT_SEEK_FAILED",
            }
    finally:
        adapter.disconnect()
    logger.info(
        "session-seek-trust seek_ok=%s region1=%s",
        (payload.get("SEEK") or {}).get("ok"),
        (payload.get("REGION_1 RECAPTURE RESULT") or {}).get("status"),
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))
    if not (payload.get("SEEK") or {}).get("ok"):
        return 2
    if (payload.get("REGION_1 RECAPTURE RESULT") or {}).get("status") != "REGION_1 TRUSTWORTHY":
        return 2
    return 0


def _reason_revalidate(evidence: Path, logger) -> int:
    from copilot.reasoning.revalidate import run_m13, write_m13

    report = run_m13(source=evidence / "reason_real_full.json")
    path = write_m13(report, evidence / "reason_real_full_revalidated.json")
    logger.info(
        "reason-revalidate model_calls=%s false_positive=%s quality=%s dest=%s",
        report["model_calls"],
        report["VALIDATOR_FALSE_POSITIVE"],
        report["MODEL_QUALITY"],
        path,
    )
    print(json.dumps(
        {
            "VALIDATOR_FALSE_POSITIVE": report["VALIDATOR_FALSE_POSITIVE"],
            "CLEAR_TEMPORAL_EVIDENCE": report["CLEAR_TEMPORAL_EVIDENCE"],
            "CLEAR_SPECTRAL_EVIDENCE": report["CLEAR_SPECTRAL_EVIDENCE"],
            "STATUS_CONTRACT": report["STATUS_CONTRACT"],
            "ASTRA_HYPOTHESIS_QUALITY": report["ASTRA_HYPOTHESIS_QUALITY"],
            "ASTRA_CALIBRATION": report["ASTRA_CALIBRATION"],
            "MODEL_QUALITY": report["MODEL_QUALITY"],
            "CORE_SAFETY_AFFECTED": report["CORE_SAFETY_AFFECTED"],
            "previous_accepted": report["previous_accepted"],
            "previous_rejected": report["previous_rejected"],
            "sole_10ms_rejections": report["sole_10ms_rejections"],
            "model_calls": report["model_calls"],
            "artifact": str(path),
        },
        indent=2,
    ))
    return 0


def _reason_eval(evidence: Path, logger) -> int:
    from copilot.reasoning.eval import run_eval

    report = run_eval(include_real_model=True)
    path = evidence / "reason_eval.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info(
        "reason-eval musical_writes=%s accepted=%s real=%s",
        report["musical_writes"],
        report["metrics"]["accepted_core_fixtures"],
        report["real_model"].get("status"),
    )
    print(json.dumps(report, indent=2))
    if report["musical_writes"] != 0:
        return 2
    if report["metrics"]["accepted_core_fixtures"] < 6:
        return 2
    if not report["adversarial"]["injected_hallucination_rejected"]:
        return 2
    return 0


def _reason_real(evidence: Path, logger, *, gate: str) -> int:
    from copilot.reasoning.eval import CORE_FIXTURES
    from copilot.reasoning.real_gate import run_real_model_gate

    names = {
        "smoke": "reason_real_smoke.json",
        "mini": "reason_real_mini.json",
        "full": "reason_real_full.json",
    }
    if gate == "smoke":
        fixtures = ("CLEAR_NO_ACTION",)
        repeats = 1
        timeout_s = 180.0
    elif gate == "mini":
        fixtures = CORE_FIXTURES
        repeats = 1
        timeout_s = 180.0
    else:
        fixtures = CORE_FIXTURES
        repeats = 5
        timeout_s = 90.0
    report = run_real_model_gate(
        repeats=repeats,
        fixtures=fixtures,
        timeout_s=timeout_s,
        close_grounding=gate == "full",
        progress_path=evidence / "reason_real_progress.jsonl",
    )
    report["gate"] = gate
    path = evidence / names[gate]
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info(
        "reason-real gate=%s status=%s writes=%s provider=%s contract=%s",
        gate,
        report.get("status"),
        report.get("musical_writes"),
        report.get("provider"),
        (report.get("provider_contract") or {}).get("text.format.type")
        or (report.get("provider_contract") or {}).get("response_format.type"),
    )
    print(json.dumps(report, indent=2))
    if report.get("status") == "REAL_MODEL_UNAVAILABLE":
        return 2
    if report.get("musical_writes") != 0:
        return 2
    if gate == "smoke":
        cases = (report.get("fixtures") or [{}])[0].get("runs") or []
        row = cases[0] if cases else {}
        if row.get("category"):
            return 0
        return 2
    if gate == "mini":
        if report.get("schema_valid_rate", 0) <= 0:
            return 2
        if report.get("core_safety_result") != "PASS":
            return 2
        accepted_bad = report.get("grounding_violations_accepted") or {}
        if any(accepted_bad.values()):
            return 2
        return 0
    if report.get("REAL_MODEL_GROUNDING") == "VERIFIED" and report.get("core_safety_result") == "PASS":
        return 0
    return 2


def _production_write(
    evidence: Path,
    logger,
    *,
    dry_run: bool = False,
    plan_path: str | None = None,
    prepare_controlled: bool = False,
    execute_controlled: bool = False,
    audible_effect: bool = False,
    audible_effect_v2: bool = False,
    autonomous_improvement: bool = False,
    autonomous_evaluation_v2: bool = False,
    autonomous_evaluation_v3: bool = False,
    arrangement_source_isolation: bool = False,
    trace_performance: bool = False,
    delta: float = -0.02,
    target_track: str | None = None,
    source_plan_id: str = "plan_82608749d172",
    source_envelope_id: str = "env_5886f4fb0b",
    expected_before: float = 0.75,
) -> int:
    """Canonical write entrypoint.

    Modes:
    - --dry-run / --prepare-controlled-plan: validate/compile only (zero mutations)
    - --execute-controlled-write: CONTROLLED_WRITE_LOOP_V1 (FROZEN — do not reopen)
    - --audible-effect-verification: AUDIBLE_EFFECT_VERIFICATION_V1
    - --audible-effect-verification-v2: AUDIBLE_EFFECT_VERIFICATION_V2
    - --first-autonomous-musical-improvement: FIRST_AUTONOMOUS_MUSICAL_IMPROVEMENT_V1
    """
    from copilot.audio.capture_journal_recovery import unresolved_capture_journals
    from copilot.daw.ableton_tcp import AbletonTcpAdapter
    from copilot.daw.state_tokens import attach_tokens
    from copilot.musicplan import (
        create_controlled_volume_plan,
        dry_run_musicplan,
        plan_from_region_c_r6,
    )
    from copilot.musicplan.execute import (
        build_agent_tools,
        execute_controlled_write_loop,
    )
    from copilot.schemas.musicplan import MusicPlan, PlanIntentClass, PlanStatus

    if (
        not dry_run
        and not prepare_controlled
        and not execute_controlled
        and not audible_effect
        and not audible_effect_v2
        and not autonomous_improvement
        and not autonomous_evaluation_v2
        and not autonomous_evaluation_v3
        and not arrangement_source_isolation
    ):
        payload = {
            "status": "EXECUTION_DISABLED",
            "CANONICAL_WRITE_PATH": CANONICAL_WRITE_PATH,
            "CANONICAL_PRE_WRITE_PATH": CANONICAL_PRE_WRITE_PATH,
            "MUSICPLAN_GATE": "CLOSED",
            "MUSICPLAN_GATE_REASON": "explicit_mode_required",
            "MUSICAL WRITES": 0,
            "note": (
                "Use --dry-run, --prepare-controlled-plan, "
                "--execute-controlled-write, --audible-effect-verification, "
                "--audible-effect-verification-v2, "
                "--first-autonomous-musical-improvement, "
                "--autonomous-musical-evaluation-v2, "
                "--autonomous-musical-evaluation-v3, "
                "or --arrangement-active-source-isolation."
            ),
        }
        print(json.dumps(payload, indent=2))
        return 2

    from contextlib import ExitStack

    from copilot.perf.ableton import instrumented_rpc, rpc_breakdown
    from copilot.perf.trace import active_trace

    adapter = AbletonTcpAdapter()
    adapter.connect()
    stack = ExitStack()
    trace = None
    if trace_performance:
        # Measurement only. active_trace/instrumented_rpc never alter the run.
        trace = stack.enter_context(active_trace("production-write"))
        stack.enter_context(instrumented_rpc(adapter))
    try:
        session = adapter.snapshot(include_notes=False)
        attach_tokens(session)
        if trace is not None:
            trace.set_project(session.project_identity or session.project_token)

        if arrangement_source_isolation:
            from copilot.audio.arrangement_active_source_isolation import (
                run_arrangement_active_source_isolation_v1,
            )

            report = run_arrangement_active_source_isolation_v1(
                adapter,
                evidence=evidence,
            )
            status = report.get("ARRANGEMENT_ACTIVE_SOURCE_ISOLATION_V1")
            logger.info(
                "production-write arrangement-source-isolation status=%s writes=%s",
                status,
                report.get("MUSICAL WRITES"),
            )
            print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
            return 0 if status == "VERIFIED" else 2

        if autonomous_evaluation_v3:
            from copilot.audio.autonomous_musical_evaluation_v3 import (
                run_autonomous_musical_evaluation_v3,
            )

            report = run_autonomous_musical_evaluation_v3(
                adapter,
                evidence=evidence,
            )
            status = report.get("AUTONOMOUS_MUSICAL_EVALUATION_V3")
            logger.info(
                "production-write autonomous-eval-v3 status=%s decision=%s writes=%s",
                status,
                report.get("final_musical_decision"),
                report.get("MUSICAL WRITES"),
            )
            print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
            ok = status in {
                "AUTONOMOUS_IMPROVEMENT_VERIFIED",
                "NO_ACTION_REQUIRED",
                "INSUFFICIENT_EVIDENCE",
                "ACTION_NOT_AVAILABLE",
                "AUTONOMOUS_CHANGE_ROLLED_BACK",
                "DIAGNOSIS_UNSTABLE",
                "NO_INDEPENDENT_ACTIVE_HOLDOUT",
            }
            return 0 if ok else 2

        if autonomous_evaluation_v2:
            from copilot.audio.autonomous_musical_evaluation_v2 import (
                run_autonomous_musical_evaluation_v2,
            )

            report = run_autonomous_musical_evaluation_v2(
                adapter,
                evidence=evidence,
            )
            status = report.get("AUTONOMOUS_MUSICAL_EVALUATION_V2")
            logger.info(
                "production-write autonomous-eval-v2 status=%s decision=%s writes=%s",
                status,
                report.get("final_musical_decision"),
                report.get("MUSICAL WRITES"),
            )
            print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
            ok = status in {
                "AUTONOMOUS_IMPROVEMENT_VERIFIED",
                "NO_ACTION_REQUIRED",
                "INSUFFICIENT_EVIDENCE",
                "ACTION_NOT_AVAILABLE",
                "AUTONOMOUS_CHANGE_ROLLED_BACK",
                "DIAGNOSIS_UNSTABLE",
            }
            return 0 if ok else 2

        if autonomous_improvement:
            from copilot.audio.first_autonomous_musical_improvement_v1 import (
                run_first_autonomous_musical_improvement_v1,
            )

            report = run_first_autonomous_musical_improvement_v1(
                adapter,
                evidence=evidence,
            )
            status = report.get("FIRST_AUTONOMOUS_MUSICAL_IMPROVEMENT_V1")
            logger.info(
                "production-write autonomous-improvement status=%s decision=%s writes=%s",
                status,
                report.get("final_musical_decision"),
                report.get("MUSICAL WRITES"),
            )
            print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
            ok = status in {
                "AUTONOMOUS_IMPROVEMENT_VERIFIED",
                "NO_ACTION_REQUIRED",
                "INSUFFICIENT_EVIDENCE",
                "ACTION_NOT_AVAILABLE",
                "AUTONOMOUS_CHANGE_ROLLED_BACK",
            }
            return 0 if ok else 2

        if audible_effect_v2:
            from copilot.audio.audible_effect_verification_v2 import (
                run_audible_effect_verification_v2,
            )

            report = run_audible_effect_verification_v2(
                adapter,
                evidence=evidence,
                restored_audio=True,
            )
            logger.info(
                "production-write audible-effect-v2 verdict=%s attribution=%s",
                report.get("AUDIBLE_EFFECT_VERIFICATION_V2"),
                report.get("CAUSAL_ATTRIBUTION"),
            )
            print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
            return 0 if report.get("AUDIBLE_EFFECT_VERIFICATION_V2") == "VERIFIED" else 2

        if audible_effect:
            from copilot.audio.audible_effect_verification import (
                run_audible_effect_verification,
            )

            report = run_audible_effect_verification(
                adapter,
                evidence=evidence,
                target_track=target_track or "Coffee Leaf",
                expected_before=expected_before,
                delta=delta,
                restored_audio=True,
            )
            logger.info(
                "production-write audible-effect verdict=%s attribution=%s",
                report.get("AUDIBLE_EFFECT_VERIFICATION_V1"),
                report.get("CAUSAL_ATTRIBUTION"),
            )
            print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
            return 0 if report.get("AUDIBLE_EFFECT_VERIFICATION_V1") == "VERIFIED" else 2

        if execute_controlled:
            name = target_track or "Coffee Leaf"
            tools = build_agent_tools(
                adapter,
                journal_path=evidence / "agent_journal.jsonl",
            )
            report = execute_controlled_write_loop(
                tools,
                target_track=name,
                delta=delta,
                expected_before=expected_before,
                source_plan_id=source_plan_id,
                source_dry_run_envelope_id=source_envelope_id,
                persist_dir=evidence / "musicplans",
            )
            tools.transactions.save(evidence / "agent_transactions.json")
            out = evidence / "controlled_write_loop_v1.json"
            out.write_text(
                json.dumps(report, indent=2, ensure_ascii=False, default=str),
                encoding="utf-8",
            )
            report["public_artifact"] = str(out)
            logger.info(
                "production-write controlled loop plan=%s txn=%s status=%s writes=%s",
                report.get("plan_id"),
                report.get("transaction_id"),
                report.get("CONTROLLED_WRITE_LOOP_V1"),
                report.get("MUSICAL_WRITE_COUNT"),
            )
            print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
            return 0 if report.get("CONTROLLED_WRITE_LOOP_V1") == "VERIFIED" else 2

        if prepare_controlled:
            # Prefer an explicit track; otherwise first musical track that is not capture/lab.
            banned = {
                "Copilot Capture",
                "Copilot Capture Bass",
                "AI Test",
                "Master",
            }
            track = None
            if target_track:
                track = session.track_by_name(target_track)
            if track is None:
                for candidate in session.tracks:
                    if candidate.name in banned:
                        continue
                    if candidate.role in {"midi", "audio"}:
                        track = candidate
                        break
            if track is None:
                print(json.dumps({"status": "NO_TARGET", "MUSICAL WRITES": 0}, indent=2))
                return 2
            plan = create_controlled_volume_plan(
                session=session,
                track=track,
                delta=delta,
            )
            result = dry_run_musicplan(plan, session=session, persist_dir=evidence / "musicplans")
            public = {
                "status": "CONTROLLED_ENGINEERING_VALIDATION_DRY_RUN",
                "intent_class": PlanIntentClass.CONTROLLED_ENGINEERING_VALIDATION.value,
                "plan_id": result.plan_id,
                "plan_status": result.plan_status.value,
                "target_track": track.name,
                "delta_ableton_volume": delta,
                "current_volume": result.current_volume,
                "compiled": None
                if result.compiled is None
                else result.compiled.model_dump(mode="json"),
                "gate": result.gate,
                "would_mutate": False,
                "MUSICAL WRITES": 0,
                "EXECUTED": False,
                "artifact": result.artifact,
                "note": "Not autonomous musical improvement. Dry-run only.",
            }
            out = evidence / "musicplan_v1_controlled_dry_run.json"
            out.write_text(json.dumps(public, indent=2, ensure_ascii=False), encoding="utf-8")
            logger.info(
                "production-write controlled dry-run plan=%s status=%s writes=0",
                result.plan_id,
                result.plan_status,
            )
            print(json.dumps(public, indent=2, ensure_ascii=False))
            return 0 if result.plan_status == PlanStatus.READY_FOR_EXECUTION else 2

        if plan_path:
            raw = json.loads(Path(plan_path).read_text(encoding="utf-8"))
            # Accept either bare plan or wrapper with "plan" key.
            plan_obj = raw.get("plan") if isinstance(raw, dict) and "plan" in raw else raw
            plan = MusicPlan.model_validate(plan_obj)
            result = dry_run_musicplan(plan, session=session, persist_dir=evidence / "musicplans")
            public = {
                "status": "DRY_RUN",
                "plan_id": result.plan_id,
                "plan_status": result.plan_status.value,
                "gate": result.gate,
                "compiled": None
                if result.compiled is None
                else result.compiled.model_dump(mode="json"),
                "would_mutate": False,
                "MUSICAL WRITES": 0,
                "EXECUTED": False,
                "unresolved_capture_journals": unresolved_capture_journals(),
                "artifact": result.artifact,
            }
            print(json.dumps(public, indent=2, ensure_ascii=False, default=str))
            return 0 if result.musical_writes == 0 else 2

        # Default dry-run without plan: REGION_C abstention check
        r6 = evidence / "session_run1_astra_r6_region_c.json"
        plan = plan_from_region_c_r6(r6)
        result = dry_run_musicplan(plan, session=session, persist_dir=evidence / "musicplans")
        public = {
            "status": "REGION_C_ABSTENTION_DRY_RUN",
            "plan_id": result.plan_id,
            "plan_status": result.plan_status.value,
            "executable_actions": 0,
            "rejection_reason": plan.rejection_reason,
            "MUSICAL WRITES": 0,
            "EXECUTED": False,
            "artifact": result.artifact,
        }
        print(json.dumps(public, indent=2, ensure_ascii=False))
        return 0
    finally:
        try:
            adapter.disconnect()
        except Exception:
            pass
        # Close the trace before persisting so total_s is final. A cancelled or
        # failed run still writes its trace: that is when it is most useful.
        stack.close()
        if trace is not None:
            try:
                path = trace.persist(evidence, name="production_write_trace.json")
                rpc = rpc_breakdown(trace)
                logger.info(
                    "performance trace: %s rpc=%s rpc_s=%.2f total_s=%.2f -> %s",
                    trace.operation_id,
                    rpc["total_calls"],
                    rpc["total_s"],
                    trace.total_s,
                    path,
                )
                print()
                print(trace.render(min_s=0.05))
            except Exception as exc:  # noqa: BLE001 — never fail a run over a report
                logger.warning("performance trace not written: %s", exc)


def _capture_journal_recover(evidence: Path, logger) -> int:
    """Append-only recovery for unresolved capture journals. No musical writes."""
    from copilot.audio.capture_journal_recovery import recover_all_unresolved
    from copilot.daw.ableton_tcp import AbletonTcpAdapter

    adapter = AbletonTcpAdapter()
    adapter.connect()
    try:
        report = recover_all_unresolved(daw=adapter)
    finally:
        try:
            adapter.disconnect()
        except Exception:
            pass
    path = evidence / "capture_journal_recovery.json"
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    logger.info(
        "capture-journal-recover ok=%s remaining=%s",
        report.get("ok"),
        len(report.get("remaining_unresolved") or []),
    )
    print(json.dumps(report, indent=2, default=str))
    return 0 if report.get("ok") else 2


def _connect_live_or_block(evidence: Path, artifact: str) -> AbletonTcpAdapter | dict:
    detection = detect_ableton()
    probe = probe_session_ready()
    if probe.status != SESSION_READY:
        payload = {
            "status": "BLOCKED",
            "verification_class": probe.status,
            "reason": probe.status,
            "session": probe.to_dict(),
            "detection": detection.to_dict(),
            "NO MOCK SUCCESS": True,
            "NO WRITE": True,
            "writes_permitted": False,
        }
        (evidence / artifact).write_text(
            json.dumps(payload, indent=2, default=str), encoding="utf-8"
        )
        return payload
    adapter = AbletonTcpAdapter()
    try:
        adapter.connect()
        live = adapter.snapshot(include_notes=False)
        from copilot.audio.cross_project_bootstrap_v1 import retain_tokens

        retain_tokens(live)
        current_identity = live.project_identity or live.project_token or ""
        check = revalidate_project(probe.project_identity, probe)
        if current_identity and probe.project_identity and current_identity != probe.project_identity:
            adapter.disconnect()
            payload = {
                "status": "BLOCKED",
                "verification_class": "STALE_PLAN",
                "reason": "PROJECT_CHANGED",
                "session": probe.to_dict(),
                "revalidate": check,
                "current_identity": current_identity,
                "detection": detection.to_dict(),
                "NO MOCK SUCCESS": True,
                "NO WRITE": True,
                "writes_permitted": False,
                "PROJECT_REVALIDATED": False,
            }
            (evidence / artifact).write_text(
                json.dumps(payload, indent=2, default=str), encoding="utf-8"
            )
            return payload
        if not check.get("PROJECT_REVALIDATED"):
            adapter.disconnect()
            payload = {
                "status": "BLOCKED",
                "verification_class": str(check.get("status") or "SESSION_NOT_READY"),
                "reason": str(check.get("reason") or "PROJECT_NOT_REVALIDATED"),
                "session": probe.to_dict(),
                "revalidate": check,
                "detection": detection.to_dict(),
                "NO MOCK SUCCESS": True,
                "NO WRITE": True,
                "writes_permitted": False,
            }
            (evidence / artifact).write_text(
                json.dumps(payload, indent=2, default=str), encoding="utf-8"
            )
            return payload
    except Exception as exc:
        try:
            adapter.disconnect()
        except Exception:
            pass
        payload = {
            "status": "BLOCKED",
            "reason": str(exc),
            "verification_class": "SESSION_NOT_READY",
            "session": probe.to_dict(),
            "detection": detection.to_dict(),
            "NO MOCK SUCCESS": True,
            "NO WRITE": True,
            "writes_permitted": False,
        }
        (evidence / artifact).write_text(
            json.dumps(payload, indent=2, default=str), encoding="utf-8"
        )
        return payload
    return adapter


def _project_bootstrap(evidence: Path, logger) -> int:
    from copilot.audio.cross_project_bootstrap_v1 import bootstrap_project, write_blocker

    connected = _connect_live_or_block(evidence, "cross_project_bootstrap_v1.json")
    if isinstance(connected, dict):
        write_blocker(evidence, str(connected.get("reason") or "LIVE_UNAVAILABLE"), **connected)
        print(json.dumps(connected, indent=2, default=str))
        return 2
    try:
        report = bootstrap_project(connected, evidence=evidence)
    finally:
        connected.disconnect()
    logger.info(
        "project-bootstrap status=%s milestone=%s",
        report.get("status"),
        report.get("CROSS_PROJECT_BOOTSTRAP_V1"),
    )
    print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    return 0 if report.get("CROSS_PROJECT_BOOTSTRAP_V1") == "VERIFIED" else 2


def _project_ready(evidence: Path, logger) -> int:
    from copilot.audio.project_ready_v1 import project_ready

    connected = _connect_live_or_block(evidence, "project_ready_v1.json")
    if isinstance(connected, dict):
        print(json.dumps(connected, indent=2, default=str))
        return 2
    try:
        report = project_ready(connected, evidence=evidence)
    finally:
        connected.disconnect()
    logger.info("project-ready status=%s", report.get("PROJECT_READY"))
    print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    return 0 if report.get("PROJECT_READY") == "VERIFIED" else 2


def _cross_project_validate(evidence: Path, logger) -> int:
    from copilot.audio.cross_project_musical_validation_v1 import (
        NEXT_READ_ONLY_VERIFIED,
        NEXT_VOLUME_LOOP,
        run_cross_project_musical_validation,
    )

    connected = _connect_live_or_block(
        evidence, "cross_project_musical_validation_v1.json"
    )
    if isinstance(connected, dict):
        print(json.dumps(connected, indent=2, default=str))
        return 2
    try:
        report = run_cross_project_musical_validation(connected, evidence=evidence)
    finally:
        connected.disconnect()
    logger.info("cross-project-validate status=%s", report.get("status"))
    print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    status = str(report.get("status") or "")
    return 0 if status in {NEXT_READ_ONLY_VERIFIED, NEXT_VOLUME_LOOP} else 2


def _import_project(evidence: Path, logger, argv: list[str]) -> int:
    from copilot.importing.project_folder_import_v1 import import_project_folder

    folder = " ".join(argv).strip()
    if not folder:
        payload = {
            "status": "BLOCKED",
            "reason": "FOLDER_REQUIRED",
            "detail": 'Usage: python -m copilot.cli import-project "<folder>"',
            "MUSICAL WRITES": 0,
            "NO WRITE": True,
        }
        print(json.dumps(payload, indent=2))
        return 2
    report = import_project_folder(folder, evidence=evidence)
    logger.info("import-project status=%s", report.get("PROJECT_FOLDER_IMPORT_V1"))
    print(json.dumps(report, indent=2, default=str))
    return 0 if report.get("PROJECT_FOLDER_IMPORT_V1") == "READY" else 2


def _capabilities() -> int:
    from copilot.audio.capability_matrix_v1 import capability_matrix
    from copilot.audio.m4l_control_contract_v1 import control_contract

    payload = {
        **capability_matrix(),
        "m4l_control_contract": control_contract(),
        "canonical_commands": list(CANONICAL_COMMANDS),
        "lab_commands_require_flag": sorted(LAB_COMMANDS),
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))
    return 0


def _performance_report(
    evidence: Path,
    logger,
    *,
    host: str,
    port: int,
    repeats: int,
    include_live: bool,
) -> int:
    """Read-only latency report. Never writes to the session."""
    from copilot.perf.report import performance_report, persist_report, render_summary

    report = performance_report(
        evidence,
        host=host,
        port=port,
        repeats=repeats,
        include_live=include_live,
    )
    path = persist_report(report, evidence)
    logger.info("performance report written: %s", path)
    print(render_summary(report))
    print()
    print(report["tree"])
    print()
    print(json.dumps({"artifact": str(path), "operation_id": report["operation_id"]}, indent=2))
    blocked = (report.get("ableton") or {}).get("status") == "BLOCKED"
    return 2 if blocked and include_live else 0


def _analyze_project(
    evidence: Path,
    logger,
    argv: list[str],
    *,
    region_id: str | None,
    start_qn: float | None,
    end_qn: float | None,
) -> int:
    """Thin CLI adapter. Orchestration lives in Producer Runtime."""
    from copilot.runtime import Producer

    folder = " ".join(argv).strip() or None
    producer = Producer(evidence=evidence)
    result = producer.analyze_project(
        folder,
        region_preference=region_id,
        start_qn=start_qn,
        end_qn=end_qn,
    )
    payload = result.to_dict()
    logger.info(
        "analyze-project status=%s commands=%s",
        payload.get("status"),
        payload.get("AGENT_HIGH_LEVEL_COMMAND_COUNT"),
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))
    status = str(result.status.value if hasattr(result.status, "value") else result.status)
    if result.musical_writes != 0:
        return 2
    if status in {"SUCCEEDED"}:
        return 0
    if status == "BLOCKED":
        return 2
    return 2


def _producer_analyze(
    evidence: Path,
    logger,
    *,
    region_id: str | None,
    start_qn: float | None,
    end_qn: float | None,
    trace_performance: bool = False,
) -> int:
    from contextlib import ExitStack

    from copilot.audio.producer_analyze_v1 import PRESERVED_STATUSES, producer_analyze
    from copilot.perf.ableton import instrumented_rpc, rpc_breakdown
    from copilot.perf.trace import active_trace

    connected = _connect_live_or_block(evidence, "producer_analyze_v1.json")
    if isinstance(connected, dict):
        print(json.dumps(connected, indent=2, default=str))
        return 2
    stack = ExitStack()
    trace = None
    if trace_performance:
        # Measurement only; neither context manager alters the analysis.
        trace = stack.enter_context(active_trace("producer-analyze"))
        stack.enter_context(instrumented_rpc(connected))
    try:
        report = producer_analyze(
            connected,
            evidence=evidence,
            region_id=region_id,
            start_qn=start_qn,
            end_qn=end_qn,
        )
    finally:
        connected.disconnect()
        stack.close()
        if trace is not None:
            try:
                path = trace.persist(evidence, name="producer_analyze_trace.json")
                rpc = rpc_breakdown(trace)
                logger.info(
                    "performance trace: %s rpc=%s rpc_s=%.2f total_s=%.2f -> %s",
                    trace.operation_id,
                    rpc["total_calls"],
                    rpc["total_s"],
                    trace.total_s,
                    path,
                )
            except Exception as exc:  # noqa: BLE001 — never fail a run over a report
                logger.warning("performance trace not written: %s", exc)
    logger.info("producer-analyze status=%s", report.get("status"))
    print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    status = str(report.get("status") or "BLOCKED")
    return 0 if status in PRESERVED_STATUSES else 2


def _producer_run(
    evidence: Path,
    logger,
    *,
    mode: str | None,
    region_id: str | None,
    start_qn: float | None,
    end_qn: float | None,
) -> int:
    from copilot.audio.producer_analyze_v1 import PRESERVED_STATUSES
    from copilot.audio.producer_run_v1 import producer_run

    if mode is None:
        payload = {
            "status": "BLOCKED",
            "reason": "MODE_REQUIRED",
            "detail": "producer-run requires --mode analyze or --mode autonomous",
        }
        print(json.dumps(payload, indent=2))
        return 2
    connected = _connect_live_or_block(evidence, "producer_run_v1.json")
    if isinstance(connected, dict):
        print(json.dumps(connected, indent=2, default=str))
        return 2
    try:
        report = producer_run(
            connected,
            evidence=evidence,
            mode=mode,
            region_id=region_id,
            start_qn=start_qn,
            end_qn=end_qn,
        )
    finally:
        connected.disconnect()
    logger.info("producer-run mode=%s status=%s", mode, report.get("status"))
    print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    status = str(report.get("status") or "BLOCKED")
    return 0 if status in PRESERVED_STATUSES else 2


def _produce_tech_house(
    evidence: Path, *, prompt_file: str | None, prompt_words: list[str],
    template: str | None, sample_index: str | None,
    sample_root: str | None, workspace: str | None,
    reference: str | None = None,
    allow_ui_save: bool = False,
    producer_context_file: str | None = None,
    enable_midi_phrases: bool = False,
) -> int:
    """New-project production; reject an incomplete goal before opening Live."""
    from uuid import uuid4

    from copilot.human_eval.store import atomic_write
    from copilot.producer.goal import ProducerGoal

    report: dict = {"status": "BLOCKED", "MUSICAL_WRITES": 0}
    run_dir = evidence / "producer" / uuid4().hex
    try:
        if not prompt_file and not prompt_words:
            raise ValueError("PRODUCER_SUPER_PROMPT_REQUIRED")
        if prompt_file and prompt_words:
            raise ValueError("PRODUCER_PROMPT_INPUT_AMBIGUOUS")
        prompt = (
            Path(prompt_file).read_text(encoding="utf-8")
            if prompt_file else " ".join(prompt_words)
        )
        goal = ProducerGoal.from_prompt(prompt)
        production_context = None
        if producer_context_file:
            from copilot.producer.context import ProducerContext

            production_context = ProducerContext.model_validate_json(
                Path(producer_context_file).read_text(encoding="utf-8")
            )
            blockers = production_context.planning_blockers()
            if blockers:
                raise ValueError(f"PRODUCER_CONTEXT_BLOCKED:{','.join(blockers)}")
            if reference:
                raise ValueError("PRODUCER_REFERENCE_INPUT_AMBIGUOUS")
            if not production_context.brief.internal_audio_capture_authorized:
                raise ValueError("PRODUCER_NO_CAPTURE_DELIVERY_NOT_CERTIFIED")
        if not all((template, sample_index, sample_root, workspace)):
            raise ValueError("PRODUCER_TEMPLATE_LIBRARY_AND_WORKSPACE_REQUIRED")
        from copilot.importing.new_project_v1 import open_new_project, prepare_new_project
        from copilot.integration.autonomous_producer_alpha_v1 import run_alpha
        from copilot.sample_library.library_v1 import load_index

        index = load_index(Path(sample_index))
        if index is None or not index.assets:
            raise ValueError("PRODUCER_SAMPLE_INDEX_MISSING_OR_EMPTY")
        library_root = Path(sample_root).resolve(strict=True)
        if not library_root.is_dir():
            raise ValueError("PRODUCER_SAMPLE_LIBRARY_ROOT_INVALID")
        copy = prepare_new_project(template_als=template, workspace=workspace)
        report["copy"] = copy
        if copy.get("status") != "CREATED":
            report["reason"] = copy.get("reason", "NEW_PROJECT_COPY_FAILED")
            return 2
        opened = open_new_project(copy)
        report["opened"] = opened
        if opened.get("status") != "OPENED_EMPTY":
            report["reason"] = opened.get("reason", "PROJECT_NOT_READY")
            return 2
        report = run_alpha(
            evidence=run_dir, goal=goal,
            expected_project_path=Path(copy["working_als"]),
            library_index=index, authorized_library_root=library_root,
            opened_project=opened,
            reference_audio=Path(reference).resolve(strict=True) if reference else None,
            allow_ui_save=allow_ui_save,
            production_context=production_context,
            enable_midi_phrases=enable_midi_phrases,
        )
        report["copy"] = copy
        report["opened"] = opened
        return 0 if report.get("status") == "COMPLETE" else 2
    except (OSError, ValueError) as exc:
        report["reason"] = f"{type(exc).__name__}: {exc}"
        return 2
    finally:
        run_dir.mkdir(parents=True, exist_ok=True)
        atomic_write(run_dir / "report.json", report)
        print(json.dumps(report, indent=2, ensure_ascii=True, default=str))


def _prepare_producer_context(
    *, brief_file: str | None, lucas_brief: bool,
    reference_context_files: list[str], sample_index: str | None,
    sample_root: str | None, output: str | None,
    authorize_library_use: bool = False,
) -> int:
    from copilot.human_eval.store import atomic_write
    from copilot.producer.context import ProducerBrief, lucas_producer_brief, prepare_producer_context
    from copilot.sample_library.library_v1 import load_index
    from copilot.schemas.lucas_integration import ReferenceContext

    try:
        if bool(brief_file) == lucas_brief:
            raise ValueError("PRODUCER_ONE_BRIEF_SOURCE_REQUIRED")
        if bool(sample_index) != bool(sample_root):
            raise ValueError("PRODUCER_INDEX_AND_ROOT_REQUIRED_TOGETHER")
        if not output or Path(output).suffix.lower() != ".json":
            raise ValueError("PRODUCER_CONTEXT_JSON_OUTPUT_REQUIRED")
        destination = Path(output).resolve()
        inputs = [value for value in [brief_file, sample_index, *reference_context_files] if value]
        if any(destination == Path(value).resolve() for value in inputs):
            raise ValueError("PRODUCER_CONTEXT_OUTPUT_OVERWRITES_INPUT")
        if destination.exists():
            raise ValueError("PRODUCER_CONTEXT_OUTPUT_ALREADY_EXISTS")
        brief = (
            ProducerBrief.model_validate_json(Path(brief_file).read_text(encoding="utf-8"))
            if brief_file else lucas_producer_brief()
        )
        if authorize_library_use:
            if sample_root is None:
                raise ValueError("PRODUCER_AUTHORIZED_LIBRARY_ROOT_REQUIRED")
            brief = ProducerBrief.model_validate({
                **brief.model_dump(mode="json"),
                "authorized_library_root": str(Path(sample_root).resolve(strict=True)),
            })
        references = [
            ReferenceContext.model_validate_json(Path(path).read_text(encoding="utf-8"))
            for path in reference_context_files
        ]
        index = load_index(Path(sample_index)) if sample_index else None
        if sample_index and index is None:
            raise ValueError("PRODUCER_INDEX_UNAVAILABLE")
        context = prepare_producer_context(
            brief=brief, references=references, index=index,
            authorized_root=Path(sample_root) if sample_root else None,
        )
        atomic_write(destination, context.model_dump(mode="json"))
        blockers = context.planning_blockers()
        print(json.dumps({
            "status": "BLOCKED" if blockers else "PLANNING_READY",
            "context_path": str(destination), "blockers": blockers,
            "MUSICAL_WRITES": 0, "audio_processed": False, "model_called": False,
            "project_delivered": False,
        }, indent=2))
        return 2 if blockers else 0
    except (OSError, ValueError) as exc:
        print(json.dumps({
            "status": "BLOCKED", "reason": str(exc), "MUSICAL_WRITES": 0,
        }, indent=2))
        return 2


def _record_producer_supervision(
    *, review_file: str | None, state_root: str | None,
    session_id: str | None, reviewed_als: str | None,
) -> int:
    from copilot.producer.state import ProducerStateStore
    from copilot.producer.supervision import TrackSupervision, record_track_supervision
    from copilot.studio.persistence import disk_evidence

    try:
        if not all((review_file, state_root, session_id, reviewed_als)):
            raise ValueError("SUPERVISION_REVIEW_STATE_AND_ALS_REQUIRED")
        review = TrackSupervision.model_validate_json(
            Path(review_file).read_text(encoding="utf-8")
        )
        path = Path(reviewed_als)
        disk = disk_evidence(path)
        if path.suffix.lower() != ".als" or disk.get("sha256") != review.reviewed_als_sha256:
            raise ValueError("SUPERVISION_ALS_DIGEST_MISMATCH")
        store = ProducerStateStore(state_root)
        state = store.load(session_id)
        if state is None or state.session_id != session_id:
            raise ValueError("SUPERVISION_PRODUCER_SESSION_NOT_FOUND")
        store.save(record_track_supervision(state, review))
        print(json.dumps({
            "status": "HUMAN_REVIEW_RECORDED", "MUSICAL_WRITES": 0,
            "human_retained_fraction": review.human_retained_fraction,
            "technical_completion_certified": False,
        }, indent=2))
        return 0
    except (OSError, ValueError) as exc:
        print(json.dumps({"status": "BLOCKED", "reason": str(exc), "MUSICAL_WRITES": 0}, indent=2))
        return 2


def _doctor(evidence: Path, logger) -> int:
    from copilot.audio.doctor_v1 import doctor
    from copilot.daw.ableton_tcp import AbletonTcpAdapter
    from copilot.daw.session_ready_v1 import probe_session_ready

    probe = probe_session_ready()
    adapter = None
    if probe.status == SESSION_READY:
        adapter = AbletonTcpAdapter()
        try:
            adapter.connect()
        except Exception:
            adapter = None
    try:
        report = doctor(evidence=evidence, daw=adapter, session_probe=probe)
    finally:
        if adapter is not None:
            try:
                adapter.disconnect()
            except Exception:
                pass
    logger.info("doctor status=%s session=%s", report.get("status"), probe.status)
    # Console encodings on Windows are not guaranteed to represent all Live
    # labels; JSON escaping keeps the machine-readable CLI path portable.
    print(json.dumps(report, indent=2, ensure_ascii=True, default=str))
    return 0 if report.get("status") == "READY" else 2


def _regression_v1(evidence: Path, logger) -> int:
    from copilot.audio.regression_v1 import run_regression_v1

    report = run_regression_v1(evidence=evidence)
    logger.info("regression-v1 overall=%s", report.get("overall"))
    print(json.dumps(report, indent=2, default=str))
    return 0 if report.get("overall") == "PASS" else 2


def _manual_control_surface_action() -> str:
    return (
        "Restart Ableton Live, then Settings → Link, Tempo & MIDI → "
        "Control Surface = AbletonMCP, Input = None, Output = None"
    )


def _track_build(evidence: Path, logger, argv: list[str], live: bool = False, leave: bool = False) -> int:
    """Build a groovy/latin tech house track from 0 (library + recipe + groove + mixing + arrangement)."""
    from copilot.sample_library.library_v1 import load_index
    from copilot.musicplan.tech_house import build_tech_house_plan, TECH_HOUSE_BPM
    from copilot.musicplan.arrangement import (
        TECH_HOUSE_ARRANGEMENT,
        build_arrangement_mute_actions,
    )
    from copilot.musicplan.mixing import MIXING_CHAINS, MASTER_CHAIN, MIXING_PHILOSOPHY

    index_path = evidence / "sample_library_index.json"
    idx = load_index(index_path)
    if idx is None:
        print(json.dumps({"status": "BLOCKED", "error": "no sample index; run 'sample-library index' first"}, ensure_ascii=False))
        return 2

    # build the plan against a blank template (mock) or the real live bridge.
    from copilot.daw.state_tokens import attach_tokens
    if live:
        from copilot.daw.ableton_tcp import AbletonTcpAdapter
        daw = AbletonTcpAdapter(); daw.connect()
    else:
        from copilot.daw.mock import MockAbletonAdapter
        daw = MockAbletonAdapter(); daw.connect()
        daw.session_path = "trackbuild_lab.als"
        daw.session_name = "trackbuild_lab"
    session = daw.snapshot()
    attach_tokens(session)

    plan = build_tech_house_plan(index=idx, session=session)
    arrangement = build_arrangement_mute_actions(project_identity=session.project_identity)
    plan.actions.extend(arrangement)
    if leave:
        # leave the set in the full-groove (DROP) state: everything active
        drop = [s for s in TECH_HOUSE_ARRANGEMENT if s.name == "DROP"][0]
        plan.actions.extend(
            build_arrangement_mute_actions(project_identity=session.project_identity, arrangement=[drop])
        )

    # ---- print structure ----
    print(f"\n=== GROOVY / LATIN TECH HOUSE — {TECH_HOUSE_BPM} BPM ===\n")
    print("SONIDO (sample por pista, percusión-first):")
    for a in plan.actions:
        if a.action_type.value == "SAMPLE_LOAD":
            print(f"  {a.target.ref.get('name','?'):12s} {a.params.sample_uri}")

    print("\nMIXING (cadenas nativas):")
    for track, devices in MIXING_CHAINS.items():
        print(f"  {track:12s} {' → '.join(devices)}")
    print(f"  {'MASTER':12s} {' → '.join(MASTER_CHAIN)}")

    print("\nARRANGEMENT (substracción/variación):")
    for sec in TECH_HOUSE_ARRANGEMENT:
        muted = 10 - len(sec.active)
        print(f"  {sec.name:8s} {sec.bars:>3d}b  activas: {', '.join(sec.active)}  (mute: {muted})")

    print(f"\nPLAN: {len(plan.actions)} acciones de build + {len(arrangement)} de arreglo")

    # silence write/transaction INFO logs so the structure reads clean
    import logging
    for _name in ("copilot.write", "copilot.transactions", "copilot.agent", "copilot"):
        logging.getLogger(_name).setLevel(logging.WARNING)

    # ---- execute against mock (write + rollback) ----
    from copilot.musicplan.execute import build_agent_tools, execute_track_build_plan
    import tempfile
    tmp = Path(tempfile.mkdtemp())
    tools = build_agent_tools(daw, journal_path=tmp / "journal.jsonl")
    build_report = execute_track_build_plan(tools, plan=plan, session=session, persist_dir=tmp, leave=leave)
    print(f"\nEJECUCIÓN: {build_report['status']}  ·  tracks {build_report['after_track_count']} → rollback {build_report['restored_track_count']}  ·  RESTORE_VERIFIED={build_report['RESTORE_VERIFIED']}")

    result = {
        "status": build_report["status"],
        "style": "groovy latin tech house",
        "bpm": TECH_HOUSE_BPM,
        "plan_actions": len(plan.actions),
        "arrangement_actions": len(arrangement),
        "tracks": [a.params.track_name for a in plan.actions if a.action_type.value == "CREATE_TRACK"],
        "restore_verified": build_report["RESTORE_VERIFIED"],
    }
    return 0 if build_report["status"] == "CONTROLLED_WRITE_LOOP_COMPLETE" else 2


def _vibe(evidence: Path, logger, argv: list[str], live: bool = False, leave: bool = False) -> int:
    """prompt -> Astra -> MusicPlan (vibe coding). Falls back to deterministic if no Astra."""
    from copilot.sample_library.library_v1 import load_index
    from copilot.musicplan.astra_plan import build_plan_from_prompt

    intent = " ".join(argv) if argv else "dark percussive groovy tech house"
    index_path = evidence / "sample_library_index.json"
    idx = load_index(index_path)
    if idx is None:
        print(json.dumps({"status": "BLOCKED", "error": "no sample index; run 'sample-library index' first"}, ensure_ascii=False))
        return 2

    from copilot.daw.state_tokens import attach_tokens
    if live:
        from copilot.daw.ableton_tcp import AbletonTcpAdapter
        daw = AbletonTcpAdapter(); daw.connect()
    else:
        from copilot.daw.mock import MockAbletonAdapter
        daw = MockAbletonAdapter(); daw.connect()
        daw.session_path = "vibe_lab.als"; daw.session_name = "vibe_lab"
    session = daw.snapshot(); attach_tokens(session)

    plan, meta = build_plan_from_prompt(index=idx, session=session, intent=intent)

    astra_arrangement = meta.get("arrangement")
    plan._astra_arrangement = astra_arrangement
    plan._astra_patch_contracts = meta.get("patch_contracts") or []
    from copilot.musicplan.arrangement import build_arrangement_mute_actions, TECH_HOUSE_ARRANGEMENT
    selected_arrangement = astra_arrangement or TECH_HOUSE_ARRANGEMENT
    plan.actions.extend(
        build_arrangement_mute_actions(
            project_identity=session.project_identity,
            arrangement=selected_arrangement,
        )
    )
    if leave:
        drop = [s for s in TECH_HOUSE_ARRANGEMENT if s.name == "DROP"][0]
        plan.actions.extend(build_arrangement_mute_actions(project_identity=session.project_identity, arrangement=[drop]))

    print(f'\n=== VIBE: "{intent}" ===\n')
    print(f"astra_used: {meta['astra_used']}")
    print(f"reasoning: {meta.get('reasoning', '')}")
    print("\nSELECCIÓN (sample por pista):")
    for a in plan.actions:
        if a.action_type.value == "SAMPLE_LOAD":
            print(f"  {a.target.ref.get('name','?'):12s} {Path(a.params.sample_uri).name}")

    import logging
    for _n in ("copilot.write", "copilot.transactions", "copilot.agent", "copilot"):
        logging.getLogger(_n).setLevel(logging.WARNING)

    import tempfile
    tmp = Path(tempfile.mkdtemp())
    if leave:
        report = execute_lucas_plan_through_core(
            plan=plan,
            session=session,
            daw=daw,
            persist_dir=tmp,
        )
    else:
        tools = build_agent_tools(daw, journal_path=tmp / "journal.jsonl")
        report = execute_track_build_plan(tools, plan=plan, session=session, persist_dir=tmp, leave=False)
    if leave:
        print(f"\nEJECUCIÓN: {report['status']} · tracks {report['after_track_count']} · LEAVE (track armado)")
    else:
        print(f"\nEJECUCIÓN: {report['status']} · tracks {report['after_track_count']} → rollback {report['restored_track_count']} · RESTORE_VERIFIED={report['RESTORE_VERIFIED']}")
    ok = report["status"] in {"CONTROLLED_WRITE_LOOP_COMPLETE", "SAFE_WRITE_COMPLETE"}
    # End-to-end finalization: arrangement timeline + mix/master, before save.
    if ok and leave:
        from copilot.integration.lucas_core_v1 import execute_lucas_patch_contracts_through_core

        final_session = daw.snapshot()

        # ASTRAL planner integration: execute patch contracts before arrangement/mix.
        patch_contracts = getattr(plan, "_astra_patch_contracts", []) or []
        patch_report = execute_lucas_patch_contracts_through_core(
            contracts=patch_contracts,
            session=final_session,
            daw=daw,
            persist_dir=tmp,
        )
        print(
            f"\nASTRAL PATCH CONTRACTS: accepted={len(patch_report['accepted'])} "
            f"deferred={len(patch_report['deferred'])} Â· AUTHORITY=SafeWrite"
        )
        if False and patch_contracts:
            applied = 0
            failed = 0
            print(f"\nASTRAL PATCH CONTRACTS: {len(patch_contracts)}")
            for c in patch_contracts:
                try:
                    rep = {"ok": False, "routing_mode": "EXECUTION_DEFERRED", "patch": {}}
                    patch = rep.get("patch", {})
                    if rep.get("ok"):
                        applied += 1
                    else:
                        failed += 1
                    print(
                        f"  - {c.get('track')} / {c.get('device')} -> ok={rep.get('ok')}"
                        f" mode={rep.get('routing_mode')}"
                        f" applied={patch.get('applied')}"
                        f" viol={len(patch.get('violations', []))}"
                    )
                except Exception as exc:  # noqa: BLE001
                    failed += 1
                    print(f"  - {c.get('track')} / {c.get('device')} -> error ({exc})")
            print(f"ASTRAL PATCH RESULT: applied={applied} failed={failed}")

        arrangement = None
        arr = {"placed": 0, "looped": 0, "errors": []}
        print(f"\nARREGLO: {arr['placed']} clips · {arr['looped']} loops · {len(arr['errors'])} errores")
        for e in arr["errors"][:6]:
            print(f"  ! {e}")
        mix = {"volumes": 0, "master_devices": 0, "master_tweaks": 0, "errors": []}
        print(f"MIX/MASTER: {mix['volumes']} volúmenes · {mix['master_devices']} dispositivos master · {mix['master_tweaks']} tweaks · {len(mix['errors'])} errores")
        for e in mix["errors"][:6]:
            print(f"  ! {e}")
    # Post-build: persist the set, then run the structured critique.
    if ok and leave:
        try:
            saved = daw.save_session()
            if saved.get("saved"):
                print(f"\nGUARDADO: {saved.get('path') or 'Sin título'}")
            else:
                print("\nGUARDADO: el LOM de Ableton no expone save — guardá con Cmd+S en Live")
        except Exception as exc:  # noqa: BLE001
            print(f"\nGUARDADO: error ({exc})")

        from copilot.integration.lucas_core_v1 import run_lucas_critique

        after = daw.snapshot()
        critique = run_lucas_critique(plan=plan, session=after)
        if critique is not None:
            print(f"\nCRÍTICA: {critique.verdict.upper()}")
            for i in critique.top_3_issues:
                print(f"  {i.priority}. [{i.area}] {i.issue} → {i.minimal_fix}")
            if critique.reasoning:
                print(f"  ({critique.reasoning})")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
