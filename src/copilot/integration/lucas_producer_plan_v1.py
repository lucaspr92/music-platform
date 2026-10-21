"""Real Lucas musical planning, deterministic rendering, direct Live audition."""

from __future__ import annotations

import json
from pathlib import Path
import re
import time

from pydantic import ValidationError

from copilot.platform.runtime_paths import runtime_root
from copilot.producer.producer_plan_v1 import INTENT, ProducerPlanV1, render_producer_plan_v1
from copilot.producer import tech_house_kit_v2 as palette
from copilot.reasoning.provider import configured_http_provider


RUNTIME = runtime_root()
OUT = RUNTIME / "production-kit" / "lucas-producer-plan-v1"
A_WAV = RUNTIME / "production-kit" / "tech-house-direct-v1" / "audio" / "tech_house_production_kit_v1_direct.wav"
B_WAV = RUNTIME / "production-kit" / "tech-house-direct-v2" / "audio" / "tech_house_production_kit_v2.wav"
C_WAV = OUT / "audio" / "lucas_producer_plan_v1.wav"


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _redact_error(error: Exception) -> str:
    return re.sub(r"sk-[A-Za-z0-9_-]+", "[REDACTED]", f"{type(error).__name__}: {error}")[:800]


def _planning_prompt(*, repair: str | None = None) -> str:
    schema = json.dumps(ProducerPlanV1.model_json_schema(), ensure_ascii=False, separators=(",", ":"))
    base = (
        "You are Lucas, the producer. Return ONE JSON object: ProducerPlanV1. "
        "Make your own compositional decisions from this exact human intent; do not reuse a known pattern, "
        "do not emit Ableton commands or code. Musical decisions belong in the JSON, not in the renderer.\n"
        f"HUMAN INTENT (copy exactly into intent):\n{INTENT}\n"
        "Experiment constraints: 126 BPM, 4/4, 16 bars, seven roles. Fixed sonic palette: "
        "Kick 909 1.aif, Clap 909.aif, Hihat Closed 909.aif, Hihat Open 909.aif, "
        "Shaker Short.aif, House Bass.adv, House Stab.adg. Do not choose presets. "
        "Choose a bounded key/root and minor or dorian mode. Four-on-the-floor kick and clap on beats 2 and 4. "
        "Decide original bass degree/offset/duration/dynamic motif with 2-6 short attacks per bar; "
        "choose your own closed-hat, open-hat, percussion, and sparse stab motifs. "
        "Motif offsets are beat positions 0 inclusive to 4 exclusive within one 4/4 bar. "
        "Use additions and omissions at specific 1-based bars for variation. "
        "Each omission has {bar,offset}; each addition has {bar,event}. "
        "For each hit specify offset,duration,dynamic (ghost|normal|accent),timing (early|on_grid|late). "
        "Bass hits also specify degree (1|b3|4|5|b6|6|b7), optional octave_shift (-1|0|1); "
        "stab hits specify degrees array and register_octave (1|2). "
        "Swing roles only closed_hat/open_hat/perc/bass, amount_beats 0..0.045. "
        "Late microtiming adds .012 beats; on a weak sixteenth, swing is added too: keep total <=.045. "
        "Sections cover bars 1-16 with no gaps/overlap, at most 5, kick/clap/bass active always. "
        "Bass and stab motif concepts and short producer rationale required. "
        "Use short human-facing rationale, not chain-of-thought. "
        "Keep main bass/stab motifs distinct from a fixed grid sequence; no arpeggio or lead melody. "
        "No source measurements claimed. Return only valid JSON, no markdown.\n"
        f"JSON SCHEMA:\n{schema}"
    )
    if repair is not None:
        base += ("\nYour previous JSON failed validation. Correct it while preserving your musical decisions. "
                 "Do not copy a stock motif. Validation error: " + repair[:2500])
    return base


def _validate_raw(raw: str) -> tuple[ProducerPlanV1, dict]:
    plan = ProducerPlanV1.model_validate_json(raw)
    rendered = render_producer_plan_v1(plan)
    return plan, rendered


def generate_real_lucas_plan() -> tuple[ProducerPlanV1, dict, dict]:
    """One real configured-model call; at most one repair, no deterministic fallback."""
    provider = configured_http_provider()
    if provider is None:
        raise RuntimeError("REAL_LUCAS_PROVIDER_NOT_CONFIGURED")
    for folder in ("input", "plan", "evidence", "audio", "report"):
        (OUT / folder).mkdir(parents=True, exist_ok=True)
    (OUT / "input" / "input_intent.txt").write_text(INTENT, encoding="utf-8")
    provenance = {"provider": provider.identity, "model": provider.version,
                  "base_url": provider._base_url, "real_call": True, "repair_pass_used": False}
    for attempt in range(2):
        started = time.perf_counter()
        prompt = _planning_prompt(repair=provenance.get("validation_error") if attempt else None)
        raw = provider.reason_json_object(prompt, timeout_s=180)
        provenance.setdefault("calls", []).append({"attempt": attempt + 1,
                                                     "seconds": round(time.perf_counter() - started, 3),
                                                     "request_contract": provider.last_request_contract})
        if attempt == 0:
            (OUT / "plan" / "producer_plan_raw_initial.json").write_text(raw, encoding="utf-8")
        (OUT / "plan" / "producer_plan_raw.json").write_text(raw, encoding="utf-8")
        try:
            plan, rendered = _validate_raw(raw)
        except (ValidationError, ValueError) as exc:
            provenance["validation_error"] = str(exc)
            if attempt == 0:
                provenance["repair_pass_used"] = True
                continue
            _write_json(OUT / "evidence" / "lucas_plan_failure.json", provenance)
            raise RuntimeError("LUCAS_PRODUCER_PLAN_INVALID_AFTER_ONE_REPAIR") from exc
        _write_json(OUT / "plan" / "producer_plan_validated.json", plan.model_dump(mode="json"))
        _write_json(OUT / "plan" / "rendered_notes.json",
                    {role: [n.model_dump(mode="json") for n in notes] for role, notes in rendered.items()})
        provenance.pop("validation_error", None)
        _write_json(OUT / "evidence" / "lucas_provider.json", provenance)
        return plan, rendered, provenance
    raise AssertionError("unreachable")


def accept_persisted_after_register_validator_fix() -> tuple[ProducerPlanV1, dict, dict]:
    """Revalidate the unchanged second response after fixing an overstrict octave rule."""
    failure_path = OUT / "evidence" / "lucas_plan_failure.json"
    failure = json.loads(failure_path.read_text(encoding="utf-8"))
    if (failure.get("model") != "gpt-6-astra" or len(failure.get("calls", [])) != 2
            or "bass_root_midi must match chosen root in octave 1" not in failure.get("validation_error", "")):
        raise RuntimeError("PERSISTED_REGISTER_FAILURE_NOT_PROVEN")
    raw = (OUT / "plan" / "producer_plan_raw.json").read_text(encoding="utf-8")
    plan, rendered = _validate_raw(raw)
    provenance = {key: value for key, value in failure.items() if key != "validation_error"}
    provenance["validation_correction"] = (
        "Core validator had incorrectly forced the upper root register. "
        "Unchanged second Lucas response chose G MIDI 31, a bounded G sub-register. "
        "No third provider call and no musical field was rewritten."
    )
    provenance["response_used"] = "repair_attempt_2_unchanged"
    _write_json(OUT / "plan" / "producer_plan_validated.json", plan.model_dump(mode="json"))
    _write_json(OUT / "plan" / "rendered_notes.json",
                {role: [n.model_dump(mode="json") for n in notes] for role, notes in rendered.items()})
    _write_json(OUT / "evidence" / "lucas_provider.json", provenance)
    return plan, rendered, provenance


def novelty_report(rendered: dict) -> dict:
    b_plan = json.loads((B_WAV.parents[1] / "plan" / "groove_plan.json").read_text(encoding="utf-8"))
    comparisons = {}
    for current, prior in (("bass", "Bass"), ("closed_hat", "Closed Hat"), ("stab", "Stab")):
        current_onsets = [round(n.start_time, 6) for n in rendered[current]]
        prior_onsets = [round(float(n["start_time"]), 6) for n in b_plan["roles"][prior]["notes"]]
        comparisons[current] = current_onsets != prior_onsets
    result = {"sequence_different": comparisons, "novel_vs_v2": any(comparisons.values())}
    if not result["novel_vs_v2"]:
        raise RuntimeError("PRODUCER_PLAN_NOT_NOVEL_VS_V2")
    return result


if __name__ == "__main__":
    try:
        plan, rendered, provenance = generate_real_lucas_plan()
        novelty = novelty_report(rendered)
        _write_json(OUT / "evidence" / "novelty_report.json", novelty)
        print(json.dumps({"plan_valid": True, "provider": provenance["provider"],
                          "model": provenance["model"], "repair_pass_used": provenance["repair_pass_used"],
                          "key": plan.tonality.root, "mode": plan.tonality.mode,
                          "sections": len(plan.sections),
                          "notes": {role: len(notes) for role, notes in rendered.items()},
                          "novelty": novelty}, ensure_ascii=False))
    except Exception as error:
        print(json.dumps({"plan_valid": False, "error": _redact_error(error)}))
        raise SystemExit(1)
