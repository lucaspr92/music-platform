"""Properties the stem-upgraded plan must hold.

The plan itself is a scratch artifact (logs/ is gitignored), so these tests are skipped
until scripts/plan_replica_v2_from_stems.py has run. What they lock down is the shape of
the answer: the bass window comes from the measurement, and the hand-percussion layers are
nested, so adding a busier section can only ever ADD a layer, never reshuffle which
instrument plays.
"""
import json
from pathlib import Path

import pytest

SCRATCH = Path(__file__).resolve().parents[1] / "logs"
PLAN = SCRATCH / "tribal_replica_plan_v2.json"
STEMS = SCRATCH / "reference_stems_report.json"

pytestmark = pytest.mark.skipif(
    not PLAN.is_file() or not STEMS.is_file(),
    reason="run scripts/measure_reference_stems.py and scripts/plan_replica_v2_from_stems.py first",
)

LAYER_POOL = ("conga", "djembe", "bongo", "shaker")


def _plan() -> dict:
    return json.loads(PLAN.read_text(encoding="utf-8"))


def _bars(element: str, plan: dict) -> set:
    out: set[int] = set()
    for run in plan["placements"][element]:
        out.update(range(int(run["start_bar"]), int(run["end_bar"])))
    return out


def test_the_bass_window_comes_from_the_bass_stem():
    plan = _plan()
    stems = json.loads(STEMS.read_text(encoding="utf-8"))
    present = [int(s["start_bar"]) for s in stems["sections"] if s["stem_present"].get("bass")]
    assert present, "the reference must have at least one bass section"
    bass = _bars("bass", plan)
    assert min(bass) == min(present), "the bass cannot start before the stem says it does"
    # and it must actually be silent early: the intro is measured well below the median
    intro = [s["bass_rms_db"] for s in stems["sections"] if int(s["start_bar"]) < 25]
    assert max(intro) < stems["stem_median_db"]["bass"] - 20.0


def test_layers_are_nested_not_reshuffled():
    plan = _plan()
    coverage = [_bars(element, plan) for element in LAYER_POOL]
    for lower, higher in zip(coverage, coverage[1:]):
        assert higher <= lower, "a busier section may only ADD a layer"
    assert coverage[0], "the base layer must cover something"
    assert len(coverage[-1]) < len(coverage[0]), "the texture layer must be the smallest"


def test_every_placement_respects_the_reference_length():
    plan = _plan()
    total = plan["total_bars"]
    for element, runs in plan["placements"].items():
        for run in runs:
            assert 0 <= run["start_bar"] < run["end_bar"] <= total + 2, f"{element} out of range"
