"""The kit can be rendered in another MINOR key and at another tempo, without
changing the frozen V1/V2 defaults or transposing the one-shot roles."""
from copilot.producer import tech_house_kit_v1 as v1, tech_house_kit_v2 as v2


def test_defaults_are_frozen():
    for plan in (v1.groove_plan(), v2.groove_plan()):
        assert plan["tempo_bpm"] == 126.0
        assert plan["key"] == "G minor"
        assert plan["key_transposition_semitones"] == 0


def test_transposition_is_shortest_motion():
    assert v1.transposition_to("G minor") == 0
    assert v1.transposition_to("D minor") == -5      # down a 5th, not up a 4th
    assert v1.transposition_to("A minor") == 2
    assert v1.transposition_to("C minor") == 5       # C is 5 up / 7 down -> up
    assert v1.transposition_to("F# minor") == -1


def test_only_pitched_roles_move():
    base = v1.groove_plan()
    moved = v1.groove_plan(tempo_bpm=128.0, key="D minor")
    assert moved["tempo_bpm"] == 128.0 and moved["key"] == "D minor"
    for role, item in ((i.role, i) for i in v1.KIT):
        first = {(n["pitch"], n["start_time"], n["velocity"]) for n in base["roles"][role]["notes"]}
        after = {(n["pitch"], n["start_time"], n["velocity"]) for n in moved["roles"][role]["notes"]}
        if item.is_one_shot:
            assert first == after, f"one-shot role {role} must not move"
        else:
            assert {(p - 5, s, v) for p, s, v in first} == after, f"{role} must move by -5"


def test_major_mode_fails_closed():
    for key in ("D major", "G major"):
        try:
            v1.transposition_to(key)
        except ValueError as exc:
            assert "UNSUPPORTED_MODE" in str(exc)
        else:
            raise AssertionError(f"{key} must not be accepted: the material is minor")


def test_v2_transposes_too_and_still_matches_v1_contract():
    plan = v2.groove_plan(tempo_bpm=128.0, key="D minor")
    assert plan["tempo_bpm"] == 128.0 and plan["key_transposition_semitones"] == -5
    kick = {n["pitch"] for n in plan["roles"]["Kick"]["notes"]}
    assert kick == {60}
    assert min(n["pitch"] for n in plan["roles"]["Bass"]["notes"]) == 36   # F1 41 -> C1
