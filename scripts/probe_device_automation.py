"""T2 feasibility: can a native effect be loaded and its filter automated from the bridge?

Answers three questions with evidence:
  1. does the browser resolve 'Auto Filter' unambiguously?
  2. does a NATIVE device expose its full parameter list (the Serum lesson was 1 param)?
  3. can set_clip_automation actually write an envelope that reads back?

Uses a temporary track and deletes it, so the replica is left untouched.
"""
from __future__ import annotations
import json, time
from pathlib import Path

# Scratch artifacts stay in the gitignored logs/ dir, resolved from the repo root so
# the script runs from any working directory.
SCRATCH = Path(__file__).resolve().parents[1] / "logs"


from copilot.daw.ableton_tcp import AbletonTcpAdapter

TEST_TRACK = "ZZ T2 PROBE"
REPORT = (SCRATCH / "t2_device_automation_probe.json")


def main() -> int:
    daw = AbletonTcpAdapter()
    daw.connect()
    report: dict = {"status": "RUNNING"}
    index = None
    started = time.perf_counter()
    try:
        # 1) resolver el efecto en el browser
        found = daw.search_browser("Auto Filter", "all")
        rows = [r for r in (found.get("items") or found.get("results") or [])
                if r.get("name") == "Auto Filter" and r.get("is_loadable")]
        report["browser"] = {"matches": len(rows), "names": [r.get("name") for r in rows]}
        if len(rows) != 1:
            report["status"] = "BROWSER_AMBIGUOUS"
            return 0
        uri = str(rows[0]["uri"])

        # track temporal
        index = int(daw.create_audio_track(TEST_TRACK)["index"])
        report["track_index"] = index
        daw.load_instrument_or_effect(index, uri)
        devices = daw.snapshot(include_notes=False).tracks[index].devices
        report["devices"] = [d.name for d in devices]
        if not devices:
            report["status"] = "DEVICE_NOT_LOADED"
            return 0

        # 2) lista de parametros del device nativo
        params = daw.get_device_parameters(index, 0)
        plist = params.get("parameters") or params.get("items") or []
        names = [p.get("name") for p in plist]
        freq = next((p for p in plist if str(p.get("name", "")).lower() == "frequency"), None)
        report["parameter_count"] = len(plist)
        report["parameters"] = [{"name": p.get("name"), "index": p.get("index"),
                                 "value": p.get("value"), "min": p.get("min"), "max": p.get("max")}
                                for p in plist[:14]]
        report["frequency_parameter"] = freq
        if freq is None:
            report["status"] = "NO_FREQUENCY_PARAMETER"
            return 0

        # 3) escribir un envelope en un clip y leerlo de vuelta
        # the bridge requires an ABSOLUTE path; take a real one from the replica plan
        plan = json.loads((SCRATCH / "tribal_replica_plan.json").read_text(encoding="utf-8"))
        source = plan["palette"]["perc"]["path"]
        report["probe_clip_source"] = source
        clip = daw.create_audio_clip(index, 0, source)
        report["clip_length_beats"] = clip.get("length")
        lo, hi = float(freq.get("min", 0.0)), float(freq.get("max", 1.0))
        envelope = [{"time": 0.0, "value": hi},
                    {"time": 8.0, "value": lo + (hi - lo) * 0.6},
                    {"time": 16.0, "value": lo + (hi - lo) * 0.25},
                    {"time": 24.0, "value": lo}]
        written = daw._command("set_clip_automation", {          # no public wrapper exists yet
            "track_index": index, "clip_index": 0,
            "parameter_name": "Frequency", "envelope_data": envelope}, side_effect=True)
        report["set_clip_automation_response"] = written
        readback = daw.get_clip_automation(index, 0, "Frequency")
        report["readback"] = readback
        report["status"] = "PROBED"
    except Exception as exc:
        report["status"] = "FAILED"
        report["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if index is not None and report.get("status") in {"PROBED", "FAILED"}:
            try:
                daw._command("delete_track", {"track_index": index}, side_effect=True)
                report["cleaned_up"] = True
            except Exception as exc:
                report["cleaned_up"] = f"FAILED: {type(exc).__name__}: {exc}"
        report["seconds"] = round(time.perf_counter() - started, 1)
        try:
            daw.disconnect()
        except Exception:
            pass
        REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
