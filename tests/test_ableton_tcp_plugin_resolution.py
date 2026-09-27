from __future__ import annotations

from copilot.daw.ableton_tcp import AbletonTcpAdapter


def test_load_device_prefers_search_match(monkeypatch):
    adapter = AbletonTcpAdapter()
    calls = {"search": [], "command": []}

    def fake_search(query: str, category: str = "all"):
        calls["search"].append((query, category))
        if category == "instruments":
            return {
                "results": [
                    {"name": "Serum2", "is_device": True, "uri": "query:Plugins#Serum2"}
                ]
            }
        return {"results": []}

    def fake_command(command_type: str, params=None, side_effect: bool = False):
        calls["command"].append((command_type, params, side_effect))
        return {"ok": True, "params": params}

    monkeypatch.setattr(adapter, "search_browser", fake_search)
    monkeypatch.setattr(adapter, "_command", fake_command)

    out = adapter.load_instrument_or_effect(2, "Serum2")

    assert out["params"]["uri"] == "query:Plugins#Serum2"
    assert calls["command"][0][0] == "load_instrument_or_effect"


def test_load_device_fallbacks_to_plugins_browse(monkeypatch):
    adapter = AbletonTcpAdapter()
    calls = {"browse": [], "command": []}

    def fake_search(query: str, category: str = "all"):
        return {"results": []}

    def fake_browse(path: list[str]):
        calls["browse"].append(list(path))
        if path == ["plugins"]:
            return {"items": [{"name": "Xfer", "is_folder": True, "is_loadable": False, "is_device": False, "uri": None}]}
        if path == ["plugins", "Xfer"]:
            return {
                "items": [
                    {
                        "name": "Serum2",
                        "is_folder": False,
                        "is_loadable": True,
                        "is_device": True,
                        "uri": "query:Plugins#Serum2",
                    }
                ]
            }
        return {"items": []}

    def fake_command(command_type: str, params=None, side_effect: bool = False):
        calls["command"].append((command_type, params, side_effect))
        return {"ok": True, "params": params}

    monkeypatch.setattr(adapter, "search_browser", fake_search)
    monkeypatch.setattr(adapter, "browse_path", fake_browse)
    monkeypatch.setattr(adapter, "_command", fake_command)

    out = adapter.load_instrument_or_effect(1, "devices/instruments/Serum2")

    assert out["params"]["uri"] == "query:Plugins#Serum2"
    assert ["plugins"] in calls["browse"]
    assert ["plugins", "Xfer"] in calls["browse"]
