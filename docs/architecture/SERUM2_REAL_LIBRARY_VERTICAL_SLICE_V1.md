# SERUM2_REAL_LIBRARY_VERTICAL_SLICE_V1

## Scope
Producer-lane milestone to prove real local Serum2 preset-first inputs are discoverable and selectable deterministically before any Ableton write.

## Implemented in this pass
- Added read-only local discovery + catalog builder: `src/copilot/producer/serum2_catalog.py`
- Added focused tests: `tests/test_serum2_catalog.py`
- Generated runtime evidence artifact: `logs/serum2_discovery_v1.json`

## Authorities
- No new write authority introduced.
- No direct Ableton writes in this pass.
- Existing SafeWrite path remains unchanged.

## Inputs discovered (this machine)
- Serum2 binaries found under `/Library/Audio/Plug-Ins` (VST3/AU).
- Preset roots found:
  - `/Library/Audio/Presets/Xfer Records/Serum 2 Presets`
  - `/Library/Audio/Presets/Xfer Records/Serum Presets`
  - `/Users/lucas/Music/Ableton/User Library`
- Total presets discovered: 3112
- Serum 2 classified presets: 980

## Deterministic selection evidence
- Existing selector reused: `PresetCatalog.select(...)` from `preset_catalog.py`
- Bass role request (`plugin='Serum 2', role='bass'`) returns `SELECTED` with deterministic ordering.
- Evidence persisted in `logs/serum2_discovery_v1.json`.

## Limitations (still pending for full vertical slice)
- No real SafeWrite device-load/preset-load executed yet in this pass.
- No authoritative preset readback from Ableton yet.
- No audio preview capture yet.

## Acceptance status (partial)
- SERUM2 INSTALLATION: VERIFIED
- REAL PRESET ROOT: VERIFIED
- REAL PRESET CATALOG: VERIFIED
- BASS PRESET CANDIDATES: VERIFIED
- DETERMINISTIC SELECTION: VERIFIED
- Remaining stages: typed action execution via ProductionCompiler/SafeWrite + readback + preview.


## Update: preset-first role selection bridge
- Added `src/copilot/producer/serum2_preset_first.py`:
  - `select_real_serum2_preset(...)` produces deterministic role-safe selection reports from the real local catalog.
  - `identify_serum2_device(...)` gives strongest available session identity evidence without writing.
- Added tests: `tests/test_serum2_preset_first.py`.
- Added artifact: `logs/serum2_preset_selection_v1.json`.

## Current blocker for next phase
- Real Ableton runtime not active (`127.0.0.1:9877` listener absent), so
  safe write execution for "load Serum2 device + apply selected preset" was not run in this pass.


## Update: external plugin URI resolution for SafeWrite device-load
- Extended `AbletonTcpAdapter.load_instrument_or_effect` with plugin-aware URI resolution:
  - multi-category `search_browser` probing (`audio_effects`, `midi_effects`, `instruments`, `drums`, `sounds`, `all`)
  - fallback recursive `browse_path(["plugins"])` resolution for third-party plugins not indexed by bridge `search_browser(all)`.
- Added tests: `tests/test_ableton_tcp_plugin_resolution.py`.
- Real probe evidence (`logs/serum2_safe_device_probe_v2.json`):
  - SafeWrite create-track: PASS
  - SafeWrite load-device (`Serum2` request): PASS (`Serum` inserted)
  - rollback cleanup: PASS (no residue track)
