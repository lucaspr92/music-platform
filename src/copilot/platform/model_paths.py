"""Host-owned locations for optional pre-trained model weights.

Weights are large, licence-encumbered artefacts that are NOT vendored in the repo.
This module resolves where they live so callers never hardcode a machine path.
"""

from __future__ import annotations

import os
from pathlib import Path

from copilot.platform.runtime_paths import runtime_root

MODELS_ROOT_VARIABLE = "MUSIC_PLATFORM_MODELS"


def models_root() -> Path:
    """Per-machine root for downloaded model weights."""
    override = os.environ.get(MODELS_ROOT_VARIABLE)
    if override:
        return Path(override).expanduser()
    return runtime_root() / "models"


def essentia_models_dir() -> Path:
    """Directory holding Essentia weight files and their metadata JSON."""
    return models_root() / "essentia"


def essentia_model_path(filename: str) -> Path:
    """Resolve one Essentia model file by bare name (weights or metadata)."""
    name = str(filename).strip()
    if not name or "/" in name or "\\" in name:
        raise ValueError("model filename must be a bare file name")
    return essentia_models_dir() / name
