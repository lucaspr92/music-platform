"""Internal DSP providers. Producer Runtime must not import these objects.

Licence note: Essentia is AGPL-3.0 and its pre-trained models are NonCommercial
(CC BY-NC-SA/ND). Using it is a deliberate, recorded product decision -- the
provider reports what is actually installed instead of hiding it, and the
obligation is carried in `pyproject.toml` (extra `essentia`) and in the tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from copilot.schemas.dsp import DspLimitation


@dataclass(frozen=True)
class ProviderStatus:
    provider_id: str
    available: bool
    version: str
    limitation: DspLimitation | None = None


def _module_version(mod: Any, fallback: str) -> str:
    for attr in ("__version__", "version"):
        value = getattr(mod, attr, None)
        if value:
            return str(value)
    return fallback


def numpy_status() -> ProviderStatus:
    try:
        import numpy as np

        return ProviderStatus("numpy", True, _module_version(np, "installed"))
    except Exception as exc:  # noqa: BLE001
        return ProviderStatus(
            "numpy",
            False,
            "missing",
            DspLimitation("NUMPY_UNAVAILABLE", str(exc)),
        )


def scipy_status() -> ProviderStatus:
    try:
        import scipy

        return ProviderStatus("scipy", True, _module_version(scipy, "installed"))
    except Exception as exc:  # noqa: BLE001
        return ProviderStatus(
            "scipy",
            False,
            "missing",
            DspLimitation("SCIPY_UNAVAILABLE", str(exc)),
        )


def pyloudnorm_status() -> ProviderStatus:
    try:
        import pyloudnorm as pyln

        return ProviderStatus("pyloudnorm", True, _module_version(pyln, "0.2.0"))
    except Exception as exc:  # noqa: BLE001
        return ProviderStatus(
            "pyloudnorm",
            False,
            "missing",
            DspLimitation("PYLOUDNORM_UNAVAILABLE", str(exc)),
        )


def soundfile_status() -> ProviderStatus:
    try:
        import soundfile as sf

        return ProviderStatus("soundfile", True, _module_version(sf, "installed"))
    except Exception as exc:  # noqa: BLE001
        return ProviderStatus(
            "soundfile",
            False,
            "missing",
            DspLimitation("SOUNDFILE_UNAVAILABLE", str(exc)),
        )


def librosa_status() -> ProviderStatus:
    try:
        import librosa  # type: ignore

        return ProviderStatus("librosa", True, _module_version(librosa, "installed"))
    except Exception:
        return ProviderStatus(
            "librosa",
            False,
            "missing",
            DspLimitation(
                "LIBROSA_NOT_INSTALLED",
                "librosa is not installed. PHYSICAL_DSP_V2 uses numpy/scipy; no silent skip of quality.",
            ),
        )


def essentia_status() -> ProviderStatus:
    """Report the installed Essentia, not the licence posture.

    An available provider must not carry a limitation: `_inventory_limitations`
    only appends limitations for unavailable providers, and `tonal.analyze_tonal`
    puts this object into a validated ``list[DspLimitation]``.
    """
    try:
        import essentia  # type: ignore

        return ProviderStatus("essentia", True, _module_version(essentia, "installed"))
    except Exception as exc:  # noqa: BLE001
        return ProviderStatus(
            "essentia",
            False,
            "missing",
            DspLimitation("ESSENTIA_UNAVAILABLE", f"essentia is not installed: {exc}"),
        )


def require(status: ProviderStatus) -> DspLimitation | None:
    if status.available:
        return None
    return status.limitation or DspLimitation(f"{status.provider_id.upper()}_UNAVAILABLE", "")


def provider_inventory() -> list[ProviderStatus]:
    return [
        numpy_status(),
        scipy_status(),
        pyloudnorm_status(),
        soundfile_status(),
        librosa_status(),
        essentia_status(),
    ]
