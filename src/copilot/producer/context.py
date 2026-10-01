"""Read-only artistic context and indexed material for a complete-track plan."""

from __future__ import annotations

import hashlib
import math
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from copilot.producer.track_spec import TrackSpec
from copilot.sample_library.retrieval import SampleRetriever
from copilot.sample_library.schemas import LibraryIndex, SampleRole, SampleSetContext, SampleType
from copilot.schemas.lucas_integration import ReferenceContext, StyleContext


class MusicalFamily(StrEnum):
    DRUMS = "DRUMS"
    BASS = "BASS"
    MUSIC = "MUSIC"
    VOCALS = "VOCALS"
    FX = "FX"


# These are executable role names already understood by the existing recipe.
ROLE_SOURCES: dict[str, tuple[MusicalFamily, tuple[SampleRole, ...], str | None]] = {
    "Kick": (MusicalFamily.DRUMS, (SampleRole.KICK,), None),
    "Clap": (MusicalFamily.DRUMS, (SampleRole.CLAP, SampleRole.SNARE), None),
    "Closed Hat": (MusicalFamily.DRUMS, (SampleRole.CLOSED_HAT,), None),
    "Shaker": (MusicalFamily.DRUMS, (SampleRole.SHAKER,), None),
    "Conga": (MusicalFamily.DRUMS, (SampleRole.PERCUSSION,), "conga"),
    "Clave": (MusicalFamily.DRUMS, (SampleRole.PERCUSSION,), "clave"),
    "Perc Loop": (MusicalFamily.DRUMS, (SampleRole.TOP_LOOP, SampleRole.DRUM_LOOP), None),
    "Bass": (MusicalFamily.BASS, (SampleRole.BASS,), None),
    "Vocal": (MusicalFamily.VOCALS, (SampleRole.VOCAL, SampleRole.VOCAL_CHOP), None),
    "Stab": (MusicalFamily.MUSIC, (SampleRole.SYNTH, SampleRole.CHORD, SampleRole.MELODY), None),
    "Texture": (MusicalFamily.MUSIC, (SampleRole.TEXTURE,), None),
    "FX": (MusicalFamily.FX, (SampleRole.FX, SampleRole.RISER), None),
    "Impact": (MusicalFamily.FX, (SampleRole.IMPACT,), None),
    "Downlifter": (MusicalFamily.FX, (SampleRole.DOWNLIFTER,), None),
}


class ProducerBrief(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    schema_version: Literal["producer-brief-v1"] = "producer-brief-v1"
    intent: str = Field(min_length=1, max_length=4000)
    artist_direction: list[str] = Field(default_factory=list, max_length=12)
    confirmed_preferences: list[str] = Field(default_factory=list, max_length=32)
    provisional_hypotheses: list[str] = Field(default_factory=list, max_length=32)
    constraints: list[str] = Field(default_factory=list, max_length=32)
    open_decisions: list[str] = Field(default_factory=list, max_length=32)
    required_families: list[MusicalFamily] = Field(
        default_factory=lambda: list(MusicalFamily), min_length=1,
    )
    authorized_sample_sha256: list[str] = Field(default_factory=list)
    authorized_library_root: str | None = Field(default=None, min_length=1)
    deliverable: Literal["EDITABLE_ALS"] = "EDITABLE_ALS"
    final_render: Literal[False] = False
    internal_audio_capture_authorized: bool = False
    target_retained_fraction: float = Field(default=0.75, gt=0, le=1)

    @model_validator(mode="after")
    def validate_permissions(self) -> "ProducerBrief":
        if len(set(self.required_families)) != len(self.required_families):
            raise ValueError("PRODUCER_DUPLICATE_REQUIRED_FAMILY")
        for digest in self.authorized_sample_sha256:
            if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError("PRODUCER_INVALID_AUTHORIZED_SHA256")
        if len(set(self.authorized_sample_sha256)) != len(self.authorized_sample_sha256):
            raise ValueError("PRODUCER_DUPLICATE_AUTHORIZED_SHA256")
        return self

    def style_context(self) -> StyleContext:
        return StyleContext(
            preferences=self.confirmed_preferences,
            constraints=self.constraints,
            provenance={
                "source": "USER_BRIEF",
                "artist_direction": self.artist_direction,
                "provisional_hypotheses": self.provisional_hypotheses,
                "open_decisions": self.open_decisions,
                "semantic_listening": False,
            },
        )


def lucas_producer_brief() -> ProducerBrief:
    """Explicit opt-in brief, not a default imposed on other producers."""
    return ProducerBrief(
        intent=(
            "Produce an original complete tech-house / latin tribal Arrangement "
            "from reference stems and my indexed library; deliver an editable "
            "saved Ableton project for my supervision and targeted revisions."
        ),
        artist_direction=["Nacho Scoppa", "Jay de Lys"],
        confirmed_preferences=[
            "Percussion and bass should interact.",
            "Include drums, bass, musical parts, vocals and effects.",
            "Make creative decisions autonomously; I supervise the complete project.",
            "Revise requested elements without rebuilding unrelated parts.",
        ],
        constraints=[
            "Do not copy phrases or recordings from reference stems.",
            "Do not export a final audio render.",
            "Never claim that artist affinity guarantees a booking or contract.",
        ],
        open_decisions=[
            "Tempo and full-track duration.",
            "Bass timbre, vocal character and break/drop intensity.",
            "Which specific references express the desired musical direction.",
        ],
    )


class ProducerContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["producer-context-v1"] = "producer-context-v1"
    brief: ProducerBrief
    references: list[ReferenceContext] = Field(default_factory=list, max_length=16)
    samples: SampleSetContext
    blockers: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    no_write: Literal[True] = True

    @model_validator(mode="after")
    def validate_sources(self) -> "ProducerContext":
        identities = [ref.identity for ref in self.references]
        tokens = [ref.reference_state_token for ref in self.references]
        if len(set(identities)) != len(identities) or len(set(tokens)) != len(tokens):
            raise ValueError("PRODUCER_DUPLICATE_REFERENCE")
        for ref in self.references:
            if not ref.identity or not ref.reference_state_token:
                raise ValueError("PRODUCER_REFERENCE_IDENTITY_REQUIRED")
            if not math.isfinite(ref.tempo_bpm) or ref.tempo_bpm <= 0:
                raise ValueError("PRODUCER_REFERENCE_TEMPO_REQUIRED")
        allowed = set(self.brief.authorized_sample_sha256)
        seen: set[str] = set()
        for candidate in self.samples.candidates:
            digest = candidate.get("sha256")
            if not digest or (
                digest not in allowed
                and (
                    self.brief.authorized_library_root is None
                    or candidate.get("authorized_library_root") != self.brief.authorized_library_root
                )
            ):
                raise ValueError("PRODUCER_SAMPLE_PERMISSION_REQUIRED")
            if digest in seen:
                raise ValueError("PRODUCER_DUPLICATE_SAMPLE")
            seen.add(digest)
        for role, candidates in self.samples.roles.items():
            if role not in ROLE_SOURCES:
                raise ValueError(f"PRODUCER_UNKNOWN_ROLE:{role}")
            for candidate in candidates:
                if candidate not in self.samples.candidates:
                    raise ValueError("PRODUCER_ROLE_SAMPLE_NOT_IN_SHORTLIST")
        return self

    def planning_blockers(self) -> list[str]:
        blockers = list(self.blockers)
        if not self.references:
            blockers.append("REFERENCE_EVIDENCE_REQUIRED")
        available = {
            ROLE_SOURCES[role][0] for role, rows in self.samples.roles.items() if rows
        }
        blockers.extend(
            f"MATERIAL_REQUIRED:{family.value}"
            for family in self.brief.required_families if family not in available
        )
        if MusicalFamily.DRUMS in self.brief.required_families and not self.samples.roles.get("Kick"):
            blockers.append("MATERIAL_REQUIRED:Kick")
        if MusicalFamily.BASS in self.brief.required_families and not self.samples.roles.get("Bass"):
            blockers.append("MATERIAL_REQUIRED:Bass")
        return list(dict.fromkeys(blockers))

    def validate_track_spec(self, spec: TrackSpec) -> None:
        active = {role for section in spec.sections for role in section.active_roles}
        unavailable = sorted(role for role in active if not self.samples.roles.get(role))
        if unavailable:
            raise ValueError(f"PRODUCER_UNAVAILABLE_ROLES:{','.join(unavailable)}")
        families = {ROLE_SOURCES[role][0] for role in active}
        missing = set(self.brief.required_families) - families
        if missing:
            raise ValueError(
                "PRODUCER_REQUIRED_FAMILIES_MISSING:"
                + ",".join(sorted(family.value for family in missing))
            )
        if MusicalFamily.DRUMS in self.brief.required_families and "Kick" not in active:
            raise ValueError("PRODUCER_KICK_ROLE_REQUIRED")

    def candidate_context(self) -> dict[str, list[dict[str, Any]]]:
        return {role: list(rows) for role, rows in self.samples.roles.items()}

    def prompt_payload(self) -> dict[str, Any]:
        """Rights constrain Core; a list of permission hashes is not musical context."""
        payload = self.model_dump(mode="json")
        payload["brief"].pop("authorized_sample_sha256")
        payload["brief"].pop("authorized_library_root")
        payload["planning_blockers"] = self.planning_blockers()
        payload["semantic_listening"] = False
        return payload


def prepare_producer_context(
    *, brief: ProducerBrief, references: list[ReferenceContext],
    index: LibraryIndex | None = None, authorized_root: Path | None = None,
    bpm: float | None = None, per_role: int = 3,
) -> ProducerContext:
    """Read indexed descriptors and verify source bytes; never decode or render audio."""
    if not 1 <= per_role <= 10:
        raise ValueError("PRODUCER_SHORTLIST_SIZE_INVALID")
    if bpm is not None and (not math.isfinite(bpm) or bpm <= 0):
        raise ValueError("PRODUCER_BPM_INVALID")
    samples = SampleSetContext(task_id="complete-track-context")
    blockers: list[str] = []
    if index is None:
        blockers.append("INDEXED_LIBRARY_REQUIRED")
    else:
        if authorized_root is None:
            raise ValueError("PRODUCER_AUTHORIZED_LIBRARY_ROOT_REQUIRED")
        root = authorized_root.resolve(strict=True)
        if not root.is_dir():
            raise ValueError("PRODUCER_LIBRARY_ROOT_INVALID")
        allowed = set(brief.authorized_sample_sha256)
        whole_library = brief.authorized_library_root is not None
        if (
            brief.authorized_library_root is not None
            and Path(brief.authorized_library_root).resolve(strict=True) != root
        ):
            raise ValueError("PRODUCER_LIBRARY_PERMISSION_ROOT_MISMATCH")
        eligible = {}
        for digest, asset in index.assets.items():
            if digest not in allowed and not whole_library:
                continue
            if asset.sha256 != digest:
                raise ValueError("PRODUCER_INDEX_DIGEST_MISMATCH")
            path = Path(asset.path).resolve()
            relative = (root / asset.relative_path).resolve()
            if not path.is_relative_to(root) or relative != path:
                if digest in allowed:
                    raise ValueError("PRODUCER_SAMPLE_OUTSIDE_AUTHORIZED_ROOT")
                continue
            eligible[digest] = asset
        retriever = SampleRetriever(LibraryIndex(assets=eligible))
        summaries: dict[str, dict[str, Any]] = {}
        for role, (family, sources, hint) in ROLE_SOURCES.items():
            if family not in brief.required_families:
                continue
            hits = []
            for source in sources:
                for kind in (SampleType.ONE_SHOT, SampleType.LOOP):
                    if role in {"Shaker", "Conga", "Clave"} and kind is not SampleType.ONE_SHOT:
                        continue
                    hits.extend(retriever.search_samples(
                        role=source, text_query=hint, one_shot_or_loop=kind,
                        bpm=bpm if kind is SampleType.LOOP else None,
                        top_k=per_role,
                    ))
            # Text retrieval is a score, not an instrument classifier.
            if hint:
                hits = [hit for hit in hits if any(
                    reason == f"token '{hint}' in filename/path" for reason in hit.reasons
                )]
            hits.sort(key=lambda hit: (-hit.score, hit.asset.sha256))
            rows = []
            for hit in hits[:per_role]:
                asset = hit.asset
                if asset.sha256 not in summaries:
                    path = Path(asset.path).resolve(strict=True)
                    if not path.is_file() or not path.is_relative_to(root):
                        raise ValueError("PRODUCER_SAMPLE_OUTSIDE_AUTHORIZED_ROOT")
                    hasher = hashlib.sha256()
                    with path.open("rb") as handle:
                        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                            hasher.update(chunk)
                    if hasher.hexdigest() != asset.sha256:
                        raise ValueError("PRODUCER_SAMPLE_DIGEST_MISMATCH")
                summary = summaries.setdefault(asset.sha256, {
                    "id": asset.id, "sha256": asset.sha256,
                    "filename": asset.filename, "relative_path": asset.relative_path,
                    "semantic_role": asset.semantic_role.value,
                    "classification_confidence": asset.classification_confidence,
                    "sample_type": asset.sample_type.value,
                    "bpm": asset.bpm.value, "bpm_confidence": asset.bpm.confidence,
                    "pitch": asset.pitch.value, "pitch_confidence": asset.pitch.confidence,
                    "duration_s": asset.descriptors.duration_s,
                    "descriptors": asset.descriptors.model_dump(mode="json"),
                    "license": asset.provenance.get("license", "UNKNOWN"),
                    "permission": (
                        "USER_AUTHORIZED_FOR_THIS_PRODUCTION" if asset.sha256 in allowed
                        else "USER_AUTHORIZED_LIBRARY"
                    ),
                    "authorized_library_root": brief.authorized_library_root if whole_library else None,
                    "available": True,
                })
                rows.append(summary)
            samples.roles[role] = rows
        samples.candidates = list(summaries.values())
    return ProducerContext(
        brief=brief, references=references, samples=samples, blockers=blockers,
        limitations=[
            "Catalog metadata and indexed descriptors are not semantic listening.",
            "Reference stems remain read-only; estimated source roles retain uncertainty.",
            "Source verification is operation-scoped, not authority for a later Live write.",
            "Complete-track editing, save/reopen and artistic effectiveness need Hermes validation.",
        ],
    )
