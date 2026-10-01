"""Bounded experimental phrase creation; observable preservation, no DAW writes."""

from copilot.daw.state_tokens import canonical_target, token_of
from copilot.producer.arrangement_score import MidiPhrase
from copilot.schemas.session import SessionState, TrackState


MIDI_PHRASE_CAPABILITIES = frozenset({
    "session.read", "clip.create", "clip.delete", "clip.read_notes", "clip.write_notes",
})


def empty_phrase_slot_blocker(track: TrackState, clip_index: int) -> str | None:
    if track.role != "midi" or not track.stable_id:
        return "MIDI_PHRASE_TARGET_NOT_MIDI"
    if track.clip_slot_count is None or track.empty_clip_slots is None:
        return "MIDI_PHRASE_SLOT_INVENTORY_UNAVAILABLE"
    if (
        type(clip_index) is not int or not 0 <= clip_index < track.clip_slot_count
        or len(track.empty_clip_slots) != len(set(track.empty_clip_slots))
        or any(type(index) is not int or not 0 <= index < track.clip_slot_count
               for index in track.empty_clip_slots)
    ):
        return "MIDI_PHRASE_SLOT_INVENTORY_INVALID"
    occupied = [clip.slot_index for clip in track.clips]
    if (
        len(occupied) != len(set(occupied))
        or any(type(index) is not int or not 0 <= index < track.clip_slot_count for index in occupied)
        or set(track.empty_clip_slots) != set(range(track.clip_slot_count)) - set(occupied)
    ):
        return "MIDI_PHRASE_SLOT_INVENTORY_INVALID"
    if clip_index not in track.empty_clip_slots or any(
        clip.slot_index == clip_index for clip in track.clips
    ):
        return "MIDI_PHRASE_SLOT_OCCUPIED"
    return None


def phrase_from_arguments(arguments: dict) -> MidiPhrase:
    notes = arguments.get("notes") or []
    if not isinstance(notes, list) or any(not isinstance(note, dict) for note in notes):
        raise ValueError("MIDI_PHRASE_NOTE_PAYLOAD_INVALID")
    if any(note.get("mute", False) is not False for note in notes):
        raise ValueError("MIDI_PHRASE_MUTED_NOTES_UNSUPPORTED")
    return MidiPhrase(
        length_qn=arguments["length_beats"], reason="Explicit producer MIDI phrase",
        notes=[{key: value for key, value in note.items() if key != "mute"} for note in notes],
    )


def phrase_preservation_token(
    session: SessionState, *, excluded_slots: dict[str, set[int]],
) -> str:
    tracks = []
    for track in session.tracks:
        excluded = excluded_slots.get(track.stable_id, set())
        visible = track.model_copy(update={
            "clips": [clip for clip in track.clips if clip.slot_index not in excluded],
        })
        tracks.append({
            "stable_id": track.stable_id, "state": canonical_target(visible),
            "devices": [device.model_dump(mode="json") for device in visible.devices],
            "clips": [
                {"stable_id": clip.stable_id, "slot_index": clip.slot_index,
                 "sample_uri": clip.sample_uri, "is_audio": clip.is_audio}
                for clip in sorted(visible.clips, key=lambda clip: clip.slot_index)
            ],
            "slot_count": track.clip_slot_count,
            "empty_slots": (
                sorted(set(track.empty_clip_slots) - excluded)
                if track.empty_clip_slots is not None else None
            ),
        })
    return token_of({
        "project_identity": session.project_identity,
        "incarnation": session.session_incarnation_id,
        "tempo": session.transport.tempo,
        "meter": [session.transport.signature_numerator, session.transport.signature_denominator],
        "tracks": tracks,
    })
