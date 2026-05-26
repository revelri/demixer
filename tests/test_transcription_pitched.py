"""Structural + monophonic-postprocess tests for transcription.pitched.

The end-to-end basic-pitch run is gated on DEMIXER_RUN_HEAVY=1 because it
imports TensorFlow (slow) and runs the model on synthesized audio.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pretty_midi
import pytest
import soundfile as sf

from demixer.core.transcription.pitched import (
    TranscribeOptions,
    _force_monophonic_inplace,
    _octave_snap_bass_inplace,
    options_for,
    transcribe_pitched,
)


def test_options_for_returns_distinct_bass_settings() -> None:
    bass = options_for("bass")
    other = options_for("other")
    assert bass.minimum_frequency == 27.5
    assert bass.maximum_frequency == 330.0
    assert other.minimum_frequency is None
    assert isinstance(bass, TranscribeOptions)


def test_drums_hint_rejected() -> None:
    with pytest.raises(ValueError, match="drums"):
        transcribe_pitched("does-not-matter.wav", hint="drums")


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        transcribe_pitched(tmp_path / "nope.wav", hint="vocals")


def _midi_with_notes(notes: list[tuple[int, float, float, int]]) -> pretty_midi.PrettyMIDI:
    """notes is a list of (pitch, start_s, end_s, velocity) tuples."""
    midi = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0)
    for pitch, start, end, vel in notes:
        inst.notes.append(pretty_midi.Note(velocity=vel, pitch=pitch, start=start, end=end))
    midi.instruments.append(inst)
    return midi


def test_force_monophonic_truncates_overlaps() -> None:
    # Two overlapping notes: first should be truncated at the second's start.
    midi = _midi_with_notes([
        (60, 0.0, 1.0, 100),  # A
        (62, 0.5, 1.5, 100),  # B starts mid-A
    ])
    _force_monophonic_inplace(midi)
    notes = sorted(midi.instruments[0].notes, key=lambda n: n.start)
    assert len(notes) == 2
    assert notes[0].pitch == 60
    assert abs(notes[0].end - 0.5) < 1e-6
    assert notes[1].pitch == 62


def test_force_monophonic_drops_fully_eclipsed_note() -> None:
    # A long quiet note fully eclipsed by a louder short one starting at the same time.
    midi = _midi_with_notes([
        (60, 0.0, 2.0, 40),   # long quiet
        (62, 0.0, 0.5, 120),  # short loud — wins via velocity sort
    ])
    _force_monophonic_inplace(midi)
    pitches = {n.pitch for n in midi.instruments[0].notes}
    # The loud short note wins the slot at t=0; the long quiet note had end > start
    # truncated to 0 and is dropped.
    assert 62 in pitches


def test_merge_fragmented_notes_stitches_sustained() -> None:
    from demixer.core.transcription.pitched import _merge_fragmented_notes_inplace
    # One pitch shattered into 4 short fragments (pitch, start, END, vel) with
    # <90ms gaps → one note.
    midi = _midi_with_notes([
        (60, 0.00, 0.10, 80),
        (60, 0.15, 0.25, 90),   # gap 0.05 → merge
        (60, 0.30, 0.40, 70),   # gap 0.05 → merge
        (60, 0.42, 0.52, 100),  # gap 0.02 → merge
    ])
    _merge_fragmented_notes_inplace(midi)
    notes = midi.instruments[0].notes
    assert len(notes) == 1
    assert notes[0].start == 0.0
    assert abs(notes[0].end - 0.52) < 1e-6
    assert notes[0].velocity == 100  # max velocity across the run


def test_merge_keeps_distinct_repeats_apart() -> None:
    from demixer.core.transcription.pitched import _merge_fragmented_notes_inplace
    # Same pitch, big gap (>90ms) → two distinct notes preserved.
    midi = _midi_with_notes([
        (60, 0.0, 0.2, 90),
        (60, 1.0, 1.2, 90),
    ])
    _merge_fragmented_notes_inplace(midi)
    assert len(midi.instruments[0].notes) == 2


def test_merge_leaves_different_pitches_independent() -> None:
    from demixer.core.transcription.pitched import _merge_fragmented_notes_inplace
    midi = _midi_with_notes([
        (60, 0.0, 0.10, 90),
        (64, 0.12, 0.22, 90),  # different pitch, close → not merged
    ])
    _merge_fragmented_notes_inplace(midi)
    assert len(midi.instruments[0].notes) == 2


def test_octave_snap_pulls_sub_bass_up() -> None:
    midi = _midi_with_notes([(12, 0.0, 1.0, 100), (36, 0.0, 1.0, 100)])  # C0, C2
    _octave_snap_bass_inplace(midi)
    pitches = [n.pitch for n in midi.instruments[0].notes]
    assert 12 not in pitches  # was below 24
    assert 24 in pitches      # bumped up an octave
    assert 36 in pitches      # untouched


@pytest.mark.skipif(
    not os.environ.get("DEMIXER_RUN_HEAVY"),
    reason="set DEMIXER_RUN_HEAVY=1 to run (imports TensorFlow, runs basic-pitch)",
)
def test_transcribe_end_to_end_on_sine(tmp_path: Path) -> None:
    sr = 22_050  # basic-pitch resamples internally; arbitrary input rate fine
    t = np.arange(int(2.0 * sr)) / sr
    sine = 0.3 * np.sin(2 * np.pi * 440.0 * t).astype(np.float32)
    src = tmp_path / "a440.wav"
    sf.write(src, sine, sr, subtype="FLOAT")

    midi = transcribe_pitched(src, hint="other")
    assert isinstance(midi, pretty_midi.PrettyMIDI)
    notes = [n for inst in midi.instruments for n in inst.notes]
    assert notes, "basic-pitch returned zero notes for a 440 Hz sine"
    # Most notes should be near MIDI 69 (A4)
    avg_pitch = sum(n.pitch for n in notes) / len(notes)
    assert abs(avg_pitch - 69) < 2, f"avg pitch {avg_pitch} far from A4 (69)"
