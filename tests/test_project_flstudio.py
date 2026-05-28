"""Tests for the FL Studio `.flp` writer + piano-roll script generator.

FL Studio is Windows/macOS only, so we verify the binary output by parsing it
back with PyFLP (the reference parser) and asserting tempo, channels, patterns,
and individual notes survive the round-trip. The `.pyscript` output is checked by
compiling it and counting its `addNote` calls.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pretty_midi
import pyflp
import soundfile as sf
from pyflp._events import EventEnum

from demixer.core.analysis.key import KeyEstimate
from demixer.core.analysis.tempo_beats import TempoBeats
from demixer.core.project.flstudio import (
    PPQ,
    StemTrack,
    write_flp,
    write_flpianoroll_scripts,
)


def _warm_pyflp_enum() -> None:
    """Work around a PyFLP 2.2.1 / CPython 3.11.15 incompatibility.

    PyFLP's parser does ``EventEnum(id)`` for every event byte, relying on the
    enum's ``_missing_`` hook to mint pseudo-members for unknown IDs. CPython's
    ``Enum.__new__`` now raises ``TypeError`` *before* calling ``_missing_`` when
    the base enum has no members (``EventEnum``'s members all live in subclasses),
    so the parser dies on the very first event. Pre-seeding the value→member map
    with the same pseudo-members ``_missing_`` would have created restores the
    pre-3.11.15 behaviour. Typed lookups still resolve via the subclass iteration
    PyFLP does separately, so this only affects unknown-ID handling.
    """
    for v in range(256):
        if v not in EventEnum._value2member_map_:
            member = int.__new__(EventEnum, v)
            member._name_ = str(v)
            member._value_ = v
            EventEnum._value2member_map_[v] = member


_warm_pyflp_enum()

_BPM = 120.0
# (pitch, start_s, end_s, velocity); at 120 BPM a beat is 0.5s → PPQ ticks/beat.
_NOTES = [(60, 0.0, 0.5, 100), (62, 0.5, 1.0, 110), (64, 1.0, 1.5, 90)]


def _fake_tempo() -> TempoBeats:
    return TempoBeats(
        tempo_bpm=_BPM,
        beat_times_s=np.array([0.0, 0.5, 1.0]),
        downbeat_times_s=np.array([0.0]),
        beats_per_bar=4,
        method="beat_this",
    )


def _fake_key() -> KeyEstimate:
    return KeyEstimate(root="G", scale="major", strength=0.9, sharps=1)


def _populated(tmp_path: Path) -> list[StemTrack]:
    stems_dir = tmp_path / "stems"
    midi_dir = tmp_path / "midi"
    stems_dir.mkdir()
    midi_dir.mkdir()
    out: list[StemTrack] = []
    for name in ("vocals", "bass", "other", "drums"):
        wav = stems_dir / f"{name}.wav"
        sf.write(wav, np.zeros((44_100, 2), dtype=np.float32), 44_100, subtype="FLOAT")
        midi: Path | None = None
        if name != "drums":
            midi = midi_dir / f"{name}.mid"
            pm = pretty_midi.PrettyMIDI()
            inst = pretty_midi.Instrument(program=0)
            for pitch, start, end, vel in _NOTES:
                inst.notes.append(
                    pretty_midi.Note(velocity=vel, pitch=pitch, start=start, end=end)
                )
            pm.instruments.append(inst)
            pm.write(str(midi))
        out.append(StemTrack(name=name, wav_path=wav, midi_path=midi))
    return out


def test_flp_parses_with_correct_header_and_tempo(tmp_path: Path) -> None:
    out = write_flp(
        tmp_path / "song", tracks=_populated(tmp_path),
        tempo=_fake_tempo(), key=_fake_key(), duration_s=30.0,
    )
    assert out.suffix == ".flp"
    proj = pyflp.parse(out)
    assert proj.ppq == PPQ
    assert round(proj.tempo) == round(_BPM)


def test_flp_channel_count(tmp_path: Path) -> None:
    out = write_flp(
        tmp_path / "song", tracks=_populated(tmp_path),
        tempo=_fake_tempo(), key=_fake_key(), duration_s=30.0,
    )
    proj = pyflp.parse(out)
    # 4 audio stems + 3 notes-host channels (drums has no MIDI) = 7.
    assert len(proj.channels) == 7


def test_flp_pattern_notes_round_trip(tmp_path: Path) -> None:
    out = write_flp(
        tmp_path / "song", tracks=_populated(tmp_path),
        tempo=_fake_tempo(), key=_fake_key(), duration_s=30.0,
    )
    proj = pyflp.parse(out)
    patterns = list(proj.patterns)
    # One pattern per transcribed stem (vocals, bass, other).
    assert len(patterns) == 3
    for pat in patterns:
        notes = list(pat.notes)
        assert len(notes) == len(_NOTES)

    # Inspect one note end-to-end: pitch, position, length, velocity preserved.
    first = sorted(patterns[0].notes, key=lambda n: n.position)[0]
    assert str(first.key) == "C5"  # PyFLP renders MIDI 60 in FL's octave convention
    assert first.position == 0
    assert first.length == PPQ   # 0.5 s at 120 BPM = one quarter note = PPQ ticks
    assert first.velocity == 100


def test_flp_sample_paths_point_at_stems(tmp_path: Path) -> None:
    tracks = _populated(tmp_path)
    out = write_flp(
        tmp_path / "song", tracks=tracks,
        tempo=_fake_tempo(), key=_fake_key(), duration_s=30.0,
    )
    proj = pyflp.parse(out)
    sample_paths = {
        str(ch.sample_path) for ch in proj.channels
        if getattr(ch, "sample_path", None)
    }
    expected = {str(t.wav_path.resolve()) for t in tracks}
    assert expected <= sample_paths


def test_pyscripts_compile_and_inject_every_note(tmp_path: Path) -> None:
    scripts = write_flpianoroll_scripts(
        tmp_path / "scripts", tracks=_populated(tmp_path),
        tempo=_fake_tempo(), key=_fake_key(),
    )
    # One per transcribed stem.
    assert {p.stem for p in scripts} == {"vocals", "bass", "other"}
    for script in scripts:
        text = script.read_text()
        compile(text, str(script), "exec")          # valid Python
        assert text.count("score.addNote(n)") == 1  # one call inside the loop
        # NOTES literal carries every transcribed note.
        assert text.count("),") >= len(_NOTES)
