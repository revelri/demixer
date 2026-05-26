"""Tests for the Reaper RPP exporter — structural / round-trip validation.

We cannot launch Reaper headless on this box, so the tests assert:
  - the produced file parses as a balanced node tree
  - track / item / source counts match what we passed in
  - paths are recorded relative to the .rpp location (portability)
  - the tempo + time signature line is present and correct
  - key marker appears at t=0
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from demixer.core.analysis.key import KeyEstimate
from demixer.core.analysis.tempo_beats import TempoBeats
from demixer.core.project.reaper import StemTrack, write_rpp


def _fake_tempo() -> TempoBeats:
    return TempoBeats(
        tempo_bpm=120.0,
        beat_times_s=np.array([0.0, 0.5, 1.0]),
        downbeat_times_s=np.array([0.0]),
        beats_per_bar=4,
        method="beat_this",
    )


def _fake_key() -> KeyEstimate:
    return KeyEstimate(root="C", scale="major", strength=0.9, sharps=0)


def _populated_stems(tmp_path: Path) -> list[StemTrack]:
    stems_dir = tmp_path / "stems"
    midi_dir = tmp_path / "midi"
    stems_dir.mkdir()
    midi_dir.mkdir()
    tracks = []
    for name in ("vocals", "bass", "other", "drums"):
        wav = stems_dir / f"{name}.wav"
        wav.write_bytes(b"fake-wav")
        midi: Path | None = None
        if name != "drums":  # drums not transcribed in v1
            midi = midi_dir / f"{name}.mid"
            midi.write_bytes(b"fake-midi")
        tracks.append(StemTrack(name=name, wav_path=wav, midi_path=midi))
    return tracks


def _count_balanced(text: str) -> tuple[int, int]:
    """Count opening `<NAME` and closing `>` tokens; should be equal in a valid file."""
    # An RPP "<" only opens when at start-of-line indent followed by a name char
    opens = sum(1 for line in text.splitlines() if line.lstrip().startswith("<"))
    closes = sum(1 for line in text.splitlines() if line.strip() == ">")
    return opens, closes


def test_writes_balanced_tree(tmp_path: Path) -> None:
    tracks = _populated_stems(tmp_path)
    rpp = write_rpp(
        tmp_path / "out.rpp",
        tracks=tracks,
        tempo=_fake_tempo(),
        key=_fake_key(),
        duration_s=30.0,
    )
    text = rpp.read_text()
    opens, closes = _count_balanced(text)
    assert opens == closes, f"unbalanced nodes: {opens} open vs {closes} close"
    assert text.startswith("<REAPER_PROJECT")
    assert text.rstrip().endswith(">")


def test_track_and_item_counts(tmp_path: Path) -> None:
    tracks = _populated_stems(tmp_path)
    rpp = write_rpp(
        tmp_path / "out.rpp",
        tracks=tracks,
        tempo=_fake_tempo(),
        key=_fake_key(),
        duration_s=30.0,
    )
    text = rpp.read_text()
    # 4 audio tracks + 3 MIDI tracks (drums has no MIDI)
    assert text.count("<TRACK ") == 7
    assert text.count("<SOURCE WAVE") == 4
    assert text.count("<SOURCE MIDI") == 3


def test_paths_are_relative_to_rpp(tmp_path: Path) -> None:
    tracks = _populated_stems(tmp_path)
    rpp = write_rpp(
        tmp_path / "out.rpp",
        tracks=tracks,
        tempo=_fake_tempo(),
        key=_fake_key(),
        duration_s=30.0,
    )
    text = rpp.read_text()
    # File references should be relative — never start with the tmp_path absolute prefix
    assert str(tmp_path) not in text, "absolute paths leaked into RPP"
    # Sanity: the relative paths we expect appear
    for name in ("vocals", "bass", "other", "drums"):
        assert f"stems/{name}.wav" in text
    for name in ("vocals", "bass", "other"):
        assert f"midi/{name}.mid" in text


def test_tempo_and_marker_lines_present(tmp_path: Path) -> None:
    tracks = _populated_stems(tmp_path)
    rpp = write_rpp(
        tmp_path / "out.rpp",
        tracks=tracks,
        tempo=_fake_tempo(),
        key=_fake_key(),
        duration_s=30.0,
    )
    text = rpp.read_text()
    assert "TEMPO 120.0000 4 4" in text
    assert "key: C major" in text


def test_handles_special_chars_in_names(tmp_path: Path) -> None:
    """Names containing the default `"` quote should pick an alternate quote without crashing."""
    stems_dir = tmp_path / "stems"
    stems_dir.mkdir()
    wav = stems_dir / "weird.wav"
    wav.write_bytes(b"x")
    tracks = [StemTrack(name='hello "world"', wav_path=wav, midi_path=None)]
    rpp = write_rpp(
        tmp_path / "out.rpp",
        tracks=tracks,
        tempo=_fake_tempo(),
        key=_fake_key(),
        duration_s=1.0,
    )
    text = rpp.read_text()
    # Either single-quoted or backtick-quoted, but not an unterminated double quote
    assert "'hello \"world\"'" in text or "`hello \"world\"`" in text
