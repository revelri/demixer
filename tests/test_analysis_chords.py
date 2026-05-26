"""Tests for chord-recognition label conversion and structural integration."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from demixer.core.analysis.chords import ChordSegment, _short_label
from demixer.core.analysis.key import KeyEstimate
from demixer.core.ingest import ingest


def test_short_label_major_minor_and_passthrough() -> None:
    assert _short_label("C:maj") == "C"
    assert _short_label("F#:maj") == "F#"
    assert _short_label("E:min") == "Em"
    assert _short_label("Bb:min") == "Bbm"
    # No-chord sentinel: passes through unchanged
    assert _short_label("N") == "N"
    # Unknown qualities are preserved verbatim so future-model output isn't lost
    assert _short_label("G:7") == "G7"
    assert _short_label("D:sus4") == "Dsus4"


def test_chord_segment_dataclass() -> None:
    seg = ChordSegment(start_s=0.0, end_s=2.0, label="C")
    assert seg.start_s == 0.0
    assert seg.end_s == 2.0
    assert seg.label == "C"


@pytest.mark.skipif(
    not os.environ.get("DEMIXER_RUN_HEAVY"),
    reason="set DEMIXER_RUN_HEAVY=1 to run (downloads autochord model + invokes TF)",
)
def test_estimate_on_c_major_triad(tmp_path: Path) -> None:
    from demixer.core.analysis import chords as chords_mod

    sr = 44_100
    t = np.arange(int(4.0 * sr)) / sr
    # C major triad: C4, E4, G4
    wave = sum(0.15 * np.sin(2 * np.pi * f * t).astype(np.float32)
               for f in (261.63, 329.63, 392.00))
    stereo = np.stack([wave, wave], axis=1).astype(np.float32)
    src = tmp_path / "c_major.wav"
    sf.write(src, stereo, sr, subtype="FLOAT")
    audio = ingest(src)
    segments = chords_mod.estimate(audio)
    assert segments, "no chord segments returned"
    # Expect mostly C-rooted labels — exact length depends on model framing
    labels = [s.label for s in segments]
    c_rooted = sum(1 for label in labels if label.startswith("C"))
    assert c_rooted >= 1, f"expected some C-rooted segments; got {labels}"


@pytest.mark.skipif(
    not os.environ.get("DEMIXER_RUN_HEAVY"),
    reason="set DEMIXER_RUN_HEAVY=1 to run (needs the isolated .venv-btc worker)",
)
def test_btc_short_clip_no_unbound_error(tmp_path: Path) -> None:
    """Regression: clips shorter than BTC's inst_len (~10s) used to crash the
    worker with UnboundLocalError ('feature'). The worker now pads short input
    and clamps segments back to the real duration."""
    from demixer.core.analysis import chords_btc
    if not chords_btc.available():
        pytest.skip("BTC worker venv not set up")
    sr = 44_100
    t = np.arange(int(4.0 * sr)) / sr  # 4s — well under inst_len
    wave = sum(0.2 * np.sin(2 * np.pi * f * t).astype(np.float32) for f in (196.0, 246.94, 293.66))
    src = tmp_path / "short.wav"
    sf.write(src, np.stack([wave, wave], axis=1), sr, subtype="FLOAT")
    segs = chords_btc.estimate(ingest(src))  # must not raise
    # All segments clamped within the 4s clip
    for s in segs:
        assert s.end_s <= 4.01, f"segment end {s.end_s} exceeds clip duration"


def test_respell_to_key_flat_key_diatonic_roots() -> None:
    from demixer.core.analysis.chords import respell_to_key
    key = KeyEstimate(root="C", scale="minor", strength=0.9, sharps=-3)  # Cm: Eb Bb Ab in sig
    chords = [ChordSegment(0, 1, "Cm"), ChordSegment(1, 2, "D#"), ChordSegment(2, 3, "A#"),
              ChordSegment(3, 4, "G"), ChordSegment(4, 5, "Gm")]
    out = {c.label for c in respell_to_key(chords, key)}
    # D# (pc3) → Eb (diatonic bIII), A# (pc10) → Bb (diatonic bVII); suffixes preserved
    assert "Eb" in out and "Bb" in out
    assert "D#" not in out and "A#" not in out
    assert "Cm" in out and "G" in out  # already correct, unchanged


def test_respell_to_key_leaves_chromatic_roots() -> None:
    from demixer.core.analysis.chords import respell_to_key
    key = KeyEstimate(root="C", scale="minor", strength=0.9, sharps=-3)
    # C# (pc1) is NOT diatonic to C natural minor (Neapolitan) → left untouched
    out = {c.label for c in respell_to_key([ChordSegment(0, 1, "C#")], key)}
    assert out == {"C#"}


def test_respell_to_key_sharp_key_unchanged() -> None:
    from demixer.core.analysis.chords import respell_to_key
    key = KeyEstimate(root="E", scale="major", strength=0.9, sharps=4)  # E: F# G# C# D#
    chords = [ChordSegment(0, 1, "E"), ChordSegment(1, 2, "A"), ChordSegment(2, 3, "B"),
              ChordSegment(3, 4, "C#m"), ChordSegment(4, 5, "F#m")]
    out = [c.label for c in respell_to_key(chords, key)]
    assert out == ["E", "A", "B", "C#m", "F#m"]  # sharp key → sharp chords, no change


def test_respell_to_key_passes_through_no_chord() -> None:
    from demixer.core.analysis.chords import respell_to_key
    key = KeyEstimate(root="C", scale="minor", strength=0.9, sharps=-3)
    out = [c.label for c in respell_to_key([ChordSegment(0, 1, "N")], key)]
    assert out == ["N"]
