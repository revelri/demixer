"""Tests for the score pipeline: quantize → MusicXML → Verovio SVG.

PDF render via MuseScore is opportunistic — tested only if `mscore` is on PATH.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pretty_midi
import pytest

from demixer.core.analysis.key import KeyEstimate
from demixer.core.analysis.tempo_beats import TempoBeats
from demixer.core.score.musicxml import build_score, to_musicxml, write_musicxml
from demixer.core.score.quantize import quantize_midi
from demixer.core.score.render import (
    musescore_available,
    open_in_musescore,
    render_audio,
    render_mscz,
    render_pdf,
    render_png,
    render_svg,
    render_svg_mscore,
)

# ---------- helpers ----------

def _tempo() -> TempoBeats:
    return TempoBeats(
        tempo_bpm=120.0,
        beat_times_s=np.array([0.0, 0.5, 1.0, 1.5, 2.0]),
        downbeat_times_s=np.array([0.0, 2.0]),
        beats_per_bar=4,
        method="beat_this",
    )


def _key() -> KeyEstimate:
    return KeyEstimate(root="G", scale="major", strength=0.9, sharps=1)


def _midi_from(notes: list[tuple[int, float, float, int]], *, is_drum: bool = False) -> pretty_midi.PrettyMIDI:
    midi = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0, is_drum=is_drum)
    for pitch, start, end, vel in notes:
        inst.notes.append(pretty_midi.Note(velocity=vel, pitch=pitch, start=start, end=end))
    midi.instruments.append(inst)
    return midi


# ---------- quantize ----------

def test_quantize_snaps_to_grid() -> None:
    # 120 BPM beats => the 120 BPM reference is 1:1, 16th grid => every 0.125s
    beats = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    midi = _midi_from([(60, 0.13, 0.62, 100), (62, 0.27, 0.78, 100)])
    out = quantize_midi(midi, beats, subdivisions_per_beat=4)
    starts = sorted(n.start for n in out.instruments[0].notes)
    # 0.13 -> beat 0.26 -> 0.25 -> 0.125s ; 0.27 -> beat 0.54 -> 0.5 -> 0.25s
    for actual, expected in zip(starts, [0.125, 0.25]):
        assert abs(actual - expected) < 1e-6


def test_quantize_output_is_tuplet_free_on_irregular_beats() -> None:
    # Irregular (jittery) beat grid — the failure mode that crashed MusicXML
    # export. Output note times must be exact 1/16-note multiples in the 120 BPM
    # reference (0.125s) regardless of input beat jitter.
    rng = np.random.default_rng(0)
    period = 0.46
    beats = np.cumsum([0.0] + [period + rng.uniform(-0.02, 0.02) for _ in range(20)])
    midi = _midi_from([(60, float(t) + 0.013, float(t) + 0.21, 90) for t in beats[:10]])
    out = quantize_midi(midi, beats, subdivisions_per_beat=4)
    sixteenth = 0.125
    for n in out.instruments[0].notes:
        # start and duration must both be integer multiples of a 16th note
        assert abs(round(n.start / sixteenth) * sixteenth - n.start) < 1e-6
        assert abs(round((n.end - n.start) / sixteenth) * sixteenth - (n.end - n.start)) < 1e-6


def test_quantize_deduplicates_collisions() -> None:
    beats = np.array([0.0, 0.5, 1.0])
    # Two notes at same pitch landing on the same grid slot after snap → dedupe
    midi = _midi_from([(60, 0.01, 0.1, 100), (60, 0.02, 0.1, 100)])
    out = quantize_midi(midi, beats, subdivisions_per_beat=4)
    assert len(out.instruments[0].notes) == 1


def test_quantize_keeps_min_duration_one_step() -> None:
    beats = np.array([0.0, 0.5, 1.0])
    midi = _midi_from([(60, 0.0, 0.001, 100)])  # near-zero duration
    out = quantize_midi(midi, beats, subdivisions_per_beat=4)
    n = out.instruments[0].notes[0]
    grid_step = 0.5 / 4
    assert abs((n.end - n.start) - grid_step) < 1e-6


def test_quantize_passes_through_when_no_beats() -> None:
    midi = _midi_from([(60, 0.13, 0.62, 100)])
    out = quantize_midi(midi, np.array([0.0]), subdivisions_per_beat=4)
    # Single-element grid → returned unchanged
    assert out is midi


def test_quantize_rejects_bad_subdivision() -> None:
    midi = _midi_from([(60, 0.0, 0.5, 100)])
    with pytest.raises(ValueError):
        quantize_midi(midi, np.array([0.0, 0.5]), subdivisions_per_beat=0)


# ---------- musicxml ----------

def test_multipart_offgrid_score_exports_without_crash() -> None:
    """Regression: dense multi-part scores from off-grid notes used to crash
    music21's MusicXML export (duplicate-Measure StreamException). build_score
    must quantize + flat-assemble so export always succeeds."""
    import numpy as np
    rng = np.random.default_rng(1)
    beats = np.cumsum([0.0] + [0.5 + rng.uniform(-0.02, 0.02) for _ in range(24)])
    # 5 parts of jittery, overlapping, off-grid notes (the failure shape)
    stems = {}
    for i, nm in enumerate(("bass", "other", "vocals", "guitar", "piano")):
        notes = [(48 + i * 3 + (j % 5), float(beats[j]) + 0.017, float(beats[j]) + 0.31, 80)
                 for j in range(20)]
        m = _midi_from(notes)
        stems[nm] = quantize_midi(m, beats, subdivisions_per_beat=4)
    score = build_score(stems, tempo=_tempo(), key=_key())
    xml = to_musicxml(score)  # must not raise
    assert "<score-partwise" in xml
    assert len(score.parts) == 5


def test_build_score_creates_part_per_stem(tmp_path: Path) -> None:
    stems = {
        "vocals": _midi_from([(60, 0.0, 0.5, 100), (62, 0.5, 1.0, 100)]),
        "bass":   _midi_from([(36, 0.0, 1.0, 100)]),
        "drums":  _midi_from([(38, 0.0, 0.25, 100)], is_drum=True),
    }
    score = build_score(stems, tempo=_tempo(), key=_key())
    assert score.metadata.title == "demixer"
    # Drums excluded by default — MuseScore can't render music21's drum-staff XML
    part_names = [p.partName for p in score.parts]
    assert part_names == ["vocals", "bass"]


def test_build_score_includes_drums_when_requested() -> None:
    stems = {
        "vocals": _midi_from([(60, 0.0, 0.5, 100)]),
        "drums":  _midi_from([(38, 0.0, 0.25, 100)], is_drum=True),
    }
    score = build_score(stems, tempo=_tempo(), key=_key(), include_drums=True)
    assert [p.partName for p in score.parts] == ["vocals", "drums"]


def test_musicxml_serializes(tmp_path: Path) -> None:
    stems = {"vocals": _midi_from([(60, 0.0, 0.5, 100)])}
    score = build_score(stems, tempo=_tempo(), key=_key())
    xml = to_musicxml(score)
    assert xml.startswith("<?xml")
    assert "<score-partwise" in xml
    # The G-major key signature has 1 sharp; music21 emits <fifths>1</fifths>
    assert "<fifths>1</fifths>" in xml
    out = write_musicxml(score, tmp_path / "score.musicxml")
    assert out.exists() and out.stat().st_size > 0


# ---------- render ----------

def test_render_svg_produces_at_least_one_page(tmp_path: Path) -> None:
    stems = {"vocals": _midi_from([(60, 0.0, 0.5, 100), (62, 0.5, 1.0, 100)])}
    score = build_score(stems, tempo=_tempo(), key=_key())
    xml_path = write_musicxml(score, tmp_path / "score.musicxml")
    svg_paths = render_svg(xml_path, tmp_path / "svg")
    assert svg_paths, "no SVG pages rendered"
    for p in svg_paths:
        text = p.read_text()
        assert text.startswith("<svg") or "<svg" in text[:200]


@pytest.mark.skipif(not musescore_available(), reason="MuseScore CLI not installed")
def test_render_pdf_runs(tmp_path: Path) -> None:
    stems = {"vocals": _midi_from([(60, 0.0, 0.5, 100)])}
    score = build_score(stems, tempo=_tempo(), key=_key())
    xml_path = write_musicxml(score, tmp_path / "score.musicxml")
    pdf = render_pdf(xml_path, tmp_path / "score.pdf")
    assert pdf is not None and pdf.exists() and pdf.stat().st_size > 0


def test_render_pdf_returns_none_without_musescore(tmp_path: Path, monkeypatch) -> None:
    """If MuseScore isn't installed, render_pdf returns None cleanly."""
    monkeypatch.setattr("demixer.core.score.render._find_musescore", lambda: None)
    stems = {"vocals": _midi_from([(60, 0.0, 0.5, 100)])}
    score = build_score(stems, tempo=_tempo(), key=_key())
    xml_path = write_musicxml(score, tmp_path / "score.musicxml")
    assert render_pdf(xml_path, tmp_path / "out.pdf") is None


def test_all_musescore_renderers_return_none_without_mscore(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("demixer.core.score.render._find_musescore", lambda: None)
    stems = {"vocals": _midi_from([(60, 0.0, 0.5, 100)])}
    xml_path = write_musicxml(build_score(stems, tempo=_tempo(), key=_key()),
                              tmp_path / "score.musicxml")
    assert render_mscz(xml_path, tmp_path / "out.mscz") is None
    assert render_svg_mscore(xml_path, tmp_path / "svg") is None
    assert render_png(xml_path, tmp_path / "png") is None
    assert render_audio(xml_path, tmp_path / "out.wav") is None
    assert open_in_musescore(xml_path) is False


@pytest.fixture
def musescore_xml(tmp_path: Path) -> Path:
    stems = {"vocals": _midi_from([(60, 0.0, 0.5, 100), (62, 0.5, 1.0, 100)])}
    score = build_score(stems, tempo=_tempo(), key=_key())
    return write_musicxml(score, tmp_path / "score.musicxml")


@pytest.mark.skipif(not musescore_available(), reason="MuseScore CLI not installed")
def test_render_mscz_produces_file(tmp_path: Path, musescore_xml: Path) -> None:
    out = render_mscz(musescore_xml, tmp_path / "score.mscz")
    assert out is not None and out.exists() and out.stat().st_size > 0


@pytest.mark.skipif(not musescore_available(), reason="MuseScore CLI not installed")
def test_render_svg_mscore_produces_at_least_one_page(tmp_path: Path, musescore_xml: Path) -> None:
    paths = render_svg_mscore(musescore_xml, tmp_path / "svg")
    assert paths is not None and paths
    for p in paths:
        assert p.suffix == ".svg" and p.stat().st_size > 0


@pytest.mark.skipif(not musescore_available(), reason="MuseScore CLI not installed")
def test_render_png_produces_at_least_one_page(tmp_path: Path, musescore_xml: Path) -> None:
    paths = render_png(musescore_xml, tmp_path / "png")
    assert paths is not None and paths
    for p in paths:
        assert p.suffix == ".png" and p.stat().st_size > 0


@pytest.mark.skipif(not musescore_available(), reason="MuseScore CLI not installed")
def test_render_audio_mp3(tmp_path: Path, musescore_xml: Path) -> None:
    out = render_audio(musescore_xml, tmp_path / "preview.mp3", format="mp3")
    assert out is not None and out.exists() and out.stat().st_size > 0
    assert out.suffix == ".mp3"


@pytest.mark.skipif(not musescore_available(), reason="MuseScore CLI not installed")
def test_render_audio_auto_corrects_extension(tmp_path: Path, musescore_xml: Path) -> None:
    """If caller passes wrong extension, render_audio rewrites to match `format`."""
    out = render_audio(musescore_xml, tmp_path / "preview.bogus", format="mp3")
    assert out is not None and out.suffix == ".mp3"
