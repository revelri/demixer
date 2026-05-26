"""Tests for the drag-in (universal-fallback) bundle writer."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

from demixer.core.analysis.key import KeyEstimate
from demixer.core.analysis.tempo_beats import TempoBeats
from demixer.core.project.dragin import StemTrack, write_dragin


def _tempo() -> TempoBeats:
    return TempoBeats(
        tempo_bpm=120.0,
        beat_times_s=np.array([0.0, 0.5]),
        downbeat_times_s=np.array([0.0]),
        beats_per_bar=4,
        method="beat_this",
    )


def _key() -> KeyEstimate:
    return KeyEstimate(root="D", scale="minor", strength=0.8, sharps=-1)


def _stems(tmp_path: Path) -> list[StemTrack]:
    stems = tmp_path / "src_stems"
    midis = tmp_path / "src_midi"
    stems.mkdir()
    midis.mkdir()
    out = []
    for name in ("vocals", "bass", "drums", "other"):
        wav = stems / f"{name}.wav"
        sf.write(wav, np.zeros((4_410, 2), dtype=np.float32), 44_100, subtype="FLOAT")
        midi = None
        if name != "drums":
            midi = midis / f"{name}.mid"
            midi.write_bytes(b"fake-midi")
        out.append(StemTrack(name=name, wav_path=wav, midi_path=midi))
    return out


def test_layout(tmp_path: Path) -> None:
    tracks = _stems(tmp_path)
    out = write_dragin(tmp_path / "out", tracks=tracks, tempo=_tempo(), key=_key(), duration_s=30.0)
    assert (out / "stems").is_dir()
    assert (out / "midi").is_dir()
    assert (out / "instruments.json").is_file()
    assert (out / "project.json").is_file()
    assert (out / "README.txt").is_file()
    for name in ("vocals", "bass", "drums", "other"):
        assert (out / "stems" / f"{name}.wav").exists()
    for name in ("vocals", "bass", "other"):
        assert (out / "midi" / f"{name}.mid").exists()
    assert not (out / "midi" / "drums.mid").exists()


def test_instruments_json_marks_drums_as_drum_channel(tmp_path: Path) -> None:
    tracks = _stems(tmp_path)
    out = write_dragin(tmp_path / "out", tracks=tracks, tempo=_tempo(), key=_key(), duration_s=30.0)
    inst = json.loads((out / "instruments.json").read_text())

    assert inst["drums"]["is_drum_channel"] is True
    assert inst["drums"]["channel"] == 10
    assert inst["drums"]["midi"] is None  # no MIDI written for drums

    for name in ("vocals", "bass", "other"):
        assert inst[name]["is_drum_channel"] is False
        assert inst[name]["channel"] == 1
        assert inst[name]["midi"] == f"midi/{name}.mid"


def test_instruments_json_gm_programs_in_valid_range(tmp_path: Path) -> None:
    tracks = _stems(tmp_path)
    out = write_dragin(tmp_path / "out", tracks=tracks, tempo=_tempo(), key=_key(), duration_s=30.0)
    inst = json.loads((out / "instruments.json").read_text())
    for stem_name, info in inst.items():
        assert 0 <= info["gm_program"] <= 127, f"{stem_name}: gm_program out of range"
        assert isinstance(info["patch_name"], str) and info["patch_name"]


def test_project_json_carries_analysis(tmp_path: Path) -> None:
    tracks = _stems(tmp_path)
    out = write_dragin(tmp_path / "out", tracks=tracks, tempo=_tempo(), key=_key(), duration_s=42.0)
    proj = json.loads((out / "project.json").read_text())
    assert proj["tempo_bpm"] == 120.0
    assert proj["beats_per_bar"] == 4
    assert proj["tempo_method"] == "beat_this"
    assert proj["key"]["root"] == "D"
    assert proj["key"]["scale"] == "minor"
    assert proj["key"]["sharps"] == -1
    assert proj["duration_s"] == 42.0


def test_readme_mentions_each_daw_and_tempo(tmp_path: Path) -> None:
    tracks = _stems(tmp_path)
    out = write_dragin(tmp_path / "out", tracks=tracks, tempo=_tempo(), key=_key(), duration_s=30.0)
    txt = (out / "README.txt").read_text()
    for daw in ("FL Studio", "Ableton Live", "Logic Pro"):
        assert daw in txt, f"README missing {daw} section"
    assert "120.00" in txt
    assert "D minor" in txt


def test_dragin_copy_media_false_references_shared(tmp_path: Path) -> None:
    """copy_media=False must NOT duplicate stems/midi and must point instruments
    at ../stems / ../midi (dedup for the .demixer archive)."""
    tracks = _stems(tmp_path)
    out = write_dragin(tmp_path / "out", tracks=tracks, tempo=_tempo(), key=_key(),
                       duration_s=30.0, copy_media=False)
    # No duplicated media in the dragin folder
    assert not (out / "stems").exists()
    assert not (out / "midi").exists()
    # instruments.json references the shared top-level media via ../
    inst = json.loads((out / "instruments.json").read_text())
    assert inst["vocals"]["stem_wav"] == "../stems/vocals.wav"
    assert inst["vocals"]["midi"] == "../midi/vocals.mid"
    assert inst["drums"]["stem_wav"] == "../stems/drums.wav"
    # metadata files still written
    assert (out / "instruments.json").is_file()
    assert (out / "project.json").is_file()
    assert (out / "README.txt").is_file()
