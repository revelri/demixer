"""Tests for the DAWproject XML exporter — structural / round-trip validation.

We cannot launch Bitwig/Studio One/Cubase headlessly, so the tests assert:
  - the .dawproject is a valid zip containing the required entries
  - project.xml parses as well-formed XML with the expected counts and ids
  - audio files end up at the path the XML references
  - inline <Notes> contains every note from the source MIDI
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import numpy as np
import pretty_midi
import soundfile as sf

from demixer.core.analysis.key import KeyEstimate
from demixer.core.analysis.tempo_beats import TempoBeats
from demixer.core.project.dawproject import StemTrack, write_dawproject


def _fake_tempo() -> TempoBeats:
    return TempoBeats(
        tempo_bpm=120.0,
        beat_times_s=np.array([0.0, 0.5, 1.0]),
        downbeat_times_s=np.array([0.0]),
        beats_per_bar=4,
        method="beat_this",
    )


def _fake_key() -> KeyEstimate:
    return KeyEstimate(root="G", scale="major", strength=0.9, sharps=1)


def _write_fake_stem(path: Path) -> None:
    sf.write(path, np.zeros((44_100, 2), dtype=np.float32), 44_100, subtype="FLOAT")


def _write_midi_with_notes(path: Path, notes: list[tuple[int, float, float, int]]) -> None:
    midi = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0)
    for pitch, start, end, vel in notes:
        inst.notes.append(pretty_midi.Note(velocity=vel, pitch=pitch, start=start, end=end))
    midi.instruments.append(inst)
    midi.write(str(path))


def _populated(tmp_path: Path) -> list[StemTrack]:
    stems_dir = tmp_path / "stems"
    midi_dir = tmp_path / "midi"
    stems_dir.mkdir()
    midi_dir.mkdir()
    out: list[StemTrack] = []
    for name in ("vocals", "bass", "other", "drums"):
        wav = stems_dir / f"{name}.wav"
        _write_fake_stem(wav)
        midi: Path | None = None
        if name != "drums":
            midi = midi_dir / f"{name}.mid"
            _write_midi_with_notes(midi, [
                (60, 0.0, 0.5, 100),
                (62, 0.5, 1.0, 110),
                (64, 1.0, 1.5, 90),
            ])
        out.append(StemTrack(name=name, wav_path=wav, midi_path=midi))
    return out


def test_zip_layout_and_entries(tmp_path: Path) -> None:
    tracks = _populated(tmp_path)
    out = write_dawproject(
        tmp_path / "out", tracks=tracks, tempo=_fake_tempo(), key=_fake_key(), duration_s=30.0,
    )
    assert out.suffix == ".dawproject"
    with zipfile.ZipFile(out) as z:
        names = set(z.namelist())
    assert {"project.xml", "metadata.xml"} <= names
    for name in ("vocals", "bass", "other", "drums"):
        assert f"audio/{name}.wav" in names


def test_project_xml_structure(tmp_path: Path) -> None:
    tracks = _populated(tmp_path)
    out = write_dawproject(
        tmp_path / "out", tracks=tracks, tempo=_fake_tempo(), key=_fake_key(), duration_s=30.0,
    )
    with zipfile.ZipFile(out) as z:
        root = ET.fromstring(z.read("project.xml"))

    assert root.tag == "Project"
    assert root.attrib["version"] == "1.0"

    transport = root.find("Transport")
    assert transport is not None
    tempo_el = transport.find("Tempo")
    assert tempo_el is not None and float(tempo_el.attrib["value"]) == 120.0
    ts_el = transport.find("TimeSignature")
    assert ts_el is not None and ts_el.attrib["numerator"] == "4"

    structure = root.find("Structure")
    assert structure is not None
    tracks_xml = structure.findall("Track")
    # 4 audio tracks + 3 instrument tracks (drums has no MIDI)
    assert len(tracks_xml) == 7
    audio_tracks = [t for t in tracks_xml if t.attrib["contentType"] == "audio"]
    notes_tracks = [t for t in tracks_xml if t.attrib["contentType"] == "notes"]
    assert len(audio_tracks) == 4
    assert len(notes_tracks) == 3


def test_audio_clips_reference_zip_paths(tmp_path: Path) -> None:
    tracks = _populated(tmp_path)
    out = write_dawproject(
        tmp_path / "out", tracks=tracks, tempo=_fake_tempo(), key=_fake_key(), duration_s=30.0,
    )
    with zipfile.ZipFile(out) as z:
        root = ET.fromstring(z.read("project.xml"))
        names_in_zip = set(z.namelist())

    files = root.iter("File")
    paths = {el.attrib["path"] for el in files}
    assert {"audio/vocals.wav", "audio/bass.wav", "audio/other.wav", "audio/drums.wav"} <= paths
    # And those paths must actually exist in the zip
    assert paths <= names_in_zip | {"project.xml", "metadata.xml"}


def test_notes_clip_contains_every_midi_note(tmp_path: Path) -> None:
    tracks = _populated(tmp_path)
    out = write_dawproject(
        tmp_path / "out", tracks=tracks, tempo=_fake_tempo(), key=_fake_key(), duration_s=30.0,
    )
    with zipfile.ZipFile(out) as z:
        root = ET.fromstring(z.read("project.xml"))

    # 3 transcribed stems × 3 notes each = 9 inline Note elements
    notes = list(root.iter("Note"))
    assert len(notes) == 9
    # Velocities normalized to [0,1]
    for n in notes:
        v = float(n.attrib["vel"])
        assert 0.0 <= v <= 1.0
    # Pitches preserved
    pitches = sorted(int(n.attrib["key"]) for n in notes)
    assert pitches == [60, 60, 60, 62, 62, 62, 64, 64, 64]


def test_metadata_has_title(tmp_path: Path) -> None:
    tracks = _populated(tmp_path)
    out = write_dawproject(
        tmp_path / "out",
        tracks=tracks, tempo=_fake_tempo(), key=_fake_key(), duration_s=30.0,
        project_name="my-song",
    )
    with zipfile.ZipFile(out) as z:
        meta = ET.fromstring(z.read("metadata.xml"))
    title = meta.find("Title")
    assert title is not None and title.text == "my-song"
